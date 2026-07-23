"""v1.2 有界 Agent Orchestrator（Slice 3 §六；Probe Comparison 於 Slice 4 插入）。

同步、有界的狀態機。核心紀律（§二正式規則 + 實作驗收條件）：

- Candidate **不得**自動取代 Parent：由 deterministic comparator（重用 v1.1
  ``evaluate_constraints`` + recommendation ordering）判定，LLM 不參與勝負。
- Candidate 被拒 → 保留 Parent、刪除 Candidate collection；但下一輪 context 仍含
  proposal / probe comparison / tune delta / coarse gate / rejection reason。
- 所有終止路徑（COMPLETED / STOPPED / FAILED / KeyboardInterrupt）都在 ``finally`` 安全清理
  最後的 Parent、Candidate 與暫存 collection；``factory_knowledge`` 由 executor 的
  ``is_optimizer_collection`` 守門，永不受影響。
- Fake determinism 只保證 proposal/probe；**promotion / stop / metrics 的決定性由本層負責**。

本層透過 :class:`TrialExecutor` 抽象執行 trial，不直接依賴 Qdrant；真實 executor（Slice 6）
包住 v1.1 ``execute_run``，測試用 fake executor 注入 scripted 結果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from evaluation.v11.run_models import TrialConfig, TrialMetric
from evaluation.v11.search.constraints import MetricLookup, evaluate_constraints
from evaluation.v11.search.recommendation import RankableCandidate, compare_candidates
from evaluation.v11.search.search_config import TIEBREAK_CONFIG_HASH
from evaluation.v11.trial_runner import (
    CANONICAL_TEXT_VERSION,
    IDENTIFIER_NORMALIZATION_VERSION,
)
from evaluation.v12.agent.context import (
    PreviousRoundSummary,
    SegmentExcerpt,
    TuneFailureView,
    build_agent_context,
)
from evaluation.v12.agent.metrics_delta import build_lookup, comparator_metric_refs
from evaluation.v12.agent.parser import ProposalParseError, parse_proposal
from evaluation.v12.agent.planner import AgentPlanner
from evaluation.v12.agent.run_config import AgentRunConfig
from evaluation.v12.agent.validator import ProposalValidator, ValidationResult

SPLIT_TUNE = "tune"
SPLIT_VALIDATION = "validation"


# --------------------------------------------------------------------------
# States / stop reasons / decisions
# --------------------------------------------------------------------------
class AgentState:
    CREATED = "created"
    BASELINE_RUNNING = "baseline_running"
    ANALYZING = "analyzing"
    PROPOSAL_VALIDATING = "proposal_validating"
    TRIAL_RUNNING = "trial_running"
    PROBE_RUNNING = "probe_running"
    EVALUATING = "evaluating"
    PROMOTION_DECIDING = "promotion_deciding"
    AWAITING_NEXT_ROUND = "awaiting_next_round"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class StopReason:
    MAX_ROUNDS = "max_rounds"
    MAX_LLM_CALLS = "max_llm_calls"
    NO_IMPROVEMENT = "no_improvement_rounds"
    DUPLICATE_PROPOSAL = "duplicate_proposal"
    NO_VALID_PROPOSAL = "no_valid_proposal"
    MAX_CONSECUTIVE_REJECTIONS = "max_consecutive_rejections"
    TARGET_METRIC_REACHED = "target_metric_reached"
    LLM_RECOMMENDED_STOP = "llm_recommended_stop"
    MANUAL_STOP = "manual_stop"
    FATAL_TRIAL_FAILURE = "fatal_trial_failure"
    LLM_ERROR = "llm_error"
    BASELINE_FAILED = "baseline_failed"


PROMOTION_PROMOTED = "promoted"
PROMOTION_REJECTED = "rejected"
PROMOTION_NONE = "none"

RECOMMENDATION_NOT_EVALUATED = "not_evaluated"

_COMPLETED_REASONS = frozenset(
    {StopReason.MAX_ROUNDS, StopReason.TARGET_METRIC_REACHED, StopReason.LLM_RECOMMENDED_STOP}
)


# --------------------------------------------------------------------------
# Trial executor abstraction
# --------------------------------------------------------------------------
@dataclass
class TrialOutcome:
    split: str
    config_hash: str
    status: str  # "completed" | "failed"
    trial_id: Optional[str] = None
    collection_name: Optional[str] = None
    metrics: tuple[TrialMetric, ...] = ()
    error: Optional[str] = None
    # 只有 tune split 需要（供 context builder）：
    tune_failures: tuple[TuneFailureView, ...] = ()
    segments: tuple[SegmentExcerpt, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    def lookup(self) -> MetricLookup:
        return build_lookup(self.metrics)


class TrialExecutor(Protocol):
    """在指定 split dataset 上執行單一 config 的 trial，並支援 probe 檢索與清理。"""

    def run(
        self, *, trial: TrialConfig, split: str, dataset_path: str, run_label: str, config_hash: str
    ) -> TrialOutcome:
        ...

    def retrieve(
        self, *, collection_name: str, trial: TrialConfig, question: str, top_k: int
    ) -> list[dict[str, Any]]:
        """在既有 collection 上做 probe 檢索（Slice 4 使用）。"""
        ...

    def cleanup(self, collection_name: Optional[str]) -> None:
        ...


@dataclass
class SplitPaths:
    tune: str
    validation: str
    test: Optional[str] = None


# --------------------------------------------------------------------------
# Internal state
# --------------------------------------------------------------------------
@dataclass
class _NodeState:
    """一個 config 節點（Parent 或 Candidate）的執行狀態。"""

    trial: TrialConfig
    config_hash: str
    tune: TrialOutcome
    validation: TrialOutcome

    def collections(self) -> list[str]:
        return [c for c in (self.tune.collection_name, self.validation.collection_name) if c]


@dataclass
class PromotionDecision:
    accepted: bool
    decision: str
    reason: str
    gate: str  # coarse validation gate: pass / fail
    constraint_result: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundRecord:
    round_number: int
    parent_config_hash: str
    candidate_config_hash: Optional[str] = None
    parent_trial_id: Optional[str] = None
    candidate_trial_id: Optional[str] = None
    context_hash: Optional[str] = None
    response_hash: Optional[str] = None
    proposal: Optional[dict[str, Any]] = None
    validation_result: Optional[dict[str, Any]] = None
    executed_config_hash: Optional[str] = None
    tune_trial_id: Optional[str] = None
    validation_trial_id: Optional[str] = None
    probe_comparison: list[dict[str, Any]] = field(default_factory=list)
    metrics_before: dict[str, Any] = field(default_factory=dict)
    metrics_after: dict[str, Any] = field(default_factory=dict)
    constraint_result: dict[str, Any] = field(default_factory=dict)
    promotion_decision: str = PROMOTION_NONE
    promotion_reason: str = ""
    candidate_accepted: bool = False
    confidence: Optional[float] = None
    stop_recommendation: Optional[bool] = None
    stop_decision: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    token_usage: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "parent_config_hash": self.parent_config_hash,
            "candidate_config_hash": self.candidate_config_hash,
            "parent_trial_id": self.parent_trial_id,
            "candidate_trial_id": self.candidate_trial_id,
            "context_hash": self.context_hash,
            "response_hash": self.response_hash,
            "proposal": self.proposal,
            "validation_result": self.validation_result,
            "executed_config_hash": self.executed_config_hash,
            "tune_trial_id": self.tune_trial_id,
            "validation_trial_id": self.validation_trial_id,
            "probe_comparison": list(self.probe_comparison),
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "constraint_result": self.constraint_result,
            "promotion_decision": self.promotion_decision,
            "promotion_reason": self.promotion_reason,
            "candidate_accepted": self.candidate_accepted,
            "confidence": self.confidence,
            "stop_recommendation": self.stop_recommendation,
            "stop_decision": self.stop_decision,
            "provider": self.provider,
            "model": self.model,
            "token_usage": self.token_usage,
            "error": self.error,
        }


@dataclass
class AgentRunResult:
    agent_run_id: str
    status: str
    stop_reason: str
    baseline_config_hash: str
    final_parent_config_hash: str
    run_name: str = ""
    planner_type: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    rounds: list[RoundRecord] = field(default_factory=list)
    llm_calls: int = 0
    candidates_proposed: int = 0
    candidates_promoted: int = 0
    candidates_rejected: int = 0
    probe_count: int = 0
    recommendation_eligibility: str = RECOMMENDATION_NOT_EVALUATED
    residual_collections: list[str] = field(default_factory=list)
    baseline_metric_summary: dict[str, Any] = field(default_factory=dict)
    final_metric_summary: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_run_id": self.agent_run_id,
            "run_name": self.run_name,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "baseline_config_hash": self.baseline_config_hash,
            "final_parent_config_hash": self.final_parent_config_hash,
            "planner_type": self.planner_type,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "llm_calls": self.llm_calls,
            "candidates_proposed": self.candidates_proposed,
            "candidates_promoted": self.candidates_promoted,
            "candidates_rejected": self.candidates_rejected,
            "probe_count": self.probe_count,
            "recommendation_eligibility": self.recommendation_eligibility,
            "residual_collections": list(self.residual_collections),
            "baseline_metric_summary": self.baseline_metric_summary,
            "final_metric_summary": self.final_metric_summary,
            "rounds": [r.to_dict() for r in self.rounds],
            "error": self.error,
        }


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def _hash(trial: TrialConfig) -> str:
    return trial.compute_hash(
        canonical_text_version=CANONICAL_TEXT_VERSION,
        identifier_normalization_version=IDENTIFIER_NORMALIZATION_VERSION,
    )


class AgentOrchestrator:
    """有界 Agent Loop。``probe_runner`` 於 Slice 4 注入；預設不跑 probe。"""

    def __init__(
        self,
        run_config: AgentRunConfig,
        planner: AgentPlanner,
        executor: TrialExecutor,
        *,
        probe_runner: Optional["ProbeRunner"] = None,
        id_factory: Callable[[], str] = None,
    ):
        self._cfg = run_config
        self._planner = planner
        self._executor = executor
        self._validator = ProposalValidator(run_config)
        self._probe_runner = probe_runner
        self._id_factory = id_factory or _default_id
        self._comparator_refs = comparator_metric_refs(
            run_config.promotion.comparator_ranking
        )

    # ------------------------------------------------------------------
    def run(self, *, splits: SplitPaths, agent_run_id: Optional[str] = None) -> AgentRunResult:
        cfg = self._cfg
        run_id = agent_run_id or self._id_factory()
        baseline_hash = _hash(cfg.baseline)
        result = AgentRunResult(
            agent_run_id=run_id,
            status=AgentState.FAILED,
            stop_reason="",
            baseline_config_hash=baseline_hash,
            final_parent_config_hash=baseline_hash,
            run_name=cfg.run_name,
            planner_type=getattr(self._planner, "planner_type", None),
            provider=getattr(self._planner, "provider", None),
            model=getattr(self._planner, "model", None),
            prompt_version=cfg.prompt_version,
        )

        live_collections: set[str] = set()
        tested_hashes: set[str] = {baseline_hash}
        parent: Optional[_NodeState] = None
        prev_summaries: list[PreviousRoundSummary] = []
        consecutive_rejections = 0
        no_improvement = 0

        try:
            # ---- Round 0: baseline on tune + validation ----
            parent = self._run_node(cfg.baseline, baseline_hash, splits, "baseline")
            for c in parent.collections():
                live_collections.add(c)
            if not (parent.tune.ok and parent.validation.ok):
                result.status = AgentState.FAILED
                result.stop_reason = StopReason.BASELINE_FAILED
                result.error = parent.tune.error or parent.validation.error
                result.baseline_metric_summary = _coarse(parent, self._comparator_refs)
                result.final_metric_summary = result.baseline_metric_summary
                return result
            result.baseline_metric_summary = _coarse(parent, self._comparator_refs)

            prev_gate = "not_evaluated"
            stop_reason: Optional[str] = None

            for round_number in range(1, cfg.budget.max_rounds + 1):
                if result.llm_calls >= cfg.budget.max_llm_calls:
                    stop_reason = StopReason.MAX_LLM_CALLS
                    break

                # ---- ANALYZING: build controlled context ----
                context = build_agent_context(
                    run_config=cfg,
                    current=parent.trial,
                    tune_failures=parent.tune.tune_failures,
                    segments=parent.tune.segments,
                    coarse_tune_metrics=_coarse(parent, self._comparator_refs)["tune"],
                    validation_gate=prev_gate,
                    previous_rounds=prev_summaries,
                )

                # ---- PROPOSAL: propose + parse + validate (one bounded retry) ----
                result.llm_calls += 1
                proposal, vres, perr, raw = self._propose_and_validate(
                    context, parent.trial, tested_hashes
                )
                if result.llm_calls < cfg.budget.max_llm_calls and (
                    proposal is None or (vres is not None and not vres.ok)
                ):
                    # bounded retry（對決定性 fake 無效，但對真實 LLM 暫時性失敗有用）
                    result.llm_calls += 1
                    proposal, vres, perr, raw = self._propose_and_validate(
                        context, parent.trial, tested_hashes
                    )

                record = RoundRecord(
                    round_number=round_number,
                    parent_config_hash=parent.config_hash,
                    parent_trial_id=parent.tune.trial_id,
                    context_hash=context.context_hash,
                    response_hash=_sha256(raw) if raw is not None else None,
                    provider=getattr(self._planner, "provider", None),
                    model=getattr(self._planner, "model", None),
                    metrics_before=_coarse(parent, self._comparator_refs),
                )
                if proposal is not None:
                    record.proposal = proposal.to_dict()
                    record.confidence = proposal.confidence
                    record.stop_recommendation = proposal.stop_recommendation
                if vres is not None:
                    record.validation_result = vres.to_dict()

                # ---- invalid proposal handling ----
                if proposal is None or vres is None or not vres.ok:
                    is_duplicate = vres is not None and any(
                        "重複" in e for e in vres.errors
                    )
                    record.promotion_decision = PROMOTION_REJECTED
                    record.promotion_reason = (
                        perr or (vres.errors[0] if vres and vres.errors else "invalid_proposal")
                    )
                    result.rounds.append(record)
                    consecutive_rejections += 1
                    no_improvement += 1
                    prev_summaries.append(
                        _summary(record, prev_gate)
                    )
                    if is_duplicate:
                        stop_reason = StopReason.DUPLICATE_PROPOSAL
                        break
                    if consecutive_rejections >= cfg.budget.max_consecutive_rejections:
                        stop_reason = StopReason.MAX_CONSECUTIVE_REJECTIONS
                        break
                    if no_improvement >= cfg.budget.no_improvement_rounds:
                        stop_reason = StopReason.NO_IMPROVEMENT
                        break
                    continue

                # ---- TRIAL_RUNNING: candidate on tune + validation ----
                candidate = self._run_node(
                    vres.candidate, vres.config_hash, splits, f"round{round_number}"
                )
                for c in candidate.collections():
                    live_collections.add(c)
                tested_hashes.add(vres.config_hash)
                result.candidates_proposed += 1
                record.candidate_config_hash = vres.config_hash
                record.candidate_trial_id = candidate.validation.trial_id
                record.executed_config_hash = vres.config_hash
                record.tune_trial_id = candidate.tune.trial_id
                record.validation_trial_id = candidate.validation.trial_id

                if not (candidate.tune.ok and candidate.validation.ok):
                    # fatal trial failure：清理 candidate、保留 parent、停止。
                    self._cleanup(candidate.collections(), live_collections)
                    record.error = candidate.tune.error or candidate.validation.error
                    record.promotion_decision = PROMOTION_REJECTED
                    record.promotion_reason = "candidate_trial_failed"
                    result.rounds.append(record)
                    prev_summaries.append(_summary(record, prev_gate))
                    stop_reason = StopReason.FATAL_TRIAL_FAILURE
                    break

                # ---- PROBE_RUNNING (Slice 4) ----
                probe_comparison: list[dict[str, Any]] = []
                if self._probe_runner is not None and proposal.probe_questions:
                    probe_comparison = self._probe_runner.compare(
                        parent=parent,
                        candidate=candidate,
                        probes=proposal.probe_questions,
                    )
                    result.probe_count += len(proposal.probe_questions)
                record.probe_comparison = probe_comparison

                # ---- EVALUATING + PROMOTION_DECIDING ----
                record.metrics_after = _coarse(candidate, self._comparator_refs)
                decision = self._decide_promotion(parent, candidate)
                record.constraint_result = decision.constraint_result
                record.promotion_decision = decision.decision
                record.promotion_reason = decision.reason
                record.candidate_accepted = decision.accepted
                prev_gate = decision.gate

                if decision.accepted:
                    # 升級：刪除舊 parent collection，candidate 成為新 parent。
                    old_parent_cols = parent.collections()
                    parent = candidate
                    result.final_parent_config_hash = candidate.config_hash
                    self._cleanup(old_parent_cols, live_collections)
                    result.candidates_promoted += 1
                    consecutive_rejections = 0
                    no_improvement = 0
                else:
                    # 拒絕：保留 parent，刪除 candidate collection。
                    self._cleanup(candidate.collections(), live_collections)
                    result.candidates_rejected += 1
                    consecutive_rejections += 1
                    no_improvement += 1

                result.rounds.append(record)
                prev_summaries.append(_summary(record, prev_gate))

                # ---- stop checks ----
                if proposal.stop_recommendation:
                    stop_reason = StopReason.LLM_RECOMMENDED_STOP
                    break
                if consecutive_rejections >= cfg.budget.max_consecutive_rejections:
                    stop_reason = StopReason.MAX_CONSECUTIVE_REJECTIONS
                    break
                if no_improvement >= cfg.budget.no_improvement_rounds:
                    stop_reason = StopReason.NO_IMPROVEMENT
                    break
                if round_number >= cfg.budget.max_rounds:
                    stop_reason = StopReason.MAX_ROUNDS
                    break

            if stop_reason is None:
                stop_reason = StopReason.MAX_ROUNDS
            result.stop_reason = stop_reason
            result.status = (
                AgentState.COMPLETED
                if stop_reason in _COMPLETED_REASONS
                else AgentState.STOPPED
            )
            if result.rounds:
                result.rounds[-1].stop_decision = stop_reason
            result.final_parent_config_hash = parent.config_hash
            result.final_metric_summary = _coarse(parent, self._comparator_refs)
            return result

        except KeyboardInterrupt:
            result.status = AgentState.STOPPED
            result.stop_reason = StopReason.MANUAL_STOP
            if parent is not None:
                result.final_parent_config_hash = parent.config_hash
                result.final_metric_summary = _coarse(parent, self._comparator_refs)
            return result
        finally:
            # 所有終止路徑：安全清理最後殘留的 collection（含 active parent）。
            self._cleanup(list(live_collections), live_collections)
            result.residual_collections = sorted(live_collections)

    # ------------------------------------------------------------------
    def _propose_and_validate(
        self, context, parent: TrialConfig, tested_hashes: set[str]
    ) -> tuple[Optional[Any], Optional[ValidationResult], Optional[str], Optional[str]]:
        try:
            raw = self._planner.propose(context)
        except Exception as exc:  # noqa: BLE001 - provider/planner 失敗轉明確訊息
            return None, None, f"llm_error:{type(exc).__name__}", None
        try:
            proposal = parse_proposal(raw)
        except ProposalParseError as exc:
            return None, None, f"parse_error:{exc}", raw
        vres = self._validator.validate(
            proposal,
            parent=parent,
            tested_config_hashes=tested_hashes,
            segment_texts=context.segment_texts(),
        )
        return proposal, vres, None, raw

    def _run_node(
        self, trial: TrialConfig, config_hash: str, splits: SplitPaths, label: str
    ) -> _NodeState:
        tune = self._executor.run(
            trial=trial,
            split=SPLIT_TUNE,
            dataset_path=splits.tune,
            run_label=f"{self._cfg.run_name}-{label}-tune",
            config_hash=config_hash,
        )
        validation = self._executor.run(
            trial=trial,
            split=SPLIT_VALIDATION,
            dataset_path=splits.validation,
            run_label=f"{self._cfg.run_name}-{label}-validation",
            config_hash=config_hash,
        )
        return _NodeState(
            trial=trial, config_hash=config_hash, tune=tune, validation=validation
        )

    def _decide_promotion(
        self, parent: _NodeState, candidate: _NodeState
    ) -> PromotionDecision:
        constraints = self._cfg.promotion.hard_constraints
        parent_val = parent.validation.lookup()
        cand_val = candidate.validation.lookup()

        # 條件 3+4：validation hard constraints（absolute）+ non-regression（relative vs parent）。
        cand_eval = evaluate_constraints(constraints, cand_val, baseline=parent_val)
        constraint_dict = cand_eval.to_dict()
        if not cand_eval.feasible:
            return PromotionDecision(
                accepted=False,
                decision=PROMOTION_REJECTED,
                reason="validation_gate_failed: " + "; ".join(cand_eval.rejection_reasons()),
                gate="fail",
                constraint_result=constraint_dict,
            )

        # 條件 5：deterministic comparator（重用 recommendation ordering）—LLM 不參與。
        parent_eval = evaluate_constraints(constraints, parent_val, baseline=parent_val)
        cand_rc = RankableCandidate(
            config_hash=candidate.config_hash,
            metrics=cand_val,
            constraint_eval=cand_eval,
            chunk_config=candidate.trial.chunk_config,
            retrieval_config=candidate.trial.retrieval_config(),
        )
        parent_rc = RankableCandidate(
            config_hash=parent.config_hash,
            metrics=parent_val,
            constraint_eval=parent_eval,
            chunk_config=parent.trial.chunk_config,
            retrieval_config=parent.trial.retrieval_config(),
        )
        cmp, steps = compare_candidates(
            cand_rc, parent_rc, self._cfg.promotion.comparator_ranking, 1e-9
        )
        deciding = next((s for s in steps if s.decided), None)
        beats = (
            cmp < 0 and deciding is not None and deciding.key != TIEBREAK_CONFIG_HASH
        )
        if not beats:
            why = (
                f"decided_by={deciding.key}" if deciding is not None else "no_substantive_difference"
            )
            return PromotionDecision(
                accepted=False,
                decision=PROMOTION_REJECTED,
                reason=f"not_better_than_parent ({why})",
                gate="pass",
                constraint_result=constraint_dict,
            )
        return PromotionDecision(
            accepted=True,
            decision=PROMOTION_PROMOTED,
            reason=f"better_than_parent (decided_by={deciding.key})",
            gate="pass",
            constraint_result=constraint_dict,
        )

    def _cleanup(self, collections: Sequence[str], live: set[str]) -> None:
        for c in collections:
            if not c:
                continue
            try:
                self._executor.cleanup(c)
            except Exception:  # noqa: BLE001 - 清理失敗不得中斷流程
                pass
            live.discard(c)


class ProbeRunner(Protocol):  # Slice 4 實作
    def compare(
        self, *, parent: "_NodeState", candidate: "_NodeState", probes: Sequence[Any]
    ) -> list[dict[str, Any]]:
        ...


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _default_id() -> str:
    import uuid

    return uuid.uuid4().hex


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coarse(node: _NodeState, refs) -> dict[str, Any]:
    """節點的 coarse metric 摘要（tune 給 LLM；validation 僅供報告，不進 LLM context）。"""

    def summarize(lookup: MetricLookup) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for ref in refs:
            v, _, _ = lookup.value(ref)
            out[ref.describe()] = v
        return out

    return {
        "tune": summarize(node.tune.lookup()),
        "validation": summarize(node.validation.lookup()),
    }


def _summary(record: RoundRecord, gate: str) -> PreviousRoundSummary:
    return PreviousRoundSummary(
        round_number=record.round_number,
        proposal_summary={
            "candidate_config_hash": record.candidate_config_hash,
            "parameter_changes": (record.validation_result or {}).get("parameter_changes", []),
        },
        promotion_decision=record.promotion_decision,
        promotion_reason=record.promotion_reason,
        tune_metric_delta=record.metrics_after.get("tune", {}) if record.metrics_after else {},
        validation_gate=gate,
        probe_comparison=tuple(record.probe_comparison),
    )
