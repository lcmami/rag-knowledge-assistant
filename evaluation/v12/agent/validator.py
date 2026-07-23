"""v1.2 Proposal Validator（Slice 1 §四 / §五）。

獨立於 LLM 的政策閘門。**不**信任 LLM 的任何自述；把 config 的結構驗證與 canonical 化
一律委派給既有 v1.1 :class:`ChunkingConfig` / :class:`RetrievalConfig` / :class:`TrialConfig`
（它們已拒絕未知欄位、強制範圍、strategy 欄位適用性、dense 拒絕 identifier 欄位）。

Validator 另外強制 agent 級政策：
- proposed_config 欄位必須落在 ``allowed_parameter_space`` 白名單（比 v1.1 更窄，例如禁止
  ``schema_version``）。
- strategy 必須在允許集合；數值必須在 agent 界限內。
- 參數變更數 <= ``max_parameter_changes``；不得與 Parent 完全相同；不得與已測試 config 重複。
- probe 數 <= ``max_probe_questions``；每個 probe 必須綁定既有 Segment，且 anchor 文字必須
  真的出現在該 Segment（不得臆造 anchor）。

Validator **不**執行任何 config；只回傳 :class:`ValidationResult`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from evaluation.v11.chunking.config import ChunkConfigError, ChunkingConfig
from evaluation.v11.retrieval.config import RetrievalConfig, RetrievalConfigError
from evaluation.v11.run_models import RunConfigError, TrialConfig
from evaluation.v11.trial_runner import (
    CANONICAL_TEXT_VERSION,
    IDENTIFIER_NORMALIZATION_VERSION,
)
from evaluation.v12.agent.proposal import AgentProposal
from evaluation.v12.agent.run_config import (
    CHUNK_BOOL_FIELDS,
    CHUNK_NUMERIC_FIELDS,
    RETRIEVAL_NUMERIC_FIELDS,
    AgentRunConfig,
)

_IDENTIFIER_FIELDS = (
    "identifier_candidate_cap",
    "exact_match_boost",
    "confusable_penalty",
    "identifier_vector_floor",
)
_RETRIEVAL_COMMON = ("top_k", "score_threshold", "candidate_multiplier", "candidate_cap")


@dataclass
class ValidationResult:
    """Validator 結果。``ok`` 為 True 時 ``candidate`` / ``config_hash`` 必為非 None。"""

    ok: bool
    errors: list[str] = field(default_factory=list)
    candidate: Optional[TrialConfig] = None
    config_hash: Optional[str] = None
    parameter_changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "config_hash": self.config_hash,
            "parameter_changes": list(self.parameter_changes),
        }


class ProposalValidator:
    def __init__(self, run_config: AgentRunConfig):
        self._cfg = run_config
        self._space = run_config.allowed_parameter_space
        self._budget = run_config.budget

    # ------------------------------------------------------------------
    def validate(
        self,
        proposal: AgentProposal,
        *,
        parent: TrialConfig,
        tested_config_hashes: set[str],
        segment_texts: Mapping[str, str],
    ) -> ValidationResult:
        errors: list[str] = []

        # 1) agent 白名單欄位（比 v1.1 更窄）。
        errors += self._check_whitelist(
            proposal.proposed_chunk_config, self._space.allowed_chunk_fields(), "chunk_config"
        )
        errors += self._check_whitelist(
            proposal.proposed_retrieval_config,
            self._space.allowed_retrieval_fields(),
            "retrieval_config",
        )

        # 2) strategy 在允許集合。
        chunk_strategy = proposal.proposed_chunk_config.get("strategy")
        if chunk_strategy is not None and chunk_strategy not in self._space.chunk_strategies:
            errors.append(
                f"chunk_config.strategy={chunk_strategy!r} 不在允許集合 "
                f"{list(self._space.chunk_strategies)} 內。"
            )
        retr_strategy = proposal.proposed_retrieval_config.get("strategy")
        if retr_strategy is not None and retr_strategy not in self._space.retrieval_strategies:
            errors.append(
                f"retrieval_config.strategy={retr_strategy!r} 不在允許集合 "
                f"{list(self._space.retrieval_strategies)} 內。"
            )

        # 3) 數值界限（agent bound）。
        errors += self._check_bounds(
            proposal.proposed_chunk_config, CHUNK_NUMERIC_FIELDS, "chunk_config"
        )
        errors += self._check_bounds(
            proposal.proposed_retrieval_config, RETRIEVAL_NUMERIC_FIELDS, "retrieval_config"
        )

        # 4) probe 政策（數量 / segment / anchor 存在性）。
        errors += self._check_probes(proposal, segment_texts)

        # 5) 委派 v1.1 做結構驗證 + canonical 化（即使前面已有錯，也嘗試以取得更多訊息）。
        candidate: Optional[TrialConfig] = None
        try:
            candidate = TrialConfig.from_raw(
                {
                    "trial_name": "candidate",
                    "chunk_config": dict(proposal.proposed_chunk_config),
                    "retrieval_config": dict(proposal.proposed_retrieval_config),
                },
                0,
            )
        except (RunConfigError, ChunkConfigError, RetrievalConfigError) as exc:
            errors.append(f"config 結構驗證失敗（v1.1）：{exc}")

        config_hash: Optional[str] = None
        parameter_changes: list[str] = []
        if candidate is not None:
            config_hash = candidate.compute_hash(
                canonical_text_version=CANONICAL_TEXT_VERSION,
                identifier_normalization_version=IDENTIFIER_NORMALIZATION_VERSION,
            )
            parent_hash = parent.compute_hash(
                canonical_text_version=CANONICAL_TEXT_VERSION,
                identifier_normalization_version=IDENTIFIER_NORMALIZATION_VERSION,
            )
            parameter_changes = _diff_config(parent, candidate)

            if config_hash == parent_hash:
                errors.append("proposed_config 與 Parent 完全相同（無變更）。")
            elif not parameter_changes:  # pragma: no cover - hash 不同必有變更
                errors.append("proposed_config 與 Parent 無可辨識的參數變更。")
            if len(parameter_changes) > self._budget.max_parameter_changes:
                errors.append(
                    f"參數變更數 {len(parameter_changes)} 超過上限 "
                    f"{self._budget.max_parameter_changes}：{parameter_changes}。"
                )
            if config_hash in tested_config_hashes:
                errors.append(f"proposed_config 與已測試 config 重複（hash={config_hash[:12]}…）。")

        ok = not errors
        return ValidationResult(
            ok=ok,
            errors=errors,
            candidate=candidate if ok else None,
            config_hash=config_hash if ok else None,
            parameter_changes=parameter_changes,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _check_whitelist(
        cfg: dict[str, Any], allowed: frozenset[str], where: str
    ) -> list[str]:
        unknown = set(cfg) - allowed
        if unknown:
            return [
                f"{where} 含白名單外欄位（LLM 不得設定）：{sorted(unknown)}；"
                f"允許：{sorted(allowed)}。"
            ]
        return []

    def _check_bounds(
        self, cfg: dict[str, Any], numeric_fields: tuple[str, ...], where: str
    ) -> list[str]:
        errors: list[str] = []
        for f in numeric_fields:
            if f not in cfg:
                continue
            val = cfg[f]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                # 型別交由 v1.1 驗證報錯；此處只做界限。
                continue
            bound = self._space.bound_for(f)
            if bound is None:
                continue
            lo, hi = bound
            if not (lo <= float(val) <= hi):
                errors.append(
                    f"{where}.{f}={val} 超出 agent 界限 [{lo}, {hi}]。"
                )
        return errors

    def _check_probes(
        self, proposal: AgentProposal, segment_texts: Mapping[str, str]
    ) -> list[str]:
        errors: list[str] = []
        if len(proposal.probe_questions) > self._budget.max_probe_questions:
            errors.append(
                f"probe 數 {len(proposal.probe_questions)} 超過上限 "
                f"{self._budget.max_probe_questions}。"
            )
        for i, probe in enumerate(proposal.probe_questions):
            where = f"probe_questions[{i}]"
            if not probe.source_segment_ids:
                errors.append(f"{where} 必須綁定至少一個 source_segment_id。")
                continue
            unknown_segs = [s for s in probe.source_segment_ids if s not in segment_texts]
            if unknown_segs:
                errors.append(f"{where} 參照不存在的 segment：{unknown_segs}。")
                continue
            corpus_text = "\n".join(segment_texts[s] for s in probe.source_segment_ids)
            for anchor in probe.anchor_text_suggestions:
                if anchor and anchor not in corpus_text:
                    errors.append(
                        f"{where} anchor 文字未出現在所綁定 segment 內（不得臆造 anchor）：{anchor!r}。"
                    )
        return errors


# --------------------------------------------------------------------------
def _diff_config(parent: TrialConfig, candidate: TrialConfig) -> list[str]:
    """列出 Parent → Candidate 的有效參數變更（strategy 切換記為 1 個變更）。

    identifier-only 欄位只有在兩者都是 identifier_aware 時才逐一比較；strategy 由 dense
    切到 identifier_aware 時，identifier 欄位屬 strategy 的內含變化，不重複計數。
    """
    changes: list[str] = []
    pc, cc = parent.chunk_config, candidate.chunk_config
    pr, cr = parent.retrieval, candidate.retrieval

    if pc.get("strategy") != cc.get("strategy"):
        changes.append(f"chunk.strategy: {pc.get('strategy')}→{cc.get('strategy')}")
    for f in CHUNK_NUMERIC_FIELDS + CHUNK_BOOL_FIELDS:
        if pc.get(f) != cc.get(f):
            changes.append(f"chunk.{f}: {pc.get(f)}→{cc.get(f)}")

    if pr.get("strategy") != cr.get("strategy"):
        changes.append(f"retrieval.strategy: {pr.get('strategy')}→{cr.get('strategy')}")
    for f in _RETRIEVAL_COMMON:
        if pr.get(f) != cr.get(f):
            changes.append(f"retrieval.{f}: {pr.get(f)}→{cr.get(f)}")
    if pr.get("strategy") == "identifier_aware" and cr.get("strategy") == "identifier_aware":
        for f in _IDENTIFIER_FIELDS:
            if pr.get(f) != cr.get(f):
                changes.append(f"retrieval.{f}: {pr.get(f)}→{cr.get(f)}")
    return changes
