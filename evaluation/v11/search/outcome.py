"""Test 驗收結果語意與 Recommendation 證據強度（Slice 5A §二～§六）。

把兩件先前被混為一談的事**明確分開**：

1. :data:`test_constraint_status` —— recommendation 在 test split 上是否滿足**絕對** hard constraint。
2. :data:`baseline_comparison_status` —— recommendation 相對 **baseline** 是進步、持平、退步或互有優劣。

「兩者都沒有通過絕對門檻」**不等於**「recommendation 相對 baseline 退化」。

另外揭露證據強度（§三～§六）：apply eligibility、metric 樣本粒度、
identifier-bearing unanswerable 的覆蓋缺口、以及 latency tie-break 的不穩定性。

本模組**不會**改變任何 recommendation：它只描述已鎖定的結果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from evaluation.v11.search.constraints import (
    STATUS_FAIL,
    ConstraintEvaluation,
    MetricLookup,
    evaluate_constraints,
)
from evaluation.v11.search.search_config import (
    DIRECTION_MAXIMIZE,
    DIRECTION_MINIMIZE,
    HardConstraint,
    MetricRef,
    ParetoObjective,
)

# -- test constraint status（§2.1）--
TEST_CONSTRAINT_PASSED = "passed"
TEST_CONSTRAINT_FAILED = "failed"
TEST_CONSTRAINT_NOT_EVALUATED = "not_evaluated"

# -- baseline comparison status（§2.2）--
CMP_IMPROVED = "improved"
CMP_SAME = "same"
CMP_REGRESSED = "regressed"
CMP_MIXED = "mixed"
CMP_NOT_EVALUATED = "not_evaluated"

# -- recommendation selection basis（§三 / §六）--
BASIS_QUALITY_RANKING = "validation_quality_ranking"
BASIS_LATENCY_TIEBREAK = "validation_latency_tiebreak"
BASIS_NOT_SELECTED = "no_recommendation"

#: 名稱含這些片段的 metric 屬「越小越好」。
_LOWER_IS_BETTER_HINTS = (
    "latency", "false_positive", "error", "duplicate", "non_computable",
)
#: 名稱含這些片段的 metric 視為「延遲類」，不計入 retrieval-quality 平手判定。
_LATENCY_HINTS = ("latency",)

#: 本 Slice 的 latency 量測次數（每個 config 只跑一次）。
LATENCY_MEASUREMENT_REPETITIONS = 1


def metric_direction(
    ref: MetricRef, objectives: Sequence[ParetoObjective] = ()
) -> str:
    """優先採用 config 明示的 objective 方向，否則依名稱推斷。"""
    for o in objectives:
        if o.ref == ref:
            return o.direction
    lowered = ref.metric.lower()
    if any(h in lowered for h in _LOWER_IS_BETTER_HINTS):
        return DIRECTION_MINIMIZE
    return DIRECTION_MAXIMIZE


def is_latency_metric(ref: MetricRef) -> bool:
    return any(h in ref.metric.lower() for h in _LATENCY_HINTS)


# --------------------------------------------------------------------------
# §2.1 Test constraint status
# --------------------------------------------------------------------------
@dataclass
class TestConstraintOutcome:
    status: str
    failed_test_constraints: list[str] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_constraint_status": self.status,
            "failed_test_constraints": list(self.failed_test_constraints),
            "results": list(self.results),
            # 這些 dict 會被 build_outcome 合併成單一 outcome，因此 note key 必須各自唯一，
            # 否則後合併者會蓋掉前者（§二／§三／§六 的揭露文字會靜默消失）。
            "test_constraint_note": self.note,
        }


def evaluate_test_constraints(
    constraints: Sequence[HardConstraint],
    recommendation: Optional[MetricLookup],
    baseline: Optional[MetricLookup],
) -> TestConstraintOutcome:
    """在 test split 上重跑同一組 hard constraint（僅供驗收，不改 recommendation）。

    對 relative constraint 而言 ``baseline`` 是 test split 的 baseline metric。
    """
    if recommendation is None:
        return TestConstraintOutcome(
            status=TEST_CONSTRAINT_NOT_EVALUATED,
            note="test 未執行或 recommendation 無 test metric。",
        )
    ce: ConstraintEvaluation = evaluate_constraints(constraints, recommendation, baseline)
    failed = [r.name for r in ce.results if r.status == STATUS_FAIL]
    return TestConstraintOutcome(
        status=TEST_CONSTRAINT_FAILED if failed else TEST_CONSTRAINT_PASSED,
        failed_test_constraints=failed,
        results=[r.to_dict() for r in ce.results],
        note=(
            "test constraint 為**絕對**門檻。即使 baseline 同樣未達標，"
            "recommendation 未達標仍記為 failed —— 這與『相對 baseline 退化』是不同概念。"
            if failed
            else "recommendation 在 test split 上滿足全部 hard constraint。"
        ),
    )


# --------------------------------------------------------------------------
# §2.2 Baseline comparison
# --------------------------------------------------------------------------
@dataclass
class MetricComparison:
    metric: str
    scope: str
    query_type: Optional[str]
    baseline_value: Optional[float]
    recommendation_value: Optional[float]
    delta: Optional[float]
    direction: str
    comparison: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "scope": self.scope,
            "query_type": self.query_type,
            "baseline_value": self.baseline_value,
            "recommendation_value": self.recommendation_value,
            "delta": self.delta,
            "direction": self.direction,
            "comparison": self.comparison,
        }


@dataclass
class BaselineComparison:
    status: str
    metrics: list[MetricComparison] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_comparison_status": self.status,
            "metrics": [m.to_dict() for m in self.metrics],
            "baseline_comparison_note": self.note,
        }


def compare_to_baseline(
    refs: Sequence[MetricRef],
    baseline: Optional[MetricLookup],
    recommendation: Optional[MetricLookup],
    objectives: Sequence[ParetoObjective] = (),
    *,
    tolerance: float = 1e-9,
) -> BaselineComparison:
    """逐 metric 比較 baseline 與 recommendation（§2.2）。"""
    if baseline is None or recommendation is None:
        return BaselineComparison(
            status=CMP_NOT_EVALUATED, note="baseline 或 recommendation 缺少 test metric。"
        )

    rows: list[MetricComparison] = []
    seen: set[tuple[str, str, Optional[str]]] = set()
    for ref in refs:
        key = (ref.metric, ref.scope, ref.query_type)
        if key in seen:
            continue
        seen.add(key)
        b, b_reason, _ = baseline.value(ref)
        r, r_reason, _ = recommendation.value(ref)
        direction = metric_direction(ref, objectives)
        if b_reason is not None or r_reason is not None:
            rows.append(MetricComparison(
                ref.metric, ref.scope, ref.query_type, b, r, None,
                direction, CMP_NOT_EVALUATED))
            continue
        delta = r - b
        if abs(delta) <= tolerance:
            cmp_ = CMP_SAME
        elif direction == DIRECTION_MAXIMIZE:
            cmp_ = CMP_IMPROVED if delta > 0 else CMP_REGRESSED
        else:
            cmp_ = CMP_IMPROVED if delta < 0 else CMP_REGRESSED
        rows.append(MetricComparison(
            ref.metric, ref.scope, ref.query_type, b, r, delta, direction, cmp_))

    rows.sort(key=lambda m: (m.metric, m.scope, m.query_type or ""))
    verdicts = {m.comparison for m in rows if m.comparison != CMP_NOT_EVALUATED}
    if not verdicts:
        status = CMP_NOT_EVALUATED
    elif verdicts == {CMP_SAME}:
        status = CMP_SAME
    elif CMP_IMPROVED in verdicts and CMP_REGRESSED in verdicts:
        status = CMP_MIXED
    elif CMP_IMPROVED in verdicts:
        status = CMP_IMPROVED
    else:
        status = CMP_REGRESSED

    return BaselineComparison(
        status=status,
        metrics=rows,
        note=(
            "此狀態只描述 recommendation 與 baseline 的相對關係；"
            "是否通過絕對門檻請見 test_constraint_status。"
        ),
    )


# --------------------------------------------------------------------------
# §六 Quality tie / latency 穩定性
# --------------------------------------------------------------------------
@dataclass
class SelectionAnalysis:
    quality_tie_candidate_count: int
    latency_measurement_repetitions: int
    latency_selection_stable: bool
    selection_basis: str
    latency_order_reversed_on_test: Optional[bool] = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_tie_candidate_count": self.quality_tie_candidate_count,
            "latency_measurement_repetitions": self.latency_measurement_repetitions,
            "latency_selection_stable": self.latency_selection_stable,
            "recommendation_selection_basis": self.selection_basis,
            "latency_order_reversed_on_test": self.latency_order_reversed_on_test,
            "selection_note": self.note,
        }


def analyze_selection(
    winner_hash: Optional[str],
    feasible_metrics: dict[str, MetricLookup],
    objectives: Sequence[ParetoObjective],
    *,
    tolerance: float = 1e-9,
    latency_order_reversed_on_test: Optional[bool] = None,
) -> SelectionAnalysis:
    """判斷 recommendation 是靠品質勝出，還是只靠 latency tie-break（§六）。"""
    if not winner_hash or winner_hash not in feasible_metrics:
        return SelectionAnalysis(
            quality_tie_candidate_count=0,
            latency_measurement_repetitions=LATENCY_MEASUREMENT_REPETITIONS,
            latency_selection_stable=False,
            selection_basis=BASIS_NOT_SELECTED,
            note="沒有選出 recommendation。",
        )

    quality = [o for o in objectives if not is_latency_metric(o.ref)]

    def vec(lookup: MetricLookup) -> Optional[tuple]:
        out: list[float] = []
        for o in quality:
            v, reason, _ = lookup.value(o.ref)
            if reason is not None:
                return None
            out.append(float(v))
        return tuple(out)

    win_vec = vec(feasible_metrics[winner_hash])
    tied = 0
    if win_vec is not None:
        for lookup in feasible_metrics.values():
            v = vec(lookup)
            if v is not None and len(v) == len(win_vec) and all(
                abs(a - b) <= tolerance for a, b in zip(v, win_vec)
            ):
                tied += 1

    has_latency_objective = any(is_latency_metric(o.ref) for o in objectives)
    if tied > 1 and has_latency_objective:
        basis = BASIS_LATENCY_TIEBREAK
        stable = False
        # §六 要求的揭露文字（英文原文照錄，數量與反轉事實依實測填入）。
        note = (
            f"All {tied} feasible validation candidates were tied on the configured "
            "retrieval-quality objectives.\n\n"
            "The selected recommendation was determined by measured P95 latency. "
            "The latency ranking was based on a small, single-machine run"
            + (
                " and reversed relative ordering on the test split"
                if latency_order_reversed_on_test
                else ""
            )
            + ", so it should be treated as unstable.\n\n"
            "不得宣稱此 candidate 在 retrieval quality 上勝過其他 candidate；"
            "它只是此次 deterministic policy 在弱證據下選出的 candidate。"
        )
    else:
        basis = BASIS_QUALITY_RANKING
        stable = True
        note = "recommendation 由 retrieval-quality 指標分出勝負。"

    return SelectionAnalysis(
        quality_tie_candidate_count=tied,
        latency_measurement_repetitions=LATENCY_MEASUREMENT_REPETITIONS,
        latency_selection_stable=stable,
        selection_basis=basis,
        latency_order_reversed_on_test=latency_order_reversed_on_test,
        note=note,
    )


# --------------------------------------------------------------------------
# §三 Apply eligibility
# --------------------------------------------------------------------------
REASON_SMALL_VALIDATION_SAMPLE = "small_validation_sample"
REASON_QUALITY_METRICS_TIED = "quality_metrics_tied"
REASON_SINGLE_RUN_LATENCY = "selection_driven_by_single_run_latency"
REASON_TEST_CONSTRAINT_FAILED = "test_constraint_failed"
REASON_DEMO_CORPUS_ONLY = "demo_corpus_only"
REASON_NO_RECOMMENDATION = "no_recommendation"

#: 低於此樣本數即視為小樣本（constraint 的分母）。
SMALL_SAMPLE_THRESHOLD = 5


@dataclass
class ApplyEligibility:
    apply_eligible: bool
    confidence: str
    reasons: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_apply_eligible": self.apply_eligible,
            "recommendation_confidence": self.confidence,
            "apply_ineligible_reasons": list(self.reasons),
            "apply_eligibility_note": self.note,
        }


def assess_apply_eligibility(
    *,
    recommended_config_hash: Optional[str],
    validation_constraints: Sequence[ConstraintEvaluation],
    selection: SelectionAnalysis,
    test_outcome: TestConstraintOutcome,
) -> ApplyEligibility:
    """判定此 recommendation 是否有足夠證據進入**未來**的 Apply（§三）。

    ``apply_eligible=False`` **不**取消 recommendation lock：
    ``recommended_config_hash`` 仍然存在且固定，只是證據不足以套用。
    """
    reasons: list[str] = []

    if not recommended_config_hash:
        reasons.append(REASON_NO_RECOMMENDATION)

    for ce in validation_constraints:
        if any(
            r.sample_count is not None and r.sample_count < SMALL_SAMPLE_THRESHOLD
            for r in ce.results
        ):
            reasons.append(REASON_SMALL_VALIDATION_SAMPLE)
            break

    if selection.quality_tie_candidate_count > 1:
        reasons.append(REASON_QUALITY_METRICS_TIED)
    if (
        selection.selection_basis == BASIS_LATENCY_TIEBREAK
        and selection.latency_measurement_repetitions <= 1
    ):
        reasons.append(REASON_SINGLE_RUN_LATENCY)
    if test_outcome.status == TEST_CONSTRAINT_FAILED:
        reasons.append(REASON_TEST_CONSTRAINT_FAILED)

    # Slice 5 的 optimizer corpus 綁定 tracked demo corpus，永遠不足以直接套用 live corpus。
    reasons.append(REASON_DEMO_CORPUS_ONLY)

    return ApplyEligibility(
        apply_eligible=False if reasons else True,
        confidence="low" if reasons else "medium",
        reasons=reasons,
        note=(
            "apply_eligible=false 不代表取消 recommendation lock："
            "recommended_config_hash 仍為 validation policy 選出的固定 candidate。"
            "它代表目前證據不足以進入未來的 Apply；Slice 7 必須拒絕 apply_eligible=false 的 candidate，"
            "除非後續 evaluation dataset／run 重新驗證通過。"
        ),
    )


# --------------------------------------------------------------------------
# §四 Metric 樣本粒度
# --------------------------------------------------------------------------
def _possible_rates(n: int) -> list[float]:
    return [round(i / n, 6) for i in range(n + 1)] if n > 0 else []


def sample_granularity(
    split_query_type_counts: dict[str, dict[str, int]],
    constraints: Sequence[HardConstraint] = (),
) -> dict[str, Any]:
    """揭露每個 split 上各 query type 的樣本數與**可能取值**（§四）。"""
    rows: list[dict[str, Any]] = []
    for split in sorted(split_query_type_counts):
        counts = split_query_type_counts[split]
        for qt in sorted(counts):
            n = counts[qt]
            rows.append({
                "split": split,
                "query_type": qt,
                "n": n,
                "possible_values": _possible_rates(n),
            })

    notes: list[str] = []
    test_exact = split_query_type_counts.get("test", {}).get("exact_code", 0)
    for c in constraints:
        if c.ref.metric == "exact_code_top1_accuracy" and c.value is not None and test_exact:
            needed = c.value * test_exact
            if needed > test_exact - 1:
                notes.append(
                    f"A threshold of {c.value:g} with {test_exact} exact-code test "
                    f"cases effectively requires {test_exact}/{test_exact} success. "
                    "The result is computable but has low statistical confidence."
                )
    notes.append(
        "指標可計算不等於樣本充足；上列 possible_values 顯示這些比率只能取到極少數離散值。"
    )
    return {"per_split": rows, "notes": notes}


# --------------------------------------------------------------------------
# §五 identifier-bearing unanswerable 覆蓋缺口
# --------------------------------------------------------------------------
def identifier_unanswerable_coverage(
    unanswerable_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """揭露各 split 的 unanswerable 題是否帶有 identifier（§五）。

    ``unanswerable_by_split`` 形如 ``{split: [{"case_id":..., "identifiers":[...]}, ...]}``。
    """
    per_split = {}
    for split in sorted(unanswerable_by_split):
        rows = unanswerable_by_split[split]
        with_id = [r for r in rows if r.get("identifiers")]
        per_split[split] = {
            "unanswerable_case_count": len(rows),
            "identifier_bearing_count": len(with_id),
            "cases": rows,
        }

    val = per_split.get("validation", {})
    gap = val.get("unanswerable_case_count", 0) > 0 and val.get("identifier_bearing_count", 0) == 0
    return {
        "per_split": per_split,
        "validation_has_identifier_bearing_unanswerable": not gap,
        "limitation": (
            "The validation split contains unanswerable cases, so the aggregate "
            "unanswerable false-positive constraint is computable.\n\n"
            "However, those validation cases do not contain an identifier that exists "
            "in the corpus. They therefore do not directly exercise the permissive "
            "identifier_vector_floor failure mode observed for identifier-bearing "
            "unanswerable queries."
        ) if gap else (
            "validation split 已包含帶 identifier 的 unanswerable case。"
        ),
        "required_followup": (
            "後續 dataset 需新增至少一個 identifier-bearing unanswerable validation case"
            "（文件確實包含某設備代碼，但詢問文件未記載的價格／供應商／保固等屬性），"
            "並新增未被目前開發過程觀察過的 held-out test group，"
            "才能重新聲稱真正的 test 驗收。"
        ) if gap else "",
    }
