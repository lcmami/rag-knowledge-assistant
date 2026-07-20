"""Hard Constraint Engine（Slice 5 §十四）。

支援 absolute 與 relative-to-baseline 兩類限制，並嚴格處理「不可計算」：

- **不可計算 ≠ 通過**（§14.5）。每個 constraint 依 ``on_not_computable`` 決定 fail / warn / skip，
  safety-critical 者強制 fail（於 :mod:`search_config` 驗證階段就擋下不合規設定）。
- 判定一律使用**完整精度浮點**（§14.10）；四捨五入只發生在報告呈現層。
- 每個結果保存 observed / baseline / required / pass-fail / sample count / reason（§14.8）。

Constraint 一律以 **validation split** 的 metric 判定（§14.1）；tune 只做預篩（§十六），
呼叫端負責傳入正確 split 的 metric，本模組不自行決定 split。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from evaluation.v11.run_models import TrialMetric
from evaluation.v11.search import metric_names as mn
from evaluation.v11.search.search_config import (
    NOT_COMPUTABLE_FAIL,
    NOT_COMPUTABLE_SKIP,
    NOT_COMPUTABLE_WARN,
    HardConstraint,
    MetricRef,
)

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_WARN = "warn"
STATUS_SKIP = "skip"


# --------------------------------------------------------------------------
# Metric 查找
# --------------------------------------------------------------------------
class MetricLookup:
    """以 (metric_name, scope, query_type) 查找某個 trial 的 :class:`TrialMetric`。"""

    def __init__(self, metrics: Iterable[TrialMetric]):
        self._by_key: dict[tuple[str, str, str], TrialMetric] = {}
        for m in metrics:
            key = (m.metric_name, m.scope, m.query_type or "")
            self._by_key[key] = m

    def find(self, ref: MetricRef) -> Optional[TrialMetric]:
        return self._by_key.get((ref.metric, ref.scope, ref.query_type or ""))

    def value(self, ref: MetricRef) -> tuple[Optional[float], Optional[str], Optional[int]]:
        """回傳 ``(value, not_computable_reason, sample_count)``。

        三態嚴格區分：row 不存在 → ``metric_absent_for_trial``；row 存在但
        ``computable=False`` → 沿用其 reason；否則回傳完整精度的 value。
        """
        m = self.find(ref)
        if m is None:
            return None, mn.REASON_METRIC_ABSENT, None
        if not m.computable or m.value is None:
            return None, m.not_computable_reason or "not_computable", m.sample_count
        return float(m.value), None, m.sample_count


# --------------------------------------------------------------------------
# 比較
# --------------------------------------------------------------------------
def compare(observed: float, operator: str, required: float) -> bool:
    """完整精度比較（§14.10：不得使用四捨五入後的顯示值）。"""
    if operator == ">=":
        return observed >= required
    if operator == ">":
        return observed > required
    if operator == "<=":
        return observed <= required
    if operator == "<":
        return observed < required
    if operator == "==":
        return observed == required
    raise ValueError(f"未知 operator：{operator!r}")


# --------------------------------------------------------------------------
# 結果
# --------------------------------------------------------------------------
@dataclass
class ConstraintResult:
    """單一 constraint 的判定結果（§14.8 全欄位保存）。"""

    name: str
    metric: str
    scope: str
    query_type: Optional[str]
    operator: str
    status: str
    passed: bool
    observed_value: Optional[float] = None
    baseline_value: Optional[float] = None
    required_value: Optional[float] = None
    sample_count: Optional[int] = None
    baseline_sample_count: Optional[int] = None
    is_relative: bool = False
    baseline_delta: Optional[float] = None
    safety_critical: bool = True
    on_not_computable: str = NOT_COMPUTABLE_FAIL
    computable: bool = True
    reason: Optional[str] = None

    @property
    def counts_as_pass(self) -> bool:
        """供 tune ranking 的 ``constraint_pass_count`` 使用。

        只有**真正計算過且通過**才計數；warn / skip 雖不阻擋可行性，但不算「通過」，
        否則不可計算的 candidate 會在排序上獲得不應得的優勢。
        """
        return self.status == STATUS_PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "scope": self.scope,
            "query_type": self.query_type,
            "operator": self.operator,
            "status": self.status,
            "passed": self.passed,
            "observed_value": self.observed_value,
            "baseline_value": self.baseline_value,
            "required_value": self.required_value,
            "sample_count": self.sample_count,
            "baseline_sample_count": self.baseline_sample_count,
            "is_relative": self.is_relative,
            "baseline_delta": self.baseline_delta,
            "safety_critical": self.safety_critical,
            "on_not_computable": self.on_not_computable,
            "computable": self.computable,
            "reason": self.reason,
        }


def _not_computable_result(
    c: HardConstraint,
    reason: str,
    *,
    observed: Optional[float] = None,
    baseline: Optional[float] = None,
    required: Optional[float] = None,
    sample_count: Optional[int] = None,
    baseline_sample_count: Optional[int] = None,
) -> ConstraintResult:
    """§14.5 / §14.6：不可計算依政策決定，**絕不默認通過**。"""
    if c.on_not_computable == NOT_COMPUTABLE_FAIL:
        status, passed = STATUS_FAIL, False
    elif c.on_not_computable == NOT_COMPUTABLE_WARN:
        status, passed = STATUS_WARN, True
    elif c.on_not_computable == NOT_COMPUTABLE_SKIP:
        status, passed = STATUS_SKIP, True
    else:  # pragma: no cover - schema 已擋
        status, passed = STATUS_FAIL, False
    return ConstraintResult(
        name=c.name,
        metric=c.ref.metric,
        scope=c.ref.scope,
        query_type=c.ref.query_type,
        operator=c.operator,
        status=status,
        passed=passed,
        observed_value=observed,
        baseline_value=baseline,
        required_value=required,
        sample_count=sample_count,
        baseline_sample_count=baseline_sample_count,
        is_relative=c.is_relative,
        baseline_delta=c.baseline_delta,
        safety_critical=c.safety_critical,
        on_not_computable=c.on_not_computable,
        computable=False,
        reason=reason,
    )


def evaluate_constraint(
    c: HardConstraint,
    candidate: MetricLookup,
    baseline: Optional[MetricLookup] = None,
) -> ConstraintResult:
    """判定單一 constraint。

    ``baseline`` 只有 relative constraint 需要；absolute constraint 會忽略它。
    """
    observed, obs_reason, obs_n = candidate.value(c.ref)
    if obs_reason is not None:
        return _not_computable_result(
            c, f"candidate metric 不可計算：{obs_reason}", sample_count=obs_n
        )

    if not c.is_relative:
        required = float(c.value)  # type: ignore[arg-type]
        passed = compare(observed, c.operator, required)
        return ConstraintResult(
            name=c.name,
            metric=c.ref.metric,
            scope=c.ref.scope,
            query_type=c.ref.query_type,
            operator=c.operator,
            status=STATUS_PASS if passed else STATUS_FAIL,
            passed=passed,
            observed_value=observed,
            baseline_value=None,
            required_value=required,
            sample_count=obs_n,
            is_relative=False,
            baseline_delta=None,
            safety_critical=c.safety_critical,
            on_not_computable=c.on_not_computable,
            computable=True,
            reason=None,
        )

    # -- relative-to-baseline --
    if baseline is None:
        return _not_computable_result(
            c,
            "relative constraint 缺少 baseline metric（未提供 baseline lookup）",
            observed=observed,
            sample_count=obs_n,
        )
    base_value, base_reason, base_n = baseline.value(c.ref)
    if base_reason is not None:
        return _not_computable_result(
            c,
            f"baseline metric 不可計算：{base_reason}",
            observed=observed,
            sample_count=obs_n,
            baseline_sample_count=base_n,
        )

    required = base_value + float(c.baseline_delta)  # type: ignore[arg-type]
    passed = compare(observed, c.operator, required)
    return ConstraintResult(
        name=c.name,
        metric=c.ref.metric,
        scope=c.ref.scope,
        query_type=c.ref.query_type,
        operator=c.operator,
        status=STATUS_PASS if passed else STATUS_FAIL,
        passed=passed,
        observed_value=observed,
        baseline_value=base_value,
        required_value=required,
        sample_count=obs_n,
        baseline_sample_count=base_n,
        is_relative=True,
        baseline_delta=c.baseline_delta,
        safety_critical=c.safety_critical,
        on_not_computable=c.on_not_computable,
        computable=True,
        reason=None,
    )


@dataclass
class ConstraintEvaluation:
    """一個 candidate 對整組 constraint 的判定結果。"""

    results: list[ConstraintResult]

    @property
    def feasible(self) -> bool:
        """所有 constraint 都未阻擋（fail 即不可行）。"""
        return all(r.passed for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.counts_as_pass)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status == STATUS_FAIL)

    def failed_names(self) -> list[str]:
        return [r.name for r in self.results if r.status == STATUS_FAIL]

    def rejection_reasons(self) -> list[str]:
        out: list[str] = []
        for r in self.results:
            if r.status != STATUS_FAIL:
                continue
            if not r.computable:
                out.append(f"{r.name}：不可計算（{r.reason}），政策 on_not_computable=fail")
            else:
                out.append(
                    f"{r.name}：observed={r.observed_value!r} "
                    f"未滿足 {r.operator} {r.required_value!r}"
                )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "failed": self.failed_names(),
            "rejection_reasons": self.rejection_reasons(),
            "results": [r.to_dict() for r in self.results],
        }


def evaluate_constraints(
    constraints: Iterable[HardConstraint],
    candidate: MetricLookup,
    baseline: Optional[MetricLookup] = None,
) -> ConstraintEvaluation:
    """依 config 順序判定整組 constraint（順序 deterministic）。"""
    return ConstraintEvaluation(
        results=[evaluate_constraint(c, candidate, baseline) for c in constraints]
    )
