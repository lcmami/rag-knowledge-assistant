"""tune / validation / test 嚴格隔離（Slice 5 §十二、§十三）。

每個 split 產出一份**獨立的 dataset 檔**，Trial Runner 只看得到該 split 的 case ——
這是最強的隔離：不是靠呼叫端記得過濾，而是 runner 物理上拿不到其他 split 的題目。

同時提供：

- Group leakage 檢查（§13.1，同 ``group_id`` 只能存在於一個 split）。
- Split coverage 分析（§五）：各 split 的 case / group 數與 query type 覆蓋。
- Constraint 可計算性**事前預測**（§五 3）：某些 metric 需要特定 query type 才存在，
  若 validation split 缺少該 query type，該 constraint 在 validation 上永遠不可計算。
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from evaluation.v11.case_schema import EvaluationCaseV11, load_dataset, parse_dataset
from evaluation.v11.hashing import dataset_hash
from evaluation.v11.search import metric_names as mn
from evaluation.v11.search.search_config import HardConstraint

SPLIT_TUNE = "tune"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
ALL_SPLITS = (SPLIT_TUNE, SPLIT_VALIDATION, SPLIT_TEST)

#: §五：每個 split 期望具備的 query type（缺少時必須誠實回報，不得偷補）。
EXPECTED_QUERY_TYPES = ("exact_code", "semantic", "unanswerable")

#: 缺乏 validation 覆蓋時的狀態碼（§五）。
STATUS_INSUFFICIENT_VALIDATION_COVERAGE = "insufficient_validation_coverage"

#: metric → 該 metric 需要哪些 query type 才可能可計算（aggregate scope）。
#: 未列出的 metric 代表不依賴特定 query type（或無法事前判定），預測為 ``None``（未知）。
_AGGREGATE_METRIC_REQUIRES: dict[str, frozenset[str]] = {
    "exact_code_top1_accuracy": frozenset({"exact_code"}),
    "confusable_code_error_rate": frozenset({"confusable"}),
    "confusable_code_error_count": frozenset({"confusable"}),
    "unanswerable_false_positive_rate": frozenset({"unanswerable"}),
    "unanswerable_false_positive_count": frozenset({"unanswerable"}),
}


@dataclass
class SplitInfo:
    split: str
    case_ids: list[str]
    group_ids: list[str]
    query_type_counts: dict[str, int]
    answerable_count: int
    unanswerable_count: int
    dataset_path: Optional[str] = None
    dataset_hash: Optional[str] = None

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    @property
    def group_count(self) -> int:
        return len(self.group_ids)

    def missing_expected_query_types(self) -> list[str]:
        return [q for q in EXPECTED_QUERY_TYPES if self.query_type_counts.get(q, 0) == 0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "case_count": self.case_count,
            "group_count": self.group_count,
            "case_ids": list(self.case_ids),
            "group_ids": list(self.group_ids),
            "query_type_counts": dict(self.query_type_counts),
            "answerable_count": self.answerable_count,
            "unanswerable_count": self.unanswerable_count,
            "missing_expected_query_types": self.missing_expected_query_types(),
            "dataset_path": self.dataset_path,
            "dataset_hash": self.dataset_hash,
        }


@dataclass
class SplitManifest:
    dataset_version: str
    full_dataset_hash: str
    splits: dict[str, SplitInfo]
    leakage_ok: bool = True
    leakage_detail: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version": self.dataset_version,
            "full_dataset_hash": self.full_dataset_hash,
            "group_leakage_ok": self.leakage_ok,
            "group_leakage_detail": list(self.leakage_detail),
            "splits": {s: self.splits[s].to_dict() for s in sorted(self.splits)},
        }


class SplitError(Exception):
    """Split 準備 / 洩漏檢查錯誤。"""

    def __init__(self, message: str, code: str = "split_error"):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
def check_group_leakage(cases: Iterable[EvaluationCaseV11]) -> list[dict[str, Any]]:
    """§13.1：回傳跨 split 的 group（空清單代表乾淨）。"""
    by_group: dict[str, set[str]] = defaultdict(set)
    for c in cases:
        by_group[c.group_id].add(c.split)
    return [
        {"group_id": g, "splits": sorted(s)}
        for g, s in sorted(by_group.items())
        if len(s) > 1
    ]


def _split_info(split: str, cases: Sequence[EvaluationCaseV11]) -> SplitInfo:
    return SplitInfo(
        split=split,
        case_ids=sorted(c.id for c in cases),
        group_ids=sorted({c.group_id for c in cases}),
        query_type_counts=dict(sorted(Counter(c.query_type for c in cases).items())),
        answerable_count=sum(1 for c in cases if c.answerable),
        unanswerable_count=sum(1 for c in cases if not c.answerable),
    )


def prepare_split_datasets(
    dataset_path: str,
    output_dir: str,
    *,
    splits: Sequence[str] = ALL_SPLITS,
) -> SplitManifest:
    """讀取完整 dataset，檢查洩漏，並為每個 split 寫出獨立 dataset 檔。

    寫出的檔案保留 ``schema_version`` / ``dataset_version``，只過濾 ``cases``。
    Trial Runner 之後只會拿到單一 split 的檔案路徑。
    """
    with open(dataset_path, encoding="utf-8") as f:
        raw = json.load(f)
    dataset = parse_dataset(raw)  # 內含 group leakage 驗證

    leakage = check_group_leakage(dataset.cases)
    if leakage:  # pragma: no cover - parse_dataset 已擋，這是第二道防線
        raise SplitError(
            f"偵測到 group split leakage：{leakage}", code="group_split_leakage"
        )

    by_split: dict[str, list[EvaluationCaseV11]] = defaultdict(list)
    for c in dataset.cases:
        by_split[c.split].append(c)

    os.makedirs(output_dir, exist_ok=True)
    infos: dict[str, SplitInfo] = {}
    raw_by_id = {str(c["id"]): c for c in raw["cases"]}

    for split in splits:
        cases = by_split.get(split, [])
        info = _split_info(split, cases)
        if cases:
            payload = {
                "schema_version": raw.get("schema_version"),
                "dataset_version": raw.get("dataset_version"),
                "notes": (
                    f"Slice 5 split isolation：僅含 split={split!r} 的 case，"
                    "由 evaluation.v11.search.split_dataset 自動產生（不得手動編輯、不進 Git）。"
                ),
                "cases": [raw_by_id[cid] for cid in info.case_ids],
            }
            path = os.path.join(output_dir, f"{split}.cases.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            info.dataset_path = path
            info.dataset_hash = dataset_hash(payload)
        infos[split] = info

    return SplitManifest(
        dataset_version=dataset.dataset_version,
        full_dataset_hash=dataset_hash(raw),
        splits=infos,
        leakage_ok=True,
        leakage_detail=[],
    )


# --------------------------------------------------------------------------
# Constraint 可計算性預測（§五 3）
# --------------------------------------------------------------------------
@dataclass
class CoverageFinding:
    constraint_name: str
    metric: str
    scope: str
    query_type: Optional[str]
    predicted_computable: Optional[bool]
    required_query_types: list[str]
    missing_query_types: list[str]
    safety_critical: bool
    on_not_computable: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_name": self.constraint_name,
            "metric": self.metric,
            "scope": self.scope,
            "query_type": self.query_type,
            "predicted_computable": self.predicted_computable,
            "required_query_types": list(self.required_query_types),
            "missing_query_types": list(self.missing_query_types),
            "safety_critical": self.safety_critical,
            "on_not_computable": self.on_not_computable,
            "note": self.note,
        }


def _required_query_types(c: HardConstraint) -> frozenset[str]:
    if c.ref.scope == mn.SCOPE_QUERY_TYPE and c.ref.query_type:
        return frozenset({c.ref.query_type})
    return _AGGREGATE_METRIC_REQUIRES.get(c.ref.metric, frozenset())


def analyze_constraint_coverage(
    constraints: Sequence[HardConstraint], info: SplitInfo
) -> list[CoverageFinding]:
    """事前預測每個 constraint 在該 split 上是否可能可計算。

    ``predicted_computable=None`` 代表「無法事前判定」（需實際執行才知道），
    不代表可計算 —— 呼叫端不得因此放寬判定。
    """
    findings: list[CoverageFinding] = []
    for c in constraints:
        required = _required_query_types(c)
        missing = sorted(q for q in required if info.query_type_counts.get(q, 0) == 0)
        if not required:
            predicted: Optional[bool] = None
            note = "此 metric 不依賴特定 query type；實際可計算性需執行後確認。"
        elif missing:
            predicted = False
            note = (
                f"split={info.split!r} 沒有任何 {missing} 類型的 case，"
                f"因此 {c.ref.describe()} 在此 split 上必然不可計算。"
            )
        else:
            predicted = True
            note = f"split={info.split!r} 具備所需 query type {sorted(required)}。"
        findings.append(
            CoverageFinding(
                constraint_name=c.name,
                metric=c.ref.metric,
                scope=c.ref.scope,
                query_type=c.ref.query_type,
                predicted_computable=predicted,
                required_query_types=sorted(required),
                missing_query_types=missing,
                safety_critical=c.safety_critical,
                on_not_computable=c.on_not_computable,
                note=note,
            )
        )
    return findings


@dataclass
class ValidationCoverageVerdict:
    sufficient: bool
    status: str
    findings: list[CoverageFinding]
    blocking: list[CoverageFinding]
    missing_expected_query_types: list[str]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "status": self.status,
            "message": self.message,
            "missing_expected_query_types": list(self.missing_expected_query_types),
            "blocking_constraints": [f.to_dict() for f in self.blocking],
            "findings": [f.to_dict() for f in self.findings],
        }


def assess_validation_coverage(
    constraints: Sequence[HardConstraint], info: SplitInfo
) -> ValidationCoverageVerdict:
    """§五：validation split 是否足以執行 hard constraints。

    若有 constraint 必然不可計算、且其政策為 ``fail``，則整個 recommendation 階段
    必定得不到任何 feasible candidate —— 這時必須明確回報
    ``insufficient_validation_coverage``，**不得**偷用 test、把 tune 當 validation、
    改動 split，或偽造 metric。
    """
    findings = analyze_constraint_coverage(constraints, info)
    blocking = [
        f
        for f in findings
        if f.predicted_computable is False and f.on_not_computable == "fail"
    ]
    missing = info.missing_expected_query_types()

    if blocking:
        names = "、".join(f"{f.constraint_name}（{f.metric}）" for f in blocking)
        return ValidationCoverageVerdict(
            sufficient=False,
            status=STATUS_INSUFFICIENT_VALIDATION_COVERAGE,
            findings=findings,
            blocking=blocking,
            missing_expected_query_types=missing,
            message=(
                f"validation split（{info.case_count} case / {info.group_count} group，"
                f"query types={sorted(info.query_type_counts)}）無法計算下列 "
                f"on_not_computable=fail 的 constraint：{names}。"
                " 依 §14.5「不可計算不得默認通過」，所有 candidate 必然 infeasible，"
                "recommendation 階段停止並回報 insufficient_validation_coverage。"
            ),
        )

    return ValidationCoverageVerdict(
        sufficient=True,
        status="ok",
        findings=findings,
        blocking=[],
        missing_expected_query_types=missing,
        message=(
            f"validation split 可計算全部 on_not_computable=fail 的 constraint"
            + (f"；但缺少期望 query type {missing}（樣本覆蓋不足，結論不確定性高）。" if missing else "。")
        ),
    )
