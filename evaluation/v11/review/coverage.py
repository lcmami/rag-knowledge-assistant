"""Dataset Coverage Summary 與 Gap Detection（Slice 6 §十六／§十七）。

對「Base Dataset ＋ Accepted Candidate」計算覆蓋率摘要，並偵測已知缺口。

**Gap 只負責提示**（§十六）：不得自動改 Split、自動接受 Candidate、自動發布
Dataset，也不得依 Candidate 的 Retrieval 結果決定題目——本模組完全不接觸任何
檢索結果或 Optimizer Metrics。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.services.identifier_extractor import IdentifierExtractor
from evaluation.v11.case_schema import ALLOWED_SPLITS, EvaluationCaseV11

#: exact_code 樣本數低於此值即提示樣本過小（§十六）。
SMALL_SAMPLE_THRESHOLD = 3

# Gap code（§十六）
GAP_VALIDATION_IDENTIFIER_UNANSWERABLE = "validation_identifier_bearing_unanswerable_missing"
GAP_INDEPENDENT_CONFUSABLE = "independent_confusable_group_missing"
GAP_SMALL_VALIDATION_EXACT_CODE = "small_validation_exact_code_sample"
GAP_SMALL_TEST_EXACT_CODE = "small_test_exact_code_sample"
#: 語料不足以支撐某個缺口時的提示（§十七.2）；不得虛構無文件依據的 answerable case。
GAP_ADDITIONAL_CORPUS_REQUIRED = "additional_corpus_required"

#: 目前已知的 confusable group（Slice 0–5 既有）；與之無關者才算「獨立」confusable。
KNOWN_CONFUSABLE_GROUP_PREFIXES = ("alarm-e10", "alarm-d900")


@dataclass
class CoverageGap:
    code: str
    message: str
    severity: str = "warning"  # Coverage Gap 恆為 warning，不得降級 schema/evidence error

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


@dataclass
class CoverageSummary:
    cases_per_split: dict[str, int] = field(default_factory=dict)
    groups_per_split: dict[str, int] = field(default_factory=dict)
    query_type_counts: dict[str, int] = field(default_factory=dict)
    query_type_counts_per_split: dict[str, dict[str, int]] = field(default_factory=dict)
    answerable_count: int = 0
    unanswerable_count: int = 0
    identifier_bearing_unanswerable_per_split: dict[str, int] = field(default_factory=dict)
    confusable_groups: list[str] = field(default_factory=list)
    independent_confusable_groups: list[str] = field(default_factory=list)
    group_leakage: list[str] = field(default_factory=list)
    total_cases: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "cases_per_split": dict(self.cases_per_split),
            "groups_per_split": dict(self.groups_per_split),
            "query_type_counts": dict(self.query_type_counts),
            "query_type_counts_per_split": {
                k: dict(v) for k, v in self.query_type_counts_per_split.items()
            },
            "answerable_count": self.answerable_count,
            "unanswerable_count": self.unanswerable_count,
            "identifier_bearing_unanswerable_per_split": dict(
                self.identifier_bearing_unanswerable_per_split
            ),
            "confusable_groups": list(self.confusable_groups),
            "independent_confusable_groups": list(self.independent_confusable_groups),
            "group_leakage": list(self.group_leakage),
        }


def is_identifier_bearing(question: str, extractor: IdentifierExtractor | None = None) -> bool:
    """Query 是否含識別碼（供 identifier-bearing unanswerable 偵測，§十五.5）。"""
    ex = extractor or IdentifierExtractor()
    return bool(ex.extract_normalized(question or ""))


def _is_independent_confusable(group_id: str) -> bool:
    g = (group_id or "").lower()
    return not any(g.startswith(p) for p in KNOWN_CONFUSABLE_GROUP_PREFIXES)


def summarize(cases: Iterable[EvaluationCaseV11]) -> CoverageSummary:
    """計算 Coverage Summary（純函式，不修改輸入）。"""
    cases = list(cases)
    extractor = IdentifierExtractor()

    summary = CoverageSummary(total_cases=len(cases))
    groups_by_split: dict[str, set[str]] = {s: set() for s in ALLOWED_SPLITS}
    group_to_splits: dict[str, set[str]] = {}
    qt_per_split: dict[str, Counter] = {s: Counter() for s in ALLOWED_SPLITS}
    ident_unans: Counter = Counter()
    confusable_groups: set[str] = set()

    for c in cases:
        summary.cases_per_split[c.split] = summary.cases_per_split.get(c.split, 0) + 1
        groups_by_split.setdefault(c.split, set()).add(c.group_id)
        group_to_splits.setdefault(c.group_id, set()).add(c.split)
        qt_per_split.setdefault(c.split, Counter())[c.query_type] += 1
        summary.query_type_counts[c.query_type] = (
            summary.query_type_counts.get(c.query_type, 0) + 1
        )
        if c.answerable:
            summary.answerable_count += 1
        else:
            summary.unanswerable_count += 1
            if is_identifier_bearing(c.question, extractor):
                ident_unans[c.split] += 1
        if c.query_type == "confusable":
            confusable_groups.add(c.group_id)

    summary.groups_per_split = {s: len(g) for s, g in groups_by_split.items() if g}
    summary.query_type_counts_per_split = {
        s: dict(counter) for s, counter in qt_per_split.items() if counter
    }
    summary.identifier_bearing_unanswerable_per_split = dict(ident_unans)
    summary.confusable_groups = sorted(confusable_groups)
    summary.independent_confusable_groups = sorted(
        g for g in confusable_groups if _is_independent_confusable(g)
    )
    # group leakage：同 group 橫跨多個 split（§十六）。
    summary.group_leakage = sorted(
        g for g, splits in group_to_splits.items() if len(splits) > 1
    )
    return summary


def detect_gaps(
    summary: CoverageSummary, *, corpus_supports_independent_confusable: bool = True
) -> list[CoverageGap]:
    """偵測已知 Coverage 缺口；只提示，不修改任何資料。"""
    gaps: list[CoverageGap] = []

    if summary.identifier_bearing_unanswerable_per_split.get("validation", 0) < 1:
        gaps.append(
            CoverageGap(
                GAP_VALIDATION_IDENTIFIER_UNANSWERABLE,
                "validation split 缺少 identifier-bearing unanswerable case："
                "無法檢驗『查詢含語料內存在的設備代碼、但文件未記載該屬性』的行為。",
            )
        )

    if not summary.independent_confusable_groups:
        gaps.append(
            CoverageGap(
                GAP_INDEPENDENT_CONFUSABLE,
                "缺少與既有 alarm-e10x／d900 無關的獨立 confusable group："
                "confusable 行為僅由單一 group 支撐。",
            )
        )
        if not corpus_supports_independent_confusable:
            gaps.append(
                CoverageGap(
                    GAP_ADDITIONAL_CORPUS_REQUIRED,
                    "目前 corpus 沒有足夠來源可建立獨立 confusable case；"
                    "需補充文件，不得虛構無文件依據的 answerable case。",
                )
            )

    for split, code in (
        ("validation", GAP_SMALL_VALIDATION_EXACT_CODE),
        ("test", GAP_SMALL_TEST_EXACT_CODE),
    ):
        n = summary.query_type_counts_per_split.get(split, {}).get("exact_code", 0)
        if n < SMALL_SAMPLE_THRESHOLD:
            gaps.append(
                CoverageGap(
                    code,
                    f"{split} split 的 exact_code 樣本僅 {n} 題"
                    f"（< {SMALL_SAMPLE_THRESHOLD}），統計結論不穩定。",
                )
            )

    return gaps


@dataclass
class CoverageReport:
    summary: CoverageSummary
    gaps: list[CoverageGap]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "gaps": [g.to_dict() for g in self.gaps],
            "gap_codes": [g.code for g in self.gaps],
        }


def build_coverage(
    cases: Iterable[EvaluationCaseV11],
    *,
    corpus_supports_independent_confusable: bool = True,
) -> CoverageReport:
    summary = summarize(cases)
    return CoverageReport(
        summary=summary,
        gaps=detect_gaps(
            summary,
            corpus_supports_independent_confusable=corpus_supports_independent_confusable,
        ),
    )
