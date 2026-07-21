"""Evaluation Config / Case / Result 結構（v1.1 Slice 0）。

- :class:`EvaluationConfig`：一次評估的完整、可明確傳入的設定。
- :class:`EvaluationCase`：單一評估案例（Slice 0 Baseline 標註格式）。
- :func:`load_cases`：讀取並驗證 case JSON；缺欄位 / 重複 id / 非法 query_type 皆丟明確錯誤。
- :class:`CaseResult` / :class:`EvaluationReport`：評估結果的 Python 結構，可輸出 JSON。

Slice 0 標註格式（非 Canonical Evidence Span）：
    expected_document / expected_section / expected_page / should_answer
另加入 Slice 0 需要、且目前資料格式可可靠計算的欄位：
    query_type / confusable_document
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Slice 0 支援的 query_type；None 代表未標註（沿用舊 questions.json 相容）。
ALLOWED_QUERY_TYPES = {"exact_code", "semantic", "confusable", "unanswerable"}

# 預設 threshold 掃描點（沿用原 evaluate_retrieval.py，行為不變）。
DEFAULT_THRESHOLDS: list[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# 固定 random seed，確保 Baseline 可重現。
DEFAULT_SEED = 20240501


class EvaluationError(Exception):
    """Evaluation 設定 / 資料錯誤的基底例外。"""


class CaseValidationError(EvaluationError):
    """Case 檔案缺欄位、重複 id 或非法值。"""


@dataclass(frozen=True)
class EvaluationConfig:
    """一次評估的完整設定，可由 CLI 或 Python import 明確傳入。

    threshold=0.0 時收集原始分數，於記憶體套用候選 threshold（沿用原行為）。
    """

    corpus_dirs: tuple[str, ...]
    cases_path: str
    qdrant_url: str
    collection: str
    top_k: int = 10
    candidate_multiplier: int = 3
    candidate_cap: int = 50
    # 檢索時的過濾 threshold；0.0 表示收集原始分數，於記憶體套用候選 threshold。
    score_threshold: float = 0.0
    # 報告用「configured」threshold（正式 /api/query 的接受門檻），供 unanswerable FP 等指標評估。
    report_threshold: float = 0.0
    thresholds: tuple[float, ...] = tuple(DEFAULT_THRESHOLDS)
    seed: int = DEFAULT_SEED
    keep_collection: bool = False
    label: str = "evaluation"

    def sanitized(self) -> dict[str, Any]:
        """報告用的設定快照；不含任何私密路徑外洩以外的敏感值。"""
        return {
            "label": self.label,
            "corpus_dirs": [_basename(d) for d in self.corpus_dirs],
            "cases_path": _basename(self.cases_path),
            "collection": self.collection,
            "top_k": self.top_k,
            "candidate_multiplier": self.candidate_multiplier,
            "candidate_cap": self.candidate_cap,
            "score_threshold": self.score_threshold,
            "report_threshold": self.report_threshold,
            "thresholds": list(self.thresholds),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class EvaluationCase:
    """單一評估案例（Slice 0 Baseline 標註格式）。"""

    id: str
    question: str
    should_answer: bool
    query_type: Optional[str] = None
    expected_document: Optional[str] = None
    expected_section: Optional[str] = None
    expected_page: Optional[int] = None
    confusable_document: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseResult:
    """單一案例的評估結果（可序列化為 JSON）。"""

    id: str
    question: str
    query_type: Optional[str]
    should_answer: bool
    expected_document: Optional[str]
    expected_section: Optional[str]
    expected_page: Optional[int]
    confusable_document: Optional[str]
    top1_score: Optional[float]
    top1_document: Optional[str]
    top1_section: Optional[str]
    top1_page: Optional[int]
    top3_documents: list[str]
    ranked_documents: list[str]
    top5: list[dict[str, Any]]
    doc_hit_at_1: Optional[bool]
    doc_hit_at_3: Optional[bool]
    doc_hit_at_5: Optional[bool]
    section_hit_at_1: Optional[bool]
    page_hit_at_1: Optional[bool]
    expected_rank: Optional[int]
    confusable_rank: Optional[int]
    reciprocal_rank: float
    retrieval_time_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_legacy_row(self) -> dict[str, Any]:
        """對照原 results.json 的 per-question 欄位（結構不變）。"""
        return {
            "id": self.id,
            "question": self.question,
            "should_answer": self.should_answer,
            "expected_document": self.expected_document,
            "expected_section": self.expected_section,
            "expected_page": self.expected_page,
            "top1_score": self.top1_score,
            "top1_document": self.top1_document,
            "top1_section": self.top1_section,
            "top1_page": self.top1_page,
            "top3_documents": self.top3_documents,
            "doc_hit_at_1": self.doc_hit_at_1,
            "doc_hit_at_3": self.doc_hit_at_3,
            "section_hit_at_1": self.section_hit_at_1,
            "page_hit_at_1": self.page_hit_at_1,
            "retrieval_time_ms": self.retrieval_time_ms,
        }


@dataclass
class EvaluationReport:
    """一次評估的完整報告（可輸出 JSON / 供程式取用）。"""

    config: dict[str, Any]
    environment: dict[str, Any]
    corpus: dict[str, Any]
    dataset: dict[str, Any]
    metrics: dict[str, Any]
    threshold_at_configured: dict[str, Any]
    threshold_sweep: list[dict[str, Any]]
    cases: list[CaseResult]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "environment": self.environment,
            "corpus": self.corpus,
            "dataset": self.dataset,
            "metrics": self.metrics,
            "threshold_at_configured": self.threshold_at_configured,
            "threshold_sweep": self.threshold_sweep,
            "cases": [c.to_dict() for c in self.cases],
            "notes": self.notes,
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """輸出與原 evaluate_retrieval.py results.json 相容的結構。"""
        env = self.environment
        m = self.metrics
        return {
            "embedding_model": env.get("embedding_model"),
            "vector_size": env.get("vector_size"),
            "distance": env.get("distance"),
            "chunk_size_chars": env.get("chunk_size_chars"),
            "chunk_overlap_chars": env.get("chunk_overlap_chars"),
            "embedding_max_tokens": env.get("embedding_max_tokens"),
            "collection": self.config.get("collection"),
            "top_k": self.config.get("top_k"),
            "threshold": self.config.get("score_threshold"),
            "total_questions": m["total_questions"],
            "answerable_count": m["answerable_count"],
            "unanswerable_count": m["unanswerable_count"],
            "hit_at_1": m["hit_at_1"],
            "hit_at_3": m["hit_at_3"],
            "section_hit_at_1": m["section_hit_at_1"],
            "page_hit_at_1": m["page_hit_at_1"],
            "section_hit_considered": m["section_hit_considered"],
            "page_hit_considered": m["page_hit_considered"],
            "average_latency_ms": m["average_latency_ms"],
            "p50_latency_ms": m["p50_latency_ms"],
            "p95_latency_ms": m["p95_latency_ms"],
            "answerable_score_summary": m["answerable_score_summary"],
            "unanswerable_score_summary": m["unanswerable_score_summary"],
            "threshold_at_configured": self.threshold_at_configured,
            "threshold_sweep": self.threshold_sweep,
            "ingested_chunk_counts": self.corpus.get("ingested_chunk_counts", {}),
            "questions": [c.to_legacy_row() for c in self.cases],
        }


def _basename(path: str) -> str:
    import os

    return os.path.basename(os.path.normpath(path))


def parse_case(raw: Any, index: int) -> EvaluationCase:
    """驗證單一 case dict 並轉為 :class:`EvaluationCase`。"""
    if not isinstance(raw, dict):
        raise CaseValidationError(f"case[{index}] 不是物件：{raw!r}")

    missing = [k for k in ("id", "question", "should_answer") if k not in raw]
    if missing:
        raise CaseValidationError(
            f"case[{index}] 缺少必要欄位：{missing}（id={raw.get('id')!r}）"
        )

    if not isinstance(raw["should_answer"], bool):
        raise CaseValidationError(
            f"case[{raw['id']}] should_answer 必須是布林值，得到 {raw['should_answer']!r}"
        )

    query_type = raw.get("query_type")
    if query_type is not None and query_type not in ALLOWED_QUERY_TYPES:
        raise CaseValidationError(
            f"case[{raw['id']}] 非法 query_type={query_type!r}；"
            f"允許值：{sorted(ALLOWED_QUERY_TYPES)} 或省略"
        )

    return EvaluationCase(
        id=str(raw["id"]),
        question=str(raw["question"]),
        should_answer=bool(raw["should_answer"]),
        query_type=query_type,
        expected_document=raw.get("expected_document"),
        expected_section=raw.get("expected_section"),
        expected_page=raw.get("expected_page"),
        confusable_document=raw.get("confusable_document"),
    )


def parse_cases(raw_list: Any) -> list[EvaluationCase]:
    """驗證整份 case 清單，含重複 id 偵測。"""
    if not isinstance(raw_list, list):
        raise CaseValidationError("case 檔案的最外層必須是陣列。")
    if not raw_list:
        raise CaseValidationError("case 清單為空，無法評估。")

    cases: list[EvaluationCase] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_list):
        case = parse_case(raw, i)
        if case.id in seen:
            raise CaseValidationError(f"重複的 case id：{case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


def load_cases(path: str) -> list[EvaluationCase]:
    """從 JSON 檔載入並驗證評估案例。"""
    with open(path, encoding="utf-8") as f:
        raw_list = json.load(f)
    return parse_cases(raw_list)
