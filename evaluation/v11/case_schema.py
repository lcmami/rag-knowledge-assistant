"""v1.1 Evaluation Case / Evidence Span Schema（Slice 1 §七）。

Gold Label 綁**原始文件 Evidence Span**（document + section/page/offset + anchor），
**不綁 chunk id**；chunking 改變時標註仍有效。

驗證規則（§七）：
1. ``answerable=false`` 可無 expected evidence。
2. ``answerable=true`` 至少一個 Evidence Span。
3. ``anchor_text`` 對 answerable case 為必要 fallback。
4. 支援多個 Evidence Span、同題多文件來源。
5. ``split`` 僅允許 ``tune`` / ``validation`` / ``test``。
6. 同 ``group_id`` 不得跨 split（防資料洩漏）。
7. Case ID 不得重複。
8. Evidence 的 ``document_sha256`` / ``canonical_text_version`` 不得為空。
9. 錯誤 schema 產生可理解的 validation error。

此模組**不修改** Slice 0 的 :mod:`evaluation.schemas`，為並行的 v1.1 版本化 schema。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.1"

# §七 6：split 僅三種，group-aware。
ALLOWED_SPLITS = {"tune", "validation", "test"}

# query_type（對齊 requirements §4.1 FR-5 的 v1.1 集合）。
ALLOWED_QUERY_TYPES = {
    "exact_code",
    "semantic",
    "procedure",
    "cross_section",
    "confusable",
    "unanswerable",
}

# offset scope（對齊 canonical §五）。
ALLOWED_OFFSET_SCOPES = {"page", "document", None}


class CaseSchemaError(Exception):
    """v1.1 dataset / case schema 驗證錯誤（訊息需可理解）。"""


@dataclass
class EvidenceSpan:
    """一個 Gold Evidence Span（綁原文來源範圍，非 chunk id）。"""

    document_sha256: str
    canonical_text_version: str
    loader_version: str | None = None
    filename: str | None = None
    page: int | None = None
    section: str | None = None
    paragraph_index: int | None = None
    offset_scope: str | None = None  # page | document | None
    char_start: int | None = None
    char_end: int | None = None
    anchor_text: list[str] = field(default_factory=list)
    evidence_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_offset(self) -> bool:
        return self.char_start is not None and self.char_end is not None


@dataclass
class EvaluationCaseV11:
    """單一 v1.1 evaluation case。"""

    id: str
    question: str
    query_type: str
    answerable: bool
    group_id: str
    split: str
    expected_evidence: list[EvidenceSpan] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    dataset_version: str | None = None
    status: str = "migrated"  # migrated | pending | accepted ...（Slice 1 僅記錄）

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "query_type": self.query_type,
            "answerable": self.answerable,
            "group_id": self.group_id,
            "split": self.split,
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "status": self.status,
            "expected_evidence": [e.to_dict() for e in self.expected_evidence],
        }


@dataclass
class EvaluationDatasetV11:
    """一份版本化 v1.1 dataset。"""

    schema_version: str
    dataset_version: str
    cases: list[EvaluationCaseV11]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "cases": [c.to_dict() for c in self.cases],
        }


# --------------------------------------------------------------------------
# 解析 / 驗證
# --------------------------------------------------------------------------
def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise CaseSchemaError(msg)


def parse_evidence(raw: Any, case_id: str, index: int) -> EvidenceSpan:
    _require(isinstance(raw, dict), f"case[{case_id}] evidence[{index}] 必須是物件。")
    where = f"case[{case_id}] evidence[{index}]"

    sha = raw.get("document_sha256")
    _require(
        isinstance(sha, str) and sha.strip() != "",
        f"{where} 缺少 document_sha256（不得為空）。",
    )
    ctv = raw.get("canonical_text_version")
    _require(
        isinstance(ctv, str) and ctv.strip() != "",
        f"{where} 缺少 canonical_text_version（不得為空）。",
    )

    scope = raw.get("offset_scope")
    _require(
        scope in ALLOWED_OFFSET_SCOPES,
        f"{where} 非法 offset_scope={scope!r}；允許 {sorted(s for s in ALLOWED_OFFSET_SCOPES if s)} 或省略。",
    )

    cs, ce = raw.get("char_start"), raw.get("char_end")
    if cs is not None or ce is not None:
        _require(
            isinstance(cs, int) and isinstance(ce, int) and 0 <= cs <= ce,
            f"{where} char_start/char_end 必須為 0<=start<=end 的整數。",
        )

    anchors = raw.get("anchor_text", [])
    _require(
        isinstance(anchors, list) and all(isinstance(a, str) for a in anchors),
        f"{where} anchor_text 必須是字串陣列。",
    )

    return EvidenceSpan(
        document_sha256=sha,
        canonical_text_version=ctv,
        loader_version=raw.get("loader_version"),
        filename=raw.get("filename"),
        page=raw.get("page"),
        section=raw.get("section"),
        paragraph_index=raw.get("paragraph_index"),
        offset_scope=scope,
        char_start=cs,
        char_end=ce,
        anchor_text=list(anchors),
        evidence_id=raw.get("evidence_id"),
    )


def parse_case(raw: Any, index: int, *, dataset_version: str | None = None) -> EvaluationCaseV11:
    _require(isinstance(raw, dict), f"case[{index}] 必須是物件：{raw!r}")

    missing = [
        k
        for k in ("id", "question", "query_type", "answerable", "group_id", "split")
        if k not in raw
    ]
    _require(not missing, f"case[{index}] 缺少必要欄位：{missing}（id={raw.get('id')!r}）")

    case_id = str(raw["id"])
    _require(
        isinstance(raw["answerable"], bool),
        f"case[{case_id}] answerable 必須是布林值，得到 {raw['answerable']!r}。",
    )
    _require(
        raw["query_type"] in ALLOWED_QUERY_TYPES,
        f"case[{case_id}] 非法 query_type={raw['query_type']!r}；允許 {sorted(ALLOWED_QUERY_TYPES)}。",
    )
    _require(
        raw["split"] in ALLOWED_SPLITS,
        f"case[{case_id}] 非法 split={raw['split']!r}；允許 {sorted(ALLOWED_SPLITS)}。",
    )

    raw_evidence = raw.get("expected_evidence", []) or []
    _require(isinstance(raw_evidence, list), f"case[{case_id}] expected_evidence 必須是陣列。")
    evidence = [parse_evidence(e, case_id, i) for i, e in enumerate(raw_evidence)]

    answerable = bool(raw["answerable"])
    if answerable:
        # §七 2：至少一個 evidence span
        _require(
            len(evidence) >= 1,
            f"case[{case_id}] answerable=true 至少需一個 expected_evidence span。",
        )
        # §七 3：anchor_text 為 answerable 的必要 fallback（每個 span 至少一個 anchor）
        for i, e in enumerate(evidence):
            _require(
                len(e.anchor_text) >= 1,
                f"case[{case_id}] evidence[{i}] answerable case 必須提供 anchor_text（必要 fallback）。",
            )

    return EvaluationCaseV11(
        id=case_id,
        question=str(raw["question"]),
        query_type=str(raw["query_type"]),
        answerable=answerable,
        group_id=str(raw["group_id"]),
        split=str(raw["split"]),
        expected_evidence=evidence,
        schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
        dataset_version=raw.get("dataset_version", dataset_version),
        status=str(raw.get("status", "migrated")),
    )


def _check_group_split_leakage(cases: list[EvaluationCaseV11]) -> None:
    """同一 group_id 不得散落於不同 split（§七 7 / requirements §8.6 group-aware）。"""
    group_split: dict[str, str] = {}
    for c in cases:
        prev = group_split.get(c.group_id)
        if prev is None:
            group_split[c.group_id] = c.split
        elif prev != c.split:
            raise CaseSchemaError(
                f"group split leakage：group_id={c.group_id!r} 同時出現於 "
                f"split={prev!r} 與 {c.split!r}，同組必須進同一 split。"
            )


def parse_dataset(raw: Any) -> EvaluationDatasetV11:
    """驗證整份 v1.1 dataset（含 schema_version / dataset_version / cases）。"""
    _require(isinstance(raw, dict), "dataset 最外層必須是物件（含 schema_version / cases）。")

    schema_version = raw.get("schema_version")
    _require(
        schema_version == SCHEMA_VERSION,
        f"schema_version 必須為 {SCHEMA_VERSION!r}，得到 {schema_version!r}。",
    )
    dataset_version = raw.get("dataset_version")
    _require(
        isinstance(dataset_version, str) and dataset_version.strip() != "",
        "dataset_version 不得為空。",
    )

    raw_cases = raw.get("cases")
    _require(isinstance(raw_cases, list) and raw_cases, "cases 必須是非空陣列。")

    cases: list[EvaluationCaseV11] = []
    seen: set[str] = set()
    for i, rc in enumerate(raw_cases):
        case = parse_case(rc, i, dataset_version=dataset_version)
        # §七 8：Case ID 不得重複
        _require(case.id not in seen, f"重複的 case id：{case.id!r}")
        seen.add(case.id)
        cases.append(case)

    _check_group_split_leakage(cases)

    return EvaluationDatasetV11(
        schema_version=schema_version,
        dataset_version=dataset_version,
        cases=cases,
    )


def load_dataset(path: str) -> EvaluationDatasetV11:
    with open(path, encoding="utf-8") as f:
        return parse_dataset(json.load(f))
