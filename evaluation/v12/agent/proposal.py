"""v1.2 Agent Proposal Schema（Slice 1 §四）。

LLM（或 Fake Planner）每輪輸出的結構化提案。此模組只定義**資料型別與受控詞彙**；
嚴格 JSON 解析在 :mod:`evaluation.v12.agent.parser`，政策驗證在
:mod:`evaluation.v12.agent.validator`。

安全邊界（§四 / §十四）：
- ``proposed_config`` 只能承載既有系統可執行的白名單欄位（Chunking / Retrieval）。
- ``probe_questions`` 只是診斷探針：不進正式 Dataset、不決定 Recommendation Eligibility、
  不自動發布；每題須綁定既有 Segment / Anchor（由 Validator 檢查存在性）。
- ``stop_recommendation`` 只是建議；是否真的停止由 Orchestrator 依 Budget 決定。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Diagnosis 的受控 failure_type 詞彙（對應 v1.1 case_status / confusable 分類的可提案集合）。
FAILURE_TYPES = frozenset(
    {
        "miss",
        "wrong_rank",
        "unanswerable_false_positive",
        "confusable_confusion",
        "threshold_too_high",
        "chunk_boundary",
        "identifier_ambiguity",
    }
)

#: expected_effects 的方向詞彙（與 v1.1 recommendation direction 對齊）。
EFFECT_DIRECTIONS = frozenset({"increase", "decrease"})


@dataclass(frozen=True)
class Diagnosis:
    """對單一 tune failure case 的診斷（LLM 只提案，不下結論）。"""

    case_id: str
    failure_type: str
    reason: str
    supporting_observations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "failure_type": self.failure_type,
            "reason": self.reason,
            "supporting_observations": list(self.supporting_observations),
        }


@dataclass(frozen=True)
class ProbeQuestion:
    """診斷探針（非正式 Dataset case）。須綁定來源 Segment 與 Anchor 建議。"""

    question: str
    query_type: str
    source_segment_ids: tuple[str, ...] = ()
    anchor_text_suggestions: tuple[str, ...] = ()
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "query_type": self.query_type,
            "source_segment_ids": list(self.source_segment_ids),
            "anchor_text_suggestions": list(self.anchor_text_suggestions),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ExpectedEffect:
    """LLM 對某 metric 的預期影響（僅供 Audit / 下一輪對照，不影響升級判定）。"""

    metric: str
    direction: str
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AgentProposal:
    """一輪的完整結構化提案。

    ``proposed_chunk_config`` / ``proposed_retrieval_config`` 為 LLM 提出的**原始** dict
    （尚未 canonical 化、尚未驗證）；Validator 會交給 v1.1 ChunkingConfig / RetrievalConfig /
    TrialConfig 驗證並 canonical 化。
    """

    diagnoses: tuple[Diagnosis, ...]
    proposed_chunk_config: dict[str, Any]
    proposed_retrieval_config: dict[str, Any]
    probe_questions: tuple[ProbeQuestion, ...]
    expected_effects: tuple[ExpectedEffect, ...]
    confidence: float
    stop_recommendation: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "proposed_config": {
                "chunk_config": dict(self.proposed_chunk_config),
                "retrieval_config": dict(self.proposed_retrieval_config),
            },
            "probe_questions": [p.to_dict() for p in self.probe_questions],
            "expected_effects": [e.to_dict() for e in self.expected_effects],
            "confidence": self.confidence,
            "stop_recommendation": self.stop_recommendation,
        }
