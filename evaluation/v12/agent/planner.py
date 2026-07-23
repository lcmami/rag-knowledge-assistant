"""v1.2 Agent Planner 介面與 Fake 實作（Slice 2 §五）。

``AgentPlanner`` 的唯一責任：**由 Context 產生一份提案（raw JSON 字串）**。

FakeAgentPlanner 的決定性保證（依實作驗收條件 §三）：**相同 Context → 相同 Proposal /
Probe / Config Hash**。它**不**負責 Promotion Decision、Stop Reason 或 Metrics 的決定性—
那些由 Orchestrator 端到端測試負責（Slice 3）。

Fake 不呼叫任何 Provider、不讀 ``.env``、不碰 Qdrant。
"""
from __future__ import annotations

import json
from typing import Any, Optional, Protocol, Sequence

from evaluation.v12.agent.context import AgentContext


class AgentPlanner(Protocol):
    """Planner 介面：Context → raw proposal JSON。"""

    planner_type: str

    def propose(self, context: AgentContext) -> str:
        ...


class FakeAgentPlanner:
    """決定性 Fake Planner（測試與離線 Demo 用）。

    兩種模式：

    - **scripted**：給定一串 proposal payload dict，依「已完成輪數」索引取用（同 Context →
      同輸出）。供 Orchestrator 測試精準編排 improving / regressing / duplicate 等情境。
    - **default**：純粹由 Context（current config + previous rounds + probe comparison）
      推導單一參數變更；展示「讀到上一輪 Probe Comparison → 提出不同 Proposal」的回饋。
    """

    planner_type = "fake"

    def __init__(
        self,
        script: Optional[Sequence[dict[str, Any]]] = None,
        model: str = "fake-agent-v1",
    ):
        self._script = tuple(script) if script is not None else None
        self.model = model

    def propose(self, context: AgentContext) -> str:
        idx = len(context.previous_rounds)
        if self._script is not None:
            payload = dict(self._script[min(idx, len(self._script) - 1)])
        else:
            payload = self._default_payload(context)
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    # ------------------------------------------------------------------
    def _default_payload(self, context: AgentContext) -> dict[str, Any]:
        cur = context.current_retrieval_config
        top_k = int(cur.get("top_k", 5))

        # 決定性單參數變更；若上一輪被拒，往另一方向調以避免重複。
        last = context.previous_rounds[-1] if context.previous_rounds else None
        rejected = last is not None and last.promotion_decision == "rejected"
        # 若上一輪 probe comparison 顯示 candidate 未變好，改走保守方向（回饋展示）。
        probe_says_worse = bool(
            last is not None
            and last.probe_comparison
            and any(not e.get("candidate_better", False) for e in last.probe_comparison)
        )
        if rejected or probe_says_worse:
            new_top_k = top_k - 1 if top_k > 3 else top_k + 2
        else:
            new_top_k = top_k + 1 if top_k < 8 else top_k - 1

        retrieval = {
            "strategy": cur.get("strategy", "dense"),
            "top_k": new_top_k,
            "score_threshold": cur.get("score_threshold", 0.55),
            "candidate_multiplier": cur.get("candidate_multiplier", 3),
            "candidate_cap": cur.get("candidate_cap", 50),
        }

        probes = self._default_probes(context)
        return {
            "diagnoses": [
                {
                    "case_id": _first_failure_case_id(context),
                    "failure_type": "miss",
                    "reason": "fake deterministic diagnosis",
                    "supporting_observations": [],
                }
            ],
            "proposed_config": {
                "chunk_config": {"strategy": context.current_chunk_config.get("strategy", "current")},
                "retrieval_config": retrieval,
            },
            "probe_questions": probes,
            "expected_effects": [
                {"metric": "mrr", "direction": "increase", "rationale": "wider candidate set"}
            ],
            "confidence": 0.5,
            "stop_recommendation": False,
        }

    @staticmethod
    def _default_probes(context: AgentContext) -> list[dict[str, Any]]:
        seg_texts = context.segment_texts()
        if not seg_texts:
            return []
        sid = sorted(seg_texts)[0]
        text = seg_texts[sid].strip()
        anchor = text.split()[0] if text.split() else ""
        probe: dict[str, Any] = {
            "question": "fake probe: is the expected chunk retrieved?",
            "query_type": "exact_code",
            "source_segment_ids": [sid],
            "anchor_text_suggestions": [anchor] if anchor else [],
            "rationale": "fake deterministic probe",
        }
        return [probe]


def _first_failure_case_id(context: AgentContext) -> str:
    failures = context.payload.get("tune_failures") or []
    if failures:
        return str(failures[0].get("case_id", "unknown"))
    return "unknown"
