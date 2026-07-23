"""Parent / Candidate Probe Comparison（Slice 4 §七）。

Candidate trial 建立後，把**同一組** probe 分別在 Parent 與 Candidate 的（tune）collection
上執行，逐 probe 比較 Top-K / rank / score / chunk / identifier，產生一份 Probe Comparison。
這份比較會被 Orchestrator 塞進**下一輪** context（回饋迴路的核心）。

Probe 的邊界（§七）：
- **不**列入正式 Metrics、**不**加入 Dataset、**不**影響 Recommendation Eligibility、
  **不**自動發布。它只是診斷訊號，供 LLM 讀取「這次改動對這些探針的效果」。
- 只在既有隔離 collection 上檢索；``factory_knowledge`` 永不受影響（由 executor 守門）。
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

#: probe 結果 chunk 原文上限（避免把大量私人原文寫入 audit）。
_PROBE_CHUNK_CAP = 400
DEFAULT_PROBE_TOP_K = 5


class DiagnosticProbeRunner:
    """在 Parent / Candidate collection 上跑同一組 probe 並產生比較。"""

    def __init__(self, executor, *, probe_top_k: int = DEFAULT_PROBE_TOP_K):
        self._executor = executor
        self._top_k = probe_top_k

    def compare(
        self, *, parent, candidate, probes: Sequence[Any]
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for probe in probes:
            parent_results = self._retrieve(parent, probe)
            candidate_results = self._retrieve(candidate, probe)
            out.append(self._entry(probe, parent_results, candidate_results))
        return out

    # ------------------------------------------------------------------
    def _retrieve(self, node, probe) -> list[dict[str, Any]]:
        collection = node.tune.collection_name
        if not collection:
            return []
        results = self._executor.retrieve(
            collection_name=collection,
            trial=node.trial,
            question=probe.question,
            top_k=self._top_k,
        )
        return [_cap_result(r) for r in results[: self._top_k]]

    def _entry(
        self, probe, parent: list[dict[str, Any]], candidate: list[dict[str, Any]]
    ) -> dict[str, Any]:
        p_score = _top_score(parent)
        c_score = _top_score(candidate)
        score_delta = (
            c_score - p_score if (p_score is not None and c_score is not None) else None
        )
        return {
            "question": probe.question,
            "query_type": probe.query_type,
            "source_segment_ids": list(probe.source_segment_ids),
            "parent_top_k": parent,
            "candidate_top_k": candidate,
            "parent_top_score": p_score,
            "candidate_top_score": c_score,
            "score_delta": score_delta,
            "parent_first_rank": _first_rank(parent),
            "candidate_first_rank": _first_rank(candidate),
            "identifier_delta": _identifier_delta(parent, candidate),
            # 診斷用啟發訊號（非 gate）：candidate top1 分數是否高於 parent。
            "candidate_better": bool(score_delta is not None and score_delta > 0),
        }


# --------------------------------------------------------------------------
def _cap_result(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    if isinstance(out.get("chunk"), str) and len(out["chunk"]) > _PROBE_CHUNK_CAP:
        out["chunk"] = out["chunk"][:_PROBE_CHUNK_CAP] + "…"
    return out


def _top_score(results: list[dict[str, Any]]) -> Optional[float]:
    if not results:
        return None
    val = results[0].get("score")
    return float(val) if isinstance(val, (int, float)) else None


def _first_rank(results: list[dict[str, Any]]) -> Optional[int]:
    if not results:
        return None
    val = results[0].get("rank")
    return int(val) if isinstance(val, int) else None


def _identifier_delta(
    parent: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    p_ids = set(parent[0].get("identifiers", [])) if parent else set()
    c_ids = set(candidate[0].get("identifiers", [])) if candidate else set()
    return {
        "gained": sorted(c_ids - p_ids),
        "lost": sorted(p_ids - c_ids),
    }
