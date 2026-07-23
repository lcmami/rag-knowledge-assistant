"""v1.2 Agent Proposal 嚴格解析（Slice 1 §四）。

鏡射 v1.1 ``review/llm_generator.parse_response`` 的做法：strip code fence → ``json.loads``
→ 逐欄位嚴格驗證。任一項不合法即**整個回應失敗**（fail-closed），不接受半完成提案，
也**絕不**執行 LLM 產生的程式碼 / Shell。

本模組只做「結構 / 型別 / 受控詞彙」驗證；是否符合 allowed parameter space、budget、
duplicate、parent-identical 等**政策**判定在 :mod:`evaluation.v12.agent.validator`。
"""
from __future__ import annotations

import json
import re
from typing import Any

from evaluation.v11.case_schema import ALLOWED_QUERY_TYPES
from evaluation.v12.agent.proposal import (
    EFFECT_DIRECTIONS,
    FAILURE_TYPES,
    AgentProposal,
    Diagnosis,
    ExpectedEffect,
    ProbeQuestion,
)

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

_KNOWN_TOP_KEYS = frozenset(
    {
        "diagnoses",
        "proposed_config",
        "probe_questions",
        "expected_effects",
        "confidence",
        "stop_recommendation",
    }
)
_DIAGNOSIS_KEYS = frozenset(
    {"case_id", "failure_type", "reason", "supporting_observations"}
)
_PROBE_KEYS = frozenset(
    {"question", "query_type", "source_segment_ids", "anchor_text_suggestions", "rationale"}
)
_EFFECT_KEYS = frozenset({"metric", "direction", "rationale"})


class ProposalParseError(Exception):
    """Proposal 解析失敗（含 JSON 錯誤、未知欄位、受控詞彙違規）。"""


def _strip_code_fence(text: str) -> str:
    m = _JSON_FENCE_RE.match(text)
    return m.group(1) if m else text


def _require_dict(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProposalParseError(f"{where} 必須是物件。")
    return value


def _reject_unknown(value: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ProposalParseError(f"{where} 含未知欄位：{sorted(unknown)}；允許：{sorted(allowed)}。")


def _str_list(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ProposalParseError(f"{where} 必須是字串陣列。")
    return tuple(value)


def _nonempty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProposalParseError(f"{where} 必須是非空字串。")
    return value.strip()


def _parse_diagnosis(raw: Any, i: int) -> Diagnosis:
    where = f"diagnoses[{i}]"
    d = _require_dict(raw, where)
    _reject_unknown(d, _DIAGNOSIS_KEYS, where)
    failure_type = _nonempty_str(d.get("failure_type"), f"{where}.failure_type")
    if failure_type not in FAILURE_TYPES:
        raise ProposalParseError(
            f"{where}.failure_type={failure_type!r} 不在受控詞彙 {sorted(FAILURE_TYPES)} 內。"
        )
    return Diagnosis(
        case_id=_nonempty_str(d.get("case_id"), f"{where}.case_id"),
        failure_type=failure_type,
        reason=_nonempty_str(d.get("reason"), f"{where}.reason"),
        supporting_observations=_str_list(
            d.get("supporting_observations"), f"{where}.supporting_observations"
        ),
    )


def _parse_probe(raw: Any, i: int) -> ProbeQuestion:
    where = f"probe_questions[{i}]"
    d = _require_dict(raw, where)
    _reject_unknown(d, _PROBE_KEYS, where)
    query_type = _nonempty_str(d.get("query_type"), f"{where}.query_type")
    if query_type not in ALLOWED_QUERY_TYPES:
        raise ProposalParseError(
            f"{where}.query_type={query_type!r} 不在允許集合 {sorted(ALLOWED_QUERY_TYPES)} 內。"
        )
    rationale = d.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        raise ProposalParseError(f"{where}.rationale 必須是字串或 null。")
    return ProbeQuestion(
        question=_nonempty_str(d.get("question"), f"{where}.question"),
        query_type=query_type,
        source_segment_ids=_str_list(d.get("source_segment_ids"), f"{where}.source_segment_ids"),
        anchor_text_suggestions=_str_list(
            d.get("anchor_text_suggestions"), f"{where}.anchor_text_suggestions"
        ),
        rationale=rationale.strip() if isinstance(rationale, str) and rationale.strip() else None,
    )


def _parse_effect(raw: Any, i: int) -> ExpectedEffect:
    where = f"expected_effects[{i}]"
    d = _require_dict(raw, where)
    _reject_unknown(d, _EFFECT_KEYS, where)
    direction = _nonempty_str(d.get("direction"), f"{where}.direction")
    if direction not in EFFECT_DIRECTIONS:
        raise ProposalParseError(
            f"{where}.direction={direction!r} 不在 {sorted(EFFECT_DIRECTIONS)} 內。"
        )
    rationale = d.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        raise ProposalParseError(f"{where}.rationale 必須是字串或 null。")
    return ExpectedEffect(
        metric=_nonempty_str(d.get("metric"), f"{where}.metric"),
        direction=direction,
        rationale=rationale.strip() if isinstance(rationale, str) and rationale.strip() else None,
    )


def parse_proposal(raw: str) -> AgentProposal:
    """嚴格解析 LLM 回應為 :class:`AgentProposal`；任一項不合法即整批失敗。"""
    text = _strip_code_fence(raw or "")
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProposalParseError(f"回應不是合法 JSON：{exc}") from exc

    payload = _require_dict(payload, "proposal")
    _reject_unknown(payload, _KNOWN_TOP_KEYS, "proposal")

    # diagnoses（可為空陣列，但必須存在且為陣列）。
    raw_diag = payload.get("diagnoses", [])
    if not isinstance(raw_diag, list):
        raise ProposalParseError("diagnoses 必須是陣列。")
    diagnoses = tuple(_parse_diagnosis(d, i) for i, d in enumerate(raw_diag))

    # proposed_config（必要；只含 chunk_config / retrieval_config）。
    proposed = _require_dict(payload.get("proposed_config"), "proposed_config")
    _reject_unknown(proposed, frozenset({"chunk_config", "retrieval_config"}), "proposed_config")
    chunk_config = _require_dict(
        proposed.get("chunk_config"), "proposed_config.chunk_config"
    )
    retrieval_config = _require_dict(
        proposed.get("retrieval_config"), "proposed_config.retrieval_config"
    )

    # probe_questions（可為空）。
    raw_probes = payload.get("probe_questions", [])
    if not isinstance(raw_probes, list):
        raise ProposalParseError("probe_questions 必須是陣列。")
    probes = tuple(_parse_probe(p, i) for i, p in enumerate(raw_probes))

    # expected_effects（可為空）。
    raw_effects = payload.get("expected_effects", [])
    if not isinstance(raw_effects, list):
        raise ProposalParseError("expected_effects 必須是陣列。")
    effects = tuple(_parse_effect(e, i) for i, e in enumerate(raw_effects))

    # confidence（0..1 數值）。
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ProposalParseError("confidence 必須是數值。")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ProposalParseError(f"confidence 必須在 [0, 1]，得到 {confidence}。")

    # stop_recommendation（布林）。
    stop = payload.get("stop_recommendation")
    if not isinstance(stop, bool):
        raise ProposalParseError("stop_recommendation 必須是布林值。")

    return AgentProposal(
        diagnoses=diagnoses,
        proposed_chunk_config=dict(chunk_config),
        proposed_retrieval_config=dict(retrieval_config),
        probe_questions=probes,
        expected_effects=effects,
        confidence=confidence,
        stop_recommendation=stop,
    )
