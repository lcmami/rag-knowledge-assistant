"""OpenAI-compatible LLM Candidate Generator（Slice 6 §十一）。

以現有 :class:`app.services.llm_client.LLMClient`（或最小相容 Adapter）取得候選問題，
要求 **Strict JSON** 輸出並嚴格驗證；任何解析失敗、未知 query_type、空問題、
Provider 失敗或 Timeout 都是**明確錯誤**，且**不落庫任何半完成 Candidate**
（§十一.9／§十一.10）——本模組只回傳「全部通過驗證」的清單，或拋
:class:`GeneratorError`。

安全性（§十一.11～§十一.13）：
- Prompt 只提供必要 Segment，不提供整份 Corpus。
- 一般 Report 不含原始 Prompt／完整 Response；只保存 provider / model /
  request id / token usage / response hash。
- **不保存 API Key。**

本 Slice 不執行真實 Provider Smoke；測試一律使用 Fake LLM。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

from evaluation.v11.case_schema import ALLOWED_QUERY_TYPES
from evaluation.v11.review.generator import (
    GeneratedQuestionCandidate,
    GenerationRequest,
    GeneratorError,
)

#: LLM 必須回傳的 JSON 欄位（§十一.4）。
REQUIRED_FIELDS = (
    "question",
    "query_type",
    "answerable_suggestion",
    "anchor_text_suggestions",
    "group_id_suggestion",
    "rationale",
)

_SYSTEM_PROMPT = (
    "你是評測資料集的候選問題提案助手。你只能『提案』，不得宣稱任何 Evidence 已驗證，"
    "也不得產生字元 offset。只依據使用者提供的文件片段提問，不得臆造片段以外的事實。"
    "必須僅輸出 JSON，不得輸出任何其他文字。"
)

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    m = _JSON_FENCE_RE.match(text)
    return m.group(1) if m else text


def build_prompt(request: GenerationRequest) -> list[dict[str, str]]:
    """組出 Chat Messages；只放入本次請求的 Segment（不放整份 Corpus）。"""
    segments_payload = [
        {
            "segment_id": s.segment_id,
            "filename": s.filename,
            "page": s.page,
            "section": s.section,
            "text": s.text,
        }
        for s in request.segments
    ]
    user = {
        "task": (
            f"依據下列文件片段提出 {request.requested_count} 個候選問題。"
            "每個問題必須可由片段內容判斷，或明確是片段未記載的（unanswerable 提案）。"
        ),
        "allowed_query_types": list(request.allowed_query_types),
        "constraints": request.constraints,
        "segments": segments_payload,
        "output_schema": {
            "candidates": [
                {
                    "question": "string",
                    "query_type": f"one of {sorted(request.allowed_query_types)}",
                    "answerable_suggestion": "boolean",
                    "anchor_text_suggestions": ["string"],
                    "group_id_suggestion": "string",
                    "rationale": "string",
                    "source_segment_ids": ["string"],
                }
            ]
        },
        "rules": [
            "僅輸出 JSON 物件，最外層必須有 candidates 陣列。",
            "不得輸出 char_start / char_end 等 offset。",
            "不得聲稱 evidence 已驗證。",
            "unanswerable 僅為建議，仍需人工確認。",
        ],
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user, ensure_ascii=False, sort_keys=True),
        },
    ]


def parse_response(
    raw: str, request: GenerationRequest
) -> list[GeneratedQuestionCandidate]:
    """嚴格解析 LLM 的 JSON 回應；任一項不合法即整批失敗（不落庫半成品）。"""
    text = _strip_code_fence(raw or "")
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise GeneratorError(f"LLM 回應不是合法 JSON：{exc}") from exc

    if not isinstance(payload, dict):
        raise GeneratorError("LLM 回應最外層必須是 JSON 物件。")
    items = payload.get("candidates")
    if not isinstance(items, list) or not items:
        raise GeneratorError("LLM 回應缺少非空的 candidates 陣列。")

    by_segment = {s.segment_id: s for s in request.segments}
    default_segment = request.segments[0]
    allowed = set(request.allowed_query_types)

    out: list[GeneratedQuestionCandidate] = []
    for i, item in enumerate(items):
        where = f"candidates[{i}]"
        if not isinstance(item, dict):
            raise GeneratorError(f"{where} 必須是物件。")
        missing = [f for f in REQUIRED_FIELDS if f not in item]
        if missing:
            raise GeneratorError(f"{where} 缺少必要欄位：{missing}")

        question = item["question"]
        if not isinstance(question, str) or not question.strip():
            raise GeneratorError(f"{where} question 不得為空。")

        query_type = item["query_type"]
        if query_type not in ALLOWED_QUERY_TYPES:
            raise GeneratorError(
                f"{where} 未知 query_type={query_type!r}；允許 {sorted(ALLOWED_QUERY_TYPES)}。"
            )
        if query_type not in allowed:
            raise GeneratorError(
                f"{where} query_type={query_type!r} 不在本次允許集合 {sorted(allowed)} 內。"
            )

        answerable = item["answerable_suggestion"]
        if not isinstance(answerable, bool):
            raise GeneratorError(f"{where} answerable_suggestion 必須是布林值。")

        anchors = item["anchor_text_suggestions"]
        if not isinstance(anchors, list) or not all(isinstance(a, str) for a in anchors):
            raise GeneratorError(f"{where} anchor_text_suggestions 必須是字串陣列。")

        seg_ids = item.get("source_segment_ids") or []
        if not isinstance(seg_ids, list) or not all(isinstance(s, str) for s in seg_ids):
            raise GeneratorError(f"{where} source_segment_ids 必須是字串陣列。")
        known = [s for s in seg_ids if s in by_segment]
        segment = by_segment[known[0]] if known else default_segment

        group_suggestion = item["group_id_suggestion"]
        if group_suggestion is not None and not isinstance(group_suggestion, str):
            raise GeneratorError(f"{where} group_id_suggestion 必須是字串或 null。")

        candidate = GeneratedQuestionCandidate(
            question=question.strip(),
            query_type=query_type,
            answerable_suggestion=answerable,
            source_document_sha256=segment.document_sha256,
            source_filename=segment.filename,
            source_segment_ids=known or [segment.segment_id],
            source_page=segment.page,
            source_section=segment.section,
            anchor_text_suggestions=list(anchors),
            group_id_suggestion=group_suggestion,
            split_suggestion=None,  # split 由人工決定，不接受 LLM 指派
            rationale=item.get("rationale") if isinstance(item.get("rationale"), str) else None,
        )
        candidate.validate()
        out.append(candidate)

    return out


def response_metadata(raw_response: str, provider: str, model: str, response: Any) -> dict[str, Any]:
    """可安全保存的 Generator metadata（不含 Prompt 全文、Response 全文、API Key）。"""
    usage = getattr(response, "usage", None)
    meta: dict[str, Any] = {
        "generator": "llm",
        "provider": provider,
        "model": model,
        "response_sha256": hashlib.sha256((raw_response or "").encode("utf-8")).hexdigest(),
    }
    request_id = getattr(response, "id", None)
    if isinstance(request_id, str):
        meta["request_id"] = request_id
    if usage is not None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, key, None)
            if isinstance(value, int):
                meta[key] = value
    return meta


class LLMCandidateGenerator:
    """OpenAI-compatible Generator。

    ``llm_client`` 需提供 async ``complete(messages) -> str``（即現有 ``LLMClient``）。
    Provider 失敗／Timeout 一律轉為 :class:`GeneratorError`，不留半完成 Candidate。
    """

    generator_type = "llm"

    def __init__(self, llm_client, provider: str = "openai_compatible"):
        self._client = llm_client
        self.provider = provider

    @property
    def model(self) -> str:
        return getattr(self._client, "model", "") or ""

    def generate(self, request: GenerationRequest) -> list[GeneratedQuestionCandidate]:
        messages = build_prompt(request)
        try:
            raw = asyncio.run(self._client.complete(messages))
        except GeneratorError:
            raise
        except Exception as exc:  # noqa: BLE001 - Provider 失敗/Timeout 皆為明確錯誤
            raise GeneratorError(
                f"LLM Provider 呼叫失敗（{type(exc).__name__}）；未建立任何 Candidate。"
            ) from exc

        candidates = parse_response(raw, request)
        meta = {
            "generator": self.generator_type,
            "provider": self.provider,
            "model": self.model,
            "response_sha256": hashlib.sha256((raw or "").encode("utf-8")).hexdigest(),
        }
        for c in candidates:
            c.generator_metadata = dict(meta)
        return candidates
