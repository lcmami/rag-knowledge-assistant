"""Slice 6 §二十三 — Candidate Generator 測試（一律使用 Fake LLM，不呼叫真實 Provider）。

驗證 Strict JSON 驗證、錯誤分類、Partial Response 不落庫、LLM Candidate 恆為
``proposed``、Fake Generator deterministic，以及 Generator 取不到 Test Metrics。
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from app.core.exceptions import LLMTimeoutError
from evaluation.v11.review.generator import (
    FakeCandidateGenerator,
    GenerationRequest,
    GeneratorError,
    SourceSegment,
)
from evaluation.v11.review.llm_generator import (
    LLMCandidateGenerator,
    build_prompt,
    parse_response,
)
from evaluation.v11.review.models import CANDIDATE_PROPOSED
from tests.unit.review_helpers import make_base_dataset, make_corpus, make_service


def _segment(**kwargs) -> SourceSegment:
    defaults = dict(
        document_sha256="a" * 64,
        filename="doc.md",
        segment_id="seg-0",
        canonical_text_version="v1",
        loader_version="v1",
        page=None,
        section="S",
        paragraph_index=0,
        text="E102 表示主軸溫度過高。",
    )
    defaults.update(kwargs)
    return SourceSegment(**defaults)


def _request(**kwargs) -> GenerationRequest:
    defaults = dict(
        segments=(_segment(),),
        allowed_query_types=("exact_code", "semantic", "unanswerable"),
        requested_count=2,
    )
    defaults.update(kwargs)
    return GenerationRequest(**defaults)


class FakeLLM:
    """最小 async LLM stub；回傳固定字串或拋出指定例外。"""

    def __init__(self, response: str | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = 0
        self.model = "fake-llm"

    async def complete(self, messages):
        self.calls += 1
        self.last_messages = messages
        if self.error is not None:
            raise self.error
        return self.response


def _valid_payload(n: int = 2) -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "question": f"E102 代表什麼？ ({i})",
                    "query_type": "exact_code",
                    "answerable_suggestion": True,
                    "anchor_text_suggestions": ["E102"],
                    "group_id_suggestion": "alarm-e102-new",
                    "rationale": "片段直接說明 E102",
                    "source_segment_ids": ["seg-0"],
                }
                for i in range(n)
            ]
        },
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------
# 介面邊界（§十）
# --------------------------------------------------------------------------
def test_generation_request_carries_no_test_metrics():
    """Generator 在型別上就拿不到 metrics / recommendation / ranking。"""
    field_names = {f.name for f in dataclasses.fields(GenerationRequest)}
    assert field_names == {"segments", "allowed_query_types", "requested_count", "constraints"}
    forbidden = {"metrics", "recommendation", "ranking", "trial", "search", "outcome"}
    assert not any(any(f in name for f in forbidden) for name in field_names)


def test_request_rejects_unknown_query_type():
    with pytest.raises(GeneratorError):
        _request(allowed_query_types=("not_a_type",))


def test_request_requires_segments_and_positive_count():
    with pytest.raises(GeneratorError):
        _request(segments=())
    with pytest.raises(GeneratorError):
        _request(requested_count=0)


# --------------------------------------------------------------------------
# Fake Generator
# --------------------------------------------------------------------------
def test_fake_generator_is_deterministic():
    gen = FakeCandidateGenerator()
    first = gen.generate(_request())
    second = gen.generate(_request())
    assert [c.question for c in first] == [c.question for c in second]
    assert [c.query_type for c in first] == [c.query_type for c in second]


def test_fake_generator_keeps_source_provenance():
    """§七.7：Candidate 需保存來源，不能只保存問題文字。"""
    candidates = FakeCandidateGenerator().generate(_request())
    for c in candidates:
        assert c.source_document_sha256 == "a" * 64
        assert c.source_segment_ids == ["seg-0"]


# --------------------------------------------------------------------------
# Strict JSON 驗證
# --------------------------------------------------------------------------
def test_parse_valid_strict_json():
    candidates = parse_response(_valid_payload(2), _request())
    assert len(candidates) == 2
    assert all(c.query_type == "exact_code" for c in candidates)
    # split 不接受 LLM 指派（§七.8）
    assert all(c.split_suggestion is None for c in candidates)


def test_parse_accepts_code_fenced_json():
    fenced = "```json\n" + _valid_payload(1) + "\n```"
    assert len(parse_response(fenced, _request())) == 1


def test_parse_invalid_json_raises():
    with pytest.raises(GeneratorError, match="不是合法 JSON"):
        parse_response("{not json", _request())


def test_parse_missing_candidates_array_raises():
    with pytest.raises(GeneratorError, match="candidates"):
        parse_response(json.dumps({"items": []}), _request())


def test_parse_unknown_query_type_raises():
    payload = json.loads(_valid_payload(1))
    payload["candidates"][0]["query_type"] = "totally_unknown"
    with pytest.raises(GeneratorError, match="query_type"):
        parse_response(json.dumps(payload), _request())


def test_parse_query_type_outside_allowed_set_raises():
    payload = json.loads(_valid_payload(1))
    payload["candidates"][0]["query_type"] = "procedure"  # 合法但不在本次允許集合
    with pytest.raises(GeneratorError, match="不在本次允許集合"):
        parse_response(json.dumps(payload), _request())


def test_parse_empty_question_raises():
    payload = json.loads(_valid_payload(1))
    payload["candidates"][0]["question"] = "   "
    with pytest.raises(GeneratorError, match="question"):
        parse_response(json.dumps(payload), _request())


def test_parse_missing_required_field_raises():
    payload = json.loads(_valid_payload(1))
    del payload["candidates"][0]["rationale"]
    with pytest.raises(GeneratorError, match="缺少必要欄位"):
        parse_response(json.dumps(payload), _request())


def test_parse_non_boolean_answerable_raises():
    payload = json.loads(_valid_payload(1))
    payload["candidates"][0]["answerable_suggestion"] = "yes"
    with pytest.raises(GeneratorError, match="布林"):
        parse_response(json.dumps(payload), _request())


# --------------------------------------------------------------------------
# Provider 失敗 / Timeout
# --------------------------------------------------------------------------
def test_provider_failure_raises_generator_error():
    gen = LLMCandidateGenerator(FakeLLM(error=RuntimeError("boom")))
    with pytest.raises(GeneratorError, match="Provider 呼叫失敗"):
        gen.generate(_request())


def test_provider_timeout_raises_generator_error():
    gen = LLMCandidateGenerator(FakeLLM(error=LLMTimeoutError()))
    with pytest.raises(GeneratorError, match="Provider 呼叫失敗"):
        gen.generate(_request())


def test_llm_metadata_excludes_secrets_and_full_response():
    gen = LLMCandidateGenerator(FakeLLM(response=_valid_payload(1)))
    candidates = gen.generate(_request())
    meta = candidates[0].generator_metadata
    assert meta["provider"] == "openai_compatible"
    assert meta["model"] == "fake-llm"
    assert "response_sha256" in meta
    # 不得保存 API Key、完整 Response 或 Prompt（§十一.11～§十一.13）
    assert not any(k in meta for k in ("api_key", "response", "prompt", "messages"))


def test_prompt_contains_only_requested_segments():
    """§十一.2：Prompt 只提供必要 Segment，不提供整份 Corpus。"""
    request = _request(segments=(_segment(segment_id="seg-0", text="only this"),))
    messages = build_prompt(request)
    body = messages[-1]["content"]
    assert "only this" in body
    assert "seg-99" not in body


# --------------------------------------------------------------------------
# 落庫行為：狀態恆為 proposed、失敗不留半成品
# --------------------------------------------------------------------------
def test_llm_candidates_are_always_proposed(tmp_path):
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    session = service.create_session(
        session_name="s", base_dataset_path=make_base_dataset(tmp_path, corpus)
    )
    doc = corpus.by_filename("equipment_alarm_manual.md")
    payload = json.loads(_valid_payload(2))
    for item in payload["candidates"]:
        item["source_segment_ids"] = [doc.segments[0].segment_id]

    gen = LLMCandidateGenerator(FakeLLM(response=json.dumps(payload, ensure_ascii=False)))
    outcomes = service.generate_candidates(
        session.session_id,
        gen,
        document_sha256=doc.document_sha256,
        allowed_query_types=("exact_code", "semantic", "unanswerable"),
        requested_count=2,
    )
    assert len(outcomes) == 2
    for c in service.repo.list_candidates(session.session_id):
        assert c.status == CANDIDATE_PROPOSED


def test_partial_or_failed_response_persists_no_candidate(tmp_path):
    """§十一.10：Provider Timeout／半完成回應不得留下 Candidate。"""
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    session = service.create_session(
        session_name="s", base_dataset_path=make_base_dataset(tmp_path, corpus)
    )
    doc = corpus.by_filename("equipment_alarm_manual.md")

    # 第二筆缺欄位 → 整批失敗
    payload = json.loads(_valid_payload(2))
    del payload["candidates"][1]["question"]
    gen = LLMCandidateGenerator(FakeLLM(response=json.dumps(payload, ensure_ascii=False)))
    with pytest.raises(GeneratorError):
        service.generate_candidates(
            session.session_id,
            gen,
            document_sha256=doc.document_sha256,
            allowed_query_types=("exact_code", "semantic", "unanswerable"),
            requested_count=2,
        )
    assert service.repo.list_candidates(session.session_id) == []

    # Timeout 同樣不留半成品
    gen_timeout = LLMCandidateGenerator(FakeLLM(error=LLMTimeoutError()))
    with pytest.raises(GeneratorError):
        service.generate_candidates(
            session.session_id,
            gen_timeout,
            document_sha256=doc.document_sha256,
            requested_count=2,
        )
    assert service.repo.list_candidates(session.session_id) == []
