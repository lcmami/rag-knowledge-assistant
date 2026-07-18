"""LLM Client 單元測試（Phase 5 §15.1）。

全部使用 mock，不呼叫網路：驗證傳入參數、錯誤分類與 API Key 不外洩。
以 asyncio.run 驅動 async 方法，避免額外的 pytest async 外掛依賴。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from app.core.config import Settings
from app.core.exceptions import (
    LLMEmptyResponseError,
    LLMNotConfiguredError,
    LLMServiceError,
    LLMTimeoutError,
)
from app.services.llm_client import LLMClient

SECRET_KEY = "sk-super-secret-key-should-never-leak-9999"
_REQ = httpx.Request("POST", "http://test/v1/chat/completions")


def _configured_settings(**kw) -> Settings:
    base = dict(
        llm_base_url="http://llm.test/v1",
        llm_api_key=SECRET_KEY,
        llm_model="test-model",
        llm_temperature=0.0,
        llm_max_tokens=1000,
        llm_timeout_seconds=60,
    )
    base.update(kw)
    return Settings(**base)


def _fake_response(content: str | None, with_choice: bool = True):
    if not with_choice:
        return SimpleNamespace(choices=[])
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client_with_create(create_mock: AsyncMock) -> LLMClient:
    fake_async = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
    )
    return LLMClient(_configured_settings(), client=fake_async)


# 1. 正確傳入 model、temperature、max_tokens；並驗證 base_url 傳給 AsyncOpenAI
def test_passes_correct_params(monkeypatch):
    create = AsyncMock(return_value=_fake_response("答案 [S1]"))
    client = _client_with_create(create)

    messages = [{"role": "user", "content": "hi"}]
    out = asyncio.run(client.complete(messages))

    assert out == "答案 [S1]"
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["messages"] == messages
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 1000
    # 不得開啟 streaming / tools
    assert "stream" not in kwargs or kwargs["stream"] is False
    assert "tools" not in kwargs


def test_constructor_passes_base_url_and_timeout(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr("app.services.llm_client.AsyncOpenAI", FakeAsyncOpenAI)
    client = LLMClient(_configured_settings())
    assert client.configured is True
    assert captured["base_url"] == "http://llm.test/v1"
    assert captured["api_key"] == SECRET_KEY
    assert captured["timeout"] == 60


# 2. 未設定時拋 LLMNotConfiguredError
def test_not_configured_raises():
    settings = Settings(llm_base_url="", llm_api_key="", llm_model="")
    client = LLMClient(settings)
    assert client.configured is False
    with pytest.raises(LLMNotConfiguredError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


# 3. Timeout 轉為 LLMTimeoutError
def test_timeout_maps_to_timeout_error():
    create = AsyncMock(side_effect=openai.APITimeoutError(request=_REQ))
    client = _client_with_create(create)
    with pytest.raises(LLMTimeoutError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


# 4. Provider error 轉為 LLMServiceError（連線與非 2xx 皆為 502）
def test_connection_error_maps_to_service_error():
    create = AsyncMock(side_effect=openai.APIConnectionError(message="conn", request=_REQ))
    client = _client_with_create(create)
    with pytest.raises(LLMServiceError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


def test_provider_status_error_maps_to_service_error():
    resp = httpx.Response(status_code=500, request=_REQ, text="internal provider detail")
    create = AsyncMock(side_effect=openai.APIStatusError("boom", response=resp, body=None))
    client = _client_with_create(create)
    with pytest.raises(LLMServiceError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


# 5. 空 choices 或空 content 被拒絕
def test_empty_choices_rejected():
    create = AsyncMock(return_value=_fake_response(None, with_choice=False))
    client = _client_with_create(create)
    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


def test_empty_content_rejected():
    create = AsyncMock(return_value=_fake_response("   "))
    client = _client_with_create(create)
    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


def test_none_content_rejected():
    create = AsyncMock(return_value=_fake_response(None))
    client = _client_with_create(create)
    with pytest.raises(LLMEmptyResponseError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


# 6. 不外洩 API Key（任何被拋出的例外訊息都不得包含 key）
def test_api_key_not_leaked_in_errors():
    resp = httpx.Response(status_code=401, request=_REQ, text=f"auth failed for {SECRET_KEY}")
    create = AsyncMock(side_effect=openai.APIStatusError("unauthorized", response=resp, body=None))
    client = _client_with_create(create)
    with pytest.raises(LLMServiceError) as exc_info:
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert SECRET_KEY not in str(exc_info.value)
    assert SECRET_KEY not in str(exc_info.value.details)
