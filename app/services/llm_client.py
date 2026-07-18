"""OpenAI-compatible LLM Client（規格 Phase 5 §3）。

以 OpenAI Python SDK 呼叫 OpenAI-compatible Chat Completions API，
不寫死特定供應商（可指向 OpenAI、vLLM、Ollama 等）。

設計原則：
- 於 FastAPI lifespan 初始化一次（build_services），不每次 Request 重建 client。
- 第一版以最大相容性為優先：不開啟 Streaming / Function Calling / Tools / JSON Schema。
- 錯誤分類為明確的應用層例外，交由 API error handler 轉譯；
  不外洩 API Key、完整 traceback 或 Provider 內部敏感資料（§3.4）。
"""
from __future__ import annotations

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
)

from app.core.config import Settings
from app.core.exceptions import (
    LLMEmptyResponseError,
    LLMNotConfiguredError,
    LLMServiceError,
    LLMTimeoutError,
)
from app.core.logging import get_logger

logger = get_logger("app.services.llm_client")


class LLMClient:
    """OpenAI-compatible Chat Completions 封裝。

    未設定必要環境變數時 configured=False，呼叫 complete() 會拋 LLMNotConfiguredError；
    /api/retrieve 仍可正常運作，/api/query 則回明確 503。
    """

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None):
        self._settings = settings
        if client is not None:
            # 測試可注入 fake / mock async client。
            self._client = client
        elif settings.llm_configured:
            self._client = AsyncOpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                timeout=settings.llm_timeout_seconds,
            )
        else:
            self._client = None

    @property
    def configured(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self._settings.llm_model

    async def complete(self, messages: list[dict]) -> str:
        """呼叫 Chat Completions 並回傳非空文字內容。

        錯誤分類（§3.4）：
        - 未設定           → LLMNotConfiguredError (503)
        - Timeout          → LLMTimeoutError (504)
        - 連線 / Provider  → LLMServiceError (502)
        - 空 / 無效回應    → LLMEmptyResponseError (502)
        """
        if self._client is None:
            raise LLMNotConfiguredError()

        try:
            response = await self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
        except APITimeoutError as exc:
            # 只記錄型別，不記錄可能含敏感資訊的完整訊息。
            logger.warning("LLM timeout: %s", type(exc).__name__)
            raise LLMTimeoutError() from exc
        except APIConnectionError as exc:
            logger.warning("LLM connection error: %s", type(exc).__name__)
            raise LLMServiceError() from exc
        except APIStatusError as exc:
            # Provider 回傳非 2xx；記錄 status code 但不外洩內容。
            status = getattr(exc, "status_code", "unknown")
            logger.warning("LLM provider error: status=%s", status)
            raise LLMServiceError() from exc
        except OpenAIError as exc:
            logger.warning("LLM error: %s", type(exc).__name__)
            raise LLMServiceError() from exc

        choices = getattr(response, "choices", None)
        if not choices:
            logger.warning("LLM returned empty choices")
            raise LLMEmptyResponseError()

        content = choices[0].message.content
        if content is None or not content.strip():
            logger.warning("LLM returned empty content")
            raise LLMEmptyResponseError()

        return content
