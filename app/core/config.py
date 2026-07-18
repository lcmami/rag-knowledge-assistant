"""應用程式設定。

所有可調參數皆由環境變數 / .env 載入（規格 §26）。
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 應用程式 ---
    app_name: str = "factory-rag-assistant"
    app_env: str = "development"
    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 8000

    # --- 檔案上傳 ---
    max_upload_size_mb: int = 10
    upload_dir: str = "storage/uploads"
    sqlite_path: str = "storage/metadata.db"

    # --- Chunking ---
    chunk_size_chars: int = 600
    chunk_overlap_chars: int = 100
    embedding_max_tokens: int = 480

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："

    # --- Qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "factory_knowledge"

    # --- Retrieval ---
    retrieval_default_top_k: int = 5
    retrieval_max_top_k: int = 10
    # 0.55 為目前資料集的 threshold 校準結果（Phase 5 §2）；正式 /api/query 預設採用。
    # 保留環境變數覆寫能力；Evaluation Script 仍可明確指定 0.0 進行原始分數分析。
    retrieval_score_threshold: float = 0.55

    # --- LLM ---
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_timeout_seconds: int = 60
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1000

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        # §3.3：必要設定為 base_url、api_key、model（本地 vLLM 可設 LLM_API_KEY=EMPTY）。
        return bool(self.llm_base_url and self.llm_api_key and self.llm_model)


def get_settings() -> Settings:
    """建立 Settings 實例（於 lifespan 呼叫一次）。"""
    return Settings()
