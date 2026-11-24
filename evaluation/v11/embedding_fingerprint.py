"""Embedding Fingerprint 與 Cache Key（v1.1 Slice 3 §十四 / §十五）。

**Fingerprint**：涵蓋實際影響向量輸出的設定（provider / model / revision / dimension /
normalize / pooling / cache schema version）。fingerprint hash 改變 → 必 cache miss。

**Cache Key**：``SHA256(key_algo_version + cache_schema_version + fingerprint_hash +
sha256(exact_embedding_input_text))``。**只**取決於模型 fingerprint 與**實際送給 embedding
model 的完整 UTF-8 文字**（含 section prefix / context / core），**不含** document_id / page /
section / chunk_index / identifier metadata / identifier_normalization_version / trial_id /
collection / threshold / top_k / candidate 參數。故：

- 相同文字 + 相同模型（即使來自不同 Trial / 文件位置）→ 重用向量（Hit）。
- 只改 threshold / top_k / identifier version → 文字與 fingerprint 不變 → Hit。
- 開啟 section prefix / 改 prefix 格式版本 → 文字改變 → Miss。

安全（§十四.1/2、§十五.4）：不含 API key；不含本機絕對模型路徑（路徑 → 穩定 model identifier）。
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Optional

from evaluation.v11.hashing import canonical_json

# Cache schema 版本（納入 fingerprint 與 cache key；§十四/§十五）。
CACHE_SCHEMA_VERSION = "v1"
# Cache Key 推導演算法版本（§十五.5：算法需版本化）。
CACHE_KEY_ALGO_VERSION = "v1"

# fingerprint 分隔用不可見字元（避免欄位串接歧義）。
_SEP = "\x1f"

REVISION_UNKNOWN = "unknown"
POOLING_UNKNOWN = "unknown"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_model_identifier(model_name: str) -> str:
    """把 model 設定轉為**穩定、不含私人絕對路徑**的 identifier（§十四.2）。

    - HuggingFace repo id（如 ``BAAI/bge-small-zh-v1.5``）原樣保留。
    - 絕對 / 含 OS 分隔符的本機路徑 → 取最後一段目錄名（穩定、不外洩私人目錄）。
    """
    if not model_name:
        return ""
    # Python 3.13+ 在 Windows 上 os.path.isabs("/opt/x") 為 False，需另判 POSIX 絕對路徑。
    looks_like_path = (
        os.path.isabs(model_name)
        or model_name.startswith("/")
        or ("\\" in model_name)
        or ("/" in model_name and (os.sep in model_name or ":" in model_name))
    )
    if looks_like_path:
        return os.path.basename(model_name.rstrip("/\\")) or model_name
    return model_name


def text_sha256(text: str) -> str:
    """embedding input text 的 exact UTF-8 SHA-256（§十五.1：不依賴 python hash()）。"""
    return _sha256(text)


@dataclass(frozen=True)
class EmbeddingFingerprint:
    provider: str
    model_name: str
    model_revision: str
    dimension: int
    normalize_embeddings: bool
    pooling_mode: str
    cache_schema_version: str = CACHE_SCHEMA_VERSION

    def to_canonical_dict(self) -> dict[str, Any]:
        """deterministic、不含敏感資訊的 fingerprint（供 hash 與 Report）。"""
        return {
            "provider": self.provider,
            "model_name": stable_model_identifier(self.model_name),
            "model_revision": self.model_revision or REVISION_UNKNOWN,
            "dimension": self.dimension,
            "normalize_embeddings": bool(self.normalize_embeddings),
            "pooling_mode": self.pooling_mode or POOLING_UNKNOWN,
            "cache_schema_version": self.cache_schema_version,
        }

    def fingerprint_hash(self) -> str:
        return _sha256(canonical_json(self.to_canonical_dict()))

    def cache_key(self, embedding_input_text: str) -> str:
        """§十五 cache key：只依 fingerprint 與實際 embedding 文字。"""
        payload = _SEP.join(
            [
                CACHE_KEY_ALGO_VERSION,
                self.cache_schema_version,
                self.fingerprint_hash(),
                text_sha256(embedding_input_text),
            ]
        )
        return _sha256(payload)

    # ------------------------------------------------------------------
    @classmethod
    def from_service(
        cls,
        embedding: Any,
        *,
        normalize_embeddings: bool = True,
        provider: str = "sentence_transformers",
        cache_schema_version: str = CACHE_SCHEMA_VERSION,
    ) -> "EmbeddingFingerprint":
        """由 EmbeddingService 建立 fingerprint（best-effort 讀 revision / pooling）。

        正式 :class:`app.services.embedding_service.EmbeddingService` 一律 normalize；本機
        BGE 於離線環境常無法取得 pinned revision → 誠實記為 ``unknown``（§十四.3：不假裝已 pin）。
        """
        return cls(
            provider=provider,
            model_name=getattr(embedding, "model_name", ""),
            model_revision=_read_revision(embedding),
            dimension=int(embedding.dimension),
            normalize_embeddings=normalize_embeddings,
            pooling_mode=_read_pooling(embedding),
            cache_schema_version=cache_schema_version,
        )


def _read_revision(embedding: Any) -> str:
    """best-effort 取得 model revision；取不到即 ``unknown``（§十四.3）。"""
    explicit = getattr(embedding, "model_revision", None)
    if explicit:
        return str(explicit)
    model = getattr(embedding, "_model", None)
    for attr in ("model_card_data", "config"):
        obj = getattr(model, attr, None)
        rev = getattr(obj, "base_model_revision", None) or getattr(obj, "_commit_hash", None)
        if rev:
            return str(rev)
    return REVISION_UNKNOWN


def _read_pooling(embedding: Any) -> str:
    """best-effort 取得 pooling mode；取不到即 ``unknown``。"""
    explicit = getattr(embedding, "pooling_mode", None)
    if explicit:
        return str(explicit)
    model = getattr(embedding, "_model", None)
    try:
        for module in model.modules():  # type: ignore[union-attr]
            name = type(module).__name__
            if name == "Pooling" and hasattr(module, "get_pooling_mode_str"):
                return str(module.get_pooling_mode_str())
    except Exception:  # noqa: BLE001 - 取不到即 unknown
        pass
    return POOLING_UNKNOWN


def make_fingerprint(
    *,
    provider: str,
    model_name: str,
    model_revision: Optional[str],
    dimension: int,
    normalize_embeddings: bool,
    pooling_mode: Optional[str],
    cache_schema_version: str = CACHE_SCHEMA_VERSION,
) -> EmbeddingFingerprint:
    """明確欄位建立 fingerprint（供單元測試不載入模型）。"""
    return EmbeddingFingerprint(
        provider=provider,
        model_name=model_name,
        model_revision=model_revision or REVISION_UNKNOWN,
        dimension=dimension,
        normalize_embeddings=normalize_embeddings,
        pooling_mode=pooling_mode or POOLING_UNKNOWN,
        cache_schema_version=cache_schema_version,
    )
