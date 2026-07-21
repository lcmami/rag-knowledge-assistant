"""Versioned Deployment Collection Builder（Slice 7 §十二）。

以 candidate 的**實際** chunk / retrieval config 重建一個全新的 versioned physical collection：

    Load Candidate Config → Load Target Corpus → Rebuild Canonical Documents
    → Apply Candidate Chunking → Embed → Create Collection → Upload Points
    → Build Payload Indexes → Return Build Manifest

與 :func:`evaluation.v11.real_services.RealTrialServiceProvider.build_index` 走同一套隔離
索引管線（loader → canonical → chunking strategy → caching embed → qdrant upsert），但：

- 目標名稱為 versioned deployment collection（非 ``optimizer_`` trial 名稱），因此改用
  :func:`~evaluation.v11.deploy.naming.assert_not_protected` 防護（**永不**建/覆寫 ``factory_knowledge``）。
- 已存在相同名稱 → 拒絕（不覆寫，§十二.3/.4）；build 失敗 → cleanup 半成品（§十二.10/.11）。
- 不修改目前 alias target（§十二.12）。
"""
from __future__ import annotations

import hashlib
import os
import uuid
from typing import Optional

from evaluation.v11.chunking import ChunkingConfig, ChunkingContext, build_strategy
from evaluation.v11.deploy.models import (
    DeploymentBuildManifest,
    DeploymentError,
    ErrorCode,
)
from evaluation.v11.deploy.naming import assert_not_protected, safe_delete_collection
from evaluation.v11.embedding_cache import (
    CachingDocumentEmbeddingService,
    EmbeddingCacheRepository,
)
from evaluation.v11.embedding_fingerprint import EmbeddingFingerprint
from evaluation.v11.hashing import collect_corpus_files, config_hash
from evaluation.v11.run_models import TrialConfig

_DOCUMENT_TYPES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "txt",
    ".pdf": "pdf",
}


class DeploymentCollectionBuilder:
    """建立 versioned deployment collection（不碰正式 collection / alias）。"""

    def __init__(
        self,
        client,
        embedding,
        *,
        cache_db_path: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        self._client = client
        self._embedding = embedding
        self._cache_enabled = cache_enabled and bool(cache_db_path)
        self._cache_repo = (
            EmbeddingCacheRepository(cache_db_path) if self._cache_enabled else None
        )
        self._fingerprint: Optional[EmbeddingFingerprint] = None
        self._caching: Optional[CachingDocumentEmbeddingService] = None

    def _caching_service(self) -> CachingDocumentEmbeddingService:
        if self._caching is None:
            self._fingerprint = EmbeddingFingerprint.from_service(
                self._embedding, normalize_embeddings=True
            )
            self._caching = CachingDocumentEmbeddingService(
                self._embedding, self._cache_repo, self._fingerprint,
                enabled=self._cache_enabled,
            )
        return self._caching

    def cleanup(self, collection_name: str, *, active_targets: tuple[str, ...] = ()) -> bool:
        """安全刪除本 builder 建立、尚未 active 的 collection（§十二.11）。"""
        return safe_delete_collection(
            self._client, collection_name, active_targets=active_targets
        )

    def build(
        self,
        collection_name: str,
        candidate_config_json: str,
        corpus_paths: list[str] | tuple[str, ...],
        *,
        corpus_base_dir: Optional[str] = None,
        expected_config_hash: str,
        corpus_hash: str,
        canonical_text_version: str,
        identifier_normalization_version: str,
    ) -> DeploymentBuildManifest:
        import json

        from app.services.canonical_builder import build_canonical_document
        from app.services.document_loader import load_document
        from app.services.vector_store import VectorStore

        assert_not_protected(collection_name)  # §四.10 硬編碼防護

        # §十二.3/.4：不得覆寫既有 collection（idempotency resume 由 service 處理）。
        if self._client.collection_exists(collection_name):
            raise DeploymentError(
                ErrorCode.COLLECTION_NAME_CONFLICT,
                f"collection {collection_name!r} 已存在，拒絕覆寫（§十二.3）。",
            )

        raw = json.loads(candidate_config_json)
        trial_config = TrialConfig.from_raw(
            {
                "trial_name": raw.get("trial_name", "deployment"),
                "chunk_config": raw.get("chunk_config") or {"strategy": "current"},
                "retrieval_config": raw.get("retrieval_config") or {},
                "collection_options": raw.get("collection_options") or {},
            },
            0,
        )
        chunk_cfg = ChunkingConfig.from_raw(trial_config.chunk_config)
        chunk_cfg_hash = config_hash(chunk_cfg.to_canonical_dict())
        strategy_name = trial_config.strategy  # dense / identifier_aware
        chunk_strategy = build_strategy(chunk_cfg.strategy)
        caching = self._caching_service()
        caching.reset_stats()

        built = False
        try:
            files = collect_corpus_files(corpus_paths, corpus_base_dir)
            store = VectorStore(self._client, collection_name)
            store.ensure_collection(caching.dimension)  # 建 collection + payload indexes
            total_chunks = 0
            payload_fields: set[str] = set()
            for name, abs_path in files:
                with open(abs_path, "rb") as f:
                    data = f.read()
                file_hash = hashlib.sha256(data).hexdigest()
                elements = load_document(name, data)
                ext = os.path.splitext(name.lower())[1]
                document_type = _DOCUMENT_TYPES.get(ext, "txt")
                canonical = build_canonical_document(
                    filename=name,
                    document_sha256=file_hash,
                    document_type=document_type,
                    elements=elements,
                )
                ctx = ChunkingContext(
                    document_id=str(uuid.uuid4()),
                    filename=name,
                    file_hash=file_hash,
                    canonical=canonical,
                    elements=elements,
                    config=chunk_cfg,
                    count_tokens=caching.count_tokens,
                )
                outcome = chunk_strategy.chunk(ctx)
                if not outcome.chunks:
                    continue
                vectors = caching.embed_documents([c.text for c in outcome.chunks])
                store.upsert_chunks(outcome.chunks, vectors)
                total_chunks += len(outcome.chunks)
                for c in outcome.chunks:
                    payload_fields.update(c.to_payload().keys())

            point_count = self._client.count(collection_name).count
            info = self._client.get_collection(collection_name)
            vector_size = info.config.params.vectors.size
            distance = str(info.config.params.vectors.distance)
            built = True
        except DeploymentError:
            raise
        except Exception as exc:  # noqa: BLE001 - build 失敗一律 cleanup 半成品（§十二.10）
            self._safe_cleanup_on_failure(collection_name)
            raise DeploymentError(
                ErrorCode.BUILD_FAILED, f"deployment collection build 失敗：{exc}"
            ) from exc

        assert self._fingerprint is not None
        return DeploymentBuildManifest(
            collection_name=collection_name,
            point_count=point_count,
            chunk_count=total_chunks,
            vector_size=vector_size,
            distance=distance,
            strategy=strategy_name,
            config_hash=expected_config_hash,
            chunk_config_hash=chunk_cfg_hash,
            corpus_hash=corpus_hash,
            canonical_text_version=canonical_text_version,
            identifier_normalization_version=identifier_normalization_version,
            embedding_fingerprint_hash=self._fingerprint.fingerprint_hash(),
            identifier_index=(strategy_name == "identifier_aware"),
            payload_fields=sorted(payload_fields),
            reused_existing=False,
        )

    def _safe_cleanup_on_failure(self, collection_name: str) -> None:
        try:
            assert_not_protected(collection_name)
            if self._client.collection_exists(collection_name):
                self._client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001
            pass
