"""RealTrialServiceProvider（Slice 2 §九 / Slice 3 §十九）。

以正式 app 服務（EmbeddingService / RetrievalService / VectorStore + Loader / Canonical
Builder）實作 :class:`~evaluation.v11.trial_runner.TrialServiceProvider`，並於 Slice 3 加入：

- 依 Trial Config 選 :class:`ChunkingStrategy`（current / fixed / paragraph / heading）。
- 以 :class:`CachingDocumentEmbeddingService` 對 **document chunk embedding** 加 persistent
  cache（cache DB 為獨立、可注入路徑；跨同一 Run 的 Trial 共用 → 相同文字重用向量）。
- Query embedding 一律用底層服務（不快取，§十七.10）。

隔離保證（不變）：
- 每 Trial 獨立 ``optimizer_`` 前綴 collection；刪除前一律驗證前綴。
- **永不**觸碰正式 ``factory_knowledge`` / 正式 ``metadata.db`` / 正式 uploads。
- 正式 IndexingService / TextSplitter **完全不使用、不修改**——Trial 索引走本模組的隔離管線
  （``current`` strategy 仍委派正式 TextSplitter 核心，故與正式切分 byte-identical）。
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from evaluation.v11.chunking import ChunkingConfig, ChunkingContext, build_strategy
from evaluation.v11.embedding_cache import (
    CachingDocumentEmbeddingService,
    EmbeddingCacheRepository,
)
from evaluation.v11.embedding_fingerprint import EmbeddingFingerprint
from evaluation.v11.hashing import config_hash
from evaluation.v11.retrieval_metrics import IndexChunk, RankedResult
from evaluation.v11.run_models import TrialConfig
from evaluation.v11.trial_collections import assert_deletable
from evaluation.v11.trial_runner import IndexOutcome, RetrievalOutcome

# 副檔名 → canonical document_type（offset scope：pdf=page、其餘=document）。
_DOCUMENT_TYPES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "txt",
    ".pdf": "pdf",
}


@dataclass
class _TrialContext:
    store: object  # VectorStore（bound 到隔離 collection；backend 由此取 client + collection）


class RealTrialServiceProvider:
    """真實服務 provider；embedding / qdrant client / cache repo 於整個 Run 共用。"""

    def __init__(
        self,
        client,
        embedding,
        base_workdir: str,
        base_settings,
        *,
        cache_db_path: Optional[str] = None,
        cache_enabled: bool = True,
    ):
        self._client = client
        self._embedding = embedding
        self._base_workdir = base_workdir
        self._base_settings = base_settings
        self._contexts: dict[str, _TrialContext] = {}
        os.makedirs(base_workdir, exist_ok=True)
        # 獨立、可注入的 embedding cache DB（**不**混入正式 metadata.db / optimizer run DB）。
        self._cache_enabled = cache_enabled
        self._cache_db_path = cache_db_path or os.path.join(
            base_workdir, "optimizer.embedding-cache.db"
        )
        self._cache_repo: Optional[EmbeddingCacheRepository] = (
            EmbeddingCacheRepository(self._cache_db_path) if cache_enabled else None
        )
        self._fingerprint: Optional[EmbeddingFingerprint] = None
        self._caching: Optional[CachingDocumentEmbeddingService] = None

    @classmethod
    def create(
        cls,
        qdrant_url: str,
        base_workdir: str,
        *,
        base_settings=None,
        embedding=None,
        cache_db_path: Optional[str] = None,
        cache_enabled: bool = True,
    ) -> "RealTrialServiceProvider":
        from app.core.config import Settings
        from app.dependencies import create_qdrant_client

        settings = base_settings or Settings(app_env="optimizer", qdrant_url=qdrant_url)
        if embedding is None:
            from app.services.embedding_service import EmbeddingService

            embedding = EmbeddingService(
                model_name=settings.embedding_model,
                query_instruction=settings.embedding_query_instruction,
                max_tokens=settings.embedding_max_tokens,
            )
        client = create_qdrant_client(qdrant_url)
        return cls(
            client, embedding, base_workdir, settings,
            cache_db_path=cache_db_path, cache_enabled=cache_enabled,
        )

    # ------------------------------------------------------------------
    def _caching_service(self) -> CachingDocumentEmbeddingService:
        """惰性建立跨 Trial 共用的 caching document embedding service。"""
        if self._caching is None:
            self._fingerprint = EmbeddingFingerprint.from_service(
                self._embedding, normalize_embeddings=True
            )
            self._caching = CachingDocumentEmbeddingService(
                self._embedding,
                self._cache_repo,
                self._fingerprint,
                enabled=self._cache_enabled,
            )
        return self._caching

    # --- provider API ---
    def build_index(
        self, collection_name: str, trial_config: TrialConfig, files: list[tuple[str, str]]
    ) -> IndexOutcome:
        from app.services.canonical_builder import build_canonical_document
        from app.services.document_loader import load_document
        from app.services.vector_store import VectorStore

        assert_deletable(collection_name)  # optimizer_ 前綴（防誤用正式名稱）

        chunk_cfg = ChunkingConfig.from_raw(trial_config.chunk_config or {"strategy": "current"})
        chunk_cfg_hash = config_hash(chunk_cfg.to_canonical_dict())
        strategy = build_strategy(chunk_cfg.strategy)
        caching = self._caching_service()
        caching.reset_stats()

        store = VectorStore(self._client, collection_name)
        # 隔離：確保 collection 為全新（存在則先刪，只刪 optimizer_ 前綴）。
        try:
            if self._client.collection_exists(collection_name):
                assert_deletable(collection_name)
                self._client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001
            pass

        start = time.perf_counter()
        store.ensure_collection(caching.dimension)

        total_chunks = 0
        fallback_count = 0
        fallback_reasons: list[str] = []
        context_added = 0
        context_reduced = 0

        for name, abs_path in files:
            with open(abs_path, "rb") as f:
                raw = f.read()
            file_hash = hashlib.sha256(raw).hexdigest()
            elements = load_document(name, raw)
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
            outcome = strategy.chunk(ctx)
            if not outcome.chunks:
                continue
            vectors = caching.embed_documents([c.text for c in outcome.chunks])
            store.upsert_chunks(outcome.chunks, vectors)
            total_chunks += len(outcome.chunks)
            fallback_count += outcome.fallback_count
            fallback_reasons.extend(outcome.fallback_reasons)
            context_added += outcome.context_added_count
            context_reduced += outcome.context_reduced_count

        build_time_ms = int((time.perf_counter() - start) * 1000)
        point_count = self._client.count(collection_name).count

        # Slice 4：retrieval 改由版本化 RetrievalStrategy 執行（dense / identifier_aware）；
        # query embedding 於 retrieve() 一次算好、走底層不快取（§十七.10 / §十一.5）。
        self._contexts[collection_name] = _TrialContext(store)

        assert self._fingerprint is not None
        stats = {
            "stats_version": "v1",
            "strategy": chunk_cfg.strategy,
            "chunk_config": chunk_cfg.to_canonical_dict(),
            "chunk_config_hash": chunk_cfg_hash,
            # 可重現性語意（Slice 3 §一）：chunk_config 為 Trial 明確展開的值，
            # **非**執行環境 .env / Settings 覆寫後的即時 Production Config。
            "chunk_config_origin": "explicit_trial_config",
            "chunk_count": total_chunks,
            "strategy_fallback_count": fallback_count,
            "strategy_fallback_reasons": sorted(set(fallback_reasons)),
            "section_prefix_enabled": chunk_cfg.prepend_section_title,
            "context_config": {
                "context_before_paragraphs": chunk_cfg.context_before_paragraphs,
                "context_after_paragraphs": chunk_cfg.context_after_paragraphs,
            },
            "context_added_count": context_added,
            "context_reduced_count": context_reduced,
            "index_build_time_ms": build_time_ms,
            "embedding_cache": caching.snapshot_stats().to_dict(),
            "embedding_fingerprint": self._fingerprint.to_canonical_dict(),
            "embedding_fingerprint_hash": self._fingerprint.fingerprint_hash(),
            "cache_schema_version": self._fingerprint.cache_schema_version,
        }
        return IndexOutcome(
            point_count=point_count,
            build_time_ms=build_time_ms,
            chunk_count=total_chunks,
            strategy=chunk_cfg.strategy,
            chunk_config_hash=chunk_cfg_hash,
            stats=stats,
        )

    def index_manifest(self, collection_name: str) -> list[IndexChunk]:
        """單次 scroll 該 Trial 自己的 collection，取回全部 Chunk 作為 Index Universe。"""
        assert_deletable(collection_name)  # 前綴保護
        chunks: list[IndexChunk] = []
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                chunks.append(
                    IndexChunk(
                        document_id=payload.get("document_id"),
                        document_sha256=payload.get("file_hash"),
                        chunk_index=payload.get("chunk_index"),
                        text=payload.get("text") or "",
                        page=payload.get("page"),
                        section=payload.get("section"),
                        provenance_spans=payload.get("provenance_spans") or [],
                        identifiers=payload.get("identifiers") or [],
                        canonical_text_version=payload.get("canonical_text_version"),
                    )
                )
            if not next_offset:
                break
        return chunks

    def retrieve(
        self, collection_name: str, trial_config: TrialConfig, question: str, top_k: int
    ) -> RetrievalOutcome:
        from evaluation.v11.retrieval.base import RetrievalRequest, build_retrieval_strategy
        from evaluation.v11.retrieval.qdrant_backend import QdrantSearchBackend

        ctx = self._contexts.get(collection_name)
        if ctx is None:
            raise RuntimeError(f"collection {collection_name!r} 尚未建索引。")

        rc = trial_config.retrieval_config_obj()
        start = time.perf_counter()
        # 每 Trial 每題算一次 query embedding（不快取，§十七.10 / §十八.11）。
        query_vector = self._embedding.embed_query(question)
        backend = QdrantSearchBackend(ctx.store.client, ctx.store.collection)
        strategy = build_retrieval_strategy(rc.strategy)
        outcome = strategy.retrieve(
            RetrievalRequest(
                query=question,
                query_vector=query_vector,
                collection_name=collection_name,
                config=rc,
            ),
            backend,
        )
        retrieval_time_ms = int((time.perf_counter() - start) * 1000)

        ranked = [
            RankedResult(
                rank=r.rank,
                document_id=r.document_id,
                document_sha256=r.file_hash,
                filename=r.filename,
                section=r.section,
                page=r.page,
                chunk_index=r.chunk_index,
                score=r.vector_score,  # §十七：score 保持 raw vector score
                text=r.text,
                provenance_spans=list(r.provenance_spans or []),
                identifiers=list(r.identifiers or []),
                canonical_text_version=r.canonical_text_version,
                ranking_score=r.ranking_score,
                exact_identifier_matches=list(r.exact_identifier_matches),
                conflicting_identifiers=list(r.conflicting_identifiers),
                exact_match_boost=r.exact_match_boost,
                confusable_penalty=r.confusable_penalty,
                candidate_sources=list(r.candidate_sources),
                admission_reason=r.admission_reason,
            )
            for r in outcome.results
        ]
        # dense 也提供 meta（strategy/ranking version + counts），供 Report 一致呈現。
        return RetrievalOutcome(
            results=ranked, retrieval_time_ms=retrieval_time_ms, meta=outcome.meta()
        )

    def collection_exists(self, collection_name: str) -> bool:
        return bool(self._client.collection_exists(collection_name))

    def delete_collection(self, collection_name: str) -> None:
        assert_deletable(collection_name)
        self._client.delete_collection(collection_name)
        self._contexts.pop(collection_name, None)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
