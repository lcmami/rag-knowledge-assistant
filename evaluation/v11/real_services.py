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
from evaluation.v11.rerank_cache import FingerprintedRerankCache, RerankCacheRepository
from evaluation.v11.rerank_fingerprint import RerankFingerprint
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
        rerank_cache_db_path: Optional[str] = None,
        rerank_cache_enabled: bool = True,
        rerank_model_name: Optional[str] = None,
        reranker=None,
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

        # --- Slice R2（v1.2.1）：rerank ------------------------------------
        # Model 與 cache 皆為**惰性建立** —— 只有真的跑到 rerank_strategy != none 的
        # Trial 時才會載入 cross-encoder，故既有 dense / identifier_aware run 的成本不變。
        #
        # ``rerank_cache_enabled=False`` 是 **uncached 量測用開關**：R2 的 end-to-end
        # feasibility 數字一律以 uncached 量得，cached latency 只作資訊（R0/R0.5 已載明）。
        from evaluation.v11.retrieval.config import DEFAULT_RERANK_MODEL

        self._rerank_model_name = rerank_model_name or DEFAULT_RERANK_MODEL
        self._rerank_cache_enabled = rerank_cache_enabled
        self._rerank_cache_db_path = rerank_cache_db_path or os.path.join(
            base_workdir, "optimizer.rerank-cache.db"
        )
        self._reranker = reranker  # 可注入（測試 / 重用已載入模型）
        self._rerank_cache_repo: Optional[RerankCacheRepository] = None
        self._rerank_cache: Optional[FingerprintedRerankCache] = None
        self._rerank_fingerprint: Optional[RerankFingerprint] = None

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
        rerank_cache_db_path: Optional[str] = None,
        rerank_cache_enabled: bool = True,
        rerank_model_name: Optional[str] = None,
        reranker=None,
    ) -> "RealTrialServiceProvider":
        from app.core.config import Settings
        from app.dependencies import create_qdrant_client

        settings = base_settings or Settings(app_env="optimizer", qdrant_url=qdrant_url)
        if embedding is None:
            from app.services.embedding_service import EmbeddingService

            embedding = EmbeddingService(
                model_name=settings.embedding_model,
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
                dimension=settings.embedding_dimension,
                query_instruction=settings.embedding_query_instruction,
                max_tokens=settings.embedding_max_tokens,
            )
        client = create_qdrant_client(qdrant_url)
        return cls(
            client, embedding, base_workdir, settings,
            cache_db_path=cache_db_path, cache_enabled=cache_enabled,
            rerank_cache_db_path=rerank_cache_db_path,
            rerank_cache_enabled=rerank_cache_enabled,
            rerank_model_name=rerank_model_name,
            reranker=reranker,
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

    # ------------------------------------------------------------------
    # Slice R2：真實 cross-encoder reranker + 正式 SQLite rerank cache
    # ------------------------------------------------------------------
    def _ensure_reranker(self):
        """惰性建立真實 :class:`CrossEncoderReranker`（整個 Run 共用一份已載入模型）。"""
        if self._reranker is None:
            from evaluation.v11.retrieval.reranker import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(self._rerank_model_name)
        return self._reranker

    def _ensure_rerank_fingerprint(self) -> RerankFingerprint:
        """由**已載入**的 cross-encoder 建立 fingerprint（cache key 綁定於此）。

        必須先觸發模型載入才能讀出 revision / num_labels / max_length / weights manifest；
        否則 fingerprint 會退化成 ``unknown``，快取語意就不可信。
        """
        if self._rerank_fingerprint is None:
            reranker = self._ensure_reranker()
            model = reranker._ensure_model()
            self._rerank_fingerprint, _manifest = RerankFingerprint.from_cross_encoder(
                model, model_name=self._rerank_model_name
            )
        return self._rerank_fingerprint

    def _ensure_rerank_cache(self) -> FingerprintedRerankCache:
        """惰性建立正式 SQLite rerank cache（獨立檔案，不混入任何其他 DB）。

        ``rerank_cache_enabled=False`` 時仍回傳 facade，但一律 miss、不寫入 ——
        這正是 uncached end-to-end 量測所需的行為（且 miss 統計仍如實計數）。
        """
        if self._rerank_cache is None:
            fingerprint = self._ensure_rerank_fingerprint()
            if self._rerank_cache_enabled and self._rerank_cache_repo is None:
                self._rerank_cache_repo = RerankCacheRepository(self._rerank_cache_db_path)
            self._rerank_cache = FingerprintedRerankCache(
                self._rerank_cache_repo,
                fingerprint,
                enabled=self._rerank_cache_enabled,
            )
        return self._rerank_cache

    def rerank_stats(self) -> Optional[dict]:
        """Run 層級 rerank 資訊（fingerprint + cache 設定）；未啟用過 rerank 時為 None。"""
        if self._rerank_fingerprint is None:
            return None
        cache = self._rerank_cache
        from evaluation.v11.retrieval.reranker import (
            RERANK_BATCH_SIZE,
            SCORED_ACTIVATION,
        )

        return {
            "stats_version": "v1",
            "rerank_model_name": self._rerank_model_name,
            "rerank_fingerprint": self._rerank_fingerprint.to_canonical_dict(),
            "rerank_fingerprint_hash": self._rerank_fingerprint.fingerprint_hash(),
            # 兩個 runtime 常數（非搜尋維度，不進 config hash / cache key）。
            "rerank_batch_size": RERANK_BATCH_SIZE,
            "scored_activation": SCORED_ACTIVATION,
            "rerank_cache_enabled": bool(cache.enabled) if cache else False,
            "rerank_cache_hits": cache.hits if cache else 0,
            "rerank_cache_misses": cache.misses if cache else 0,
            "cache_schema_version": self._rerank_fingerprint.cache_schema_version,
            # 誠實性註記隨產物一起走，避免脫離 runbook 後被誤讀。
            "latency_note": (
                "cached rerank latency is INFORMATIONAL ONLY. Feasibility must use the "
                "uncached end-to-end p95_retrieval_latency_ms, never a cache-warmed number."
            ),
        }

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

        # 跨文件 chunk batching：避免每份文件各呼叫一次遠端 Embedding API。
        embedding_batch_size = 512
        pending_chunks: list = []
        embedding_batch_count = 0
        embedded_chunk_count = 0
        embedding_stage_seconds = 0.0
        upsert_stage_seconds = 0.0

        def _flush_batch(batch: list, *, force_print: bool = False) -> None:
            nonlocal embedding_batch_count, embedded_chunk_count
            nonlocal embedding_stage_seconds, upsert_stage_seconds
            if not batch:
                return
            t0 = time.perf_counter()
            vectors = caching.embed_documents([c.text for c in batch])
            embedding_stage_seconds += time.perf_counter() - t0
            embedded_chunk_count += len(batch)
            t1 = time.perf_counter()
            store.upsert_chunks(batch, vectors)
            upsert_stage_seconds += time.perf_counter() - t1
            embedding_batch_count += 1
            # progress：第 1 batch / 之後每 10 batch / 最後一個 batch。
            if (
                embedding_batch_count == 1
                or embedding_batch_count % 10 == 0
                or force_print
            ):
                print(
                    f"[INDEX] batches={embedding_batch_count} chunks={embedded_chunk_count}",
                    flush=True,
                )

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
            total_chunks += len(outcome.chunks)
            fallback_count += outcome.fallback_count
            fallback_reasons.extend(outcome.fallback_reasons)
            context_added += outcome.context_added_count
            context_reduced += outcome.context_reduced_count
            # 將本文件 chunks 加入 pending buffer；滿 batch 就 flush。
            pending_chunks.extend(outcome.chunks)
            while len(pending_chunks) >= embedding_batch_size:
                batch = pending_chunks[:embedding_batch_size]
                pending_chunks = pending_chunks[embedding_batch_size:]
                _flush_batch(batch)

        # 所有 files 處理完成後，flush 最後不足 batch 的 pending_chunks。
        if pending_chunks:
            _flush_batch(pending_chunks, force_print=True)
            pending_chunks = []

        build_time_ms = int((time.perf_counter() - start) * 1000)
        point_count = self._client.count(collection_name).count

        print(
            f"[INDEX] DONE chunks={total_chunks} batches={embedding_batch_count} "
            f"embedding_s={embedding_stage_seconds:.3f} upsert_s={upsert_stage_seconds:.3f}",
            flush=True,
        )

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
        # Slice R2：只有真的啟用 rerank 才建構 reranker / cache（否則不載入模型）。
        rerank_on = bool(getattr(rc, "is_rerank_enabled", False))
        reranker = self._ensure_reranker() if rerank_on else None
        rerank_cache = self._ensure_rerank_cache() if rerank_on else None

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
                # 注入既有 strategy dispatch —— dense / identifier_aware 共用同一條路徑。
                reranker=reranker,
                rerank_cache=rerank_cache,
            ),
            backend,
        )
        # **這才是 end-to-end**：query embedding + vector search + admission + rerank
        # + final ranking 全部在內。rerank stage 的分項時間另由 meta 提供，兩者不可相加。
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
                # Slice R2：rerank 明細。未啟用時三者皆 None → 既有輸出 byte-identical。
                rerank_score=r.rerank_score if rerank_on else None,
                rerank_score_normalized=(
                    r.rerank_score_normalized if rerank_on else None
                ),
                rerank_applied=bool(r.rerank_applied) if rerank_on else None,
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
