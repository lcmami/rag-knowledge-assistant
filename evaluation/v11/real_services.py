"""RealTrialServiceProvider（Slice 2 §九）。

以正式 app 服務（EmbeddingService / IndexingService / RetrievalService / VectorStore）
實作 :class:`~evaluation.v11.trial_runner.TrialServiceProvider`，供 CLI / 整合測試使用。

隔離保證：
- 每 Trial 使用獨立 collection（``optimizer_`` 前綴）、獨立暫存 SQLite / Upload 目錄。
- **永不**觸碰正式 ``factory_knowledge``、正式 ``metadata.db``、正式 uploads。
- 刪除 collection 前一律驗證 ``optimizer_`` 前綴（防誤刪）。
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from evaluation.v11.retrieval_metrics import IndexChunk, RankedResult
from evaluation.v11.run_models import TrialConfig
from evaluation.v11.trial_collections import assert_deletable
from evaluation.v11.trial_runner import IndexOutcome, RetrievalOutcome


@dataclass
class _TrialContext:
    store: object
    retrieval: object
    workdir: str


class RealTrialServiceProvider:
    """真實服務 provider；embedding 與 qdrant client 共用，per-trial 建 collection-scoped 服務。"""

    def __init__(self, client, embedding, base_workdir: str, base_settings):
        self._client = client
        self._embedding = embedding
        self._base_workdir = base_workdir
        self._base_settings = base_settings
        self._contexts: dict[str, _TrialContext] = {}
        os.makedirs(base_workdir, exist_ok=True)

    @classmethod
    def create(
        cls,
        qdrant_url: str,
        base_workdir: str,
        *,
        base_settings=None,
        embedding=None,
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
        return cls(client, embedding, base_workdir, settings)

    # --- provider API ---
    def build_index(
        self, collection_name: str, trial_config: TrialConfig, files: list[tuple[str, str]]
    ) -> IndexOutcome:
        from app.core.config import Settings
        from app.repositories.document_repository import DocumentRepository
        from app.services.indexing_service import IndexingService
        from app.services.vector_store import VectorStore

        assert_deletable(collection_name)  # 確保是 optimizer_ 前綴（防誤用正式名稱）

        base = self._base_settings
        chunk = trial_config.chunk_config or {}
        workdir = os.path.join(self._base_workdir, f"trial_{uuid.uuid4().hex[:8]}")
        os.makedirs(workdir, exist_ok=True)
        settings = Settings(
            app_env="optimizer",
            qdrant_url=base.qdrant_url,
            qdrant_collection=collection_name,
            sqlite_path=os.path.join(workdir, "trial_metadata.db"),
            upload_dir=os.path.join(workdir, "uploads"),
            chunk_size_chars=int(chunk.get("chunk_size_chars", base.chunk_size_chars)),
            chunk_overlap_chars=int(chunk.get("chunk_overlap_chars", base.chunk_overlap_chars)),
            embedding_max_tokens=base.embedding_max_tokens,
            retrieval_score_threshold=trial_config.score_threshold,
            retrieval_max_top_k=max(trial_config.top_k, 10),
        )

        store = VectorStore(self._client, collection_name)
        # 隔離：確保 collection 為全新（存在則先刪，只刪 optimizer_ 前綴）
        try:
            if self._client.collection_exists(collection_name):
                assert_deletable(collection_name)
                self._client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001
            pass

        start = time.perf_counter()
        store.ensure_collection(self._embedding.dimension)
        repo = DocumentRepository(settings.sqlite_path)
        indexing = IndexingService(settings, repo, self._embedding, store)
        for name, abs_path in files:
            with open(abs_path, "rb") as f:
                raw = f.read()
            indexing.index_document(name, raw)
        build_time_ms = int((time.perf_counter() - start) * 1000)

        point_count = self._client.count(collection_name).count

        from app.services.retrieval_service import RetrievalService

        retrieval = RetrievalService(
            self._embedding,
            store,
            score_threshold=trial_config.score_threshold,
            candidate_multiplier=trial_config.candidate_multiplier,
            candidate_cap=trial_config.candidate_cap,
        )
        self._contexts[collection_name] = _TrialContext(store, retrieval, workdir)
        return IndexOutcome(point_count=point_count, build_time_ms=build_time_ms)

    def index_manifest(self, collection_name: str) -> list[IndexChunk]:
        """單次 scroll 該 Trial 自己的 collection，取回全部 Chunk 作為 Index Universe。

        僅該 Trial 的 collection（``optimizer_`` 前綴），不觸碰正式 ``factory_knowledge``；
        不做向量檢索（scroll 為列舉），也不為每題重複 scroll。
        """
        assert_deletable(collection_name)  # 前綴保護（順帶確保非正式 collection）
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
        ctx = self._contexts.get(collection_name)
        if ctx is None:
            raise RuntimeError(f"collection {collection_name!r} 尚未建索引。")
        result = ctx.retrieval.retrieve(question, top_k)
        ranked = [
            RankedResult(
                rank=rank,
                document_id=c.document_id,
                document_sha256=c.file_hash,
                filename=c.filename,
                section=c.section,
                page=c.page,
                chunk_index=c.chunk_index,
                score=c.score,
                text=c.text,
                provenance_spans=list(c.provenance_spans or []),
                identifiers=list(c.identifiers or []),
                canonical_text_version=c.canonical_text_version,
            )
            for rank, c in enumerate(result.results, start=1)
        ]
        return RetrievalOutcome(results=ranked, retrieval_time_ms=result.retrieval_time_ms)

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
