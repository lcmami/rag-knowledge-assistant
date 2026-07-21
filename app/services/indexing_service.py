"""Indexing Service（規格 §14 / §15，本切片 §二、§三、§五）。

負責文件生命週期的商業邏輯，讓 API Router 保持精簡：

- 上傳匯入：驗證 → SHA-256 去重 → 建立 registry(indexing) → 存檔 → Loader → Splitter
  → Embedding → 寫入 Qdrant → status=indexed。
- 匯入失敗回滾：刪除該 document_id 已寫入 Qdrant points、保留 SQLite 記錄、status=failed、
  寫入 error_message；單一文件失敗不得使服務停止。
- 刪除：先刪 Qdrant points → 再刪本地檔案 → 最後刪 SQLite 記錄；Qdrant 刪除失敗不得先刪 SQLite。

本切片僅支援 Markdown 與 TXT；PDF 留待下一階段。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid

from app.core.config import Settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from app.core.logging import get_logger, mask_hash
from app.repositories.document_repository import DocumentRecord, DocumentRepository
from app.services.canonical_builder import build_canonical_document
from app.services.document_loader import load_document
from app.services.embedding_service import EmbeddingService
from app.services.text_splitter import TextSplitter
from app.services.vector_store import VectorStore

# 副檔名 → canonical document_type（offset scope 由此決定：pdf=page、其餘=document）。
_DOCUMENT_TYPES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "txt",
    ".pdf": "pdf",
}

logger = get_logger("app.indexing")

# 支援的副檔名（Phase 4 起加入 PDF）
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}

_CONTENT_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}


def _ext(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return ext


class IndexingService:
    def __init__(
        self,
        settings: Settings,
        repository: DocumentRepository,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.settings = settings
        self.repository = repository
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.splitter = TextSplitter(
            chunk_size_chars=settings.chunk_size_chars,
            chunk_overlap_chars=settings.chunk_overlap_chars,
            max_tokens=settings.embedding_max_tokens,
            count_tokens=embedding_service.count_tokens,
        )

    # --- 匯入 ---
    def index_document(self, filename: str, raw_bytes: bytes) -> DocumentRecord:
        # 1. 驗證副檔名
        ext = _ext(filename)
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                "不支援的檔案格式，僅支援 .md / .markdown / .txt / .pdf。",
                details={"filename": filename},
            )

        # 2. 驗證檔案不是空白
        if not raw_bytes or not raw_bytes.strip():
            raise EmptyDocumentError("上傳的檔案為空白。")

        # 3. 計算原始 bytes 的 SHA-256
        file_hash = hashlib.sha256(raw_bytes).hexdigest()

        # 4~5. 檢查重複，重複回 409 並附既有 document_id
        existing = self.repository.get_by_hash(file_hash)
        if existing is not None:
            raise DuplicateDocumentError(
                "此文件已經上傳。",
                details={"document_id": existing.id},
            )

        # 6. 建立 registry 記錄，status=indexing
        document_id = str(uuid.uuid4())
        storage_path = os.path.join(
            self.settings.upload_dir, document_id, os.path.basename(filename)
        )
        record = self.repository.create(
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            file_type=_CONTENT_TYPES.get(ext, "application/octet-stream"),
            file_size=len(raw_bytes),
            status="indexing",
            storage_path=storage_path,
        )
        logger.info(
            "開始 indexing document_id=%s filename=%s hash=%s size=%d",
            document_id,
            filename,
            mask_hash(file_hash),
            len(raw_bytes),
        )

        try:
            # 7. 儲存原始檔案至 storage/uploads/{document_id}/
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, "wb") as f:
                f.write(raw_bytes)

            # 8. 使用現有 Loader 解析
            elements = load_document(filename, raw_bytes)

            # 8b. 建立版本化 Canonical Document（v1.1 §五），供 Chunk provenance 對映。
            #     切分邏輯不變；canonical 僅提供附加的可追蹤 offset / 版本 metadata。
            canonical = build_canonical_document(
                filename=filename,
                document_sha256=file_hash,
                document_type=_DOCUMENT_TYPES.get(ext, "txt"),
                elements=elements,
            )

            # 9. 使用現有 Splitter 切 Chunk（附上 provenance / identifiers / 版本）
            chunks = self.splitter.split(
                elements,
                document_id=document_id,
                filename=filename,
                file_hash=file_hash,
                canonical=canonical,
            )
            if not chunks:
                raise EmptyDocumentError("文件解析後沒有可索引的內容。")

            # 10. 使用現有 Embedding Service 產生向量
            vectors = self.embedding_service.embed_documents([c.text for c in chunks])

            # 11. 寫入真實 Qdrant Server
            self.vector_store.upsert_chunks(chunks, vectors)

            # 12. 成功後更新 status=indexed、chunk_count
            self.repository.mark_indexed(document_id, len(chunks))
            logger.info(
                "indexing 完成 document_id=%s chunk_count=%d", document_id, len(chunks)
            )
            return self.repository.get(document_id)  # type: ignore[return-value]

        except Exception as exc:  # noqa: BLE001 - 任一步失敗都要回滾
            # 三、匯入失敗回滾
            logger.warning(
                "indexing 失敗，執行回滾 document_id=%s error=%s", document_id, exc
            )
            try:
                self.vector_store.delete_by_document_id(document_id)
            except Exception as rollback_exc:  # noqa: BLE001
                logger.error(
                    "回滾刪除 Qdrant points 失敗 document_id=%s error=%s",
                    document_id,
                    rollback_exc,
                )
            self.repository.mark_failed(document_id, str(exc)[:500])
            raise

    # --- 刪除（本切片 §五）---
    def delete_document(self, document_id: str) -> None:
        # 1. 確認文件存在
        record = self.repository.get(document_id)
        if record is None:
            raise DocumentNotFoundError(details={"document_id": document_id})

        # 2. 先刪除 Qdrant 中該 document_id 的所有 points
        #    Qdrant 刪除失敗 -> 不得先刪 SQLite，回 503
        try:
            self.vector_store.delete_by_document_id(document_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "刪除 Qdrant points 失敗，保留 SQLite 記錄 document_id=%s error=%s",
                document_id,
                exc,
            )
            raise VectorStoreError("刪除向量資料失敗，請稍後重試。") from exc

        # 3. Qdrant 刪除成功後，再刪除本地檔案
        if record.storage_path:
            doc_dir = os.path.dirname(record.storage_path)
            if os.path.isdir(doc_dir):
                shutil.rmtree(doc_dir, ignore_errors=True)

        # 4. 最後刪除 SQLite 記錄
        self.repository.delete(document_id)
        logger.info("已刪除 document_id=%s", document_id)
