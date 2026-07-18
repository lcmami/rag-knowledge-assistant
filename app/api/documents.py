"""文件 API（規格 §19.3~§19.5，本切片 §二、§四、§五）。

Router 保持精簡：商業邏輯委由 IndexingService 與 DocumentRepository。
同步端點於 FastAPI threadpool 執行，讀取上傳檔案使用 file.file.read()。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from app.dependencies import get_document_repository, get_indexing_service
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.indexing_service import IndexingService

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=DocumentUploadResponse,
)
def upload_document(
    file: UploadFile = File(...),
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> DocumentUploadResponse:
    """上傳並索引文件（§19.3）。成功回 201。"""
    raw_bytes = file.file.read()
    record = indexing_service.index_document(file.filename or "", raw_bytes)
    return DocumentUploadResponse.from_record(record)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    repository: DocumentRepository = Depends(get_document_repository),
) -> DocumentListResponse:
    """列出所有文件（§19.4）。"""
    records = repository.list_all()
    items = [DocumentListItem.from_record(r) for r in records]
    return DocumentListResponse(documents=items, total=len(items))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> Response:
    """刪除文件與其全部向量（§19.5）。成功回 204，不存在回 404。"""
    indexing_service.delete_document(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
