"""Application exceptions（規格 §22）。

每個例外帶有機器可讀 code 與對應 HTTP status，供 API 統一轉譯。
"""
from __future__ import annotations


class AppError(Exception):
    """所有應用程式例外的基底類別。"""

    code: str = "APP_ERROR"
    http_status: int = 500
    message: str = "應用程式錯誤。"

    def __init__(self, message: str | None = None, details: dict | None = None):
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)


class UnsupportedFileTypeError(AppError):
    code = "UNSUPPORTED_FILE_TYPE"
    http_status = 415
    message = "不支援的檔案格式。"


class FileTooLargeError(AppError):
    code = "FILE_TOO_LARGE"
    http_status = 413
    message = "檔案超過大小限制。"


class EmptyDocumentError(AppError):
    code = "EMPTY_DOCUMENT"
    http_status = 400
    message = "文件解析後沒有有效文字內容。"


class DuplicateDocumentError(AppError):
    code = "DUPLICATE_DOCUMENT"
    http_status = 409
    message = "此文件已經上傳。"


class DocumentParseError(AppError):
    code = "DOCUMENT_PARSE_ERROR"
    http_status = 400
    message = "文件解析失敗。"


class CorruptPdfError(AppError):
    """PDF 檔案損壞、無法開啟或格式錯誤（§2.4）。"""

    code = "CORRUPT_PDF"
    http_status = 400
    message = "PDF 檔案損壞或無法解析。"


class EncryptedPdfError(AppError):
    """加密 / 需密碼的 PDF，本階段不支援（§2.4）。"""

    code = "ENCRYPTED_PDF"
    http_status = 400
    message = "不支援加密或需要密碼的 PDF。"


class NoExtractableTextError(AppError):
    """PDF 可解析但沒有任何可擷取文字（例如純掃描圖片型，§2.4）。"""

    code = "NO_EXTRACTABLE_TEXT"
    http_status = 400
    message = "PDF 沒有可擷取的文字內容（可能為掃描圖片型，本階段不支援 OCR）。"


class EmbeddingError(AppError):
    code = "EMBEDDING_ERROR"
    http_status = 500
    message = "Embedding 產生失敗。"


class VectorStoreError(AppError):
    code = "VECTOR_STORE_ERROR"
    http_status = 503
    message = "向量資料庫不可用。"


class CollectionMismatchError(AppError):
    code = "COLLECTION_MISMATCH"
    http_status = 503
    message = "Qdrant Collection 與目前 Embedding 模型不相容。"


class LLMNotConfiguredError(AppError):
    """LLM 必要設定（base_url / api_key / model）缺少（§3.3）。"""

    code = "LLM_NOT_CONFIGURED"
    http_status = 503
    message = "LLM 服務尚未完成設定。"


class LLMTimeoutError(AppError):
    """LLM 呼叫逾時（§3.4）。不外洩 provider 內部資訊。"""

    code = "LLM_TIMEOUT"
    http_status = 504
    message = "LLM 服務回應逾時。"


class LLMServiceError(AppError):
    """LLM 連線失敗或 Provider 回傳錯誤（§3.4）。"""

    code = "LLM_SERVICE_ERROR"
    http_status = 502
    message = "LLM 服務暫時無法使用。"


class LLMEmptyResponseError(AppError):
    """LLM 回傳空 choices 或空內容（§3.4）。"""

    code = "LLM_EMPTY_RESPONSE"
    http_status = 502
    message = "LLM 回傳空的或無效的回應。"


class DocumentNotFoundError(AppError):
    code = "DOCUMENT_NOT_FOUND"
    http_status = 404
    message = "找不到指定文件。"
