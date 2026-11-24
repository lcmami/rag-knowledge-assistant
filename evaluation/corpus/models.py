"""corpus 資料模型（docs/external-corpus-api-plan.md §4.2、§5.2、§5.5、§6.6）。

- Slice A：:class:`LawArticle` / :class:`LawRecord` / :class:`NormalizedDocument`
  與 :class:`SkippedRecordDiagnostic`。
- Slice B：:class:`SnapshotManifest`（依計畫不得在 Slice A 提前建立）。

設計原則：``LawRecord`` 是 MOJ 來源欄位的**忠實對映**，不臆造來源沒有的欄位。
唯二的衍生欄位是 :attr:`LawRecord.pcode`（由 ``LawURL`` query parameter 解析）
與 :attr:`LawRecord.is_repealed`（由 ``LawAbandonNote`` 是否非空判定），兩者皆為
既有 pipeline 實際需要的識別與 metadata，非為未來功能預留。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict

# 來源以「行首空白數」編碼編章節巢狀深度，且為 3 的倍數（實測：0/3/6/9/12）。
CHAPTER_INDENT_UNIT = 3

ARTICLE_TYPE_ARTICLE = "A"  # 條文
ARTICLE_TYPE_CHAPTER = "C"  # 編章節標題


@dataclass(frozen=True)
class LawArticle:
    """``LawArticles[]`` 的單一元素。

    - ``article_type``：``"A"`` 條文、``"C"`` 編章節標題。
    - ``article_no``：type-``C`` 實測一律為空字串。
    - ``article_content``：type-``C`` 以行首空白編碼巢狀深度。
    """

    article_type: str
    article_no: str
    article_content: str

    @property
    def is_article(self) -> bool:
        return self.article_type == ARTICLE_TYPE_ARTICLE

    @property
    def is_chapter(self) -> bool:
        return self.article_type == ARTICLE_TYPE_CHAPTER

    @property
    def chapter_depth(self) -> int:
        """編章節巢狀深度（``第 一 章`` = 1、``第 一 節`` = 2）。

        非 type-``C`` 一律回傳 0。深度由行首空白數整除
        :data:`CHAPTER_INDENT_UNIT` 得出；不足一個單位者視為深度 0。
        """
        if not self.is_chapter:
            return 0
        stripped = self.article_content.lstrip(" ")
        return (len(self.article_content) - len(stripped)) // CHAPTER_INDENT_UNIT

    @property
    def chapter_title(self) -> str:
        """去除縮排後的編章節標題文字（保留來源原本的字間空白）。"""
        return self.article_content.strip()


@dataclass(frozen=True)
class LawRecord:
    """單一法規，忠實對映 MOJ ``Laws[]`` 的實際欄位。

    欄位名稱採 snake_case，但與來源 PascalCase 一一對應；不新增來源沒有的語意。
    """

    pcode: str  # 衍生：由 law_url 的 pcode query parameter 解析
    law_level: str
    law_name: str
    law_url: str
    law_category: str
    law_modified_date: str
    law_effective_date: str
    law_effective_note: str
    law_abandon_note: str
    law_has_eng_version: str
    eng_law_name: str
    law_histories: str
    law_foreword: str
    articles: tuple[LawArticle, ...] = ()
    # 來源 ``LawAttachements``（官方拼字即如此）。Slice A 明確忽略、不下載，
    # 但仍忠實保留於記憶體中，以免正規化階段誤以為來源不存在此欄位。
    law_attachements: tuple[dict[str, str], ...] = ()

    @property
    def is_repealed(self) -> bool:
        """是否已廢止（``LawAbandonNote`` 非空即為已廢止）。"""
        return bool(self.law_abandon_note.strip())

    @property
    def article_count(self) -> int:
        """type-``A`` 條文數（不含編章節標題）。"""
        return sum(1 for a in self.articles if a.is_article)


class SkippedRecordDiagnostic(TypedDict):
    """單筆被跳過的來源 record 的可機器讀取診斷。

    ``parse_raw`` 逐筆跳過 malformed record 是刻意的容錯，但跳過本身必須留下
    provenance：Slice B 的 manifest 需要 ``skipped_records`` 數量與 ``reason``
    分佈，才能分辨「來源本來就這樣」與「來源契約悄悄改變了」。

    - ``index``：於來源 ``Laws[]`` 中的位置（便於回頭比對原始資料）。
    - ``law_name``：無法判定時為 ``None``（例如整筆不是物件、或 ``LawName`` 為空）。
    - ``reason``：穩定的機器可讀代碼，目前為 ``record_not_object`` /
      ``missing_law_name`` / ``missing_pcode`` / ``missing_law_articles``。
    """

    index: int
    law_name: Optional[str]
    reason: str


@dataclass(frozen=True)
class NormalizedDocument:
    """正規化後、可直接寫入 corpus 目錄的單一文件。

    只包含目前 normalization 真正需要的欄位；不為 crawler / PDF / OCR / Vision
    預留任何抽象。
    """

    doc_id: str  # == pcode
    filename: str  # == f"{pcode}.md"
    markdown: str
    metadata: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Slice B：snapshot manifest
# ---------------------------------------------------------------------------

# manifest 自身的 schema 版本。**已發布**的欄位集合或語意改變時必須 bump；
# 尚未 commit、尚無任何既存 manifest 需要遷移時，維持初始版本（不製造假的 migration）。
MANIFEST_SCHEMA_VERSION = 1

# 計畫 §6.6 要求的 manifest 欄位。集中於此，讓 contract test 有單一斷言來源。
MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "snapshot_id",
    "provider",
    "source_endpoint",
    "source_update_date",
    "fetched_at",
    "raw_sha256",
    "raw_bytes",
    "selection",
    "selection_hash",
    "source_record_count",
    "selected_record_count",
    "normalized_document_count",
    "article_count",
    "skipped_article_count",
    "skipped_record_count",
    "skipped_record_reason_counts",
    "skipped_records",
    "corpus_hash",
    "git_state",
)


@dataclass(frozen=True)
class SnapshotManifest:
    """一份 snapshot 的完整 provenance（計畫 §6.6）。

    **明確不包含**：API key / Authorization、任何本機絕對路徑（含 ``output_root``）、
    來源文件全文。前者本 API 根本不需要；後兩者會讓 manifest 無法跨機器比對，
    也違反計畫的安全聲明。

    ``skipped_records`` 保留逐筆診斷明細（依 ``index`` 排序），而非只留計數：
    唯有原因分佈可機器讀取，才能分辨「來源本來就這樣」與「來源契約悄悄改變了」。

    **``skipped_record_count`` 與 ``skipped_article_count`` 是兩個不同層級的統計**：
    前者是**來源 record** 因 malformed 而未進入 pipeline 的筆數；後者是**已挑選的
    合法 record** 中，因 normalization 規則而未輸出的 type-``A`` 條文數。一部法規
    所有條文皆空時，它仍是 valid selected record（只是正規化後沒有 document），
    其條文必須計入 ``skipped_article_count``，**不得**被誤算為 malformed record。
    """

    schema_version: int
    snapshot_id: str
    provider: str
    source_endpoint: str
    source_update_date: str
    fetched_at: str
    raw_sha256: str
    raw_bytes: int
    selection: dict[str, object]
    selection_hash: str
    source_record_count: int
    selected_record_count: int
    normalized_document_count: int
    article_count: int
    skipped_article_count: int
    skipped_record_count: int
    skipped_record_reason_counts: dict[str, int]
    skipped_records: tuple[SkippedRecordDiagnostic, ...]
    corpus_hash: str
    git_state: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """轉為可 canonical 序列化的 dict。

        ``skipped_records`` 依 ``index`` 排序、每筆欄位順序固定，確保相同輸入
        必得相同位元組（key 排序由 ``canonical_json`` 負責）。
        """
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "provider": self.provider,
            "source_endpoint": self.source_endpoint,
            "source_update_date": self.source_update_date,
            "fetched_at": self.fetched_at,
            "raw_sha256": self.raw_sha256,
            "raw_bytes": self.raw_bytes,
            "selection": dict(self.selection),
            "selection_hash": self.selection_hash,
            "source_record_count": self.source_record_count,
            "selected_record_count": self.selected_record_count,
            "normalized_document_count": self.normalized_document_count,
            "article_count": self.article_count,
            "skipped_article_count": self.skipped_article_count,
            "skipped_record_count": self.skipped_record_count,
            "skipped_record_reason_counts": dict(self.skipped_record_reason_counts),
            "skipped_records": [
                {
                    "index": entry["index"],
                    "law_name": entry["law_name"],
                    "reason": entry["reason"],
                }
                for entry in sorted(self.skipped_records, key=lambda e: e["index"])
            ],
            "corpus_hash": self.corpus_hash,
            "git_state": dict(self.git_state),
        }
