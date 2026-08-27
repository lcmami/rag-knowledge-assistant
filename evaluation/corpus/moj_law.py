"""MOJ 全國法規資料庫 provider（docs/external-corpus-api-plan.md §5）。

這是目前**唯一**的 provider。依計畫 §4.2，刻意不建立 ``ExternalCorpusSource``
Protocol、provider registry 或 plugin framework — 只有一個實作時那些都屬過早抽象。

已驗證的來源契約（規劃期實際請求一次確認）：

- ``GET https://law.moj.gov.tw/api/Ch/Law/JSON`` — 無認證、**無 pagination**。
- 回應為 ZIP，成員含 ``ChLaw.json`` / ``schema.csv`` / ``manifest.csv``。
- ``ChLaw.json`` 為 **UTF-8 with BOM**，須以 ``utf-8-sig`` 解碼。
- 頂層結構 ``{"UpdateDate": str, "Laws": [...]}``。
- ``pcode`` 由 ``LawURL`` 的 query parameter 解析，實測全數唯一。
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from evaluation.corpus.models import LawArticle, LawRecord, SkippedRecordDiagnostic

DEFAULT_ENDPOINT = "https://law.moj.gov.tw/api/Ch/Law/JSON"
LAW_JSON_MEMBER = "ChLaw.json"

# 來源結構若真的改變，逐筆跳過會把問題掩蓋成「資料變少」。超過此比例即整體失敗。
DEFAULT_MAX_SKIP_RATIO = 0.05

_PCODE_RE = re.compile(r"^[A-Za-z0-9]+$")


class MojLawError(Exception):
    """MOJ provider 的錯誤基底。"""


class MojLawFetchError(MojLawError):
    """網路取得階段失敗（連線、逾時、長度不符）。"""


class MojLawParseError(MojLawError):
    """來源結構不符契約（非 ZIP、缺成員、頂層結構錯誤、pcode 重複）。"""


class MojLawSelectionError(MojLawError):
    """挑選結果為空等，無法產生有意義的 corpus。"""


@dataclass(frozen=True)
class SourceConfig:
    """Provider / fetch / 挑選設定（計畫 §7.1–§7.2）。

    Slice A 以 dataclass 預設值提供；Slice B 才會從 ``evaluation/configs/*.json``
    載入。此處**不含任何 secret** — 該 API 不需認證。
    """

    endpoint: str = DEFAULT_ENDPOINT
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    # 挑選
    max_laws: Optional[int] = 100
    category_prefix: Optional[str] = None
    include_repealed: bool = True
    order_by: str = "pcode"
    # 容錯
    max_skip_ratio: float = DEFAULT_MAX_SKIP_RATIO


# ---------------------------------------------------------------------------
# 1. fetch
# ---------------------------------------------------------------------------
def _httpx_fetcher(url: str, timeout: float) -> bytes:
    """預設 fetcher。延遲 import，讓 unit test 完全不必碰 httpx。"""
    import httpx

    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    declared = response.headers.get("Content-Length")
    body = response.content
    # partial fetch：宣告長度與實得位元組不符時視為失敗，交由外層重試。
    if declared is not None and declared.isdigit() and int(declared) != len(body):
        raise MojLawFetchError(
            f"回應長度不符：Content-Length={declared}、實得 {len(body)} bytes"
        )
    return body


def fetch_raw(
    config: SourceConfig,
    *,
    fetcher: Optional[Callable[[str, float], bytes]] = None,
    sleeper: Optional[Callable[[float], None]] = None,
) -> bytes:
    """取得原始 ZIP 位元組，失敗時以指數退避重試。

    ``fetcher`` / ``sleeper`` 皆可注入：unit test 一律注入 fake，**不碰網路、
    也不真的等待 backoff**（計畫 Clarification C3）。

    重試次數為 ``max_retries``，即最多嘗試 ``max_retries`` 次；退避序列為
    ``retry_backoff_seconds * 2**i``，最後一次失敗後不再 sleep。
    """
    do_fetch = fetcher or _httpx_fetcher
    do_sleep = sleeper if sleeper is not None else time.sleep

    attempts = max(1, config.max_retries)
    last_error: Optional[BaseException] = None
    for attempt in range(attempts):
        try:
            body = do_fetch(config.endpoint, config.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - 統一轉為 MojLawFetchError
            last_error = exc
            if attempt < attempts - 1:
                do_sleep(config.retry_backoff_seconds * (2**attempt))
            continue
        if not body:
            last_error = MojLawFetchError("回應為空")
            if attempt < attempts - 1:
                do_sleep(config.retry_backoff_seconds * (2**attempt))
            continue
        return body

    raise MojLawFetchError(
        f"取得來源失敗：{config.endpoint}（已嘗試 {attempts} 次）；最後錯誤：{last_error!r}"
    ) from last_error


# ---------------------------------------------------------------------------
# 2. parse
# ---------------------------------------------------------------------------
def extract_pcode(law_url: str) -> Optional[str]:
    """由 ``LawURL`` 的 query parameter 解析 pcode；無法解析回傳 ``None``。

    比對不分大小寫（來源同時出現 ``pcode`` 與 ``PCode``）。
    """
    if not law_url:
        return None
    try:
        query = parse_qs(urlparse(law_url).query)
    except ValueError:
        return None
    for key, values in query.items():
        if key.lower() != "pcode":
            continue
        for value in values:
            candidate = value.strip()
            if candidate and _PCODE_RE.match(candidate):
                return candidate
    return None


def _as_str(value: Any) -> str:
    """來源偶有 ``null``；統一成字串，避免下游到處 None 檢查。"""
    return "" if value is None else str(value)


def _parse_articles(raw_articles: Any) -> list[LawArticle]:
    articles: list[LawArticle] = []
    for raw in raw_articles:
        if not isinstance(raw, dict):
            continue
        articles.append(
            LawArticle(
                article_type=_as_str(raw.get("ArticleType")),
                article_no=_as_str(raw.get("ArticleNo")),
                article_content=_as_str(raw.get("ArticleContent")),
            )
        )
    return articles


def _parse_attachements(raw: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(raw, list):
        return ()
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                {
                    "FileName": _as_str(item.get("FileName")),
                    "FileURL": _as_str(item.get("FileURL")),
                }
            )
    return tuple(out)


def _parse_one(
    raw: Any, index: int
) -> tuple[Optional[LawRecord], Optional[SkippedRecordDiagnostic]]:
    """解析單筆；回傳 ``(record, diagnostic)``，兩者必有其一為 ``None``。"""

    def skip(reason: str, name: Optional[str] = None) -> tuple[None, SkippedRecordDiagnostic]:
        return None, SkippedRecordDiagnostic(index=index, law_name=name, reason=reason)

    if not isinstance(raw, dict):
        return skip("record_not_object")

    law_name = _as_str(raw.get("LawName")).strip()
    if not law_name:
        return skip("missing_law_name")

    law_url = _as_str(raw.get("LawURL"))
    pcode = extract_pcode(law_url)
    if not pcode:
        return skip("missing_pcode", law_name)

    raw_articles = raw.get("LawArticles")
    if not isinstance(raw_articles, list):
        return skip("missing_law_articles", law_name)

    return (
        LawRecord(
            pcode=pcode,
            law_level=_as_str(raw.get("LawLevel")),
            law_name=law_name,
            law_url=law_url,
            law_category=_as_str(raw.get("LawCategory")),
            law_modified_date=_as_str(raw.get("LawModifiedDate")),
            law_effective_date=_as_str(raw.get("LawEffectiveDate")),
            law_effective_note=_as_str(raw.get("LawEffectiveNote")),
            law_abandon_note=_as_str(raw.get("LawAbandonNote")),
            law_has_eng_version=_as_str(raw.get("LawHasEngVersion")),
            eng_law_name=_as_str(raw.get("EngLawName")),
            law_histories=_as_str(raw.get("LawHistories")),
            law_foreword=_as_str(raw.get("LawForeword")),
            articles=tuple(_parse_articles(raw_articles)),
            law_attachements=_parse_attachements(raw.get("LawAttachements")),
        ),
        None,
    )


def parse_raw(
    raw_zip: bytes,
    *,
    max_skip_ratio: float = DEFAULT_MAX_SKIP_RATIO,
) -> tuple[str, list[LawRecord], list[SkippedRecordDiagnostic]]:
    """``ZIP -> ChLaw.json -> utf-8-sig -> JSON -> (UpdateDate, records, skipped)``。

    結構性錯誤（非 ZIP、缺 ``ChLaw.json``、頂層結構不符、pcode 重複）一律
    立即 :class:`MojLawParseError`；單筆 malformed record 則跳過並累計於
    ``skipped``。跳過比例超過 ``max_skip_ratio`` 代表來源契約已改變，整體失敗。

    第三個回傳值是 :class:`SkippedRecordDiagnostic` 清單，而非僅一個計數：逐筆
    跳過是有損的，Slice B 的 manifest 必須能記錄跳過**數量**與**原因分佈**，
    這是 provenance 的一部分。
    """
    if not zipfile.is_zipfile(io.BytesIO(raw_zip)):
        head = raw_zip[:16]
        raise MojLawParseError(f"回應不是 ZIP；前導 bytes = {head!r}")

    try:
        archive = zipfile.ZipFile(io.BytesIO(raw_zip))
        members = archive.namelist()
        if LAW_JSON_MEMBER not in members:
            raise MojLawParseError(
                f"ZIP 缺少 {LAW_JSON_MEMBER}；實際成員 = {members}"
            )
        payload = archive.read(LAW_JSON_MEMBER)
    except MojLawParseError:
        raise
    except zipfile.BadZipFile as exc:
        raise MojLawParseError(f"ZIP 損毀：{exc}") from exc

    try:
        document = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MojLawParseError(f"{LAW_JSON_MEMBER} 無法解析為 JSON：{exc}") from exc

    if not isinstance(document, dict):
        raise MojLawParseError("頂層結構不是物件")
    if "UpdateDate" not in document:
        raise MojLawParseError("頂層缺少 UpdateDate")
    raw_laws = document.get("Laws")
    if not isinstance(raw_laws, list):
        raise MojLawParseError("頂層 Laws 缺少或不是 list")

    records: list[LawRecord] = []
    skipped: list[SkippedRecordDiagnostic] = []
    seen: dict[str, str] = {}
    for index, raw in enumerate(raw_laws):
        record, diagnostic = _parse_one(raw, index)
        if record is None:
            assert diagnostic is not None  # _parse_one 契約：兩者必有其一
            skipped.append(diagnostic)
            continue
        if record.pcode in seen:
            # 實測 1,346/1,346 唯一。重複代表來源契約變動，不可默默吞掉。
            raise MojLawParseError(
                f"pcode 重複：{record.pcode}"
                f"（已見於「{seen[record.pcode]}」、又見於「{record.law_name}」）"
            )
        seen[record.pcode] = record.law_name
        records.append(record)

    total = len(raw_laws)
    if total and len(skipped) / total > max_skip_ratio:
        raise MojLawParseError(
            f"跳過比例 {len(skipped)}/{total} 超過上限 {max_skip_ratio:.0%}；"
            f"來源結構可能已改變。前幾筆原因：{skipped[:5]}"
        )

    return _as_str(document["UpdateDate"]), records, skipped


# ---------------------------------------------------------------------------
# 3. select
# ---------------------------------------------------------------------------
def select(records: list[LawRecord], config: SourceConfig) -> list[LawRecord]:
    """決定性挑選：``filter -> pcode 升冪 -> max_laws``。

    輸入順序不影響輸出（排序在截斷之前），故同一份來源必得同一份子集。
    """
    if config.order_by != "pcode":
        raise MojLawSelectionError(f"不支援的 order_by：{config.order_by!r}（目前只支援 'pcode'）")

    filtered = [
        r
        for r in records
        if (config.include_repealed or not r.is_repealed)
        and (
            config.category_prefix is None
            or r.law_category.startswith(config.category_prefix)
        )
    ]
    filtered.sort(key=lambda r: r.pcode)

    if config.max_laws is not None:
        if config.max_laws <= 0:
            raise MojLawSelectionError(f"max_laws 必須為正整數或 None，得到 {config.max_laws}")
        filtered = filtered[: config.max_laws]

    if not filtered:
        raise MojLawSelectionError(
            "挑選結果為空（"
            f"來源 {len(records)} 筆、include_repealed={config.include_repealed}、"
            f"category_prefix={config.category_prefix!r}）"
        )
    return filtered


def with_overrides(config: SourceConfig, **overrides: Any) -> SourceConfig:
    """回傳套用覆寫後的新 config（frozen dataclass 的便利包裝）。"""
    return replace(config, **overrides)
