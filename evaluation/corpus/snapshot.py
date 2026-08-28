"""Snapshot 產生與 corpus fingerprint（docs/external-corpus-api-plan.md §6、Slice B）。

把「取得的來源 bytes」變成一份**不可變、可稽核、以 hash 版本化**的 snapshot 目錄：

    <output_root>/<snapshot_id>/
      manifest.json        受追蹤
      normalized/<pcode>.md 受追蹤（UTF-8 無 BOM、LF）
      raw/ChLaw.zip        gitignored
      raw/ChLaw.json       gitignored
      raw/schema.csv       gitignored（來源自帶欄位字典，稽核用）

三個不變式貫穿本模組：

1. **原子性** — 全部寫入先落在同層的 ``.<snapshot_id>.tmp/``，所有驗證（含
   ``corpus_hash`` 與 manifest）完成後才 ``os.replace`` 更名。任何一步失敗都不得
   留下正式 snapshot 目錄。tmp 與 final 同 parent，避免跨 filesystem rename。
2. **不可變** — 正式 snapshot 目錄一旦存在即為 immutable artifact；同名再建只會
   得到明確錯誤，既有內容完全不動（不覆寫、不 merge、不刪除重建）。
3. **可重現** — ``snapshot_id`` / ``selection_hash`` / ``corpus_hash`` 皆只由來源
   內容與挑選語意決定，與 ``output_root``、``fetched_at``、OS 路徑分隔符無關。

``corpus_hash`` **直接呼叫既有的** :func:`evaluation.v11.hashing.corpus_hash`
（計畫 §6.5）；本模組不自寫、不複製、不修改該演算法。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from evaluation.corpus.models import (
    MANIFEST_SCHEMA_VERSION,
    NormalizedDocument,
    SkippedRecordDiagnostic,
    SnapshotManifest,
)
from evaluation.corpus.moj_law import SourceConfig, fetch_raw, parse_raw, select
from evaluation.corpus.normalize import to_markdown
from evaluation.v11 import hashing

PROVIDER_MOJ_LAW = "moj_law"
SNAPSHOT_ID_PREFIX = "moj-chlaw"

MANIFEST_FILENAME = "manifest.json"
NORMALIZED_DIRNAME = "normalized"
RAW_DIRNAME = "raw"
RAW_ZIP_FILENAME = "ChLaw.zip"
# 來源 ZIP 內值得原樣保留以供稽核的成員；不存在就略過（不因此改寫 parser 架構）。
RAW_MEMBERS = ("ChLaw.json", "schema.csv")

# 只有真正影響「挑選出哪些法規、以什麼順序」的設定才進 selection_hash。
# timeout / retry / output_root / fetched_at 明確排除：它們不改變挑選結果。
SELECTION_FIELDS = ("max_laws", "category_prefix", "include_repealed", "order_by")

# config 的完整合法 key 集合；未知 key 一律拒絕（避免 typo 靜默退回預設值）。
_REQUIRED_CONFIG_KEYS = ("provider", "endpoint", "output_root")
_OPTIONAL_CONFIG_KEYS = (
    "timeout_seconds",
    "max_retries",
    "retry_backoff_seconds",
    "max_laws",
    "category_prefix",
    "include_repealed",
    "order_by",
)

# 來源 UpdateDate 實測形如 ``2026/8/21 上午 12:00:00``（月／日**不補零**），
# 亦相容 ``2026-08-21`` 與 ``20260821``。snapshot_id 需要固定寬度的 YYYYMMDD。
_UPDATE_DATE_RE = re.compile(r"^\s*(\d{4})\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})")

_TMP_NAME_RE = re.compile(r"^\.[A-Za-z0-9._-]+\.tmp$")


class SnapshotError(Exception):
    """Snapshot 產生失敗。"""


class SnapshotExistsError(SnapshotError):
    """目標 snapshot 已存在。正式 snapshot 為 immutable artifact，不得覆寫。"""


class CorpusConfigError(Exception):
    """Corpus config 檔內容不符契約。"""


# ---------------------------------------------------------------------------
# 1. config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusConfig:
    """一份 corpus pilot config（計畫 §7）。

    ``source`` 是既有的 :class:`~evaluation.corpus.moj_law.SourceConfig`；本類別
    只額外承載 provider 識別與 snapshot 輸出位置。**不含任何 secret** — 該 API
    不需認證，亦不得為此新增 ``.env`` 欄位。
    """

    provider: str
    source: SourceConfig
    output_root: str


def _require(raw: dict[str, Any], key: str, types: tuple[type, ...], *, allow_none: bool = False) -> Any:
    value = raw.get(key)
    if value is None and allow_none:
        return None
    if isinstance(value, bool) and bool not in types:
        # JSON 的 true/false 是 int 的子類別，不可誤當數值欄位。
        raise CorpusConfigError(f"config 欄位 {key!r} 型別錯誤：不接受 boolean")
    if not isinstance(value, types):
        names = "/".join(t.__name__ for t in types)
        raise CorpusConfigError(f"config 欄位 {key!r} 型別錯誤：期望 {names}，得到 {type(value).__name__}")
    return value


def parse_config(raw: Any) -> CorpusConfig:
    """驗證並轉換 config dict；不合契約一律 :class:`CorpusConfigError`。"""
    if not isinstance(raw, dict):
        raise CorpusConfigError(f"config 頂層必須是物件，得到 {type(raw).__name__}")

    allowed = set(_REQUIRED_CONFIG_KEYS) | set(_OPTIONAL_CONFIG_KEYS)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise CorpusConfigError(f"config 含未知欄位：{unknown}（合法欄位：{sorted(allowed)}）")
    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in raw]
    if missing:
        raise CorpusConfigError(f"config 缺少必要欄位：{missing}")

    provider = _require(raw, "provider", (str,))
    if provider != PROVIDER_MOJ_LAW:
        raise CorpusConfigError(f"不支援的 provider：{provider!r}（目前只支援 {PROVIDER_MOJ_LAW!r}）")

    endpoint = _require(raw, "endpoint", (str,))
    if not endpoint.strip():
        raise CorpusConfigError("config 欄位 'endpoint' 不得為空")

    output_root = _require(raw, "output_root", (str,))
    if not output_root.strip():
        raise CorpusConfigError("config 欄位 'output_root' 不得為空")
    if os.path.isabs(output_root):
        # 絕對路徑會讓 config 綁死單一機器；一律以 repo root 為基準的相對路徑。
        raise CorpusConfigError(f"config 欄位 'output_root' 必須為 repo 相對路徑，得到 {output_root!r}")

    defaults = SourceConfig()
    source = SourceConfig(
        endpoint=endpoint,
        timeout_seconds=float(
            _require(raw, "timeout_seconds", (int, float)) if "timeout_seconds" in raw else defaults.timeout_seconds
        ),
        max_retries=int(_require(raw, "max_retries", (int,)) if "max_retries" in raw else defaults.max_retries),
        retry_backoff_seconds=float(
            _require(raw, "retry_backoff_seconds", (int, float))
            if "retry_backoff_seconds" in raw
            else defaults.retry_backoff_seconds
        ),
        max_laws=(
            _require(raw, "max_laws", (int,), allow_none=True) if "max_laws" in raw else defaults.max_laws
        ),
        category_prefix=(
            _require(raw, "category_prefix", (str,), allow_none=True)
            if "category_prefix" in raw
            else defaults.category_prefix
        ),
        include_repealed=bool(
            _require(raw, "include_repealed", (bool,)) if "include_repealed" in raw else defaults.include_repealed
        ),
        order_by=str(_require(raw, "order_by", (str,)) if "order_by" in raw else defaults.order_by),
    )
    return CorpusConfig(provider=provider, source=source, output_root=output_root)


def load_config(path: str) -> CorpusConfig:
    """讀取並驗證 ``evaluation/configs/corpus-*.json``。"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError as exc:
        raise CorpusConfigError(f"找不到 config：{path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusConfigError(f"config 不是合法 JSON：{path}（{exc}）") from exc
    return parse_config(raw)


# ---------------------------------------------------------------------------
# 2. selection hash / snapshot id
# ---------------------------------------------------------------------------
def selection_payload(source: SourceConfig) -> dict[str, Any]:
    """挑選語意的完整表述 — 且**只有**挑選語意。"""
    return {field: getattr(source, field) for field in SELECTION_FIELDS}


def selection_hash(source: SourceConfig) -> str:
    """挑選設定的 canonical hash（沿用既有 ``canonical_json`` 慣例）。"""
    return hashlib.sha256(
        hashing.canonical_json(selection_payload(source)).encode("utf-8")
    ).hexdigest()


def compact_update_date(update_date: str) -> str:
    """來源 ``UpdateDate`` -> ``YYYYMMDD``。無法判讀即失敗，不臆測。"""
    match = _UPDATE_DATE_RE.match(update_date or "")
    if not match:
        raise SnapshotError(f"無法由 UpdateDate 取得日期：{update_date!r}")
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        raise SnapshotError(f"UpdateDate 日期不合理：{update_date!r}")
    return f"{year:04d}{month:02d}{day:02d}"


def build_snapshot_id(update_date: str, selection_hash_hex: str, raw_sha256: str) -> str:
    """``moj-chlaw-<YYYYMMDD>-<selection_hash[:8]>-<raw_sha256[:8]>``（計畫 §6.2）。

    **必須含 ``raw_sha256``**：``UpdateDate`` 不足以單獨保證來源 identity —— 來源
    bytes 改變但 ``UpdateDate`` 未同步更新時，不得產生相同的 snapshot_id。
    完整 64 字元 ``raw_sha256`` 仍記入 manifest。

    id 只由來源內容與挑選語意組成；不含 ``fetched_at``、絕對路徑、git 狀態、
    OS 路徑分隔符。
    """
    return (
        f"{SNAPSHOT_ID_PREFIX}-{compact_update_date(update_date)}"
        f"-{selection_hash_hex[:8]}-{raw_sha256[:8]}"
    )


def tmp_dirname(snapshot_id: str) -> str:
    """暫存目錄名稱。與 final 同 parent，避免跨 filesystem rename。"""
    return f".{snapshot_id}.tmp"


# ---------------------------------------------------------------------------
# 3. 寫入原語
# ---------------------------------------------------------------------------
def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)


def _write_text(path: str, text: str) -> None:
    """UTF-8 **無 BOM**、**LF only**。

    刻意走 binary + 明確 encode，而不是 ``open(..., "w")``：後者在 Windows 上會做
    newline translation，讓同一份內容在不同 OS 產生不同 bytes，進而汙染
    ``corpus_hash``。
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    _write_bytes(path, normalized.encode("utf-8"))


def _remove_own_tmp(tmp_dir: str) -> None:
    """只清理**符合自己精確命名慣例**的 tmp 目錄。

    刻意不做任何 wildcard 掃描：正式 snapshot 與他人的 corpus 目錄絕不能被
    這裡碰到。
    """
    if not os.path.exists(tmp_dir):
        return
    name = os.path.basename(os.path.normpath(tmp_dir))
    if not _TMP_NAME_RE.match(name):
        raise SnapshotError(f"拒絕清理不符 tmp 命名慣例的目錄：{name!r}")
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_raw_members(raw_zip: bytes, raw_dir: str) -> list[str]:
    """從來源 ZIP 抽出稽核用成員（best-effort，成員不存在就略過）。

    成員名稱取自固定白名單 :data:`RAW_MEMBERS`，不使用 ZIP 內的任意名稱，故無
    path traversal 風險。此處只是讀既有 archive，不改動 ``moj_law`` 的解析架構。
    """
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        names = set(archive.namelist())
        for member in RAW_MEMBERS:
            if member not in names:
                continue
            _write_bytes(os.path.join(raw_dir, member), archive.read(member))
            written.append(member)
    return written


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_root(output_root: str) -> str:
    return output_root if os.path.isabs(output_root) else os.path.join(hashing.repo_root(), output_root)


# ---------------------------------------------------------------------------
# 4. build_snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SnapshotResult:
    """``build_snapshot`` 的回傳值。

    ``path`` 是本機絕對路徑，供呼叫端使用；**不會**進入 manifest。
    """

    snapshot_id: str
    path: str
    manifest: SnapshotManifest


def build_snapshot(
    config: CorpusConfig,
    *,
    output_root: Optional[str] = None,
    raw_zip: Optional[bytes] = None,
    fetcher: Optional[Callable[[str, float], bytes]] = None,
    sleeper: Optional[Callable[[float], None]] = None,
    now: Callable[[], str] = _utc_now_iso,
    git_state_fn: Callable[[], dict[str, Any]] = hashing.git_state,
) -> SnapshotResult:
    """依計畫 §6 產生一份 immutable snapshot。

    步驟順序是契約的一部分：fetch -> raw_sha256 -> parse -> select -> normalize
    -> selection_hash -> snapshot_id -> tmp 目錄 -> 寫 raw/normalized ->
    ``hashing.corpus_hash`` -> manifest -> 全部成功才 ``os.replace``。

    ``raw_zip`` 可直接提供已取得的來源 bytes（CLI ``--raw-zip`` 與 offline 測試
    使用），語意等同於 ``fetch_raw`` 回傳該 bytes。``fetcher`` / ``sleeper`` /
    ``now`` / ``git_state_fn`` 皆可注入，讓 unit test 完全 offline 且決定性。

    失敗一律不留下正式 snapshot 目錄；自己的 tmp 目錄會被清除。
    """
    root = _resolve_root(output_root if output_root is not None else config.output_root)

    raw = raw_zip if raw_zip is not None else fetch_raw(config.source, fetcher=fetcher, sleeper=sleeper)
    fetched_at = now()
    raw_sha256 = hashlib.sha256(raw).hexdigest()

    update_date, records, skipped = parse_raw(raw, max_skip_ratio=config.source.max_skip_ratio)
    selected = select(records, config.source)

    documents: list[NormalizedDocument] = []
    skipped_article_count = 0
    for record in selected:
        document = to_markdown(record)
        # 已輸出的 type-A 條文數；正規化後完全沒有 document 時為 0。
        emitted = int(document.metadata.get("article_count", 0)) if document is not None else 0
        # 差額即「因 normalization 規則而未輸出」的條文數。刻意用「來源條文數 - 已輸出數」
        # 而非加總各 document 的 metadata：後者會漏掉**整部法規條文皆空**的情形
        # （to_markdown 回傳 None，但那些條文確實被跳過了）。同時避免在此重複一份
        # normalize.py 的 skip 判定邏輯。
        skipped_article_count += record.article_count - emitted
        if document is not None:
            documents.append(document)
    # 檔名排序，讓寫入順序與來源 Laws[] 原始順序完全無關。
    documents.sort(key=lambda d: d.filename)
    if not documents:
        raise SnapshotError(
            f"正規化後無任何文件（挑選 {len(selected)} 筆，全部無可用條文）；不產生空 snapshot"
        )

    sel_hash = selection_hash(config.source)
    snapshot_id = build_snapshot_id(update_date, sel_hash, raw_sha256)

    final_dir = os.path.join(root, snapshot_id)
    tmp_dir = os.path.join(root, tmp_dirname(snapshot_id))
    if os.path.exists(final_dir):
        raise SnapshotExistsError(
            f"snapshot 已存在，不覆寫：{snapshot_id}"
            "（正式 snapshot 為 immutable artifact；如需重建請先自行移除或改用其他 output_root）"
        )

    os.makedirs(root, exist_ok=True)
    _remove_own_tmp(tmp_dir)
    try:
        raw_dir = os.path.join(tmp_dir, RAW_DIRNAME)
        normalized_dir = os.path.join(tmp_dir, NORMALIZED_DIRNAME)
        os.makedirs(raw_dir)
        os.makedirs(normalized_dir)

        _write_bytes(os.path.join(raw_dir, RAW_ZIP_FILENAME), raw)
        _write_raw_members(raw, raw_dir)

        for document in documents:
            _write_text(os.path.join(normalized_dir, document.filename), document.markdown)

        # base_dir 指向 snapshot 目錄本身 -> 相對路徑恆為 "normalized/<pcode>.md"，
        # 與 output_root 位於何處無關。這是「兩個 output root 得到相同 corpus_hash」
        # 的關鍵。演算法本身完全來自既有 evaluation/v11/hashing.py。
        corpus_hash = hashing.corpus_hash([normalized_dir], base_dir=tmp_dir)

        manifest = _build_manifest(
            config=config,
            snapshot_id=snapshot_id,
            update_date=update_date,
            fetched_at=fetched_at,
            raw=raw,
            raw_sha256=raw_sha256,
            selection_hash_hex=sel_hash,
            records=records,
            skipped=skipped,
            selected_count=len(selected),
            documents=documents,
            skipped_article_count=skipped_article_count,
            corpus_hash=corpus_hash,
            git_state=git_state_fn(),
        )
        _write_text(
            os.path.join(tmp_dir, MANIFEST_FILENAME),
            hashing.canonical_json(manifest.to_dict()) + "\n",
        )

        if os.path.exists(final_dir):
            # 兩個 build 同時進行時的最後一道防線：既有 snapshot 完全不動。
            raise SnapshotExistsError(f"snapshot 已存在，不覆寫：{snapshot_id}")
        os.replace(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return SnapshotResult(snapshot_id=snapshot_id, path=final_dir, manifest=manifest)


def _build_manifest(
    *,
    config: CorpusConfig,
    snapshot_id: str,
    update_date: str,
    fetched_at: str,
    raw: bytes,
    raw_sha256: str,
    selection_hash_hex: str,
    records: list,
    skipped: list[SkippedRecordDiagnostic],
    selected_count: int,
    documents: list[NormalizedDocument],
    skipped_article_count: int,
    corpus_hash: str,
    git_state: dict[str, Any],
) -> SnapshotManifest:
    reason_counts: dict[str, int] = {}
    for entry in skipped:
        reason_counts[entry["reason"]] = reason_counts.get(entry["reason"], 0) + 1

    return SnapshotManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        snapshot_id=snapshot_id,
        provider=config.provider,
        source_endpoint=config.source.endpoint,
        source_update_date=update_date,
        fetched_at=fetched_at,
        raw_sha256=raw_sha256,
        raw_bytes=len(raw),
        selection=selection_payload(config.source),
        selection_hash=selection_hash_hex,
        # 來源 Laws[] 的總筆數：每筆不是成為 record 就是成為 skipped diagnostic。
        source_record_count=len(records) + len(skipped),
        selected_record_count=selected_count,
        normalized_document_count=len(documents),
        article_count=sum(int(d.metadata.get("article_count", 0)) for d in documents),
        skipped_article_count=skipped_article_count,
        skipped_record_count=len(skipped),
        skipped_record_reason_counts=dict(sorted(reason_counts.items())),
        skipped_records=tuple(sorted(skipped, key=lambda e: e["index"])),
        corpus_hash=corpus_hash,
        git_state=git_state,
    )
