"""Snapshot + corpus fingerprint 測試（Slice B）。

全部 offline：來源以 Slice A 的 fixture + 記憶體 ZIP 取代，網路以 injected fake
fetcher 取代，``fetched_at`` / ``git_state`` 以注入的決定性 stub 取代。
**不碰**網路、模型、Qdrant、embedding、retrieval、LLM。

本檔鎖住三組不變式：

1. **Identity** — ``snapshot_id`` / ``selection_hash`` / ``corpus_hash`` 只由來源
   內容與挑選語意決定；``output_root`` / ``fetched_at`` / 來源陣列順序皆不得影響。
2. **Atomicity + immutability** — 任一步驟失敗不得留下正式 snapshot；既有 snapshot
   不得被覆寫、merge 或刪除重建。
3. **Manifest provenance** — §6.6 欄位齊備、skipped record 有可機器讀取的統計，
   且不含 secret 或本機絕對路徑。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest

from evaluation.corpus import snapshot as snapshot_module
from evaluation.corpus.models import (
    MANIFEST_REQUIRED_FIELDS,
    MANIFEST_SCHEMA_VERSION,
    SnapshotManifest,
)
from evaluation.corpus.moj_law import MojLawSelectionError, SourceConfig, parse_raw, select
from evaluation.corpus.normalize import to_markdown
from evaluation.corpus.snapshot import (
    CorpusConfig,
    CorpusConfigError,
    MANIFEST_FILENAME,
    NORMALIZED_DIRNAME,
    PROVIDER_MOJ_LAW,
    RAW_DIRNAME,
    RAW_ZIP_FILENAME,
    SnapshotError,
    SnapshotExistsError,
    build_snapshot,
    build_snapshot_id,
    compact_update_date,
    load_config,
    parse_config,
    selection_hash,
    selection_payload,
    tmp_dirname,
)
from evaluation.v11 import hashing
from tests.unit.test_corpus_moj_law import (
    ALL_PCODES_SORTED,
    PCODE_UNIFORM,
    RecordingFetcher,
    RecordingSleeper,
    build_zip,
    fixture_zip,
    load_fixture_document,
)

REPO_ROOT = hashing.repo_root()
PILOT_CONFIG_PATH = os.path.join(REPO_ROOT, "evaluation", "configs", "corpus-moj-law-pilot.json")
GITIGNORE_PATH = os.path.join(REPO_ROOT, ".gitignore")

FIXED_FETCHED_AT = "2026-01-02T03:04:05Z"
FIXED_GIT_STATE = {"commit": "0" * 40, "branch": "test-branch", "working_tree_dirty": False}

# ``build_zip`` 會把**當下時間**寫進 ZIP 的 local header，因此每次呼叫的 bytes 都不同。
# raw_sha256 對 bytes 敏感（這正是 §6.2 要的），故測試必須固定一份來源 bytes；
# 真實來源是同一次下載的同一份 ZIP，本常數即代表那份 bytes。
FIXTURE_ZIP = fixture_zip()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_config(**source_overrides) -> CorpusConfig:
    """測試用 config；``output_root`` 一律由呼叫端以 tmp_path 覆寫。"""
    defaults = dict(max_laws=None, category_prefix=None, include_repealed=True, order_by="pcode")
    defaults.update(source_overrides)
    return CorpusConfig(
        provider=PROVIDER_MOJ_LAW,
        source=SourceConfig(**defaults),
        output_root="evaluation/corpora/moj-law",
    )


def build(tmp_root, *, config=None, raw=None, **kwargs):
    """以決定性 stub 建立 snapshot（無網路、無真實時間、無真實 git）。"""
    kwargs.setdefault("now", lambda: FIXED_FETCHED_AT)
    kwargs.setdefault("git_state_fn", lambda: dict(FIXED_GIT_STATE))
    return build_snapshot(
        config or make_config(),
        output_root=str(tmp_root),
        raw_zip=FIXTURE_ZIP if raw is None else raw,
        **kwargs,
    )


def read_manifest_bytes(path: str) -> bytes:
    with open(os.path.join(path, MANIFEST_FILENAME), "rb") as handle:
        return handle.read()


def read_manifest(path: str) -> dict:
    return json.loads(read_manifest_bytes(path).decode("utf-8"))


def altered_raw_bytes() -> bytes:
    """相同 UpdateDate、相同 selection，但**不同 raw bytes**。

    只改動不影響挑選與正規化輸出的欄位（``LawHasEngVersion``），故
    ``selection_hash`` 與 ``corpus_hash`` 不變，唯有 ``raw_sha256`` 改變。
    """
    document = load_fixture_document()
    for law in document["Laws"]:
        law["LawHasEngVersion"] = "Y" if law.get("LawHasEngVersion") != "Y" else "N"
    return build_zip(document)


# ---------------------------------------------------------------------------
# 1. SnapshotManifest contract
# ---------------------------------------------------------------------------
def test_manifest_contract_has_all_required_fields(tmp_path):
    result = build(tmp_path)
    payload = result.manifest.to_dict()

    assert set(payload) == set(MANIFEST_REQUIRED_FIELDS)
    assert payload["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["provider"] == PROVIDER_MOJ_LAW
    # 寫出的 manifest.json 必須與 to_dict() 完全一致（canonical 序列化）
    assert read_manifest(result.path) == payload


def test_manifest_dataclass_is_frozen_and_serializes_deterministically():
    manifest = SnapshotManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        snapshot_id="moj-chlaw-20260821-aaaaaaaa-bbbbbbbb",
        provider=PROVIDER_MOJ_LAW,
        source_endpoint="https://example.invalid/api",
        source_update_date="2026/8/21",
        fetched_at=FIXED_FETCHED_AT,
        raw_sha256="a" * 64,
        raw_bytes=10,
        selection={"max_laws": 1, "category_prefix": None, "include_repealed": True, "order_by": "pcode"},
        selection_hash="b" * 64,
        source_record_count=3,
        selected_record_count=1,
        normalized_document_count=1,
        article_count=5,
        skipped_article_count=3,
        skipped_record_count=2,
        skipped_record_reason_counts={"missing_pcode": 2},
        # 刻意亂序輸入，to_dict() 必須依 index 排序
        skipped_records=(
            {"index": 7, "law_name": "乙法", "reason": "missing_pcode"},
            {"index": 2, "law_name": None, "reason": "missing_pcode"},
        ),
        corpus_hash="c" * 64,
        git_state=dict(FIXED_GIT_STATE),
    )

    with pytest.raises(Exception):
        manifest.snapshot_id = "mutated"  # type: ignore[misc]

    assert [e["index"] for e in manifest.to_dict()["skipped_records"]] == [2, 7]


# ---------------------------------------------------------------------------
# 2/3. snapshot_id 決定性與對 raw bytes 的敏感度
# ---------------------------------------------------------------------------
def test_snapshot_id_format_and_components():
    snapshot_id = build_snapshot_id("2026/8/21 上午 12:00:00", "s" * 64, "r" * 64)
    assert snapshot_id == f"moj-chlaw-20260821-{'s' * 8}-{'r' * 8}"


@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ("2026/8/21 上午 12:00:00", "20260821"),
        ("2026/12/5", "20261205"),
        ("2026-08-21", "20260821"),
        ("20260821", "20260821"),
    ],
)
def test_compact_update_date_handles_source_formats(raw_value, expected):
    """來源實測為 ``YYYY/M/D``（**月／日不補零**），snapshot_id 需固定寬度。"""
    assert compact_update_date(raw_value) == expected


@pytest.mark.parametrize("raw_value", ["", "not-a-date", "2026/13/01", "2026/1/99"])
def test_compact_update_date_rejects_unparseable(raw_value):
    with pytest.raises(SnapshotError):
        compact_update_date(raw_value)


def test_snapshot_id_is_stable_for_identical_inputs(tmp_path):
    first = build(tmp_path / "a")
    second = build(tmp_path / "b")
    assert first.snapshot_id == second.snapshot_id


def test_different_raw_bytes_yield_different_snapshot_id(tmp_path):
    """相同 UpdateDate + 相同 selection + 不同 raw bytes -> 不同 snapshot_id。

    這正是 §6.2 要求 id 必含 ``raw_sha256[:8]`` 的理由：來源 bytes 改變但
    ``UpdateDate`` 未同步更新時，**不得**被誤認為同一份 snapshot。
    """
    baseline = build(tmp_path / "a")
    altered = build(tmp_path / "b", raw=altered_raw_bytes())

    assert altered.manifest.raw_sha256 != baseline.manifest.raw_sha256
    assert altered.snapshot_id != baseline.snapshot_id
    # 對照組：UpdateDate 與 selection 完全相同，差異只可能來自 raw_sha256 片段
    assert altered.manifest.source_update_date == baseline.manifest.source_update_date
    assert altered.manifest.selection_hash == baseline.manifest.selection_hash
    assert altered.snapshot_id.rsplit("-", 1)[0] == baseline.snapshot_id.rsplit("-", 1)[0]


def test_snapshot_id_excludes_volatile_provenance(tmp_path):
    """id 不得含 fetched_at、絕對路徑、git 狀態或 OS 路徑分隔符。"""
    first = build(tmp_path / "a", now=lambda: "2020-01-01T00:00:00Z")
    second = build(
        tmp_path / "b",
        now=lambda: "2031-12-31T23:59:59Z",
        git_state_fn=lambda: {"commit": "f" * 40, "branch": "other", "working_tree_dirty": True},
    )

    assert first.snapshot_id == second.snapshot_id
    for token in ("/", "\\", ":", str(tmp_path)):
        assert token not in first.snapshot_id


# ---------------------------------------------------------------------------
# 4. selection_hash 語意
# ---------------------------------------------------------------------------
def test_selection_hash_covers_only_selection_fields():
    assert set(selection_payload(SourceConfig())) == {
        "max_laws",
        "category_prefix",
        "include_repealed",
        "order_by",
    }


_SELECTION_BASE = dict(max_laws=None, category_prefix=None, include_repealed=True, order_by="pcode")


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_laws": 3},
        {"category_prefix": "行政"},
        {"include_repealed": False},
    ],
)
def test_selection_hash_changes_with_selection_semantics(overrides):
    base = SourceConfig(**_SELECTION_BASE)
    assert selection_hash(base) != selection_hash(SourceConfig(**{**_SELECTION_BASE, **overrides}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"timeout_seconds": 9.0},
        {"max_retries": 9},
        {"retry_backoff_seconds": 9.0},
        {"endpoint": "https://example.invalid/other"},
        {"max_skip_ratio": 0.5},
    ],
)
def test_selection_hash_ignores_non_selection_settings(overrides):
    """timeout / retry / endpoint / 容錯門檻不改變挑選結果，故不得影響 hash。"""
    base = SourceConfig(**_SELECTION_BASE)
    assert selection_hash(base) == selection_hash(SourceConfig(**{**_SELECTION_BASE, **overrides}))


def test_selection_hash_is_reproducible_from_manifest(tmp_path):
    """manifest 的 selection 欄位必須足以自行重算 selection_hash（可稽核）。"""
    manifest = read_manifest(build(tmp_path).path)
    recomputed = hashlib.sha256(
        hashing.canonical_json(manifest["selection"]).encode("utf-8")
    ).hexdigest()
    assert recomputed == manifest["selection_hash"]


def test_output_root_does_not_affect_selection_hash(tmp_path):
    a = build(tmp_path / "a")
    b = build(tmp_path / "b")
    assert a.manifest.selection_hash == b.manifest.selection_hash


# ---------------------------------------------------------------------------
# 5/6. corpus_hash 與輸入順序無關
# ---------------------------------------------------------------------------
def test_corpus_hash_comes_from_existing_hashing_module(tmp_path):
    """corpus_hash 必須就是既有 ``evaluation/v11/hashing.py`` 對 normalized/ 的結果。"""
    result = build(tmp_path)
    normalized_dir = os.path.join(result.path, NORMALIZED_DIRNAME)

    expected = hashing.corpus_hash([normalized_dir], base_dir=result.path)
    assert result.manifest.corpus_hash == expected


def test_shuffled_source_order_yields_identical_normalized_corpus(tmp_path):
    """來源 ``Laws[]`` 順序改變，不得改變 normalized corpus 或 corpus_hash。"""
    document = load_fixture_document()
    shuffled = copy.deepcopy(document)
    shuffled["Laws"] = list(reversed(shuffled["Laws"]))

    original = build(tmp_path / "a", raw=build_zip(document))
    reordered = build(tmp_path / "b", raw=build_zip(shuffled))

    assert original.manifest.corpus_hash == reordered.manifest.corpus_hash
    assert _normalized_bytes(original.path) == _normalized_bytes(reordered.path)


def test_raw_metadata_does_not_affect_corpus_hash(tmp_path):
    """raw ZIP 內容改變（不影響正規化輸出時）不得改變 corpus_hash。"""
    baseline = build(tmp_path / "a")
    altered = build(tmp_path / "b", raw=altered_raw_bytes())

    assert altered.manifest.raw_sha256 != baseline.manifest.raw_sha256
    assert altered.manifest.corpus_hash == baseline.manifest.corpus_hash


def test_corpus_hash_changes_when_normalized_content_changes(tmp_path):
    document = load_fixture_document()
    target = next(law for law in document["Laws"] if law["LawURL"].lower().endswith(PCODE_UNIFORM.lower()))
    target["LawName"] = target["LawName"] + "（修正）"

    baseline = build(tmp_path / "a")
    changed = build(tmp_path / "b", raw=build_zip(document))

    assert changed.manifest.corpus_hash != baseline.manifest.corpus_hash


def _normalized_bytes(snapshot_path: str) -> dict[str, bytes]:
    normalized_dir = os.path.join(snapshot_path, NORMALIZED_DIRNAME)
    out: dict[str, bytes] = {}
    for name in sorted(os.listdir(normalized_dir)):
        with open(os.path.join(normalized_dir, name), "rb") as handle:
            out[name] = handle.read()
    return out


# ---------------------------------------------------------------------------
# 7/8/9. manifest 欄位與 skipped-record provenance
# ---------------------------------------------------------------------------
def test_manifest_counts_match_snapshot_contents(tmp_path):
    result = build(tmp_path)
    manifest = read_manifest(result.path)
    files = sorted(os.listdir(os.path.join(result.path, NORMALIZED_DIRNAME)))

    assert manifest["source_record_count"] == 4
    assert manifest["selected_record_count"] == 4
    assert manifest["normalized_document_count"] == len(files) == 4
    assert manifest["article_count"] > 0
    assert manifest["skipped_article_count"] >= 0
    assert manifest["raw_bytes"] == len(FIXTURE_ZIP)
    assert manifest["raw_sha256"] == hashlib.sha256(FIXTURE_ZIP).hexdigest()
    assert len(manifest["raw_sha256"]) == 64, "manifest 必須保留完整 sha256，非 id 中的短碼"
    assert manifest["source_update_date"] == load_fixture_document()["UpdateDate"]
    assert manifest["fetched_at"] == FIXED_FETCHED_AT
    assert manifest["git_state"] == FIXED_GIT_STATE


def test_manifest_records_skipped_count_and_reason_aggregation(tmp_path):
    document = load_fixture_document()
    document["Laws"][0]["LawURL"] = "https://example.invalid/no-code"
    document["Laws"][1]["LawName"] = "  "
    document["Laws"].insert(0, "not-an-object")

    config = make_config()
    result = build(
        tmp_path,
        config=CorpusConfig(
            provider=config.provider,
            source=SourceConfig(
                max_laws=None, include_repealed=True, order_by="pcode", max_skip_ratio=0.9
            ),
            output_root=config.output_root,
        ),
        raw=build_zip(document),
    )
    manifest = read_manifest(result.path)

    assert manifest["skipped_record_count"] == 3
    assert manifest["skipped_record_reason_counts"] == {
        "missing_law_name": 1,
        "missing_pcode": 1,
        "record_not_object": 1,
    }
    # 明細必須可機器讀取且依 index 排序（不得只留一行 log）
    indices = [entry["index"] for entry in manifest["skipped_records"]]
    assert indices == sorted(indices)
    assert {entry["reason"] for entry in manifest["skipped_records"]} == set(
        manifest["skipped_record_reason_counts"]
    )
    # source_record_count 是來源 Laws[] 總筆數 = 有效 + 跳過
    assert manifest["source_record_count"] == manifest["selected_record_count"] + 3


def test_manifest_has_no_skipped_records_for_clean_source(tmp_path):
    manifest = read_manifest(build(tmp_path).path)
    assert manifest["skipped_record_count"] == 0
    assert manifest["skipped_record_reason_counts"] == {}
    assert manifest["skipped_records"] == []


# ---------------------------------------------------------------------------
# skipped_article_count（record 層與 article 層是兩個不同的統計）
# ---------------------------------------------------------------------------
def make_law(pcode: str, contents: list[str]) -> dict:
    """一部合法法規；``contents`` 每個元素即一條 type-``A`` 條文的內容。"""
    return {
        "LawLevel": "法律",
        "LawName": f"測試法規 {pcode}",
        "LawURL": f"https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={pcode}",
        "LawCategory": "行政＞測試部",
        "LawModifiedDate": "20260101",
        "LawEffectiveDate": "",
        "LawEffectiveNote": "",
        "LawAbandonNote": "",
        "LawHasEngVersion": "N",
        "EngLawName": "",
        "LawHistories": "",
        "LawForeword": "",
        "LawAttachements": [],
        "LawArticles": [
            {"ArticleType": "A", "ArticleNo": f"第 {i + 1} 條", "ArticleContent": content}
            for i, content in enumerate(contents)
        ],
    }


def test_skipped_article_count_includes_laws_that_normalize_to_nothing(tmp_path):
    """一部法規所有條文皆空時，它仍是 valid selected record。

    ``to_markdown`` 對它回傳 ``None``，因此**不能**只加總成功產出的
    ``NormalizedDocument.metadata``；那些被跳過的條文仍必須計入
    ``skipped_article_count``，且該法規**不得**被誤算為 malformed source record。
    """
    document = load_fixture_document()
    document["Laws"] = [
        make_law("T0000001", ["第一條內容", "第二條內容", "   "]),  # 2 可用 + 1 空
        make_law("T0000002", ["", "   "]),  # 全空 -> 正規化後無 document
    ]

    result = build(tmp_path, raw=build_zip(document))
    manifest = read_manifest(result.path)

    assert manifest["selected_record_count"] == 2
    assert manifest["normalized_document_count"] == 1
    assert manifest["article_count"] == 2
    assert manifest["skipped_article_count"] == 3

    # 全空的法規是合法來源 record，不是 malformed
    assert manifest["skipped_record_count"] == 0
    assert manifest["skipped_records"] == []
    assert manifest["source_record_count"] == 2

    # snapshot 仍必須成功建立，且只含有內容的那一部法規
    assert sorted(os.listdir(os.path.join(result.path, NORMALIZED_DIRNAME))) == ["T0000001.md"]
    assert manifest["corpus_hash"]


def test_skipped_article_count_is_zero_when_every_article_is_emitted(tmp_path):
    document = load_fixture_document()
    document["Laws"] = [make_law("T0000001", ["甲", "乙", "丙"])]

    manifest = read_manifest(build(tmp_path, raw=build_zip(document)).path)

    assert manifest["article_count"] == 3
    assert manifest["skipped_article_count"] == 0


def test_skipped_article_count_agrees_with_per_document_metadata(tmp_path):
    """全部法規都成功正規化時，manifest 值必須等於各 document metadata 的加總。

    這鎖住 ``record.article_count - emitted`` 與 ``normalize.py`` 的 skip 判定
    不會悄悄漂移。
    """
    document = load_fixture_document()
    document["Laws"] = [
        make_law("T0000001", ["甲", "", "乙"]),
        make_law("T0000002", ["丙", "   "]),
    ]

    result = build(tmp_path, raw=build_zip(document))
    per_document_total = sum(
        int(to_markdown(record).metadata["skipped_article_count"])
        for record in _selected_records(build_zip(document))
    )

    assert result.manifest.normalized_document_count == 2
    assert result.manifest.skipped_article_count == per_document_total == 2


def _selected_records(raw_zip: bytes) -> list:
    _update, records, _skipped = parse_raw(raw_zip)
    return select(records, make_config().source)


# ---------------------------------------------------------------------------
# 10/11/12. normalized 輸出的位元組契約
# ---------------------------------------------------------------------------
def test_normalized_files_are_utf8_without_bom_and_lf_only(tmp_path):
    result = build(tmp_path)
    for name, raw in _normalized_bytes(result.path).items():
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{name} 不得含 UTF-8 BOM"
        assert b"\r" not in raw, f"{name} 不得含 CR（一律 LF）"
        raw.decode("utf-8")  # 必須是合法 UTF-8


def test_manifest_json_is_utf8_without_bom_and_lf_only(tmp_path):
    raw = read_manifest_bytes(build(tmp_path).path)
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw


def test_normalized_filenames_are_pcode_markdown(tmp_path):
    result = build(tmp_path)
    files = sorted(os.listdir(os.path.join(result.path, NORMALIZED_DIRNAME)))
    assert files == [f"{pcode}.md" for pcode in ALL_PCODES_SORTED]


def test_snapshot_layout_and_raw_artifacts(tmp_path):
    result = build(tmp_path)
    raw_dir = os.path.join(result.path, RAW_DIRNAME)

    assert sorted(os.listdir(result.path)) == [MANIFEST_FILENAME, NORMALIZED_DIRNAME, RAW_DIRNAME]
    with open(os.path.join(raw_dir, RAW_ZIP_FILENAME), "rb") as handle:
        assert handle.read() == FIXTURE_ZIP, "raw ZIP 必須原樣保留，可重跑正規化"
    # 來源自帶成員在低成本可得時一併保留供稽核
    assert set(os.listdir(raw_dir)) >= {RAW_ZIP_FILENAME, "ChLaw.json", "schema.csv"}


# ---------------------------------------------------------------------------
# 13. raw artifact 的 gitignore 政策
# ---------------------------------------------------------------------------
def test_gitignore_covers_raw_but_not_tracked_snapshot_artifacts():
    with open(GITIGNORE_PATH, "r", encoding="utf-8") as handle:
        lines = {line.strip() for line in handle}

    assert "evaluation/corpora/*/*/raw/" in lines
    assert "evaluation/corpora/*/.*.tmp/" in lines
    # manifest.json 與 normalized/ 是受追蹤產物，不得被任何規則忽略
    assert not any(
        pattern.startswith("evaluation/corpora") and ("normalized" in pattern or "manifest" in pattern)
        for pattern in lines
    )


def test_raw_artifact_paths_match_the_gitignore_pattern(tmp_path):
    """實際產出的 raw 路徑必須落在 ``evaluation/corpora/*/*/raw/`` 之下。"""
    result = build(tmp_path)
    snapshot_id = result.snapshot_id
    rel = f"evaluation/corpora/moj-law/{snapshot_id}/{RAW_DIRNAME}/{RAW_ZIP_FILENAME}"
    segments = rel.split("/")

    assert segments[:2] == ["evaluation", "corpora"]
    assert segments[4] == RAW_DIRNAME
    # 對照：受追蹤產物不在 raw/ 之下
    assert f"evaluation/corpora/moj-law/{snapshot_id}/{MANIFEST_FILENAME}".split("/")[4] != RAW_DIRNAME


# ---------------------------------------------------------------------------
# 14. immutability
# ---------------------------------------------------------------------------
def test_existing_snapshot_is_immutable(tmp_path):
    first = build(tmp_path)
    before = _normalized_bytes(first.path)
    before_manifest = read_manifest_bytes(first.path)

    with pytest.raises(SnapshotExistsError, match="已存在"):
        build(tmp_path, now=lambda: "2099-01-01T00:00:00Z")

    assert _normalized_bytes(first.path) == before
    assert read_manifest_bytes(first.path) == before_manifest, "既有 manifest 不得被覆寫或 merge"
    assert not os.path.exists(os.path.join(tmp_path, tmp_dirname(first.snapshot_id)))


def test_occupied_snapshot_directory_is_never_merged_or_rebuilt(tmp_path):
    """目標目錄已被佔用時：不得覆寫、不得 merge、不得刪除重建。"""
    snapshot_id = build(tmp_path / "probe").snapshot_id
    root = tmp_path / "real"
    occupied = root / snapshot_id
    occupied.mkdir(parents=True)
    (occupied / "sentinel.txt").write_text("do not touch", encoding="utf-8")

    with pytest.raises(SnapshotExistsError, match=snapshot_id):
        build(root)

    assert sorted(os.listdir(occupied)) == ["sentinel.txt"], "既有內容不得被 merge 或重建"
    assert (occupied / "sentinel.txt").read_text(encoding="utf-8") == "do not touch"
    assert sorted(os.listdir(root)) == [snapshot_id], "不得留下 tmp 目錄"


# ---------------------------------------------------------------------------
# 15. atomicity：失敗不留下正式 snapshot
# ---------------------------------------------------------------------------
def test_normalize_failure_leaves_no_snapshot(tmp_path, monkeypatch):
    def boom(record):
        raise RuntimeError("normalize failed")

    monkeypatch.setattr(snapshot_module, "to_markdown", boom)

    with pytest.raises(RuntimeError, match="normalize failed"):
        build(tmp_path)

    # normalize 在建立任何目錄之前就失敗 -> output root 完全乾淨
    assert os.listdir(tmp_path) == []


def test_write_failure_leaves_no_snapshot_and_no_tmp(tmp_path):
    snapshot_id = build(tmp_path / "probe").snapshot_id
    calls = {"n": 0}
    real_write_text = snapshot_module._write_text

    def failing_write_text(path, text):
        calls["n"] += 1
        if os.path.basename(path) == MANIFEST_FILENAME:
            raise OSError("disk full")
        return real_write_text(path, text)

    target_root = tmp_path / "real"
    original = snapshot_module._write_text
    snapshot_module._write_text = failing_write_text
    try:
        with pytest.raises(OSError, match="disk full"):
            build(target_root)
    finally:
        snapshot_module._write_text = original

    assert calls["n"] > 0
    assert not os.path.exists(os.path.join(str(target_root), snapshot_id))
    assert not os.path.exists(os.path.join(str(target_root), tmp_dirname(snapshot_id)))


def test_failed_rebuild_does_not_damage_existing_snapshot(tmp_path):
    """既有 snapshot 存在時，另一次失敗的 build 不得動到它。"""
    first = build(tmp_path)
    before = read_manifest_bytes(first.path)

    # 不同 raw bytes -> 不同 snapshot_id；讓寫入在 manifest 階段失敗
    real_write_text = snapshot_module._write_text

    def failing_write_text(path, text):
        if os.path.basename(path) == MANIFEST_FILENAME:
            raise OSError("boom")
        return real_write_text(path, text)

    snapshot_module._write_text = failing_write_text
    try:
        with pytest.raises(OSError):
            build(tmp_path, raw=altered_raw_bytes())
    finally:
        snapshot_module._write_text = real_write_text

    assert read_manifest_bytes(first.path) == before
    assert _normalized_bytes(first.path)
    # output root 內只剩下第一份正式 snapshot，無任何 tmp 殘留
    assert sorted(os.listdir(tmp_path)) == [first.snapshot_id]


def test_tmp_directory_is_sibling_of_final_snapshot(tmp_path):
    """tmp 必須與 final 同 parent，避免跨 filesystem rename。"""
    result = build(tmp_path)
    tmp_name = tmp_dirname(result.snapshot_id)

    assert os.path.dirname(os.path.join(str(tmp_path), tmp_name)) == os.path.dirname(result.path)
    assert tmp_name.startswith(".") and tmp_name.endswith(".tmp")


def test_stale_own_tmp_is_reclaimed_but_foreign_directories_are_not(tmp_path):
    """只清理符合自己精確命名慣例的 tmp；不得 wildcard 刪除其他 corpus 目錄。"""
    snapshot_id = build(tmp_path / "probe").snapshot_id
    root = tmp_path / "real"
    root.mkdir()
    stale = root / tmp_dirname(snapshot_id)
    stale.mkdir()
    (stale / "leftover.txt").write_text("half-written", encoding="utf-8")
    bystander = root / "some-other-snapshot"
    bystander.mkdir()
    (bystander / "keep.md").write_text("keep me", encoding="utf-8")

    result = build(root)

    assert result.snapshot_id == snapshot_id
    assert not os.path.exists(str(stale))
    assert (bystander / "keep.md").read_text(encoding="utf-8") == "keep me"


def test_remove_own_tmp_refuses_non_tmp_directory(tmp_path):
    victim = tmp_path / "moj-chlaw-20260821-aaaaaaaa-bbbbbbbb"
    victim.mkdir()
    (victim / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SnapshotError, match="tmp 命名慣例"):
        snapshot_module._remove_own_tmp(str(victim))

    assert (victim / "manifest.json").exists()


# ---------------------------------------------------------------------------
# 16. manifest 不含 secret / 絕對路徑
# ---------------------------------------------------------------------------
def test_manifest_contains_no_absolute_paths_or_secrets(tmp_path):
    result = build(tmp_path)
    text = read_manifest_bytes(result.path).decode("utf-8")

    assert str(tmp_path) not in text
    assert REPO_ROOT not in text
    assert os.path.basename(REPO_ROOT) not in text
    assert "\\" not in text, "manifest 不得含 OS 特定路徑分隔符"
    # 唯一允許出現的絕對 URL 是來源 endpoint / 法規來源網址；不得有本機路徑或授權資訊
    for forbidden in ("api_key", "apikey", "Authorization", "authorization", "token", "C:/", "C:\\"):
        assert forbidden not in text


def test_manifest_excludes_output_root_and_raw_full_text(tmp_path):
    manifest = read_manifest(build(tmp_path).path)

    assert "output_root" not in manifest
    assert "path" not in manifest
    # 條文全文不得進 manifest（只留數量與 hash）
    serialized = hashing.canonical_json(manifest)
    assert len(serialized) < 8000, "manifest 應為輕量 provenance，不含來源文件全文"


# ---------------------------------------------------------------------------
# 17. 空結果明確失敗
# ---------------------------------------------------------------------------
def test_empty_selection_fails_without_creating_snapshot(tmp_path):
    config = make_config(category_prefix="不存在的類別前綴")

    with pytest.raises(MojLawSelectionError, match="挑選結果為空"):
        build(tmp_path, config=config)

    assert not os.path.isdir(tmp_path) or os.listdir(tmp_path) == []


def test_all_documents_empty_fails_without_creating_snapshot(tmp_path):
    document = load_fixture_document()
    for law in document["Laws"]:
        for article in law["LawArticles"]:
            article["ArticleContent"] = "   "

    with pytest.raises(SnapshotError, match="無任何文件"):
        build(tmp_path, raw=build_zip(document))

    assert not os.path.isdir(tmp_path) or os.listdir(tmp_path) == []


# ---------------------------------------------------------------------------
# 決定性總測：兩個不同 output root
# ---------------------------------------------------------------------------
def test_two_output_roots_produce_identical_snapshot(tmp_path):
    """同一份 fixture、兩個不同 output root -> 除 fetched_at 外完全一致。

    ``git_state`` 在此以注入的 stub 固定；真實 ``git_state`` 於
    ``test_real_git_state_shape`` 單獨驗證其欄位契約。
    """
    first = build(tmp_path / "root-one", now=lambda: "2026-01-01T00:00:00Z")
    second = build(tmp_path / "root-two", now=lambda: "2026-06-30T12:00:00Z")

    assert first.snapshot_id == second.snapshot_id
    assert _normalized_bytes(first.path) == _normalized_bytes(second.path)
    assert first.manifest.corpus_hash == second.manifest.corpus_hash
    assert first.manifest.selection_hash == second.manifest.selection_hash
    assert first.manifest.raw_sha256 == second.manifest.raw_sha256

    a = read_manifest(first.path)
    b = read_manifest(second.path)
    assert a["fetched_at"] != b["fetched_at"], "fetched_at 是唯一允許變動的 provenance 欄位"
    a.pop("fetched_at")
    b.pop("fetched_at")
    assert hashing.canonical_json(a) == hashing.canonical_json(b)


def test_repeated_build_in_fresh_root_is_byte_identical_apart_from_fetched_at(tmp_path):
    first = read_manifest_bytes(build(tmp_path / "a").path)
    second = read_manifest_bytes(build(tmp_path / "b").path)
    assert first == second, "fetched_at 固定時，manifest 必須位元組相同"


def test_real_git_state_shape(tmp_path):
    """不注入 stub 時，git_state 必須是既有 ``hashing.git_state`` 的輸出形狀。"""
    result = build_snapshot(
        make_config(),
        output_root=str(tmp_path),
        raw_zip=FIXTURE_ZIP,
        now=lambda: FIXED_FETCHED_AT,
    )
    assert set(result.manifest.git_state) == {"commit", "branch", "working_tree_dirty"}


# ---------------------------------------------------------------------------
# fetch 接縫（仍完全 offline）
# ---------------------------------------------------------------------------
def test_build_snapshot_uses_injected_fetcher_when_no_raw_zip(tmp_path):
    fetcher = RecordingFetcher([FIXTURE_ZIP])
    sleeper = RecordingSleeper()

    result = build_snapshot(
        make_config(),
        output_root=str(tmp_path),
        fetcher=fetcher,
        sleeper=sleeper,
        now=lambda: FIXED_FETCHED_AT,
        git_state_fn=lambda: dict(FIXED_GIT_STATE),
    )

    assert len(fetcher.calls) == 1
    assert sleeper.delays == []
    assert result.manifest.normalized_document_count == 4


# ---------------------------------------------------------------------------
# config 驗證與受追蹤的 pilot config
# ---------------------------------------------------------------------------
def test_pilot_config_matches_plan_defaults():
    config = load_config(PILOT_CONFIG_PATH)

    assert config.provider == PROVIDER_MOJ_LAW
    assert config.source.endpoint == "https://law.moj.gov.tw/api/Ch/Law/JSON"
    assert config.source.timeout_seconds == 120
    assert config.source.max_retries == 3
    assert config.source.retry_backoff_seconds == 2.0
    assert config.source.max_laws == 100
    assert config.source.category_prefix is None
    assert config.source.include_repealed is True
    assert config.source.order_by == "pcode"
    assert config.output_root == "evaluation/corpora/moj-law"


def test_pilot_config_contains_no_secret():
    with open(PILOT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        text = handle.read()
    lowered = text.lower()
    for forbidden in ("api_key", "apikey", "authorization", "token", "secret", "password"):
        assert forbidden not in lowered


@pytest.mark.parametrize(
    "mutate,pattern",
    [
        (lambda c: c.__setitem__("provider", "other"), "provider"),
        (lambda c: c.__setitem__("endpoint", ""), "endpoint"),
        (lambda c: c.__setitem__("max_laws", "100"), "max_laws"),
        (lambda c: c.__setitem__("include_repealed", "yes"), "include_repealed"),
        (lambda c: c.__setitem__("output_root", "/abs/path"), "output_root"),
        (lambda c: c.__setitem__("api_key", "secret"), "未知欄位"),
        (lambda c: c.pop("endpoint"), "缺少必要欄位"),
    ],
)
def test_config_validation_rejects_contract_violations(mutate, pattern):
    with open(PILOT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    mutate(raw)

    with pytest.raises(CorpusConfigError, match=pattern):
        parse_config(raw)


def test_load_config_reports_missing_file():
    with pytest.raises(CorpusConfigError, match="找不到 config"):
        load_config(os.path.join(REPO_ROOT, "evaluation", "configs", "does-not-exist.json"))


# ---------------------------------------------------------------------------
# explicit pcode selection（Slice D2）：backward compatibility 是首要契約
# ---------------------------------------------------------------------------
CHALLENGE_CONFIG_PATH = os.path.join(
    REPO_ROOT, "evaluation", "configs", "corpus-moj-law-challenge.json"
)

#: 既有 pcode-first pilot 的 ``selection_hash``，已固化於受追蹤 snapshot
#: ``moj-chlaw-20260821-2e815ced-9629909b`` 的 manifest。新增任何 optional
#: selection 能力都**不得**改變這個值 —— 舊 pilot 必須永遠可重現。
PILOT_SELECTION_HASH = "2e815ced283adb3d04556db44b4300078bc3fb09e39c87fc2450da7c1476ad14"


def test_pilot_selection_hash_is_backward_compatible():
    """回歸測試：舊 pilot config 的 selection_hash 必須仍等於既有值。"""
    assert selection_hash(load_config(PILOT_CONFIG_PATH).source) == PILOT_SELECTION_HASH


def test_optional_selection_field_stays_out_of_payload_when_unused():
    """``include_pcodes`` 未使用時不得出現在 payload —— 否則舊 hash 會改變。"""
    payload = selection_payload(SourceConfig())

    assert "include_pcodes" not in payload
    assert set(payload) == {"max_laws", "category_prefix", "include_repealed", "order_by"}


def test_optional_selection_field_enters_payload_only_when_used():
    payload = selection_payload(SourceConfig(include_pcodes=("A0000001", "A0000002")))

    assert payload["include_pcodes"] == ["A0000001", "A0000002"]
    assert selection_hash(SourceConfig(include_pcodes=("A0000001",))) != PILOT_SELECTION_HASH


def test_explicit_selection_hash_is_deterministic_and_order_sensitive_only_to_content():
    """相同指名集合必得相同 hash；不同集合必得不同 hash。"""
    a = SourceConfig(include_pcodes=("A0000001", "A0000002"))
    b = SourceConfig(include_pcodes=("A0000001", "A0000002"))
    c = SourceConfig(include_pcodes=("A0000001", "A0000003"))

    assert selection_hash(a) == selection_hash(b)
    assert selection_hash(a) != selection_hash(c)


def test_challenge_config_is_a_frozen_explicit_selection():
    config = load_config(CHALLENGE_CONFIG_PATH)

    assert config.provider == PROVIDER_MOJ_LAW
    assert config.output_root == "evaluation/corpora/moj-law"
    assert config.source.max_laws == 100
    assert config.source.include_repealed is True
    assert config.source.order_by == "pcode"
    assert config.source.include_pcodes is not None
    assert len(config.source.include_pcodes) == 100
    assert len(set(config.source.include_pcodes)) == 100


def test_challenge_config_does_not_disturb_the_pilot_config():
    """兩份 config 是獨立 identity；challenge 的存在不改變 pilot 的 hash。"""
    pilot = load_config(PILOT_CONFIG_PATH)
    challenge = load_config(CHALLENGE_CONFIG_PATH)

    assert selection_hash(pilot.source) == PILOT_SELECTION_HASH
    assert selection_hash(challenge.source) != selection_hash(pilot.source)


def test_challenge_config_contains_no_secret():
    with open(CHALLENGE_CONFIG_PATH, "r", encoding="utf-8") as handle:
        lowered = handle.read().lower()
    for forbidden in ("api_key", "apikey", "authorization", "token", "secret", "password"):
        assert forbidden not in lowered


@pytest.mark.parametrize(
    "value,pattern",
    [
        (["A0000001", "A0000001"], "重複"),
        (["A0000001", "A0000001", "A0000002", "A0000002"], "重複"),
        ([], "不得為空"),
        ("A0000001", "型別錯誤"),
        ([123], "型別錯誤"),
        (["not a pcode!"], "不是合法 pcode"),
    ],
)
def test_include_pcodes_config_validation(value, pattern):
    with open(PILOT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    raw["include_pcodes"] = value

    with pytest.raises(CorpusConfigError, match=pattern):
        parse_config(raw)


def test_include_pcodes_null_is_equivalent_to_absent():
    """明確寫 ``null`` 與省略欄位必須完全等價（含 hash）。"""
    with open(PILOT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    absent = parse_config(dict(raw))
    raw["include_pcodes"] = None
    explicit_null = parse_config(raw)

    assert explicit_null.source.include_pcodes is None
    assert selection_hash(explicit_null.source) == selection_hash(absent.source)
    assert selection_hash(explicit_null.source) == PILOT_SELECTION_HASH


def test_explicit_selection_snapshot_does_not_collide_with_pcode_first_snapshot(tmp_path):
    """同一份來源 bytes、不同 selection 語意 -> 不同 snapshot_id，互不覆寫。"""
    pcode_first = build(tmp_path, config=make_config(max_laws=2))
    explicit = build(
        tmp_path,
        config=make_config(max_laws=None, include_pcodes=tuple(ALL_PCODES_SORTED[:2])),
    )

    assert pcode_first.snapshot_id != explicit.snapshot_id
    assert pcode_first.manifest.selection_hash != explicit.manifest.selection_hash
    # 先建立的 snapshot 完全不受影響
    assert os.path.isdir(pcode_first.path)
    assert read_manifest(pcode_first.path)["snapshot_id"] == pcode_first.snapshot_id
    # 兩者挑到同一批法規，故 normalized 內容（corpus_hash）相同 —— 差別只在 identity
    assert pcode_first.manifest.corpus_hash == explicit.manifest.corpus_hash


def test_explicit_selection_manifest_records_the_frozen_list(tmp_path):
    """manifest 的 selection 欄位必須足以自行重算 selection_hash（含指名清單）。"""
    requested = tuple(ALL_PCODES_SORTED[:3])
    manifest = read_manifest(
        build(tmp_path, config=make_config(max_laws=None, include_pcodes=requested)).path
    )

    assert manifest["selection"]["include_pcodes"] == list(requested)
    recomputed = hashlib.sha256(
        hashing.canonical_json(manifest["selection"]).encode("utf-8")
    ).hexdigest()
    assert recomputed == manifest["selection_hash"]


def test_explicit_selection_snapshot_is_immutable_like_any_other(tmp_path):
    config = make_config(max_laws=None, include_pcodes=tuple(ALL_PCODES_SORTED[:2]))
    first = build(tmp_path, config=config)

    with pytest.raises(SnapshotExistsError):
        build(tmp_path, config=config)

    assert read_manifest(first.path)["snapshot_id"] == first.snapshot_id


def test_explicit_selection_failure_leaves_no_snapshot(tmp_path):
    config = make_config(max_laws=None, include_pcodes=(ALL_PCODES_SORTED[0], "Z9999999"))

    with pytest.raises(MojLawSelectionError, match="不存在於來源"):
        build(tmp_path, config=config)

    assert os.listdir(tmp_path) == []


def test_tracked_pilot_snapshot_manifest_still_matches_the_pilot_config():
    """受追蹤的 pcode-first snapshot 必須仍能由現行 pilot config 重算出同一 identity。

    這是 backward compatibility 最直接的證據：manifest 是**已 commit 的產物**，
    它記錄的 selection payload 與 selection_hash 必須與今天的 ``selection_payload``
    / ``selection_hash`` 逐位元組相符。
    """
    manifest_path = os.path.join(
        REPO_ROOT, "evaluation", "corpora", "moj-law",
        "moj-chlaw-20260821-2e815ced-9629909b", MANIFEST_FILENAME,
    )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    source = load_config(PILOT_CONFIG_PATH).source

    assert manifest["selection"] == selection_payload(source)
    assert manifest["selection_hash"] == selection_hash(source) == PILOT_SELECTION_HASH
    assert manifest["snapshot_id"] == build_snapshot_id(
        manifest["source_update_date"], manifest["selection_hash"], manifest["raw_sha256"]
    )
