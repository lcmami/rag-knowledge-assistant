"""MOJ provider 的 source contract 測試（Slice A）。

本模組鎖定的是**來源契約**，不是正規化輸出（那在 ``test_corpus_normalize.py``）。

測試一律 offline：
- 網路由 injected fake fetcher 取代，**永不呼叫 httpx**；
- backoff 由 injected sleeper 取代，**wall-clock 等待接近 0**（計畫 Clarification C3）。

fixture ``evaluation/corpus/fixtures/chlaw_sample.json`` 取自真實 MOJ 資料，
內含 4 部法規（其中 ``遺產稅法`` 的 ``LawArticles`` 已截取為代表性子集，以控制
fixture 體積；所有保留的欄位與條目皆為來源原文）。涵蓋：巢狀編章節、
``第 9-1 條`` 條號、附件、前言、已廢止狀態。

依計畫 Slice A 邊界，本檔**不測** snapshot / corpus_hash / manifest /
atomic write / Qdrant / embedding / retrieval。
"""
from __future__ import annotations

import copy
import io
import json
import os
import zipfile

import pytest

from evaluation.corpus.models import LawRecord
from evaluation.corpus.moj_law import (
    LAW_JSON_MEMBER,
    MojLawFetchError,
    MojLawParseError,
    MojLawSelectionError,
    SourceConfig,
    extract_pcode,
    fetch_raw,
    parse_raw,
    select,
)

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "evaluation",
    "corpus",
    "fixtures",
    "chlaw_sample.json",
)

# fixture 內 4 部法規的 pcode（升冪，即 select 的期望輸出順序）。
PCODE_TEMP = "A0000005"  # 動員戡亂時期臨時條款（已廢止、有前言）
PCODE_ADVISORY = "A0010013"  # 國策顧問委員會組織條例（已廢止）
PCODE_UNIFORM = "F0060007"  # 陸海空軍服制條例（現行、3 附件、有 第 9-1 條）
PCODE_ESTATE = "G0340074"  # 遺產稅法（已廢止、巢狀編章節）
ALL_PCODES_SORTED = [PCODE_TEMP, PCODE_ADVISORY, PCODE_UNIFORM, PCODE_ESTATE]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_fixture_document() -> dict:
    """讀取 fixture 的 JSON document（可安全 mutate 的深拷貝）。"""
    with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_zip(document: dict, *, with_bom: bool = True, member: str = LAW_JSON_MEMBER) -> bytes:
    """把 document 打包成與來源同形的 ZIP。

    來源實測為 UTF-8 **with BOM**，故預設加上 BOM，讓解析路徑真的走
    ``utf-8-sig``。
    """
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8-sig" if with_bom else "utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, payload)
        archive.writestr("schema.csv", "name,title\nLawName,法規名稱\n")
        archive.writestr("manifest.csv", "name,schema,description\n")
    return buffer.getvalue()


def fixture_zip(**kwargs) -> bytes:
    return build_zip(load_fixture_document(), **kwargs)


class RecordingFetcher:
    """可注入的 fake fetcher，記錄每次呼叫並依序回放結果。"""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[tuple[str, float]] = []

    def __call__(self, url: str, timeout: float) -> bytes:
        self.calls.append((url, timeout))
        outcome = self.results[min(len(self.calls) - 1, len(self.results) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RecordingSleeper:
    """記錄 backoff 序列但**不真的等待**。"""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


# ---------------------------------------------------------------------------
# 1. fixture parse / BOM / pcode
# ---------------------------------------------------------------------------
def test_parse_fixture_returns_update_date_and_all_records():
    update_date, records, skipped = parse_raw(fixture_zip())

    assert update_date  # 來源原字串，不做格式假設
    assert skipped == []
    assert [r.pcode for r in records] == [
        PCODE_ESTATE,
        PCODE_UNIFORM,
        PCODE_ADVISORY,
        PCODE_TEMP,
    ], "parse 必須保留來源順序；排序是 select 的責任"
    assert all(isinstance(r, LawRecord) for r in records)


def test_parse_handles_utf8_bom_and_produces_identical_records():
    """來源實測為 UTF-8-BOM；有無 BOM 都必須解析出相同結果。"""
    with_bom = parse_raw(fixture_zip(with_bom=True))
    without_bom = parse_raw(fixture_zip(with_bom=False))

    assert with_bom == without_bom
    # BOM 不得洩漏進第一個欄位值
    assert not with_bom[1][0].law_level.startswith("﻿")


def test_parse_preserves_source_fields_faithfully():
    _update, records, _skipped = parse_raw(fixture_zip())
    uniform = next(r for r in records if r.pcode == PCODE_UNIFORM)

    assert uniform.law_name == "陸海空軍服制條例"
    assert uniform.law_url.endswith(f"pcode={PCODE_UNIFORM}")
    assert uniform.is_repealed is False
    # 附件忠實保留於記憶體（Slice A 不下載，但不得假裝來源沒有此欄位）
    assert len(uniform.law_attachements) == 3
    assert uniform.law_attachements[0]["FileName"].endswith(".PDF")
    # 第 N-N 條 條號必須原樣保留
    assert "第 9-1 條" in [a.article_no for a in uniform.articles]


def test_repealed_flag_derived_from_abandon_note():
    _update, records, _skipped = parse_raw(fixture_zip())
    by_pcode = {r.pcode: r for r in records}

    assert by_pcode[PCODE_ADVISORY].is_repealed is True
    assert by_pcode[PCODE_TEMP].is_repealed is True
    assert by_pcode[PCODE_UNIFORM].is_repealed is False


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=A0000001", "A0000001"),
        ("https://law.moj.gov.tw/LawClass/LawAll.aspx?PCode=G0340074", "G0340074"),
        ("https://law.moj.gov.tw/LawClass/LawAll.aspx?x=1&pcode=F0060007", "F0060007"),
        ("https://law.moj.gov.tw/LawClass/LawAll.aspx", None),
        ("https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=", None),
        ("", None),
    ],
)
def test_extract_pcode(url, expected):
    assert extract_pcode(url) == expected


def test_chapter_depth_encoded_by_leading_spaces():
    """來源以行首空白（3 的倍數）編碼編章節深度。"""
    _update, records, _skipped = parse_raw(fixture_zip())
    estate = next(r for r in records if r.pcode == PCODE_ESTATE)

    chapters = [a for a in estate.articles if a.is_chapter]
    depths = [a.chapter_depth for a in chapters]

    assert 1 in depths and 2 in depths, "fixture 必須同時含 章 與 節，才能證明巢狀行為"
    deepest = next(a for a in chapters if a.chapter_depth == 2)
    assert deepest.chapter_title == deepest.chapter_title.strip()
    assert not deepest.chapter_title.startswith(" ")


# ---------------------------------------------------------------------------
# 2. malformed / 結構性錯誤
# ---------------------------------------------------------------------------
def test_non_zip_payload_rejected():
    with pytest.raises(MojLawParseError, match="不是 ZIP"):
        parse_raw(b"<html>not a zip</html>")


def test_corrupt_zip_rejected():
    raw = bytearray(fixture_zip())
    # 破壞 central directory 尾端，讓 is_zipfile 之後的讀取失敗
    raw[-6:] = b"\x00" * 6
    with pytest.raises(MojLawParseError):
        parse_raw(bytes(raw))


def test_missing_chlaw_json_member_rejected():
    raw = build_zip(load_fixture_document(), member="SomethingElse.json")
    with pytest.raises(MojLawParseError, match=LAW_JSON_MEMBER):
        parse_raw(raw)


def test_invalid_json_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(LAW_JSON_MEMBER, b"{not json")
    with pytest.raises(MojLawParseError, match="無法解析為 JSON"):
        parse_raw(buffer.getvalue())


@pytest.mark.parametrize(
    "mutate,pattern",
    [
        (lambda d: d.pop("UpdateDate"), "UpdateDate"),
        (lambda d: d.pop("Laws"), "Laws"),
        (lambda d: d.__setitem__("Laws", {}), "Laws"),
    ],
)
def test_top_level_structure_violations_rejected(mutate, pattern):
    document = load_fixture_document()
    mutate(document)
    with pytest.raises(MojLawParseError, match=pattern):
        parse_raw(build_zip(document))


def test_duplicate_pcode_is_fatal():
    """實測 1,346/1,346 唯一；重複代表來源契約變動，不得默默吞掉。"""
    document = load_fixture_document()
    document["Laws"].append(copy.deepcopy(document["Laws"][0]))
    with pytest.raises(MojLawParseError, match="pcode 重複"):
        parse_raw(build_zip(document))


@pytest.mark.parametrize(
    "mutate,reason,expect_law_name",
    [
        (lambda law: law.__setitem__("LawName", "  "), "missing_law_name", False),
        (
            lambda law: law.__setitem__("LawURL", "https://example.invalid/no-code"),
            "missing_pcode",
            True,
        ),
        (lambda law: law.__setitem__("LawArticles", None), "missing_law_articles", True),
    ],
)
def test_malformed_record_is_skipped_not_fatal(mutate, reason, expect_law_name):
    """單筆 malformed 應跳過並記錄原因，其餘法規仍須解析成功。

    診斷必須是可機器讀取的 :class:`SkippedRecordDiagnostic`（Slice B 的 manifest
    需要 skipped 數量與 reason 分佈），而非僅一個計數或一行 log。
    """
    document = load_fixture_document()
    original_name = document["Laws"][0]["LawName"]
    mutate(document["Laws"][0])

    _update, records, skipped = parse_raw(build_zip(document), max_skip_ratio=0.5)

    assert len(records) == 3
    assert len(skipped) == 1
    diagnostic = skipped[0]
    assert set(diagnostic) == {"index", "law_name", "reason"}
    assert diagnostic["reason"] == reason
    assert diagnostic["index"] == 0
    # 名稱可判定時保留，無法判定時為 None（不以空字串偽裝成「有名字」）
    assert diagnostic["law_name"] == (original_name if expect_law_name else None)


def test_skip_ratio_above_threshold_is_fatal():
    """來源結構真的改變時，逐筆跳過會把問題掩蓋成「資料變少」。"""
    document = load_fixture_document()
    for law in document["Laws"]:
        law["LawArticles"] = None

    with pytest.raises(MojLawParseError, match="超過上限"):
        parse_raw(build_zip(document), max_skip_ratio=0.05)


def test_non_object_record_skipped():
    document = load_fixture_document()
    document["Laws"].insert(0, "not-an-object")
    _update, records, skipped = parse_raw(build_zip(document), max_skip_ratio=0.5)

    assert len(records) == 4
    assert skipped[0]["reason"] == "record_not_object"
    assert skipped[0]["law_name"] is None


# ---------------------------------------------------------------------------
# 3. deterministic selection
# ---------------------------------------------------------------------------
def test_selection_is_deterministic_regardless_of_input_order():
    """輸入順序改變不得影響輸出（排序發生在截斷之前）。"""
    _update, records, _skipped = parse_raw(fixture_zip())
    config = SourceConfig(max_laws=3)

    forward = [r.pcode for r in select(list(records), config)]
    reversed_ = [r.pcode for r in select(list(reversed(records)), config)]
    rotated = [r.pcode for r in select(records[2:] + records[:2], config)]

    assert forward == reversed_ == rotated == ALL_PCODES_SORTED[:3]


def test_selection_sorts_by_pcode_ascending():
    _update, records, _skipped = parse_raw(fixture_zip())
    selected = select(records, SourceConfig(max_laws=None))
    assert [r.pcode for r in selected] == ALL_PCODES_SORTED


def test_include_repealed_true_keeps_repealed_laws():
    _update, records, _skipped = parse_raw(fixture_zip())
    selected = select(records, SourceConfig(max_laws=None, include_repealed=True))

    assert any(r.is_repealed for r in selected)
    assert len(selected) == 4


def test_include_repealed_false_drops_repealed_laws():
    _update, records, _skipped = parse_raw(fixture_zip())
    selected = select(records, SourceConfig(max_laws=None, include_repealed=False))

    assert [r.pcode for r in selected] == [PCODE_UNIFORM]
    assert all(not r.is_repealed for r in selected)


def test_include_repealed_defaults_to_true():
    assert SourceConfig().include_repealed is True


def test_category_prefix_filter():
    _update, records, _skipped = parse_raw(fixture_zip())
    prefix = next(r for r in records if r.pcode == PCODE_UNIFORM).law_category[:2]

    selected = select(records, SourceConfig(max_laws=None, category_prefix=prefix))

    assert selected
    assert all(r.law_category.startswith(prefix) for r in selected)


def test_max_laws_truncates_after_sorting():
    _update, records, _skipped = parse_raw(fixture_zip())
    assert [r.pcode for r in select(records, SourceConfig(max_laws=1))] == [PCODE_TEMP]


def test_empty_selection_is_an_error():
    _update, records, _skipped = parse_raw(fixture_zip())
    config = SourceConfig(max_laws=None, category_prefix="不存在的類別前綴")

    with pytest.raises(MojLawSelectionError, match="挑選結果為空"):
        select(records, config)


def test_invalid_max_laws_rejected():
    _update, records, _skipped = parse_raw(fixture_zip())
    with pytest.raises(MojLawSelectionError, match="max_laws"):
        select(records, SourceConfig(max_laws=0))


def test_unsupported_order_by_rejected():
    _update, records, _skipped = parse_raw(fixture_zip())
    with pytest.raises(MojLawSelectionError, match="order_by"):
        select(records, SourceConfig(order_by="law_name"))


# ---------------------------------------------------------------------------
# 4. fetch：retry / timeout（injected fake，wall-clock 接近 0）
# ---------------------------------------------------------------------------
def test_fetch_returns_body_on_first_success():
    fetcher = RecordingFetcher([b"payload"])
    sleeper = RecordingSleeper()

    body = fetch_raw(SourceConfig(), fetcher=fetcher, sleeper=sleeper)

    assert body == b"payload"
    assert len(fetcher.calls) == 1
    assert sleeper.delays == [], "成功時不得 sleep"


def test_fetch_passes_configured_timeout_to_fetcher():
    fetcher = RecordingFetcher([b"payload"])
    fetch_raw(
        SourceConfig(timeout_seconds=7.5),
        fetcher=fetcher,
        sleeper=RecordingSleeper(),
    )
    assert fetcher.calls[0][1] == 7.5


def test_fetch_retries_then_succeeds_with_exponential_backoff():
    fetcher = RecordingFetcher([TimeoutError("t1"), TimeoutError("t2"), b"ok"])
    sleeper = RecordingSleeper()

    body = fetch_raw(
        SourceConfig(max_retries=3, retry_backoff_seconds=2.0),
        fetcher=fetcher,
        sleeper=sleeper,
    )

    assert body == b"ok"
    assert len(fetcher.calls) == 3
    assert sleeper.delays == [2.0, 4.0], "backoff 必須為 2.0 * 2**i，且成功後不再 sleep"


def test_fetch_exhausts_retries_and_reports_deterministically():
    fetcher = RecordingFetcher([TimeoutError("boom")])
    sleeper = RecordingSleeper()

    with pytest.raises(MojLawFetchError) as excinfo:
        fetch_raw(
            SourceConfig(max_retries=3, retry_backoff_seconds=1.0),
            fetcher=fetcher,
            sleeper=sleeper,
        )

    message = str(excinfo.value)
    assert "已嘗試 3 次" in message
    assert SourceConfig().endpoint in message
    assert len(fetcher.calls) == 3
    assert sleeper.delays == [1.0, 2.0], "最後一次失敗後不得再 sleep"


def test_fetch_treats_empty_body_as_failure():
    fetcher = RecordingFetcher([b"", b"", b"recovered"])
    sleeper = RecordingSleeper()

    body = fetch_raw(SourceConfig(max_retries=3), fetcher=fetcher, sleeper=sleeper)

    assert body == b"recovered"
    assert len(fetcher.calls) == 3


def test_fetch_does_not_sleep_when_retries_disabled():
    fetcher = RecordingFetcher([TimeoutError("once")])
    sleeper = RecordingSleeper()

    with pytest.raises(MojLawFetchError):
        fetch_raw(SourceConfig(max_retries=1), fetcher=fetcher, sleeper=sleeper)

    assert len(fetcher.calls) == 1
    assert sleeper.delays == []


def test_fetch_then_parse_round_trip():
    """fetch -> parse 的接縫（仍完全 offline）。"""
    fetcher = RecordingFetcher([fixture_zip()])

    raw = fetch_raw(SourceConfig(), fetcher=fetcher, sleeper=RecordingSleeper())
    _update, records, skipped = parse_raw(raw)

    assert len(records) == 4
    assert skipped == []


# ---------------------------------------------------------------------------
# 4. explicit pcode selection（Slice D2 challenge corpus）
# ---------------------------------------------------------------------------
# 契約重點：``include_pcodes`` 是一份**明確指名**的清單，挑選結果必須恰好等於它。
# 任何會讓結果變少的情形都必須明確失敗，而不是回傳較小的子集 —— challenge corpus
# 的 mandatory confuser 不得被任何設定悄悄削減。


def test_explicit_pcode_selection_returns_exactly_the_requested_laws():
    _update, records, _skipped = parse_raw(fixture_zip())
    requested = (PCODE_UNIFORM, PCODE_TEMP)

    selected = select(records, SourceConfig(max_laws=None, include_pcodes=requested))

    assert [r.pcode for r in selected] == sorted(requested)


def test_explicit_pcode_selection_is_deterministic_regardless_of_input_order():
    """來源 ``Laws[]`` 順序改變不得影響結果。"""
    _update, records, _skipped = parse_raw(fixture_zip())
    config = SourceConfig(max_laws=None, include_pcodes=(PCODE_ESTATE, PCODE_TEMP, PCODE_UNIFORM))

    forward = [r.pcode for r in select(list(records), config)]
    reversed_ = [r.pcode for r in select(list(reversed(records)), config)]
    rotated = [r.pcode for r in select(records[2:] + records[:2], config)]

    assert forward == reversed_ == rotated == sorted(config.include_pcodes)


def test_explicit_pcode_selection_ignores_requested_list_ordering():
    """frozen 清單的書寫順序不影響 canonical 結果（mandatory 先寫或後寫皆同）。"""
    _update, records, _skipped = parse_raw(fixture_zip())
    members = (PCODE_TEMP, PCODE_ADVISORY, PCODE_UNIFORM)

    ascending = select(records, SourceConfig(max_laws=None, include_pcodes=members))
    descending = select(records, SourceConfig(max_laws=None, include_pcodes=tuple(reversed(members))))
    interleaved = select(
        records, SourceConfig(max_laws=None, include_pcodes=(members[1], members[2], members[0]))
    )

    assert [r.pcode for r in ascending] == [r.pcode for r in descending]
    assert [r.pcode for r in ascending] == [r.pcode for r in interleaved]
    assert [r.pcode for r in ascending] == sorted(members)


def test_missing_requested_pcode_fails_clearly():
    _update, records, _skipped = parse_raw(fixture_zip())
    config = SourceConfig(max_laws=None, include_pcodes=(PCODE_UNIFORM, "Z9999999"))

    with pytest.raises(MojLawSelectionError, match="不存在於來源") as excinfo:
        select(records, config)
    assert "Z9999999" in str(excinfo.value)


def test_requested_pcode_filtered_out_by_other_settings_fails_instead_of_shrinking():
    """被 ``include_repealed`` 濾掉的指名法規必須報錯，不得靜默捨棄。"""
    _update, records, _skipped = parse_raw(fixture_zip())
    config = SourceConfig(
        max_laws=None, include_repealed=False, include_pcodes=(PCODE_UNIFORM, PCODE_TEMP)
    )

    with pytest.raises(MojLawSelectionError, match="被其他挑選條件濾除") as excinfo:
        select(records, config)
    assert PCODE_TEMP in str(excinfo.value)


def test_requested_pcodes_exceeding_max_laws_fail_instead_of_being_truncated():
    """mandatory 清單超過 max_laws 時必須 STOP，不得偷偷截斷。"""
    _update, records, _skipped = parse_raw(fixture_zip())
    config = SourceConfig(max_laws=2, include_pcodes=tuple(ALL_PCODES_SORTED))

    with pytest.raises(MojLawSelectionError, match="不得被截斷"):
        select(records, config)


def test_explicit_selection_at_exactly_max_laws_is_allowed():
    _update, records, _skipped = parse_raw(fixture_zip())
    config = SourceConfig(max_laws=len(ALL_PCODES_SORTED), include_pcodes=tuple(ALL_PCODES_SORTED))

    assert [r.pcode for r in select(records, config)] == ALL_PCODES_SORTED


def test_include_pcodes_defaults_to_none_preserving_original_semantics():
    """預設為 ``None`` —— 未使用此能力時走的是與新增前完全相同的路徑。"""
    _update, records, _skipped = parse_raw(fixture_zip())

    assert SourceConfig().include_pcodes is None
    assert [r.pcode for r in select(records, SourceConfig(max_laws=None))] == ALL_PCODES_SORTED
