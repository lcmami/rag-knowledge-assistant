"""MOJ 來源契約漂移偵測（docs/external-corpus-api-plan.md §9.3、Slice D）。

**這是本 repo 唯一會對外部公開 API 發出請求的測試。**

預設**不執行** —— 需顯式 ``-m external_api``，故不會讓預設測試套件變 flaky。
該 API 為公開資料、無需認證，測試不含任何 secret。

範圍嚴格限縮為**契約漂移偵測**：endpoint 是單一 bulk ZIP，一次 request 必然取得
完整 ZIP，故不存在「只下載一部分」的選項。本檔**只**驗證 response / ZIP / JSON
的結構契約，**不做** normalize、ingest、embedding、Qdrant index、retrieval 或任何
品質評測 —— 那些都由 offline 的 unit / integration 測試以受追蹤 fixture 覆蓋。

契約若真的改變，這裡會失敗；失敗即代表 ``moj_law.py`` 的假設需要重新檢視，
而不是把這個測試改鬆。
"""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from evaluation.corpus.moj_law import (
    DEFAULT_ENDPOINT,
    LAW_JSON_MEMBER,
    SourceConfig,
    extract_pcode,
    fetch_raw,
    parse_raw,
)

pytestmark = pytest.mark.external_api

#: 規劃期與 Slice D pilot 皆實測到的成員集合。
EXPECTED_MEMBERS = {LAW_JSON_MEMBER, "schema.csv", "manifest.csv"}
#: 來源規模的下界（實測 1,346）。設下界而非固定值：法規本來就會增修，
#: 數量變動是正常的；掉到個位數才代表契約真的壞了。
MIN_LAW_COUNT = 500


@pytest.fixture(scope="module")
def raw_zip() -> bytes:
    """對來源發出**唯一一次**請求，整個模組共用。"""
    return fetch_raw(SourceConfig(endpoint=DEFAULT_ENDPOINT))


def test_endpoint_returns_a_zip_archive(raw_zip):
    assert raw_zip, "來源回應為空"
    assert zipfile.is_zipfile(io.BytesIO(raw_zip)), f"回應不是 ZIP；前導 bytes={raw_zip[:16]!r}"


def test_zip_contains_the_expected_members(raw_zip):
    members = set(zipfile.ZipFile(io.BytesIO(raw_zip)).namelist())
    assert LAW_JSON_MEMBER in members, f"ZIP 缺少 {LAW_JSON_MEMBER}；實際成員={sorted(members)}"
    assert EXPECTED_MEMBERS <= members, f"ZIP 成員集合縮減；實際={sorted(members)}"


def test_law_json_is_utf8_with_bom(raw_zip):
    """來源實測為 UTF-8 **with BOM**；``parse_raw`` 依此以 utf-8-sig 解碼。"""
    payload = zipfile.ZipFile(io.BytesIO(raw_zip)).read(LAW_JSON_MEMBER)
    assert payload[:3] == b"\xef\xbb\xbf", "BOM 消失 —— utf-8-sig 解碼假設需重新檢視"
    payload.decode("utf-8-sig")


def test_top_level_structure_is_unchanged(raw_zip):
    payload = zipfile.ZipFile(io.BytesIO(raw_zip)).read(LAW_JSON_MEMBER)
    document = json.loads(payload.decode("utf-8-sig"))

    assert isinstance(document, dict)
    assert "UpdateDate" in document, "頂層缺少 UpdateDate"
    assert str(document["UpdateDate"]).strip(), "UpdateDate 為空"
    assert isinstance(document.get("Laws"), list), "頂層 Laws 缺少或不是 list"
    assert len(document["Laws"]) >= MIN_LAW_COUNT, (
        f"來源法規數 {len(document['Laws'])} 低於下界 {MIN_LAW_COUNT}"
    )


def test_law_record_shape_is_unchanged(raw_zip):
    """逐筆欄位契約：``LawURL`` 可解出 pcode、``LawArticles`` 為 list、A/C 兩型仍在。"""
    payload = zipfile.ZipFile(io.BytesIO(raw_zip)).read(LAW_JSON_MEMBER)
    laws = json.loads(payload.decode("utf-8-sig"))["Laws"]
    sample = laws[:200]

    assert all(isinstance(law, dict) for law in sample)
    assert all(str(law.get("LawName", "")).strip() for law in sample)
    assert all(extract_pcode(str(law.get("LawURL", ""))) for law in sample), "LawURL 已無法解出 pcode"
    assert all(isinstance(law.get("LawArticles"), list) for law in sample)

    types = {a.get("ArticleType") for law in sample for a in law["LawArticles"]}
    assert {"A", "C"} <= types, f"ArticleType 集合改變：{sorted(t for t in types if t)}"


def test_parse_raw_accepts_the_live_payload(raw_zip):
    """既有 parser 能吃下 live payload，且跳過率仍在容忍範圍內。

    這是最實質的一條：它把契約驗證交給**正式解析路徑**本身，而不是另寫一套斷言。
    """
    update_date, records, skipped = parse_raw(raw_zip)

    assert update_date.strip()
    assert len(records) >= MIN_LAW_COUNT
    # parse_raw 內建 5% 跳過率上限；此處另外記錄實際值，讓漂移可被看見
    assert len(skipped) / (len(records) + len(skipped)) <= 0.05
    assert len({r.pcode for r in records}) == len(records), "pcode 不再唯一"
