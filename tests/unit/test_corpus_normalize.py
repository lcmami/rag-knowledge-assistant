"""LawRecord -> Markdown 正規化測試（Slice A）。

本模組最重要的責任是鎖住計畫 §5.5 / Clarification C1 的不變式：

    ``ArticleType == "C"``（編章節）只能作為其後條文的章節 context，
    **不得**成為既有 loader 眼中的獨立法律條文 evidence。

因此測試不只斷言字串，還會把產出的 Markdown 真的餵進
``app/services/document_loader.py::load_markdown``，雙向證明：

  (a) 沒有任何 section／heading element 等同於某個編章節標題；
  (b) 該編章節的 context 仍完整保留在其後條文的 section 名稱中。

``load_markdown`` 在此僅供**讀取驗證**；依計畫 Slice A 不得修改 ``app/**``。

依 Slice A 邊界，本檔不測 snapshot / corpus_hash / manifest / Qdrant。
"""
from __future__ import annotations

import pytest

from app.core.constants import ELEMENT_HEADING, ELEMENT_LIST, ELEMENT_PARAGRAPH
from app.services.document_loader import load_markdown
from evaluation.corpus.models import LawArticle, LawRecord
from evaluation.corpus.normalize import CHAPTER_SEPARATOR, to_markdown
from evaluation.corpus.moj_law import parse_raw
from tests.unit.test_corpus_moj_law import (
    PCODE_ESTATE,
    PCODE_TEMP,
    PCODE_UNIFORM,
    fixture_zip,
)


def article(no: str, content: str) -> LawArticle:
    return LawArticle(article_type="A", article_no=no, article_content=content)


def chapter(title: str, depth: int) -> LawArticle:
    """以行首空白編碼深度，比照來源實際格式（3 空白 = 1 層）。"""
    return LawArticle(article_type="C", article_no="", article_content=" " * (depth * 3) + title)


def make_record(**overrides) -> LawRecord:
    base = dict(
        pcode="T0000001",
        law_level="法律",
        law_name="測試條例",
        law_url="https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=T0000001",
        law_category="行政＞測試部",
        law_modified_date="20260101",
        law_effective_date="",
        law_effective_note="",
        law_abandon_note="",
        law_has_eng_version="N",
        eng_law_name="",
        law_histories="",
        law_foreword="",
        articles=(),
        law_attachements=(),
    )
    base.update(overrides)
    return LawRecord(**base)


def sections_of(markdown: str) -> list[str]:
    """以既有 loader 解析出的所有 heading element 文字。"""
    return [e.text for e in load_markdown(markdown) if e.element_type == ELEMENT_HEADING]


# ---------------------------------------------------------------------------
# 1. golden：完整結構
# ---------------------------------------------------------------------------
GOLDEN_MARKDOWN = """# 測試條例

- 法規名稱：測試條例
- 法規位階：法律
- 法規類別：行政＞測試部
- 最後修正日期：20260101
- 廢止狀態：現行
- 來源網址：https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=T0000001

茲制定測試條例如左：

## 第 一 章 總則｜第 1 條

本條例依測試法第一條制定之。

## 第 一 章 總則｜第 一 節 定義｜第 2 條

本條例用詞定義如下。

## 第 二 章 罰則｜第 3-1 條

違反前條規定者，處新臺幣一萬元罰鍰。
"""


def test_golden_markdown_structure():
    record = make_record(
        law_foreword="茲制定測試條例如左：\r\n",
        articles=(
            chapter("第 一 章 總則", 1),
            article("第 1 條", "本條例依測試法第一條制定之。"),
            chapter("第 一 節 定義", 2),
            article("第 2 條", "本條例用詞定義如下。"),
            chapter("第 二 章 罰則", 1),
            article("第 3-1 條", "違反前條規定者，處新臺幣一萬元罰鍰。"),
        ),
    )

    document = to_markdown(record)

    assert document is not None
    assert document.markdown == GOLDEN_MARKDOWN
    assert document.doc_id == "T0000001"
    assert document.filename == "T0000001.md"


def test_crlf_from_source_is_normalised_to_lf():
    """來源實測含 CRLF；輸出必須一律 LF，否則 corpus hash 會漂移。"""
    record = make_record(
        law_foreword="第一行\r\n第二行",
        articles=(article("第 1 條", "內容第一行\r\n內容第二行"),),
    )

    document = to_markdown(record)

    assert "\r" not in document.markdown
    assert "內容第一行\n內容第二行" in document.markdown


# ---------------------------------------------------------------------------
# 2. C1 不變式：編章節不得成為獨立 evidence，但 context 必須保留
# ---------------------------------------------------------------------------
def test_chapter_never_becomes_its_own_section():
    """(a) 編章節標題不得單獨成為 heading／section。"""
    record = make_record(
        articles=(
            chapter("第 一 章 總則", 1),
            chapter("第 一 節 定義", 2),
            article("第 1 條", "條文內容。"),
        ),
    )

    document = to_markdown(record)
    headings = sections_of(document.markdown)

    assert "第 一 章 總則" not in headings
    assert "第 一 節 定義" not in headings
    # 唯二的 heading：法規名稱（H1）與該條文（H2）
    assert headings == ["測試條例", f"第 一 章 總則{CHAPTER_SEPARATOR}第 一 節 定義{CHAPTER_SEPARATOR}第 1 條"]


def test_chapter_context_is_preserved_on_following_articles():
    """(b) 編章節 context 必須保留在其後條文的 section 名稱中。"""
    record = make_record(
        articles=(
            chapter("第 一 章 總則", 1),
            article("第 1 條", "甲。"),
            chapter("第 二 章 罰則", 1),
            article("第 2 條", "乙。"),
        ),
    )

    elements = load_markdown(to_markdown(record).markdown)
    by_text = {e.text: e.section for e in elements}

    assert by_text["甲。"] == f"第 一 章 總則{CHAPTER_SEPARATOR}第 1 條"
    assert by_text["乙。"] == f"第 二 章 罰則{CHAPTER_SEPARATOR}第 2 條"


def test_chapter_stack_truncates_by_depth():
    """同深度的新編章節取代舊的，不累積。"""
    record = make_record(
        articles=(
            chapter("第 一 章 甲章", 1),
            chapter("第 一 節 甲節", 2),
            article("第 1 條", "一。"),
            chapter("第 二 節 乙節", 2),
            article("第 2 條", "二。"),
            chapter("第 二 章 乙章", 1),
            article("第 3 條", "三。"),
        ),
    )

    headings = sections_of(to_markdown(record).markdown)

    sep = CHAPTER_SEPARATOR
    assert headings[1:] == [
        f"第 一 章 甲章{sep}第 一 節 甲節{sep}第 1 條",
        f"第 一 章 甲章{sep}第 二 節 乙節{sep}第 2 條",
        f"第 二 章 乙章{sep}第 3 條",
    ]


def test_article_without_chapter_context_uses_bare_article_no():
    record = make_record(articles=(article("第 1 條", "無章節脈絡。"),))
    headings = sections_of(to_markdown(record).markdown)
    assert headings == ["測試條例", "第 1 條"]


def test_depth_zero_chapter_resets_the_stack():
    record = make_record(
        articles=(
            chapter("第 一 章 甲", 1),
            chapter("附則", 0),
            article("第 9 條", "內容。"),
        ),
    )
    headings = sections_of(to_markdown(record).markdown)
    assert headings[1] == f"附則{CHAPTER_SEPARATOR}第 9 條"


def test_no_element_originates_from_a_chapter_row():
    """編章節不得以任何形式成為獨立 element（heading 或段落）。"""
    record = make_record(
        articles=(
            chapter("第 一 章 獨特章名", 1),
            article("第 1 條", "條文內容。"),
        ),
    )

    elements = load_markdown(to_markdown(record).markdown)
    standalone = [e for e in elements if e.text.strip() == "第 一 章 獨特章名"]

    assert standalone == [], "編章節不得成為獨立 element"


# ---------------------------------------------------------------------------
# 2b. Loader contract regression — 凍結既有 load_markdown 對本格式的實際行為
# ---------------------------------------------------------------------------
# 以下測試**只記錄現況**，不主張現況是理想設計。它們存在的理由是：normalize.py
# 的輸出格式是為了配合 ``load_markdown`` 的已知行為而設計的，若日後有人調整
# normalize 的排版（或有人動了 loader），這些斷言會立刻指出契約已改變。
#
# 已知並在此凍結的行為（實測，非推論）：
#   - metadata 清單 → 一個 **獨立的 ELEMENT_LIST element**，section 為法規名稱；
#   - foreword     → 一個 **獨立的 ELEMENT_PARAGRAPH element**，section 為法規名稱；
#   - ArticleType C → **不產生任何 element**（見 §2 的 C1 不變式測試）；
#   - 條文 heading  → ELEMENT_HEADING，且其後條文內容的 section 與之一致。
#
# 即 metadata 與 foreword **確實**會成為 standalone element。是否構成 retrieval
# 問題（例如被當成 evidence 回傳）留待 Slice C 判斷；Slice A 不因此重設計格式，
# 也不修改 ``app/**``。


LOADER_CONTRACT_RECORD_ARTICLES = (
    chapter("第 一 章 總則", 1),
    article("第 1 條", "本條例依測試法第一條制定之。"),
)


def loader_contract_elements():
    record = make_record(
        law_foreword="茲制定測試條例如左：\r\n謹此。",
        articles=LOADER_CONTRACT_RECORD_ARTICLES,
    )
    return load_markdown(to_markdown(record).markdown)


def test_loader_contract_full_element_sequence():
    """完整凍結 element 序列：(type, section, text) 三元組。"""
    elements = loader_contract_elements()

    assert [(e.element_type, e.section) for e in elements] == [
        (ELEMENT_HEADING, "測試條例"),
        (ELEMENT_LIST, "測試條例"),
        (ELEMENT_PARAGRAPH, "測試條例"),
        (ELEMENT_HEADING, f"第 一 章 總則{CHAPTER_SEPARATOR}第 1 條"),
        (ELEMENT_PARAGRAPH, f"第 一 章 總則{CHAPTER_SEPARATOR}第 1 條"),
    ]


def test_loader_contract_metadata_becomes_one_standalone_list_element():
    """metadata 六行合併為**單一** ELEMENT_LIST，不逐行拆散、不成為 heading。"""
    elements = loader_contract_elements()
    lists = [e for e in elements if e.element_type == ELEMENT_LIST]

    assert len(lists) == 1
    metadata = lists[0]
    assert metadata.section == "測試條例", "metadata 掛在法規名稱 section 之下"
    assert metadata.text.startswith("- 法規名稱：測試條例")
    assert metadata.text.count("\n") == 5, "六行 metadata 必須留在同一個 element"
    assert "- 來源網址：" in metadata.text


def test_loader_contract_foreword_becomes_one_standalone_paragraph_element():
    """foreword 成為獨立段落 element（現況；非本 slice 要解決的問題）。"""
    elements = loader_contract_elements()
    paragraphs = [e for e in elements if e.element_type == ELEMENT_PARAGRAPH]

    foreword_elements = [e for e in paragraphs if "茲制定測試條例如左" in e.text]
    assert len(foreword_elements) == 1
    foreword = foreword_elements[0]
    assert foreword.section == "測試條例", "foreword 掛在法規名稱 section，非任何條文"
    assert foreword.text == "茲制定測試條例如左：\n謹此。", "CRLF 內部換行不拆成兩個 element"


def test_loader_contract_chapter_produces_no_element_of_any_type():
    """ArticleType C 不產生 standalone evidence — 任何 element type 都不行。"""
    elements = loader_contract_elements()

    assert not any(e.text.strip() == "第 一 章 總則" for e in elements)
    assert not any(
        e.element_type != ELEMENT_HEADING and "第 一 章 總則" in e.text for e in elements
    ), "編章節文字只能出現在條文 heading 的路徑中"


def test_loader_contract_article_identity_survives_the_loader():
    """條文 section identity：heading 文字 == 其後內容的 section。"""
    elements = loader_contract_elements()
    expected = f"第 一 章 總則{CHAPTER_SEPARATOR}第 1 條"

    heading = next(e for e in elements if e.element_type == ELEMENT_HEADING and e.text == expected)
    body = next(e for e in elements if e.text == "本條例依測試法第一條制定之。")

    assert heading.section == expected
    assert body.section == expected
    assert body.element_type == ELEMENT_PARAGRAPH


def test_loader_contract_holds_on_real_fixture_law(fixture_records):
    """真實法規（有 foreword、已廢止）上重現同一 element 佈局。"""
    document = to_markdown(fixture_records[PCODE_TEMP])
    elements = load_markdown(document.markdown)

    assert elements[0].element_type == ELEMENT_HEADING
    assert elements[0].text == fixture_records[PCODE_TEMP].law_name
    assert elements[1].element_type == ELEMENT_LIST
    assert elements[1].text.startswith("- 法規名稱：")
    assert elements[1].section == elements[0].text
    # foreword 段落，仍掛在法規名稱 section 之下
    assert elements[2].element_type == ELEMENT_PARAGRAPH
    assert elements[2].section == elements[0].text
    # 其後才是第一個條文 heading
    assert elements[3].element_type == ELEMENT_HEADING


# ---------------------------------------------------------------------------
# 3. metadata / 附件 / 空條文
# ---------------------------------------------------------------------------
def test_attachments_are_ignored_but_counted():
    """PDF 附件屬 out of scope：不下載、不寫入 Markdown，但保留數量供統計。"""
    record = make_record(
        articles=(article("第 1 條", "內容。"),),
        law_attachements=(
            {"FileName": "附圖一：制式圖.PDF", "FileURL": "https://example.invalid/a.pdf"},
            {"FileName": "附圖二：制式圖.PDF", "FileURL": "https://example.invalid/b.pdf"},
        ),
    )

    document = to_markdown(record)

    assert "附圖一" not in document.markdown
    assert ".PDF" not in document.markdown
    assert "example.invalid" not in document.markdown
    assert document.metadata["attachment_count"] == 2


def test_repealed_metadata_and_header():
    record = make_record(law_abandon_note="廢", articles=(article("第 1 條", "內容。"),))
    document = to_markdown(record)

    assert document.metadata["is_repealed"] is True
    assert "- 廢止狀態：已廢止" in document.markdown
    assert "- 廢止註記：廢" in document.markdown


def test_current_law_marks_metadata_as_not_repealed():
    record = make_record(articles=(article("第 1 條", "內容。"),))
    document = to_markdown(record)

    assert document.metadata["is_repealed"] is False
    assert "- 廢止狀態：現行" in document.markdown


def test_empty_article_content_is_skipped_and_counted():
    record = make_record(
        articles=(
            article("第 1 條", "有內容。"),
            article("第 2 條", "   \n  "),
            article("第 3 條", ""),
        ),
    )

    document = to_markdown(record)

    assert document.metadata["article_count"] == 1
    assert document.metadata["skipped_article_count"] == 2
    assert "第 2 條" not in document.markdown
    assert "第 3 條" not in document.markdown


def test_law_with_no_usable_articles_returns_none():
    """整部法規無可用條文時回傳 None，由呼叫端整筆跳過。"""
    record = make_record(articles=(article("第 1 條", "  "), chapter("第 一 章 空", 1)))
    assert to_markdown(record) is None


def test_law_with_only_chapters_returns_none():
    record = make_record(articles=(chapter("第 一 章 空", 1),))
    assert to_markdown(record) is None


def test_article_without_number_or_chapter_is_skipped():
    record = make_record(
        articles=(article("", "無法定位的內容。"), article("第 2 條", "可定位。")),
    )
    document = to_markdown(record)

    assert document.metadata["article_count"] == 1
    assert "無法定位的內容。" not in document.markdown


# ---------------------------------------------------------------------------
# 4. 真實 fixture 上的行為
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fixture_records():
    _update, records, _skipped = parse_raw(fixture_zip())
    return {r.pcode: r for r in records}


def test_real_nested_law_produces_no_chapter_only_sections(fixture_records):
    """真實巢狀編章節（章 + 節）上的 C1 不變式。"""
    record = fixture_records[PCODE_ESTATE]
    document = to_markdown(record)
    headings = sections_of(document.markdown)

    chapter_titles = {a.chapter_title for a in record.articles if a.is_chapter}
    assert chapter_titles, "fixture 必須含編章節，否則本測試無意義"

    # (a) 沒有任何 heading 等同於某個編章節標題
    assert chapter_titles.isdisjoint(set(headings))
    # (b) 每個編章節標題都出現在某個條文 heading 的路徑中
    for title in chapter_titles:
        assert any(title in h for h in headings), f"章節 context 遺失：{title}"


def test_real_nested_law_builds_two_level_chapter_path(fixture_records):
    document = to_markdown(fixture_records[PCODE_ESTATE])
    headings = sections_of(document.markdown)

    two_level = [h for h in headings if h.count(CHAPTER_SEPARATOR) == 2]
    assert two_level, "應存在 章｜節｜條 的三段式 heading"


def test_real_law_preserves_hyphenated_article_number(fixture_records):
    """``第 9-1 條`` 這類條號必須原樣進入 section 名稱。"""
    document = to_markdown(fixture_records[PCODE_UNIFORM])
    assert "## 第 9-1 條" in document.markdown


def test_real_law_foreword_is_emitted(fixture_records):
    record = fixture_records[PCODE_TEMP]
    document = to_markdown(record)

    assert record.law_foreword.strip()
    first_line = record.law_foreword.replace("\r\n", "\n").strip().split("\n")[0]
    assert first_line in document.markdown


def test_real_laws_all_normalise_and_filenames_are_unique(fixture_records):
    documents = [to_markdown(r) for r in fixture_records.values()]

    assert all(d is not None for d in documents)
    filenames = [d.filename for d in documents]
    assert len(set(filenames)) == len(filenames)
    assert all(f.endswith(".md") for f in filenames)


def test_normalisation_is_deterministic(fixture_records):
    """同一輸入必得位元組相同的輸出。"""
    record = fixture_records[PCODE_ESTATE]
    assert to_markdown(record).markdown == to_markdown(record).markdown


def test_normalised_markdown_survives_the_real_loader(fixture_records):
    """產出的 Markdown 必須能被既有 loader 解析出非空 element。"""
    for record in fixture_records.values():
        elements = load_markdown(to_markdown(record).markdown)
        assert elements
        assert any(e.text.strip() for e in elements)
