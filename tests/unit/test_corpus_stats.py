"""Corpus statistics 測試（Slice D）。

完全 offline、完全決定性：來源是 Slice A fixture 經 Slice B builder 產出的
snapshot，``count_tokens`` 注入決定性 fake，因此**不載入任何模型**、不碰網路、
不碰 Qdrant。

重點在於統計本身可被稽核：百分位不插值、集合輸出排序、同一輸入必得相同
canonical JSON。
"""
from __future__ import annotations

import os

import pytest

from evaluation.corpus.moj_law import SourceConfig
from evaluation.corpus.snapshot import CorpusConfig, build_snapshot
from evaluation.corpus.stats import (
    METADATA_KEYS,
    PILOT_CHUNK_CONFIG,
    build_stats,
    chunk_stats,
    collect_documents,
    confusability_stats,
    distribution,
    identity_stats,
    metadata_stats,
    parse_normalized_document,
    percentile,
    render_markdown,
    title_stem,
)
from evaluation.v11 import hashing
from tests.unit.test_corpus_moj_law import fixture_zip

#: fixture 的 4 部法規（依 pcode 升冪）與其 type-A 條文數。
FIXTURE_ARTICLES = {"A0000005": 11, "A0010013": 7, "F0060007": 12, "G0340074": 5}
TOTAL_ARTICLES = sum(FIXTURE_ARTICLES.values())  # 35
TOTAL_LAWS = len(FIXTURE_ARTICLES)


def char_tokens(text: str) -> int:
    """決定性的 fake tokenizer：1 字元 = 1 token。避免測試載入真實模型。"""
    return len(text)


@pytest.fixture(scope="module")
def snapshot_dir(tmp_path_factory) -> str:
    config = CorpusConfig(
        provider="moj_law",
        source=SourceConfig(max_laws=None, include_repealed=True, order_by="pcode"),
        output_root="evaluation/corpora/moj-law",
    )
    result = build_snapshot(
        config,
        output_root=str(tmp_path_factory.mktemp("stats-corpora")),
        raw_zip=fixture_zip(),
    )
    return result.path


@pytest.fixture(scope="module")
def documents(snapshot_dir):
    return collect_documents(snapshot_dir)


# ---------------------------------------------------------------------------
# 1. 由 normalized Markdown 還原文件事實
# ---------------------------------------------------------------------------
SAMPLE_MD = """# 測試條例

- 法規名稱：測試條例
- 法規位階：法律
- 法規類別：行政＞測試部
- 最後修正日期：20260101
- 廢止狀態：已廢止
- 來源網址：https://example.invalid/x?pcode=T0000001
- 廢止註記：本條例自中華民國一百年一月一日廢止。

本條例依測試法第一條之規定制定之。

## 第 一 章 總則｜第 1 條

第一條內容。

## 第 2 條

第二條內容。
"""


def test_parse_normalized_document_recovers_all_facts():
    facts = parse_normalized_document("T0000001.md", SAMPLE_MD)

    assert facts.pcode == "T0000001"
    assert facts.law_name == "測試條例"
    assert facts.metadata["法規位階"] == "法律"
    assert facts.metadata["廢止狀態"] == "已廢止"
    assert facts.is_repealed is True
    assert facts.metadata["廢止註記"].startswith("本條例自")
    assert facts.has_foreword is True
    assert facts.sections == ("第 一 章 總則｜第 1 條", "第 2 條")
    assert facts.article_count == 2


def test_foreword_detection_distinguishes_metadata_from_prose():
    """metadata 清單本身不得被誤判為前言。"""
    without = SAMPLE_MD.replace("本條例依測試法第一條之規定制定之。\n\n", "")
    facts = parse_normalized_document("T0000001.md", without)

    assert facts.has_foreword is False
    assert facts.metadata  # metadata 仍完整解析
    assert facts.sections  # 條文仍完整解析


def test_active_law_is_not_flagged_repealed():
    facts = parse_normalized_document("T0000001.md", SAMPLE_MD.replace("已廢止", "現行"))
    assert facts.is_repealed is False


def test_collect_documents_is_sorted_and_complete(documents):
    assert [d.pcode for d in documents] == sorted(FIXTURE_ARTICLES)
    assert {d.pcode: d.article_count for d in documents} == FIXTURE_ARTICLES
    # 只有 A0000005（動員戡亂時期臨時條款）有前言
    assert [d.pcode for d in documents if d.has_foreword] == ["A0000005"]


# ---------------------------------------------------------------------------
# 2. 百分位 / 分布（nearest-rank，不插值）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pct,expected",
    [(50, 5), (90, 9), (95, 10), (99, 10), (100, 10)],
)
def test_percentile_is_nearest_rank(pct, expected):
    values = list(range(1, 11))  # 1..10
    assert percentile(values, pct) == expected


def test_percentile_returns_a_real_corpus_value():
    """nearest-rank 保證結果一定是語料中真實存在的值（不插值）。"""
    values = [3, 3, 100]
    assert percentile(values, 50) in values
    assert percentile(values, 99) in values


def test_percentile_handles_empty_input():
    assert percentile([], 50) is None
    assert distribution([])["count"] == 0
    assert distribution([])["p50"] is None


def test_distribution_is_order_independent():
    a = distribution([5, 1, 9, 3])
    b = distribution([9, 3, 5, 1])
    assert a == b
    assert a["min"] == 1 and a["max"] == 9 and a["count"] == 4
    assert a["mean"] == 4.5


# ---------------------------------------------------------------------------
# 3. identity / metadata 完整度
# ---------------------------------------------------------------------------
def test_identity_stats_on_fixture(documents):
    stats = identity_stats(documents)
    assert stats["unique_pcodes"] == TOTAL_LAWS
    assert stats["unique_law_names"] == TOTAL_LAWS
    assert stats["duplicate_pcode_count"] == 0
    assert stats["duplicate_law_name_count"] == 0


def test_identity_stats_detects_duplicates():
    docs = [
        parse_normalized_document("T0000001.md", SAMPLE_MD),
        parse_normalized_document("T0000002.md", SAMPLE_MD),  # 同名、不同 pcode
    ]
    stats = identity_stats(docs)
    assert stats["duplicate_law_name_count"] == 1
    assert stats["duplicate_pcode_count"] == 0


def test_metadata_stats_on_fixture(documents):
    stats = metadata_stats(documents)
    assert stats["document_count"] == TOTAL_LAWS
    # normalize.py 一律寫出全部六個欄位
    assert stats["field_incomplete"] == []
    assert sorted(stats["field_complete"]) == sorted(METADATA_KEYS)
    assert stats["all_fields_complete_count"] == TOTAL_LAWS
    assert stats["laws_with_foreword"] == 1
    assert stats["laws_without_foreword"] == TOTAL_LAWS - 1
    assert stats["repealed_law_count"] == 3  # fixture 中 3 部已廢止
    assert stats["repealed_with_abandon_note"] <= stats["repealed_law_count"]


def test_metadata_stats_reports_incomplete_fields():
    broken = SAMPLE_MD.replace("- 法規位階：法律\n", "")
    docs = [parse_normalized_document("T0000001.md", broken)]
    stats = metadata_stats(docs)

    assert "法規位階" in stats["field_incomplete"]
    assert stats["all_fields_complete_count"] == 0


# ---------------------------------------------------------------------------
# 4. 混淆性（確定性規則）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("內政部警政署組織條例", "內政部警政署"),
        ("內政部警政署組織法", "內政部警政署"),
        ("戶籍法施行細則", "戶籍法"),
        ("法院組織法", "法院"),
        ("憲法", "憲法"),  # 剝除後為空 -> 保留原字串
    ],
)
def test_title_stem_strips_a_single_legal_form_suffix(name, expected):
    assert title_stem(name) == expected


def test_confusability_detects_stem_collisions_and_mixed_status():
    def doc(pcode, name, repealed):
        md = SAMPLE_MD.replace("測試條例", name).replace(
            "已廢止", "已廢止" if repealed else "現行"
        )
        return parse_normalized_document(f"{pcode}.md", md)

    docs = [
        doc("T0000001", "內政部警政署組織條例", True),
        doc("T0000002", "內政部警政署組織法", False),
        doc("T0000003", "完全無關的法律", False),
    ]
    stats = confusability_stats(docs)

    assert stats["identical_title_count"] == 0
    assert stats["title_stem_collision_group_count"] == 1
    assert stats["title_stem_collision_law_count"] == 2
    assert stats["title_stem_collision_groups"]["內政部警政署"] == ["T0000001", "T0000002"]
    # 一現行一已廢止 -> 現行／已廢止交錯
    assert stats["active_repealed_confuser_group_count"] == 1


def test_confusability_detects_identical_titles():
    docs = [
        parse_normalized_document("T0000001.md", SAMPLE_MD),
        parse_normalized_document("T0000002.md", SAMPLE_MD),
    ]
    stats = confusability_stats(docs)

    assert stats["identical_title_count"] == 1
    assert stats["identical_title_groups"]["測試條例"] == ["T0000001", "T0000002"]


def test_confusability_groups_are_sorted_and_exclude_singletons(documents):
    stats = confusability_stats(documents)
    for groups in (
        stats["identical_title_groups"],
        stats["title_stem_collision_groups"],
        stats["active_repealed_confuser_groups"],
    ):
        for members in groups.values():
            assert members == sorted(members)
            assert len(members) > 1


# ---------------------------------------------------------------------------
# 5. chunk 統計（走既有管線，注入 fake tokenizer）
# ---------------------------------------------------------------------------
def test_chunk_stats_classifies_header_and_article_chunks(snapshot_dir, documents):
    stats = chunk_stats(
        snapshot_dir, documents, count_tokens=char_tokens, chunk_config=PILOT_CHUNK_CONFIG
    )

    # 每部法規恰好一個「法規標頭 chunk」（metadata + 前言合併），其餘為條文 chunk
    assert stats["law_header_chunk_count"] == TOTAL_LAWS
    assert stats["article_chunk_count"] == TOTAL_ARTICLES
    assert stats["total_chunk_count"] == TOTAL_ARTICLES + TOTAL_LAWS
    assert stats["law_header_chunk_share_pct"] == round(
        100.0 * TOTAL_LAWS / (TOTAL_ARTICLES + TOTAL_LAWS), 2
    )
    assert stats["law_header_with_foreword"] == 1
    assert stats["law_header_metadata_only"] == TOTAL_LAWS - 1
    assert stats["article_section_count"] == TOTAL_ARTICLES
    assert stats["strategy_fallback_count"] == 0


def test_chunk_stats_distributions_are_consistent(snapshot_dir, documents):
    stats = chunk_stats(
        snapshot_dir, documents, count_tokens=char_tokens, chunk_config=PILOT_CHUNK_CONFIG
    )
    assert stats["chunk_char_length"]["count"] == stats["total_chunk_count"]
    assert stats["chunk_token_length"]["count"] == stats["total_chunk_count"]
    assert stats["chunks_per_law"]["count"] == TOTAL_LAWS
    assert stats["chunk_char_length"]["min"] <= stats["chunk_char_length"]["p50"]
    assert stats["chunk_char_length"]["p50"] <= stats["chunk_char_length"]["max"]


def test_no_chunk_exceeds_the_token_limit(snapshot_dir, documents):
    """token 上限強制必須生效：切分後不得有任何 chunk 超標。"""
    stats = chunk_stats(
        snapshot_dir, documents, count_tokens=char_tokens, chunk_config=PILOT_CHUNK_CONFIG
    )
    assert stats["chunks_over_token_limit"] == 0
    assert stats["chunk_token_length"]["max"] <= PILOT_CHUNK_CONFIG["max_tokens"]


def test_hard_token_split_counts_articles_that_exceed_the_limit(snapshot_dir, documents):
    """壓低 max_tokens 後，超長條文必須被計為 hard token split。

    定義是「條文全文本身即超過 max_tokens」——這由語料單獨決定，故可重現。
    """
    relaxed = chunk_stats(
        snapshot_dir, documents, count_tokens=char_tokens,
        chunk_config={**PILOT_CHUNK_CONFIG, "max_tokens": 480},
    )
    strict = chunk_stats(
        snapshot_dir, documents, count_tokens=char_tokens,
        chunk_config={**PILOT_CHUNK_CONFIG, "max_tokens": 100},
    )

    assert relaxed["hard_token_split_count"] == 0
    assert strict["hard_token_split_count"] > 0
    assert 0 < strict["hard_token_split_rate_pct"] <= 100
    # 上限壓低後仍不得有 chunk 超標（強制切分確實執行）
    assert strict["chunks_over_token_limit"] == 0
    assert strict["chunk_token_length"]["max"] <= 100
    # 超長條文必然被切成多個 chunk
    assert strict["chunks_per_article"]["max"] > 1


# ---------------------------------------------------------------------------
# 6. 組合輸出與決定性
# ---------------------------------------------------------------------------
def test_build_stats_is_deterministic(snapshot_dir):
    first = build_stats(snapshot_dir, count_tokens=char_tokens)
    second = build_stats(snapshot_dir, count_tokens=char_tokens)
    assert hashing.canonical_json(first) == hashing.canonical_json(second)


def test_build_stats_agrees_with_the_manifest(snapshot_dir):
    import json

    stats = build_stats(snapshot_dir, count_tokens=char_tokens)
    with open(os.path.join(snapshot_dir, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert stats["snapshot_id"] == manifest["snapshot_id"]
    assert stats["corpus_hash"] == manifest["corpus_hash"]
    assert stats["scale"]["article_count"] == manifest["article_count"] == TOTAL_ARTICLES
    assert stats["scale"]["selected_law_count"] == manifest["selected_record_count"]
    assert stats["scale"]["repealed_law_count"] + stats["scale"]["active_law_count"] == TOTAL_LAWS
    # 條文 chunk 數必須等於 manifest 的條文數（每條至少一個 chunk）
    assert stats["chunks"]["article_chunk_count"] >= manifest["article_count"]


def test_build_stats_runtime_is_optional_and_sorted(snapshot_dir):
    assert build_stats(snapshot_dir, count_tokens=char_tokens)["runtime"] == {}
    with_runtime = build_stats(
        snapshot_dir, count_tokens=char_tokens,
        runtime={"index_ms": 2, "fetch_ms": 1},
    )
    assert list(with_runtime["runtime"]) == ["fetch_ms", "index_ms"]


def test_rendered_markdown_has_no_absolute_paths(snapshot_dir):
    stats = build_stats(snapshot_dir, count_tokens=char_tokens)
    markdown = render_markdown(stats)

    assert stats["snapshot_id"] in markdown
    assert "hard token split" in markdown
    assert snapshot_dir not in markdown
    assert ":\\" not in markdown
    assert hashing.repo_root() not in markdown
