"""Slice C —— 外部法規 corpus 接上既有 pipeline 的端到端整合測試。

本檔要回答的**唯一**問題：Slice B 產出的 snapshot，能不能直接被既有管線吃下去，
而且 ``app/**`` 與 ``evaluation/v11/**`` **一行都不用改**？

    fixture ZIP（記憶體）
      -> evaluation.corpus.snapshot.build_snapshot        （Slice B）
      -> normalized/*.md
      -> evaluation.v11.hashing.collect_corpus_files      （既有）
      -> app.services.document_loader.load_document       （既有）
      -> app.services.canonical_builder                   （既有）
      -> evaluation.v11.chunking（current strategy）      （既有）
      -> app.services.embedding_service                   （既有）
      -> app.services.vector_store -> 真實 Qdrant         （既有）
      -> evaluation.v11.retrieval（dense / identifier_aware / rerank）（既有）

本檔**不是**品質評測，**不**建立 evaluation dataset，**不**設定 metric threshold，
**不**碰 MOJ 網路，**不**呼叫 LLM。

隔離：只建立、只清理自己 ``optimizer_`` 前綴的 collection；對整個 Qdrant namespace
不做任何斷言（見 ``tests/integration/conftest.py``）。正式 ``factory_knowledge``
以 before/after 快照語意保護，**不**斷言任何固定 point count。
"""
from __future__ import annotations

import os

import pytest

from tests.integration.conftest import (
    PROD_COLLECTION,
    drop_collections,
    list_collections,
    prod_point_count,
    unique_token,
    wait_collection_absent,
)

pytestmark = pytest.mark.integration

# --------------------------------------------------------------------------
# 設定
# --------------------------------------------------------------------------
#: 計畫 §7.4 / L2 的 pilot 值。刻意由**測試**注入既有 ChunkingConfig，而不是塞進
#: corpus-moj-law-pilot.json —— 那份 config 的責任是 source / fetch / snapshot，
#: 與 chunking / retrieval 是兩份不相關的契約，不該為了少一個檔案而混在一起。
CHUNK_CONFIG = {
    "strategy": "current",
    "target_chars": 450,
    "overlap_chars": 80,
    "max_tokens": 480,
}
EMBEDDING_MAX_TOKENS = 480

TOP_K = 5
DENSE = {
    "strategy": "dense", "top_k": TOP_K, "score_threshold": 0.55,
    "candidate_multiplier": 3, "candidate_cap": 50,
}
DENSE_RERANK = {**DENSE, "rerank_strategy": "cross_encoder", "rerank_top_n": 5}
IDENTIFIER_AWARE = {
    "strategy": "identifier_aware", "top_k": TOP_K, "score_threshold": 0.55,
    "candidate_multiplier": 3, "candidate_cap": 50, "identifier_candidate_cap": 20,
    "exact_match_boost": 0.2, "confusable_penalty": 0.15, "identifier_vector_floor": 0.4,
}
RERANK_MODEL = "BAAI/bge-reranker-base"

#: 每部 fixture 法規的預期結構：``filename -> (type-A 條文數, chunk 數)``。
#: chunk 數 = 條文數 + 1，多出來的那一個是 chunk_index 0 的「法規標頭 chunk」
#: （法規名稱 heading + metadata 清單 + 前言，全部落在 section == 法規名稱）。
EXPECTED = {
    "A0000005.md": (11, 12),  # 動員戡亂時期臨時條款：有前言、ArticleNo 為純數字
    "A0010013.md": (7, 8),    # 國策顧問委員會組織條例
    "F0060007.md": (12, 13),  # 陸海空軍服制條例：有 第 9-1 條、3 個附件
    "G0340074.md": (5, 6),    # 遺產稅法：巢狀編章節
}
EXPECTED_ARTICLE_CHUNKS = sum(a for a, _ in EXPECTED.values())      # 35
EXPECTED_TOTAL_CHUNKS = sum(c for _, c in EXPECTED.values())        # 39
EXPECTED_HEADER_CHUNKS = len(EXPECTED)                             # 4

#: fixture 中出現的編章節標題。它們**不得**成為獨立 section／獨立 evidence，
#: 只能作為其後條文 section 名稱的前綴（計畫 §5.5 / C1）。
CHAPTER_TITLES = ("第 一 章 總則", "第 四 章 稽徵程序", "第 一 節 申報", "第 二 節 核稅")
CHAPTER_SEPARATOR = "｜"

#: A0000005 的前言片段（只存在於前言，不存在於任何條文）。
FOREWORD_FRAGMENT = "第一百七十四條第一款"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def qdrant_url():
    """真實 Qdrant；同時強制離線，確保絕不下載模型。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        from qdrant_client import QdrantClient

        probe = QdrantClient(url=url)
        probe.get_collections()
        probe.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"真實 Qdrant 未就緒（{url}）：{exc}")
    return url


@pytest.fixture(scope="session")
def cached_embedding():
    """v1.3 起為 OpenAI-compatible Embedding API client。

    建構不需連線；``count_tokens`` 為本地計算，chunking 類測試不需 API。
    需要真正產生向量的測試請改用 ``api_embedding``（未設定 API 時 skip）。
    """
    try:
        from app.core.config import Settings
        from app.services.embedding_service import EmbeddingService

        settings = Settings(app_env="optimizer")
        service = EmbeddingService(
            model_name=settings.embedding_model,
            query_instruction=settings.embedding_query_instruction,
            max_tokens=settings.embedding_max_tokens,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            dimension=settings.embedding_dimension,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"無法建立 EmbeddingService：{exc}")
    return service


@pytest.fixture(scope="session")
def api_embedding(cached_embedding):
    """需要真實向量的測試使用；未設定 EMBEDDING_BASE_URL / EMBEDDING_API_KEY 即 skip。"""
    from app.core.config import Settings

    settings = Settings(app_env="optimizer")
    if not (settings.embedding_base_url and settings.embedding_api_key):
        pytest.skip("未設定 EMBEDDING_BASE_URL / EMBEDDING_API_KEY（v1.3 Embedding API）")
    return cached_embedding


@pytest.fixture(scope="session")
def cached_reranker():
    """本機**已快取**的 cross-encoder；未快取即 skip（絕不下載）。"""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from evaluation.v11.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(RERANK_MODEL)
        reranker._ensure_model()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"本機未快取 rerank 模型 {RERANK_MODEL}（不下載）：{exc}")
    return reranker


@pytest.fixture(scope="session")
def snapshot(tmp_path_factory):
    """以 Slice A fixture + Slice B builder 建一份 snapshot（完全 offline）。

    建在 pytest tmp_path，**不**寫入正式的 ``evaluation/corpora/moj-law``。
    """
    from evaluation.corpus.moj_law import SourceConfig
    from evaluation.corpus.snapshot import CorpusConfig, build_snapshot
    from tests.unit.test_corpus_moj_law import fixture_zip

    config = CorpusConfig(
        provider="moj_law",
        source=SourceConfig(max_laws=None, include_repealed=True, order_by="pcode"),
        output_root="evaluation/corpora/moj-law",
    )
    return build_snapshot(
        config,
        output_root=str(tmp_path_factory.mktemp("corpora")),
        raw_zip=fixture_zip(),
    )


@pytest.fixture(scope="session")
def corpus_files(snapshot):
    """經**既有** ``collect_corpus_files`` 列舉出的 ``(basename, abs_path)``。"""
    from evaluation.v11 import hashing

    normalized = os.path.join(snapshot.path, "normalized")
    return [
        (os.path.basename(rel), abs_path)
        for rel, abs_path in hashing.collect_corpus_files([normalized], snapshot.path)
    ]


@pytest.fixture(scope="session", autouse=True)
def production_collection_guard(qdrant_url):
    """正式 collection 的 before/after 快照語意（計畫 §8 / Owner Decision Q3）。

    存在 → 結束時必須完全相同；不存在 → 結束時必須仍不存在。
    **不**斷言任何固定 point count，**不**對整個 namespace 做斷言。
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    before_exists = PROD_COLLECTION in list_collections(client)
    before_count = prod_point_count(client)
    try:
        yield {"exists": before_exists, "point_count": before_count}
    finally:
        after_exists = PROD_COLLECTION in list_collections(client)
        after_count = prod_point_count(client)
        client.close()
        assert after_exists == before_exists, (
            f"正式 collection 存在狀態改變：before={before_exists} after={after_exists}"
        )
        assert after_count == before_count, (
            f"正式 collection point 數改變：before={before_count} after={after_count}"
        )


def _trial_config(name, retrieval):
    from evaluation.v11.run_models import TrialConfig

    return TrialConfig.from_raw(
        {"trial_name": name, "chunk_config": CHUNK_CONFIG, "retrieval_config": retrieval},
        0,
    )


def _make_provider(qdrant_url, workdir, embedding, reranker=None, tag="slicec"):
    from evaluation.v11.real_services import RealTrialServiceProvider

    return RealTrialServiceProvider.create(
        qdrant_url,
        os.path.join(str(workdir), f"{tag}-work"),
        embedding=embedding,
        cache_db_path=os.path.join(str(workdir), f"{tag}.embedding-cache.db"),
        rerank_cache_db_path=os.path.join(str(workdir), f"{tag}.rerank-cache.db"),
        reranker=reranker,
    )


@pytest.fixture(scope="session")
def indexed(qdrant_url, tmp_path_factory, api_embedding, cached_reranker, corpus_files):
    """建一次索引供多個唯讀測試共用；結束時只刪自己建立的 collection。"""
    workdir = tmp_path_factory.mktemp("indexed")
    provider = _make_provider(qdrant_url, workdir, api_embedding, cached_reranker, tag="idx")
    collection = f"optimizer_slicec_{unique_token()}"
    try:
        trial = _trial_config("slice-c", DENSE)
        outcome = provider.build_index(collection, trial, corpus_files)
        yield {"provider": provider, "collection": collection,
               "trial": trial, "outcome": outcome}
    finally:
        drop_collections(provider._client, [collection])
        provider.close()


def _canonical(name, abs_path):
    """走**既有** loader + canonical builder，不做任何客製。"""
    import hashlib

    from app.services.canonical_builder import build_canonical_document
    from app.services.document_loader import load_document

    with open(abs_path, "rb") as handle:
        raw = handle.read()
    elements = load_document(name, raw)
    canonical = build_canonical_document(
        filename=name,
        document_sha256=hashlib.sha256(raw).hexdigest(),
        document_type="markdown",
        elements=elements,
    )
    return elements, canonical


def _chunks(name, abs_path, count_tokens):
    """走**既有** chunking strategy。"""
    import hashlib

    from evaluation.v11.chunking import ChunkingConfig, ChunkingContext, build_strategy

    with open(abs_path, "rb") as handle:
        raw = handle.read()
    elements, canonical = _canonical(name, abs_path)
    cfg = ChunkingConfig.from_raw(CHUNK_CONFIG)
    ctx = ChunkingContext(
        document_id="slice-c",
        filename=name,
        file_hash=hashlib.sha256(raw).hexdigest(),
        canonical=canonical,
        elements=elements,
        config=cfg,
        count_tokens=count_tokens,
    )
    return build_strategy(cfg.strategy).chunk(ctx)


# --------------------------------------------------------------------------
# 1. snapshot -> collect_corpus_files
# --------------------------------------------------------------------------
def test_snapshot_is_consumable_as_a_corpus_path(snapshot, corpus_files):
    """snapshot 的 ``normalized/`` 可直接當成 ``corpus_paths`` 交給既有列舉器。"""
    from evaluation.v11 import hashing

    assert sorted(name for name, _ in corpus_files) == sorted(EXPECTED)
    # 檔名即 <pcode>.md（ASCII、跨 OS 安全、與唯一識別碼一致）
    for name, _ in corpus_files:
        assert name.endswith(".md")
        assert name[:-3].isalnum()

    # manifest 可讀，且 corpus_hash 確實對得上這份 normalized corpus
    normalized = os.path.join(snapshot.path, "normalized")
    assert snapshot.manifest.corpus_hash == hashing.corpus_hash([normalized], snapshot.path)
    assert snapshot.manifest.normalized_document_count == len(corpus_files)
    assert snapshot.manifest.article_count == EXPECTED_ARTICLE_CHUNKS


# --------------------------------------------------------------------------
# 2/3. loader + canonical builder：條文 identity 與編章節處理
# --------------------------------------------------------------------------
def test_canonical_documents_expose_traceable_article_identity(corpus_files):
    """每個 type-``A`` 條文都成為可追溯的獨立 section。"""
    for name, path in corpus_files:
        elements, canonical = _canonical(name, path)
        assert canonical.segments, f"{name} 未產生任何 canonical segment"

        sections = {s.section for s in canonical.segments}
        law_name = elements[0].text.strip()
        article_sections = sections - {law_name}
        expected_articles, _ = EXPECTED[name]
        assert len(article_sections) == expected_articles, (
            f"{name} 的條文 section 數 {len(article_sections)} != 預期 {expected_articles}"
        )
        # 每個 segment 都掛著 section，且 filename 一路保留
        assert all(s.section for s in canonical.segments)
        assert canonical.filename == name


def test_chapter_headings_never_become_standalone_evidence(corpus_files):
    """編章節只能是條文 section 的前綴，**永不**自成獨立 section／element。

    這是計畫 C1 的雙向斷言：(a) 沒有任何 section 等同於某個編章節標題；
    (b) 該編章節的 context 仍完整保留在其後條文的 section 名稱中。
    """
    all_sections: set[str] = set()
    for name, path in corpus_files:
        _elements, canonical = _canonical(name, path)
        all_sections |= {s.section for s in canonical.segments}

    for title in CHAPTER_TITLES:
        # (a) 不得有任何 section 就等於編章節標題本身
        assert title not in all_sections, f"編章節 {title!r} 變成了獨立 section"
        # (b) context 必須保留在條文 section 中，且以分隔符與條號相連
        carriers = [s for s in all_sections if title in s]
        assert carriers, f"編章節 {title!r} 的 context 遺失"
        assert all(CHAPTER_SEPARATOR in s for s in carriers)

    # 巢狀章節：章與節必須同時出現在同一個條文 section 內
    nested = [s for s in all_sections if "第 四 章 稽徵程序" in s and "第 一 節 申報" in s]
    assert nested, "巢狀編章節的完整路徑未被保留"


# --------------------------------------------------------------------------
# 4. metadata / foreword 的最終 chunk 行為（freeze 實際觀察到的行為）
# --------------------------------------------------------------------------
def test_metadata_and_foreword_collapse_into_one_law_header_chunk(
    corpus_files, cached_embedding
):
    """**freeze 觀察結果**：metadata 與 foreword 不各自成 chunk。

    ``# {法規名稱}`` heading、metadata 清單、前言段落三者的 ``section`` 都是法規
    名稱，``TextSplitter`` 依 ``(page, section)`` 分組後它們合併成**單一**
    ``chunk_index == 0`` 的「法規標頭 chunk」。

    因此每部法規的 chunk 數 = 條文數 + 1，且這個標頭 chunk 是**可被 retrieval
    命中的獨立 chunk**（見 ``test_retrieval_sanity_probes``）。此處只 freeze 行為，
    **不**修改 ``normalize.py`` / ``TextSplitter``，也不刪除 metadata / foreword。
    """
    for name, path in corpus_files:
        outcome = _chunks(name, path, cached_embedding.count_tokens)
        expected_articles, expected_chunks = EXPECTED[name]
        assert len(outcome.chunks) == expected_chunks, (
            f"{name} chunk 數 {len(outcome.chunks)} != 預期 {expected_chunks}"
        )

        header = outcome.chunks[0]
        assert header.chunk_index == 0
        # 標頭 chunk 的 section 就是法規名稱，且確實含 metadata 清單
        assert header.section == name[:-3] or "法規名稱：" in header.text
        assert "- 法規名稱：" in header.text
        assert "- 廢止狀態：" in header.text
        # 其餘 chunk 一律是條文，且不含 metadata 清單
        article_chunks = outcome.chunks[1:]
        assert len(article_chunks) == expected_articles
        assert all("- 法規名稱：" not in c.text for c in article_chunks)

    # 前言：只出現在該法規的標頭 chunk 內，不另成 chunk、也不混進任何條文
    foreword_doc = "A0000005.md"
    path = dict(corpus_files)[foreword_doc]
    outcome = _chunks(foreword_doc, path, cached_embedding.count_tokens)
    carriers = [c for c in outcome.chunks if FOREWORD_FRAGMENT in c.text]
    assert len(carriers) == 1, "前言必須只落在一個 chunk 內"
    assert carriers[0].chunk_index == 0, "前言被合併進法規標頭 chunk（非獨立 chunk）"


# --------------------------------------------------------------------------
# 5/6. chunking 契約：token 上限與 identity
# --------------------------------------------------------------------------
def test_all_chunks_respect_the_embedding_token_limit(corpus_files, cached_embedding):
    """pilot chunk 設定下，所有 chunk 都在 embedding token 上限內。

    這只證明 ``target_chars=450`` 在此語料**能正常工作**；**不**主張它優於 600。
    真正的 hard-split 比例與 corpus-scale statistics 屬 Slice D。
    """
    total = 0
    for name, path in corpus_files:
        outcome = _chunks(name, path, cached_embedding.count_tokens)
        total += len(outcome.chunks)
        for chunk in outcome.chunks:
            tokens = cached_embedding.count_tokens(chunk.text)
            assert tokens <= EMBEDDING_MAX_TOKENS, (
                f"{name} chunk#{chunk.chunk_index} 超過 token 上限：{tokens}"
            )
        # 此 fixture 規模下不應觸發 strategy fallback
        assert outcome.fallback_count == 0, f"{name} 觸發 fallback：{outcome.fallback_reasons}"
    assert total == EXPECTED_TOTAL_CHUNKS


def test_chunk_identity_preserves_law_and_article_provenance(corpus_files, cached_embedding):
    """``(filename, section, chunk_index)`` 足以還原法規與條文脈絡。

    不使用 runtime ``document_id`` 作 identity —— 它每次建索引都是新的 uuid4，
    跨 rebuild 沒有意義。
    """
    seen: set[tuple[str, str, int]] = set()
    for name, path in corpus_files:
        outcome = _chunks(name, path, cached_embedding.count_tokens)
        for chunk in outcome.chunks:
            assert chunk.filename == name
            assert chunk.section, f"{name} chunk#{chunk.chunk_index} 缺少 section"
            key = (chunk.filename, chunk.section, chunk.chunk_index)
            assert key not in seen, f"chunk identity 重複：{key}"
            seen.add(key)
            # filename -> pcode（法規 identity）
            assert name[:-3] == chunk.filename[:-3]
    assert len(seen) == EXPECTED_TOTAL_CHUNKS


# --------------------------------------------------------------------------
# 7. 真實 Qdrant 索引
# --------------------------------------------------------------------------
def test_snapshot_indexes_into_real_qdrant(indexed):
    """snapshot 的 normalized 目錄能完成 chunk -> embed -> Qdrant upsert。"""
    outcome = indexed["outcome"]
    assert outcome.chunk_count == EXPECTED_TOTAL_CHUNKS
    assert outcome.point_count == outcome.chunk_count, "Qdrant point 數必須等於 chunk 數"
    assert outcome.strategy == "current"
    assert outcome.stats["strategy_fallback_count"] == 0
    assert outcome.stats["chunk_config"]["target_chars"] == 450
    assert outcome.stats["chunk_config"]["overlap_chars"] == 80
    assert outcome.stats["chunk_config"]["max_tokens"] == 480


# --------------------------------------------------------------------------
# 8. Qdrant payload 契約
# --------------------------------------------------------------------------
def test_qdrant_payload_contract_survives_the_external_corpus(indexed):
    """外部語料不得破壞既有 payload 契約；且可由 payload 還原法規 + 條文脈絡。"""
    chunks = indexed["provider"].index_manifest(indexed["collection"])
    assert len(chunks) == EXPECTED_TOTAL_CHUNKS

    identities = set()
    header_count = 0
    chapter_carrying = 0
    for chunk in chunks:
        assert chunk.text, "payload 必須保留 chunk 文字"
        assert chunk.section, "payload 必須保留 section"
        assert chunk.document_sha256, "payload 必須保留 file_hash"
        assert chunk.canonical_text_version, "canonical_text_version 不得遺失"
        assert isinstance(chunk.identifiers, list)
        assert isinstance(chunk.provenance_spans, list) and chunk.provenance_spans
        identities.add((chunk.document_sha256, chunk.section, chunk.chunk_index))
        if chunk.chunk_index == 0:
            header_count += 1
        if CHAPTER_SEPARATOR in chunk.section:
            chapter_carrying += 1

    assert len(identities) == EXPECTED_TOTAL_CHUNKS, "chunk identity 必須唯一"
    assert header_count == EXPECTED_HEADER_CHUNKS
    assert chapter_carrying > 0, "編章節 context 必須進到 Qdrant payload 的 section"


def test_retrieval_results_carry_filename_and_section(indexed):
    """retrieval 結果本身也必須帶著 filename / section，才能還原條文出處。"""
    provider, collection, trial = (
        indexed["provider"], indexed["collection"], indexed["trial"]
    )
    out = provider.retrieve(collection, trial, "現役軍人穿著軍常服的時機為何", TOP_K)

    assert out.results
    for result in out.results:
        assert result.filename in EXPECTED
        assert result.section
        assert result.text
        # 由 filename + section 即可還原：法規 identity + 條文/章節脈絡
        assert result.filename[:-3].isalnum()


# --------------------------------------------------------------------------
# 9. retrieval sanity probes（觀察用，非評測）
# --------------------------------------------------------------------------
def test_retrieval_sanity_probes(indexed):
    """少量 sanity probe：確認回傳合理條文。**不是**評測，無 metric threshold。

    C（前言）與 D（metadata）只作**觀察**：確認標頭 chunk 確實可被 retrieval
    命中。這是 Slice C 要記錄的既有行為，不得因此調整 retrieval 或 normalization。
    """
    provider, collection, trial = (
        indexed["provider"], indexed["collection"], indexed["trial"]
    )

    # A：以條文內容查詢 -> 命中正確法規的條文 chunk
    out = provider.retrieve(collection, trial, "現役軍人穿著軍常服的時機為何", TOP_K)
    assert out.results
    assert out.results[0].filename == "F0060007.md"
    assert out.results[0].chunk_index != 0, "應命中條文 chunk，而非法規標頭"
    assert "條" in out.results[0].section

    out = provider.retrieve(collection, trial, "遺產稅申報書類的調查及估價期限", TOP_K)
    assert out.results
    assert out.results[0].filename == "G0340074.md"
    assert CHAPTER_SEPARATOR in out.results[0].section, "應命中帶章節脈絡的條文"

    # B：法規名稱 + 條號式查詢 -> 至少落在正確的那部法規
    out = provider.retrieve(collection, trial, "陸海空軍服制條例 第 9-1 條", TOP_K)
    assert out.results
    assert out.results[0].filename == "F0060007.md"

    # C（觀察）：前言查詢 -> 標頭 chunk 可被命中
    out = provider.retrieve(
        collection, trial, "依照憲法第一百七十四條第一款程序制定動員戡亂時期臨時條款", TOP_K
    )
    assert out.results
    top = out.results[0]
    assert top.filename == "A0000005.md"
    assert top.chunk_index == 0, "前言所在的法規標頭 chunk 確實是可 retrieval 的獨立 chunk"
    assert FOREWORD_FRAGMENT in top.text

    # D（觀察）：metadata 式查詢 -> 標頭 chunk 可被命中
    out = provider.retrieve(collection, trial, "法規類別 廢止法規 財政部 法規位階 法律", TOP_K)
    assert out.results
    headers = [r for r in out.results if r.chunk_index == 0]
    assert headers, "metadata 查詢確實會命中法規標頭 chunk（已知行為，僅記錄）"
    assert "- 法規名稱：" in headers[0].text


# --------------------------------------------------------------------------
# 10. 唯一命名 collection 的建立與清理
# --------------------------------------------------------------------------
def test_unique_collection_is_created_and_fully_cleaned_up(
    qdrant_url, tmp_path, api_embedding, corpus_files
):
    """只建立、只清理自己的 collection；清理後以有限輪詢確認真的消失。"""
    provider = _make_provider(qdrant_url, tmp_path, api_embedding, tag="cleanup")
    collection = f"optimizer_sliceccleanup_{unique_token()}"
    others_before = [
        n for n in list_collections(provider._client) if n != collection
    ]
    try:
        provider.build_index(collection, _trial_config("cleanup", DENSE), corpus_files)
        assert provider._client.collection_exists(collection)
    finally:
        drop_collections(provider._client, [collection])

    wait_collection_absent(provider._client, collection)
    assert not provider._client.collection_exists(collection)
    # 不做 whole-namespace 斷言，只確認自己沒有動到別人的 collection
    others_after = [n for n in list_collections(provider._client) if n != collection]
    assert others_after == others_before
    provider.close()


# --------------------------------------------------------------------------
# 11. 正式 collection before/after
# --------------------------------------------------------------------------
def test_production_collection_is_untouched(production_collection_guard, indexed, qdrant_url):
    """索引外部語料的過程完全不觸碰正式 collection。

    採 before/after 快照語意：存在則狀態相同、不存在則仍不存在。
    **不**斷言任何固定 point count（10 只是某個時點的狀態，不是不變量）。
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    try:
        now_exists = PROD_COLLECTION in list_collections(client)
        assert now_exists == production_collection_guard["exists"]
        assert prod_point_count(client) == production_collection_guard["point_count"]
        # 本測試自己的 collection 一定不是正式名稱
        assert indexed["collection"].startswith("optimizer_")
        assert indexed["collection"] != PROD_COLLECTION
    finally:
        client.close()


# --------------------------------------------------------------------------
# 12. dense -> cross-encoder rerank wiring smoke
# --------------------------------------------------------------------------
def test_dense_to_rerank_wiring_smoke(
    qdrant_url, tmp_path, api_embedding, cached_reranker, corpus_files
):
    """rerank stage 能接受法律語料的 chunk 並產出合法結果。

    只做接線 smoke：**不**比較品質、**不**量測 latency、**不**設任何 threshold。
    """
    from evaluation.v11.retrieval.reranker import sigmoid

    provider = _make_provider(
        qdrant_url, tmp_path, api_embedding, cached_reranker, tag="rerank"
    )
    collection = f"optimizer_slicecrr_{unique_token()}"
    try:
        trial = _trial_config("rerank", DENSE_RERANK)
        provider.build_index(collection, trial, corpus_files)
        out = provider.retrieve(collection, trial, "現役軍人穿著軍常服的時機為何", TOP_K)

        assert out.meta["rerank_strategy"] == "cross_encoder"
        assert out.meta["ranking_version"] == "dense_v1+cross_encoder_v1"
        assert out.results
        assert out.meta["rerank_window_count"] + out.meta["rerank_tail_count"] == \
               out.meta["admitted_count"]

        scored = [r for r in out.results if r.rerank_applied]
        assert scored, "至少要有被 rerank 過的結果"
        raws = [r.rerank_score for r in scored]
        assert raws == sorted(raws, reverse=True), "排序主鍵必須是 raw logit"
        for result in scored:
            assert result.rerank_score_normalized == pytest.approx(
                sigmoid(result.rerank_score), rel=1e-12
            )
            assert result.filename in EXPECTED
            assert result.section
    finally:
        drop_collections(provider._client, [collection])
        provider.close()


# --------------------------------------------------------------------------
# 13. identifier-aware 在此語料上的惰性（已知 limitation L1）
# --------------------------------------------------------------------------
def test_identifier_aware_is_inert_on_this_legal_corpus(indexed):
    """**觀察並鎖住既有 limitation L1**：法律識別碼不會被 IdentifierExtractor 辨識。

    ``extract_normalized()`` 對「第 N 條」/「第 N-N 條」/ pcode 全數回傳空陣列，
    故 ``IdentifierAwareRetrievalStrategy`` 走「查詢無識別碼」路徑而委派給 dense。
    **這是既有行為的記錄，不是缺陷修復請求。**
    依 Owner Decision Q1，Slice A–D 不修改 ``IdentifierExtractor``。
    """
    from app.services.identifier_extractor import IdentifierExtractor

    extractor = IdentifierExtractor()
    for probe in ("第 9-1 條", "陸海空軍服制條例 第 9-1 條", "F0060007", "遺產稅法 第 16 條"):
        assert extractor.extract_normalized(probe) == [], (
            f"L1 已不成立：{probe!r} 現在可被辨識；需重新評估 identifier-aware 的行為"
        )

    provider, collection = indexed["provider"], indexed["collection"]
    trial = _trial_config("ida", IDENTIFIER_AWARE)
    out = provider.retrieve(collection, trial, "陸海空軍服制條例 第 9-1 條", TOP_K)

    assert out.meta["ranking_version"] == "identifier_aware_v1"
    assert out.meta["query_identifiers"] == []
    assert out.meta["identifier_candidate_count"] == 0
    assert out.results, "退化為 dense 後仍必須有結果"
    for result in out.results:
        assert result.candidate_sources == ["dense"], "候選只可能來自 dense 路徑"
        assert result.exact_match_boost == 0.0
        assert result.confusable_penalty == 0.0
        assert result.identifiers == [], "chunk 的 identifiers payload 幾乎全為空"
