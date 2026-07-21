"""Identifiers keyword payload index 整合測試（v1.1 Slice 1 §十三 Qdrant Integration）。

真實 Qdrant Server，使用**唯一暫時 collection**，驗證：
- identifiers keyword payload index 可建立。
- exact keyword match 可找回 E102；查 E102 不會 match D900。
- CX-500 query 經 IdentifierExtractor 正規化為 CX500 後可 match payload。
- 測試結束清除暫時 collection。
- 全程不觸碰、不修改正式 collection ``factory_knowledge``。

未就緒（無真實 Qdrant）時整個模組 skip 而非失敗。
"""
from __future__ import annotations

import os
import uuid

import pytest

from app.models.chunk import Chunk
from app.services.identifier_extractor import IdentifierExtractor
from app.services.vector_store import VectorStore

pytestmark = pytest.mark.integration

PROTECTED = "factory_knowledge"
_DIM = 4


@pytest.fixture
def client():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        from qdrant_client import QdrantClient

        c = QdrantClient(url=qdrant_url)
        c.get_collections()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"真實 Qdrant 未就緒（{qdrant_url}）：{exc}")
    try:
        yield c
    finally:
        c.close()


def _chunk(cid: str, identifiers: list[str], text: str) -> Chunk:
    return Chunk(
        id=cid,
        document_id="doc-x",
        filename="f.md",
        file_hash="h",
        page=None,
        section="s",
        chunk_index=0,
        text=text,
        created_at="2026-01-01T00:00:00+00:00",
        identifiers=identifiers,
        canonical_text_version="v1",
        loader_version="v1",
        identifier_normalization_version="v1",
    )


def _count_with_identifier(client, collection: str, value: str) -> int:
    from qdrant_client import models

    return client.count(
        collection_name=collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="identifiers", match=models.MatchValue(value=value))]
        ),
    ).count


def _prod_snapshot(client) -> tuple[bool, int | None]:
    """讀取正式 collection 狀態（存在與否 + point 數），供前後比對其未變。"""
    try:
        if not client.collection_exists(PROTECTED):
            return False, None
        return True, client.count(collection_name=PROTECTED).count
    except Exception:  # noqa: BLE001
        return False, None


def test_identifiers_keyword_index_exact_match(client):
    collection = f"factory_test_identifiers_{uuid.uuid4().hex[:8]}"
    assert collection != PROTECTED

    prod_before = _prod_snapshot(client)

    store = VectorStore(client, collection)
    try:
        # 建立 collection（含 identifiers keyword index）
        store.ensure_collection(_DIM)
        # 明確再建一次 identifiers keyword index（idempotent）
        store.create_keyword_index("identifiers")

        chunks = [
            _chunk("11111111-1111-1111-1111-111111111111", ["E102"], "E102 感測器通訊異常"),
            _chunk("22222222-2222-2222-2222-222222222222", ["D900"], "D900 混淆警報"),
            _chunk("33333333-3333-3333-3333-333333333333", ["CX500"], "CX500 設備保養"),
        ]
        vectors = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.4, 0.3], [0.3, 0.4, 0.1, 0.2]]
        store.upsert_chunks(chunks, vectors)

        # exact match 找回 E102（且僅一筆）
        assert _count_with_identifier(client, collection, "E102") == 1
        # 查 E102 不會 match D900
        assert _count_with_identifier(client, collection, "D900") == 1
        # 交叉確認：E102 filter 不會回傳 D900 那筆
        from qdrant_client import models

        e102_points = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="identifiers", match=models.MatchValue(value="E102"))]
            ),
            limit=10,
            with_payload=True,
        )[0]
        assert len(e102_points) == 1
        assert "E102" in e102_points[0].payload["identifiers"]
        assert "D900" not in e102_points[0].payload["identifiers"]

        # CX-500 query → 正規化 → CX500 → match payload
        normalized = IdentifierExtractor().normalize("CX-500")
        assert normalized == "CX500"
        assert _count_with_identifier(client, collection, normalized) == 1
        # 未正規化的原始查詢字串（含 hyphen）不會誤中
        assert _count_with_identifier(client, collection, "CX-500") == 0
    finally:
        try:
            client.delete_collection(collection)
        except Exception:  # noqa: BLE001
            pass

    # 暫時 collection 已清理
    assert not client.collection_exists(collection)
    # 正式 collection 未被本測試建立 / 改動
    prod_after = _prod_snapshot(client)
    assert prod_after == prod_before
