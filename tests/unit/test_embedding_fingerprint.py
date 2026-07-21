"""Embedding Fingerprint / Cache Key 單元測試（v1.1 Slice 3 §十四 / §十五 / §二十二.7）。"""
from __future__ import annotations

from evaluation.v11.embedding_fingerprint import (
    make_fingerprint,
    stable_model_identifier,
    text_sha256,
)


def _fp(**over):
    base = dict(provider="sentence_transformers", model_name="BAAI/bge-small-zh-v1.5",
               model_revision=None, dimension=512, normalize_embeddings=True, pooling_mode="cls")
    base.update(over)
    return make_fingerprint(**base)


def test_fingerprint_deterministic():
    assert _fp().fingerprint_hash() == _fp().fingerprint_hash()


def test_model_change_changes_key():
    assert _fp().cache_key("x") != _fp(model_name="BAAI/bge-large-zh").cache_key("x")


def test_dimension_change_changes_key():
    assert _fp().cache_key("x") != _fp(dimension=768).cache_key("x")


def test_normalize_change_changes_key():
    assert _fp().cache_key("x") != _fp(normalize_embeddings=False).cache_key("x")


def test_same_text_and_model_same_key():
    assert _fp().cache_key("相同文字") == _fp().cache_key("相同文字")


def test_same_model_diff_text_diff_key():
    assert _fp().cache_key("文字甲") != _fp().cache_key("文字乙")


def test_threshold_topk_identifier_version_do_not_affect_key():
    # cache key 只取決於 fingerprint + 文字；threshold / top_k / identifier version 不在其中。
    fp = _fp()
    assert fp.cache_key("同一段文字") == fp.cache_key("同一段文字")


def test_section_prefix_changes_text_changes_key():
    fp = _fp()
    assert fp.cache_key("core") != fp.cache_key("[SECTION] S1\ncore")


def test_no_api_key_or_absolute_path_in_fingerprint():
    fp = _fp(model_name=r"C:\Users\me\models\bge-small")
    d = fp.to_canonical_dict()
    assert d["model_name"] == "bge-small"
    for v in d.values():
        s = str(v)
        assert "C:\\" not in s and "/Users/" not in s


def test_revision_unknown_when_absent():
    assert _fp(model_revision=None).to_canonical_dict()["model_revision"] == "unknown"


def test_stable_model_identifier():
    assert stable_model_identifier("BAAI/bge-small-zh-v1.5") == "BAAI/bge-small-zh-v1.5"
    assert stable_model_identifier(r"C:\models\bge") == "bge"
    assert stable_model_identifier("/opt/models/bge") == "bge"


def test_text_sha256_exact_utf8():
    # 明確 UTF-8 bytes，不依賴 python hash()。
    assert text_sha256("héllo") == text_sha256("héllo")
    assert text_sha256("a") != text_sha256("b")
