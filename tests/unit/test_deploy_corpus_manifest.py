"""Target Corpus Manifest 測試（Slice 7 §十一）。"""
from __future__ import annotations

from evaluation.v11.deploy.corpus_manifest import CORPUS_HASH_VERSION, build_corpus_manifest


def _corpus(tmp_path, name, content):
    d = tmp_path / name
    d.mkdir()
    (d / "a.md").write_text(content, encoding="utf-8")
    return str(d)


def test_manifest_fields_and_no_absolute_path(tmp_path):
    c = _corpus(tmp_path, "corpus", "# E900\n\n內容\n")
    m = build_corpus_manifest((c,), base_dir=str(tmp_path))
    d = m.to_dict()
    assert d["corpus_hash_version"] == CORPUS_HASH_VERSION
    assert d["document_count"] == 1
    assert d["canonical_text_version"] == "v1" and d["loader_version"] == "v1"
    # 檔案清單只含穩定相對路徑 + sha256，不含絕對路徑 / mtime（§十一.2）。
    for doc in d["documents"]:
        assert set(doc) == {"path", "sha256"}
        assert ":" not in doc["path"] and "\\" not in doc["path"]


def test_same_content_same_hash_diff_content_diff_hash(tmp_path):
    c1 = _corpus(tmp_path, "c1", "same content\n")
    c2 = _corpus(tmp_path, "c2", "same content\n")
    c3 = _corpus(tmp_path, "c3", "DIFFERENT content\n")
    h1 = build_corpus_manifest((c1,), base_dir=str(tmp_path / "c1")).corpus_hash
    h2 = build_corpus_manifest((c2,), base_dir=str(tmp_path / "c2")).corpus_hash
    h3 = build_corpus_manifest((c3,), base_dir=str(tmp_path / "c3")).corpus_hash
    # 相同檔名 + 相同內容 → 相同 hash（與絕對路徑無關）。
    assert h1 == h2
    # 相同檔名但內容不同 → 不同 hash（§十一.4）。
    assert h1 != h3
