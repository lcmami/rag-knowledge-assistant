"""Identifier Extractor / Normalizer 單元測試（v1.1 Slice 1 §十三 Identifier）。"""
from __future__ import annotations

from app.services.identifier_extractor import (
    IDENTIFIER_NORMALIZATION_VERSION,
    IdentifierExtractor,
)


def ex() -> IdentifierExtractor:
    return IdentifierExtractor()


def test_e102_variants_normalize_same():
    assert ex().normalize("E102") == "E102"
    assert ex().normalize("e102") == "E102"
    assert ex().normalize("E-102") == "E102"


def test_cx500_variants_normalize_same():
    assert ex().normalize("CX-500") == "CX500"
    assert ex().normalize("CX500") == "CX500"


def test_extract_e102_from_chinese_sentence():
    assert ex().extract_normalized("設備出現 E102 警報時應如何處理？") == ["E102"]


def test_extract_e102_without_surrounding_space():
    assert ex().extract_normalized("E102警報") == ["E102"]


def test_extract_hyphen_and_lowercase():
    assert ex().extract_normalized("請參考 E-102 與 e102 說明") == ["E102"]


def test_extract_cx500_hyphen():
    assert ex().extract_normalized("CX-500 高速貼片機") == ["CX500"]


def test_multiple_codes_in_one_chunk_all_kept():
    got = ex().extract_normalized("PM-100 與 PM-1000 是不同模組，另有 D900。")
    assert got == ["PM100", "PM1000", "D900"]


def test_duplicate_codes_deduped_deterministic_order():
    # 重複代碼去重，維持首次出現順序
    got = ex().extract_normalized("E102 再次出現 E-102，接著 D900，再一次 e102")
    assert got == ["E102", "D900"]


def test_chinese_sentence_not_misdetected():
    assert ex().extract_normalized("設備維護作業應由合格技術員執行。") == []


def test_pure_year_not_misdetected():
    assert ex().extract_normalized("此文件於 2024 年發布，民國 113 年適用。") == []


def test_normalization_version_recorded():
    ids = ex().extract("E102 與 CX-500")
    assert all(i.identifier_normalization_version == IDENTIFIER_NORMALIZATION_VERSION for i in ids)
    assert ex().version == IDENTIFIER_NORMALIZATION_VERSION


def test_raw_preserved_for_display():
    ids = ex().extract("型號 CX-500")
    assert ids[0].raw_identifier == "CX-500"
    assert ids[0].normalized_identifier == "CX500"
