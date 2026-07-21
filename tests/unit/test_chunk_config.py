"""Chunk Config 單元測試（v1.1 Slice 3 §二十二.1）。"""
from __future__ import annotations

import pytest

from evaluation.v11.chunking.config import ChunkConfigError, ChunkingConfig
from evaluation.v11.hashing import config_hash


def _h(raw: dict) -> str:
    return config_hash(ChunkingConfig.from_raw(raw).to_canonical_dict())


def test_valid_current():
    cfg = ChunkingConfig.from_raw({"strategy": "current"})
    assert cfg.strategy == "current"
    assert cfg.target_chars == 600 and cfg.overlap_chars == 100 and cfg.max_tokens == 480


def test_valid_fixed():
    cfg = ChunkingConfig.from_raw(
        {"strategy": "fixed", "target_chars": 400, "overlap_chars": 50, "max_tokens": 300}
    )
    assert cfg.strategy == "fixed"


def test_valid_paragraph():
    assert ChunkingConfig.from_raw({"strategy": "paragraph"}).strategy == "paragraph"


def test_valid_heading_with_prefix():
    cfg = ChunkingConfig.from_raw({"strategy": "heading", "prepend_section_title": True})
    assert cfg.prepend_section_title is True


def test_unknown_strategy_rejected():
    with pytest.raises(ChunkConfigError):
        ChunkingConfig.from_raw({"strategy": "semantic"})


def test_negative_rejected():
    for field in ("target_chars", "overlap_chars", "max_tokens",
                  "context_before_paragraphs", "context_after_paragraphs"):
        with pytest.raises(ChunkConfigError):
            ChunkingConfig.from_raw({"strategy": "fixed", field: -1})


def test_overlap_ge_target_rejected():
    with pytest.raises(ChunkConfigError):
        ChunkingConfig.from_raw({"strategy": "fixed", "target_chars": 100, "overlap_chars": 100})
    with pytest.raises(ChunkConfigError):
        ChunkingConfig.from_raw({"strategy": "fixed", "target_chars": 100, "overlap_chars": 150})


def test_max_tokens_non_positive_rejected():
    with pytest.raises(ChunkConfigError):
        ChunkingConfig.from_raw({"strategy": "fixed", "max_tokens": 0})


def test_unknown_key_rejected_not_silently_ignored():
    with pytest.raises(ChunkConfigError):
        ChunkingConfig.from_raw({"strategy": "fixed", "bogus_field": 1})


def test_inapplicable_fields_rejected():
    # current / paragraph 不支援 prefix；current 不支援 context。
    with pytest.raises(ChunkConfigError):
        ChunkingConfig.from_raw({"strategy": "current", "prepend_section_title": True})
    with pytest.raises(ChunkConfigError):
        ChunkingConfig.from_raw({"strategy": "paragraph", "prepend_section_title": True})
    with pytest.raises(ChunkConfigError):
        ChunkingConfig.from_raw({"strategy": "current", "context_after_paragraphs": 1})


def test_bool_typed_int_field_rejected():
    # target_chars=True 不得被當成 1（bool 非法）。
    with pytest.raises(ChunkConfigError):
        ChunkingConfig.from_raw({"strategy": "fixed", "target_chars": True})


def test_config_hash_deterministic_and_order_independent():
    a = _h({"strategy": "fixed", "target_chars": 600, "overlap_chars": 100})
    b = _h({"overlap_chars": 100, "target_chars": 600, "strategy": "fixed"})
    assert a == b


def test_config_hash_changes_on_any_field():
    base = {"strategy": "fixed", "target_chars": 600, "overlap_chars": 100, "max_tokens": 480}
    h0 = _h(base)
    assert _h({**base, "target_chars": 601}) != h0
    assert _h({**base, "overlap_chars": 99}) != h0
    assert _h({**base, "max_tokens": 479}) != h0
    assert _h({**base, "context_before_paragraphs": 1}) != h0
    assert _h({**base, "context_after_paragraphs": 1}) != h0
    assert _h({**base, "strategy": "paragraph"}) != h0


def test_canonical_dict_includes_prefix_format_version():
    d = ChunkingConfig.from_raw({"strategy": "heading", "prepend_section_title": True}).to_canonical_dict()
    assert d["prefix_format_version"] == "v1"


def test_canonical_dict_no_absolute_paths():
    d = ChunkingConfig.from_raw({"strategy": "fixed"}).to_canonical_dict()
    for v in d.values():
        assert not (isinstance(v, str) and ("/" in v or "\\" in v and ":" in v))


# --- Slice 3 §一：current strategy 的可重現性語意 ---


def test_current_expands_to_explicit_values_not_bare_strategy():
    # {"strategy":"current"} 必須展開為完整 chunk_config（含 target/overlap/max_tokens），
    # 不得只保存 {"strategy":"current"} 再於 runtime 偷讀 Settings/.env 補值。
    d = ChunkingConfig.from_raw({"strategy": "current"}).to_canonical_dict()
    assert d["strategy"] == "current"
    assert d["target_chars"] == 600
    assert d["overlap_chars"] == 100
    assert d["max_tokens"] == 480


def test_current_defaults_match_v10_repo_settings_defaults():
    # current = repository-defined reproducible v1.0 default；pin 對正式 Settings 的**類別預設**
    # （非 instance，避免 .env 覆寫），若任一方漂移即失敗。
    from app.core.config import Settings

    d = ChunkingConfig.from_raw({"strategy": "current"}).to_canonical_dict()
    fields = Settings.model_fields
    assert d["target_chars"] == fields["chunk_size_chars"].default
    assert d["overlap_chars"] == fields["chunk_overlap_chars"].default
    assert d["max_tokens"] == fields["embedding_max_tokens"].default


def test_current_config_ignores_env_chunk_overrides(monkeypatch):
    # 即使環境變數覆寫正式 chunk 尺寸，Optimizer 的 current strategy 仍用 repo canonical 值
    # （ChunkingConfig 不讀 Settings/.env）→ 保證可重現性。
    monkeypatch.setenv("CHUNK_SIZE_CHARS", "999")
    monkeypatch.setenv("CHUNK_OVERLAP_CHARS", "222")
    monkeypatch.setenv("EMBEDDING_MAX_TOKENS", "111")
    cfg = ChunkingConfig.from_raw({"strategy": "current"})
    assert (cfg.target_chars, cfg.overlap_chars, cfg.max_tokens) == (600, 100, 480)


def test_trial_config_persists_expanded_current_config():
    # TrialConfig 由 {"strategy":"current"} 展開後持久化：config_json 與 hash 輸入都含明確值，
    # 不會只保存 {"strategy":"current"} 再於 runtime 補值。
    import json

    from evaluation.v11.run_models import TrialConfig

    tc = TrialConfig.from_raw({"trial_name": "A", "chunk_config": {"strategy": "current"}}, 0)
    assert tc.chunk_config["target_chars"] == 600
    assert tc.chunk_config["overlap_chars"] == 100
    assert tc.chunk_config["max_tokens"] == 480

    persisted = json.loads(tc.config_json())
    assert persisted["chunk_config"]["target_chars"] == 600

    hashable = tc.hashable(canonical_text_version="v1", identifier_normalization_version="v1")
    assert hashable["chunk_config"]["overlap_chars"] == 100
