"""Slice 5 §25.1 —— Grid / Random candidate 生成、conditional validation、去重、budget。"""
from __future__ import annotations

import pytest

from evaluation.v11.search.search_space import (
    ERROR_SEARCH_SPACE_EXCEEDS_BUDGET,
    GEN_DUPLICATE,
    GEN_INVALID,
    GEN_NOT_SAMPLED,
    IDENTIFIER_ONLY_FIELDS,
    SearchSpaceError,
    candidate_manifest_signature,
    generate_candidates,
)
from tests.unit.search_helpers import make_search_config

VERSIONS = {"canonical_text_version": "v1", "identifier_normalization_version": "v1"}


def gen(**overrides):
    return generate_candidates(make_search_config(**overrides), **VERSIONS)


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------
def test_grid_cartesian_product_counts():
    r = gen(search_space={"retrieval": {"top_k": [3, 5], "score_threshold": [0.5, 0.6]}})
    assert r.raw_combination_count == 4
    assert r.executable_count == 4
    assert r.invalid_count == 0
    assert r.duplicate_count == 0


def test_grid_order_is_deterministic_across_runs():
    sigs = {candidate_manifest_signature(gen(
        search_space={"retrieval": {"top_k": [3, 5, 7], "score_threshold": [0.5, 0.6]}}
    )) for _ in range(5)}
    assert len(sigs) == 1


def test_grid_order_independent_of_json_key_order():
    a = gen(search_space={"retrieval": {"top_k": [3, 5], "score_threshold": [0.5, 0.6]}})
    b = gen(search_space={"retrieval": {"score_threshold": [0.5, 0.6], "top_k": [3, 5]}})
    assert candidate_manifest_signature(a) == candidate_manifest_signature(b)


def test_grid_exceeding_budget_fails_loudly_and_does_not_truncate():
    with pytest.raises(SearchSpaceError) as exc:
        gen(
            max_candidates=3,
            search_space={"retrieval": {"top_k": [1, 2, 3, 4, 5, 6, 7, 8]}},
        )
    assert exc.value.code == ERROR_SEARCH_SPACE_EXCEEDS_BUDGET


def test_grid_exactly_at_budget_is_allowed():
    r = gen(max_candidates=4, search_space={"retrieval": {"top_k": [1, 2, 3, 4]}})
    assert r.executable_count == 4


# --------------------------------------------------------------------------
# Conditional parameters（§八）
# --------------------------------------------------------------------------
def test_dense_candidates_never_carry_identifier_fields():
    r = gen(
        search_space={
            "retrieval": {
                "strategy": ["dense", "identifier_aware"],
                "identifier_vector_floor": [0.4, 0.5],
            }
        }
    )
    for c in r.executable:
        if c.retrieval_config["strategy"] == "dense":
            for f in IDENTIFIER_ONLY_FIELDS:
                assert f not in c.retrieval_config, f"dense config 不得帶 {f}"
        else:
            assert c.retrieval_config["identifier_vector_floor"] in (0.4, 0.5)


def test_dense_projection_collapses_into_duplicates_not_silent_drop():
    """dense × 兩個 floor → 只剩一個 dense candidate，另一個明確記為 duplicate。"""
    r = gen(
        search_space={
            "retrieval": {
                "strategy": ["dense", "identifier_aware"],
                "identifier_vector_floor": [0.4, 0.5],
            }
        }
    )
    assert r.raw_combination_count == 4
    assert r.executable_count == 3  # dense×1 + identifier_aware×2
    assert r.duplicate_count == 1
    dup = [c for c in r.rejected if c.generation_status == GEN_DUPLICATE]
    assert len(dup) == 1
    assert dup[0].generation_rejection_reason.startswith("duplicate_config_hash")


def test_current_strategy_rejects_section_prefix_with_recorded_reason():
    r = gen(
        search_space={
            "chunking": {"strategy": ["current", "heading"], "prepend_section_title": [True]}
        }
    )
    invalid = [c for c in r.rejected if c.generation_status == GEN_INVALID]
    assert len(invalid) == 1
    assert invalid[0].assignment["chunking.strategy"] == "current"
    assert "prepend_section_title" in invalid[0].generation_rejection_reason
    # 無效組合不得產生可執行 candidate
    assert all(c.chunk_config["strategy"] == "heading" for c in r.executable)


def test_invalid_candidates_are_not_executable_and_keep_reason():
    r = gen(search_space={"retrieval": {"top_k": [5, 100]}, },
            baseline={"chunk_config": {"strategy": "current"},
                      "retrieval_config": {"strategy": "dense", "candidate_cap": 50}})
    # candidate_cap(50) >= top_k 必須成立 → top_k=100 違規
    invalid = [c for c in r.rejected if c.generation_status == GEN_INVALID]
    assert len(invalid) == 1
    assert "candidate_cap" in invalid[0].generation_rejection_reason
    assert invalid[0].config_hash is None


def test_current_chunking_expands_to_explicit_values_not_env():
    r = gen(search_space={"chunking": {"strategy": ["current"]}})
    cc = r.executable[0].chunk_config
    assert (cc["target_chars"], cc["overlap_chars"], cc["max_tokens"]) == (600, 100, 480)


def test_candidate_hash_matches_trial_config_hash():
    r = gen(search_space={"retrieval": {"top_k": [5]}})
    c = r.executable[0]
    assert c.to_trial_config().compute_hash(**VERSIONS) == c.config_hash


def test_no_legal_candidate_raises():
    with pytest.raises(SearchSpaceError) as exc:
        gen(
            search_space={"chunking": {"strategy": ["current"], "prepend_section_title": [True]}}
        )
    assert exc.value.code == "no_legal_candidate"


# --------------------------------------------------------------------------
# Random（§十）
# --------------------------------------------------------------------------
def _random_cfg(**kw):
    base = dict(
        search_mode="random",
        max_candidates=3,
        search_space={"retrieval": {"top_k": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}},
    )
    base.update(kw)
    return base


def test_random_same_seed_is_reproducible():
    a = candidate_manifest_signature(gen(**_random_cfg(random_seed=42)))
    b = candidate_manifest_signature(gen(**_random_cfg(random_seed=42)))
    assert a == b


def test_random_different_seed_can_differ():
    sigs = {
        candidate_manifest_signature(gen(**_random_cfg(random_seed=s)))
        for s in (1, 2, 3, 4, 5, 6, 7, 8)
    }
    assert len(sigs) > 1, "不同 seed 應能產生不同取樣結果"


def test_random_respects_budget_and_is_without_replacement():
    r = gen(**_random_cfg(random_seed=42, max_candidates=4))
    assert r.executable_count == 4
    hashes = [c.config_hash for c in r.executable]
    assert len(set(hashes)) == 4, "without replacement：不得有重複 config hash"


def test_random_with_fewer_legal_than_budget_runs_all_and_records_it():
    r = gen(
        search_mode="random",
        max_candidates=50,
        search_space={"retrieval": {"top_k": [1, 2, 3]}},
    )
    assert r.executable_count == 3
    assert r.budget_exhausted is False
    assert any("未超過 max_candidates" in n for n in r.notes)


def test_random_keeps_full_manifest_including_unsampled():
    r = gen(**_random_cfg(random_seed=42, max_candidates=3))
    manifest = r.manifest()
    assert len(manifest) == 10, "manifest 必須保留全部合法 candidate"
    not_sampled = [m for m in manifest if m["generation_status"] == GEN_NOT_SAMPLED]
    assert len(not_sampled) == 7
    assert all("not_sampled" in m["generation_rejection_reason"] for m in not_sampled)


def test_random_sampling_depends_only_on_config_hash_and_seed():
    """不得使用 process-dependent hash（PYTHONHASHSEED 之類）作為亂數來源。"""
    import subprocess
    import sys

    script = (
        "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'tests/unit');"
        "from tests.unit.search_helpers import make_search_config;"
        "from evaluation.v11.search.search_space import generate_candidates, candidate_manifest_signature;"
        "cfg=make_search_config(search_mode='random',max_candidates=3,random_seed=42,"
        "search_space={'retrieval':{'top_k':[1,2,3,4,5,6,7,8,9,10]}});"
        "r=generate_candidates(cfg,canonical_text_version='v1',identifier_normalization_version='v1');"
        "print(candidate_manifest_signature(r))"
    )
    outs = set()
    for seed_env in ("0", "1", "12345"):
        p = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            env={"PYTHONHASHSEED": seed_env, "PATH": ""},
        )
        assert p.returncode == 0, p.stderr
        outs.add(p.stdout.strip())
    assert len(outs) == 1, "取樣結果不得受 PYTHONHASHSEED 影響"
