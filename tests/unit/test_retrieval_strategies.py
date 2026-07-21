"""Retrieval Strategy 行為單元測試（v1.1 Slice 4 §二十一.2–.7）。

以 fake backend 驗證 dense parity 邏輯、candidate union、boost/penalty、admission/threshold、
deterministic ranking、query identifier 抽取——不需真實 Qdrant / BGE。
"""
from __future__ import annotations

from app.services.identifier_extractor import IdentifierExtractor
from tests.unit.retrieval_helpers import FakeBackend, cand, run_strategy

DENSE = {"strategy": "dense", "top_k": 5, "score_threshold": 0.55,
         "candidate_multiplier": 3, "candidate_cap": 50}
IDA = {"strategy": "identifier_aware", "top_k": 5, "score_threshold": 0.55,
       "candidate_multiplier": 3, "candidate_cap": 50, "identifier_candidate_cap": 20,
       "exact_match_boost": 0.2, "confusable_penalty": 0.15, "identifier_vector_floor": 0.4}


# ============================ §21.3 Query identifier 抽取 ============================
def test_query_identifier_extraction_matches_index_time():
    ex = IdentifierExtractor()
    assert ex.extract_normalized("E102") == ["E102"]
    assert ex.extract_normalized("e102 警報") == ["E102"]
    assert ex.extract_normalized("E-102 是什麼") == ["E102"]
    assert ex.extract_normalized("CX-500 保養") == ["CX500"]
    assert ex.extract_normalized("CX500 週期") == ["CX500"]
    assert set(ex.extract_normalized("E102 與 D900 有何不同")) == {"E102", "D900"}
    assert ex.extract_normalized("設備通訊異常怎麼處理") == []  # 一般中文無代碼
    assert ex.extract_normalized("2024 年報告") == []           # 純年份不誤判


# ============================ §21.2 Dense parity 邏輯 ============================
def test_dense_threshold_rejects_below():
    backend = FakeBackend(dense=[cand("a", 0, 0.60), cand("b", 0, 0.50)])
    out = run_strategy("dense", DENSE, "任意查詢", backend)
    assert [r.document_id for r in out.results] == ["a"]
    assert out.threshold_rejected_count == 1


def test_dense_top_k_and_sort():
    ds = [cand(chr(97 + i), 0, 0.9 - i * 0.05) for i in range(8)]  # 8 個遞減
    out = run_strategy("dense", {**DENSE, "top_k": 3}, "q", FakeBackend(dense=ds))
    assert [r.rank for r in out.results] == [1, 2, 3]
    assert [round(r.vector_score, 3) for r in out.results] == [0.9, 0.85, 0.8]


def test_dense_candidate_limit_used():
    backend = FakeBackend(dense=[cand("a", 0, 0.9)])
    run_strategy("dense", DENSE, "q", backend)
    assert backend.dense_calls == [15]  # max(5, min(5*3, 50))


def test_dense_ranking_score_equals_vector_and_zero_components():
    out = run_strategy("dense", DENSE, "E102 查詢", FakeBackend(dense=[cand("a", 0, 0.7, ["E102"])]))
    r = out.results[0]
    assert r.ranking_score == r.vector_score == 0.7
    assert r.exact_match_boost == 0.0 and r.confusable_penalty == 0.0
    assert r.candidate_sources == ["dense"] and r.admission_reason == "dense_threshold"


def test_dense_unaffected_by_query_identifiers():
    ds = [cand("d900", 0, 0.70, ["D900"]), cand("e102", 0, 0.60, ["E102"])]
    with_id = run_strategy("dense", DENSE, "E102 警報", FakeBackend(dense=list(ds)))
    without_id = run_strategy("dense", DENSE, "一般查詢", FakeBackend(dense=list(ds)))
    assert [r.document_id for r in with_id.results] == [r.document_id for r in without_id.results]
    # dense 依 vector score：D900(0.70) 在 E102(0.60) 前（誠實保留混淆）。
    assert [r.document_id for r in with_id.results] == ["d900", "e102"]


# ============================ §21.4 Candidate Union ============================
def test_union_no_duplicate_distinct_chunks():
    backend = FakeBackend(dense=[cand("a", 0, 0.7)], identifier=[cand("b", 0, 0.65, ["E102"])])
    out = run_strategy("identifier_aware", IDA, "E102", backend)
    assert out.dense_candidate_count == 1 and out.identifier_candidate_count == 1
    assert out.union_candidate_count == 2 and out.deduplicated_count == 2


def test_union_same_chunk_dedup_merges_sources():
    same = lambda: cand("e102", 0, 0.60, ["E102"])
    backend = FakeBackend(dense=[same()], identifier=[same()])
    out = run_strategy("identifier_aware", IDA, "E102", backend)
    assert out.union_candidate_count == 2 and out.deduplicated_count == 1
    assert sorted(out.results[0].candidate_sources) == ["dense", "identifier"]


def test_exact_recovered_when_not_in_dense_topk():
    # E102 不在 dense（全是別的），僅由 identifier search 找回。
    dense = [cand("misc", i, 0.62, []) for i in range(5)]
    backend = FakeBackend(dense=dense, identifier=[cand("e102", 0, 0.50, ["E102"])])
    out = run_strategy("identifier_aware", IDA, "E102 警報", backend)
    ids = [r.document_id for r in out.results]
    assert "e102" in ids
    e = next(r for r in out.results if r.document_id == "e102")
    assert e.candidate_sources == ["identifier"]
    assert e.vector_score == 0.50  # 保留 raw vector score（不填假分數）


def test_identifier_candidate_retains_raw_vector_score():
    backend = FakeBackend(dense=[], identifier=[cand("e102", 0, 0.48, ["E102"])])
    out = run_strategy("identifier_aware", IDA, "E102", backend)
    assert out.results[0].vector_score == 0.48


def test_multi_identifier_matchany_passed():
    backend = FakeBackend(dense=[cand("a", 0, 0.7)], identifier=[])
    run_strategy("identifier_aware", IDA, "E102 與 D900 差異", backend)
    assert len(backend.identifier_calls) == 1
    ids, cap = backend.identifier_calls[0]
    assert set(ids) == {"E102", "D900"} and cap == 20


def test_empty_identifier_query_skips_keyword_search():
    backend = FakeBackend(dense=[cand("a", 0, 0.7)], identifier=[cand("x", 0, 0.9, ["E102"])])
    out = run_strategy("identifier_aware", IDA, "一般語意查詢無代碼", backend)
    assert backend.identifier_calls == []          # 未做 keyword search
    assert out.identifier_candidate_count == 0
    assert [r.document_id for r in out.results] == ["a"]  # 與 dense 等價


def test_union_does_not_drop_exact_below_threshold():
    # exact E102 raw=0.50 < threshold 0.55，但 >= floor 0.40 → union 後不得被截掉。
    backend = FakeBackend(dense=[], identifier=[cand("e102", 0, 0.50, ["E102"])])
    out = run_strategy("identifier_aware", IDA, "E102", backend)
    assert [r.document_id for r in out.results] == ["e102"]


# ============================ §21.5 Boost / Penalty ============================
def test_exact_match_boost_applied():
    out = run_strategy("identifier_aware", IDA, "E102", FakeBackend(dense=[cand("e102", 0, 0.60, ["E102"])]))
    r = out.results[0]
    assert r.exact_match_boost == 0.2 and r.ranking_score == 0.80 and r.vector_score == 0.60


def test_confusable_penalty_applied():
    out = run_strategy("identifier_aware", IDA, "E102", FakeBackend(dense=[cand("d900", 0, 0.70, ["D900"])]))
    r = out.results[0]
    assert r.confusable_penalty == 0.15 and round(r.ranking_score, 5) == 0.55
    assert r.conflicting_identifiers == ["D900"]


def test_no_penalty_when_candidate_has_no_identifier():
    out = run_strategy("identifier_aware", IDA, "E102", FakeBackend(dense=[cand("misc", 0, 0.70, [])]))
    r = out.results[0]
    assert r.confusable_penalty == 0.0 and r.exact_match_boost == 0.0
    assert r.ranking_score == 0.70


def test_no_components_when_query_has_no_identifier():
    out = run_strategy("identifier_aware", IDA, "一般查詢", FakeBackend(dense=[cand("d900", 0, 0.70, ["D900"])]))
    r = out.results[0]
    assert r.exact_match_boost == 0.0 and r.confusable_penalty == 0.0
    assert r.ranking_score == 0.70


def test_multi_query_identifier_not_mutually_conflicting():
    # E102 與 D900 都是 query id → 含 D900 的結果不因 E102 被視為 conflicting。
    backend = FakeBackend(dense=[cand("d900", 0, 0.60, ["D900"]), cand("e102", 0, 0.60, ["E102"])])
    out = run_strategy("identifier_aware", IDA, "E102 與 D900 差異", backend)
    for r in out.results:
        assert r.confusable_penalty == 0.0
        assert r.exact_match_boost == 0.2  # 各自命中一個 query id


def test_binary_boost_not_stacked_for_repeated_code():
    # candidate 含多個 query id 也只加一次 binary boost（§十四.7）。
    backend = FakeBackend(dense=[cand("both", 0, 0.60, ["E102", "D900"])])
    out = run_strategy("identifier_aware", IDA, "E102 與 D900", backend)
    assert out.results[0].exact_match_boost == 0.2
    assert round(out.results[0].ranking_score, 5) == 0.80


def test_zero_config_equals_no_weighting():
    zero = {**IDA, "exact_match_boost": 0.0, "confusable_penalty": 0.0}
    out = run_strategy("identifier_aware", zero, "E102",
                       FakeBackend(dense=[cand("e102", 0, 0.60, ["E102"]), cand("d900", 0, 0.70, ["D900"])]))
    for r in out.results:
        assert r.ranking_score == r.vector_score


# ============================ §21.6 Admission / Threshold ============================
def test_dense_result_below_threshold_rejected():
    out = run_strategy("identifier_aware", IDA, "E102", FakeBackend(dense=[cand("misc", 0, 0.50, [])]))
    assert out.results == [] and out.threshold_rejected_count == 1


def test_exact_above_floor_below_threshold_admitted():
    out = run_strategy("identifier_aware", IDA, "E102", FakeBackend(identifier=[cand("e102", 0, 0.50, ["E102"])]))
    assert len(out.results) == 1
    assert out.results[0].admission_reason == "exact_identifier_floor"


def test_exact_below_floor_rejected():
    out = run_strategy("identifier_aware", IDA, "E102", FakeBackend(identifier=[cand("e102", 0, 0.30, ["E102"])]))
    assert out.results == [] and out.identifier_floor_rejected_count == 1


def test_non_exact_cannot_bypass_threshold_via_boost():
    # 一般候選 raw=0.50 < threshold 0.55；即使有 boost 概念也不得放行（無 exact → 無 boost）。
    out = run_strategy("identifier_aware", IDA, "E102", FakeBackend(dense=[cand("misc", 0, 0.50, [])]))
    assert out.results == []


def test_ranking_score_not_used_for_threshold():
    # D900 raw=0.60 >= threshold → 通過 admission；penalty 後 ranking=0.45 不影響 admission。
    out = run_strategy("identifier_aware", IDA, "E102", FakeBackend(dense=[cand("d900", 0, 0.60, ["D900"])]))
    assert len(out.results) == 1
    assert round(out.results[0].ranking_score, 5) == 0.45
    assert out.results[0].admission_reason == "dense_threshold"


def test_threshold_and_floor_boundary_values():
    # 邊界：vector_score 恰等於 threshold / floor → 允許（>=）。
    thr = run_strategy("identifier_aware", IDA, "一般", FakeBackend(dense=[cand("a", 0, 0.55, [])]))
    assert len(thr.results) == 1
    floor = run_strategy("identifier_aware", IDA, "E102", FakeBackend(identifier=[cand("e102", 0, 0.40, ["E102"])]))
    assert len(floor.results) == 1


# ============================ §21.7 Ranking ============================
def test_ranking_score_priority():
    # e102 boosted 0.60->0.80 應排在 raw 更高但無 boost 的 misc(0.70) 之前。
    out = run_strategy("identifier_aware", IDA, "E102",
                       FakeBackend(dense=[cand("misc", 0, 0.70, []), cand("e102", 0, 0.60, ["E102"])]))
    assert [r.document_id for r in out.results] == ["e102", "misc"]


def test_exact_match_tie_break():
    # ranking_score 相同（皆 0.70）：有 exact 者優先。
    cfg = {**IDA, "exact_match_boost": 0.0}  # 無 boost 讓分數相同
    out = run_strategy("identifier_aware", cfg, "E102",
                       FakeBackend(dense=[cand("plain", 0, 0.70, []), cand("e102", 0, 0.70, ["E102"])]))
    assert out.results[0].document_id == "e102"


def test_vector_score_tie_break():
    # ranking_score 相同、皆非 exact：vector_score 高者優先。
    out = run_strategy("identifier_aware", IDA, "一般",
                       FakeBackend(dense=[cand("lo", 0, 0.60, []), cand("hi", 0, 0.65, [])]))
    assert [r.document_id for r in out.results] == ["hi", "lo"]


def test_stable_identity_tie_break_independent_of_input_order():
    # query 有 identifier（觸發 identifier-aware ranking），但候選皆無匹配 id → 皆非 exact、
    # ranking_score/vector_score 全同 → 走 §十六 deterministic identity tie-break（不依輸入順序）。
    a = [cand("bbb", 1, 0.70, []), cand("aaa", 2, 0.70, []), cand("aaa", 1, 0.70, [])]
    out1 = run_strategy("identifier_aware", IDA, "E102 查詢", FakeBackend(dense=list(a)))
    out2 = run_strategy("identifier_aware", IDA, "E102 查詢", FakeBackend(dense=list(reversed(a))))
    order1 = [(r.document_id, r.chunk_index) for r in out1.results]
    order2 = [(r.document_id, r.chunk_index) for r in out2.results]
    assert order1 == order2 == [("aaa", 1), ("aaa", 2), ("bbb", 1)]


def test_no_reorder_from_formatting_rounding():
    # 極接近但不同的 ranking_score 不得因四捨五入而改變順序（內部全精度）。
    out = run_strategy("identifier_aware", IDA, "一般",
                       FakeBackend(dense=[cand("a", 0, 0.600001, []), cand("b", 0, 0.600002, [])]))
    assert [r.document_id for r in out.results] == ["b", "a"]
