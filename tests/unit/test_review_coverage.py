"""Slice 6 §二十三 — Coverage Summary 與 Gap Detection 測試。

Coverage 只提示，不得自動改 Split、自動接受 Candidate 或自動發布。
"""
from __future__ import annotations

import inspect

from evaluation.v11.case_schema import EvaluationCaseV11, EvidenceSpan
from evaluation.v11.review import coverage as coverage_module
from evaluation.v11.review.coverage import (
    GAP_ADDITIONAL_CORPUS_REQUIRED,
    GAP_INDEPENDENT_CONFUSABLE,
    GAP_SMALL_TEST_EXACT_CODE,
    GAP_SMALL_VALIDATION_EXACT_CODE,
    GAP_VALIDATION_IDENTIFIER_UNANSWERABLE,
    build_coverage,
    detect_gaps,
    summarize,
)


def _case(
    case_id,
    *,
    question="問題？",
    query_type="semantic",
    answerable=True,
    group_id="g",
    split="tune",
):
    evidence = (
        [EvidenceSpan(document_sha256="a" * 64, canonical_text_version="v1", anchor_text=["x"])]
        if answerable
        else []
    )
    return EvaluationCaseV11(
        id=case_id,
        question=question,
        query_type=query_type,
        answerable=answerable,
        group_id=group_id,
        split=split,
        expected_evidence=evidence,
    )


def _codes(gaps):
    return {g.code for g in gaps}


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def test_counts_per_split_and_query_type():
    cases = [
        _case("a", query_type="exact_code", split="tune", group_id="g1"),
        _case("b", query_type="semantic", split="validation", group_id="g2"),
        _case("c", query_type="semantic", split="validation", group_id="g3"),
        _case("d", query_type="unanswerable", answerable=False, split="test", group_id="g4"),
    ]
    s = summarize(cases)
    assert s.total_cases == 4
    assert s.cases_per_split == {"tune": 1, "validation": 2, "test": 1}
    assert s.groups_per_split == {"tune": 1, "validation": 2, "test": 1}
    assert s.query_type_counts == {"exact_code": 1, "semantic": 2, "unanswerable": 1}
    assert s.query_type_counts_per_split["validation"] == {"semantic": 2}
    assert s.answerable_count == 3
    assert s.unanswerable_count == 1


def test_group_leakage_detected():
    cases = [
        _case("a", group_id="shared", split="tune"),
        _case("b", group_id="shared", split="test"),
        _case("c", group_id="clean", split="tune"),
    ]
    assert summarize(cases).group_leakage == ["shared"]


def test_no_group_leakage_when_groups_are_split_local():
    cases = [_case("a", group_id="g1", split="tune"), _case("b", group_id="g2", split="test")]
    assert summarize(cases).group_leakage == []


def test_identifier_bearing_unanswerable_counted_per_split():
    cases = [
        _case(
            "a",
            question="CX-500 的售價是多少？",
            query_type="unanswerable",
            answerable=False,
            split="validation",
            group_id="g1",
        ),
        _case(
            "b",
            question="這台機器貴嗎？",
            query_type="unanswerable",
            answerable=False,
            split="validation",
            group_id="g2",
        ),
    ]
    s = summarize(cases)
    assert s.identifier_bearing_unanswerable_per_split == {"validation": 1}


def test_confusable_groups_and_independence():
    cases = [
        _case("a", query_type="confusable", group_id="alarm-e102", split="tune"),
        _case("b", query_type="confusable", group_id="pump-x77", split="tune"),
    ]
    s = summarize(cases)
    assert s.confusable_groups == ["alarm-e102", "pump-x77"]
    assert s.independent_confusable_groups == ["pump-x77"]


# --------------------------------------------------------------------------
# Gap Detection
# --------------------------------------------------------------------------
def test_validation_identifier_unanswerable_gap_detected():
    cases = [_case("a", split="validation", group_id="g1")]
    assert GAP_VALIDATION_IDENTIFIER_UNANSWERABLE in _codes(detect_gaps(summarize(cases)))


def test_validation_identifier_unanswerable_gap_disappears_when_candidate_added():
    """§二十三：Candidate 加入後 Gap 消失。"""
    before = [_case("a", split="validation", group_id="g1")]
    assert GAP_VALIDATION_IDENTIFIER_UNANSWERABLE in _codes(detect_gaps(summarize(before)))

    after = before + [
        _case(
            "new",
            question="CX-500 的保固期限是多久？",
            query_type="unanswerable",
            answerable=False,
            split="validation",
            group_id="cx500-warranty",
        )
    ]
    assert GAP_VALIDATION_IDENTIFIER_UNANSWERABLE not in _codes(detect_gaps(summarize(after)))


def test_independent_confusable_gap_detected_and_resolved():
    only_known = [_case("a", query_type="confusable", group_id="alarm-e102", split="tune")]
    assert GAP_INDEPENDENT_CONFUSABLE in _codes(detect_gaps(summarize(only_known)))

    with_independent = only_known + [
        _case("b", query_type="confusable", group_id="pump-x77", split="tune")
    ]
    assert GAP_INDEPENDENT_CONFUSABLE not in _codes(detect_gaps(summarize(with_independent)))


def test_additional_corpus_required_reported_when_corpus_insufficient():
    """§十七.2：語料不足時提示補充，不得虛構無文件依據的 answerable case。"""
    cases = [_case("a", query_type="confusable", group_id="alarm-e102", split="tune")]
    gaps = detect_gaps(summarize(cases), corpus_supports_independent_confusable=False)
    assert GAP_ADDITIONAL_CORPUS_REQUIRED in _codes(gaps)


def test_small_exact_code_sample_warnings():
    cases = [_case("a", query_type="exact_code", split="validation", group_id="g1")]
    codes = _codes(detect_gaps(summarize(cases)))
    assert GAP_SMALL_VALIDATION_EXACT_CODE in codes
    assert GAP_SMALL_TEST_EXACT_CODE in codes


def test_sufficient_exact_code_sample_has_no_small_sample_warning():
    cases = [
        _case(f"v{i}", query_type="exact_code", split="validation", group_id=f"gv{i}")
        for i in range(3)
    ] + [
        _case(f"t{i}", query_type="exact_code", split="test", group_id=f"gt{i}")
        for i in range(3)
    ]
    codes = _codes(detect_gaps(summarize(cases)))
    assert GAP_SMALL_VALIDATION_EXACT_CODE not in codes
    assert GAP_SMALL_TEST_EXACT_CODE not in codes


def test_gaps_are_warnings_only():
    cases = [_case("a", split="validation", group_id="g1")]
    report = build_coverage(cases)
    assert all(g.severity == "warning" for g in report.gaps)


# --------------------------------------------------------------------------
# 邊界：不得自動修改
# --------------------------------------------------------------------------
def test_coverage_does_not_mutate_input_cases():
    cases = [_case("a", split="validation", group_id="g1")]
    build_coverage(cases)
    assert cases[0].split == "validation"  # 未被自動改 split
    assert len(cases) == 1  # 未自動新增 case


def test_coverage_module_has_no_split_mutation_or_publish():
    """§十六：Coverage 不得自動改 Split／接受 Candidate／發布 Dataset。"""
    source = inspect.getsource(coverage_module)
    for forbidden in ("def publish", "accept_candidate", "update_candidate", ".split ="):
        assert forbidden not in source


def test_coverage_module_does_not_read_retrieval_results():
    source = inspect.getsource(coverage_module)
    for forbidden in ("retrieve", "trial", "search_runner", "recommendation"):
        assert forbidden not in source
