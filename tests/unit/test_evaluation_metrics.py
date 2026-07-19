"""Evaluation 指標 / schema 的 deterministic 單元測試（v1.1 Slice 0）。

不載入 BGE、不連 Qdrant、不呼叫 LLM。全部以 deterministic fake retrieval results
驗證指標計算與 case 驗證邏輯，並確認 CLI 與 import 走同一核心 runner。

刻意不驗證「D900 永遠排第一」或「固定 0.xxxx 分數」這類脆弱期望值——
真實排名/分數屬整合測試與 Baseline 報告的範疇。
"""
from __future__ import annotations

import json

import pytest

from evaluation import metrics
from evaluation.schemas import (
    CaseValidationError,
    EvaluationCase,
    EvaluationReport,
    parse_cases,
)


def _case(**kw) -> EvaluationCase:
    base = dict(
        id="X",
        question="q",
        should_answer=True,
        query_type=None,
        expected_document=None,
        expected_section=None,
        expected_page=None,
        confusable_document=None,
    )
    base.update(kw)
    return EvaluationCase(**base)


def _ranked(*pairs) -> list[dict]:
    """pairs：(document, score) 或 (document, score, section, page)。"""
    out = []
    for p in pairs:
        doc, score = p[0], p[1]
        section = p[2] if len(p) > 2 else None
        page = p[3] if len(p) > 3 else None
        out.append({"document": doc, "score": score, "section": section, "page": page})
    return out


# --------------------------------------------------------------------------
# build_case_result：hit / rank / reciprocal rank
# --------------------------------------------------------------------------
def test_build_case_result_basic_hit_and_ranks():
    case = _case(
        id="A",
        query_type="exact_code",
        expected_document="alarm.md",
        expected_section="E102",
        confusable_document="d900.md",
    )
    ranked = _ranked(("alarm.md", 0.9, "E102", None), ("d900.md", 0.8), ("other.md", 0.5))
    r = metrics.build_case_result(case, ranked, retrieval_time_ms=12, top_k=5)

    assert r.expected_rank == 1
    assert r.confusable_rank == 2
    assert r.doc_hit_at_1 is True
    assert r.doc_hit_at_3 is True
    assert r.doc_hit_at_5 is True
    assert r.section_hit_at_1 is True
    assert r.reciprocal_rank == 1.0
    assert r.top1_document == "alarm.md"
    assert [t["rank"] for t in r.top5] == [1, 2, 3]


def test_build_case_result_confusable_beats_expected():
    case = _case(
        id="B",
        query_type="confusable",
        expected_document="alarm.md",
        confusable_document="d900.md",
    )
    ranked = _ranked(("d900.md", 0.85), ("alarm.md", 0.80))
    r = metrics.build_case_result(case, ranked, retrieval_time_ms=9, top_k=5)

    assert r.expected_rank == 2
    assert r.confusable_rank == 1
    assert r.doc_hit_at_1 is False
    assert r.doc_hit_at_5 is True
    assert r.reciprocal_rank == 0.5


def test_build_case_result_no_results():
    case = _case(id="Z", expected_document="alarm.md")
    r = metrics.build_case_result(case, [], retrieval_time_ms=3, top_k=5)
    assert r.expected_rank is None
    assert r.doc_hit_at_1 is False
    assert r.reciprocal_rank == 0.0
    assert r.top1_document is None


# --------------------------------------------------------------------------
# 聚合指標
# --------------------------------------------------------------------------
def _sample_results() -> list:
    a = metrics.build_case_result(
        _case(id="A", query_type="exact_code", expected_document="alarm.md",
              confusable_document="d900.md"),
        _ranked(("alarm.md", 0.90), ("d900.md", 0.80)), 10, top_k=5,
    )
    b = metrics.build_case_result(
        _case(id="B", query_type="confusable", expected_document="alarm.md",
              confusable_document="d900.md"),
        _ranked(("d900.md", 0.85), ("alarm.md", 0.80)), 20, top_k=5,
    )
    c = metrics.build_case_result(
        _case(id="C", query_type="semantic", expected_document="pdf.pdf",
              expected_section="S", expected_page=2),
        _ranked(("pdf.pdf", 0.70, "S", 2), ("alarm.md", 0.4)), 30, top_k=5,
    )
    d = metrics.build_case_result(
        _case(id="D", query_type="unanswerable", should_answer=False),
        _ranked(("alarm.md", 0.60)), 40, top_k=5,
    )
    e = metrics.build_case_result(
        _case(id="E", query_type="unanswerable", should_answer=False),
        _ranked(("alarm.md", 0.40)), 50, top_k=5,
    )
    return [a, b, c, d, e]


def test_mrr():
    # A rr=1.0, B rr=0.5, C rr=1.0 -> mean over 3 answerable = 0.8333
    value, n = metrics.mrr(_sample_results())
    assert n == 3
    assert value == pytest.approx(0.8333, abs=1e-4)


def test_recall_at_5_and_hit_at_5():
    m = metrics.aggregate_metrics(_sample_results(), configured_threshold=0.55)
    # A,B,C 皆命中 top-5（考量對象 = 有 expected_document 的 3 題）
    assert m["hit_at_5_considered"] == 3
    assert m["hit_at_5"] == 1.0
    assert m["recall_at_5"] == m["hit_at_5"]


def test_exact_code_top1_accuracy():
    out = metrics.exact_code_top1_accuracy(_sample_results())
    assert out["considered"] == 1  # 只有 A 是 exact_code
    assert out["accuracy"] == 1.0
    assert out["miss_ids"] == []


def test_confusable_error_rate():
    out = metrics.confusable_error(_sample_results())
    # A(no error) + B(error) -> 1/2
    assert out["considered"] == 2
    assert out["errors"] == 1
    assert out["rate"] == 0.5
    assert out["error_ids"] == ["B"]


def test_unanswerable_false_positive_rate():
    out = metrics.unanswerable_false_positive_rate(_sample_results(), threshold=0.55)
    # D(0.60>=0.55 FP) + E(0.40 not) -> 1/2
    assert out["considered"] == 2
    assert out["false_positives"] == 1
    assert out["rate"] == 0.5
    assert out["false_positive_ids"] == ["D"]


def test_group_by_query_type():
    g = metrics.group_by_query_type(_sample_results())
    assert set(g) == {"exact_code", "confusable", "semantic", "unanswerable"}
    assert g["unanswerable"]["count"] == 2
    assert g["exact_code"]["hit_at_1"] == 1.0


def test_empty_dataset_does_not_crash():
    m = metrics.aggregate_metrics([], configured_threshold=0.55)
    assert m["total_questions"] == 0
    assert m["hit_at_1"] == 0.0
    assert m["mrr"] == 0.0
    assert m["p95_latency_ms"] == 0.0
    assert m["by_query_type"] == {}


# --------------------------------------------------------------------------
# schema 驗證：缺欄位 / 重複 id / 非法值
# --------------------------------------------------------------------------
def test_parse_cases_missing_field():
    with pytest.raises(CaseValidationError, match="缺少必要欄位"):
        parse_cases([{"id": "A", "question": "q"}])  # 缺 should_answer


def test_parse_cases_duplicate_id():
    with pytest.raises(CaseValidationError, match="重複的 case id"):
        parse_cases(
            [
                {"id": "A", "question": "q1", "should_answer": True},
                {"id": "A", "question": "q2", "should_answer": True},
            ]
        )


def test_parse_cases_invalid_query_type():
    with pytest.raises(CaseValidationError, match="非法 query_type"):
        parse_cases([{"id": "A", "question": "q", "should_answer": True, "query_type": "bogus"}])


def test_parse_cases_should_answer_not_bool():
    with pytest.raises(CaseValidationError, match="should_answer"):
        parse_cases([{"id": "A", "question": "q", "should_answer": "yes"}])


def test_parse_cases_empty_list():
    with pytest.raises(CaseValidationError, match="為空"):
        parse_cases([])


def test_real_cases_json_is_valid():
    """實際 baseline cases.json 可被驗證載入。"""
    import os

    from evaluation.schemas import load_cases

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cases = load_cases(os.path.join(root, "evaluation", "cases.json"))
    assert len(cases) >= 5
    types = {c.query_type for c in cases}
    assert {"exact_code", "semantic", "confusable", "unanswerable"} <= types


# --------------------------------------------------------------------------
# CLI 與 import 使用同一核心 runner
# --------------------------------------------------------------------------
def _report_from(results) -> EvaluationReport:
    m = metrics.aggregate_metrics(results, 0.0)
    return EvaluationReport(
        config={"collection": "factory_evaluation", "top_k": 10, "score_threshold": 0.0},
        environment={
            "embedding_model": "fake", "vector_size": 8, "distance": "Cosine",
            "chunk_size_chars": 600, "chunk_overlap_chars": 100, "embedding_max_tokens": 480,
        },
        corpus={"ingested_chunk_counts": {}},
        dataset={},
        metrics=m,
        threshold_at_configured=metrics.threshold_report(results, 0.0),
        threshold_sweep=[metrics.threshold_report(results, 0.55)],
        cases=results,
        notes=[],
    )


def test_cli_delegates_to_runner_core(monkeypatch, tmp_path):
    """run_cli 必須透過 evaluation.runner.run_evaluation 執行，不另建一套實作。"""
    from evaluation import evaluate_retrieval

    calls = {}

    def fake_run_evaluation(config, embedding=None):
        calls["config"] = config
        return _report_from(_sample_results())

    # 攔截共用核心；若 CLI 自行複製一套邏輯，此攔截就不會被呼叫。
    monkeypatch.setattr(evaluate_retrieval.runner, "run_evaluation", fake_run_evaluation)
    out_path = tmp_path / "results.json"
    monkeypatch.setattr(evaluate_retrieval, "RESULTS_PATH", str(out_path))

    legacy = evaluate_retrieval.run_cli(["--collection", "factory_evaluation", "--top-k", "10"])

    assert calls, "run_cli 未呼叫 runner.run_evaluation（可能複製了第二套實作）"
    assert calls["config"].collection == "factory_evaluation"
    assert calls["config"].cases_path.endswith("questions.json")
    # 產出 legacy 結構且可寫檔
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert "hit_at_1" in written and "questions" in written
    assert legacy["hit_at_1"] == written["hit_at_1"]
