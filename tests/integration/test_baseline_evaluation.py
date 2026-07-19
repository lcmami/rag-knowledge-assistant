"""v1.1 Slice 0 Baseline 整合測試（真實 BGE + 隔離 Qdrant）。

驗證 Baseline 評估流程可完整跑完、輸出 Top-5 與真實分數、報告記錄 E102/D900 實際排名，
且**不碰正式 factory_knowledge**、評估後不留孤兒 collection。

刻意**不**斷言固定分數或「D900 必為 Top-1」；只誠實記錄實際排名（confusable 壓力測試）。

執行：
    pytest -m integration tests/integration/test_baseline_evaluation.py
（需先啟動 localhost Qdrant 且已快取 BGE 模型；未就緒時整個模組 skip 而非失敗。）
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLE_DIR = os.path.join(ROOT, "sample_docs")
FIXTURES_DIR = os.path.join(ROOT, "evaluation", "fixtures")
CASES_PATH = os.path.join(ROOT, "evaluation", "cases.json")

TEST_COLLECTION = "factory_test_baseline_v1_1"
PROD_COLLECTION = "factory_knowledge"
ALARM_DOC = "equipment_alarm_manual.md"
CONFUSABLE_DOC = "confusable_alarm_d900.md"


@pytest.fixture
def qdrant_url(monkeypatch):
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert TEST_COLLECTION != PROD_COLLECTION
    try:
        from qdrant_client import QdrantClient

        probe = QdrantClient(url=url)
        probe.get_collections()
        probe.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"真實 Qdrant 未就緒（{url}）：{exc}")
    if not os.path.exists(os.path.join(FIXTURES_DIR, CONFUSABLE_DOC)):
        pytest.skip("缺少 D900 fixture")
    return url


def _prod_snapshot(url):
    from qdrant_client import QdrantClient

    c = QdrantClient(url=url)
    try:
        names = [x.name for x in c.get_collections().collections]
        prod_points = c.count(PROD_COLLECTION).count if PROD_COLLECTION in names else None
        return names, prod_points
    finally:
        c.close()


def test_baseline_runs_and_records_ranks(qdrant_url):
    from evaluation import runner
    from evaluation.schemas import EvaluationConfig

    before_names, before_prod = _prod_snapshot(qdrant_url)

    config = EvaluationConfig(
        corpus_dirs=(SAMPLE_DIR, FIXTURES_DIR),
        cases_path=CASES_PATH,
        qdrant_url=qdrant_url,
        collection=TEST_COLLECTION,
        top_k=5,
        score_threshold=0.0,
        report_threshold=0.55,
        label="baseline-integration-test",
    )

    report = runner.run_evaluation(config)

    # 1. 流程完整跑完，每題都有結果列
    assert len(report.cases) == report.metrics["total_questions"]
    assert report.metrics["total_questions"] >= 5

    # 2. 可回答題輸出 Top-5 與真實 float 分數（不斷言固定值）
    answerable = [c for c in report.cases if c.should_answer]
    assert answerable
    for c in answerable:
        assert c.top5, f"{c.id} 無 Top-5 結果"
        assert isinstance(c.top5[0]["score"], float)
        assert c.top5[0]["score"] > 0

    # 3. 報告記錄 E102 / D900 實際排名（只記錄，不強制名次）
    e102_cases = [c for c in report.cases if c.expected_document == ALARM_DOC]
    assert e102_cases
    for c in e102_cases:
        # E102 正確文件應可在候選內被檢索到（Baseline 現況；名次由報告誠實記錄）
        assert c.expected_rank is not None, f"{c.id}: E102 未進入 Top-{config.top_k}"
    # D900 為隔離 fixture，確實有進入 corpus 被索引（confusable 壓力來源存在）
    assert CONFUSABLE_DOC in report.corpus["ingested_chunk_counts"]

    # 4. Slice 0 指標齊備且型別正確
    m = report.metrics
    for key in ("hit_at_1", "hit_at_5", "recall_at_5", "mrr"):
        assert isinstance(m[key], float)
    assert m["exact_code_top1"]["considered"] >= 1
    assert m["confusable_code_error"]["considered"] >= 1
    assert m["unanswerable_false_positive"]["threshold"] == 0.55

    # 5./6. 不污染正式 collection；評估後不留孤兒（測試 collection 已刪除）
    after_names, after_prod = _prod_snapshot(qdrant_url)
    assert TEST_COLLECTION not in after_names, "評估後殘留孤兒 collection"
    assert after_prod == before_prod, "factory_knowledge point 數改變"
    # 正式 collection 集合不因評估而新增/減少（除了本測試自身 collection 已清掉）
    assert set(before_names) == set(after_names)
