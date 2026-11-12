"""v1.2 Optimizer Grid Runner Smoke Script（最小版,standalone）。

目的：確認 :func:`evaluation.v12.optimizer.runner.run_grid_candidates` 能：
1. 在**既有** baseline index 上跑多個 candidate（不重建 chunk / embedding / collection）。
2. 對每個 candidate 產出 MRR 等既有 metric。
3. GridCandidateGenerator 自動跳過與 baseline 完全相同的 candidate。

限制：
- 使用 :class:`FakeTrialServiceProvider`（不連 Qdrant、不載 BGE、不呼叫 LLM）。
- 不執行完整 SciFact benchmark；只取 tune 前 10 個 cases。
- search space 固定為 ``top_k=[5,7]`` × ``score_threshold=[0.45,0.55]``。
- baseline 為 ``(top_k=5, score_threshold=0.55)``（與 v12-agent-scifact.json 一致），
  故 generator 會自動跳過該組 → 預期 3 個 candidate。

執行：
    python tests/unit/test_v12_optimizer_grid_smoke.py
"""
from __future__ import annotations

import json
import os

from evaluation.v11.case_schema import parse_dataset
from evaluation.v11.retrieval.config import RetrievalConfig
from evaluation.v11.retrieval_metrics import IndexChunk, RankedResult
from evaluation.v11.run_models import TrialConfig
from evaluation.v11.trial_runner import IndexOutcome
from evaluation.v12.optimizer.runner import run_grid_candidates
from tests.unit.v11_fakes import FakeTrialServiceProvider

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCIFACT_CASES_PATH = os.path.join(
    ROOT, "evaluation", "datasets", "scifact_agent", "cases-v1.json"
)
COLLECTION_NAME = "optimizer_smoke_baseline_index"


def _make_baseline_trial() -> TrialConfig:
    """Baseline 對齊 v12-agent-scifact.json 的 retrieval_config。"""
    baseline_rc = RetrievalConfig.from_raw(
        {
            "strategy": "dense",
            "top_k": 5,
            "score_threshold": 0.55,
            "candidate_multiplier": 3,
            "candidate_cap": 50,
        }
    )
    return TrialConfig(
        trial_name="baseline-dense",
        chunk_config={"strategy": "current"},
        retrieval=baseline_rc.to_canonical_dict(),
    )


def _make_index_universe(n: int = 20) -> list[IndexChunk]:
    """Fake index universe（每個 chunk 帶 provenance 供 NDCG 計算）。"""
    return [
        IndexChunk(
            document_id=f"doc-{i}",
            document_sha256=f"sha-{i}",
            chunk_index=i,
            text=f"chunk text {i}",
            page=None,
            section=f"S{i}",
            provenance_spans=[
                {
                    "document_sha256": f"sha-{i}",
                    "canonical_text_version": "v1",
                    "loader_version": "v1",
                    "page": None,
                    "section": f"S{i}",
                    "paragraph_index": 0,
                    "offset_scope": "document",
                    "char_start": 0,
                    "char_end": 100,
                }
            ],
            identifiers=[],
            canonical_text_version="v1",
        )
        for i in range(n)
    ]


def _make_results_for(trial_name: str, collection: str, question: str) -> list[RankedResult]:
    """腳本化檢索結果：固定 5 筆,score 隨 trial_name 變化以模擬不同 candidate。"""
    # 用 trial_name 末段的 hash 當作 score 偏移,讓不同 candidate 拿到不同結果。
    seed = sum(ord(c) for c in trial_name) % 7
    return [
        RankedResult(
            rank=r + 1,
            document_id=f"doc-{(r + seed) % 20}",
            document_sha256=f"sha-{(r + seed) % 20}",
            filename=f"doc-{(r + seed) % 20}.md",
            section=f"S{(r + seed) % 20}",
            page=None,
            chunk_index=(r + seed) % 20,
            score=round(0.9 - r * 0.05 - seed * 0.01, 4),
            text=f"chunk text {(r + seed) % 20}",
            provenance_spans=[
                {
                    "document_sha256": f"sha-{(r + seed) % 20}",
                    "canonical_text_version": "v1",
                    "loader_version": "v1",
                    "page": None,
                    "section": f"S{(r + seed) % 20}",
                    "paragraph_index": 0,
                    "offset_scope": "document",
                    "char_start": 0,
                    "char_end": 100,
                }
            ],
            identifiers=[],
            canonical_text_version="v1",
        )
        for r in range(5)
    ]


def _load_tune_cases(limit: int = 10) -> list:
    """載入 SciFact tune 前 N 個 cases（不跑完整 benchmark）。"""
    with open(SCIFACT_CASES_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    dataset = parse_dataset(raw)
    tune_cases = [c for c in dataset.cases if c.split == "tune"]
    return tune_cases[:limit]


def main() -> None:
    """Smoke test：跑 grid candidates,輸出每個 candidate 的 top_k / score_threshold / MRR / hash。"""
    baseline_trial = _make_baseline_trial()
    cases = _load_tune_cases(limit=10)
    index_universe = _make_index_universe(n=20)

    provider = FakeTrialServiceProvider(
        point_count=len(index_universe),
        results_for=_make_results_for,
        universe_for=lambda c, _c: index_universe,
    )

    # 1. 建立 baseline index（只建一次,所有 candidate 共用）。
    idx = provider.build_index(COLLECTION_NAME, baseline_trial, files=[])
    assert isinstance(idx, IndexOutcome)
    assert idx.point_count == len(index_universe)
    assert provider.collection_exists(COLLECTION_NAME)

    # 2. 取得 index universe（所有 candidate 共用）。
    universe = provider.index_manifest(COLLECTION_NAME)
    assert len(universe) == len(index_universe)

    # 3. 跑 grid candidates。
    search_space = {
        "top_k": [5, 7],
        "score_threshold": [0.45, 0.55],
    }
    outcomes = run_grid_candidates(
        baseline_trial=baseline_trial,
        collection_name=COLLECTION_NAME,
        provider=provider,
        cases=cases,
        index_universe=universe,
        search_space=search_space,
    )

    # 4. 驗證：baseline (5, 0.55) 應被跳過 → 預期 3 個 candidate。
    assert len(outcomes) == 3, f"預期 3 個 candidate,實際 {len(outcomes)}"

    # 5. 驗證：所有 candidate 共用同一個 collection（index 被重用）。
    assert provider.build_calls == ["baseline-dense"], (
        f"build_index 只應被呼叫一次,實際 {provider.build_calls}"
    )

    # 6. 驗證：每個 candidate 都有 MRR metric。
    for outcome in outcomes:
        metric_names = {m.metric_name for m in outcome.metrics}
        assert "mrr" in metric_names, f"candidate {outcome.candidate_hash} 缺少 mrr metric"

    # 7. 輸出結果。
    print(f"\n=== Grid Runner Smoke Test ===")
    print(f"collection_name: {COLLECTION_NAME}")
    print(f"baseline: top_k=5, score_threshold=0.55")
    print(f"search_space: {search_space}")
    print(f"cases: {len(cases)} (tune 前 10)")
    print(f"candidate_count: {len(outcomes)}")
    print(f"build_index_calls: {provider.build_calls} (應只有 baseline)")
    print()
    print(f"{'top_k':>6} {'score_threshold':>16} {'MRR':>10} {'candidate_hash':>20}")
    print("-" * 60)
    for outcome in outcomes:
        rc = outcome.retrieval_config
        mrr_metric = next(m for m in outcome.metrics if m.metric_name == "mrr")
        mrr_val = mrr_metric.value if mrr_metric.value is not None else float("nan")
        print(
            f"{rc['top_k']:>6} {rc['score_threshold']:>16} {mrr_val:>10.4f} "
            f"{outcome.candidate_hash[:16]:>20}"
        )


if __name__ == "__main__":
    main()
