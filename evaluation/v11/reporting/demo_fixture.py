"""Deterministic UI Demo Fixture 定義（Slice 8 §十三）。

產生三個**清楚標示為 demo** 的 fixture DB（search / review / deployment），供 UI 與
Playwright 展示，且：

- 每個 fixture 都帶 ``demo_fixture=true`` metadata（§十三.4）。
- **不**修改真實 search / locked search、**不**建立 Qdrant collection / alias、**不**呼叫 LLM。
- 忠實重現目前真實的 **ineligible** recommendation（apply_eligible=false + 五個原因，§十三.12）；
  絕不偽造 apply 成功（§十三.13）。
- Deployment history 為空（alias 未初始化）。

本模組只含 deterministic 定義（可入庫，§十三.15）；產生的 DB 檔放 ignored 路徑，不入庫。
"""
from __future__ import annotations

import json
import os
from typing import Any

from evaluation.v11.case_schema import parse_dataset
from evaluation.v11.deploy.repository import DeploymentRepository
from evaluation.v11.review.models import (
    CANDIDATE_ACCEPTED,
    CANDIDATE_PROPOSED,
    CANDIDATE_REJECTED,
    EVENT_CANDIDATE_ACCEPTED,
    EVENT_CANDIDATE_CREATED,
    EVENT_CANDIDATE_REJECTED,
    SESSION_REVIEWING,
    CandidateRevision,
    QuestionCandidate,
    ReviewSession,
)
from evaluation.v11.review.repository import ReviewRepository
from evaluation.v11.run_models import (
    CleanupStatus,
    EvaluationRun,
    RetrievalTrial,
    TrialCaseResult,
    TrialMetric,
)
from evaluation.v11.search.search_models import (
    CandidateStageStatus,
    OptimizerSearch,
    SearchCandidate,
    SearchStatus,
)
from evaluation.v11.search.search_repository import SearchRepository

# --------------------------------------------------------------------------
# 常數（deterministic）
# --------------------------------------------------------------------------
TS = "2026-07-20T00:00:00+00:00"
DATASET_VERSION = "v1.1-coverage-0002"
CORPUS_HASH = "corpusdemo0001abcdef0123456789abcdef0123456789abcdef0123456789ab"
DATASET_HASH = "datasetdemo0001abcdef0123456789abcdef0123456789abcdef01234567"
BASELINE_HASH = "cfgbaseline0001abcdef0123456789abcdef0123456789abcdef01234567"
RECOMMENDED_HASH = "e246697d41aa0011223344556677889900aabbccddeeff00112233445566"

INELIGIBLE_REASONS = [
    "small_validation_sample",
    "quality_metrics_tied",
    "selection_driven_by_single_run_latency",
    "test_constraint_failed",
    "demo_corpus_only",
]

SEARCH_DB_NAME = "optimizer-demo.search.db"
REVIEW_DB_NAME = "optimizer-demo.review.db"
DEPLOYMENT_DB_NAME = "optimizer-demo.deployment.db"


def _config_json(chunk: str, strategy: str, floor: float, threshold: float,
                 boost: float | None = None, penalty: float | None = None) -> str:
    retrieval: dict[str, Any] = {
        "strategy": strategy,
        "score_threshold": threshold,
        "identifier_vector_floor": floor,
    }
    if boost is not None:
        retrieval["exact_match_boost"] = boost
    if penalty is not None:
        retrieval["confusable_penalty"] = penalty
    return json.dumps(
        {
            "schema_version": "1.0",
            "trial_name": f"{chunk}-{strategy}",
            "chunk_config": {"strategy": chunk},
            "retrieval_config": retrieval,
            "collection_options": {},
        },
        ensure_ascii=False, sort_keys=True,
    )


# --------------------------------------------------------------------------
# Evaluation cases（問題文字；供 per-case diff 顯示，無文件全文）
# --------------------------------------------------------------------------
_SHA_ALARM = "a" * 64
_SHA_MAINT = "b" * 64
_DEMO_CASES = {
    "schema_version": "1.1",
    "dataset_version": DATASET_VERSION,
    "cases": [
        {
            "id": "eval-e102-001",
            "question": "設備出現 E102 感測器通訊異常時應如何處理？",
            "query_type": "exact_code", "answerable": True,
            "group_id": "alarm-e102", "split": "test",
            "expected_evidence": [
                {"filename": "equipment_alarm_manual.md", "section": "E102 感測器通訊異常",
                 "anchor_text": ["E102"], "document_sha256": _SHA_ALARM,
                 "canonical_text_version": "v1"}
            ],
        },
        {
            "id": "eval-maint-cycle-001",
            "question": "設備的定期保養週期是多久？",
            "query_type": "semantic", "answerable": True,
            "group_id": "maint-cycle", "split": "test",
            "expected_evidence": [
                {"filename": "equipment_maintenance_manual.pdf", "section": "保養週期",
                 "anchor_text": ["保養"], "document_sha256": _SHA_MAINT,
                 "canonical_text_version": "v1"}
            ],
        },
        {
            "id": "eval-unanswerable-price",
            "question": "CX-500 的售價是多少？",
            "query_type": "unanswerable", "answerable": False,
            "group_id": "unanswerable-price", "split": "test",
            "expected_evidence": [],
        },
    ],
}


# --------------------------------------------------------------------------
# Metrics（§八 清單；honest：品質幾乎全平手，只有 latency 有差異）
# --------------------------------------------------------------------------
def _metrics(trial_id: str, *, baseline: bool) -> list[TrialMetric]:
    def m(name, value, scope="aggregate", qt=None, n=None, computable=True):
        return TrialMetric(
            trial_id=trial_id, metric_name=name, scope=scope, query_type=qt,
            value=value, sample_count=n, computable=computable,
        )

    return [
        m("exact_code_top1_accuracy", 0.5, n=2),                 # 兩者皆 0.5 → 絕對門檻 0.75 未達
        m("unanswerable_false_positive_rate", 0.0, n=2),
        m("ndcg_at_5", 0.8367 if baseline else 0.9193, n=7),      # recommendation 改善 +0.0826
        m("mrr", 1.0, n=7),
        m("relevant_context_hit_at_1", 0.8, n=7),
        m("relevant_context_hit_at_3", 1.0, n=7),
        m("relevant_context_hit_at_5", 1.0, n=7),
        m("duplicate_ratio", 0.0, n=7),
        m("latency_p50_ms", 40.0 if baseline else 45.0, n=7),
        m("latency_p95_ms", 47.6 if baseline else 67.9, n=7),     # 變差但仍 < 86 ceiling
        m("index_build_time_ms", 1200.0 if baseline else 1250.0, n=1),
        m("cache_hit_rate", 0.0 if baseline else 1.0, n=1),
    ]


def _top5_e102(baseline: bool) -> str:
    """C01 的 Top-5：兩者 first_relevant_rank 都是 2（§九 誠實呈現，未解決）。"""
    return json.dumps([
        {"rank": 1, "document_id": "d900", "filename": "equipment_alarm_manual.md",
         "section": "D900 與 E102 的差異", "page": None, "chunk_index": 7,
         "score": 0.71, "vector_score": 0.71, "ranking_score": 0.91,
         "relevance_level": 0, "identifiers": ["E102", "D900"],
         "exact_identifier_matches": ["E102"], "exact_match_boost": 0.2,
         "confusable_penalty": 0.0, "candidate_sources": ["dense", "identifier"],
         "admission_reason": "exact_identifier_match"},
        {"rank": 2, "document_id": "e102", "filename": "equipment_alarm_manual.md",
         "section": "E102 感測器通訊異常", "page": None, "chunk_index": 3,
         "score": 0.66, "vector_score": 0.66, "ranking_score": 0.86,
         "relevance_level": 3, "identifiers": ["E102"],
         "exact_identifier_matches": ["E102"], "exact_match_boost": 0.2,
         "confusable_penalty": 0.0, "candidate_sources": ["dense", "identifier"],
         "admission_reason": "exact_identifier_match"},
    ], ensure_ascii=False)


def _top5_generic(section: str, filename: str, rank1_rel: int) -> str:
    return json.dumps([
        {"rank": 1, "document_id": "doc", "filename": filename, "section": section,
         "page": 2, "chunk_index": 1, "score": 0.72, "vector_score": 0.72,
         "ranking_score": 0.72, "relevance_level": rank1_rel, "identifiers": [],
         "admission_reason": "score_threshold"},
    ], ensure_ascii=False)


def _case_results(trial_id: str, *, baseline: bool) -> list[TrialCaseResult]:
    return [
        TrialCaseResult(
            trial_id=trial_id, case_id="eval-e102-001", case_status="answered",
            query_type="exact_code", answerable=True, latency_ms=48 if baseline else 68,
            first_relevant_rank=2,  # 兩者皆 2 → verdict "same"
            top_results_json=_top5_e102(baseline), evidence_match_json="{}",
        ),
        TrialCaseResult(
            trial_id=trial_id, case_id="eval-maint-cycle-001", case_status="answered",
            query_type="semantic", answerable=True, latency_ms=40 if baseline else 44,
            first_relevant_rank=2 if baseline else 1,  # recommendation 改善 → "improved"
            top_results_json=_top5_generic("保養週期", "equipment_maintenance_manual.pdf",
                                           0 if baseline else 3),
            evidence_match_json="{}",
        ),
        TrialCaseResult(
            trial_id=trial_id, case_id="eval-unanswerable-price", case_status="correct_refusal",
            query_type="unanswerable", answerable=False, latency_ms=38 if baseline else 42,
            first_relevant_rank=None, top_results_json="[]", evidence_match_json="{}",
        ),
    ]


def _split_manifest() -> dict[str, Any]:
    def stat(split, cases, groups, qtc, missing_ok=True):
        return {
            "split": split, "case_count": cases, "group_count": groups,
            "case_ids": [], "group_ids": [],
            "query_type_counts": qtc,
            "answerable_count": sum(1 for _ in range(cases)),
            "unanswerable_count": qtc.get("unanswerable", 0),
            "missing_expected_query_types": [] if missing_ok else ["identifier_bearing_unanswerable"],
            # dataset_path 故意省略（store 也會 redact），避免任何私人路徑。
            "dataset_hash": DATASET_HASH,
        }

    return {
        "tune": stat("tune", 4, 3, {"exact_code": 1, "semantic": 2, "cross_section": 1}),
        "validation": stat("validation", 5, 5,
                           {"exact_code": 1, "procedure": 1, "semantic": 1, "unanswerable": 2},
                           missing_ok=False),
        "test": stat("test", 7, 3,
                    {"exact_code": 2, "semantic": 1, "confusable": 2, "unanswerable": 2}),
        "group_leakage_ok": True,
        "group_leakage_detail": [],
    }


def _outcome() -> dict[str, Any]:
    return {
        "test_constraint_status": "failed",
        "failed_test_constraints": ["exact_code_top1_accuracy"],
        "test_constraint_note": (
            "test constraint 為絕對門檻。exact_code_top1_accuracy 0.5 < 0.75；"
            "baseline 亦為 0.5，故此為『絕對門檻未達』而非『相對 baseline 退化』。"
        ),
        "baseline_comparison_status": "improved",
        "baseline_comparison_note": "recommendation 於 ndcg_at_5 改善，其餘品質指標與 baseline 持平。",
        "quality_tie_candidate_count": 8,
        "latency_measurement_repetitions": 1,
        "latency_selection_stable": False,
        "recommendation_selection_basis": "validation_latency_tiebreak",
        "latency_order_reversed_on_test": True,
        "selection_note": (
            "All 8 feasible validation candidates were tied on the configured retrieval-quality "
            "objectives.\n\nThe selected recommendation was determined by measured P95 latency. "
            "The latency ranking was based on a small, single-machine run and reversed relative "
            "ordering on the test split, so it should be treated as unstable."
        ),
        "recommendation_apply_eligible": False,
        "recommendation_confidence": "low",
        "apply_ineligible_reasons": list(INELIGIBLE_REASONS),
        "apply_eligibility_note": (
            "apply_eligible=false 不代表取消 recommendation lock：recommended_config_hash 仍固定。"
            "它代表目前證據不足以進入未來的 Apply。"
        ),
        "known_limitations": [
            "First relevant rank for eval-e102-001 remains 2.",
            "Same-chunk multi-identifier ambiguity remains unresolved.",
        ],
    }


def _seed_search_db(path: str) -> None:
    repo = SearchRepository(path)
    repo.import_dataset(parse_dataset(_DEMO_CASES))

    run = EvaluationRun(
        run_id="run-demo-test", status="completed", run_name="demo-test-split",
        dataset_version=DATASET_VERSION, dataset_hash=DATASET_HASH, corpus_hash=CORPUS_HASH,
        code_commit="commitdemo0001", working_tree_dirty=False,
        canonical_text_version="v1", loader_version="v1",
        identifier_normalization_version="v1", random_seed=42,
        created_at=TS, completed_at=TS,
    )
    repo.upsert_run(run)

    baseline_trial = RetrievalTrial(
        trial_id="trial-baseline-test", run_id="run-demo-test", ordinal=0,
        trial_name="baseline", status="completed",
        config_json=_config_json("current", "dense", 0.0, 0.55),
        config_hash=BASELINE_HASH, collection_name=None, point_count=24,
        cleanup_status=CleanupStatus.CLEANED, index_build_time_ms=1200,
        completed_at=TS,
    )
    rec_trial = RetrievalTrial(
        trial_id="trial-rec-test", run_id="run-demo-test", ordinal=1,
        trial_name="recommendation", status="completed",
        config_json=_config_json("current", "identifier_aware", 0.5, 0.55, 0.2, 0.1),
        config_hash=RECOMMENDED_HASH, collection_name=None, point_count=24,
        cleanup_status=CleanupStatus.CLEANED, index_build_time_ms=1250,
        completed_at=TS,
    )
    for t in (baseline_trial, rec_trial):
        repo.upsert_trial(t)
    repo.save_trial_metrics("trial-baseline-test", _metrics("trial-baseline-test", baseline=True))
    repo.save_trial_metrics("trial-rec-test", _metrics("trial-rec-test", baseline=False))
    repo.save_trial_case_results("trial-baseline-test", _case_results("trial-baseline-test", baseline=True))
    repo.save_trial_case_results("trial-rec-test", _case_results("trial-rec-test", baseline=False))

    search = OptimizerSearch(
        search_id="search-demo", search_name="DEMO optimizer search (fixture)",
        status=SearchStatus.TEST_CONSTRAINT_FAILED, search_mode="grid",
        search_config_json=json.dumps({"mode": "grid"}), search_config_hash="scfgdemo",
        dataset_version=DATASET_VERSION, dataset_hash=DATASET_HASH, corpus_hash=CORPUS_HASH,
        code_commit="commitdemo0001", working_tree_dirty=False, random_seed=42,
        max_candidates=32, validation_top_n=8,
        baseline_config_hash=BASELINE_HASH, recommended_config_hash=RECOMMENDED_HASH,
        recommendation_locked=True, recommendation_locked_at=TS, test_evaluated=True,
        split_manifest_json=json.dumps(_split_manifest(), ensure_ascii=False),
        outcome_json=json.dumps(_outcome(), ensure_ascii=False),
        generation_stats_json=json.dumps({"demo_fixture": True, "raw": 24, "executable": 12}),
        created_at=TS, completed_at=TS,
    )
    repo.upsert_search(search)  # 先鎖，才能寫 candidate 的 test_trial_id

    candidates = [
        SearchCandidate(
            search_id="search-demo", candidate_id="cand-baseline", ordinal=0,
            config_json=_config_json("current", "dense", 0.0, 0.55),
            generation_status="executable", config_hash=BASELINE_HASH, is_baseline=True,
            tune_status=CandidateStageStatus.COMPLETED,
            validation_status=CandidateStageStatus.COMPLETED,
            test_status=CandidateStageStatus.COMPLETED,
            test_trial_id="trial-baseline-test", feasible=True, pareto_member=True,
            pareto_rank=1, recommendation_rank=2,
        ),
        SearchCandidate(
            search_id="search-demo", candidate_id="cand-recommended", ordinal=1,
            config_json=_config_json("current", "identifier_aware", 0.5, 0.55, 0.2, 0.1),
            generation_status="executable", config_hash=RECOMMENDED_HASH,
            tune_status=CandidateStageStatus.COMPLETED,
            validation_status=CandidateStageStatus.COMPLETED,
            test_status=CandidateStageStatus.COMPLETED,
            test_trial_id="trial-rec-test", feasible=True, pareto_member=True,
            pareto_rank=1, recommendation_rank=1,
        ),
        SearchCandidate(
            search_id="search-demo", candidate_id="cand-tie-1", ordinal=2,
            config_json=_config_json("paragraph", "identifier_aware", 0.4, 0.55, 0.2, 0.1),
            generation_status="executable", config_hash="cfgtie0001aabbccddeeff",
            tune_status=CandidateStageStatus.COMPLETED,
            validation_status=CandidateStageStatus.COMPLETED,
            test_status=CandidateStageStatus.LOCKED_OUT,
            feasible=True, pareto_member=False, recommendation_rank=3,
        ),
        SearchCandidate(
            search_id="search-demo", candidate_id="cand-infeasible", ordinal=3,
            config_json=_config_json("heading", "identifier_aware", 0.4, 0.55, 0.2, 0.1),
            generation_status="executable", config_hash="cfginfeas0001aabbcc",
            tune_status=CandidateStageStatus.COMPLETED,
            validation_status=CandidateStageStatus.FAILED,
            test_status=CandidateStageStatus.LOCKED_OUT,
            feasible=False,
            rejection_reasons_json=json.dumps(["latency_p95_ms_exceeds_ceiling"]),
        ),
    ]
    for c in candidates:
        repo.upsert_candidate(c)


def _seed_review_db(path: str) -> None:
    repo = ReviewRepository(path)
    session = ReviewSession(
        session_id="review-demo", session_name="DEMO review session (fixture)",
        status=SESSION_REVIEWING,
        base_dataset_path="evaluation/datasets/v1.1/cases-v2.json",
        base_dataset_version=DATASET_VERSION, base_dataset_hash=DATASET_HASH,
        corpus_hash=CORPUS_HASH, canonical_text_version="v1", loader_version="v1",
        generator_type="fake",
        generator_config_json=json.dumps({"demo_fixture": True}),
        created_at=TS, updated_at=TS,
    )
    repo.create_session(session)

    specs = [
        ("cand-r1", "E102 與 D900 警報的差異為何？", "confusable", True,
         CANDIDATE_ACCEPTED, "alarm-e102", "test"),
        ("cand-r2", "設備保養時需要準備哪些工具？", "procedure", True,
         CANDIDATE_PROPOSED, "maint-cycle", "tune"),
        ("cand-r3", "CX-500 的保固期限是多久？", "unanswerable", False,
         CANDIDATE_REJECTED, "unanswerable-warranty", "validation"),
    ]
    for cid, q, qt, answerable, status, group, split in specs:
        cand = QuestionCandidate(
            candidate_id=cid, session_id="review-demo",
            proposed_question=q, proposed_query_type=qt, proposed_answerable=answerable,
            status=CANDIDATE_PROPOSED, current_revision=1,
            suggested_group_id=group, suggested_split=split,
            suggested_anchor_text_json=json.dumps(["E102"] if answerable else []),
            unanswerable_confirmed=not answerable,
            created_at=TS, updated_at=TS,
        )
        repo.create_candidate(cand)
        repo.add_revision(CandidateRevision(
            candidate_id=cid, revision_number=1, question=q, query_type=qt,
            answerable=answerable, group_id=group, split=split, created_at=TS,
        ))
        repo.add_event("review-demo", EVENT_CANDIDATE_CREATED,
                       candidate_id=cid, actor="local_user")
        if status == CANDIDATE_ACCEPTED:
            cand.status = CANDIDATE_ACCEPTED
            repo.update_candidate(cand)
            repo.add_event("review-demo", EVENT_CANDIDATE_ACCEPTED,
                           candidate_id=cid, actor="local_user")
        elif status == CANDIDATE_REJECTED:
            cand.status = CANDIDATE_REJECTED
            repo.update_candidate(cand)
            repo.add_event("review-demo", EVENT_CANDIDATE_REJECTED,
                           candidate_id=cid, actor="local_user")


def _seed_deployment_db(path: str) -> None:
    # 僅建立 schema（migration）；不插入任何 operation / record → alias 未初始化、history 為空。
    DeploymentRepository(path)


def build_demo_fixtures(output_dir: str, *, overwrite: bool = False) -> dict[str, str]:
    """在 ``output_dir`` 產生三個 fixture DB；回傳 {search,review,deployment} → 路徑。

    - ``output_dir`` 必須明確傳入（§十三.1）。
    - 既有檔案預設**拒絕覆寫**（§十三.2），除非 ``overwrite=True``。
    - 產生的 DB 檔應放 ignored 路徑（呼叫端負責），本函式不入庫任何內容。
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "search": os.path.join(output_dir, SEARCH_DB_NAME),
        "review": os.path.join(output_dir, REVIEW_DB_NAME),
        "deployment": os.path.join(output_dir, DEPLOYMENT_DB_NAME),
    }
    if not overwrite:
        existing = [p for p in paths.values() if os.path.exists(p)]
        if existing:
            raise FileExistsError(
                "拒絕覆寫既有 fixture（加 --overwrite 才會重建）：" + ", ".join(existing)
            )
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
    _seed_search_db(paths["search"])
    _seed_review_db(paths["review"])
    _seed_deployment_db(paths["deployment"])
    return paths
