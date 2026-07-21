import { Page, Route } from "@playwright/test";

// 這些 payload 形狀對應 OptimizerReportingStore 的實際輸出（read-only API），
// 但為 E2E **mock**：明確標示 demo，不觸及真實 Qdrant / DB / LLM。

export function demoOverview(over: any = {}) {
  return {
    is_demo_fixture: true,
    search: {
      configured: true, available: true, reason: null,
      dataset_version: "v1.1-coverage-0002", corpus_hash_short: "corpusdemo00",
      case_count: 16, group_count: 11,
      split_counts: { tune: 4, validation: 5, test: 7 },
      coverage_warnings: ["validation:missing_query_type:identifier_bearing_unanswerable"],
      latest_search: {
        search_id: "search-demo", status: "test_constraint_failed",
        executable_candidate_count: 4, feasible_candidate_count: 3, pareto_count: 2,
        recommended_config_hash: "e246697d41aa0011", recommended_config_hash_short: "e246697d41aa",
        recommendation_locked: true, apply_eligible: false, confidence: "low",
        test_constraint_status: "failed", baseline_comparison_status: "improved",
        recommendation_selection_basis: "validation_latency_tiebreak",
        quality_tie_candidate_count: 8,
        apply_ineligible_reasons: [
          "small_validation_sample", "quality_metrics_tied",
          "selection_driven_by_single_run_latency", "test_constraint_failed", "demo_corpus_only",
        ],
      },
    },
    review: { configured: true, available: true, reason: null, session_count: 1, sessions: [] },
    deployment: {
      configured: true, available: true, reason: null,
      admin_write_enabled: false, active_alias: "factory_knowledge_active",
      alias_initialized: false, history_count: 0,
    },
    ...over,
  };
}

export function unavailableOverview() {
  const un = { configured: false, available: false, reason: "optimizer_data_not_available" };
  return {
    is_demo_fixture: false,
    search: { ...un, latest_search: null, split_counts: {}, coverage_warnings: [] },
    review: { ...un, session_count: 0, sessions: [] },
    deployment: { ...un, admin_write_enabled: false, active_alias: "factory_knowledge_active",
      alias_initialized: false, history_count: 0 },
  };
}

export function searchesList() {
  return { configured: true, available: true, reason: null,
    searches: [{ search_id: "search-demo", status: "test_constraint_failed" }] };
}

export function candidates() {
  return {
    configured: true, available: true, reason: null, found: true,
    candidates: [
      { candidate_id: "cand-baseline", is_baseline: true, chunk_strategy: "current",
        retrieval_strategy: "dense", identifier_vector_floor: 0.0, score_threshold: 0.55,
        exact_match_boost: null, confusable_penalty: null, tune_status: "completed",
        validation_status: "completed", constraint_status: "pass", feasible: true,
        pareto_member: true, recommendation_rank: 2 },
      { candidate_id: "cand-recommended", is_baseline: false, chunk_strategy: "current",
        retrieval_strategy: "identifier_aware", identifier_vector_floor: 0.5, score_threshold: 0.55,
        exact_match_boost: 0.2, confusable_penalty: 0.1, tune_status: "completed",
        validation_status: "completed", constraint_status: "pass", feasible: true,
        pareto_member: true, recommendation_rank: 1 },
    ],
  };
}

export function comparison() {
  return {
    configured: true, available: true, reason: null, found: true,
    comparison: {
      search_id: "search-demo", baseline_comparison_status: "improved",
      metrics: [
        { metric_name: "exact_code_top1_accuracy", scope: "aggregate", query_type: null,
          baseline_value: 0.5, candidate_value: 0.5, delta: 0.0, direction: "higher_is_better",
          sample_count: 2, baseline_computable: true, candidate_computable: true },
        { metric_name: "ndcg_at_5", scope: "aggregate", query_type: null,
          baseline_value: 0.8367, candidate_value: 0.9193, delta: 0.0826, direction: "higher_is_better",
          sample_count: 7, baseline_computable: true, candidate_computable: true },
        { metric_name: "latency_p95_ms", scope: "aggregate", query_type: null,
          baseline_value: 47.6, candidate_value: 67.9, delta: 20.3, direction: "lower_is_better",
          sample_count: 7, baseline_computable: true, candidate_computable: true },
      ],
      per_case: {
        label: "test", improved: 1, same: 1, regressed: 0,
        cases: [
          { case_id: "eval-e102-001", question: "設備出現 E102 感測器通訊異常時應如何處理？",
            query_type: "exact_code", baseline_first_relevant_rank: 2,
            candidate_first_relevant_rank: 2, verdict: "same",
            baseline_top5: [{ rank: 1, source: "equipment_alarm_manual.md", section: "D900 與 E102 的差異",
              vector_score: 0.71, ranking_score: 0.91, relevance_level: 0, admission_reason: "exact_identifier_match" }],
            candidate_top5: [{ rank: 2, source: "equipment_alarm_manual.md", section: "E102 感測器通訊異常",
              vector_score: 0.66, ranking_score: 0.86, relevance_level: 3, admission_reason: "exact_identifier_match" }] },
        ],
      },
    },
  };
}

export function reviews() {
  return {
    configured: true, available: true, reason: null,
    sessions: [{ session_id: "review-demo", session_name: "DEMO review session (fixture)",
      status: "reviewing", base_dataset_version: "v1.1-coverage-0002", revision_count: 3,
      counts: { proposed: 1, accepted: 1, rejected: 1, published: 0 } }],
  };
}

export function reviewDetail() {
  return {
    configured: true, available: true, reason: null, found: true,
    session: {
      session_id: "review-demo", session_name: "DEMO review session (fixture)", status: "reviewing",
      base_dataset_version: "v1.1-coverage-0002", revision_count: 3,
      counts: { proposed: 1, accepted: 1, rejected: 1, published: 0 },
      candidates: [
        { candidate_id: "cand-r1", question: "E102 與 D900 警報的差異為何？", query_type: "confusable",
          answerable: true, status: "accepted", suggested_split: "test", evidence_count: 1, revision_count: 1 },
      ],
      audit_events: [{ event_type: "candidate_created", candidate_id: "cand-r1", created_at: "2026-07-20T00:00:00+00:00" }],
    },
  };
}

export function coverage() {
  return {
    configured: true, available: true, reason: null,
    coverage: { dataset_version: "v1.1-coverage-0002", corpus_hash_short: "corpusdemo00",
      split_manifest: {}, coverage_warnings: ["validation:missing_query_type:identifier_bearing_unanswerable"],
      apply_ineligible_reasons: [] },
  };
}

export function deploymentSummary(over: any = {}) {
  return {
    configured: true, available: true, reason: null,
    active_alias: "factory_knowledge_active", alias_initialized: false,
    current_collection: null, previous_collection: null, history_count: 0,
    records: [], operations: [], admin_write_enabled: false, ...over,
  };
}

export function aliasStatus(over: any = {}) {
  return { alias: "factory_knowledge_active", alias_target: null, db_active_target: null,
    initialized: false, ...over };
}

// 安裝一組預設 route（可用 overrides 覆寫個別端點）。
export async function installRoutes(page: Page, overrides: Record<string, any> = {}) {
  const json = (route: Route, body: any, status = 200) =>
    route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

  // app.js 開場會打這些；直接 mock 掉避免依賴真實服務。
  await page.route("**/ready", (r) => json(r, { checks: {} }));
  await page.route("**/api/documents", (r) => json(r, { total: 0, documents: [] }));

  await page.route("**/api/optimizer/overview", (r) =>
    json(r, overrides.overview ?? demoOverview()));
  await page.route("**/api/optimizer/searches", (r) =>
    json(r, overrides.searches ?? searchesList()));
  await page.route("**/api/optimizer/searches/*/candidates", (r) =>
    json(r, overrides.candidates ?? candidates()));
  await page.route("**/api/optimizer/searches/*/comparison", (r) =>
    json(r, overrides.comparison ?? comparison()));
  await page.route("**/api/optimizer/reviews", (r) =>
    json(r, overrides.reviews ?? reviews()));
  await page.route("**/api/optimizer/reviews/*", (r) =>
    json(r, overrides.reviewDetail ?? reviewDetail()));
  await page.route("**/api/optimizer/coverage", (r) =>
    json(r, overrides.coverage ?? coverage()));
  await page.route("**/api/optimizer/deployment/summary", (r) =>
    json(r, overrides.deploymentSummary ?? deploymentSummary()));
  await page.route("**/api/optimizer/alias/status", (r) =>
    json(r, overrides.aliasStatus ?? aliasStatus()));
}

export async function openOptimizer(page: Page) {
  await page.goto("/static/index.html");
  await page.click('.tab[data-tab="optimizer"]');
}
