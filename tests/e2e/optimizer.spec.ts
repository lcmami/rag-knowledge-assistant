import { test, expect } from "@playwright/test";
import * as F from "./fixtures";

// Slice 8 §十六 Retrieval Optimizer UI E2E。API 一律 mock；不碰真實 Qdrant / LLM / DB。

test.describe("Retrieval Optimizer UI", () => {
  // 16.1 無 optimizer DB
  test("16.1 no optimizer data: tab opens, shows unavailable, other tabs unaffected", async ({ page }) => {
    await F.installRoutes(page, {
      overview: F.unavailableOverview(),
      searches: { configured: false, available: false, reason: "optimizer_data_not_available", searches: [] },
    });
    await F.openOptimizer(page);
    const overview = page.locator("#opt-subpanel-overview");
    await expect(overview.getByText(/optimizer_data_not_available|尚未設定/)).toBeVisible();
    // App 沒 crash：其他三個既有 tab 仍可切換
    await page.click('.tab[data-tab="documents"]');
    await expect(page.locator("#panel-documents")).toBeVisible();
    await page.click('.tab[data-tab="retrieve"]');
    await expect(page.locator("#panel-retrieve")).toBeVisible();
  });

  // 16.2 Demo fixture overview
  test("16.2 demo overview: dataset / recommendation locked / apply not eligible / demo badge", async ({ page }) => {
    await F.installRoutes(page);
    await F.openOptimizer(page);
    const ov = page.locator("#opt-subpanel-overview");
    await expect(ov.getByText("Demo fixture")).toBeVisible();
    await expect(ov.getByText("v1.1-coverage-0002")).toBeVisible();
    await expect(ov.locator(".badge", { hasText: "Recommendation locked" })).toBeVisible();
    await expect(ov.getByText("Apply not eligible")).toBeVisible();
    await expect(ov.getByText(/Test constraints failed/)).toBeVisible();
    await expect(ov.getByText("small_validation_sample")).toBeVisible();
    await expect(ov.getByText("Admin write disabled")).toBeVisible();
  });

  // 16.3 Search results
  test("16.3 search results: candidate table, metrics comparison, per-case C01 FRR=2", async ({ page }) => {
    await F.installRoutes(page);
    await F.openOptimizer(page);
    await page.click('.subtab[data-sub="search"]');
    const sp = page.locator("#opt-subpanel-search");
    await expect(sp.getByText("cand-recommended")).toBeVisible();
    await expect(sp.getByText("identifier_aware").first()).toBeVisible();
    await expect(sp.getByText("ndcg_at_5")).toBeVisible();
    await expect(sp.getByText("latency_p95_ms")).toBeVisible();
    // C01 誠實：first relevant rank 保持 2 → verdict same
    await expect(sp.getByText(/baseline FRR 2 → candidate FRR 2/)).toBeVisible();
    await expect(sp.locator(".opt-case-head").getByText("same")).toBeVisible();
  });

  // 16.4 Human review
  test("16.4 human review: session list, counts, coverage gaps, no auto-accept/publish button", async ({ page }) => {
    await F.installRoutes(page);
    await F.openOptimizer(page);
    await page.click('.subtab[data-sub="review"]');
    const rp = page.locator("#opt-subpanel-review");
    await expect(rp.getByText("DEMO review session (fixture)").first()).toBeVisible();
    await expect(rp.getByText(/identifier_bearing_unanswerable/)).toBeVisible();
    // 不得有 auto-accept / batch / publish / LLM 生成按鈕
    await expect(rp.getByRole("button", { name: /accept|publish|generate|批次|接受|發布|生成/i })).toHaveCount(0);
  });

  // 16.5 Deployment admin disabled
  test("16.5 deployment admin disabled: message + disabled buttons + eligibility reasons", async ({ page }) => {
    await F.installRoutes(page);
    await F.openOptimizer(page);
    await page.click('.subtab[data-sub="deploy"]');
    const dp = page.locator("#opt-subpanel-deploy");
    await expect(dp.getByText("Administrative write operations are disabled.")).toBeVisible();
    await expect(dp.locator('[data-testid="apply-blocked"]')).toBeVisible();
    await expect(dp.getByText("selection_driven_by_single_run_latency")).toBeVisible();
    // Initialize / Rollback / Reconcile 全部 disabled
    for (const op of ["initialize", "rollback", "reconcile"]) {
      await expect(dp.locator(`.opt-action[data-op="${op}"]`)).toBeDisabled();
    }
    // Apply 的 confirmation 送出鈕 disabled（admin 停用）
    await expect(dp.locator('.opt-confirm-submit[data-op="apply"]')).toBeDisabled();
  });

  // 16.6 Explicit confirmation（admin enabled + eligible 的 isolated mock）
  test("16.6 typed confirmation: enable on exact text, payload has idempotency_key, success re-fetch, error shown", async ({ page }) => {
    const eligibleLatest = {
      search_id: "search-demo", status: "completed", executable_candidate_count: 4,
      feasible_candidate_count: 3, pareto_count: 2,
      recommended_config_hash: "cfg_elig_0001", recommended_config_hash_short: "cfg_elig_0001",
      recommendation_locked: true, apply_eligible: true, confidence: "medium",
      test_constraint_status: "passed", baseline_comparison_status: "improved",
      recommendation_selection_basis: "validation_quality_ranking", quality_tie_candidate_count: 1,
      apply_ineligible_reasons: [],
    };
    const overview = F.demoOverview({
      search: { ...F.demoOverview().search, latest_search: eligibleLatest },
      deployment: { ...F.demoOverview().deployment, admin_write_enabled: true,
        alias_initialized: true },
    });
    const summary = F.deploymentSummary({
      admin_write_enabled: true, alias_initialized: true,
      current_collection: "factory_knowledge_v1_demo", history_count: 1,
    });

    let applyRequestBody: any = null;
    let applyCalls = 0;
    await F.installRoutes(page, { overview, deploymentSummary: summary,
      aliasStatus: F.aliasStatus({ alias_target: "factory_knowledge_v1_demo", initialized: true }) });
    await page.route("**/api/optimizer/candidates/*/apply", (route) => {
      applyCalls += 1;
      applyRequestBody = route.request().postDataJSON();
      if (applyCalls === 1) {
        // 第一次：模擬後端錯誤，錯誤訊息要顯示
        return route.fulfill({ status: 409, contentType: "application/json",
          body: JSON.stringify({ error: { code: "stale_alias_target", message: "alias moved" } }) });
      }
      return route.fulfill({ status: 200, contentType: "application/json",
        body: JSON.stringify({ operation: { status: "completed" } }) });
    });

    await F.openOptimizer(page);
    await page.click('.subtab[data-sub="deploy"]');
    const form = page.locator('.opt-confirm[data-op="apply"]');
    const input = form.locator(".opt-confirm-input");
    const submit = form.locator(".opt-confirm-submit");

    // 輸入不完整 → submit disabled
    await input.fill("APPLY wrong");
    await expect(submit).toBeDisabled();

    // 完整輸入後端要求的精確字串 → enabled
    const expected = "APPLY factory_knowledge_active CONFIG cfg_elig_0001 FROM factory_knowledge_v1_demo";
    await input.fill(expected);
    await expect(submit).toBeEnabled();

    // 送出 → payload 正確且含 idempotency_key，錯誤訊息顯示
    await submit.click();
    await expect(form.locator(".opt-confirm-msg")).toContainText("stale_alias_target");
    expect(applyRequestBody.confirmation).toBe(expected);
    expect(typeof applyRequestBody.idempotency_key).toBe("string");
    expect(applyRequestBody.idempotency_key.length).toBeGreaterThan(0);

    // 重試（沿用同一 idempotency key）→ 成功 → 重新 fetch 狀態
    const firstIdem = applyRequestBody.idempotency_key;
    await expect(submit).toBeEnabled();
    const okResp = page.waitForResponse(
      (r) => r.url().includes("/apply") && r.status() === 200);
    await submit.click();
    await okResp;
    expect(applyCalls).toBe(2);
    expect(applyRequestBody.idempotency_key).toBe(firstIdem);
  });
});
