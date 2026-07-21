// 擷取 Retrieval Optimizer UI 的 demo 截圖（Slice 8 §十九）。
//
// 使用真實 FastAPI + 真實 read-only API，資料來自 seeder 產生的 demo fixture
// （demo_fixture=true，非真實 production run）。不需 Qdrant / LLM。
//
// 前置：先產生 fixture：
//   .venv/Scripts/python -m scripts.seed_optimizer_ui_demo --output-dir evaluation/demo --overwrite
// 執行：
//   node scripts/capture_optimizer_screenshots.mjs
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { chromium } from "@playwright/test";

const PORT = 8097;
const PY = process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";
const OUT = "docs/screenshots";
const BASE = `http://127.0.0.1:${PORT}`;

const env = {
  ...process.env,
  SKIP_STARTUP_SERVICES: "1",
  PYTHONPATH: ".",
  OPTIMIZER_SEARCH_DB_PATH: "evaluation/demo/optimizer-demo.search.db",
  OPTIMIZER_REVIEW_DB_PATH: "evaluation/demo/optimizer-demo.review.db",
  OPTIMIZER_DEPLOYMENT_DB_PATH: "evaluation/demo/optimizer-demo.deployment.db",
};

const server = spawn(PY, ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(PORT)], {
  env, stdio: "inherit",
});

async function waitReady() {
  for (let i = 0; i < 60; i++) {
    try {
      const r = await fetch(`${BASE}/static/index.html`);
      if (r.ok) return;
    } catch (_) { /* not up yet */ }
    await sleep(500);
  }
  throw new Error("server did not start");
}

async function main() {
  await waitReady();
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
  await page.goto(`${BASE}/static/index.html`);
  await page.click('.tab[data-tab="optimizer"]');
  await page.waitForSelector("#opt-subpanel-overview .opt-card");
  await page.screenshot({ path: `${OUT}/v1.1-optimizer-overview.png`, fullPage: true });

  await page.click('.subtab[data-sub="search"]');
  await page.waitForSelector("#opt-subpanel-search .opt-table");
  await page.screenshot({ path: `${OUT}/v1.1-search-comparison.png`, fullPage: true });

  await page.click('.subtab[data-sub="review"]');
  await page.waitForSelector("#opt-subpanel-review .opt-table");
  await page.screenshot({ path: `${OUT}/v1.1-human-review.png`, fullPage: true });

  await page.click('.subtab[data-sub="deploy"]');
  await page.waitForSelector('[data-testid="apply-blocked"]');
  await page.screenshot({ path: `${OUT}/v1.1-deployment-blocked.png`, fullPage: true });

  await browser.close();
  console.log("screenshots written to", OUT);
}

main()
  .catch((e) => { console.error(e); process.exitCode = 1; })
  .finally(() => { server.kill(); });
