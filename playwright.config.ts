import { defineConfig, devices } from "@playwright/test";

// Slice 8 E2E：以真實 FastAPI 服務靜態前端，API 由各測試以 page.route 「mock／isolate」，
// 因此不需真實 Qdrant / LLM / DB，且測試間互不影響（§十六：唯一 port、無固定 sleep、
// 以 locator/response 等待、測後自動清理）。
const PORT = Number(process.env.OPT_E2E_PORT || 8099);
// Windows cmd.exe 不接受開頭帶正斜線的相對執行檔路徑；預設用平台對應的 venv python。
const PYTHON =
  process.env.OPT_E2E_PYTHON ||
  (process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python");

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "off",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `${PYTHON} -m uvicorn app.main:app --host 127.0.0.1 --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/static/index.html`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
    env: {
      SKIP_STARTUP_SERVICES: "1",
      PYTHONPATH: ".",
    },
  },
});
