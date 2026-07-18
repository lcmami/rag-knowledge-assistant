"""前端自動化 Demo 腳本（僅供 Demo，不修改應用程式行為）。

使用 Playwright Async API、headed 模式、slow_mo 放慢動作，
操作既有網站（http://localhost:8000）並留下截圖與錄影證據。

流程：
  1. 開啟首頁，確認標題 / 三個分頁 / LLM 未設定提示。
  2. 展示現有兩份正式文件（indexed / chunk 5）。
  3. 建立唯一內容的暫時 Markdown 並透過前端上傳。
  4. E102 向量檢索（Top-1 為 equipment_alarm_manual.md）。
  5. 無關問題檢索（0 筆，顯示 threshold 提示）。
  6. RAG 問答（503 LLM_NOT_CONFIGURED，前端顯示固定提示）。
  7. 刪除本次暫時文件，清理本機檔案，回到 2 份。

執行：
  .venv/Scripts/python.exe scripts/demo_frontend.py
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import async_playwright

BASE_URL = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent
SHOT_DIR = ROOT / "docs" / "screenshots" / "demo-run"
VIDEO_DIR = SHOT_DIR / "video"
STORAGE_DIR = ROOT / "storage"
TEMP_FILE = STORAGE_DIR / "demo_frontend_upload.md"
TEMP_NAME = "demo_frontend_upload.md"

OFFICIAL_DOCS = {"equipment_alarm_manual.md", "equipment_maintenance_manual.pdf"}

# 放慢每步間隔（毫秒），並用於「停留」秒數轉換。
STEP_PAUSE_MS = 900


def log(msg: str) -> None:
    print(f"[demo] {msg}", flush=True)


def build_temp_markdown(extra: str = "") -> str:
    run_id = datetime.now(timezone.utc).isoformat()
    # 內容刻意選用與「E102 警報處理」語意無關的主題（展示台清潔），
    # 以免暫時文件在 E102 檢索中誤排到 Top-1；仍為唯一內容並帶 demo_run_id。
    return (
        "# Demo 展示台清潔紀錄\n\n"
        "本文件為前端自動化展示所建立的暫時虛構資料，內容與任何警報代碼無關。\n\n"
        "## Z1 展示台每日清潔\n\n"
        "Z1 展示台僅用於前端 Demo 錄影，不連接任何實際產線或警報系統。\n\n"
        "清潔項目：\n\n"
        "1. 以乾布擦拭展示台外殼表面。\n"
        "2. 檢查展示用指示燈是否正常點亮。\n"
        "3. 記錄本次 Demo 的執行時間並歸檔。\n\n"
        f"demo_run_id: {run_id}\n"
        f"{extra}"
    )


async def screenshot(page, name: str) -> None:
    path = SHOT_DIR / name
    await page.screenshot(path=str(path), full_page=True)
    log(f"截圖：{path}")


async def pause(page, seconds: float) -> None:
    await page.wait_for_timeout(int(seconds * 1000))


async def run() -> int:
    STORAGE_DIR.mkdir(exist_ok=True)
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # 產生唯一內容暫時檔案（避免與過去 SHA-256 重複）。
    TEMP_FILE.write_text(build_temp_markdown(), encoding="utf-8")
    log(f"已建立暫時檔案：{TEMP_FILE}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, slow_mo=STEP_PAUSE_MS)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 1000},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 1440, "height": 1000},
        )
        page = await context.new_page()

        # 接受瀏覽器 confirm（刪除時）。
        page.on("dialog", lambda d: asyncio.ensure_future(d.accept()))

        temp_doc_id = None
        try:
            # ---- Step 1：首頁 ----
            log("Step 1：開啟首頁")
            await page.goto(BASE_URL, wait_until="networkidle")
            await page.wait_for_selector("h1:has-text('Factory RAG')")
            # 等待 /ready 與文件列表載入（docs-msg 會顯示「共 N 份文件」）。
            await page.wait_for_selector("#docs-msg:has-text('份文件')", timeout=15000)
            await page.wait_for_selector("#llm-banner", state="visible", timeout=15000)

            for txt in ["Factory RAG 知識助理", "文件管理", "向量檢索", "RAG 問答"]:
                assert await page.locator(f"text={txt}").count() > 0, f"缺少文字：{txt}"
            banner = page.locator("#llm-banner")
            assert await banner.is_visible(), "LLM 未設定提示未顯示"
            log(f"LLM banner 文字：{(await banner.inner_text()).strip()}")
            await pause(page, 2)
            await screenshot(page, "01-home.png")
            await pause(page, 2)

            # ---- Step 2：現有文件列表 ----
            log("Step 2：展示現有文件列表")
            await page.get_by_role("tab", name="文件管理").click()
            await page.wait_for_selector("#docs-table tbody tr")
            rows = page.locator("#docs-tbody tr")
            row_count = await rows.count()
            log(f"文件列表列數：{row_count}")
            for name in OFFICIAL_DOCS:
                assert await page.locator(f"#docs-tbody tr:has-text('{name}')").count() == 1, (
                    f"找不到正式文件：{name}"
                )
                row = page.locator(f"#docs-tbody tr:has-text('{name}')")
                row_text = await row.inner_text()
                assert "indexed" in row_text, f"{name} 狀態非 indexed：{row_text}"
                assert await row.locator("button.danger").count() == 1, f"{name} 缺刪除按鈕"
                log(f"  {name} → {row_text.strip().splitlines()[-1] if False else 'indexed'} / row ok")
            await pause(page, 2)
            await screenshot(page, "02-document-list.png")
            await pause(page, 2)

            # ---- Step 3：上傳暫時文件 ----
            log("Step 3：上傳暫時 Demo 文件")
            await page.locator("#file-input").set_input_files(str(TEMP_FILE))
            await page.locator("#upload-btn").click()
            # 等待成功或 409。
            await page.wait_for_selector("#upload-msg.ok, #upload-msg.error", timeout=30000)
            upload_msg = (await page.locator("#upload-msg").inner_text()).strip()
            log(f"上傳訊息：{upload_msg}")

            if "已存在" in upload_msg or "409" in upload_msg or "重複" in upload_msg:
                # 409：重新產生不同 run_id，只重試一次。
                log("偵測到重複，重試一次（新 run_id）")
                TEMP_FILE.write_text(build_temp_markdown(extra=f"retry: {datetime.now(timezone.utc).isoformat()}\n"), encoding="utf-8")
                await page.locator("#file-input").set_input_files(str(TEMP_FILE))
                await page.locator("#upload-btn").click()
                await page.wait_for_selector("#upload-msg.ok, #upload-msg.error", timeout=30000)
                upload_msg = (await page.locator("#upload-msg").inner_text()).strip()
                log(f"重試上傳訊息：{upload_msg}")

            assert "已索引" in upload_msg, f"上傳未成功：{upload_msg}"
            # 確認列表出現暫時文件並取得 document_id。
            temp_row = page.locator(f"#docs-tbody tr:has-text('{TEMP_NAME}')")
            await temp_row.wait_for(timeout=10000)
            temp_doc_id = await temp_row.locator("button.danger").get_attribute("data-id")
            log(f"暫時文件 document_id：{temp_doc_id}")
            await pause(page, 3)
            await screenshot(page, "03-upload-success.png")
            await pause(page, 1)

            # ---- Step 4：E102 檢索 ----
            log("Step 4：E102 向量檢索")
            await page.get_by_role("tab", name="向量檢索").click()
            await page.locator("#retrieve-q").fill("設備出現 E102 警報時，應該如何處理？")
            await page.locator("#retrieve-topk").fill("5")
            await page.locator("#retrieve-btn").click()
            await page.wait_for_selector("#retrieve-msg.ok, #retrieve-msg.error", timeout=30000)
            r_msg = (await page.locator("#retrieve-msg").inner_text()).strip()
            log(f"檢索訊息：{r_msg}")
            assert "命中" in r_msg, f"E102 檢索失敗：{r_msg}"
            cards = page.locator("#retrieve-results .result-card")
            assert await cards.count() > 0, "E102 檢索結果為空"
            top1 = cards.first
            top1_text = await top1.inner_text()
            log(f"Top-1：{top1_text.strip()[:120]}")
            assert "equipment_alarm_manual.md" in top1_text, f"Top-1 filename 非預期：{top1_text}"
            await top1.scroll_into_view_if_needed()
            await pause(page, 4)
            await screenshot(page, "04-retrieval-e102.png")
            await pause(page, 1)

            # ---- Step 5：無關問題（0 筆）----
            log("Step 5：無關問題檢索（threshold 拒絕）")
            await page.locator("#retrieve-q").fill("")
            await page.locator("#retrieve-q").fill("台灣的首都是哪一個城市？")
            await page.locator("#retrieve-btn").click()
            await page.wait_for_selector("#retrieve-msg.ok, #retrieve-msg.error", timeout=30000)
            r2_msg = (await page.locator("#retrieve-msg").inner_text()).strip()
            log(f"檢索訊息：{r2_msg}")
            assert "命中 0" in r2_msg, f"預期 0 筆，實際：{r2_msg}"
            assert "ms" in r2_msg, "缺 retrieval latency"
            no_result = page.locator("#retrieve-results .result-card")
            assert await no_result.count() == 1, "應顯示無結果提示卡"
            assert "沒有通過 threshold" in (await no_result.inner_text()), "缺 threshold 提示"
            await pause(page, 3)
            await screenshot(page, "05-retrieval-no-result.png")
            await pause(page, 1)

            # ---- Step 6：RAG 問答（未設定 LLM）----
            log("Step 6：RAG 問答（LLM 未設定）")
            await page.get_by_role("tab", name="RAG 問答").click()
            await page.locator("#query-q").fill("設備出現 E102 警報時，應該如何處理？")
            await page.locator("#query-btn").click()
            await page.wait_for_selector("#query-msg.warn, #query-msg.error, #query-msg.ok", timeout=30000)
            q_msg = (await page.locator("#query-msg").inner_text()).strip()
            log(f"RAG 訊息：{q_msg}")
            assert "LLM 服務尚未設定" in q_msg, f"未顯示 LLM 未設定提示：{q_msg}"
            # 不應有偽造回答或 sources。
            assert await page.locator("#query-answer").is_hidden(), "不應顯示回答"
            assert await page.locator("#query-sources .result-card").count() == 0, "不應顯示偽造來源"
            await pause(page, 4)
            await screenshot(page, "06-llm-not-configured.png")
            await pause(page, 1)

            # ---- Step 7：刪除暫時文件 ----
            log("Step 7：刪除本次暫時文件")
            await page.get_by_role("tab", name="文件管理").click()
            await page.wait_for_selector(f"#docs-tbody tr:has-text('{TEMP_NAME}')", timeout=10000)
            del_row = page.locator(f"#docs-tbody tr:has-text('{TEMP_NAME}')")
            await del_row.locator("button.danger").click()  # dialog 自動 accept
            # 等待列表回到 2 份且暫時文件消失。
            await page.wait_for_selector(
                f"#docs-tbody tr:has-text('{TEMP_NAME}')", state="detached", timeout=15000
            )
            await page.wait_for_selector("#docs-msg:has-text('共 2 份文件')", timeout=10000)
            final_msg = (await page.locator("#docs-msg").inner_text()).strip()
            log(f"最終文件列表訊息：{final_msg}")
            for name in OFFICIAL_DOCS:
                assert await page.locator(f"#docs-tbody tr:has-text('{name}')").count() == 1, (
                    f"正式文件遺失：{name}"
                )
            assert await page.locator(f"#docs-tbody tr:has-text('{TEMP_NAME}')").count() == 0
            await pause(page, 2)
            await screenshot(page, "07-final-document-list.png")
            await pause(page, 1)

            log("Demo 全部步驟完成")
            result = 0
        except Exception as exc:  # noqa: BLE001
            log(f"步驟失敗：{exc!r}")
            try:
                await page.screenshot(path=str(SHOT_DIR / "failure.png"), full_page=True)
                log(f"已保留失敗截圖：{SHOT_DIR / 'failure.png'}")
            except Exception as se:  # noqa: BLE001
                log(f"保留失敗截圖時再次出錯：{se!r}")
            result = 1
        finally:
            # 正常關閉，確保錄影檔完整寫入。
            video_path = None
            try:
                video_path = await page.video.path() if page.video else None
            except Exception:  # noqa: BLE001
                pass
            await context.close()
            await browser.close()
            if video_path:
                log(f"錄影檔：{video_path}")

    # 清理本機暫時檔案。
    try:
        if TEMP_FILE.exists():
            TEMP_FILE.unlink()
            log(f"已刪除本機暫時檔案：{TEMP_FILE}")
    except Exception as e:  # noqa: BLE001
        log(f"刪除暫時檔案失敗：{e!r}")

    return result


if __name__ == "__main__":
    try:
        code = asyncio.run(run())
    except KeyboardInterrupt:
        log("偵測到人工中斷（KeyboardInterrupt），腳本結束。")
        code = 2
    sys.exit(code)
