"""可重現的虛構 PDF 產生腳本（Phase 4 §4）。

產生 sample_docs/equipment_maintenance_manual.pdf：
- 內容為自行撰寫的虛構資料，頂端含作品集聲明。
- 至少 3 頁，刻意安排各頁不同主題，供頁碼驗證：
    第 1 頁：文件聲明與設備基本資訊
    第 2 頁：定期保養週期（刻意放入多個相近但不同的週期）
    第 3 頁：異常保養與通訊模組檢查

字型策略（§4）：
- 使用 PyMuPDF 內建的「cjk」字型（Droid Sans Fallback），隨 pymupdf 套件散佈，
  可跨環境穩定處理中文，且不需將任何字型檔提交到 Repository。
- 若執行環境無法提供該字型，直接以清楚錯誤中止，不靜默產生亂碼 PDF。

執行：
    python scripts/generate_sample_pdf.py
"""
from __future__ import annotations

import os
import sys

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_docs",
    "equipment_maintenance_manual.pdf",
)

DISCLAIMER = "本文件為作品集示範所建立的虛構資料，不代表任何真實公司或設備。"

# 每頁：list[(tag, text)]，tag 為 "h2"（標題→section）或 "p"（段落）。
PAGES: list[list[tuple[str, str]]] = [
    # --- 第 1 頁：文件聲明與設備基本資訊 ---
    [
        ("disclaimer", DISCLAIMER),
        ("h2", "設備基本資訊"),
        ("p", "本手冊適用於 CX-500 高速貼片機，為虛構示範設備，用於說明保養流程。"),
        ("p", "CX-500 的額定功率為 3.2 千瓦，適用於中小型 SMT 表面黏著產線。"),
        ("p", "設備控制器韌體版本為 v2.4，並搭配 CM-12 通訊模組與雙軸伺服系統。"),
        ("p", "建議操作環境溫度為攝氏 18 度至 28 度，相對濕度維持在百分之 40 至 60。"),
    ],
    # --- 第 2 頁：定期保養週期（多個相近但不同的週期，供混淆測試）---
    [
        ("h2", "定期保養週期"),
        ("p", "CX-500 的基本保養每運轉 500 小時執行一次，包含外觀清潔與螺絲扭力複查。"),
        ("p", "空氣過濾網屬於高頻耗材，每運轉 250 小時必須清潔一次，必要時更換。"),
        ("p", "潤滑系統每運轉 1000 小時檢查油量並補充指定規格潤滑油。"),
        ("p", "傳動皮帶每運轉 2000 小時檢查張力與磨耗，張力不足時立即調整。"),
        ("p", "所有定期保養紀錄須登錄於保養履歷，並由當班技術員簽核。"),
    ],
    # --- 第 3 頁：異常保養與通訊模組檢查 ---
    [
        ("h2", "通訊模組檢查"),
        ("p", "當設備出現通訊中斷或封包遺失時，應優先檢查 CM-12 通訊模組的指示燈狀態。"),
        ("p", "CM-12 通訊模組每運轉 3000 小時進行一次預防性更換，以避免突發性斷線。"),
        ("p", "若指示燈為紅色恆亮，代表通訊模組故障，須立即更換並記錄故障時間。"),
        ("h2", "異常保養"),
        ("p", "當同一設備在 24 小時內連續發生三次通訊警報時，須啟動異常保養程序。"),
        ("p", "異常保養須由資深技術員執行完整通訊鏈路檢測，並保留檢測報告至少一年。"),
    ],
]

# 版面參數
_MARGIN = 50
_PAGE_W, _PAGE_H = 595, 842  # A4（point）
_CSS = (
    "@font-face{font-family:CJK;src:url(cjk.ttf);}"
    "*{font-family:CJK;line-height:1.6;}"
    "h2{font-size:18px;margin:14px 0 6px 0;}"
    "p{font-size:12px;margin:0 0 8px 0;}"
    ".disclaimer{font-size:11px;margin:0 0 10px 0;}"
)


def _build_html(blocks: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for tag, text in blocks:
        if tag == "h2":
            parts.append(f"<h2>{text}</h2>")
        elif tag == "disclaimer":
            parts.append(f'<p class="disclaimer">{text}</p>')
        else:
            parts.append(f"<p>{text}</p>")
    return "".join(parts)


def _load_cjk_font():
    """取得可穩定處理中文的內建字型；不可用時給出清楚錯誤。"""
    try:
        import pymupdf
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("需要安裝 pymupdf 才能產生 PDF：pip install pymupdf") from exc

    try:
        font = pymupdf.Font("cjk")  # Droid Sans Fallback，隨 pymupdf 散佈
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "無法載入內建中文字型（cjk / Droid Sans Fallback）。"
            "請確認 pymupdf 版本完整，或改用其他可處理 CJK 的字型方案。"
        ) from exc

    if not getattr(font, "buffer", None):
        raise SystemExit("內建中文字型無字型緩衝資料，無法安全產生中文 PDF。")
    return pymupdf, font


def generate(output_path: str = OUTPUT_PATH) -> str:
    pymupdf, font = _load_cjk_font()

    archive = pymupdf.Archive()
    archive.add(font.buffer, "cjk.ttf")

    doc = pymupdf.open()
    try:
        for page_no, blocks in enumerate(PAGES, start=1):
            page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
            rect = pymupdf.Rect(
                _MARGIN, _MARGIN, _PAGE_W - _MARGIN, _PAGE_H - _MARGIN
            )
            html = _build_html(blocks)
            spare, _ = page.insert_htmlbox(
                rect, html, css=_CSS, archive=archive
            )
            if spare < 0:
                raise SystemExit(
                    f"第 {page_no} 頁內容溢出版面（spare={spare:.1f}），請縮短內容。"
                )
        # 子集化嵌入字型並壓縮，避免整套 CJK 字型（數 MB）被重複嵌入撐大檔案。
        try:
            doc.subset_fonts()
        except Exception:  # noqa: BLE001 - 子集化失敗仍可存檔（僅檔案較大）
            pass
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc.save(output_path, garbage=4, deflate=True, clean=True)
    finally:
        doc.close()

    return output_path


def main() -> None:
    path = generate()
    # 產生後立即回讀驗證頁數與可擷取文字，避免產出亂碼 / 空白 PDF。
    import pymupdf

    doc = pymupdf.open(path)
    try:
        page_count = doc.page_count
        non_empty = sum(1 for i in range(page_count) if doc.load_page(i).get_text().strip())
    finally:
        doc.close()
    print(f"已產生 {path}")
    print(f"頁數={page_count}，含可擷取文字頁數={non_empty}")
    if page_count < 3 or non_empty < 3:
        print("警告：頁數或可擷取文字頁數不足 3，請檢查產生流程。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
