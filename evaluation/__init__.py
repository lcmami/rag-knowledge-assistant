"""Retrieval Evaluation 套件（v1.1 Slice 0 — Baseline 與失敗案例固定化）。

本套件將原本 `evaluate_retrieval.py` 的可重用邏輯抽為模組：

- :mod:`evaluation.schemas`  — Config / Case / Result 結構與 case 載入驗證。
- :mod:`evaluation.metrics`  — 純函式指標計算（可用 fake retrieval 做 deterministic 測試）。
- :mod:`evaluation.runner`   — 共用核心：建服務、匯入 corpus、跑 case、組報告。
- :mod:`evaluation.evaluate_retrieval` — 保留原本 CLI，薄包裝呼叫 runner（同一核心）。
- :mod:`evaluation.generate_baseline`  — 產生 v1.1 Slice 0 Baseline 報告（JSON + Markdown）。

Slice 0 僅做 Baseline 所需最小抽象，不建立完整 v1.1 TrialRunner 或 SQLite Trial Schema。
Case 標註沿用 Slice 0 Baseline 格式（expected_document / expected_section /
expected_page / should_answer），Slice 1 才遷移至 Canonical Evidence Span。
"""
from __future__ import annotations

__all__ = [
    "schemas",
    "metrics",
    "runner",
]
