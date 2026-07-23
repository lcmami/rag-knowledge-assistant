"""v1.2 — LLM-guided Retrieval Optimization Agent.

本套件建立在既有 v1.1 Optimizer（``evaluation.v11``）之上：重用其 Trial Runner、
Chunking / Retrieval Config、Metrics、Hard Constraints、Recommendation Ordering 與
Split 隔離，**不修改** v1.1 或 production ``app/``。

v1.2 新增的核心是一個「有界的 LLM Agent Loop」：LLM 閱讀 tune split 的失敗、Chunk 與
Retrieval 結果 → 診斷 → 提出下一輪 Chunking / Retrieval 設定與 Probe Questions →
既有 Trial Runner 執行 → 以人工審核資料集計算正式 Metrics → 讀取結果（含 Parent/
Candidate Probe Comparison）進入下一輪，直到硬性停止條件。
"""
