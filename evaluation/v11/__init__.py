"""v1.1 Retrieval Optimizer — Slice 1 元件。

僅包含 Slice 1 範圍：
- :mod:`evaluation.v11.case_schema`：v1.1 Evaluation Case / Evidence Span schema 與驗證。
- :mod:`evaluation.v11.evidence_matcher`：Gold Evidence ↔ Chunk Provenance 的 relevance 判定。
- :mod:`evaluation.v11.optimizer_repository`：獨立於正式 registry 的 Optimizer SQLite 儲存。

不含 Slice 2+ 的 TrialRunner / Metrics / OptimizerService / Apply / Rollback。
"""
