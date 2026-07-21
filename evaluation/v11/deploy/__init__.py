"""Safe Apply / Rollback / Alias / Eligibility / Audit（Slice 7）。

本套件實作 Candidate Deployment 的**安全機制與隔離驗證**：

- :mod:`~evaluation.v11.deploy.eligibility`：``ApplyEligibilityGuard``（純函式，可單元測試）。
- :mod:`~evaluation.v11.deploy.corpus_manifest`：Target Active Corpus Manifest。
- :mod:`~evaluation.v11.deploy.naming`：versioned physical collection 命名 + 正式 collection 硬編碼防護。
- :mod:`~evaluation.v11.deploy.builder`：``DeploymentCollectionBuilder``。
- :mod:`~evaluation.v11.deploy.readiness`：``DeploymentReadinessVerifier``。
- :mod:`~evaluation.v11.deploy.alias_ops`：Qdrant Logical Alias 的 atomic 操作。
- :mod:`~evaluation.v11.deploy.repository`：獨立、可注入的 Deployment 持久化。
- :mod:`~evaluation.v11.deploy.service`：initialize / apply / rollback / reconcile workflow。

**安全不變式（Slice 7 §四 / §六）**：本 Slice 不對真實 ``factory_knowledge`` 執行任何
Apply / Rollback / Alias 初始化；成功流程僅以隔離、deterministic 的 eligible fixture 驗證。
"""
