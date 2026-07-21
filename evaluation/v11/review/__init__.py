"""v1.1 Slice 6 — Human Review 與 Dataset Authoring Workflow。

最小、可稽核的人工審核流程：

    Canonical Document → 產生候選問題 → 人工接受／修改／拒絕
    → 人工確認 Answerability → 人工標註 Evidence
    → Schema／Coverage 驗證 → 發布新的 Versioned Dataset

核心不變量：**LLM 只能提出候選，不得自行把 Case 放入正式 Dataset。**
所有 Candidate（不論 LLM 或手動）一律從 ``proposed`` 開始，必須經明確人工操作
才能 ``accepted``，且發布前需通過 Evidence／Coverage／Schema 驗證。

本套件不執行 Optimizer Search、不 Apply 參數、不修改正式 Qdrant／SQLite／Uploads。
"""
