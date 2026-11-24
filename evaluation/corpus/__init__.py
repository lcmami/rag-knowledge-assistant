"""外部結構化 API corpus 取得與正規化（docs/external-corpus-api-plan.md）。

本套件是既有 evaluation pipeline 的**生產者**：把外部來源正規化成磁碟上的
Markdown，寫進一個 corpus 目錄；其後由既有的
``evaluation/v11/hashing.py::collect_corpus_files`` → ``load_document`` →
chunking → embedding → Qdrant 全程接手。

因此 ``app/**`` 與 ``evaluation/v11/**`` 皆不需要、也不得為此修改。

Slice A 範圍：``models`` / ``moj_law`` / ``normalize``。
snapshot / stats / cli 屬後續 slice，尚未實作。
"""
