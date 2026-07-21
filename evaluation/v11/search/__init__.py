"""Slice 5 —— Optimizer Search Strategy / Hard Constraints / Pareto / Recommendation。

本套件只做「搜尋與選擇」，**不**套用任何結果到正式系統：

- :mod:`search_config`：版本化 Search Config schema + 嚴格驗證 + deterministic hash。
- :mod:`search_space`：Grid / Random candidate 生成（conditional validation、去重、budget）。
- :mod:`constraints`：Hard Constraint Engine（absolute / relative-to-baseline、不可計算政策）。
- :mod:`pareto`：Pareto Frontier（明確 maximize / minimize，無不透明加權總分）。
- :mod:`complexity`：版本化 Config Complexity（僅作最後 tie-break）。
- :mod:`recommendation`：deterministic lexicographic Recommendation Policy。
- :mod:`split_dataset`：tune / validation / test 嚴格隔離 + group leakage 檢查。
- :mod:`search_models`：持久化資料模型與狀態常數。
- :mod:`search_runner`：baseline → tune → validation → constraints → pareto →
  recommendation → **lock** → test 的流程編排。
- :mod:`search_report`：JSON + Markdown Optimizer Report。

安全邊界（Slice 5 §四）：不讀 ``.env``、不呼叫任何 LLM、不觸碰 ``factory_knowledge`` /
正式 ``metadata.db`` / uploads、不建立 alias、不自動套用 recommendation。本套件產生的是
``recommended_candidate``，**不是** deployed / active / production candidate。
"""
