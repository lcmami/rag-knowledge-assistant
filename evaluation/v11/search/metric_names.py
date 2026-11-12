"""已知 Metric 名稱註冊表（Slice 5 §14.3：metric 名稱不存在必須報錯）。

這裡列出的是 :mod:`evaluation.v11.retrieval_metrics` **實際會產生** 的 metric 名稱。
Constraint / Pareto objective 只能引用這些名稱；引用不存在的名稱屬**設定錯誤**，
必須在 Search Config 驗證階段就失敗，而不是在執行期默默當成「不可計算」。

三種狀態必須嚴格區分（§14.5「不可計算不得默認通過」）：

1. **名稱不存在**  → 設定錯誤，:class:`SearchConfigError`（本模組負責偵測）。
2. **名稱存在但該 trial 沒有這筆 row** → 不可計算，reason ``metric_absent_for_trial``。
3. **row 存在但 ``computable=False``** → 不可計算，沿用 metric 自身的 reason。

注意：per-query-type scope 只產生四個 metric（見 ``retrieval_metrics._per_query_type_metrics``），
**沒有** ``exact_code_top1_accuracy`` / latency。若要對某個 query type 設限，
只能用下方 :data:`QUERY_TYPE_METRICS` 內的名稱。

v1.1 retrieval evaluator 不再產生 ``recall_at_*``：``recall_at_{1,3,5}`` 與
``relevant_context_hit_at_{1,3,5}`` 在 v1.1 中公式、case 篩選、relevance threshold
完全相同（皆為 ``_hit_at_k(..., level=1)``），屬重複 metric。請改用
``relevant_context_hit_at_*``。
"""
from __future__ import annotations

from evaluation.v11.metric_config import DEFAULT_METRIC_CONFIG

SCOPE_AGGREGATE = "aggregate"
SCOPE_QUERY_TYPE = "query_type"
ALLOWED_SCOPES = (SCOPE_AGGREGATE, SCOPE_QUERY_TYPE)

_TOP_K = DEFAULT_METRIC_CONFIG.top_k
_NDCG_K = DEFAULT_METRIC_CONFIG.ndcg_k

#: scope="aggregate" 時可引用的 metric 名稱。
AGGREGATE_METRICS: frozenset[str] = frozenset(
    [
        *(f"strong_evidence_hit_at_{k}" for k in (1, 3, 5)),
        *(f"relevant_context_hit_at_{k}" for k in (1, 3, 5)),
        "mrr",
        f"graded_ndcg_at_{_NDCG_K}",
        "graded_ndcg_computable_count",
        "graded_ndcg_non_computable_count",
        f"evidence_coverage_at_{_TOP_K}",
        f"relevant_section_precision_at_{_TOP_K}",
        "exact_code_top1_accuracy",
        "confusable_code_error_count",
        "confusable_code_error_rate",
        "unanswerable_false_positive_count",
        "unanswerable_false_positive_rate",
        "duplicate_ratio",
        "mean_retrieval_latency_ms",
        "p50_retrieval_latency_ms",
        "p95_retrieval_latency_ms",
        "index_point_count",
        "index_build_time_ms",
        "index_universe_size",
    ]
)

#: scope="query_type" 時可引用的 metric 名稱（僅這四個會被產生）。
QUERY_TYPE_METRICS: frozenset[str] = frozenset(
    [
        "relevant_context_hit_at_1",
        f"relevant_context_hit_at_{_TOP_K}",
        "mrr",
        "case_count",
    ]
)

#: 沿用 :mod:`evaluation.v11.case_schema` 的合法 query type。
ALLOWED_QUERY_TYPES: frozenset[str] = frozenset(
    ["exact_code", "semantic", "procedure", "cross_section", "confusable", "unanswerable"]
)

#: metric 在該 trial 完全沒有對應 row 時使用的 not-computable reason。
REASON_METRIC_ABSENT = "metric_absent_for_trial"


def metric_exists(metric_name: str, scope: str, query_type: str | None) -> bool:
    """該 (metric_name, scope) 組合是否為 retrieval_metrics 會產生的名稱。"""
    if scope == SCOPE_AGGREGATE:
        return metric_name in AGGREGATE_METRICS
    if scope == SCOPE_QUERY_TYPE:
        return metric_name in QUERY_TYPE_METRICS
    return False


def describe_metric(metric_name: str, scope: str, query_type: str | None) -> str:
    """人類可讀的 metric 參照（報告 / 錯誤訊息用）。"""
    if scope == SCOPE_QUERY_TYPE:
        return f"{metric_name}[{scope}:{query_type}]"
    return f"{metric_name}[{scope}]"
