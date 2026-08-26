"""版本化 Optimizer Search Config（Slice 5 §六、§七、§十四、§十七、§十八）。

嚴格驗證的 search config：任何未知欄位、未實作的搜尋維度、或不存在的 metric 名稱都會
在**載入階段**失敗，而不是等到跑了半天才炸。Config hash 涵蓋 §6.8 要求的全部語意欄位。

安全（§6.9 / §6.11）：``dataset_path`` / ``corpus_paths`` 必須是 repo 相對路徑，
不得是絕對路徑；search config 不隱性讀取 ``.env``（baseline chunking 的 ``current`` strategy
由 :mod:`evaluation.v11.chunking.config` 展開成明確數值，見 Slice 3 §一）。
"""
from __future__ import annotations

import ntpath
import posixpath
from dataclasses import dataclass, field
from typing import Any, Optional

from evaluation.v11.chunking.config import ChunkConfigError, ChunkingConfig
from evaluation.v11.hashing import canonical_json, config_hash
from evaluation.v11.retrieval.config import RetrievalConfig, RetrievalConfigError
from evaluation.v11.search import metric_names as mn

SEARCH_CONFIG_SCHEMA_VERSION = "1.0"

SEARCH_MODE_GRID = "grid"
SEARCH_MODE_RANDOM = "random"
ALLOWED_SEARCH_MODES = (SEARCH_MODE_GRID, SEARCH_MODE_RANDOM)

#: §14.1 支援的比較運算子。
ALLOWED_OPERATORS = (">=", ">", "<=", "<", "==")

#: §14.6 不可計算政策。
NOT_COMPUTABLE_FAIL = "fail"
NOT_COMPUTABLE_WARN = "warn"
NOT_COMPUTABLE_SKIP = "skip"
ALLOWED_NOT_COMPUTABLE = (NOT_COMPUTABLE_FAIL, NOT_COMPUTABLE_WARN, NOT_COMPUTABLE_SKIP)

#: §17.2 objective 不可計算時的處理。
OBJ_EXCLUDE_CANDIDATE = "exclude_candidate"
OBJ_IGNORE_OBJECTIVE = "ignore_objective"
ALLOWED_OBJECTIVE_NOT_COMPUTABLE = (OBJ_EXCLUDE_CANDIDATE, OBJ_IGNORE_OBJECTIVE)

DIRECTION_MAXIMIZE = "maximize"
DIRECTION_MINIMIZE = "minimize"
ALLOWED_DIRECTIONS = (DIRECTION_MAXIMIZE, DIRECTION_MINIMIZE)

#: §7.1 允許搜尋的 chunking 維度。
ALLOWED_CHUNKING_SEARCH_KEYS = (
    "strategy",
    "target_chars",
    "overlap_chars",
    "max_tokens",
    "prepend_section_title",
    "context_before_paragraphs",
    "context_after_paragraphs",
)

#: §7.2 允許搜尋的 retrieval 維度。
ALLOWED_RETRIEVAL_SEARCH_KEYS = (
    "strategy",
    "top_k",
    "score_threshold",
    "candidate_multiplier",
    "candidate_cap",
    "identifier_candidate_cap",
    "exact_match_boost",
    "confusable_penalty",
    "identifier_vector_floor",
    # --- v1.2.1 Slice R3：rerank（與 strategy 正交，不是第三個 strategy）---
    # `rerank_strategy` 可搜尋（none vs cross_encoder 正是要比較的對象）。
    # `rerank_top_n` 技術上可搜尋，但 **R0.5 已 FROZEN 為 5**：它由 candidate coverage
    # audit（tune + validation）選出，不是效能調參旋鈕。此處放行只是為了讓 search config
    # 能明示寫出 `[5]`，**不代表核准對它做多值掃描**。
    "rerank_strategy",
    "rerank_top_n",
)

#: §7 明確**不得**搜尋的欄位（固定版本資訊，不是搜尋參數）。給出精準錯誤訊息。
FORBIDDEN_SEARCH_KEYS = {
    "embedding_model": "embedding model 為固定版本資訊，不是 Slice 5 搜尋參數",
    "embedding_dimension": "embedding dimension 為固定版本資訊，不是 Slice 5 搜尋參數",
    "llm_model": "LLM model 不屬於 retrieval 搜尋空間",
    "prompt": "prompt 不屬於 retrieval 搜尋空間",
    "citation": "citation 規則不屬於 retrieval 搜尋空間",
    "identifier_normalization_version": "identifier normalization version 為固定版本資訊",
    "canonical_text_version": "canonical text version 為固定版本資訊",
    "ranking_version": "ranking_version 由 strategy 決定，不得直接搜尋",
    "prefix_format_version": "prefix_format_version 為固定版本資訊",
    "gain_mapping": "gain mapping 屬 MetricConfig 版本資訊，不是搜尋參數",
    "ndcg_variant": "NDCG variant 屬 MetricConfig 版本資訊，不是搜尋參數",
    "ndcg_gains": "NDCG gains 屬 MetricConfig 版本資訊，不是搜尋參數",
    # v1.2.1 §R3：比照 embedding_model —— 換 reranker 等於換 fingerprint、換 baseline，
    # 是版本決策而非候選比較。R0 已把 BAAI/bge-reranker-base 與其 revision 鎖進 fingerprint。
    "rerank_model": "rerank model 為固定版本資訊（換模型＝換 fingerprint＝換 baseline），不是搜尋參數",
}


class SearchConfigError(Exception):
    """Search Config 驗證錯誤（訊息需可理解、指出違規欄位）。"""


# --------------------------------------------------------------------------
# Metric 參照
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricRef:
    """一個 metric 的完整參照 (metric_name, scope, query_type)。"""

    metric: str
    scope: str = mn.SCOPE_AGGREGATE
    query_type: Optional[str] = None

    def describe(self) -> str:
        return mn.describe_metric(self.metric, self.scope, self.query_type)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"metric": self.metric, "scope": self.scope}
        if self.scope == mn.SCOPE_QUERY_TYPE:
            d["query_type"] = self.query_type
        return d


def _parse_metric_ref(raw: dict[str, Any], where: str) -> MetricRef:
    metric = raw.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        raise SearchConfigError(f"{where} 缺少 metric 名稱。")
    metric = metric.strip()

    scope = raw.get("scope", mn.SCOPE_AGGREGATE)
    if scope not in mn.ALLOWED_SCOPES:
        raise SearchConfigError(
            f"{where} scope={scope!r} 無效，只允許 {list(mn.ALLOWED_SCOPES)}。"
        )

    query_type = raw.get("query_type")
    if scope == mn.SCOPE_QUERY_TYPE:
        if not isinstance(query_type, str) or not query_type.strip():
            raise SearchConfigError(f"{where} scope=query_type 時必須指定 query_type。")
        query_type = query_type.strip()
        if query_type not in mn.ALLOWED_QUERY_TYPES:
            raise SearchConfigError(
                f"{where} query_type={query_type!r} 無效，只允許 "
                f"{sorted(mn.ALLOWED_QUERY_TYPES)}。"
            )
    else:
        if query_type is not None:
            raise SearchConfigError(f"{where} scope=aggregate 時不得指定 query_type。")

    # §14.3：metric 名稱不存在 → 設定錯誤（不是「不可計算」）。
    if not mn.metric_exists(metric, scope, query_type):
        allowed = (
            sorted(mn.AGGREGATE_METRICS)
            if scope == mn.SCOPE_AGGREGATE
            else sorted(mn.QUERY_TYPE_METRICS)
        )
        raise SearchConfigError(
            f"{where} metric={metric!r} 在 scope={scope!r} 不存在（retrieval_metrics 不會產生）。"
            f" 可用名稱：{allowed}"
        )
    return MetricRef(metric=metric, scope=scope, query_type=query_type)


# --------------------------------------------------------------------------
# Hard Constraint
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HardConstraint:
    """單一硬限制（§14）。

    ``value`` 與 ``baseline_delta`` 二擇一：

    - ``value``：absolute constraint，``observed <op> value``。
    - ``baseline_delta``：relative-to-baseline，``observed <op> (baseline + delta)``。
    """

    name: str
    ref: MetricRef
    operator: str
    value: Optional[float] = None
    baseline_delta: Optional[float] = None
    on_not_computable: str = NOT_COMPUTABLE_FAIL
    safety_critical: bool = True

    @property
    def is_relative(self) -> bool:
        return self.baseline_delta is not None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            **self.ref.to_dict(),
            "operator": self.operator,
            "on_not_computable": self.on_not_computable,
            "safety_critical": self.safety_critical,
        }
        if self.is_relative:
            d["baseline_delta"] = self.baseline_delta
        else:
            d["value"] = self.value
        return d


def _parse_constraint(raw: Any, index: int) -> HardConstraint:
    where = f"hard_constraints[{index}]"
    if not isinstance(raw, dict):
        raise SearchConfigError(f"{where} 必須是物件。")

    ref = _parse_metric_ref(raw, where)

    operator = raw.get("operator")
    if operator not in ALLOWED_OPERATORS:
        raise SearchConfigError(
            f"{where} operator={operator!r} 無效，只允許 {list(ALLOWED_OPERATORS)}。"
        )

    has_value = "value" in raw and raw["value"] is not None
    has_delta = "baseline_delta" in raw and raw["baseline_delta"] is not None
    if has_value == has_delta:
        raise SearchConfigError(
            f"{where} 必須且只能指定 value（absolute）或 baseline_delta（relative）其中一項。"
        )

    value = _finite(raw["value"], f"{where}.value") if has_value else None
    delta = _finite(raw["baseline_delta"], f"{where}.baseline_delta") if has_delta else None

    onc = raw.get("on_not_computable", NOT_COMPUTABLE_FAIL)
    if onc not in ALLOWED_NOT_COMPUTABLE:
        raise SearchConfigError(
            f"{where} on_not_computable={onc!r} 無效，只允許 {list(ALLOWED_NOT_COMPUTABLE)}。"
        )

    safety_critical = raw.get("safety_critical", True)
    if not isinstance(safety_critical, bool):
        raise SearchConfigError(f"{where} safety_critical 必須是布林值。")

    # §14.7：safety-critical 預設且應維持 fail；若刻意放寬必須是明確的設定決定。
    if safety_critical and onc != NOT_COMPUTABLE_FAIL:
        raise SearchConfigError(
            f"{where} 標為 safety_critical 卻設定 on_not_computable={onc!r}。"
            " safety-critical constraint 不可計算時必須 fail（§14.7）；"
            " 若確實非安全限制，請明確設 safety_critical=false。"
        )

    name = raw.get("name")
    if name is None:
        name = f"{ref.describe()} {operator} " + (
            f"baseline{delta:+g}" if has_delta else f"{value:g}"
        )
    if not isinstance(name, str) or not name.strip():
        raise SearchConfigError(f"{where} name 必須是非空字串。")

    return HardConstraint(
        name=name.strip(),
        ref=ref,
        operator=operator,
        value=value,
        baseline_delta=delta,
        on_not_computable=onc,
        safety_critical=safety_critical,
    )


# --------------------------------------------------------------------------
# Pareto objective
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ParetoObjective:
    ref: MetricRef
    direction: str
    on_not_computable: str = OBJ_EXCLUDE_CANDIDATE

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.ref.to_dict(),
            "direction": self.direction,
            "on_not_computable": self.on_not_computable,
        }

    def describe(self) -> str:
        arrow = "↑" if self.direction == DIRECTION_MAXIMIZE else "↓"
        return f"{self.ref.describe()} {arrow}"


def _parse_objective(raw: Any, index: int) -> ParetoObjective:
    where = f"pareto_objectives[{index}]"
    if not isinstance(raw, dict):
        raise SearchConfigError(f"{where} 必須是物件。")
    ref = _parse_metric_ref(raw, where)
    direction = raw.get("direction")
    if direction not in ALLOWED_DIRECTIONS:
        raise SearchConfigError(
            f"{where} direction={direction!r} 無效，只允許 {list(ALLOWED_DIRECTIONS)}。"
        )
    onc = raw.get("on_not_computable", OBJ_EXCLUDE_CANDIDATE)
    if onc not in ALLOWED_OBJECTIVE_NOT_COMPUTABLE:
        raise SearchConfigError(
            f"{where} on_not_computable={onc!r} 無效，"
            f"只允許 {list(ALLOWED_OBJECTIVE_NOT_COMPUTABLE)}。"
        )
    return ParetoObjective(ref=ref, direction=direction, on_not_computable=onc)


# --------------------------------------------------------------------------
# Recommendation policy
# --------------------------------------------------------------------------
#: lexicographic 排序中可使用的非 metric tie-break 鍵。
TIEBREAK_CONSTRAINT_PASS_COUNT = "constraint_pass_count"
TIEBREAK_CONFIG_COMPLEXITY = "config_complexity"
TIEBREAK_CONFIG_HASH = "config_hash"
SPECIAL_SORT_KEYS = (
    TIEBREAK_CONSTRAINT_PASS_COUNT,
    TIEBREAK_CONFIG_COMPLEXITY,
    TIEBREAK_CONFIG_HASH,
)


@dataclass(frozen=True)
class SortKey:
    """排序鍵：metric 參照或特殊鍵，加上方向。"""

    key: str
    direction: str
    ref: Optional[MetricRef] = None

    @property
    def is_special(self) -> bool:
        return self.ref is None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"key": self.key, "direction": self.direction}
        if self.ref is not None and self.ref.scope == mn.SCOPE_QUERY_TYPE:
            d["scope"] = self.ref.scope
            d["query_type"] = self.ref.query_type
        return d

    def describe(self) -> str:
        arrow = "↑" if self.direction == DIRECTION_MAXIMIZE else "↓"
        return f"{self.key}{'' if self.is_special else ''} {arrow}"


def _parse_sort_key(raw: Any, where: str) -> SortKey:
    if not isinstance(raw, dict):
        raise SearchConfigError(f"{where} 必須是物件。")
    key = raw.get("key")
    if not isinstance(key, str) or not key.strip():
        raise SearchConfigError(f"{where} 缺少 key。")
    key = key.strip()
    direction = raw.get("direction")
    if direction not in ALLOWED_DIRECTIONS:
        raise SearchConfigError(
            f"{where} direction={direction!r} 無效，只允許 {list(ALLOWED_DIRECTIONS)}。"
        )
    if key in SPECIAL_SORT_KEYS:
        return SortKey(key=key, direction=direction, ref=None)
    ref = _parse_metric_ref({**raw, "metric": key}, where)
    return SortKey(key=key, direction=direction, ref=ref)


def _parse_sort_keys(raw: Any, where: str) -> tuple[SortKey, ...]:
    if not isinstance(raw, list) or not raw:
        raise SearchConfigError(f"{where} 必須是非空陣列。")
    keys = tuple(_parse_sort_key(r, f"{where}[{i}]") for i, r in enumerate(raw))
    # §18.4：順序不得由 dict 決定；且結尾必須是完全 deterministic 的 config_hash。
    if keys[-1].key != TIEBREAK_CONFIG_HASH:
        raise SearchConfigError(
            f"{where} 最後一個排序鍵必須是 {TIEBREAK_CONFIG_HASH!r}，"
            "以保證完全 deterministic 的 tie-break（§18.9）。"
        )
    seen: set[str] = set()
    for k in keys:
        sig = f"{k.key}|{'' if k.ref is None else k.ref.describe()}"
        if sig in seen:
            raise SearchConfigError(f"{where} 排序鍵重複：{sig}")
        seen.add(sig)
    return keys


@dataclass(frozen=True)
class TunePrefilter:
    """§十六 Tune 預篩策略（明確可配置，避免小樣本過度淘汰）。

    ``eliminate_on_not_computable`` 預設為 **False**：在小型 tune split 上，某個 metric
    往往因為該 split 缺少對應 query type 而**結構性**不可計算（例如 tune 沒有 unanswerable
    案例）。若此時就套用 §16.2 淘汰，會在進入 validation 前把**所有** candidate 都刷掉，
    正好違反 §十六 開頭「需避免小樣本造成過度淘汰」的要求。

    注意：這只影響 **tune 預篩**。Validation 階段的正式判定一律遵守 §14.5
    「不可計算不得默認通過」，不受此旗標影響。
    """

    eliminate_on_failed_trial: bool = True
    eliminate_on_computable_safety_violation: bool = True
    eliminate_on_not_computable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "eliminate_on_failed_trial": self.eliminate_on_failed_trial,
            "eliminate_on_computable_safety_violation": (
                self.eliminate_on_computable_safety_violation
            ),
            "eliminate_on_not_computable": self.eliminate_on_not_computable,
        }


@dataclass(frozen=True)
class RecommendationPolicy:
    """§16 / §18：可配置、deterministic 的 lexicographic 排序政策。"""

    tune_ranking: tuple[SortKey, ...]
    recommendation_ranking: tuple[SortKey, ...]
    complexity_version: str = "v1"
    float_tolerance: float = 1e-9
    tune_prefilter: TunePrefilter = field(default_factory=TunePrefilter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tune_ranking": [k.to_dict() for k in self.tune_ranking],
            "recommendation_ranking": [k.to_dict() for k in self.recommendation_ranking],
            "complexity_version": self.complexity_version,
            "float_tolerance": self.float_tolerance,
            "tune_prefilter": self.tune_prefilter.to_dict(),
        }


def _parse_tune_prefilter(raw: Any) -> TunePrefilter:
    if raw is None:
        return TunePrefilter()
    if not isinstance(raw, dict):
        raise SearchConfigError("recommendation_policy.tune_prefilter 必須是物件。")
    known = {
        "eliminate_on_failed_trial",
        "eliminate_on_computable_safety_violation",
        "eliminate_on_not_computable",
    }
    unknown = set(raw) - known
    if unknown:
        raise SearchConfigError(
            f"recommendation_policy.tune_prefilter 含未知欄位：{sorted(unknown)}"
        )
    for key in known:
        if key in raw and not isinstance(raw[key], bool):
            raise SearchConfigError(
                f"recommendation_policy.tune_prefilter.{key} 必須是布林值。"
            )
    return TunePrefilter(
        eliminate_on_failed_trial=raw.get("eliminate_on_failed_trial", True),
        eliminate_on_computable_safety_violation=raw.get(
            "eliminate_on_computable_safety_violation", True
        ),
        eliminate_on_not_computable=raw.get("eliminate_on_not_computable", False),
    )


def _parse_policy(raw: Any) -> RecommendationPolicy:
    if not isinstance(raw, dict):
        raise SearchConfigError("recommendation_policy 必須是物件。")
    unknown = set(raw) - {
        "tune_ranking",
        "recommendation_ranking",
        "complexity_version",
        "float_tolerance",
        "tune_prefilter",
    }
    if unknown:
        raise SearchConfigError(f"recommendation_policy 含未知欄位：{sorted(unknown)}")
    tol = raw.get("float_tolerance", 1e-9)
    tol = _finite(tol, "recommendation_policy.float_tolerance")
    if tol < 0:
        raise SearchConfigError("recommendation_policy.float_tolerance 不得為負。")
    version = raw.get("complexity_version", "v1")
    if not isinstance(version, str) or not version.strip():
        raise SearchConfigError("recommendation_policy.complexity_version 必須是非空字串。")
    return RecommendationPolicy(
        tune_ranking=_parse_sort_keys(
            raw.get("tune_ranking"), "recommendation_policy.tune_ranking"
        ),
        recommendation_ranking=_parse_sort_keys(
            raw.get("recommendation_ranking"), "recommendation_policy.recommendation_ranking"
        ),
        complexity_version=version.strip(),
        float_tolerance=tol,
        tune_prefilter=_parse_tune_prefilter(raw.get("tune_prefilter")),
    )


# --------------------------------------------------------------------------
# 共用驗證小工具
# --------------------------------------------------------------------------
def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SearchConfigError(f"{where} 必須是數字。")
    f = float(value)
    if f != f or f in (float("inf"), float("-inf")):
        raise SearchConfigError(f"{where} 不得為 NaN / Inf。")
    return f


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SearchConfigError(f"{where} 必須是整數。")
    if value <= 0:
        raise SearchConfigError(f"{where} 必須 > 0（收到 {value}）。")
    return value


def _relative_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SearchConfigError(f"{where} 必須是非空字串。")
    p = value.strip()
    # §6.9：不得含絕對路徑（POSIX 與 Windows 皆檢查，避免跨平台漏網）。
    if posixpath.isabs(p) or ntpath.isabs(p) or ":" in p.replace("://", ""):
        raise SearchConfigError(f"{where} 不得為絕對路徑（收到 {p!r}）；請使用 repo 相對路徑。")
    if ".." in p.replace("\\", "/").split("/"):
        raise SearchConfigError(f"{where} 不得含 '..'（收到 {p!r}）。")
    return p


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchSpace:
    """每個維度對應一個「明確列出的候選值」清單（保留作者順序 → 生成順序 deterministic）。"""

    chunking: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    retrieval: dict[str, tuple[Any, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunking": {k: list(v) for k, v in self.chunking.items()},
            "retrieval": {k: list(v) for k, v in self.retrieval.items()},
        }


def _parse_space_section(
    raw: Any, allowed: tuple[str, ...], section: str
) -> dict[str, tuple[Any, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SearchConfigError(f"search_space.{section} 必須是物件。")
    out: dict[str, tuple[Any, ...]] = {}
    for key, values in raw.items():
        where = f"search_space.{section}.{key}"
        if key in FORBIDDEN_SEARCH_KEYS:
            raise SearchConfigError(
                f"{where} 不得出現在搜尋空間：{FORBIDDEN_SEARCH_KEYS[key]}（§七）。"
            )
        if key not in allowed:
            raise SearchConfigError(
                f"{where} 不是本 Slice 可搜尋的維度。允許：{list(allowed)}（§六.6）。"
            )
        if not isinstance(values, list) or not values:
            raise SearchConfigError(f"{where} 必須是非空陣列。")
        # 值必須是 JSON scalar；重複值視為設定錯誤（避免無聲產生重複 candidate）。
        seen: list[str] = []
        for v in values:
            if not isinstance(v, (str, int, float, bool)):
                raise SearchConfigError(f"{where} 只允許 string / number / boolean 值。")
            sig = canonical_json(v)
            if sig in seen:
                raise SearchConfigError(f"{where} 含重複值：{v!r}")
            seen.append(sig)
        out[key] = tuple(values)
    # 維度順序 canonical 化（依 key 排序）→ Cartesian product 順序與 JSON 撰寫順序無關。
    return {k: out[k] for k in sorted(out)}


# --------------------------------------------------------------------------
# SearchConfig
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchConfig:
    """完整、可執行、版本化的 Search Config。"""

    search_name: str
    search_mode: str
    dataset_path: str
    corpus_paths: tuple[str, ...]
    baseline_chunk_config: dict[str, Any]
    baseline_retrieval_config: dict[str, Any]
    search_space: SearchSpace
    hard_constraints: tuple[HardConstraint, ...]
    pareto_objectives: tuple[ParetoObjective, ...]
    recommendation_policy: RecommendationPolicy
    random_seed: int = 42
    max_candidates: int = 32
    validation_top_n: int = 8
    policy_source: str = "unspecified"
    policy_confidence: str = "unspecified"
    keep_trial_collections: bool = False
    schema_version: str = SEARCH_CONFIG_SCHEMA_VERSION

    # -- hashing -----------------------------------------------------------
    def hashable(self) -> dict[str, Any]:
        """§6.8：hash 涵蓋 baseline / search space / constraint / objective /
        recommendation policy / seed / budget。

        刻意**不含** ``search_name``（人類標籤）與 ``keep_trial_collections``（執行旗標），
        兩者不改變搜尋語意。含 dataset / corpus 以確保換資料集不會誤判為同一搜尋。
        """
        return {
            "schema_version": self.schema_version,
            "search_mode": self.search_mode,
            "dataset_path": self.dataset_path,
            "corpus_paths": list(self.corpus_paths),
            "baseline": {
                "chunk_config": self.baseline_chunk_config,
                "retrieval_config": self.baseline_retrieval_config,
            },
            "search_space": self.search_space.to_dict(),
            "hard_constraints": [c.to_dict() for c in self.hard_constraints],
            "pareto_objectives": [o.to_dict() for o in self.pareto_objectives],
            "recommendation_policy": self.recommendation_policy.to_dict(),
            "random_seed": self.random_seed,
            "max_candidates": self.max_candidates,
            "validation_top_n": self.validation_top_n,
        }

    def compute_hash(self) -> str:
        return config_hash(self.hashable())

    def sanitized(self) -> dict[str, Any]:
        """報告用快照（皆為相對路徑，無敏感資料）。"""
        return {
            **self.hashable(),
            "search_name": self.search_name,
            "policy_source": self.policy_source,
            "policy_confidence": self.policy_confidence,
            "search_config_hash": self.compute_hash(),
        }

    # -- parsing -----------------------------------------------------------
    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "SearchConfig":
        if not isinstance(raw, dict):
            raise SearchConfigError("search config 最外層必須是物件。")

        known = {
            "schema_version",
            "search_name",
            "search_mode",
            "random_seed",
            "max_candidates",
            "validation_top_n",
            "dataset_path",
            "corpus_paths",
            "baseline",
            "search_space",
            "hard_constraints",
            "pareto_objectives",
            "recommendation_policy",
            "policy_source",
            "policy_confidence",
            "keep_trial_collections",
        }
        unknown = set(raw) - known
        if unknown:
            raise SearchConfigError(f"search config 含未知欄位：{sorted(unknown)}")

        schema_version = str(raw.get("schema_version", SEARCH_CONFIG_SCHEMA_VERSION))
        if schema_version != SEARCH_CONFIG_SCHEMA_VERSION:
            raise SearchConfigError(
                f"不支援的 schema_version={schema_version!r}"
                f"（本版支援 {SEARCH_CONFIG_SCHEMA_VERSION!r}）。"
            )

        search_name = raw.get("search_name")
        if not isinstance(search_name, str) or not search_name.strip():
            raise SearchConfigError("search config 缺少 search_name。")

        search_mode = raw.get("search_mode")
        if search_mode not in ALLOWED_SEARCH_MODES:
            raise SearchConfigError(
                f"search_mode={search_mode!r} 無效，只允許 {list(ALLOWED_SEARCH_MODES)}（§6.2）。"
            )

        max_candidates = _positive_int(raw.get("max_candidates", 32), "max_candidates")
        validation_top_n = _positive_int(raw.get("validation_top_n", 8), "validation_top_n")

        seed = raw.get("random_seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SearchConfigError("random_seed 必須是整數。")

        dataset_path = _relative_path(raw.get("dataset_path"), "dataset_path")
        corpus_raw = raw.get("corpus_paths")
        if not isinstance(corpus_raw, list) or not corpus_raw:
            raise SearchConfigError("corpus_paths 必須是非空陣列（§6.10 明確指定 corpus）。")
        corpus_paths = tuple(
            _relative_path(p, f"corpus_paths[{i}]") for i, p in enumerate(corpus_raw)
        )

        # -- baseline（§6.5 必須完整、可執行）--
        baseline = raw.get("baseline")
        if not isinstance(baseline, dict):
            raise SearchConfigError("baseline 必須是物件。")
        unknown_b = set(baseline) - {"chunk_config", "retrieval_config"}
        if unknown_b:
            raise SearchConfigError(f"baseline 含未知欄位：{sorted(unknown_b)}")
        try:
            baseline_chunk = ChunkingConfig.from_raw(
                baseline.get("chunk_config") or {"strategy": "current"}
            ).to_canonical_dict()
        except ChunkConfigError as exc:
            raise SearchConfigError(f"baseline.chunk_config 無效：{exc}") from exc
        try:
            baseline_retrieval = RetrievalConfig.from_raw(
                baseline.get("retrieval_config") or {}
            ).to_canonical_dict()
        except RetrievalConfigError as exc:
            raise SearchConfigError(f"baseline.retrieval_config 無效：{exc}") from exc

        # -- search space --
        space_raw = raw.get("search_space")
        if not isinstance(space_raw, dict):
            raise SearchConfigError("search_space 必須是物件。")
        unknown_s = set(space_raw) - {"chunking", "retrieval"}
        if unknown_s:
            raise SearchConfigError(f"search_space 含未知區段：{sorted(unknown_s)}")
        space = SearchSpace(
            chunking=_parse_space_section(
                space_raw.get("chunking"), ALLOWED_CHUNKING_SEARCH_KEYS, "chunking"
            ),
            retrieval=_parse_space_section(
                space_raw.get("retrieval"), ALLOWED_RETRIEVAL_SEARCH_KEYS, "retrieval"
            ),
        )
        if not space.chunking and not space.retrieval:
            raise SearchConfigError("search_space 至少要有一個可搜尋維度。")

        # -- constraints / objectives / policy --
        raw_constraints = raw.get("hard_constraints")
        if not isinstance(raw_constraints, list):
            raise SearchConfigError("hard_constraints 必須是陣列（可為空）。")
        constraints = tuple(
            _parse_constraint(c, i) for i, c in enumerate(raw_constraints)
        )
        names = [c.name for c in constraints]
        if len(set(names)) != len(names):
            raise SearchConfigError("hard_constraints 的 name 不得重複。")

        raw_objectives = raw.get("pareto_objectives")
        if not isinstance(raw_objectives, list) or not raw_objectives:
            raise SearchConfigError("pareto_objectives 必須是非空陣列。")
        objectives = tuple(_parse_objective(o, i) for i, o in enumerate(raw_objectives))
        sigs = [f"{o.ref.describe()}|{o.direction}" for o in objectives]
        if len(set(sigs)) != len(sigs):
            raise SearchConfigError("pareto_objectives 不得重複。")

        policy = _parse_policy(raw.get("recommendation_policy"))

        policy_source = str(raw.get("policy_source", "unspecified")).strip() or "unspecified"
        policy_confidence = (
            str(raw.get("policy_confidence", "unspecified")).strip() or "unspecified"
        )

        keep = raw.get("keep_trial_collections", False)
        if not isinstance(keep, bool):
            raise SearchConfigError("keep_trial_collections 必須是布林值。")

        return cls(
            search_name=search_name.strip(),
            search_mode=search_mode,
            dataset_path=dataset_path,
            corpus_paths=corpus_paths,
            baseline_chunk_config=baseline_chunk,
            baseline_retrieval_config=baseline_retrieval,
            search_space=space,
            hard_constraints=constraints,
            pareto_objectives=objectives,
            recommendation_policy=policy,
            random_seed=seed,
            max_candidates=max_candidates,
            validation_top_n=validation_top_n,
            policy_source=policy_source,
            policy_confidence=policy_confidence,
            keep_trial_collections=keep,
            schema_version=schema_version,
        )
