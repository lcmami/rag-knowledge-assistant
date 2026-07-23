"""v1.2 Agent Run Contract（Slice 0 §三）。

定義一次 Agent 最佳化的完整、版本化、嚴格驗證設定：

- ``baseline``：固定的起始 :class:`~evaluation.v11.run_models.TrialConfig`（重用 v1.1 的
  Chunking / Retrieval 嚴格驗證與 canonical 化）。
- ``allowed_parameter_space``：LLM **唯一**可提案的白名單欄位與界限（Slice 1 Validator 依此
  收斂；Slice 2 Context Builder 據此告知 LLM 允許的搜尋空間）。LLM 不得提出白名單外的欄位。
- ``budget``：所有硬性上限（rounds / llm calls / probes / 參數變更數 / no-improvement /
  連續拒絕 / context 字元預算），這些是 Orchestrator 停止條件的來源。
- ``promotion``：Candidate 是否升級為下一輪 Parent 的判定政策——重用 v1.1 的
  :class:`~evaluation.v11.search.search_config.HardConstraint`（validation gate 與 non-
  regression）與 recommendation ordering（deterministic comparator）。**LLM 不參與勝負判定。**

本模組不呼叫 LLM、不碰 Qdrant、不讀 ``.env``，可純單元測試。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from evaluation.v11.run_models import RunConfig, RunConfigError, TrialConfig
from evaluation.v11.search.search_config import (
    HardConstraint,
    SearchConfigError,
    SortKey,
    _parse_constraint,
    _parse_sort_keys,
    _relative_path,
)

AGENT_RUN_SCHEMA_VERSION = "1.0"
DEFAULT_PROMPT_VERSION = "v1"

# LLM 可提案的白名單欄位（對應 v1.1 canonical config 欄位；strategy 另行處理）。
CHUNK_NUMERIC_FIELDS = (
    "target_chars",
    "overlap_chars",
    "max_tokens",
    "context_before_paragraphs",
    "context_after_paragraphs",
)
CHUNK_BOOL_FIELDS = ("prepend_section_title",)
RETRIEVAL_NUMERIC_FIELDS = (
    "top_k",
    "score_threshold",
    "candidate_multiplier",
    "candidate_cap",
    "identifier_candidate_cap",
    "exact_match_boost",
    "confusable_penalty",
    "identifier_vector_floor",
)

# 需以整數界限驗證的數值欄位（其餘為浮點）。
_INT_FIELDS = frozenset(
    {
        "target_chars",
        "overlap_chars",
        "max_tokens",
        "context_before_paragraphs",
        "context_after_paragraphs",
        "top_k",
        "candidate_multiplier",
        "candidate_cap",
        "identifier_candidate_cap",
    }
)

# 預設 agent 界限（比 v1.1 結構性驗證更緊，用於收斂 LLM 的搜尋空間）。
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "target_chars": (200, 1200),
    "overlap_chars": (0, 400),
    "max_tokens": (128, 1024),
    "context_before_paragraphs": (0, 3),
    "context_after_paragraphs": (0, 3),
    "top_k": (1, 10),
    "score_threshold": (0.0, 1.0),
    "candidate_multiplier": (1, 10),
    "candidate_cap": (1, 200),
    "identifier_candidate_cap": (1, 100),
    "exact_match_boost": (0.0, 1.0),
    "confusable_penalty": (0.0, 1.0),
    "identifier_vector_floor": (0.0, 1.0),
}

_DEFAULT_CHUNK_STRATEGIES = ("current", "paragraph", "heading")
_DEFAULT_RETRIEVAL_STRATEGIES = ("dense", "identifier_aware")


class AgentRunConfigError(Exception):
    """Agent Run Config 驗證錯誤（訊息需可理解、指出違規欄位）。"""


# --------------------------------------------------------------------------
# Allowed parameter space
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AllowedParameterSpace:
    """LLM 唯一可提案的白名單欄位與數值界限。

    Slice 1 Validator 會拒絕：白名單外的欄位、超出界限的值、不在允許集合的 strategy。
    """

    chunk_strategies: tuple[str, ...]
    retrieval_strategies: tuple[str, ...]
    bounds: dict[str, tuple[float, float]]
    bool_fields: tuple[str, ...] = CHUNK_BOOL_FIELDS

    def allowed_chunk_fields(self) -> frozenset[str]:
        return frozenset(("strategy",) + CHUNK_NUMERIC_FIELDS + self.bool_fields)

    def allowed_retrieval_fields(self) -> frozenset[str]:
        return frozenset(("strategy",) + RETRIEVAL_NUMERIC_FIELDS)

    def bound_for(self, field_name: str) -> Optional[tuple[float, float]]:
        return self.bounds.get(field_name)

    def is_int_field(self, field_name: str) -> bool:
        return field_name in _INT_FIELDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_strategies": list(self.chunk_strategies),
            "retrieval_strategies": list(self.retrieval_strategies),
            "bounds": {k: list(v) for k, v in sorted(self.bounds.items())},
            "bool_fields": list(self.bool_fields),
        }

    @classmethod
    def from_raw(cls, raw: Optional[dict[str, Any]]) -> "AllowedParameterSpace":
        raw = dict(raw or {})
        unknown = set(raw) - {
            "chunk_strategies",
            "retrieval_strategies",
            "bounds",
            "bool_fields",
        }
        if unknown:
            raise AgentRunConfigError(
                f"allowed_parameter_space 含未知欄位：{sorted(unknown)}"
            )

        chunk_strategies = _str_tuple(
            raw.get("chunk_strategies", _DEFAULT_CHUNK_STRATEGIES),
            "allowed_parameter_space.chunk_strategies",
        )
        retrieval_strategies = _str_tuple(
            raw.get("retrieval_strategies", _DEFAULT_RETRIEVAL_STRATEGIES),
            "allowed_parameter_space.retrieval_strategies",
        )

        bounds = dict(_DEFAULT_BOUNDS)
        raw_bounds = raw.get("bounds")
        if raw_bounds is not None:
            if not isinstance(raw_bounds, dict):
                raise AgentRunConfigError("allowed_parameter_space.bounds 必須是物件。")
            for key, val in raw_bounds.items():
                if key not in _DEFAULT_BOUNDS:
                    raise AgentRunConfigError(
                        f"allowed_parameter_space.bounds 含未知欄位 {key!r}；"
                        f"允許：{sorted(_DEFAULT_BOUNDS)}。"
                    )
                if (
                    not isinstance(val, (list, tuple))
                    or len(val) != 2
                    or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in val)
                ):
                    raise AgentRunConfigError(
                        f"allowed_parameter_space.bounds.{key} 必須是 [min, max] 數值。"
                    )
                lo, hi = float(val[0]), float(val[1])
                if lo > hi:
                    raise AgentRunConfigError(
                        f"allowed_parameter_space.bounds.{key} 的 min({lo}) 不得大於 max({hi})。"
                    )
                bounds[key] = (lo, hi)

        bool_fields = _str_tuple(
            raw.get("bool_fields", CHUNK_BOOL_FIELDS),
            "allowed_parameter_space.bool_fields",
        )
        for b in bool_fields:
            if b not in CHUNK_BOOL_FIELDS:
                raise AgentRunConfigError(
                    f"allowed_parameter_space.bool_fields 含不支援欄位 {b!r}；"
                    f"允許：{list(CHUNK_BOOL_FIELDS)}。"
                )
        return cls(
            chunk_strategies=chunk_strategies,
            retrieval_strategies=retrieval_strategies,
            bounds=bounds,
            bool_fields=bool_fields,
        )


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentBudget:
    """所有硬性上限；Orchestrator 停止條件的唯一來源。"""

    max_rounds: int = 2
    max_llm_calls: int = 4
    max_probe_questions: int = 3
    max_parameter_changes: int = 2
    no_improvement_rounds: int = 1
    max_consecutive_rejections: int = 2
    context_char_budget: int = 12000

    _FIELDS = (
        "max_rounds",
        "max_llm_calls",
        "max_probe_questions",
        "max_parameter_changes",
        "no_improvement_rounds",
        "max_consecutive_rejections",
        "context_char_budget",
    )

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self._FIELDS}

    @classmethod
    def from_raw(cls, raw: Optional[dict[str, Any]]) -> "AgentBudget":
        raw = dict(raw or {})
        unknown = set(raw) - set(cls._FIELDS)
        if unknown:
            raise AgentRunConfigError(f"budget 含未知欄位：{sorted(unknown)}")
        values: dict[str, int] = {}
        defaults = cls()
        for f in cls._FIELDS:
            val = raw.get(f, getattr(defaults, f))
            if isinstance(val, bool) or not isinstance(val, int):
                raise AgentRunConfigError(f"budget.{f} 必須是整數，得到 {val!r}。")
            if val < 1:
                raise AgentRunConfigError(f"budget.{f} 必須 >= 1，得到 {val}。")
            values[f] = val
        return cls(**values)


# --------------------------------------------------------------------------
# Promotion policy（重用 v1.1 constraint / recommendation ordering）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PromotionPolicy:
    """Candidate → 下一輪 Parent 的判定政策（§二正式規則）。

    - ``hard_constraints`` 的 absolute constraint 用於 validation gate（升級條件 3）；
      relative（``baseline_delta``）constraint 用於 non-regression（升級條件 4，baseline
      在執行期綁定為 **當前 Parent** 的 metrics）。
    - ``comparator_ranking`` 為 deterministic lexicographic 排序（重用 v1.1 recommendation
      ordering，最後一鍵必為 ``config_hash``），用於升級條件 5「Candidate 是否優於 Parent」。
      **LLM 不參與。**
    """

    hard_constraints: tuple[HardConstraint, ...]
    comparator_ranking: tuple[SortKey, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_constraints": [c.to_dict() for c in self.hard_constraints],
            "comparator_ranking": [k.to_dict() for k in self.comparator_ranking],
        }

    @classmethod
    def from_raw(cls, raw: Optional[dict[str, Any]]) -> "PromotionPolicy":
        if not isinstance(raw, dict):
            raise AgentRunConfigError("promotion 必須是物件。")
        unknown = set(raw) - {"hard_constraints", "comparator_ranking"}
        if unknown:
            raise AgentRunConfigError(f"promotion 含未知欄位：{sorted(unknown)}")

        raw_constraints = raw.get("hard_constraints")
        if not isinstance(raw_constraints, list) or not raw_constraints:
            raise AgentRunConfigError("promotion.hard_constraints 必須是非空陣列。")
        try:
            constraints = tuple(
                _parse_constraint(c, i) for i, c in enumerate(raw_constraints)
            )
            comparator = _parse_sort_keys(
                raw.get("comparator_ranking"), "promotion.comparator_ranking"
            )
        except SearchConfigError as exc:
            raise AgentRunConfigError(f"promotion 設定無效：{exc}") from exc

        names = [c.name for c in constraints]
        if len(set(names)) != len(names):
            raise AgentRunConfigError("promotion.hard_constraints 的 name 不得重複。")
        return cls(hard_constraints=constraints, comparator_ranking=comparator)


# --------------------------------------------------------------------------
# AgentRunConfig
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentRunConfig:
    """一次 Agent 最佳化的完整設定。"""

    run_name: str
    dataset_path: str
    corpus_paths: tuple[str, ...]
    baseline: TrialConfig
    allowed_parameter_space: AllowedParameterSpace
    budget: AgentBudget
    promotion: PromotionPolicy
    random_seed: int = 42
    prompt_version: str = DEFAULT_PROMPT_VERSION
    schema_version: str = AGENT_RUN_SCHEMA_VERSION

    _KNOWN_KEYS = frozenset(
        {
            "schema_version",
            "run_name",
            "dataset_path",
            "corpus_paths",
            "random_seed",
            "prompt_version",
            "baseline",
            "allowed_parameter_space",
            "budget",
            "promotion",
        }
    )

    def build_run_config(
        self, dataset_path: str, trial: TrialConfig, run_name: str
    ) -> RunConfig:
        """組出可交給 v1.1 ``execute_run`` 的單一 Trial RunConfig（在指定 split dataset 上）。

        Orchestrator 會以 tune / validation 的 split dataset 路徑各呼叫一次。
        """
        return RunConfig(
            run_name=run_name,
            dataset_path=dataset_path,
            corpus_paths=self.corpus_paths,
            trials=(trial,),
            random_seed=self.random_seed,
            fail_fast=True,
            keep_trial_collections=True,  # collection 生命週期由 Orchestrator 管理（Slice 3/4）
        )

    def sanitized(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "dataset_path": self.dataset_path,
            "corpus_paths": list(self.corpus_paths),
            "random_seed": self.random_seed,
            "prompt_version": self.prompt_version,
            "baseline": {
                "chunk_config": self.baseline.chunk_config,
                "retrieval_config": self.baseline.retrieval_config(),
            },
            "allowed_parameter_space": self.allowed_parameter_space.to_dict(),
            "budget": self.budget.to_dict(),
            "promotion": self.promotion.to_dict(),
        }

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "AgentRunConfig":
        if not isinstance(raw, dict):
            raise AgentRunConfigError("agent run config 最外層必須是物件。")
        unknown = set(raw) - cls._KNOWN_KEYS
        if unknown:
            raise AgentRunConfigError(f"agent run config 含未知欄位：{sorted(unknown)}")

        schema_version = str(raw.get("schema_version", AGENT_RUN_SCHEMA_VERSION))
        if schema_version != AGENT_RUN_SCHEMA_VERSION:
            raise AgentRunConfigError(
                f"不支援的 schema_version={schema_version!r}"
                f"（本版支援 {AGENT_RUN_SCHEMA_VERSION!r}）。"
            )

        run_name = raw.get("run_name")
        if not isinstance(run_name, str) or not run_name.strip():
            raise AgentRunConfigError("agent run config 缺少 run_name。")

        try:
            dataset_path = _relative_path(raw.get("dataset_path"), "dataset_path")
            corpus_raw = raw.get("corpus_paths")
            if not isinstance(corpus_raw, list) or not corpus_raw:
                raise AgentRunConfigError("corpus_paths 必須是非空陣列。")
            corpus_paths = tuple(
                _relative_path(p, f"corpus_paths[{i}]")
                for i, p in enumerate(corpus_raw)
            )
        except SearchConfigError as exc:
            raise AgentRunConfigError(str(exc)) from exc

        seed = raw.get("random_seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AgentRunConfigError("random_seed 必須是整數。")

        prompt_version = str(raw.get("prompt_version", DEFAULT_PROMPT_VERSION)).strip()
        if not prompt_version:
            raise AgentRunConfigError("prompt_version 必須是非空字串。")

        # -- baseline（重用 v1.1 TrialConfig 嚴格驗證）--
        baseline_raw = raw.get("baseline")
        if not isinstance(baseline_raw, dict):
            raise AgentRunConfigError("baseline 必須是物件。")
        unknown_b = set(baseline_raw) - {"chunk_config", "retrieval_config"}
        if unknown_b:
            raise AgentRunConfigError(f"baseline 含未知欄位：{sorted(unknown_b)}")
        try:
            baseline = TrialConfig.from_raw(
                {
                    "trial_name": "baseline",
                    "chunk_config": baseline_raw.get("chunk_config") or {"strategy": "current"},
                    "retrieval_config": baseline_raw.get("retrieval_config") or {},
                },
                0,
            )
        except RunConfigError as exc:
            raise AgentRunConfigError(f"baseline 無效：{exc}") from exc

        space = AllowedParameterSpace.from_raw(raw.get("allowed_parameter_space"))
        budget = AgentBudget.from_raw(raw.get("budget"))
        promotion = PromotionPolicy.from_raw(raw.get("promotion"))

        # baseline 的 strategy 必須落在 allowed space（否則第一輪起點即非法）。
        if baseline.chunk_config["strategy"] not in space.chunk_strategies:
            raise AgentRunConfigError(
                f"baseline chunk strategy={baseline.chunk_config['strategy']!r} "
                f"不在 allowed_parameter_space.chunk_strategies {list(space.chunk_strategies)} 內。"
            )
        if baseline.retrieval["strategy"] not in space.retrieval_strategies:
            raise AgentRunConfigError(
                f"baseline retrieval strategy={baseline.retrieval['strategy']!r} "
                f"不在 allowed_parameter_space.retrieval_strategies "
                f"{list(space.retrieval_strategies)} 內。"
            )

        return cls(
            run_name=run_name.strip(),
            dataset_path=dataset_path,
            corpus_paths=corpus_paths,
            baseline=baseline,
            allowed_parameter_space=space,
            budget=budget,
            promotion=promotion,
            random_seed=seed,
            prompt_version=prompt_version,
            schema_version=schema_version,
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise AgentRunConfigError(f"{where} 必須是非空陣列。")
    out: list[str] = []
    for i, v in enumerate(value):
        if not isinstance(v, str) or not v.strip():
            raise AgentRunConfigError(f"{where}[{i}] 必須是非空字串。")
        if v in out:
            raise AgentRunConfigError(f"{where} 含重複值：{v!r}")
        out.append(v)
    return tuple(out)


def load_agent_run_config(path: str) -> AgentRunConfig:
    """由 JSON 檔載入並驗證 Agent Run Config。"""
    import json

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return AgentRunConfig.from_raw(raw)
