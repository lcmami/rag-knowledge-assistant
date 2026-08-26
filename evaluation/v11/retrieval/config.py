"""版本化 Retrieval Strategy Config（v1.1 Slice 4 §八）。

嚴格驗證、deterministic canonical 序列化的 Retrieval 設定。Config Hash 由
:mod:`evaluation.v11.hashing` 對本模組 :meth:`RetrievalConfig.to_canonical_dict`
輸出計算，故：

- 任一有效 ranking 參數改變 → canonical dict 改變 → config hash 改變（§八.9）。
- JSON key 順序不影響 hash（§八.10）。

Strategy 欄位適用矩陣（§八.11：dense 不適用的 identifier 欄位必須**明確拒絕**，不得默默忽略）：

===========================  =====  ================
field                        dense  identifier_aware
===========================  =====  ================
top_k                        用     用
score_threshold              用     用
candidate_multiplier         用     用
candidate_cap                用     用
identifier_candidate_cap     拒絕   用
exact_match_boost            拒絕   用
confusable_penalty           拒絕   用
identifier_vector_floor      拒絕   用
===========================  =====  ================

**v1.2.1 rerank 為正交欄位**（與 strategy 無關，dense / identifier_aware 皆可搭配）：

========================  ==========  =========================
field                     none        cross_encoder
========================  ==========  =========================
rerank_strategy           用（預設）  用
rerank_top_n              拒絕        用（預設 5，須 >= top_k）
========================  ==========  =========================

``ranking_version`` 為 (strategy, rerank_strategy) 的衍生常數（§八.13）：canonical dict
一律以對應版本寫入；若使用者提供且不符即報錯（§六.12 不默默忽略）。

**回溯相容硬需求**：``rerank_strategy=none`` 時 :meth:`RetrievalConfig.to_canonical_dict`
的輸出與 v1.1 **byte-identical**（不輸出任何 rerank 欄位、``ranking_version`` 不變），
故所有既有 config hash、dataset、baseline 與已 locked 的推薦全部維持有效。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

RETRIEVAL_CONFIG_SCHEMA_VERSION = "1.0"

STRATEGY_DENSE = "dense"
STRATEGY_IDENTIFIER_AWARE = "identifier_aware"

SUPPORTED_STRATEGIES = (STRATEGY_DENSE, STRATEGY_IDENTIFIER_AWARE)

# --- v1.2.1 rerank（正交於 strategy，非第三個 strategy）---------------------
RERANK_NONE = "none"
RERANK_CROSS_ENCODER = "cross_encoder"
SUPPORTED_RERANK_STRATEGIES = (RERANK_NONE, RERANK_CROSS_ENCODER)

#: rerank model 是**固定版本資訊,不是搜尋維度**（比照 embedding_model）。
#: 換模型 = 換 fingerprint = 換 baseline,不是候選比較。
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"

#: R0.5 candidate coverage audit（tune + validation only）選出並 FROZEN 的 window。
DEFAULT_RERANK_TOP_N = 5

# Ranking 演算法版本（§八.13 / §十六）：任一 strategy 的 ranking 邏輯變更須 bump。
DENSE_RANKING_VERSION = "dense_v1"
IDENTIFIER_AWARE_RANKING_VERSION = "identifier_aware_v1"
CROSS_ENCODER_RANKING_SUFFIX = "+cross_encoder_v1"

_RANKING_VERSION_BY_STRATEGY = {
    STRATEGY_DENSE: DENSE_RANKING_VERSION,
    STRATEGY_IDENTIFIER_AWARE: IDENTIFIER_AWARE_RANKING_VERSION,
}

# identifier_aware 專屬欄位（dense 若出現即拒絕）。
_IDENTIFIER_FIELDS = (
    "identifier_candidate_cap",
    "exact_match_boost",
    "confusable_penalty",
    "identifier_vector_floor",
)

# 共用欄位。
_COMMON_FIELDS = ("top_k", "score_threshold", "candidate_multiplier", "candidate_cap")

#: rerank 專屬欄位（``rerank_strategy=none`` 時若出現即拒絕，比照 _IDENTIFIER_FIELDS）。
_RERANK_ONLY_FIELDS = ("rerank_top_n",)

_META_KEYS = {"schema_version", "strategy", "ranking_version"}
_ALLOWED_KEYS = (
    _META_KEYS
    | set(_COMMON_FIELDS)
    | set(_IDENTIFIER_FIELDS)
    | {"rerank_strategy"}
    | set(_RERANK_ONLY_FIELDS)
)

# identifier_aware 預設值（Slice 4 明確；Slice 5 才做搜尋/最佳化）。
_DEFAULTS: dict[str, Any] = {
    "top_k": 5,
    "score_threshold": 0.55,
    "candidate_multiplier": 3,
    "candidate_cap": 50,
    "identifier_candidate_cap": 20,
    "exact_match_boost": 0.2,
    "confusable_penalty": 0.15,
    "identifier_vector_floor": 0.4,
}


class RetrievalConfigError(ValueError):
    """Retrieval Config 驗證錯誤（訊息需可理解、指出違反的規則）。"""


def _finite_number(name: str, val: Any) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise RetrievalConfigError(f"{name} 必須為數值，得到 {val!r}。")
    f = float(val)
    if not math.isfinite(f):
        raise RetrievalConfigError(f"{name} 不得為 NaN / Infinity，得到 {val!r}。")
    return f


def _positive_int(name: str, val: Any) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise RetrievalConfigError(f"{name} 必須為整數，得到 {val!r}。")
    if val <= 0:
        raise RetrievalConfigError(f"{name} 必須 > 0，得到 {val}。")
    return val


@dataclass(frozen=True)
class RetrievalConfig:
    """單一 Retrieval Strategy 的嚴格驗證、版本化設定。"""

    strategy: str
    top_k: int = _DEFAULTS["top_k"]
    score_threshold: float = _DEFAULTS["score_threshold"]
    candidate_multiplier: int = _DEFAULTS["candidate_multiplier"]
    candidate_cap: int = _DEFAULTS["candidate_cap"]
    # identifier_aware 專屬（dense 時為 None，且 canonical dict 不輸出）。
    identifier_candidate_cap: int | None = None
    exact_match_boost: float | None = None
    confusable_penalty: float | None = None
    identifier_vector_floor: float | None = None
    # v1.2.1 rerank（正交欄位；none 時不輸出於 canonical dict，故舊 hash 不變）。
    rerank_strategy: str = RERANK_NONE
    rerank_top_n: int | None = None
    schema_version: str = RETRIEVAL_CONFIG_SCHEMA_VERSION

    @property
    def ranking_version(self) -> str:
        """(strategy, rerank_strategy) 的衍生常數。

        ``rerank_strategy=none`` 時**與 v1.1 完全相同**，確保既有 config hash 不變。
        """
        base = _RANKING_VERSION_BY_STRATEGY[self.strategy]
        if self.rerank_strategy == RERANK_CROSS_ENCODER:
            return base + CROSS_ENCODER_RANKING_SUFFIX
        return base

    @property
    def is_identifier_aware(self) -> bool:
        return self.strategy == STRATEGY_IDENTIFIER_AWARE

    @property
    def is_rerank_enabled(self) -> bool:
        return self.rerank_strategy != RERANK_NONE

    def candidate_limit(self, top_k: int | None = None) -> int:
        """Dense 候選數（§十一.1）：沿用既有公式 max(top_k, min(top_k*mult, cap))。"""
        k = top_k if top_k is not None else self.top_k
        return max(k, min(k * self.candidate_multiplier, self.candidate_cap))

    # ------------------------------------------------------------------
    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "RetrievalConfig":
        """由 dict 建立並嚴格驗證。``None`` / 空 dict → 預設 ``dense`` config（向後相容）。"""
        raw = dict(raw or {})
        raw.setdefault("strategy", STRATEGY_DENSE)

        unknown = set(raw) - _ALLOWED_KEYS
        if unknown:
            raise RetrievalConfigError(
                f"retrieval_config 含未知欄位（不得默默忽略）：{sorted(unknown)}；"
                f"允許欄位：{sorted(_ALLOWED_KEYS)}。"
            )

        strategy = raw["strategy"]
        if strategy not in SUPPORTED_STRATEGIES:
            raise RetrievalConfigError(
                f"不支援的 retrieval strategy={strategy!r}；支援：{list(SUPPORTED_STRATEGIES)}。"
            )

        # §八.11：dense 不適用 identifier 欄位——若出現即明確拒絕。
        if strategy == STRATEGY_DENSE:
            present = [k for k in _IDENTIFIER_FIELDS if k in raw]
            if present:
                raise RetrievalConfigError(
                    f"strategy=dense 不適用 identifier 欄位 {present}；"
                    f"請改用 strategy=identifier_aware 或移除。"
                )

        # --- v1.2.1 rerank（正交欄位）---
        rerank_strategy = raw.get("rerank_strategy", RERANK_NONE)
        if rerank_strategy not in SUPPORTED_RERANK_STRATEGIES:
            raise RetrievalConfigError(
                f"不支援的 rerank_strategy={rerank_strategy!r}；"
                f"支援：{list(SUPPORTED_RERANK_STRATEGIES)}。"
            )
        # 比照 §八.11：不適用的欄位必須**明確拒絕**，不得默默忽略。
        if rerank_strategy == RERANK_NONE:
            present = [k for k in _RERANK_ONLY_FIELDS if k in raw]
            if present:
                raise RetrievalConfigError(
                    f"rerank_strategy=none 不適用 rerank 欄位 {present}；"
                    f"請設 rerank_strategy={RERANK_CROSS_ENCODER!r} 或移除。"
                )

        # §八.13：ranking_version 為 (strategy, rerank_strategy) 衍生常數；不符即拒絕。
        expected_rv = _RANKING_VERSION_BY_STRATEGY[strategy]
        if rerank_strategy == RERANK_CROSS_ENCODER:
            expected_rv += CROSS_ENCODER_RANKING_SUFFIX
        provided_rv = raw.get("ranking_version")
        if provided_rv is not None and provided_rv != expected_rv:
            raise RetrievalConfigError(
                f"strategy={strategy!r} / rerank_strategy={rerank_strategy!r} 的 "
                f"ranking_version 必須為 {expected_rv!r}，得到 {provided_rv!r}（不得覆寫）。"
            )

        top_k = _positive_int("top_k", raw.get("top_k", _DEFAULTS["top_k"]))
        candidate_multiplier = _positive_int(
            "candidate_multiplier", raw.get("candidate_multiplier", _DEFAULTS["candidate_multiplier"])
        )
        candidate_cap = _positive_int(
            "candidate_cap", raw.get("candidate_cap", _DEFAULTS["candidate_cap"])
        )
        if candidate_cap < top_k:
            raise RetrievalConfigError(
                f"candidate_cap ({candidate_cap}) 必須 >= top_k ({top_k})。"
            )
        score_threshold = _finite_number(
            "score_threshold", raw.get("score_threshold", _DEFAULTS["score_threshold"])
        )

        id_cap = boost = penalty = floor = None
        if strategy == STRATEGY_IDENTIFIER_AWARE:
            id_cap = _positive_int(
                "identifier_candidate_cap",
                raw.get("identifier_candidate_cap", _DEFAULTS["identifier_candidate_cap"]),
            )
            boost = _finite_number(
                "exact_match_boost", raw.get("exact_match_boost", _DEFAULTS["exact_match_boost"])
            )
            penalty = _finite_number(
                "confusable_penalty", raw.get("confusable_penalty", _DEFAULTS["confusable_penalty"])
            )
            floor = _finite_number(
                "identifier_vector_floor",
                raw.get("identifier_vector_floor", _DEFAULTS["identifier_vector_floor"]),
            )
            if boost < 0:
                raise RetrievalConfigError("exact_match_boost 不得為負。")
            if penalty < 0:
                raise RetrievalConfigError("confusable_penalty 不得為負。")

        rerank_top_n = None
        if rerank_strategy == RERANK_CROSS_ENCODER:
            rerank_top_n = _positive_int(
                "rerank_top_n", raw.get("rerank_top_n", DEFAULT_RERANK_TOP_N)
            )
            # W/T 語義要求 window 至少涵蓋最終輸出，否則 tail 會擠掉被 rerank 的結果。
            if rerank_top_n < top_k:
                raise RetrievalConfigError(
                    f"rerank_top_n ({rerank_top_n}) 必須 >= top_k ({top_k})。"
                )

        return cls(
            strategy=strategy,
            top_k=top_k,
            score_threshold=score_threshold,
            candidate_multiplier=candidate_multiplier,
            candidate_cap=candidate_cap,
            identifier_candidate_cap=id_cap,
            exact_match_boost=boost,
            confusable_penalty=penalty,
            identifier_vector_floor=floor,
            rerank_strategy=rerank_strategy,
            rerank_top_n=rerank_top_n,
            schema_version=str(raw.get("schema_version", RETRIEVAL_CONFIG_SCHEMA_VERSION)),
        )

    # ------------------------------------------------------------------
    def to_canonical_dict(self) -> dict[str, Any]:
        """deterministic、可還原的 config dict（不含私密路徑）。

        永遠包含 strategy 對應的所有欄位（含 ranking_version），故任一有效 ranking 參數
        改變都會反映於 config hash。dense 不輸出 identifier 欄位（§八.11）。
        """
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "candidate_multiplier": self.candidate_multiplier,
            "candidate_cap": self.candidate_cap,
            "ranking_version": self.ranking_version,
        }
        if self.is_identifier_aware:
            d["identifier_candidate_cap"] = self.identifier_candidate_cap
            d["exact_match_boost"] = self.exact_match_boost
            d["confusable_penalty"] = self.confusable_penalty
            d["identifier_vector_floor"] = self.identifier_vector_floor
        # **回溯相容硬需求**：rerank_strategy=none 時完全不輸出 rerank 欄位，
        # 使 canonical dict 與 v1.1 byte-identical → 所有既有 config_hash 不變
        # （含已 locked 的推薦 e246697d41 與既有 dataset / baseline）。
        if self.is_rerank_enabled:
            d["rerank_strategy"] = self.rerank_strategy
            d["rerank_top_n"] = self.rerank_top_n
        return d
