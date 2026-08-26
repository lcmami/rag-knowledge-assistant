"""Grid / Random Candidate 生成（Slice 5 §八、§九、§十）。

流程：

1. 以 baseline config 為底，套用搜尋維度的值 → raw combination（deterministic Cartesian product）。
2. **Conditional projection**（§八）：``dense`` strategy 不得帶 identifier 欄位，
   故生成 dense candidate 時明確移除這些欄位（並記錄為 projection，不是默默塞值）。
3. 交由既有 :class:`ChunkingConfig` / :class:`RetrievalConfig` 正規化與驗證；
   失敗 → 保存 rejection reason，**不執行 trial**（§9.4）。
4. 以 canonical config hash 去重（§9.5）。
5. Budget（§9.7 / §10.7）：grid 超過 ``max_candidates`` → 明確失敗，**不默默截斷**；
   random 則以固定 seed without replacement 取樣。

candidate 的 ``config_hash`` 直接以 :class:`TrialConfig` 計算，與 ``retrieval_trials.config_hash``
完全一致，因此 candidate ↔ trial 可以用 hash 直接 join。
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from evaluation.v11.chunking.config import ChunkConfigError, ChunkingConfig
from evaluation.v11.hashing import canonical_json
from evaluation.v11.retrieval.config import RetrievalConfig, RetrievalConfigError
from evaluation.v11.run_models import TrialConfig
from evaluation.v11.search.search_config import (
    SEARCH_MODE_GRID,
    SEARCH_MODE_RANDOM,
    SearchConfig,
)

#: 只在 identifier_aware strategy 才適用的 retrieval 欄位（§八）。
IDENTIFIER_ONLY_FIELDS = (
    "identifier_candidate_cap",
    "exact_match_boost",
    "confusable_penalty",
    "identifier_vector_floor",
)

#: 只在 ``rerank_strategy != none`` 才適用的欄位（v1.2.1 §R3，比照 IDENTIFIER_ONLY_FIELDS）。
#:
#: :class:`RetrievalConfig` 在 ``rerank_strategy=none`` 時會**明確拒絕**這些欄位
#: （不默默忽略）。若不做 projection，``rerank_strategy × rerank_top_n`` 的
#: Cartesian product 會產生一整排 invalid candidate 而非乾淨的 duplicate，
#: 使 generation stats 失去意義。projection 後 duplicate 語義維持 deterministic：
#: 所有 ``none × 任意 rerank_top_n`` 收斂成同一個 config hash。
RERANK_ONLY_FIELDS = ("rerank_top_n",)

GEN_EXECUTABLE = "executable"
GEN_INVALID = "invalid"
GEN_DUPLICATE = "duplicate"
#: random search 中合法但未被取樣者（既非無效也非重複，需與兩者區分）。
GEN_NOT_SAMPLED = "not_sampled"

#: §9.7 budget 超出時的錯誤代碼。
ERROR_SEARCH_SPACE_EXCEEDS_BUDGET = "search_space_exceeds_budget"


class SearchSpaceError(Exception):
    """Candidate 生成錯誤。``code`` 供報告與測試判定。"""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


@dataclass
class GeneratedCandidate:
    """一個生成出來的 candidate（可能是 executable / invalid / duplicate）。"""

    ordinal: int
    assignment: dict[str, Any]
    generation_status: str
    chunk_config: Optional[dict[str, Any]] = None
    retrieval_config: Optional[dict[str, Any]] = None
    config_hash: Optional[str] = None
    generation_rejection_reason: Optional[str] = None
    duplicate_of_hash: Optional[str] = None
    projected_fields: tuple[str, ...] = ()

    @property
    def is_executable(self) -> bool:
        return self.generation_status == GEN_EXECUTABLE

    def trial_name(self) -> str:
        return f"candidate-{self.ordinal:03d}"

    def to_trial_config(self) -> TrialConfig:
        if not self.is_executable:
            raise SearchSpaceError(
                f"candidate ordinal={self.ordinal} 非 executable，不得建立 TrialConfig。",
                code="candidate_not_executable",
            )
        return TrialConfig(
            trial_name=self.trial_name(),
            chunk_config=dict(self.chunk_config or {}),
            retrieval=dict(self.retrieval_config or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "assignment": self.assignment,
            "generation_status": self.generation_status,
            "config_hash": self.config_hash,
            "generation_rejection_reason": self.generation_rejection_reason,
            "duplicate_of_hash": self.duplicate_of_hash,
            "projected_fields": list(self.projected_fields),
            "chunk_config": self.chunk_config,
            "retrieval_config": self.retrieval_config,
        }


@dataclass
class GenerationResult:
    """§9.9 報告所需的生成統計 + 完整 candidate manifest。"""

    executable: list[GeneratedCandidate] = field(default_factory=list)
    rejected: list[GeneratedCandidate] = field(default_factory=list)
    raw_combination_count: int = 0
    invalid_count: int = 0
    duplicate_count: int = 0
    legal_candidate_count: int = 0
    sampled_count: int = 0
    budget_exhausted: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def executable_count(self) -> int:
        return len(self.executable)

    def manifest(self) -> list[dict[str, Any]]:
        """完整 candidate manifest（含被拒者），依 ordinal 排序（§10.9）。"""
        allc = sorted(self.executable + self.rejected, key=lambda c: c.ordinal)
        return [c.to_dict() for c in allc]

    def stats(self) -> dict[str, Any]:
        return {
            "raw_combination_count": self.raw_combination_count,
            "invalid_count": self.invalid_count,
            "duplicate_count": self.duplicate_count,
            "legal_candidate_count": self.legal_candidate_count,
            "executable_candidate_count": self.executable_count,
            "sampled_count": self.sampled_count,
            "budget_exhausted": self.budget_exhausted,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
def _base_retrieval(baseline: dict[str, Any]) -> dict[str, Any]:
    """baseline retrieval canonical dict 去掉 strategy 衍生欄位，作為覆寫底稿。

    ``ranking_version`` 由 strategy 決定，若沿用 baseline 的值會在切換 strategy 時衝突，
    因此一律移除交給 :class:`RetrievalConfig` 重新推導。
    """
    base = {k: v for k, v in baseline.items() if k != "ranking_version"}
    return base


def _build_raw_configs(
    search_config: SearchConfig, assignment: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """以 baseline 為底套用 assignment，並做 dense 的 conditional projection。"""
    chunk_raw = dict(search_config.baseline_chunk_config)
    retrieval_raw = _base_retrieval(search_config.baseline_retrieval_config)

    for key, value in assignment.items():
        section, _, name = key.partition(".")
        if section == "chunking":
            chunk_raw[name] = value
        else:
            retrieval_raw[name] = value

    # §八：dense 不得帶 identifier 欄位 —— 明確移除並記錄（不是默默塞值）。
    projected: list[str] = []
    if retrieval_raw.get("strategy") == "dense":
        for f in IDENTIFIER_ONLY_FIELDS:
            if f in retrieval_raw:
                retrieval_raw.pop(f)
                projected.append(f)

    # v1.2.1 §R3：rerank_strategy=none 不得帶 rerank 欄位 —— 同樣明確移除並記錄。
    # 注意 rerank 與 strategy **正交**：這個 projection 對 dense 與 identifier_aware
    # 皆適用，不能併進上面的 dense 分支。
    if retrieval_raw.get("rerank_strategy", "none") == "none":
        for f in RERANK_ONLY_FIELDS:
            if f in retrieval_raw:
                retrieval_raw.pop(f)
                projected.append(f)

    # chunk strategy 切換時，prefix_format_version 由 ChunkingConfig 重新產生。
    chunk_raw.pop("prefix_format_version", None)
    return chunk_raw, retrieval_raw, tuple(projected)


def _canonicalize(
    chunk_raw: dict[str, Any], retrieval_raw: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """交給既有 config 類別做正規化 + 嚴格驗證（§八首句）。"""
    chunk = ChunkingConfig.from_raw(chunk_raw).to_canonical_dict()
    retrieval = RetrievalConfig.from_raw(retrieval_raw).to_canonical_dict()
    return chunk, retrieval


def _dimension_axes(search_config: SearchConfig) -> list[tuple[str, tuple[Any, ...]]]:
    """deterministic 的維度順序：chunking.* 先、retrieval.* 後，各自依 key 排序。

    :class:`SearchSpace` 建構時已依 key 排序，這裡再固定 section 順序，
    使 Cartesian product 完全不依賴 dict / set 的插入順序（§9.6）。
    """
    axes: list[tuple[str, tuple[Any, ...]]] = []
    for key in sorted(search_config.search_space.chunking):
        axes.append((f"chunking.{key}", search_config.search_space.chunking[key]))
    for key in sorted(search_config.search_space.retrieval):
        axes.append((f"retrieval.{key}", search_config.search_space.retrieval[key]))
    return axes


def _enumerate_legal(
    search_config: SearchConfig,
    *,
    canonical_text_version: str,
    identifier_normalization_version: str,
) -> GenerationResult:
    """列舉全部 raw combination，做 conditional validation 與去重。

    回傳的 ``executable`` 依「首次出現順序」排序 —— 與 Cartesian product 順序一致，
    因此完全 deterministic（§9.2）。
    """
    result = GenerationResult()
    axes = _dimension_axes(search_config)
    names = [a[0] for a in axes]
    value_lists = [a[1] for a in axes]

    seen_hash: dict[str, int] = {}
    ordinal = 0

    for combo in itertools.product(*value_lists):
        result.raw_combination_count += 1
        assignment = {names[i]: combo[i] for i in range(len(names))}

        chunk_raw, retrieval_raw, projected = _build_raw_configs(search_config, assignment)
        try:
            chunk, retrieval = _canonicalize(chunk_raw, retrieval_raw)
        except (ChunkConfigError, RetrievalConfigError) as exc:
            # §9.4：保存 rejection reason，不執行 trial。
            result.invalid_count += 1
            result.rejected.append(
                GeneratedCandidate(
                    ordinal=ordinal,
                    assignment=assignment,
                    generation_status=GEN_INVALID,
                    generation_rejection_reason=f"{type(exc).__name__}: {exc}",
                    projected_fields=projected,
                )
            )
            ordinal += 1
            continue

        tc = TrialConfig(trial_name="probe", chunk_config=chunk, retrieval=retrieval)
        chash = tc.compute_hash(
            canonical_text_version=canonical_text_version,
            identifier_normalization_version=identifier_normalization_version,
        )

        if chash in seen_hash:
            # §9.5：相同 canonical config hash 去重（dense projection 常造成這種重複）。
            result.duplicate_count += 1
            result.rejected.append(
                GeneratedCandidate(
                    ordinal=ordinal,
                    assignment=assignment,
                    generation_status=GEN_DUPLICATE,
                    chunk_config=chunk,
                    retrieval_config=retrieval,
                    config_hash=chash,
                    duplicate_of_hash=chash,
                    generation_rejection_reason=(
                        "duplicate_config_hash: 與先前 candidate 正規化後完全相同"
                    ),
                    projected_fields=projected,
                )
            )
            ordinal += 1
            continue

        seen_hash[chash] = ordinal
        result.executable.append(
            GeneratedCandidate(
                ordinal=ordinal,
                assignment=assignment,
                generation_status=GEN_EXECUTABLE,
                chunk_config=chunk,
                retrieval_config=retrieval,
                config_hash=chash,
                projected_fields=projected,
            )
        )
        ordinal += 1

    result.legal_candidate_count = len(result.executable)
    all_candidates = result.executable + result.rejected
    if any(f in c.projected_fields for c in all_candidates for f in IDENTIFIER_ONLY_FIELDS):
        result.notes.append(
            "部分組合套用了 conditional projection：dense strategy 不接受 identifier 欄位，"
            f"生成時已明確移除 {list(IDENTIFIER_ONLY_FIELDS)}（§八），"
            "因此多個 identifier 參數組合會收斂成同一個 dense candidate 並計入 duplicate。"
        )
    if any(f in c.projected_fields for c in all_candidates for f in RERANK_ONLY_FIELDS):
        result.notes.append(
            "部分組合套用了 conditional projection：rerank_strategy=none 不接受 rerank 欄位，"
            f"生成時已明確移除 {list(RERANK_ONLY_FIELDS)}（v1.2.1 §R3），"
            "因此多個 rerank_top_n 組合會收斂成同一個 no-rerank candidate 並計入 duplicate。"
        )
    return result


def generate_candidates(
    search_config: SearchConfig,
    *,
    canonical_text_version: str,
    identifier_normalization_version: str,
) -> GenerationResult:
    """依 ``search_mode`` 產生 candidate 清單。

    - ``grid``：完整 Cartesian product；合法 candidate 超過 budget → 明確失敗（§9.7/§9.8）。
    - ``random``：固定 seed、without replacement 取樣（§10）。

    :raises SearchSpaceError: grid 超出 budget，或無任何合法 candidate。
    """
    result = _enumerate_legal(
        search_config,
        canonical_text_version=canonical_text_version,
        identifier_normalization_version=identifier_normalization_version,
    )

    if not result.executable:
        raise SearchSpaceError(
            "搜尋空間沒有任何合法 candidate（全部組合皆無效或重複）。",
            code="no_legal_candidate",
        )

    if search_config.search_mode == SEARCH_MODE_GRID:
        if result.legal_candidate_count > search_config.max_candidates:
            # §9.8：不得默默截斷 grid。
            raise SearchSpaceError(
                f"合法 candidate 數 {result.legal_candidate_count} 超過 "
                f"max_candidates={search_config.max_candidates}；"
                "請縮小 search_space 或提高 budget（grid 不得默默截斷）。",
                code=ERROR_SEARCH_SPACE_EXCEEDS_BUDGET,
            )
        result.sampled_count = result.legal_candidate_count
        return result

    if search_config.search_mode != SEARCH_MODE_RANDOM:  # pragma: no cover - schema 已擋
        raise SearchSpaceError(
            f"未知 search_mode={search_config.search_mode!r}", code="unknown_search_mode"
        )

    # -- Random（§十）--
    legal = result.executable
    budget = search_config.max_candidates
    if len(legal) <= budget:
        # §10.7：合法 candidate 少於 budget → 全部執行並明確記錄。
        result.sampled_count = len(legal)
        result.budget_exhausted = False
        result.notes.append(
            f"random search：合法 candidate 數 {len(legal)} 未超過 max_candidates={budget}，"
            "因此執行全部合法 candidate（未進行取樣）。"
        )
        return result

    # 固定 seed、without replacement。以 config_hash 排序後再取樣，使取樣結果
    # 只依賴 (search_config_hash, seed)，不依賴列舉順序以外的任何 process 狀態（§10.8）。
    rng = random.Random(f"{search_config.compute_hash()}|{search_config.random_seed}")
    pool = sorted(legal, key=lambda c: c.config_hash or "")
    chosen = rng.sample(pool, budget)
    chosen_hashes = {c.config_hash for c in chosen}

    dropped = [c for c in legal if c.config_hash not in chosen_hashes]
    for c in dropped:
        c.generation_status = GEN_NOT_SAMPLED
        c.generation_rejection_reason = (
            "not_sampled: 合法 candidate，但在 random search budget 內未被取樣（未執行）"
        )
    # 保留原 ordinal 順序（deterministic），只留下被取樣者為 executable。
    result.executable = sorted(
        [c for c in legal if c.config_hash in chosen_hashes], key=lambda c: c.ordinal
    )
    result.rejected.extend(sorted(dropped, key=lambda c: c.ordinal))
    result.sampled_count = len(result.executable)
    result.budget_exhausted = True
    result.notes.append(
        f"random search：自 {len(legal)} 個合法 candidate 中以 seed={search_config.random_seed} "
        f"without replacement 取樣 {result.sampled_count} 個；"
        "取樣僅依 search_config_hash 與 seed，未使用任何 tune / validation / test metric。"
    )
    return result


def candidate_manifest_signature(result: GenerationResult) -> str:
    """Candidate manifest 的 deterministic 簽章（供 random 可重現性測試 / resume 驗證）。"""
    return canonical_json(
        [
            {"ordinal": c.ordinal, "config_hash": c.config_hash}
            for c in sorted(result.executable, key=lambda c: c.ordinal)
        ]
    )
