"""Search / Candidate 持久化模型（Slice 5 §十一）。

純資料模型，不依賴 FastAPI、不依賴 Qdrant。對應 Optimizer SQLite 的
``optimizer_searches`` / ``optimizer_candidates`` 兩張表（migration ``0005``）。

**Recommendation Lock**（§12.4）是本 Slice 最重要的狀態機不變式：

- ``recommendation_locked`` 為 False 時，``test_trial_id`` 必須全為 NULL、
  ``test_evaluated`` 必須為 False（§13.3：test metric 在 lock 前不存在）。
- 一旦 lock，``recommended_config_hash`` 不得再變更 —— 即使 test 結果較差（§12.5）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class SearchStatus:
    PENDING = "pending"
    RUNNING = "running"
    #: tune + validation 都跑完、constraint / pareto / recommendation 都算完且已 lock。
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    #: §18.5：沒有任何 candidate 通過 validation hard constraints。
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    #: §五：validation split 不足以計算必要的 hard constraint。
    INSUFFICIENT_VALIDATION_COVERAGE = "insufficient_validation_coverage"
    #: Slice 5A §2.3：已 lock 並跑完 test，但 recommendation 未通過 test 的**絕對**
    #: hard constraint（recommendation 不變）。這與「相對 baseline 退化」是不同概念 ——
    #: 即使 baseline 同樣未達標，仍記為 constraint failed。
    TEST_CONSTRAINT_FAILED = "test_constraint_failed"
    #: §12.5：test 絕對門檻通過，但**相對 baseline** 全面退化（recommendation 不變）。
    TEST_REGRESSION = "test_regression"
    #: §12.5：test 階段執行失敗（recommendation 保留為原鎖定結果）。
    TEST_FAILED = "test_failed"


class CandidateStageStatus:
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    #: 未進入該階段（例如未被選入 validation top-N，或未被選為 test 的兩個 config）。
    NOT_SELECTED = "not_selected"
    #: lock 之前 test 階段一律為此狀態（§12.4）。
    LOCKED_OUT = "locked_out"


@dataclass
class OptimizerSearch:
    """一次 search 的完整狀態（§11 optimizer_searches）。"""

    search_id: str
    search_name: str
    status: str
    search_mode: str
    search_config_json: str
    search_config_hash: str
    dataset_version: Optional[str] = None
    dataset_hash: Optional[str] = None
    corpus_hash: Optional[str] = None
    code_commit: Optional[str] = None
    working_tree_dirty: Optional[bool] = None
    random_seed: int = 42
    max_candidates: int = 32
    validation_top_n: int = 8
    baseline_config_hash: Optional[str] = None
    recommended_config_hash: Optional[str] = None
    recommendation_locked: bool = False
    recommendation_locked_at: Optional[str] = None
    test_evaluated: bool = False
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    report_json_path: Optional[str] = None
    report_markdown_path: Optional[str] = None
    # -- 額外保存（§9.9 / §10.9 / §21.6 resume 驗證）--
    tune_run_id: Optional[str] = None
    validation_run_id: Optional[str] = None
    test_run_id: Optional[str] = None
    generation_stats_json: Optional[str] = None
    candidate_manifest_json: Optional[str] = None
    split_manifest_json: Optional[str] = None
    #: Slice 5A（§二／§三）：test 驗收語意 + apply eligibility + 樣本粒度 + latency tie 分析。
    #: 單一 JSON blob，避免為每個欄位各開一欄。
    outcome_json: Optional[str] = None

    def assert_test_allowed(self) -> None:
        """§12.4 / §13.3：lock 之前不得建立或執行任何 test trial。"""
        if not self.recommendation_locked:
            raise RecommendationLockError(
                f"search {self.search_id!r} 尚未 lock recommendation，"
                "不得執行或建立 test trial（§12.4 / §13.3）。"
            )
        if not self.recommended_config_hash:
            raise RecommendationLockError(
                f"search {self.search_id!r} 已標記 locked 但無 recommended_config_hash，"
                "狀態不一致，拒絕進入 test 階段。"
            )


@dataclass
class SearchCandidate:
    """單一 candidate 在整條流程中的狀態（§11 optimizer_candidates）。"""

    search_id: str
    candidate_id: str
    ordinal: int
    config_json: str
    generation_status: str
    config_hash: Optional[str] = None
    generation_rejection_reason: Optional[str] = None
    duplicate_of_hash: Optional[str] = None
    tune_trial_id: Optional[str] = None
    validation_trial_id: Optional[str] = None
    test_trial_id: Optional[str] = None
    tune_status: str = CandidateStageStatus.PENDING
    validation_status: str = CandidateStageStatus.NOT_SELECTED
    test_status: str = CandidateStageStatus.LOCKED_OUT
    constraint_results_json: Optional[str] = None
    feasible: Optional[bool] = None
    pareto_member: Optional[bool] = None
    pareto_rank: Optional[int] = None
    recommendation_rank: Optional[int] = None
    rejection_reasons_json: Optional[str] = None
    tune_rank: Optional[int] = None
    is_baseline: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "ordinal": self.ordinal,
            "config_hash": self.config_hash,
            "generation_status": self.generation_status,
            "generation_rejection_reason": self.generation_rejection_reason,
            "duplicate_of_hash": self.duplicate_of_hash,
            "is_baseline": self.is_baseline,
            "tune_trial_id": self.tune_trial_id,
            "validation_trial_id": self.validation_trial_id,
            "test_trial_id": self.test_trial_id,
            "tune_status": self.tune_status,
            "validation_status": self.validation_status,
            "test_status": self.test_status,
            "tune_rank": self.tune_rank,
            "feasible": self.feasible,
            "pareto_member": self.pareto_member,
            "pareto_rank": self.pareto_rank,
            "recommendation_rank": self.recommendation_rank,
        }


class RecommendationLockError(Exception):
    """違反 Recommendation Lock 不變式（§12.4 / §12.5 / §13.3）。"""


class SearchResumeError(Exception):
    """Resume 前置條件不符（§21.6 / §21.7）。"""

    def __init__(self, message: str, code: str = "resume_rejected"):
        super().__init__(message)
        self.code = code
