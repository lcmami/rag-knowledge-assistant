"""Run / Trial 執行模型（Slice 2 §六）。

純資料模型（dataclass）+ 狀態常數 + Run/Trial Config 解析驗證。不依賴 FastAPI、
不依賴 Qdrant，可單元測試。

- :class:`TrialConfig`：單一 Trial 的固定 Chunking / Dense Retrieval 設定。
- :class:`RunConfig`：一個 Run 的完整設定（明確列出的多個 Trial + 隔離 / retry 旗標）。
- :class:`EvaluationRun` / :class:`RetrievalTrial` / :class:`TrialCaseResult` /
  :class:`TrialMetric`：持久化用的執行模型（對應 Optimizer SQLite 各表）。

本 Slice 只支援目前已存在的固定 Chunking（``strategy="current"``）與 Dense Retrieval，
**不**實作新的 chunking strategy 或 identifier-aware retrieval；config 欄位保留完整以利
未來 Slice，但不得放入本 Slice 無法執行的設定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from evaluation.v11.chunking.config import (
    SUPPORTED_STRATEGIES,
    ChunkConfigError,
    ChunkingConfig,
)
from evaluation.v11.hashing import canonical_json, config_hash

CONFIG_SCHEMA_VERSION = "1.0"

# Slice 3：支援 current / fixed / paragraph / heading（明確列出的 chunking 搜尋空間）。
SUPPORTED_CHUNK_STRATEGIES = set(SUPPORTED_STRATEGIES)


# --- 狀態常數（§六：語意清楚且有測試）---
class RunStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class TrialStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CleanupStatus:
    NOT_CREATED = "not_created"
    ACTIVE = "active"
    CLEANED = "cleaned"
    CLEANUP_FAILED = "cleanup_failed"


class RunConfigError(Exception):
    """Run / Trial Config 驗證錯誤（訊息需可理解）。"""


# --------------------------------------------------------------------------
# TrialConfig
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialConfig:
    """單一 Trial 的固定 Chunking / Dense Retrieval 設定（§六）。"""

    trial_name: str
    chunk_config: dict[str, Any] = field(default_factory=lambda: {"strategy": "current"})
    top_k: int = 5
    score_threshold: float = 0.55
    candidate_multiplier: int = 3
    candidate_cap: int = 50
    collection_options: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CONFIG_SCHEMA_VERSION

    def retrieval_config(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "candidate_multiplier": self.candidate_multiplier,
            "candidate_cap": self.candidate_cap,
        }

    def hashable(
        self,
        *,
        canonical_text_version: str,
        identifier_normalization_version: str,
    ) -> dict[str, Any]:
        """config_hash 的輸入：涵蓋 chunk / retrieval / 版本欄位（§八 Config Hash）。

        不含 ``trial_name``（僅為人類標籤，不影響語意）；相同語意設定 → 相同 hash。
        """
        return {
            "schema_version": self.schema_version,
            "chunk_config": self.chunk_config,
            "retrieval_config": self.retrieval_config(),
            "canonical_text_version": canonical_text_version,
            "identifier_normalization_version": identifier_normalization_version,
        }

    def compute_hash(
        self,
        *,
        canonical_text_version: str,
        identifier_normalization_version: str,
    ) -> str:
        return config_hash(
            self.hashable(
                canonical_text_version=canonical_text_version,
                identifier_normalization_version=identifier_normalization_version,
            )
        )

    def config_json(self) -> str:
        """完整、可還原的 config（供 retrial_trials.config_json）。"""
        return canonical_json(
            {
                "schema_version": self.schema_version,
                "trial_name": self.trial_name,
                "chunk_config": self.chunk_config,
                "retrieval_config": self.retrieval_config(),
                "collection_options": self.collection_options,
            }
        )

    @classmethod
    def from_raw(cls, raw: dict[str, Any], index: int) -> "TrialConfig":
        if not isinstance(raw, dict):
            raise RunConfigError(f"trials[{index}] 必須是物件。")
        name = raw.get("trial_name")
        if not isinstance(name, str) or not name.strip():
            raise RunConfigError(f"trials[{index}] 缺少 trial_name。")

        raw_chunk = raw.get("chunk_config") or {"strategy": "current"}
        if not isinstance(raw_chunk, dict):
            raise RunConfigError(f"trials[{index}] chunk_config 必須是物件。")
        # 嚴格驗證 + canonical 化（§六）：以 canonical dict 作為 chunk_config，確保 config_hash
        # 涵蓋所有有效 chunk 欄位（含 prefix_format_version）且與 JSON key 順序無關。
        try:
            chunk_config = ChunkingConfig.from_raw(raw_chunk).to_canonical_dict()
        except ChunkConfigError as exc:
            raise RunConfigError(f"trials[{index}] chunk_config 無效：{exc}") from exc

        retrieval = raw.get("retrieval_config") or {}
        if not isinstance(retrieval, dict):
            raise RunConfigError(f"trials[{index}] retrieval_config 必須是物件。")

        top_k = retrieval.get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1:
            raise RunConfigError(f"trials[{index}] top_k 必須為 >=1 的整數。")
        threshold = retrieval.get("score_threshold", 0.55)
        if not isinstance(threshold, (int, float)):
            raise RunConfigError(f"trials[{index}] score_threshold 必須為數值。")

        return cls(
            trial_name=name.strip(),
            chunk_config=chunk_config,
            top_k=top_k,
            score_threshold=float(threshold),
            candidate_multiplier=int(retrieval.get("candidate_multiplier", 3)),
            candidate_cap=int(retrieval.get("candidate_cap", 50)),
            collection_options=raw.get("collection_options") or {},
            schema_version=str(raw.get("schema_version", CONFIG_SCHEMA_VERSION)),
        )


# --------------------------------------------------------------------------
# RunConfig
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """一個 Run 的完整設定（明確列出的多個 Trial）。"""

    run_name: str
    dataset_path: str
    corpus_paths: tuple[str, ...]
    trials: tuple[TrialConfig, ...]
    random_seed: int = 42
    fail_fast: bool = False
    retry_failed: bool = False
    retry_interrupted: bool = True
    keep_trial_collections: bool = False
    schema_version: str = CONFIG_SCHEMA_VERSION

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "RunConfig":
        if not isinstance(raw, dict):
            raise RunConfigError("run config 最外層必須是物件。")

        run_name = raw.get("run_name")
        if not isinstance(run_name, str) or not run_name.strip():
            raise RunConfigError("run config 缺少 run_name。")

        dataset_path = raw.get("dataset_path")
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise RunConfigError("run config 缺少 dataset_path。")

        corpus_paths = raw.get("corpus_paths")
        if not isinstance(corpus_paths, list) or not corpus_paths:
            raise RunConfigError("run config corpus_paths 必須是非空陣列。")
        if not all(isinstance(p, str) and p.strip() for p in corpus_paths):
            raise RunConfigError("run config corpus_paths 必須全為非空字串。")

        raw_trials = raw.get("trials")
        if not isinstance(raw_trials, list) or not raw_trials:
            raise RunConfigError("run config trials 必須是非空陣列（明確列出的 Trial）。")
        trials = tuple(TrialConfig.from_raw(t, i) for i, t in enumerate(raw_trials))

        names = [t.trial_name for t in trials]
        if len(set(names)) != len(names):
            raise RunConfigError("trials 內 trial_name 不得重複。")

        return cls(
            run_name=run_name.strip(),
            dataset_path=dataset_path.strip(),
            corpus_paths=tuple(corpus_paths),
            trials=trials,
            random_seed=int(raw.get("random_seed", 42)),
            fail_fast=bool(raw.get("fail_fast", False)),
            retry_failed=bool(raw.get("retry_failed", False)),
            retry_interrupted=bool(raw.get("retry_interrupted", True)),
            keep_trial_collections=bool(raw.get("keep_trial_collections", False)),
            schema_version=str(raw.get("schema_version", CONFIG_SCHEMA_VERSION)),
        )

    def sanitized(self) -> dict[str, Any]:
        """報告用設定快照（不含私密絕對路徑；corpus_paths 已是相對路徑）。"""
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "dataset_path": self.dataset_path,
            "corpus_paths": list(self.corpus_paths),
            "random_seed": self.random_seed,
            "fail_fast": self.fail_fast,
            "retry_failed": self.retry_failed,
            "retry_interrupted": self.retry_interrupted,
            "keep_trial_collections": self.keep_trial_collections,
            "trials": [
                {
                    "trial_name": t.trial_name,
                    "chunk_config": t.chunk_config,
                    "retrieval_config": t.retrieval_config(),
                }
                for t in self.trials
            ],
        }


# --------------------------------------------------------------------------
# 持久化執行模型
# --------------------------------------------------------------------------
@dataclass
class EvaluationRun:
    run_id: str
    status: str
    run_name: str
    dataset_version: Optional[str] = None
    dataset_hash: Optional[str] = None
    corpus_hash: Optional[str] = None
    code_commit: Optional[str] = None
    working_tree_dirty: Optional[bool] = None
    canonical_text_version: Optional[str] = None
    loader_version: Optional[str] = None
    identifier_normalization_version: Optional[str] = None
    random_seed: int = 42
    config_json: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    report_json_path: Optional[str] = None
    report_markdown_path: Optional[str] = None


@dataclass
class RetrievalTrial:
    trial_id: str
    run_id: str
    ordinal: int
    trial_name: str
    status: str
    config_json: str
    config_hash: str
    collection_name: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    index_build_time_ms: Optional[int] = None
    retrieval_time_ms: Optional[int] = None
    point_count: Optional[int] = None
    cleanup_status: str = CleanupStatus.NOT_CREATED
    error: Optional[str] = None
    attempt_count: int = 0
    # Slice 3（§十八 / §二十三）：Chunking + Embedding Cache 統計。
    # ``chunk_cache_stats_json`` 為版本化 JSON blob（chunk config / strategy fallback /
    # section prefix / context / cache stats / embedding fingerprint）。
    strategy: Optional[str] = None
    chunk_config_hash: Optional[str] = None
    chunk_count: Optional[int] = None
    chunk_cache_stats_json: Optional[str] = None


@dataclass
class TrialMetric:
    trial_id: str
    metric_name: str
    scope: str  # aggregate | query_type
    query_type: Optional[str]
    value: Optional[float]
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    sample_count: Optional[int] = None
    computable: bool = True
    not_computable_reason: Optional[str] = None


@dataclass
class TrialCaseResult:
    trial_id: str
    case_id: str
    case_status: str
    query_type: Optional[str]
    answerable: bool
    latency_ms: Optional[int]
    first_relevant_rank: Optional[int]
    top_results_json: str
    evidence_match_json: str
    error: Optional[str] = None
