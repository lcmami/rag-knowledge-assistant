"""Search / Candidate 儲存（Slice 5 §十一）。

繼承 :class:`OptimizerRepository`，沿用同一個獨立 Optimizer SQLite、同一套 migration runner
與同樣的連線慣例（``isolation_level=None`` + 每次呼叫自開自關 + ``PRAGMA foreign_keys=ON``）。

**Recommendation Lock 在這一層強制執行**（§12.4 / §13.3 / §25.6）：任何試圖在
``recommendation_locked=0`` 時寫入 ``test_trial_id`` 的操作都會被拒絕。把守衛放在
persistence 層而不是只放在 runner，是為了讓「test 在 lock 前不存在」成為資料庫層級的
不變式 —— 即使未來有別的呼叫端，也繞不過去。
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional

from evaluation.v11.optimizer_repository import OptimizerRepository
from evaluation.v11.search.search_models import (
    OptimizerSearch,
    RecommendationLockError,
    SearchCandidate,
)


class SearchRepository(OptimizerRepository):
    """Optimizer Search 的 CRUD（不依賴 FastAPI）。"""

    # ----------------------------------------------------------------------
    # Search CRUD
    # ----------------------------------------------------------------------
    _SEARCH_COLUMNS = (
        "search_id", "search_name", "status", "search_mode", "search_config_json",
        "search_config_hash", "dataset_version", "dataset_hash", "corpus_hash",
        "code_commit", "working_tree_dirty", "random_seed", "max_candidates",
        "validation_top_n", "baseline_config_hash", "recommended_config_hash",
        "recommendation_locked", "recommendation_locked_at", "test_evaluated",
        "created_at", "started_at", "completed_at", "error",
        "report_json_path", "report_markdown_path",
        "tune_run_id", "validation_run_id", "test_run_id",
        "generation_stats_json", "candidate_manifest_json", "split_manifest_json",
    )

    def upsert_search(self, search: OptimizerSearch) -> None:
        """建立或更新一個 Search（依 search_id upsert；重複不產生重複 Search）。"""
        values = (
            search.search_id, search.search_name, search.status, search.search_mode,
            search.search_config_json, search.search_config_hash,
            search.dataset_version, search.dataset_hash, search.corpus_hash,
            search.code_commit, self._bool_to_int(search.working_tree_dirty),
            search.random_seed, search.max_candidates, search.validation_top_n,
            search.baseline_config_hash, search.recommended_config_hash,
            self._bool_to_int(search.recommendation_locked) or 0,
            search.recommendation_locked_at,
            self._bool_to_int(search.test_evaluated) or 0,
            search.created_at, search.started_at, search.completed_at, search.error,
            search.report_json_path, search.report_markdown_path,
            search.tune_run_id, search.validation_run_id, search.test_run_id,
            search.generation_stats_json, search.candidate_manifest_json,
            search.split_manifest_json,
        )
        placeholders = ",".join("?" for _ in self._SEARCH_COLUMNS)
        updates = ",".join(
            f"{c}=excluded.{c}" for c in self._SEARCH_COLUMNS if c != "search_id"
        )
        conn = self._connect()
        try:
            conn.execute(
                f"INSERT INTO optimizer_searches ({','.join(self._SEARCH_COLUMNS)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT(search_id) DO UPDATE SET {updates}",
                values,
            )
        finally:
            conn.close()

    def get_search(self, search_id: str) -> Optional[OptimizerSearch]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM optimizer_searches WHERE search_id=?", (search_id,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_search(row) if row else None

    def list_searches(self) -> list[OptimizerSearch]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM optimizer_searches ORDER BY created_at, search_id"
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_search(r) for r in rows]

    def delete_search(self, search_id: str) -> None:
        """刪除 Search；FK ON DELETE CASCADE 連帶刪除其 candidates。"""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM optimizer_searches WHERE search_id=?", (search_id,))
        finally:
            conn.close()

    def _row_to_search(self, row: sqlite3.Row) -> OptimizerSearch:
        return OptimizerSearch(
            search_id=row["search_id"],
            search_name=row["search_name"],
            status=row["status"],
            search_mode=row["search_mode"],
            search_config_json=row["search_config_json"],
            search_config_hash=row["search_config_hash"],
            dataset_version=row["dataset_version"],
            dataset_hash=row["dataset_hash"],
            corpus_hash=row["corpus_hash"],
            code_commit=row["code_commit"],
            working_tree_dirty=self._int_to_bool(row["working_tree_dirty"]),
            random_seed=row["random_seed"],
            max_candidates=row["max_candidates"],
            validation_top_n=row["validation_top_n"],
            baseline_config_hash=row["baseline_config_hash"],
            recommended_config_hash=row["recommended_config_hash"],
            recommendation_locked=bool(row["recommendation_locked"]),
            recommendation_locked_at=row["recommendation_locked_at"],
            test_evaluated=bool(row["test_evaluated"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error=row["error"],
            report_json_path=row["report_json_path"],
            report_markdown_path=row["report_markdown_path"],
            tune_run_id=self._row_get(row, "tune_run_id"),
            validation_run_id=self._row_get(row, "validation_run_id"),
            test_run_id=self._row_get(row, "test_run_id"),
            generation_stats_json=self._row_get(row, "generation_stats_json"),
            candidate_manifest_json=self._row_get(row, "candidate_manifest_json"),
            split_manifest_json=self._row_get(row, "split_manifest_json"),
        )

    # ----------------------------------------------------------------------
    # Recommendation Lock
    # ----------------------------------------------------------------------
    def lock_recommendation(
        self, search_id: str, recommended_config_hash: str, locked_at: str
    ) -> OptimizerSearch:
        """鎖定 recommendation（§12.4）。

        - 必須有實際的 ``recommended_config_hash``（不得鎖定 null）。
        - 已鎖定者**不得**改成別的 config hash（§12.5：test 結果不能換人）。
          以同一個 hash 重複鎖定則為 idempotent no-op，方便 resume。
        """
        search = self.get_search(search_id)
        if search is None:
            raise RecommendationLockError(f"找不到 search_id={search_id!r}")
        if not recommended_config_hash:
            raise RecommendationLockError(
                "recommended_config_hash 為空，不得 lock（§18.5 應改記 no_feasible_candidate）。"
            )
        if search.recommendation_locked:
            if search.recommended_config_hash != recommended_config_hash:
                raise RecommendationLockError(
                    f"search {search_id!r} 已鎖定 "
                    f"recommended_config_hash={search.recommended_config_hash!r}，"
                    f"不得改為 {recommended_config_hash!r}（§12.4 / §12.5）。"
                )
            return search

        conn = self._connect()
        try:
            conn.execute(
                "UPDATE optimizer_searches SET recommended_config_hash=?, "
                "recommendation_locked=1, recommendation_locked_at=? WHERE search_id=?",
                (recommended_config_hash, locked_at, search_id),
            )
        finally:
            conn.close()
        locked = self.get_search(search_id)
        assert locked is not None
        return locked

    def is_recommendation_locked(self, search_id: str) -> bool:
        search = self.get_search(search_id)
        return bool(search and search.recommendation_locked)

    # ----------------------------------------------------------------------
    # Candidate CRUD
    # ----------------------------------------------------------------------
    _CANDIDATE_COLUMNS = (
        "search_id", "candidate_id", "ordinal", "config_json", "config_hash",
        "generation_status", "generation_rejection_reason", "duplicate_of_hash",
        "is_baseline", "tune_trial_id", "validation_trial_id", "test_trial_id",
        "tune_status", "validation_status", "test_status", "tune_rank",
        "constraint_results_json", "feasible", "pareto_member", "pareto_rank",
        "recommendation_rank", "rejection_reasons_json",
    )

    def upsert_candidate(self, candidate: SearchCandidate) -> None:
        """建立或更新一個 Candidate（依 candidate_id upsert）。

        §12.4 / §13.3 守衛：``test_trial_id`` 非空時，該 search 必須已 lock。
        """
        if candidate.test_trial_id:
            search = self.get_search(candidate.search_id)
            if search is None:
                raise RecommendationLockError(
                    f"找不到 search_id={candidate.search_id!r}，不得寫入 test_trial_id。"
                )
            search.assert_test_allowed()

        values = (
            candidate.search_id, candidate.candidate_id, candidate.ordinal,
            candidate.config_json, candidate.config_hash, candidate.generation_status,
            candidate.generation_rejection_reason, candidate.duplicate_of_hash,
            self._bool_to_int(candidate.is_baseline) or 0,
            candidate.tune_trial_id, candidate.validation_trial_id,
            candidate.test_trial_id, candidate.tune_status, candidate.validation_status,
            candidate.test_status, candidate.tune_rank,
            candidate.constraint_results_json, self._bool_to_int(candidate.feasible),
            self._bool_to_int(candidate.pareto_member), candidate.pareto_rank,
            candidate.recommendation_rank, candidate.rejection_reasons_json,
        )
        placeholders = ",".join("?" for _ in self._CANDIDATE_COLUMNS)
        updates = ",".join(
            f"{c}=excluded.{c}" for c in self._CANDIDATE_COLUMNS if c != "candidate_id"
        )
        conn = self._connect()
        try:
            conn.execute(
                f"INSERT INTO optimizer_candidates ({','.join(self._CANDIDATE_COLUMNS)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT(candidate_id) DO UPDATE SET {updates}",
                values,
            )
        finally:
            conn.close()

    def save_candidates(self, candidates: list[SearchCandidate]) -> None:
        """整批寫入（單一 transaction，失敗即 rollback）。"""
        for c in candidates:
            if c.test_trial_id:
                search = self.get_search(c.search_id)
                if search is None:
                    raise RecommendationLockError(
                        f"找不到 search_id={c.search_id!r}，不得寫入 test_trial_id。"
                    )
                search.assert_test_allowed()

        placeholders = ",".join("?" for _ in self._CANDIDATE_COLUMNS)
        updates = ",".join(
            f"{c}=excluded.{c}" for c in self._CANDIDATE_COLUMNS if c != "candidate_id"
        )
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            for c in candidates:
                conn.execute(
                    f"INSERT INTO optimizer_candidates "
                    f"({','.join(self._CANDIDATE_COLUMNS)}) VALUES ({placeholders}) "
                    f"ON CONFLICT(candidate_id) DO UPDATE SET {updates}",
                    (
                        c.search_id, c.candidate_id, c.ordinal, c.config_json,
                        c.config_hash, c.generation_status,
                        c.generation_rejection_reason, c.duplicate_of_hash,
                        self._bool_to_int(c.is_baseline) or 0,
                        c.tune_trial_id, c.validation_trial_id, c.test_trial_id,
                        c.tune_status, c.validation_status, c.test_status, c.tune_rank,
                        c.constraint_results_json, self._bool_to_int(c.feasible),
                        self._bool_to_int(c.pareto_member), c.pareto_rank,
                        c.recommendation_rank, c.rejection_reasons_json,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def list_candidates(self, search_id: str) -> list[SearchCandidate]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM optimizer_candidates WHERE search_id=? ORDER BY ordinal",
                (search_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_candidate(r) for r in rows]

    def get_candidate_by_config_hash(
        self, search_id: str, config_hash: str
    ) -> Optional[SearchCandidate]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM optimizer_candidates WHERE search_id=? AND config_hash=?",
                (search_id, config_hash),
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_candidate(row) if row else None

    def _row_to_candidate(self, row: sqlite3.Row) -> SearchCandidate:
        return SearchCandidate(
            search_id=row["search_id"],
            candidate_id=row["candidate_id"],
            ordinal=row["ordinal"],
            config_json=row["config_json"],
            config_hash=row["config_hash"],
            generation_status=row["generation_status"],
            generation_rejection_reason=row["generation_rejection_reason"],
            duplicate_of_hash=row["duplicate_of_hash"],
            is_baseline=bool(row["is_baseline"]),
            tune_trial_id=row["tune_trial_id"],
            validation_trial_id=row["validation_trial_id"],
            test_trial_id=row["test_trial_id"],
            tune_status=row["tune_status"],
            validation_status=row["validation_status"],
            test_status=row["test_status"],
            tune_rank=row["tune_rank"],
            constraint_results_json=row["constraint_results_json"],
            feasible=self._int_to_bool(row["feasible"]),
            pareto_member=self._int_to_bool(row["pareto_member"]),
            pareto_rank=row["pareto_rank"],
            recommendation_rank=row["recommendation_rank"],
            rejection_reasons_json=row["rejection_reasons_json"],
        )

    # ----------------------------------------------------------------------
    # 洩漏防護輔助
    # ----------------------------------------------------------------------
    def count_test_trials(self, search_id: str) -> int:
        """§13.3：lock 前應為 0。供測試與 report 前置檢查使用。"""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM optimizer_candidates "
                "WHERE search_id=? AND test_trial_id IS NOT NULL",
                (search_id,),
            ).fetchone()
        finally:
            conn.close()
        return int(row["n"]) if row else 0
