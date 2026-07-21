"""Read-only Optimizer 資料存取（Slice 8 §六 / §五）。

設計目標（§六）：
- 以真正的 SQLite ``mode=ro`` URI 開啟既有 Search / Review / Deployment DB。
- **永不** migration、**永不** ``CREATE TABLE``、DB 不存在時**不建立**檔案。
- Schema 不相容 → 回 ``optimizer_schema_incompatible``（不當機、不改資料）。
- Search / Review / Deployment DB 路徑**分開**配置；回應**不輸出**實際 DB path。

實作方式：以 ``_ReadOnlyMixin`` 覆寫既有 Repository 的 ``__init__`` / ``_connect`` /
``_migrate``，如此可**完全重用**既有 read 方法（``list_searches`` / ``get_trial_case_results`` /
``list_sessions`` / ``list_operations`` …）與報表輔助函式，而不觸發任何寫入。

去識別化（§五.4/.5）：只回傳白名單欄位；不輸出私人絕對路徑、API key、prompt / LLM
response 全文、embedding vector 或文件全文（只允許短 snippet）。
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Optional

from evaluation.v11.deploy.repository import DeploymentRepository
from evaluation.v11.review.repository import ReviewRepository
from evaluation.v11.search.search_report import per_case_diff
from evaluation.v11.search.search_repository import SearchRepository

# --- unavailable reasons（§五.3 / §六.5）---
REASON_NOT_AVAILABLE = "optimizer_data_not_available"
REASON_SCHEMA_INCOMPATIBLE = "optimizer_schema_incompatible"

#: 名稱含這些片段的 metric 屬「越小越好」（與 outcome.py 一致，避免耦合私有名稱）。
_LOWER_IS_BETTER_HINTS = (
    "latency", "false_positive", "error", "duplicate", "non_computable", "p50", "p95",
)


class OptimizerDataUnavailable(Exception):
    """Read-only DB 不可用（缺檔或 schema 不相容）；``reason`` 為穩定的機器可讀碼。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------
# Read-only repository：覆寫連線 / migration，重用既有 read 方法。
# --------------------------------------------------------------------------
class _ReadOnlyMixin:
    #: 該 DB 必須已套用的 migration 版本（缺一即 schema 不相容）。
    _REQUIRED_VERSIONS: tuple[str, ...] = ()

    def __init__(self, db_path: str):  # noqa: D401 - 覆寫既有 __init__
        # 不建立父目錄、不建立 DB：檔案不存在直接判定 unavailable（§六.4）。
        if not db_path or not os.path.exists(db_path):
            raise OptimizerDataUnavailable(REASON_NOT_AVAILABLE)
        self.db_path = db_path
        self._migrations = None
        self._assert_schema()

    def _connect(self) -> sqlite3.Connection:
        # 真正唯讀：mode=ro 會拒絕任何寫入，且對不存在的檔案直接報錯（我們已先 exists 檢查）。
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro", uri=True, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self) -> None:  # pragma: no cover - read-only 永不 migration
        raise AssertionError("read-only repository 不得執行 migration")

    def _assert_schema(self) -> None:
        try:
            conn = self._connect()
        except sqlite3.OperationalError as exc:  # pragma: no cover - 已先 exists
            raise OptimizerDataUnavailable(REASON_NOT_AVAILABLE) from exc
        try:
            try:
                rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
            except sqlite3.OperationalError as exc:
                # 連 schema_migrations 都沒有 → 不是本系統的 DB。
                raise OptimizerDataUnavailable(REASON_SCHEMA_INCOMPATIBLE) from exc
            applied = {r[0] for r in rows}
            if not set(self._REQUIRED_VERSIONS).issubset(applied):
                raise OptimizerDataUnavailable(REASON_SCHEMA_INCOMPATIBLE)
        finally:
            conn.close()


class ReadOnlySearchRepository(_ReadOnlyMixin, SearchRepository):
    # 0001 建 evaluation_cases（問題文字）；0005/0006 建 search / outcome。
    _REQUIRED_VERSIONS = ("0001_initial", "0005_optimizer_search", "0006_search_outcome")


class ReadOnlyReviewRepository(_ReadOnlyMixin, ReviewRepository):
    _REQUIRED_VERSIONS = ("0007_review_workflow",)


class ReadOnlyDeploymentRepository(_ReadOnlyMixin, DeploymentRepository):
    _REQUIRED_VERSIONS = ("0008_deployment",)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _short(h: Optional[str], n: int = 12) -> Optional[str]:
    if not h:
        return None
    return h[:n]


def _loads(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _dig(obj: Any, key: str) -> Any:
    """在巢狀 dict 內尋找第一個名為 ``key`` 的值（config 結構有 chunk_config / retrieval_config）。"""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _dig(v, key)
            if found is not None:
                return found
    return None


def _metric_direction(name: str) -> str:
    lowered = (name or "").lower()
    return "lower_is_better" if any(h in lowered for h in _LOWER_IS_BETTER_HINTS) else "higher_is_better"


def _availability(configured: bool, available: bool, reason: Optional[str]) -> dict[str, Any]:
    return {"configured": configured, "available": available, "reason": reason}


def _redact_split_manifest(sm: Any) -> Any:
    """移除 split_manifest 內的私人路徑（SplitStat.to_dict 帶有 dataset_path，§五.4）。"""
    if not isinstance(sm, dict):
        return sm
    out: dict[str, Any] = {}
    for k, v in sm.items():
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if kk != "dataset_path"}
        out[k] = v
    return out


class OptimizerReportingStore:
    """三個獨立 read-only DB 的彙總存取層（回傳 UI 就緒、已去識別化的 dict）。

    每個 DB 路徑皆為 optional；未配置 → configured=false；配置但缺檔 →
    available=false / reason=optimizer_data_not_available；schema 不符 →
    reason=optimizer_schema_incompatible。任何情況都**不會建立** DB。
    """

    def __init__(
        self,
        search_db_path: Optional[str] = None,
        review_db_path: Optional[str] = None,
        deployment_db_path: Optional[str] = None,
    ):
        self.search_db_path = search_db_path
        self.review_db_path = review_db_path
        self.deployment_db_path = deployment_db_path

    # -- repo 取得（catch unavailable → 回 (None, availability)）--
    def _open(self, path: Optional[str], cls) -> tuple[Any, dict[str, Any]]:
        if not path:
            return None, _availability(False, False, REASON_NOT_AVAILABLE)
        try:
            repo = cls(path)
        except OptimizerDataUnavailable as exc:
            return None, _availability(True, False, exc.reason)
        return repo, _availability(True, True, None)

    def _search_repo(self):
        return self._open(self.search_db_path, ReadOnlySearchRepository)

    def _review_repo(self):
        return self._open(self.review_db_path, ReadOnlyReviewRepository)

    def _deploy_repo(self):
        return self._open(self.deployment_db_path, ReadOnlyDeploymentRepository)

    # ------------------------------------------------------------------
    # Demo fixture 標記（seeder 寫在 search.generation_stats_json /
    # review session.generator_config_json 的 demo_fixture=true）。
    # ------------------------------------------------------------------
    def is_demo_fixture(self) -> bool:
        repo, av = self._search_repo()
        if repo is not None:
            for s in repo.list_searches():
                if _loads(s.generation_stats_json, {}).get("demo_fixture") is True:
                    return True
        repo, av = self._review_repo()
        if repo is not None:
            for sess in repo.list_sessions():
                if _loads(sess.generator_config_json, {}).get("demo_fixture") is True:
                    return True
        return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _questions(self, repo) -> dict[str, str]:
        """case_id → question（來自 evaluation_cases；無此表則空）。"""
        try:
            conn = repo._connect()
        except Exception:  # noqa: BLE001
            return {}
        try:
            try:
                rows = conn.execute(
                    "SELECT case_id, question FROM evaluation_cases"
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
            return {r["case_id"]: r["question"] for r in rows}
        finally:
            conn.close()

    def _search_summary(self, repo, s) -> dict[str, Any]:
        cands = repo.list_candidates(s.search_id)
        executable = [c for c in cands if c.generation_status == "executable"]
        feasible = [c for c in cands if c.feasible]
        pareto = [c for c in cands if c.pareto_member]
        outcome = _loads(s.outcome_json, {})
        return {
            "search_id": s.search_id,
            "search_name": s.search_name,
            "status": s.status,
            "search_mode": s.search_mode,
            "dataset_version": s.dataset_version,
            "corpus_hash": s.corpus_hash,
            "corpus_hash_short": _short(s.corpus_hash),
            "candidate_count": len(cands),
            "executable_candidate_count": len(executable),
            "feasible_candidate_count": len(feasible),
            "pareto_count": len(pareto),
            "recommended_config_hash": s.recommended_config_hash,
            "recommended_config_hash_short": _short(s.recommended_config_hash),
            "recommendation_locked": bool(s.recommendation_locked),
            "recommendation_locked_at": s.recommendation_locked_at,
            "test_evaluated": bool(s.test_evaluated),
            "apply_eligible": outcome.get("recommendation_apply_eligible"),
            "confidence": outcome.get("recommendation_confidence"),
            "apply_ineligible_reasons": outcome.get("apply_ineligible_reasons", []),
            "test_constraint_status": outcome.get("test_constraint_status"),
            "baseline_comparison_status": outcome.get("baseline_comparison_status"),
            "recommendation_selection_basis": outcome.get("recommendation_selection_basis"),
            "quality_tie_candidate_count": outcome.get("quality_tie_candidate_count"),
            "created_at": s.created_at,
            "completed_at": s.completed_at,
        }

    def list_searches(self) -> dict[str, Any]:
        repo, av = self._search_repo()
        if repo is None:
            return {**av, "searches": []}
        searches = [self._search_summary(repo, s) for s in repo.list_searches()]
        return {**av, "searches": searches}

    def latest_search_summary(self) -> Optional[dict[str, Any]]:
        repo, av = self._search_repo()
        if repo is None:
            return None
        searches = repo.list_searches()
        if not searches:
            return None
        # list_searches 依 created_at DESC（見 search_repository），取第一筆為最新。
        return self._search_summary(repo, searches[0])

    def get_search(self, search_id: str) -> dict[str, Any]:
        """回傳 search detail；找不到 → {'found': False}。"""
        repo, av = self._search_repo()
        if repo is None:
            return {**av, "found": False, "search": None}
        s = repo.get_search(search_id)
        if s is None:
            return {**av, "found": False, "search": None}
        summary = self._search_summary(repo, s)
        outcome = _loads(s.outcome_json, {})
        split_manifest = _redact_split_manifest(_loads(s.split_manifest_json, {}))
        detail = {
            **summary,
            "search_config_hash": s.search_config_hash,
            "random_seed": s.random_seed,
            "max_candidates": s.max_candidates,
            "validation_top_n": s.validation_top_n,
            "baseline_config_hash": s.baseline_config_hash,
            "baseline_config_hash_short": _short(s.baseline_config_hash),
            "code_commit": s.code_commit,
            "working_tree_dirty": s.working_tree_dirty,
            "split_manifest": split_manifest,
            "outcome": outcome,  # 已為指標 / 註解，無私人路徑（§五.4/.5）。
        }
        return {**av, "found": True, "search": detail}

    def _candidate_row(self, c) -> dict[str, Any]:
        config = _loads(c.config_json, {})
        constraints = _loads(c.constraint_results_json, None)
        return {
            "candidate_id": c.candidate_id,
            "ordinal": c.ordinal,
            "config_hash": c.config_hash,
            "config_hash_short": _short(c.config_hash),
            "is_baseline": bool(c.is_baseline),
            "generation_status": c.generation_status,
            "generation_rejection_reason": c.generation_rejection_reason,
            "chunk_strategy": _dig(_dig(config, "chunk_config") or config, "strategy")
            if _dig(config, "chunk_config") else _dig(config, "strategy"),
            "retrieval_strategy": _dig(_dig(config, "retrieval_config") or {}, "strategy"),
            "identifier_vector_floor": _dig(config, "identifier_vector_floor"),
            "score_threshold": _dig(config, "score_threshold"),
            "exact_match_boost": _dig(config, "exact_match_boost"),
            "confusable_penalty": _dig(config, "confusable_penalty"),
            "tune_status": c.tune_status,
            "validation_status": c.validation_status,
            "test_status": c.test_status,
            "constraint_status": "pass" if c.feasible else ("fail" if c.feasible is False else None),
            "constraint_results": constraints,
            "feasible": c.feasible,
            "pareto_member": c.pareto_member,
            "pareto_rank": c.pareto_rank,
            "recommendation_rank": c.recommendation_rank,
            "tune_rank": c.tune_rank,
        }

    def list_candidates(self, search_id: str) -> dict[str, Any]:
        repo, av = self._search_repo()
        if repo is None:
            return {**av, "found": False, "candidates": []}
        s = repo.get_search(search_id)
        if s is None:
            return {**av, "found": False, "candidates": []}
        rows = [self._candidate_row(c) for c in repo.list_candidates(search_id)]
        return {**av, "found": True, "candidates": rows}

    def _metric_key(self, m) -> tuple[str, str, Optional[str]]:
        return (m.metric_name, m.scope, m.query_type)

    def comparison(self, search_id: str) -> dict[str, Any]:
        """Baseline vs Recommendation：test-split 指標比較 + per-case Top-5 diff（§八 / §九）。"""
        repo, av = self._search_repo()
        if repo is None:
            return {**av, "found": False, "comparison": None}
        s = repo.get_search(search_id)
        if s is None:
            return {**av, "found": False, "comparison": None}

        cands = repo.list_candidates(search_id)
        baseline = next(
            (c for c in cands if c.is_baseline or c.config_hash == s.baseline_config_hash),
            None,
        )
        recommended = next(
            (c for c in cands if c.config_hash == s.recommended_config_hash and s.recommended_config_hash),
            None,
        )
        b_trial = baseline.test_trial_id if baseline else None
        r_trial = recommended.test_trial_id if recommended else None

        # -- metrics comparison（涵蓋兩個 test trial 出現的所有 metric）--
        b_metrics = {self._metric_key(m): m for m in repo.get_trial_metrics(b_trial)} if b_trial else {}
        r_metrics = {self._metric_key(m): m for m in repo.get_trial_metrics(r_trial)} if r_trial else {}
        rows: list[dict[str, Any]] = []
        for key in sorted(set(b_metrics) | set(r_metrics), key=lambda k: (k[0], k[1], k[2] or "")):
            name, scope, qt = key
            bm = b_metrics.get(key)
            rm = r_metrics.get(key)
            b_val = bm.value if bm and bm.computable else None
            r_val = rm.value if rm and rm.computable else None
            delta = (r_val - b_val) if (b_val is not None and r_val is not None) else None
            rows.append({
                "metric_name": name,
                "scope": scope,
                "query_type": qt,
                "baseline_value": b_val,
                "candidate_value": r_val,
                "delta": delta,
                "direction": _metric_direction(name),
                "sample_count": (rm.sample_count if rm else None) or (bm.sample_count if bm else None),
                "baseline_computable": bool(bm.computable) if bm else False,
                "candidate_computable": bool(rm.computable) if rm else False,
            })

        # -- per-case Top-5 diff（重用既有 helper，並補上題目文字）--
        diff = per_case_diff(repo, b_trial, r_trial, label="test")
        questions = self._questions(repo)
        for row in diff["cases"]:
            if row.get("question") is None:
                row["question"] = questions.get(row["case_id"])

        return {
            **av,
            "found": True,
            "comparison": {
                "search_id": search_id,
                "baseline_config_hash": s.baseline_config_hash,
                "baseline_config_hash_short": _short(s.baseline_config_hash),
                "recommended_config_hash": s.recommended_config_hash,
                "recommended_config_hash_short": _short(s.recommended_config_hash),
                "baseline_comparison_status": _loads(s.outcome_json, {}).get("baseline_comparison_status"),
                "metrics": rows,
                "per_case": diff,
            },
        }

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------
    def _session_summary(self, repo, sess) -> dict[str, Any]:
        cands = repo.list_candidates(sess.session_id)
        counts = {"proposed": 0, "accepted": 0, "rejected": 0, "published": 0}
        for c in cands:
            counts[c.status] = counts.get(c.status, 0) + 1
        return {
            "session_id": sess.session_id,
            "session_name": sess.session_name,
            "status": sess.status,
            "base_dataset_version": sess.base_dataset_version,
            "base_dataset_hash": sess.base_dataset_hash,
            "base_dataset_hash_short": _short(sess.base_dataset_hash),
            "corpus_hash_short": _short(sess.corpus_hash),
            "generator_type": sess.generator_type,
            "candidate_count": len(cands),
            "counts": counts,
            "revision_count": repo.count_revisions(sess.session_id),
            "published_dataset_version": sess.published_dataset_version,
            "created_at": sess.created_at,
            "completed_at": sess.completed_at,
        }

    def list_reviews(self) -> dict[str, Any]:
        repo, av = self._review_repo()
        if repo is None:
            return {**av, "sessions": []}
        sessions = [self._session_summary(repo, s) for s in repo.list_sessions()]
        return {**av, "sessions": sessions}

    def get_review(self, session_id: str) -> dict[str, Any]:
        repo, av = self._review_repo()
        if repo is None:
            return {**av, "found": False, "session": None}
        sess = repo.get_session(session_id)
        if sess is None:
            return {**av, "found": False, "session": None}
        summary = self._session_summary(repo, sess)
        candidates = []
        for c in repo.list_candidates(session_id):
            evidence = _loads(c.suggested_anchor_text_json, [])
            candidates.append({
                "candidate_id": c.candidate_id,
                "question": c.proposed_question,
                "query_type": c.proposed_query_type,
                "answerable": bool(c.proposed_answerable),
                "status": c.status,
                "suggested_group_id": c.suggested_group_id,
                "suggested_split": c.suggested_split,
                "unanswerable_confirmed": bool(c.unanswerable_confirmed),
                "evidence_count": len(evidence) if isinstance(evidence, list) else 0,
                "current_revision": c.current_revision,
                "revision_count": len(repo.list_revisions(c.candidate_id)),
            })
        events = [
            {"event_type": e.event_type, "candidate_id": e.candidate_id, "created_at": e.created_at}
            for e in repo.list_events(session_id)
        ]
        return {
            **av,
            "found": True,
            "session": {**summary, "candidates": candidates, "audit_events": events},
        }

    def coverage(self) -> dict[str, Any]:
        """Dataset Coverage / Gap（§五 coverage）。

        優先用最新 search 的 split_manifest（tune/validation/test 覆蓋）＋ outcome 揭露；
        review DB 若有 session 則附上 review coverage gap 提示。
        """
        repo, av = self._search_repo()
        if repo is None:
            return {**av, "coverage": None}
        searches = repo.list_searches()
        if not searches:
            return {**av, "coverage": None, "reason": REASON_NOT_AVAILABLE}
        s = searches[0]
        split_manifest = _redact_split_manifest(_loads(s.split_manifest_json, {}))
        outcome = _loads(s.outcome_json, {})
        warnings: list[str] = []
        # split coverage 缺 query type → 警告。
        for split in ("tune", "validation", "test"):
            stat = split_manifest.get(split) or {}
            for missing in stat.get("missing_expected_query_types", []) or []:
                warnings.append(f"{split}:missing_query_type:{missing}")
        # outcome 揭露的 identifier-unanswerable / 小樣本 gap（若 seeder 有寫入）。
        limitation = outcome.get("apply_ineligible_reasons", [])
        return {
            **av,
            "coverage": {
                "dataset_version": s.dataset_version,
                "corpus_hash_short": _short(s.corpus_hash),
                "split_manifest": split_manifest,
                "coverage_warnings": warnings,
                "apply_ineligible_reasons": limitation,
            },
        }

    # ------------------------------------------------------------------
    # Deployment（read-only 摘要；write / live-alias 仍走 Slice 7 admin API）
    # ------------------------------------------------------------------
    def deployment_summary(self, active_alias: Optional[str]) -> dict[str, Any]:
        repo, av = self._deploy_repo()
        if repo is None:
            return {**av, "active_alias": active_alias, "alias_initialized": False,
                    "history_count": 0, "operations": [], "records": []}
        records = repo.list_records(active_alias) if active_alias else []
        active = next((r for r in records if r.active), None)
        ops = repo.list_operations(active_alias) if active_alias else repo.list_operations()
        return {
            **av,
            "active_alias": active_alias,
            "alias_initialized": active is not None,
            "current_collection": active.physical_collection if active else None,
            "previous_collection": active.previous_collection if active else None,
            "history_count": len(records),
            "records": [r.to_public_dict() for r in records],
            "operations": [o.to_public_dict() for o in ops],
        }

    # ------------------------------------------------------------------
    # Overview（§七）
    # ------------------------------------------------------------------
    def overview(self, *, active_alias: Optional[str], admin_write_enabled: bool) -> dict[str, Any]:
        search_av = self.list_searches()
        latest = self.latest_search_summary()
        cov = self.coverage()
        reviews = self.list_reviews()
        deploy = self.deployment_summary(active_alias)

        split_counts = {}
        cov_block = cov.get("coverage") or {}
        sm = cov_block.get("split_manifest") or {}
        case_count = 0
        group_count = 0
        for split in ("tune", "validation", "test"):
            stat = sm.get(split) or {}
            n = stat.get("case_count", len(stat.get("case_ids", []) or []))
            split_counts[split] = n
            case_count += n or 0
            group_count += stat.get("group_count", 0) or 0

        return {
            "is_demo_fixture": self.is_demo_fixture(),
            "search": {
                "configured": search_av["configured"],
                "available": search_av["available"],
                "reason": search_av["reason"],
                "dataset_version": cov_block.get("dataset_version"),
                "corpus_hash_short": cov_block.get("corpus_hash_short"),
                "case_count": case_count,
                "group_count": group_count,
                "split_counts": split_counts,
                "coverage_warnings": cov_block.get("coverage_warnings", []),
                "latest_search": latest,
            },
            "review": {
                "configured": reviews["configured"],
                "available": reviews["available"],
                "reason": reviews["reason"],
                "session_count": len(reviews.get("sessions", [])),
                "sessions": reviews.get("sessions", []),
            },
            "deployment": {
                "configured": deploy["configured"],
                "available": deploy["available"],
                "reason": deploy["reason"],
                "admin_write_enabled": admin_write_enabled,
                "active_alias": active_alias,
                "alias_initialized": deploy.get("alias_initialized", False),
                "history_count": deploy.get("history_count", 0),
            },
        }
