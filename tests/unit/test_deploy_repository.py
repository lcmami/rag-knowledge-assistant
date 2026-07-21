"""DeploymentRepository / Migration 測試（Slice 7 §二十四.2 / §十八）。"""
from __future__ import annotations

import sqlite3

import pytest

from evaluation.v11.deploy.models import (
    DeploymentError,
    DeploymentOperation,
    DeploymentRecord,
    ErrorCode,
    OperationStatus,
    OperationType,
)
from evaluation.v11.deploy.repository import DEPLOY_MIGRATIONS, DeploymentRepository


def _op(**over) -> DeploymentOperation:
    base = dict(
        operation_id="op1", idempotency_key="k1", operation_type=OperationType.APPLY,
        status=OperationStatus.PENDING, alias_name="al", request_payload_hash="ph1",
    )
    base.update(over)
    return DeploymentOperation(**base)


def _record(op_id="op1", **over) -> DeploymentRecord:
    base = dict(
        deployment_id="dep1", alias_name="al", physical_collection="coll_v1",
        active=True, operation_id=op_id, activated_at="2026-07-21T00:00:00+00:00",
    )
    base.update(over)
    return DeploymentRecord(**base)


def test_migrations_include_deployment(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    assert repo.applied_migrations() == [v for v, _ in DEPLOY_MIGRATIONS]
    assert repo.applied_migrations()[-1] == "0008_deployment"


def test_migrations_idempotent(tmp_path):
    path = str(tmp_path / "x.deployment.db")
    first = DeploymentRepository(path).applied_migrations()
    second = DeploymentRepository(path).applied_migrations()
    assert first == second == [v for v, _ in DEPLOY_MIGRATIONS]


def test_migration_id_gap_is_intentional_and_not_missing(tmp_path):
    """Global migration numbering policy: 0007 lives in the review namespace.

    The Deployment DB applies 0001..0006 then 0008, skipping 0007. Prove:
      (a) the applied set is exactly {0001,0002,0003,0004,0005,0006,0008},
          i.e. 0007 is intentionally absent (a gap, not a bug);
      (b) re-running the migrations is idempotent and never re-applies or errors;
      (c) the runner does NOT treat the absent 0007 as a "missing migration"
          (no contiguity assumption): a fresh run over the same DB applies
          nothing new and raises nothing.
    """
    from evaluation.v11.optimizer_repository import run_migrations

    path = str(tmp_path / "gap.deployment.db")
    repo = DeploymentRepository(path)
    applied = repo.applied_migrations()

    # (a) exact global-unique, non-contiguous sequence — 0007 deliberately skipped.
    assert applied == [
        "0001_initial",
        "0002_runs_trials",
        "0003_chunk_cache_stats",
        "0004_retrieval_meta",
        "0005_optimizer_search",
        "0006_search_outcome",
        "0008_deployment",
    ]
    assert "0007_review" not in applied
    assert "0007_review_workflow" not in applied

    # (b)+(c) re-run the migration list against the already-migrated DB:
    # nothing new is applied and no error is raised for the 0007 gap.
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        newly = run_migrations(conn, DEPLOY_MIGRATIONS)
    finally:
        conn.close()
    assert newly == []  # gap does not trigger a re-apply nor a "missing" failure


def test_no_documents_table(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    assert "documents" not in repo.table_names()
    assert "deployment_operations" in repo.table_names()


def test_migration_failure_rolls_back(tmp_path):
    from evaluation.v11.optimizer_repository import MigrationError, run_migrations

    conn = sqlite3.connect(str(tmp_path / "bad.db"), isolation_level=None)
    conn.row_factory = sqlite3.Row
    bad = [("0001_ok", ["CREATE TABLE t (id TEXT)"]),
           ("0002_bad", ["CREATE TABLE u (id TEXT)", "THIS IS NOT SQL"])]
    with pytest.raises(MigrationError):
        run_migrations(conn, bad)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "t" in tables and "u" not in tables  # 0002 整批回滾
    conn.close()


def test_operation_crud(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    op, created = repo.create_operation(_op())
    assert created is True
    got = repo.get_operation("op1")
    assert got is not None and got.idempotency_key == "k1"
    got.status = OperationStatus.COMPLETED
    got.target_collection = "coll_v2"
    repo.update_operation(got)
    assert repo.get_operation("op1").status == OperationStatus.COMPLETED
    assert repo.get_operation("op1").target_collection == "coll_v2"
    assert [o.operation_id for o in repo.list_operations("al")] == ["op1"]


def test_idempotency_same_key_same_payload_returns_original(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    op1, created1 = repo.create_operation(_op())
    op2, created2 = repo.create_operation(_op(operation_id="op2"))  # 同 key + 同 payload
    assert created1 is True and created2 is False
    assert op2.operation_id == "op1"  # 回傳原 operation


def test_idempotency_same_key_different_payload_conflict(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    repo.create_operation(_op())
    with pytest.raises(DeploymentError) as exc:
        repo.create_operation(_op(operation_id="op2", request_payload_hash="DIFFERENT"))
    assert exc.value.code == ErrorCode.IDEMPOTENCY_CONFLICT


def test_deployment_record_crud_and_active_query(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    repo.create_operation(_op())
    repo.insert_record(_record())
    active = repo.get_active_record("al")
    assert active is not None and active.physical_collection == "coll_v1"
    assert repo.active_targets() == ("coll_v1",)
    assert repo.known_collections("al") == {"coll_v1"}


def test_switch_active_record_atomic(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    repo.create_operation(_op())
    repo.insert_record(_record())
    new = _record(deployment_id="dep2", physical_collection="coll_v2",
                  previous_collection="coll_v1")
    repo.switch_active_record(new, "2026-07-21T01:00:00+00:00")
    active = repo.get_active_record("al")
    assert active.physical_collection == "coll_v2"
    # 只有一筆 active。
    actives = [r for r in repo.list_records("al") if r.active]
    assert len(actives) == 1
    assert repo.known_collections("al") == {"coll_v1", "coll_v2"}


def test_audit_events_append_and_list(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    repo.create_operation(_op())
    repo.add_audit_event("e1", "op1", "operation_created", actor="local-admin", details={"a": 1})
    repo.add_audit_event("e2", "op1", "alias_switched", actor="local-admin", details={"to": "coll_v2"})
    events = repo.list_audit_events("op1")
    assert [e.event_type for e in events] == ["operation_created", "alias_switched"]
    import json
    assert json.loads(events[0].details_json) == {"a": 1}


def test_lock_acquire_release(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    lock = repo.acquire_lock("al", "op1")
    assert lock.alias_name == "al"
    assert repo.get_lock("al") is not None
    repo.release_lock("al", "op1")
    assert repo.get_lock("al") is None


def test_concurrent_acquire_rejected(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    repo.acquire_lock("al", "op1")
    with pytest.raises(DeploymentError) as exc:
        repo.acquire_lock("al", "op2")
    assert exc.value.code == ErrorCode.DEPLOYMENT_IN_PROGRESS
    # 不同 alias 不衝突。
    repo.acquire_lock("al2", "op3")


def test_steal_stale_lock(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    repo.acquire_lock("al", "old_op")
    assert repo.steal_stale_lock("al", "old_op", "new_op") is True
    assert repo.get_lock("al").operation_id == "new_op"
    # 只有 owner 相符才可接管。
    assert repo.steal_stale_lock("al", "not_owner", "x") is False


def test_json_round_trip(tmp_path):
    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    repo.create_operation(_op(readiness_json='{"ready": true, "failures": []}'))
    import json
    got = repo.get_operation("op1")
    assert json.loads(got.readiness_json) == {"ready": True, "failures": []}


def test_can_read_seeded_search_same_db(tmp_path):
    """同一 DB 內可讀 optimizer_searches（Eligibility Guard 依賴，§二十一）。"""
    from tests.unit.deploy_helpers import seed_eligible_search

    repo = DeploymentRepository(str(tmp_path / "x.deployment.db"))
    search, cand = seed_eligible_search(repo)
    assert repo.get_search(search.search_id) is not None
    assert repo.get_candidate_by_config_hash(search.search_id, cand.config_hash) is not None
