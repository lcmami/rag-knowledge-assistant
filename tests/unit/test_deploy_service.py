"""DeploymentService orchestration 測試（Slice 7 §二十四.5/.6/.7/.8）。

以 fake alias/builder/verifier 驗證 workflow 邏輯（順序 / lock / idempotency / stale /
cleanup / audit / preserve），不需真實 Qdrant（真實 Qdrant 由 integration 測試涵蓋）。
"""
from __future__ import annotations

import pytest

from evaluation.v11.deploy import confirmation as conf
from evaluation.v11.deploy.models import ErrorCode, OperationStatus
from evaluation.v11.deploy.service import (
    DeploymentService,
    InitializeRequest,
    ReconcileRequest,
    RollbackRequest,
)
from tests.unit.deploy_helpers import (
    FakeAliasManager,
    FakeBuilder,
    FakeVerifier,
    eligible_apply_request,
    eligible_env,
    make_repo,
    seed_current_ineligible_search,
    seed_eligible_search,
    service_config,
)

ALIAS = "factory_test_alias_svc"
BASE = "factory_test_deploy_v1_base"


class Setup:
    def __init__(self, tmp_path, *, admin_enabled=True, verifier_ready=True,
                 builder_fail=False, on_build=None):
        self.repo = make_repo(tmp_path)
        self.alias = FakeAliasManager()
        self.builder = FakeBuilder(self.alias, fail=builder_fail, on_build=on_build)
        self.verifier = FakeVerifier(ready=verifier_ready)
        cfg = service_config()
        cfg.admin_enabled = admin_enabled
        self.config = cfg
        self.service = DeploymentService(
            self.repo, self.alias, self.builder, self.verifier, cfg,
            env_provider=lambda: eligible_env(admin_enabled=admin_enabled),
        )

    def initialize(self):
        self.alias.add_collection(BASE)
        return self.service.initialize_alias(InitializeRequest(
            alias=ALIAS, collection=BASE,
            confirmation=conf.initialize_confirmation(ALIAS, BASE),
            idempotency_key="init-1",
        ))

    def seed_and_init(self):
        seed_eligible_search(self.repo)
        r = self.initialize()
        assert r.operation.status == OperationStatus.COMPLETED
        return r

    def apply(self, **over):
        req = eligible_apply_request(alias=ALIAS, current_collection=BASE, **over)
        return self.service.apply(req)

    def audit_types(self, op_id):
        return [e.event_type for e in self.repo.list_audit_events(op_id)]


# ------------------------------------------------------------------
# §十四 initialize
# ------------------------------------------------------------------
def test_initialize_success(tmp_path):
    s = Setup(tmp_path)
    r = s.seed_and_init()
    assert s.alias.get_alias_target(ALIAS) == BASE
    rec = s.repo.get_active_record(ALIAS)
    assert rec.physical_collection == BASE and rec.previous_collection is None
    assert "alias_initialized" in s.audit_types(r.operation.operation_id)


def test_initialize_alias_already_exists(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    r = s.service.initialize_alias(InitializeRequest(
        alias=ALIAS, collection=BASE,
        confirmation=conf.initialize_confirmation(ALIAS, BASE),
        idempotency_key="init-2"))
    assert r.operation.status == OperationStatus.REJECTED
    assert r.operation.error_code == ErrorCode.ALIAS_ALREADY_INITIALIZED


def test_initialize_collection_missing(tmp_path):
    s = Setup(tmp_path)
    seed_eligible_search(s.repo)
    r = s.service.initialize_alias(InitializeRequest(
        alias=ALIAS, collection="nonexistent",
        confirmation=conf.initialize_confirmation(ALIAS, "nonexistent"),
        idempotency_key="init-3"))
    assert r.operation.error_code == ErrorCode.COLLECTION_NOT_FOUND


def test_initialize_confirmation_mismatch(tmp_path):
    s = Setup(tmp_path)
    s.alias.add_collection(BASE)
    r = s.service.initialize_alias(InitializeRequest(
        alias=ALIAS, collection=BASE, confirmation="WRONG", idempotency_key="init-4"))
    assert r.operation.error_code == ErrorCode.CONFIRMATION_MISMATCH


def test_initialize_real_active_alias_blocked(tmp_path):
    """本 Slice 不對真實 active alias 初始化（§六.5 / §二十四.5）。"""
    s = Setup(tmp_path)
    from evaluation.v11.deploy.models import DeploymentError

    s.alias.add_collection("factory_knowledge")
    with pytest.raises(DeploymentError):
        s.service.initialize_alias(InitializeRequest(
            alias=s.config.active_alias, collection="factory_knowledge",
            confirmation=conf.initialize_confirmation(s.config.active_alias, "factory_knowledge"),
            idempotency_key="init-real"))


# ------------------------------------------------------------------
# §十五 apply
# ------------------------------------------------------------------
def test_apply_eligible_success_and_order(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    r = s.apply()
    assert r.operation.status == OperationStatus.COMPLETED
    new = r.record.physical_collection
    assert s.alias.get_alias_target(ALIAS) == new
    # previous collection 保留（§十五.7）。
    assert BASE in s.alias.collections
    # build → readiness → switch 順序（§十五）。
    types = s.audit_types(r.operation.operation_id)
    assert types.index("build_completed") < types.index("readiness_passed") < types.index("alias_switched")
    assert "deployment_activated" in types
    # 新 record active、previous 記錄 deactivated（只有一筆 active）。
    actives = [x for x in s.repo.list_records(ALIAS) if x.active]
    assert len(actives) == 1 and actives[0].physical_collection == new
    assert actives[0].previous_collection == BASE


def test_apply_current_ineligible_rejected_no_collection(tmp_path):
    """目前真實 ineligible recommendation 必須被拒絕，且不建 collection、不切 alias（§九）。"""
    s = Setup(tmp_path)
    # env 與真實 search 完全相符，使**唯一**拒絕原因是 ineligibility 本身（apply_rejected）。
    s.service._env_provider = lambda: {
        "code_commit": "commit_real", "working_tree_dirty": False,
        "target_corpus_hash": "corpus_real", "canonical_text_version": "v1",
        "identifier_normalization_version": "v1", "embedding_fingerprint_hash": None,
    }
    seed_current_ineligible_search(s.repo, search_id="search-real", config_hash="cfg_real_e246697d41")
    s.initialize()
    from evaluation.v11.deploy.eligibility import ApplyRequestSnapshot

    req = ApplyRequestSnapshot(
        candidate_id="cand-real", search_id="search-real", alias_name=ALIAS,
        expected_config_hash="cfg_real_e246697d41", expected_dataset_hash="dataset_real",
        expected_corpus_hash="corpus_real", expected_code_commit="commit_real",
        expected_current_collection=BASE,
        confirmation=conf.apply_confirmation(ALIAS, "cfg_real_e246697d41", BASE),
        idempotency_key="apply-real",
    )
    r = s.service.apply(req)
    assert r.operation.status == OperationStatus.REJECTED
    assert r.operation.error_code == ErrorCode.APPLY_REJECTED
    # 五個真實原因保留。
    for reason in ["small_validation_sample", "quality_metrics_tied",
                   "selection_driven_by_single_run_latency", "test_constraint_failed",
                   "demo_corpus_only"]:
        assert reason in r.decision.reasons
    assert s.builder.build_calls == 0            # 未建 collection
    assert s.alias.get_alias_target(ALIAS) == BASE  # 未切 alias
    assert "eligibility_rejected" in s.audit_types(r.operation.operation_id)


def test_apply_idempotent_retry_no_rebuild(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    r1 = s.apply()
    target = r1.record.physical_collection
    r2 = s.apply()  # 同 idempotency_key + 同 payload
    assert r2.idempotent_replay is True
    assert s.builder.build_calls == 1            # 未重建
    assert s.alias.get_alias_target(ALIAS) == target  # 未重切


def test_apply_concurrent_guard(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    s.repo.acquire_lock(ALIAS, "holder-op")      # 另一操作持鎖
    r = s.apply(idempotency_key="apply-conc")
    assert r.operation.status == OperationStatus.REJECTED
    assert r.operation.error_code == ErrorCode.DEPLOYMENT_IN_PROGRESS
    assert s.builder.build_calls == 0


def test_apply_readiness_failure_cleanup_no_switch(tmp_path):
    s = Setup(tmp_path, verifier_ready=False)
    s.seed_and_init()
    r = s.apply()
    assert r.operation.status == OperationStatus.FAILED
    assert r.operation.error_code == ErrorCode.READINESS_FAILED
    assert s.alias.get_alias_target(ALIAS) == BASE          # 未切 alias
    assert s.builder.cleanup_calls                          # 新建 collection 已清除
    assert "readiness_failed" in s.audit_types(r.operation.operation_id)


def test_apply_build_failure_no_switch(tmp_path):
    s = Setup(tmp_path, builder_fail=True)
    s.seed_and_init()
    r = s.apply()
    assert r.operation.status == OperationStatus.FAILED
    assert r.operation.error_code == ErrorCode.BUILD_FAILED
    assert s.alias.get_alias_target(ALIAS) == BASE
    assert "build_failed" in s.audit_types(r.operation.operation_id)


def test_apply_stale_alias_during_build(tmp_path):
    # build 期間 alias 被切走 → apply 拒絕、不覆蓋、清除新建 collection（§十五.4）。
    s = Setup(tmp_path)
    s.seed_and_init()
    s.builder.on_build = lambda: s.alias.aliases.__setitem__(ALIAS, "hijacked_by_other")
    r = s.apply()
    assert r.operation.status == OperationStatus.REJECTED
    assert r.operation.error_code == ErrorCode.STALE_ALIAS_TARGET
    assert s.alias.get_alias_target(ALIAS) == "hijacked_by_other"  # 未覆蓋別人的切換
    assert s.builder.cleanup_calls


def test_apply_readback_mismatch_needs_reconcile_keeps_collection(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    s.alias.readback_override = "unexpected_readback"
    r = s.apply()
    assert r.operation.status == OperationStatus.NEEDS_RECONCILE
    assert r.operation.reconcile_required is True
    # alias 已切但 read-back 不符 → 不刪 collection（§十五.6）。
    assert s.builder.cleanup_calls == []


def test_apply_does_not_delete_previous_collection(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    r = s.apply()
    assert BASE in s.alias.collections
    assert r.record.physical_collection in s.alias.collections


# ------------------------------------------------------------------
# §十六 rollback
# ------------------------------------------------------------------
def _apply_then(s):
    r = s.apply()
    return r.record.physical_collection


def test_rollback_success_and_preserve(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    v2 = _apply_then(s)
    r = s.service.rollback(RollbackRequest(
        alias=ALIAS, expected_current_collection=v2, target_collection=BASE,
        confirmation=conf.rollback_confirmation(ALIAS, v2, BASE), idempotency_key="rb-1"))
    assert r.operation.status == OperationStatus.COMPLETED
    assert s.alias.get_alias_target(ALIAS) == BASE
    # 兩個 collection 皆保留（§十六.7）。
    assert BASE in s.alias.collections and v2 in s.alias.collections
    assert "rollback_completed" in s.audit_types(r.operation.operation_id)


def test_roll_forward_after_rollback(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    v2 = _apply_then(s)
    s.service.rollback(RollbackRequest(
        alias=ALIAS, expected_current_collection=v2, target_collection=BASE,
        confirmation=conf.rollback_confirmation(ALIAS, v2, BASE), idempotency_key="rb-1"))
    # roll-forward 回 v2（已知 target）。
    r = s.service.rollback(RollbackRequest(
        alias=ALIAS, expected_current_collection=BASE, target_collection=v2,
        confirmation=conf.rollback_confirmation(ALIAS, BASE, v2), idempotency_key="rf-1"))
    assert r.operation.status == OperationStatus.COMPLETED
    assert s.alias.get_alias_target(ALIAS) == v2


def test_rollback_unknown_target_rejected(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    v2 = _apply_then(s)
    r = s.service.rollback(RollbackRequest(
        alias=ALIAS, expected_current_collection=v2, target_collection="never_deployed",
        confirmation=conf.rollback_confirmation(ALIAS, v2, "never_deployed"),
        idempotency_key="rb-x"))
    assert r.operation.error_code == ErrorCode.ROLLBACK_TARGET_NOT_ALLOWED


def test_rollback_optimizer_trial_target_rejected(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    v2 = _apply_then(s)
    # 把一個 optimizer_ collection 塞進部署歷史（known），仍必須被 optimizer 前綴 guard 擋下。
    from evaluation.v11.deploy.models import DeploymentRecord

    # 直接插一筆 record（借用既有 operation_id）。
    existing_op = s.repo.list_operations(ALIAS)[0].operation_id
    s.alias.add_collection("optimizer_trial_x")
    s.repo.insert_record(DeploymentRecord(
        deployment_id="dep-opt", alias_name=ALIAS, physical_collection="optimizer_trial_x",
        active=False, operation_id=existing_op, readiness_json='{"ready": true}',
        activated_at="2026-07-21T02:00:00+00:00"))
    r = s.service.rollback(RollbackRequest(
        alias=ALIAS, expected_current_collection=v2, target_collection="optimizer_trial_x",
        confirmation=conf.rollback_confirmation(ALIAS, v2, "optimizer_trial_x"),
        idempotency_key="rb-opt"))
    assert r.operation.error_code == ErrorCode.ROLLBACK_TARGET_NOT_ALLOWED


def test_rollback_target_missing_collection(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    v2 = _apply_then(s)
    s.alias.collections.discard(BASE)   # target 消失
    r = s.service.rollback(RollbackRequest(
        alias=ALIAS, expected_current_collection=v2, target_collection=BASE,
        confirmation=conf.rollback_confirmation(ALIAS, v2, BASE), idempotency_key="rb-m"))
    assert r.operation.error_code == ErrorCode.COLLECTION_NOT_FOUND


def test_rollback_current_stale_rejected(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    v2 = _apply_then(s)
    r = s.service.rollback(RollbackRequest(
        alias=ALIAS, expected_current_collection="wrong_current", target_collection=BASE,
        confirmation=conf.rollback_confirmation(ALIAS, "wrong_current", BASE),
        idempotency_key="rb-s"))
    assert r.operation.error_code == ErrorCode.STALE_ALIAS_TARGET


def test_rollback_confirmation_mismatch(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    v2 = _apply_then(s)
    r = s.service.rollback(RollbackRequest(
        alias=ALIAS, expected_current_collection=v2, target_collection=BASE,
        confirmation="WRONG", idempotency_key="rb-c"))
    assert r.operation.error_code == ErrorCode.CONFIRMATION_MISMATCH


# ------------------------------------------------------------------
# §十九 reconcile
# ------------------------------------------------------------------
def test_reconcile_consistent(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    r = s.service.reconcile(ReconcileRequest(alias=ALIAS, idempotency_key="rc-1"))
    assert r.operation.status == OperationStatus.COMPLETED
    assert r.reconciliation["resolution"] == "consistent"


def test_reconcile_alias_switched_db_not_updated(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    v2 = _apply_then(s)
    # 模擬 alias 已切到已知 v2，但 DB active 仍為某舊值：手動把 DB active 改回 BASE。
    s.repo.deactivate_all(ALIAS, "2026-07-21T03:00:00+00:00")
    base_rec = [x for x in s.repo.list_records(ALIAS) if x.physical_collection == BASE][0]
    s.repo.set_record_active(base_rec.deployment_id, True, "2026-07-21T03:00:00+00:00")
    assert s.repo.get_active_record(ALIAS).physical_collection == BASE
    assert s.alias.get_alias_target(ALIAS) == v2
    r = s.service.reconcile(ReconcileRequest(alias=ALIAS, idempotency_key="rc-2"))
    assert r.reconciliation["resolution"] == "db_updated_to_alias"
    assert s.repo.get_active_record(ALIAS).physical_collection == v2  # DB 修正為 alias
    assert s.alias.get_alias_target(ALIAS) == v2                       # alias 未被切（§十九.2）
    assert v2 in s.alias.collections and BASE in s.alias.collections  # 未刪 collection


def test_reconcile_stale_lock_released(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    # 建一筆終態 operation 並留下其鎖。
    ops = s.repo.list_operations(ALIAS)
    terminal_op = ops[0]
    s.repo.acquire_lock(ALIAS, terminal_op.operation_id)
    r = s.service.reconcile(ReconcileRequest(alias=ALIAS, idempotency_key="rc-3"))
    assert r.reconciliation["released_stale_lock"] is True
    assert s.repo.get_lock(ALIAS) is None


def test_reconcile_undeterminable_manual(tmp_path):
    s = Setup(tmp_path)
    s.seed_and_init()
    s.alias.aliases[ALIAS] = "mystery_unknown_collection"  # alias 指向未知 collection
    r = s.service.reconcile(ReconcileRequest(alias=ALIAS, idempotency_key="rc-4"))
    assert r.operation.status == OperationStatus.NEEDS_RECONCILE
    assert r.operation.error_code == ErrorCode.MANUAL_RECONCILIATION_REQUIRED
    assert r.reconciliation["resolution"] == "manual_reconciliation_required"
    assert s.alias.get_alias_target(ALIAS) == "mystery_unknown_collection"  # 未切 alias
