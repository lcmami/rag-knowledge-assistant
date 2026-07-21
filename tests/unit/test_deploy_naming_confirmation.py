"""Naming 防護 + Confirmation 格式測試（Slice 7 §四.10 / §七 / §十四 / §十五.3 / §十六.6）。"""
from __future__ import annotations

import pytest

from evaluation.v11.deploy import confirmation as conf
from evaluation.v11.deploy import naming
from evaluation.v11.deploy.models import DeploymentError, ErrorCode


def test_versioned_name_deterministic_and_legal():
    a = naming.build_versioned_collection_name("cfghash123456789", "token123456789")
    b = naming.build_versioned_collection_name("cfghash123456789", "token123456789")
    assert a == b
    assert a.startswith("factory_knowledge_v")
    assert not naming.is_protected(a)
    # 不同 payload token → 不同名稱。
    c = naming.build_versioned_collection_name("cfghash123456789", "OTHERTOKEN")
    assert c != a


def test_protected_collection_hardcoded():
    assert naming.is_protected("factory_knowledge")
    with pytest.raises(DeploymentError) as e:
        naming.assert_not_protected("factory_knowledge")
    assert e.value.code == ErrorCode.UNSAFE_COLLECTION


def test_assert_safe_to_delete_blocks_protected_and_active():
    with pytest.raises(DeploymentError):
        naming.assert_safe_to_delete("factory_knowledge")
    with pytest.raises(DeploymentError):
        naming.assert_safe_to_delete("coll_v2", active_targets=("coll_v2",))
    naming.assert_safe_to_delete("coll_v9", active_targets=("coll_v2",))  # ok


def test_alias_target_rejects_optimizer_trial():
    with pytest.raises(DeploymentError) as e:
        naming.assert_valid_alias_target("optimizer_run_trial")
    assert e.value.code == ErrorCode.ROLLBACK_TARGET_NOT_ALLOWED
    naming.assert_valid_alias_target("factory_knowledge")  # 正式 collection 可為 alias target


def test_safe_delete_never_touches_protected():
    class FakeClient:
        def __init__(self):
            self.deleted = []
        def collection_exists(self, n):
            return True
        def delete_collection(self, n):
            self.deleted.append(n)

    client = FakeClient()
    with pytest.raises(DeploymentError):
        naming.safe_delete_collection(client, "factory_knowledge")
    assert client.deleted == []
    assert naming.safe_delete_collection(client, "factory_test_deploy_v1_x") is True
    assert client.deleted == ["factory_test_deploy_v1_x"]


def test_confirmation_formats_and_match():
    assert conf.initialize_confirmation("al", "coll") == "INITIALIZE al -> coll"
    assert conf.apply_confirmation("al", "hash", "cur") == "APPLY al CONFIG hash FROM cur"
    assert conf.rollback_confirmation("al", "cur", "tgt") == "ROLLBACK al FROM cur TO tgt"
    assert conf.confirmation_matches(" INITIALIZE al -> coll ", "INITIALIZE al -> coll")
    assert not conf.confirmation_matches("INITIALIZE al -> other", "INITIALIZE al -> coll")


def test_confirmation_hash_stable_and_no_plaintext():
    h1 = conf.confirmation_hash("APPLY al CONFIG hash FROM cur")
    h2 = conf.confirmation_hash(" APPLY al CONFIG hash FROM cur ")
    assert h1 == h2 and len(h1) == 64
    assert "APPLY" not in h1
