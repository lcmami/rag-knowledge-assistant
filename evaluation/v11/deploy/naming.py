"""Versioned physical collection 命名 + 正式 collection 硬編碼防護（Slice 7 §四.10 / §七 / §十二）。

Logical Alias 與 Physical Collection **明確分離**：

- Logical Alias（可設定）：``factory_knowledge_active``
- Physical Collections：``factory_knowledge``、``factory_knowledge_v<config>_<token>``
- Integration Test：``factory_test_alias_<uuid>``、``factory_test_deploy_v<...>``

硬編碼防護（§四.10）：正式 ``factory_knowledge`` **永遠**不得被本 Slice 的任何路徑刪除，
即使被錯誤地當成 build target / rollback target / cleanup target。
"""
from __future__ import annotations

import re

from evaluation.v11.deploy.models import DeploymentError, ErrorCode

#: 硬編碼、任何情況都不得刪除的正式 collection（§四.3 / §四.10）。
PROTECTED_COLLECTIONS: tuple[str, ...] = ("factory_knowledge",)

#: 正式概念預設值（§六）。App 未初始化前不得自動建立。
DEFAULT_ACTIVE_ALIAS = "factory_knowledge_active"
DEFAULT_VERSIONED_PREFIX = "factory_knowledge_v"

#: optimizer trial collection 前綴；alias / rollback target **不得**指向這些（§七.6 / §十六.3）。
OPTIMIZER_TRIAL_PREFIX = "optimizer_"

_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")
_HASH_SHORT = 12
_TOKEN_SHORT = 12


def _sanitize(token: str) -> str:
    t = _SANITIZE_RE.sub("", (token or "").lower())
    return t or "x"


def is_protected(name: str) -> bool:
    return name in PROTECTED_COLLECTIONS


def is_optimizer_trial(name: str) -> bool:
    return bool(name) and name.startswith(OPTIMIZER_TRIAL_PREFIX)


def build_versioned_collection_name(
    config_hash: str,
    payload_token: str,
    *,
    prefix: str = DEFAULT_VERSIONED_PREFIX,
) -> str:
    """deterministic、合法、含 config-hash-short + per-payload token 的 collection 名稱（§十二.1/.2）。

    同一 (config_hash, payload_token) → 同一名稱（idempotent resume）；不同 request payload →
    不同 token → 不同名稱（避免不同語意共用實體 collection）。名稱**絕不**等於受保護正式 collection。
    """
    name = (
        f"{_sanitize(prefix)}"
        f"{_sanitize(config_hash)[:_HASH_SHORT]}"
        f"_{_sanitize(payload_token)[:_TOKEN_SHORT]}"
    )
    if is_protected(name):
        # 理論上不可能（prefix 固定含 _v）；防禦性保險。
        raise DeploymentError(
            ErrorCode.UNSAFE_COLLECTION,
            "產生的 versioned collection 名稱與受保護正式 collection 相同，已拒絕。",
        )
    return name


def assert_not_protected(name: str) -> None:
    """任何刪除 / 覆寫 / build target 前呼叫：受保護正式 collection 一律拒絕。"""
    if is_protected(name):
        raise DeploymentError(
            ErrorCode.UNSAFE_COLLECTION,
            f"拒絕操作受保護的正式 collection {name!r}（硬編碼防護）。",
        )


def assert_valid_alias_target(name: str) -> None:
    """Alias 不得指向 optimizer trial collection（§七.6）。"""
    if not name:
        raise DeploymentError(ErrorCode.INVALID_REQUEST, "alias target 不得為空。")
    if is_optimizer_trial(name):
        raise DeploymentError(
            ErrorCode.ROLLBACK_TARGET_NOT_ALLOWED,
            f"alias 不得指向 optimizer trial collection {name!r}（§七.6）。",
        )


def assert_safe_to_delete(name: str, *, active_targets: tuple[str, ...] = ()) -> None:
    """Cleanup 防護：受保護正式 collection、以及目前 active target 一律不得刪除（§十二.11 / §十五.11）。"""
    assert_not_protected(name)
    if name in active_targets:
        raise DeploymentError(
            ErrorCode.UNSAFE_COLLECTION,
            f"拒絕刪除目前 active 的 collection {name!r}。",
        )


def safe_delete_collection(client, name: str, *, active_targets: tuple[str, ...] = ()) -> bool:
    """防誤刪的刪除：先驗證受保護 / active，再刪。回傳是否實際刪除。"""
    assert_safe_to_delete(name, active_targets=active_targets)
    try:
        if client.collection_exists(name):
            client.delete_collection(name)
            return True
    except Exception:  # noqa: BLE001 - cleanup 不得遮蔽原始錯誤
        return False
    return False
