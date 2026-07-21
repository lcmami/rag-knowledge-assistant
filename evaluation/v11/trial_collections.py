"""Trial Collection 命名與防誤刪 Cleanup（Slice 2 §九、§十一）。

- Trial 專屬 collection 命名：``optimizer_<run>_<trial>``，合法化並避免過長。
- **絕不**接受正式 collection ``factory_knowledge``。
- Cleanup 只能刪除 ``optimizer_`` 前綴 collection；傳入錯誤 ID 也不得刪到正式資料。
"""
from __future__ import annotations

import re

OPTIMIZER_PREFIX = "optimizer_"
PROTECTED_COLLECTION = "factory_knowledge"

_TOKEN_MAX = 20
_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


class UnsafeCollectionError(Exception):
    """嘗試刪除 / 使用非 ``optimizer_`` 前綴（或正式）collection。"""


def _sanitize_token(token: str) -> str:
    t = _SANITIZE_RE.sub("", (token or "").lower())
    if not t:
        t = "x"
    return t[:_TOKEN_MAX]


def trial_collection_name(run_id: str, trial_id: str) -> str:
    """由 run_id / trial_id 產生合法、長度受控的 Trial collection 名稱。"""
    name = f"{OPTIMIZER_PREFIX}{_sanitize_token(run_id)}_{_sanitize_token(trial_id)}"
    if name == PROTECTED_COLLECTION or not name.startswith(OPTIMIZER_PREFIX):
        # 理論上不會發生（前綴固定）；防禦性保險。
        raise UnsafeCollectionError(f"產生的 collection 名稱不安全：{name!r}")
    return name


def is_optimizer_collection(name: str) -> bool:
    return bool(name) and name.startswith(OPTIMIZER_PREFIX) and name != PROTECTED_COLLECTION


def assert_deletable(name: str) -> None:
    """僅允許刪除 ``optimizer_`` 前綴 collection；否則拋 :class:`UnsafeCollectionError`。"""
    if not is_optimizer_collection(name):
        raise UnsafeCollectionError(
            f"拒絕刪除非 optimizer_ 前綴 collection：{name!r}（保護 {PROTECTED_COLLECTION}）。"
        )


def safe_delete_collection(client, name: str) -> None:
    """防誤刪的 collection 刪除：先驗證前綴，再呼叫 client.delete_collection。"""
    assert_deletable(name)
    client.delete_collection(name)


def cleanup_run_collections(client, collection_names: list[str]) -> dict[str, str]:
    """刪除一組 Trial collection（跳過非 optimizer_ 前綴者）。回傳每個名稱的結果。"""
    results: dict[str, str] = {}
    for name in collection_names:
        if not is_optimizer_collection(name):
            results[name] = "skipped_not_optimizer"
            continue
        try:
            if client.collection_exists(name):
                client.delete_collection(name)
                results[name] = "cleaned"
            else:
                results[name] = "absent"
        except Exception as exc:  # noqa: BLE001
            results[name] = f"cleanup_failed: {exc}"
    return results
