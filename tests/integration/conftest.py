"""整合測試共用的 Qdrant 資源隔離工具（Pre-Slice 6 Stability Gate）。

背景：整合測試曾間歇性失敗（完整測試失敗、單獨執行通過）。根因為兩點：

1. **固定 collection 名稱／run id**：中斷或失敗的執行會留下殘留 collection，
   下一次執行沿用同一名稱，狀態被繼承。
2. **全域範圍斷言**：測試以 ``set(before_names) == set(after_names)`` 或
   ``not any(n.startswith("optimizer_"))`` 斷言「整個 Qdrant namespace」，
   因此任何與本測試無關的殘留都會使其失敗；殘留被本測試清掉後，
   重跑就通過——正是觀察到的間歇性症狀。

本模組提供：唯一資源名稱、Session 起始殘留清掃、以及**有限 Timeout Polling**
確認 collection 真正消失（而非只發出刪除命令）。

安全性：所有刪除路徑都以前綴防護，正式 ``factory_knowledge`` 永不被觸及。
"""
from __future__ import annotations

import os
import time
import uuid

import pytest

PROD_COLLECTION = "factory_knowledge"
#: 僅允許清理這些前綴的測試 collection；正式 collection 不在其中。
TEST_COLLECTION_PREFIXES = ("optimizer_", "factory_test_")


def unique_token() -> str:
    """每次呼叫皆不同的短 token，用於組出唯一 collection / run id。"""
    return uuid.uuid4().hex[:8]


def is_test_collection(name: str) -> bool:
    """僅測試前綴且非正式 collection 才視為可清理。"""
    if not name or name == PROD_COLLECTION:
        return False
    return name.startswith(TEST_COLLECTION_PREFIXES)


def list_collections(client) -> list[str]:
    return sorted(x.name for x in client.get_collections().collections)


def prod_point_count(client) -> int | None:
    """正式 collection 的 point 數；不存在時回傳 None。"""
    if PROD_COLLECTION not in list_collections(client):
        return None
    return client.count(PROD_COLLECTION).count


def wait_collection_absent(client, name: str, timeout: float = 15.0) -> None:
    """有限 Timeout Polling 確認 collection 真正消失。

    不以固定 ``sleep()`` 作為同步方式：輪詢實際狀態，超時才失敗，
    並輸出可理解的資源狀態以利診斷。
    """
    deadline = time.monotonic() + timeout
    while True:
        if not client.collection_exists(name):
            return
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"collection {name!r} 於 {timeout}s 內未消失；"
                f"目前 collections={list_collections(client)}"
            )
        time.sleep(0.05)


def drop_collections(client, names, timeout: float = 15.0) -> None:
    """刪除測試 collection 並確認消失；供 ``finally`` 使用。

    只刪除測試前綴 collection，非測試名稱直接跳過（不拋錯，避免遮蔽原始失敗）。
    """
    for name in names:
        if not is_test_collection(name):
            continue
        try:
            if client.collection_exists(name):
                client.delete_collection(name)
            wait_collection_absent(client, name, timeout=timeout)
        except AssertionError:
            raise
        except Exception:  # noqa: BLE001 - cleanup 不得遮蔽測試本身的失敗
            pass


def sweep_stale_test_collections(client) -> list[str]:
    """清掉先前中斷執行留下的測試 collection；回傳被清理的名稱。"""
    stale = [n for n in list_collections(client) if is_test_collection(n)]
    drop_collections(client, stale)
    return stale


def qdrant_client_or_none(url: str):
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url)
        client.get_collections()
        return client
    except Exception:  # noqa: BLE001
        return None


@pytest.fixture(scope="session", autouse=True)
def _sweep_stale_collections_at_session_start():
    """Session 起始清掃殘留測試 collection，避免繼承前次中斷執行的狀態。

    正式 ``factory_knowledge`` 不在清掃範圍內。Qdrant 未就緒時靜默略過
    （個別測試自身會 skip）。
    """
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = qdrant_client_or_none(url)
    if client is None:
        yield
        return
    try:
        sweep_stale_test_collections(client)
        yield
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
