"""Qdrant Logical Alias 的 atomic 操作（Slice 7 §七 / §十四 / §十五）。

Logical Alias 與 Physical Collection 明確分離：Apply / Rollback **不 rename collection**，
一律以單一 atomic ``update_collection_aliases`` 切換 alias target（delete+create 同一次呼叫）。

安全：alias 不得指向 ``optimizer_`` trial collection（§七.6）；切換前 target collection 必須存在。
本模組**不刪除任何 collection**（§十九.7）。
"""
from __future__ import annotations

from typing import Optional

from evaluation.v11.deploy.models import DeploymentError, ErrorCode
from evaluation.v11.deploy.naming import assert_valid_alias_target


class AliasManager:
    """封裝 Qdrant alias API；只做 alias 操作，不碰 collection 生命週期。"""

    def __init__(self, client):
        self.client = client

    # ------------------------------------------------------------------
    def get_alias_target(self, alias_name: str) -> Optional[str]:
        """讀取 alias 目前指向的 physical collection；alias 不存在回 None。"""
        try:
            resp = self.client.get_aliases()
        except Exception as exc:  # noqa: BLE001
            raise DeploymentError(
                ErrorCode.INVALID_REQUEST, f"無法讀取 Qdrant aliases：{exc}"
            ) from exc
        for desc in getattr(resp, "aliases", []) or []:
            if getattr(desc, "alias_name", None) == alias_name:
                return getattr(desc, "collection_name", None)
        return None

    def alias_exists(self, alias_name: str) -> bool:
        return self.get_alias_target(alias_name) is not None

    def _collection_exists(self, name: str) -> bool:
        return bool(self.client.collection_exists(name))

    # ------------------------------------------------------------------
    def create_alias(self, alias_name: str, collection_name: str) -> None:
        """First-time atomic create（§十四）。collection 必須存在；不覆寫既有 alias。"""
        from qdrant_client import models

        assert_valid_alias_target(collection_name)
        if self.alias_exists(alias_name):
            raise DeploymentError(
                ErrorCode.ALIAS_ALREADY_INITIALIZED,
                f"alias {alias_name!r} 已存在，拒絕重複初始化（§十四.1）。",
            )
        if not self._collection_exists(collection_name):
            raise DeploymentError(
                ErrorCode.COLLECTION_NOT_FOUND,
                f"target collection {collection_name!r} 不存在，拒絕建立 alias（§十四.2）。",
            )
        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=collection_name, alias_name=alias_name
                    )
                )
            ]
        )

    def switch_alias(
        self, alias_name: str, new_collection: str, *, expected_current: Optional[str]
    ) -> str:
        """Atomic 切換 alias→new_collection（§十五 delete+create 同一次呼叫）。

        - ``expected_current`` 非 None 時，先確認 alias 目前確實指向它（TOCTOU 防護，§十五.4）；
          不符即 :class:`DeploymentError`(``stale_alias_target``)，**不覆蓋別人的操作**。
        - new collection 必須存在且非 optimizer trial。
        回傳切換後 read-back 的 target。
        """
        from qdrant_client import models

        assert_valid_alias_target(new_collection)
        current = self.get_alias_target(alias_name)
        if current is None:
            raise DeploymentError(
                ErrorCode.ALIAS_NOT_INITIALIZED,
                f"alias {alias_name!r} 尚未初始化，無法切換（§十五.1）。",
            )
        if expected_current is not None and current != expected_current:
            raise DeploymentError(
                ErrorCode.STALE_ALIAS_TARGET,
                f"alias {alias_name!r} 目前指向 {current!r}，與 expected {expected_current!r} 不符（§十五.4）。",
            )
        if not self._collection_exists(new_collection):
            raise DeploymentError(
                ErrorCode.COLLECTION_NOT_FOUND,
                f"target collection {new_collection!r} 不存在，拒絕切換 alias。",
            )
        self.client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias_name)
                ),
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=new_collection, alias_name=alias_name
                    )
                ),
            ]
        )
        # §十五.5：Read-back。
        return self.get_alias_target(alias_name)
