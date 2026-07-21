"""Deployment Readiness Verification（Slice 7 §十三）。

在 Alias 切換**之前**驗證新建 collection 完整且可搜尋。Probe 為 deterministic，**不呼叫 LLM**
（§十三.14），且只驗證「index 可搜尋 / 資料完整 / 預期 point 可被找回」，不重選 candidate（§十三）。

Probe 失敗 → ``ready=False``；service 依此**不切 alias**（§十三.15）。
"""
from __future__ import annotations

from typing import Any, Optional

from evaluation.v11.deploy.models import DeploymentBuildManifest, ReadinessReport

#: 每個 point 必須具備的 payload 欄位（§十三.6）。
REQUIRED_PAYLOAD_FIELDS = ("document_id", "file_hash", "text", "chunk_index")
#: 必須存在的 keyword payload index（§十三.6/.7）。
REQUIRED_INDEXES = ("document_id", "file_hash")
IDENTIFIER_INDEX = "identifiers"


class DeploymentReadinessVerifier:
    """驗證 versioned collection 是否 ready（可入 deployment record）。"""

    def __init__(self, client, embedding):
        self._client = client
        self._embedding = embedding

    def verify(
        self,
        manifest: DeploymentBuildManifest,
        *,
        expected_corpus_hash: str,
        expected_config_hash: str,
        expected_canonical_text_version: str,
        expected_identifier_normalization_version: str,
        expected_embedding_fingerprint_hash: str,
        probe_top_k: int = 5,
    ) -> ReadinessReport:
        collection = manifest.collection_name
        checks: list[dict[str, Any]] = []

        def check(name: str, passed: bool, detail: Any = None) -> bool:
            checks.append({"name": name, "passed": bool(passed), "detail": detail})
            return bool(passed)

        # 1. Collection 存在。
        if not check("collection_exists", self._client.collection_exists(collection)):
            return self._report(collection, checks, probe={})

        info = self._client.get_collection(collection)
        vector_size = info.config.params.vectors.size
        distance = str(info.config.params.vectors.distance)

        # 2/3. Vector size / distance。
        check("vector_size", vector_size == manifest.vector_size,
              {"expected": manifest.vector_size, "actual": vector_size})
        check("distance_metric", distance == manifest.distance,
              {"expected": manifest.distance, "actual": distance})

        # 4. Point count == build manifest。
        point_count = self._client.count(collection).count
        check("point_count", point_count == manifest.point_count,
              {"expected": manifest.point_count, "actual": point_count})

        # 掃描全部 points（payload），供 5/6 與 probe 使用。
        points = self._scroll_all(collection)

        # 5. Chunk identity 無重複（(document_id, chunk_index)）。
        identities = [
            (p.get("document_id"), p.get("chunk_index")) for p in points
        ]
        check("no_duplicate_identity", len(identities) == len(set(identities)),
              {"points": len(identities), "distinct": len(set(identities))})

        # 6. Required payload fields。
        missing_fields = set()
        for p in points:
            for f in REQUIRED_PAYLOAD_FIELDS:
                if f not in p:
                    missing_fields.add(f)
        check("required_payload_fields", not missing_fields,
              {"missing": sorted(missing_fields)})

        # 7. Required keyword indexes（identifier-aware 需 identifiers index）。
        schema = getattr(info, "payload_schema", None) or {}
        index_fields = set(schema.keys())
        needed = list(REQUIRED_INDEXES)
        if manifest.identifier_index:
            needed.append(IDENTIFIER_INDEX)
        missing_idx = [f for f in needed if f not in index_fields]
        check("required_payload_indexes", not missing_idx,
              {"missing": missing_idx, "present": sorted(index_fields)})

        # 8–12. Hash / version 一致（來自 build manifest；manifest 由 candidate 驅動）。
        check("corpus_hash", manifest.corpus_hash == expected_corpus_hash,
              {"expected": expected_corpus_hash, "actual": manifest.corpus_hash})
        check("config_hash", manifest.config_hash == expected_config_hash,
              {"expected": expected_config_hash, "actual": manifest.config_hash})
        check("canonical_text_version",
              manifest.canonical_text_version == expected_canonical_text_version)
        check("identifier_normalization_version",
              manifest.identifier_normalization_version
              == expected_identifier_normalization_version)
        check("embedding_fingerprint",
              manifest.embedding_fingerprint_hash == expected_embedding_fingerprint_hash)

        # 13–15. Deterministic retrieval probe（無 LLM）。
        probe = self._probe(collection, points, probe_top_k)
        check("retrieval_probe", probe.get("passed", False), probe)

        return self._report(collection, checks, probe=probe)

    # ------------------------------------------------------------------
    def _scroll_all(self, collection: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        next_offset = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=collection,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                out.append(dict(p.payload or {}))
            if not next_offset:
                break
        return out

    def _probe(
        self, collection: str, points: list[dict[str, Any]], top_k: int
    ) -> dict[str, Any]:
        """embed 一個已索引 chunk 的文字為 query，驗證其可被找回（deterministic，無 LLM）。"""
        if not points:
            return {"passed": False, "reason": "empty_index"}
        # 以 (document_id, chunk_index) 排序取第一個，確保 deterministic。
        target = sorted(
            points, key=lambda p: (str(p.get("document_id")), int(p.get("chunk_index") or 0))
        )[0]
        text = target.get("text") or ""
        if not text:
            return {"passed": False, "reason": "probe_point_missing_text"}
        try:
            qv = self._embedding.embed_query(text)
            results = self._client.query_points(
                collection_name=collection, query=qv, limit=top_k, with_payload=True
            ).points
        except Exception as exc:  # noqa: BLE001
            return {"passed": False, "reason": f"probe_search_failed: {exc}"}
        want = (target.get("document_id"), target.get("chunk_index"))
        found = any(
            (r.payload or {}).get("document_id") == want[0]
            and (r.payload or {}).get("chunk_index") == want[1]
            for r in results
        )
        return {
            "passed": found,
            "probe_identity": {"document_id": want[0], "chunk_index": want[1]},
            "result_count": len(results),
            "reason": None if found else "probe_point_not_retrieved",
        }

    @staticmethod
    def _report(
        collection: str, checks: list[dict[str, Any]], *, probe: dict[str, Any]
    ) -> ReadinessReport:
        failures = [c["name"] for c in checks if not c["passed"]]
        return ReadinessReport(
            ready=len(failures) == 0,
            collection_name=collection,
            checks=checks,
            failures=failures,
            probe=probe,
        )
