"""共用 fake 件（Slice 2 Trial Runner 單元測試用）。

不載入 BGE、不連 Qdrant。以記憶體集合模擬 collection 生命週期，以腳本化 map 模擬檢索。
"""
from __future__ import annotations

from typing import Callable, Optional

from evaluation.v11.retrieval_metrics import IndexChunk, RankedResult
from evaluation.v11.trial_collections import assert_deletable
from evaluation.v11.trial_runner import IndexOutcome, RetrievalOutcome


def ranked(rank, *, doc="doc-a", sha="sha-a", section="S", page=None, chunk_index=0,
           score=0.9, text="t", provenance=None, identifiers=None):
    return RankedResult(
        rank=rank, document_id=doc, document_sha256=sha, filename=f"{doc}.md",
        section=section, page=page, chunk_index=chunk_index, score=score, text=text,
        provenance_spans=provenance or [], identifiers=identifiers or [],
        canonical_text_version="v1",
    )


class FakeTrialServiceProvider:
    """腳本化 provider：collection 生命週期用集合模擬；檢索用 (trial_name, question) map。"""

    def __init__(
        self,
        *,
        point_count: int = 10,
        results_for: Optional[Callable[[str, str, str], list[RankedResult]]] = None,
        universe_for: Optional[Callable[[str, str], list[IndexChunk]]] = None,
        fail_build_trials: Optional[set[str]] = None,
        fail_cleanup_collections: Optional[set[str]] = None,
    ):
        self.collections: set[str] = set()
        self.deleted: list[str] = []
        self.build_calls: list[str] = []
        self.manifest_calls: list[str] = []
        self.point_count = point_count
        self._results_for = results_for
        self._universe_for = universe_for
        self._fail_build = fail_build_trials or set()
        self._fail_cleanup = fail_cleanup_collections or set()

    def build_index(self, collection_name, trial_config, files) -> IndexOutcome:
        self.build_calls.append(trial_config.trial_name)
        if trial_config.trial_name in self._fail_build:
            raise RuntimeError(f"注入的 build 失敗：{trial_config.trial_name}")
        self.collections.add(collection_name)
        return IndexOutcome(point_count=self.point_count, build_time_ms=5)

    def index_manifest(self, collection_name) -> list[IndexChunk]:
        self.manifest_calls.append(collection_name)
        if self._universe_for is not None:
            return self._universe_for(collection_name, collection_name)
        return []

    def retrieve(self, collection_name, trial_config, question, top_k) -> RetrievalOutcome:
        if self._results_for is not None:
            res = self._results_for(trial_config.trial_name, collection_name, question)
        else:
            res = []
        return RetrievalOutcome(results=res[:top_k], retrieval_time_ms=3)

    def collection_exists(self, collection_name) -> bool:
        return collection_name in self.collections

    def delete_collection(self, collection_name) -> None:
        assert_deletable(collection_name)
        if collection_name in self._fail_cleanup:
            raise RuntimeError(f"注入的 cleanup 失敗：{collection_name}")
        self.collections.discard(collection_name)
        self.deleted.append(collection_name)

    def close(self) -> None:  # CLI 會呼叫；fake 為 no-op
        pass
