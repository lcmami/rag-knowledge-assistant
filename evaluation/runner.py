"""Baseline Evaluation Runner（v1.1 Slice 0）—— CLI 與 import 共用的單一核心。

流程：固定 seed → 建隔離服務（隔離 SQLite / Upload / Qdrant collection）
→ 匯入 corpus → 逐題真實檢索 → 以 :mod:`evaluation.metrics` 計算指標
→ 組 :class:`~evaluation.schemas.EvaluationReport`。

安全與隔離（Slice 0 §三）：
- 只操作 config.collection 指定的隔離 collection，永不觸碰 factory_knowledge。
- 使用暫存 SQLite 與暫存 Upload 目錄，結束即清除。
- 不呼叫任何 LLM API。
- 評估結束（含失敗）一律清掉 collection 與暫存目錄，不留孤兒。

本 Slice 只做 Baseline 所需最小抽象，不建立完整 v1.1 TrialRunner／SQLite Trial Schema。
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional

from evaluation import metrics
from evaluation.schemas import (
    EvaluationCase,
    EvaluationConfig,
    EvaluationError,
    EvaluationReport,
    load_cases,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 保護欄：正式 collection 名稱，runner 絕不接受以此為評估 collection。
PROTECTED_COLLECTION = "factory_knowledge"

SUPPORTED_EXTENSIONS = (".md", ".markdown", ".txt", ".pdf")


# --------------------------------------------------------------------------
# 可重現性 / 環境資訊
# --------------------------------------------------------------------------
def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    try:  # numpy 為選用；BGE encode 本身在 eval 模式下即為 deterministic
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def _git_commit() -> dict[str, Any]:
    def _run(args: list[str]) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:  # noqa: BLE001
            return None
        return None

    commit = _run(["rev-parse", "HEAD"])
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _collect_corpus_files(corpus_dirs: tuple[str, ...]) -> list[tuple[str, str]]:
    """回傳 [(basename, absolute_path)]，依 basename 排序；重複 basename 視為錯誤。"""
    collected: dict[str, str] = {}
    for d in corpus_dirs:
        if not os.path.isdir(d):
            raise EvaluationError(f"corpus 目錄不存在：{d}")
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(SUPPORTED_EXTENSIONS):
                if name in collected:
                    raise EvaluationError(f"corpus 出現重複檔名：{name}")
                collected[name] = os.path.join(d, name)
    if not collected:
        raise EvaluationError(f"corpus 目錄無任何支援檔案：{list(corpus_dirs)}")
    return sorted(collected.items())


def _corpus_manifest(files: list[tuple[str, str]]) -> dict[str, Any]:
    """建立 corpus 文件清單 + 每檔 hash + 整體 corpus hash。"""
    entries = []
    hasher = hashlib.sha256()
    for name, path in files:
        with open(path, "rb") as f:
            raw = f.read()
        file_hash = _sha256_bytes(raw)
        entries.append({"filename": name, "sha256": file_hash, "bytes": len(raw)})
        hasher.update(name.encode("utf-8"))
        hasher.update(file_hash.encode("utf-8"))
    return {
        "documents": entries,
        "document_count": len(entries),
        "corpus_hash": hasher.hexdigest(),
    }


def _dataset_manifest(cases_path: str, cases: list[EvaluationCase]) -> dict[str, Any]:
    import json

    normalized = json.dumps(
        [c.to_dict() for c in cases], ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return {
        "cases_file": os.path.basename(cases_path),
        "case_count": len(cases),
        "dataset_hash": _sha256_bytes(normalized),
    }


# --------------------------------------------------------------------------
# 服務建立 / corpus 匯入（抽自原 evaluate_retrieval.py，行為等價）
# --------------------------------------------------------------------------
def build_services(config: EvaluationConfig, workdir: str, embedding=None):
    """建立隔離的評估服務堆疊。

    ``embedding`` 可注入以避免測試載入真實 BGE；預設建立真實 EmbeddingService。
    """
    from app.core.config import Settings
    from app.dependencies import create_qdrant_client
    from app.repositories.document_repository import DocumentRepository
    from app.services.indexing_service import IndexingService
    from app.services.retrieval_service import RetrievalService
    from app.services.vector_store import VectorStore

    settings = Settings(
        app_env="evaluation",
        qdrant_url=config.qdrant_url,
        qdrant_collection=config.collection,
        sqlite_path=os.path.join(workdir, "eval_metadata.db"),
        upload_dir=os.path.join(workdir, "uploads"),
        retrieval_score_threshold=config.score_threshold,
        retrieval_max_top_k=max(config.top_k, 10),
    )

    if embedding is None:
        from app.services.embedding_service import EmbeddingService

        embedding = EmbeddingService(
            model_name=settings.embedding_model,
            query_instruction=settings.embedding_query_instruction,
            max_tokens=settings.embedding_max_tokens,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            dimension=settings.embedding_dimension,
        )

    client = create_qdrant_client(config.qdrant_url)
    store = VectorStore(client, config.collection)
    repo = DocumentRepository(settings.sqlite_path)
    indexing = IndexingService(settings, repo, embedding, store)
    retrieval = RetrievalService(
        embedding,
        store,
        score_threshold=config.score_threshold,
        candidate_multiplier=config.candidate_multiplier,
        candidate_cap=config.candidate_cap,
    )
    return settings, embedding, store, repo, indexing, retrieval, client


def ingest_corpus(store, embedding, indexing, files: list[tuple[str, str]]) -> dict[str, int]:
    """乾淨的評估 collection：先刪舊 collection 再重建，匯入 corpus 全部文件。"""
    try:
        store.client.delete_collection(store.collection)
    except Exception:  # noqa: BLE001 - 不存在即可忽略
        pass
    store.ensure_collection(embedding.dimension)

    ingested: dict[str, int] = {}
    for name, path in files:
        with open(path, "rb") as f:
            raw = f.read()
        record = indexing.index_document(name, raw)
        ingested[name] = record.chunk_count
    return ingested


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def run_evaluation(config: EvaluationConfig, embedding=None) -> EvaluationReport:
    """執行一次完整 Baseline 評估並回傳 :class:`EvaluationReport`。

    失敗時清掉隔離 collection 與暫存目錄後再拋出，不留半成品／孤兒。
    """
    if config.collection == PROTECTED_COLLECTION:
        raise EvaluationError(
            f"拒絕在正式 collection「{PROTECTED_COLLECTION}」上執行評估。"
        )

    _seed_everything(config.seed)

    cases = load_cases(config.cases_path)
    files = _collect_corpus_files(config.corpus_dirs)

    workdir = tempfile.mkdtemp(prefix="factory_eval_")
    client = None
    collection_created = False
    try:
        (settings, embedding, store, repo, indexing, retrieval, client) = build_services(
            config, workdir, embedding=embedding
        )

        collection_created = True
        ingested = ingest_corpus(store, embedding, indexing, files)

        case_results = []
        for case in cases:
            result = retrieval.retrieve(case.question, config.top_k)
            ranked = [
                {
                    "document": r.filename,
                    "section": r.section,
                    "page": r.page,
                    "score": r.score,
                }
                for r in result.results
            ]
            case_results.append(
                metrics.build_case_result(
                    case, ranked, result.retrieval_time_ms, top_k=config.top_k
                )
            )

        agg = metrics.aggregate_metrics(case_results, config.report_threshold)
        sweep = [metrics.threshold_report(case_results, t) for t in config.thresholds]
        configured = metrics.threshold_report(case_results, config.report_threshold)

        corpus = _corpus_manifest(files)
        corpus["ingested_chunk_counts"] = ingested
        corpus["total_chunks"] = sum(ingested.values())

        report = EvaluationReport(
            config=config.sanitized(),
            environment={
                "embedding_model": settings.embedding_model,
                "vector_size": embedding.dimension,
                "distance": "Cosine",
                "chunk_size_chars": settings.chunk_size_chars,
                "chunk_overlap_chars": settings.chunk_overlap_chars,
                "embedding_max_tokens": settings.embedding_max_tokens,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "git": _git_commit(),
            },
            corpus=corpus,
            dataset=_dataset_manifest(config.cases_path, cases),
            metrics=agg,
            threshold_at_configured=configured,
            threshold_sweep=sweep,
            cases=case_results,
            notes=[
                "Case 標註為 Slice 0 Baseline 格式（expected_document / expected_section / "
                "expected_page / should_answer）；Slice 1 才遷移至 Canonical Evidence Span。",
                "Evidence Coverage、Graded NDCG、Evidence-level Precision@5、Canonical "
                "Evidence overlap 於 Slice 0 尚無可靠標註，未計算（留給 Slice 1／2）。",
                "分數與排名為本次真實 BGE 執行結果，僅供 Baseline 記錄，不作為固定期望值斷言。",
            ],
        )
        return report
    except Exception as exc:  # noqa: BLE001
        raise EvaluationError(f"評估失敗：{exc}") from exc
    finally:
        # 清理順序：collection -> client -> workdir，確保不留孤兒。
        if client is not None:
            if collection_created and not config.keep_collection:
                try:
                    client.delete_collection(config.collection)
                except Exception:  # noqa: BLE001
                    pass
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(workdir, ignore_errors=True)
