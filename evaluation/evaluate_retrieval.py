"""Retrieval Evaluation（規格 §29 / Phase 4 §8、§9）。

直接呼叫 Retrieval Service（不經 LLM），在獨立的 evaluation collection 上量測：
- Hit@1 / Hit@3（以 expected_document 判定），另記 section hit 與 page hit。
- 平均 / P50 / P95 retrieval latency。
- 可回答問題 Top-1 score 分布、無關問題最高 score 分布。
- 多個候選 threshold 下的接受 / 拒答 / 誤拒 / 誤接受統計，供 threshold 校準。

實際數字只能來自本次執行結果，不得預先捏造。

使用：
    python evaluation/evaluate_retrieval.py
              [--qdrant-url http://localhost:6333]
              [--collection factory_evaluation]
              [--top-k 10]
              [--keep-collection]      # 保留 evaluation collection（預設執行後刪除）

注意：預設使用獨立 collection「factory_evaluation」，不會觸碰正式 factory_knowledge。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# 允許以 `python evaluation/evaluate_retrieval.py` 直接執行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import Settings  # noqa: E402
from app.dependencies import create_qdrant_client  # noqa: E402
from app.repositories.document_repository import DocumentRepository  # noqa: E402
from app.services.embedding_service import EmbeddingService  # noqa: E402
from app.services.indexing_service import IndexingService  # noqa: E402
from app.services.retrieval_service import RetrievalService  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "sample_docs")
QUESTIONS_PATH = os.path.join(ROOT, "evaluation", "questions.json")
RESULTS_PATH = os.path.join(ROOT, "evaluation", "results.json")

CANDIDATE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# --- 小工具（避免額外依賴，純 Python 統計）---
def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "p25": round(_percentile(values, 0.25), 4),
        "median": round(_percentile(values, 0.50), 4),
        "mean": round(sum(values) / len(values), 4),
        "p75": round(_percentile(values, 0.75), 4),
        "max": round(max(values), 4),
    }


def _load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _sample_files() -> list[str]:
    files = []
    for name in sorted(os.listdir(SAMPLE_DIR)):
        if name.lower().endswith((".md", ".markdown", ".txt", ".pdf")):
            files.append(name)
    return files


def build_services(qdrant_url: str, collection: str, top_k_cap: int):
    workdir = tempfile.mkdtemp(prefix="factory_eval_")
    settings = Settings(
        app_env="evaluation",
        qdrant_url=qdrant_url,
        qdrant_collection=collection,
        sqlite_path=os.path.join(workdir, "eval_metadata.db"),
        upload_dir=os.path.join(workdir, "uploads"),
        retrieval_score_threshold=0.0,
        retrieval_max_top_k=max(10, top_k_cap),
    )
    embedding = EmbeddingService(
        model_name=settings.embedding_model,
        query_instruction=settings.embedding_query_instruction,
        max_tokens=settings.embedding_max_tokens,
    )
    client = create_qdrant_client(qdrant_url)
    store = VectorStore(client, collection)
    repo = DocumentRepository(settings.sqlite_path)
    indexing = IndexingService(settings, repo, embedding, store)
    # threshold 0.0 收集原始分數，後續在記憶體中套用候選 threshold
    retrieval = RetrievalService(embedding, store, score_threshold=0.0)
    return settings, embedding, store, repo, indexing, retrieval, client, workdir


def ingest_samples(store: VectorStore, embedding: EmbeddingService, indexing: IndexingService) -> dict:
    # 乾淨的 evaluation collection：先刪除舊 collection 再重建
    try:
        store.client.delete_collection(store.collection)
    except Exception:  # noqa: BLE001 - 不存在即可忽略
        pass
    store.ensure_collection(embedding.dimension)

    ingested = {}
    for name in _sample_files():
        path = os.path.join(SAMPLE_DIR, name)
        with open(path, "rb") as f:
            raw = f.read()
        record = indexing.index_document(name, raw)
        ingested[name] = record.chunk_count
    return ingested


def evaluate(retrieval: RetrievalService, questions: list[dict], top_k: int) -> list[dict]:
    rows = []
    for q in questions:
        result = retrieval.retrieve(q["question"], top_k)
        results = result.results
        top_docs = [r.filename for r in results]
        top1 = results[0] if results else None

        expected_doc = q.get("expected_document")
        expected_section = q.get("expected_section")
        expected_page = q.get("expected_page")

        doc_hit_1 = bool(top1 and top1.filename == expected_doc) if expected_doc else None
        doc_hit_3 = (expected_doc in top_docs[:3]) if expected_doc else None
        section_hit_1 = (
            bool(top1 and top1.filename == expected_doc and top1.section == expected_section)
            if (expected_doc and expected_section)
            else None
        )
        page_hit_1 = (
            bool(top1 and top1.filename == expected_doc and top1.page == expected_page)
            if (expected_doc and expected_page is not None)
            else None
        )

        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "should_answer": q["should_answer"],
                "expected_document": expected_doc,
                "expected_section": expected_section,
                "expected_page": expected_page,
                "top1_score": round(top1.score, 4) if top1 else None,
                "top1_document": top1.filename if top1 else None,
                "top1_section": top1.section if top1 else None,
                "top1_page": top1.page if top1 else None,
                "top3_documents": top_docs[:3],
                "doc_hit_at_1": doc_hit_1,
                "doc_hit_at_3": doc_hit_3,
                "section_hit_at_1": section_hit_1,
                "page_hit_at_1": page_hit_1,
                "retrieval_time_ms": result.retrieval_time_ms,
            }
        )
    return rows


def _rate(numer: int, denom: int) -> float:
    return round(numer / denom, 4) if denom else 0.0


def threshold_report(rows: list[dict], threshold: float) -> dict:
    answerable = [r for r in rows if r["should_answer"]]
    unanswerable = [r for r in rows if not r["should_answer"]]

    def accepted(r) -> bool:
        return r["top1_score"] is not None and r["top1_score"] >= threshold

    ans_accepted = [r for r in answerable if accepted(r)]
    ans_rejected = [r for r in answerable if not accepted(r)]
    un_accepted = [r for r in unanswerable if accepted(r)]
    un_rejected = [r for r in unanswerable if not accepted(r)]

    # precision：所有被接受的問題中，屬於可回答且 top-1 命中正確文件的比例
    accepted_all = ans_accepted + un_accepted
    true_accept = [r for r in ans_accepted if r["doc_hit_at_1"]]
    precision = _rate(len(true_accept), len(accepted_all))

    return {
        "threshold": threshold,
        "answerable_accept_rate": _rate(len(ans_accepted), len(answerable)),
        "answerable_false_reject_rate": _rate(len(ans_rejected), len(answerable)),
        "unanswerable_refuse_rate": _rate(len(un_rejected), len(unanswerable)),
        "unanswerable_false_accept_rate": _rate(len(un_accepted), len(unanswerable)),
        "precision": precision,
        "false_accepted_unanswerable_ids": [r["id"] for r in un_accepted],
        "false_rejected_answerable_ids": [r["id"] for r in ans_rejected],
    }


def _hit_rate(rows: list[dict], key: str) -> tuple[float, int, int]:
    considered = [r for r in rows if r[key] is not None]
    hits = [r for r in considered if r[key]]
    return _rate(len(hits), len(considered)), len(hits), len(considered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--collection", default="factory_evaluation")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--keep-collection", action="store_true")
    args = parser.parse_args()

    questions = _load_questions()

    (settings, embedding, store, repo, indexing, retrieval, client, workdir) = build_services(
        args.qdrant_url, args.collection, args.top_k
    )

    try:
        print(f"[1/3] 匯入 sample documents 至 collection「{args.collection}」…")
        ingested = ingest_samples(store, embedding, indexing)
        for name, cc in ingested.items():
            print(f"      - {name}: {cc} chunks")

        print(f"[2/3] 執行 {len(questions)} 題檢索評估（top_k={args.top_k}, threshold=0.0）…")
        rows = evaluate(retrieval, questions, args.top_k)

        answerable = [r for r in rows if r["should_answer"]]
        unanswerable = [r for r in rows if not r["should_answer"]]

        hit1, hit1_n, hit1_d = _hit_rate(rows, "doc_hit_at_1")
        hit3, hit3_n, hit3_d = _hit_rate(rows, "doc_hit_at_3")
        sec1, sec1_n, sec1_d = _hit_rate(rows, "section_hit_at_1")
        page1, page1_n, page1_d = _hit_rate(rows, "page_hit_at_1")

        latencies = [r["retrieval_time_ms"] for r in rows]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        p50 = round(_percentile([float(x) for x in latencies], 0.50), 2)
        p95 = round(_percentile([float(x) for x in latencies], 0.95), 2)

        ans_scores = [r["top1_score"] for r in answerable if r["top1_score"] is not None]
        un_scores = [r["top1_score"] for r in unanswerable if r["top1_score"] is not None]

        sweep = [threshold_report(rows, t) for t in CANDIDATE_THRESHOLDS]
        configured_report = threshold_report(rows, settings.retrieval_score_threshold)

        results = {
            "embedding_model": settings.embedding_model,
            "vector_size": embedding.dimension,
            "distance": "Cosine",
            "chunk_size_chars": settings.chunk_size_chars,
            "chunk_overlap_chars": settings.chunk_overlap_chars,
            "embedding_max_tokens": settings.embedding_max_tokens,
            "collection": args.collection,
            "top_k": args.top_k,
            "threshold": settings.retrieval_score_threshold,
            "total_questions": len(rows),
            "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "section_hit_at_1": sec1,
            "page_hit_at_1": page1,
            "section_hit_considered": sec1_d,
            "page_hit_considered": page1_d,
            "average_latency_ms": avg_latency,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "answerable_score_summary": _summary(ans_scores),
            "unanswerable_score_summary": _summary(un_scores),
            "threshold_at_configured": configured_report,
            "threshold_sweep": sweep,
            "ingested_chunk_counts": ingested,
            "questions": rows,
        }

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # --- 終端摘要 ---
        print(f"[3/3] 評估完成，結果已寫入 {RESULTS_PATH}\n")
        print("=== 摘要 ===")
        print(f"題目總數={len(rows)}  可回答={len(answerable)}  無關={len(unanswerable)}")
        print(f"Hit@1={hit1} ({hit1_n}/{hit1_d})   Hit@3={hit3} ({hit3_n}/{hit3_d})")
        print(f"Section Hit@1={sec1} ({sec1_n}/{sec1_d})   Page Hit@1={page1} ({page1_n}/{page1_d})")
        print(f"Latency  avg={avg_latency}ms  p50={p50}ms  p95={p95}ms")
        print(f"可回答 Top-1 score 分布: {_summary(ans_scores)}")
        print(f"無關   最高 score 分布: {_summary(un_scores)}")
        print("\n=== Threshold 掃描 ===")
        header = f"{'thr':>5} {'ans_accept':>10} {'ans_falsereject':>15} {'un_refuse':>10} {'un_falseaccept':>14} {'precision':>9}"
        print(header)
        for rep in sweep:
            print(
                f"{rep['threshold']:>5.2f} {rep['answerable_accept_rate']:>10.2f} "
                f"{rep['answerable_false_reject_rate']:>15.2f} {rep['unanswerable_refuse_rate']:>10.2f} "
                f"{rep['unanswerable_false_accept_rate']:>14.2f} {rep['precision']:>9.2f}"
            )
    finally:
        if not args.keep_collection:
            try:
                store.client.delete_collection(args.collection)
                print(f"\n已刪除 evaluation collection「{args.collection}」。")
            except Exception as exc:  # noqa: BLE001
                print(f"\n刪除 evaluation collection 失敗：{exc}", file=sys.stderr)
        else:
            print(f"\n已保留 evaluation collection「{args.collection}」。")
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
