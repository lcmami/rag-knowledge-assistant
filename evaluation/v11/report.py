"""Run Report 產生（Slice 2 §十五）。

由 Optimizer Repository（單一真相）讀出 Run / Trial / Metrics / Case Results，
輸出 ``<output_dir>/<run_id>/report.json`` 與 ``report.md``。

報告**不得**包含：.env、API key、私人絕對路徑、Docker secrets、文件原文全文。
可包含：相對 filename、section、page、score、provenance 摘要、identifier、chunk 摘要。
"""
from __future__ import annotations

import json
import os
from typing import Any

from evaluation.v11.metric_config import DEFAULT_METRIC_CONFIG, MetricConfig
from evaluation.v11.optimizer_repository import OptimizerRepository
from evaluation.v11.run_models import EvaluationRun, RunConfig

KNOWN_LIMITATIONS = [
    "Slice 2 只量測現有固定 Chunking + Dense Retrieval；未實作 chunking 搜尋空間、"
    "identifier candidate union、exact-code boost、hybrid/BM25、reranker。",
    "不自動選擇 Best Candidate、不自動修改任何正式設定。",
    "評估樣本小（約 12 case 級）；小樣本指標為低可信度，附樣本數。",
    "Graded NDCG / Evidence Coverage 僅在 gold evidence 具 offset 時計算；"
    "anchor-only 標為不可計算，不以 0 假裝。",
    "Slice 3 只快取 Document Chunk Embedding；Query Embedding 不快取（避免污染 latency）。"
    "Embedding Cache 為獨立 SQLite（*.embedding-cache.db），本 Slice 不做 LRU/TTL/容量管理，"
    "跨 Run 預設使用臨時 cache（run 結束清除）。",
    # §一：current strategy 的可重現性語意（Slice 3 收尾要求）。
    "current strategy means the repository-defined reproducible v1.0 default behavior. "
    "It does not automatically mirror environment-specific chunking overrides. "
    "每個 Trial 保存完整展開的 chunk_config（含 target_chars/overlap_chars/max_tokens），"
    "config hash 涵蓋展開後的實際值，chunk_config_origin=explicit_trial_config；Optimizer 不讀取 "
    ".env / Settings 覆寫值。若要評估 Production Active Config，須由使用者明確匯出為 Trial Config。",
]


def _ndcg_of(evidence_match) -> dict[str, Any] | None:
    """evidence_match 現為 {"matches": [...], "ndcg": {...}}；相容舊 list 格式回 None。"""
    if isinstance(evidence_match, dict):
        return evidence_match.get("ndcg")
    return None


def _ndcg_summary(case_results) -> dict[str, Any]:
    computable = 0
    non_computable = 0
    reasons: dict[str, int] = {}
    for c in case_results:
        em = json.loads(c.evidence_match_json)
        ndcg = _ndcg_of(em)
        if not ndcg:
            continue
        if ndcg.get("computable"):
            computable += 1
        else:
            non_computable += 1
            r = ndcg.get("not_computable_reason") or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
    return {"computable_cases": computable, "non_computable_cases": non_computable,
            "not_computable_reasons": reasons}


def _metric_to_dict(m) -> dict[str, Any]:
    return {
        "metric_name": m.metric_name,
        "scope": m.scope,
        "query_type": m.query_type,
        "value": m.value,
        "numerator": m.numerator,
        "denominator": m.denominator,
        "sample_count": m.sample_count,
        "computable": m.computable,
        "not_computable_reason": m.not_computable_reason,
    }


def build_report(
    repo: OptimizerRepository,
    run: EvaluationRun,
    run_config: RunConfig,
    corpus_manifest: dict[str, Any],
    *,
    remaining_collections: list[str] | None = None,
    metric_config: MetricConfig = DEFAULT_METRIC_CONFIG,
    extra_errors: list[str] | None = None,
) -> dict[str, Any]:
    trials = repo.list_trials(run.run_id)
    trial_blocks: list[dict[str, Any]] = []
    not_computable: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = list({"scope": "run", "error": e} for e in (extra_errors or []))

    for t in trials:
        metrics = repo.get_trial_metrics(t.trial_id)
        case_results = repo.get_trial_case_results(t.trial_id)
        for m in metrics:
            if not m.computable:
                not_computable.append(
                    {
                        "trial_name": t.trial_name,
                        "metric_name": m.metric_name,
                        "scope": m.scope,
                        "query_type": m.query_type,
                        "reason": m.not_computable_reason,
                    }
                )
        if t.error:
            errors.append({"scope": f"trial:{t.trial_name}", "error": t.error})

        # Slice 3（§二十三）：Chunking + Embedding Cache 統計（版本化 JSON blob）。
        chunk_cache_stats = (
            json.loads(t.chunk_cache_stats_json) if t.chunk_cache_stats_json else None
        )
        trial_blocks.append(
            {
                "trial_id": t.trial_id,
                "trial_name": t.trial_name,
                "ordinal": t.ordinal,
                "status": t.status,
                "config_hash": t.config_hash,
                "config": json.loads(t.config_json) if t.config_json else None,
                "collection_name": t.collection_name,
                "cleanup_status": t.cleanup_status,
                "point_count": t.point_count,
                "index_build_time_ms": t.index_build_time_ms,
                "retrieval_time_ms": t.retrieval_time_ms,
                "attempt_count": t.attempt_count,
                "error": t.error,
                "strategy": t.strategy,
                "chunk_count": t.chunk_count,
                "chunk_config_hash": t.chunk_config_hash,
                "chunk_cache_stats": chunk_cache_stats,
                "aggregate_metrics": [_metric_to_dict(m) for m in metrics if m.scope == "aggregate"],
                "per_query_type_metrics": [
                    _metric_to_dict(m) for m in metrics if m.scope == "query_type"
                ],
                "cases": [
                    {
                        "case_id": c.case_id,
                        "case_status": c.case_status,
                        "query_type": c.query_type,
                        "answerable": c.answerable,
                        "latency_ms": c.latency_ms,
                        "first_relevant_rank": c.first_relevant_rank,
                        "top_results": json.loads(c.top_results_json),
                        "evidence_match": json.loads(c.evidence_match_json),
                        "error": c.error,
                    }
                    for c in case_results
                ],
                "ndcg_summary": _ndcg_summary(case_results),
            }
        )

    return {
        "run_id": run.run_id,
        "run_name": run.run_name,
        "run_status": run.status,
        "run_config": run_config.sanitized(),
        "git_commit": run.code_commit,
        "working_tree_dirty": run.working_tree_dirty,
        "corpus_manifest": corpus_manifest.get("documents"),
        "corpus_hash": run.corpus_hash,
        "dataset_version": run.dataset_version,
        "dataset_hash": run.dataset_hash,
        "canonical_text_version": run.canonical_text_version,
        "loader_version": run.loader_version,
        "identifier_normalization_version": run.identifier_normalization_version,
        "random_seed": run.random_seed,
        "metric_config": metric_config.to_dict(),
        "trials": trial_blocks,
        "not_computable_metrics": not_computable,
        "errors": errors,
        "remaining_collections": remaining_collections or [],
        "known_limitations": KNOWN_LIMITATIONS,
    }


def _fmt_metric(m: dict[str, Any]) -> str:
    if not m["computable"]:
        return f"n/a (不可計算：{m['not_computable_reason']})"
    val = m["value"]
    val_s = f"{val:.4f}" if isinstance(val, float) else str(val)
    sc = m.get("sample_count")
    suffix = f" (n={sc})" if sc is not None else ""
    return f"{val_s}{suffix}"


def _render_chunk_cache(ap, t: dict[str, Any]) -> None:
    """Slice 3（§二十三）：Chunking + Embedding Cache 統計。

    **不**輸出 embedding vectors / chunk 全文 / API key / 私人絕對路徑（stats blob 本身不含）。
    """
    stats = t.get("chunk_cache_stats")
    if not stats:
        return
    ap("### Chunking + Embedding Cache")
    ctx = stats.get("context_config") or {}
    ap(f"- Strategy: **{stats.get('strategy')}** · Chunk Config Hash: "
       f"`{(stats.get('chunk_config_hash') or '')[:12]}…`")
    ap(f"- Chunk Config Origin: **{stats.get('chunk_config_origin', 'explicit_trial_config')}** "
       f"（不讀 .env / Settings 覆寫）")
    ap(f"- Chunk Count: {stats.get('chunk_count')} · Section Prefix: "
       f"**{stats.get('section_prefix_enabled')}**")
    ap(f"- Strategy Fallback: {stats.get('strategy_fallback_count')} "
       f"{stats.get('strategy_fallback_reasons') or []}")
    ap(f"- Context: before={ctx.get('context_before_paragraphs', 0)} "
       f"after={ctx.get('context_after_paragraphs', 0)} · added={stats.get('context_added_count')} "
       f"· reduced={stats.get('context_reduced_count')}")
    fp = stats.get("embedding_fingerprint") or {}
    ap(f"- Embedding Fingerprint: `{(stats.get('embedding_fingerprint_hash') or '')[:12]}…` "
       f"(model=`{fp.get('model_name')}` dim={fp.get('dimension')} "
       f"revision=`{fp.get('model_revision')}` cache_schema=`{stats.get('cache_schema_version')}`)")
    cache = stats.get("embedding_cache") or {}
    rate = cache.get("embedding_cache_hit_rate")
    rate_s = f"{rate:.4f}" if isinstance(rate, float) else "n/a"
    ap(f"- Embedding Cache Enabled: **{cache.get('cache_enabled')}**")
    ap(f"- Unique Embedding Text: {cache.get('unique_embedding_text_count')} · "
       f"Persistent Hits: {cache.get('embedding_cache_hit_count')} · "
       f"Misses: {cache.get('embedding_cache_miss_count')} · "
       f"Batch Dedup Savings: {cache.get('batch_dedup_savings')}")
    ap(f"- Underlying Model Text Count: {cache.get('embedding_model_call_text_count')} · "
       f"Corrupt: {cache.get('embedding_cache_corrupt_count')} · "
       f"Write Failures: {cache.get('embedding_cache_write_failure_count')} · "
       f"Hit Rate: {rate_s}")
    ap("")


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    ap = lines.append
    ap(f"# Run Report — {report['run_name']}")
    ap("")
    ap(f"- Run ID: `{report['run_id']}`")
    ap(f"- Run Status: **{report['run_status']}**")
    ap(f"- Git Commit: `{report['git_commit']}`  ")
    dirty = report["working_tree_dirty"]
    ap(f"- Working Tree Dirty: **{dirty}**"
       + ("  ⚠️ 非 Release Baseline（開發期 dirty run）" if dirty else ""))
    ap(f"- Corpus Hash: `{report['corpus_hash']}`")
    ap(f"- Dataset Version / Hash: `{report['dataset_version']}` / `{report['dataset_hash']}`")
    ap(f"- Canonical Text Version: `{report['canonical_text_version']}`")
    ap(f"- Loader Version: `{report['loader_version']}`")
    ap(f"- Identifier Normalization Version: `{report['identifier_normalization_version']}`")
    ap(f"- Random Seed: `{report['random_seed']}`")
    ap(f"- Metric Config Version: `{report['metric_config']['version']}`")
    ap("")

    ap("## Corpus Manifest")
    for d in report.get("corpus_manifest") or []:
        ap(f"- `{d['path']}` — sha256 `{d['sha256'][:12]}…` ({d['bytes']} bytes)")
    ap("")

    for t in report["trials"]:
        ap(f"## Trial: {t['trial_name']} (ordinal {t['ordinal']})")
        ap(f"- Status: **{t['status']}** · Attempts: {t['attempt_count']}")
        ap(f"- Config Hash: `{t['config_hash']}`")
        ap(f"- Collection: `{t['collection_name']}` · Cleanup: **{t['cleanup_status']}**")
        ap(f"- Point Count: {t['point_count']} · Index Build: {t['index_build_time_ms']} ms "
           f"· Retrieval: {t['retrieval_time_ms']} ms")
        if t["error"]:
            ap(f"- Error: `{t['error']}`")
        ap("")
        _render_chunk_cache(ap, t)
        ap("### Aggregate Metrics")
        ap("| metric | value |")
        ap("| --- | --- |")
        for m in t["aggregate_metrics"]:
            ap(f"| {m['metric_name']} | {_fmt_metric(m)} |")
        ap("")
        if t["per_query_type_metrics"]:
            ap("### Per Query Type")
            ap("| query_type | metric | value |")
            ap("| --- | --- | --- |")
            for m in t["per_query_type_metrics"]:
                ap(f"| {m['query_type']} | {m['metric_name']} | {_fmt_metric(m)} |")
            ap("")
        nd = t.get("ndcg_summary") or {}
        ap(f"### NDCG (trial_index_gold) — computable {nd.get('computable_cases', 0)} / "
           f"non-computable {nd.get('non_computable_cases', 0)}")
        for reason, count in sorted((nd.get("not_computable_reasons") or {}).items()):
            ap(f"- non-computable `{reason}`: {count}")
        ap("")
        ap("### Per Case (Top-5 + Evidence Match + NDCG)")
        for c in t["cases"]:
            ap(f"- **{c['case_id']}** [{c['query_type']}] status=`{c['case_status']}` "
               f"first_relevant_rank={c['first_relevant_rank']} latency={c['latency_ms']}ms")
            for r in c["top_results"]:
                ident = ",".join(r.get("identifiers") or []) or "-"
                dup = " dup" if r.get("is_duplicate") else ""
                ap(f"    - #{r['rank']} `{r.get('filename')}` / {r.get('section')} "
                   f"(p{r.get('page')}) score={r.get('score')} rel={r['relevance_level']} "
                   f"id=[{ident}]{dup}")
            em = c["evidence_match"]
            matches = em.get("matches", []) if isinstance(em, dict) else em
            reasons = sorted({rr for mm in matches for rr in mm.get("match_reasons", [])})
            if reasons:
                ap(f"    - match_reasons: {', '.join(reasons)}")
            ndcg = em.get("ndcg") if isinstance(em, dict) else None
            if ndcg:
                if ndcg.get("computable"):
                    ap(f"    - ndcg={ndcg['value']} (dcg={ndcg['dcg']} idcg={ndcg['idcg']} "
                       f"relevant_chunks={ndcg['relevant_chunk_count']})")
                else:
                    ap(f"    - ndcg=n/a ({ndcg.get('not_computable_reason')})")
        ap("")

    if report["not_computable_metrics"]:
        ap("## 不可計算 Metrics")
        for nc in report["not_computable_metrics"]:
            ap(f"- {nc['trial_name']} · {nc['metric_name']} ({nc['scope']}"
               f"{'/' + nc['query_type'] if nc['query_type'] else ''}) — {nc['reason']}")
        ap("")

    if report["errors"]:
        ap("## Errors")
        for e in report["errors"]:
            ap(f"- [{e['scope']}] {e['error']}")
        ap("")

    ap("## Remaining Collections")
    if report["remaining_collections"]:
        for c in report["remaining_collections"]:
            ap(f"- ⚠️ `{c}`（未清理，需手動 cleanup）")
    else:
        ap("- （無殘留 Trial collection）")
    ap("")

    ap("## Known Limitations")
    for lim in report["known_limitations"]:
        ap(f"- {lim}")
    ap("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: str) -> tuple[str, str]:
    run_dir = os.path.join(output_dir, report["run_id"])
    os.makedirs(run_dir, exist_ok=True)
    json_path = os.path.join(run_dir, "report.json")
    md_path = os.path.join(run_dir, "report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    return json_path, md_path
