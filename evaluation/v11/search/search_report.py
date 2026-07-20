"""Optimizer Report（Slice 5 §二十二）—— JSON + Markdown。

**洩漏防護**（§13.4）：只要 ``recommendation_locked`` 為 False，報告就不會輸出任何
test split 統計 —— 連 section 都不會出現，取而代之的是一段明確說明。

**不輸出**：API key、``.env``、embedding vector、文件全文、私人絕對路徑、cache DB 路徑。
所有路徑一律轉為 repo 相對路徑後才寫入。
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from evaluation.v11 import hashing
from evaluation.v11.report import _fmt_metric, _metric_to_dict
from evaluation.v11.search import complexity as complexity_mod
from evaluation.v11.search import recommendation as rec_mod
from evaluation.v11.search.search_config import SearchConfig
from evaluation.v11.search.search_models import SearchStatus
from evaluation.v11.search.search_repository import SearchRepository

#: §十九：本 Slice 已知無法以搜尋空間解決的限制。
C01_CASE_ID = "eval-e102-001"

KNOWN_LIMITATIONS: list[str] = [
    "本 Slice 產生的是 recommended_candidate，**不是** deployed / active / production "
    "candidate；未套用至任何正式系統，未建立 alias，未修改正式 retrieval config。",
    "Hard constraint 門檻來自 Slice 0 baseline 的建議值，樣本數極小（n=4~5），"
    "policy_confidence=low；不適用於其他資料集或其他機器，亦非正式 Production SLA。",
    "P95 latency 門檻高度依賴硬體與 Qdrant 部署方式，跨環境不可直接沿用。",
    "每個 candidate 建立自己的隔離 trial collection；共用單一 collection 的 index variant "
    "屬未來效能優化，非本 Slice 阻塞條件。",
    "Random search 以「先列舉完整合法空間再取樣」實作，因此可保證 without replacement 與"
    "可重現性，但不適用於無法窮舉的巨大搜尋空間。",
    "Tune 預篩的 eliminate_on_not_computable 預設為 false，避免小樣本 split 上"
    "結構性不可計算的 metric 把全部 candidate 提前刷掉；validation 正式判定不受此影響。",
]


def _rel(path: Optional[str], base: str) -> Optional[str]:
    """轉為 repo 相對路徑（§22：不得輸出私人絕對路徑）。"""
    if not path:
        return None
    try:
        return os.path.relpath(path, base).replace("\\", "/")
    except ValueError:
        return os.path.basename(path)


def _metrics_dict(repo: SearchRepository, trial_id: Optional[str]) -> dict[str, Any]:
    if not trial_id:
        return {}
    out: dict[str, Any] = {}
    for m in repo.get_trial_metrics(trial_id):
        key = m.metric_name if m.scope == "aggregate" else f"{m.metric_name}@{m.query_type}"
        out[key] = _metric_to_dict(m)
    return out


def _case_rows(repo: SearchRepository, trial_id: Optional[str]) -> dict[str, Any]:
    if not trial_id:
        return {}
    return {r.case_id: r for r in repo.get_trial_case_results(trial_id)}


def _top5(case_row) -> list[dict[str, Any]]:
    try:
        results = json.loads(case_row.top_results_json)
    except (TypeError, ValueError):
        return []
    out = []
    for r in results[:5]:
        out.append(
            {
                "rank": r.get("rank"),
                "source": r.get("filename"),
                "section": r.get("section"),
                "score": r.get("score"),
                "vector_score": r.get("vector_score"),
                "ranking_score": r.get("ranking_score"),
                "relevance_level": r.get("relevance_level"),
                "identifiers": r.get("identifiers"),
                "exact_identifier_matches": r.get("exact_identifier_matches"),
                "admission_reason": r.get("admission_reason"),
            }
        )
    return out


def per_case_diff(
    repo: SearchRepository,
    baseline_trial_id: Optional[str],
    candidate_trial_id: Optional[str],
    *,
    label: str,
) -> dict[str, Any]:
    """§22 per-case Top-K diff（baseline vs candidate）。"""
    base_rows = _case_rows(repo, baseline_trial_id)
    cand_rows = _case_rows(repo, candidate_trial_id)
    cases: list[dict[str, Any]] = []
    for case_id in sorted(set(base_rows) | set(cand_rows)):
        b = base_rows.get(case_id)
        c = cand_rows.get(case_id)
        b_rank = getattr(b, "first_relevant_rank", None)
        c_rank = getattr(c, "first_relevant_rank", None)
        if b_rank == c_rank:
            verdict = "same"
        elif c_rank is None:
            verdict = "regressed"
        elif b_rank is None:
            verdict = "improved"
        else:
            verdict = "improved" if c_rank < b_rank else "regressed"
        cases.append(
            {
                "case_id": case_id,
                "question": None,
                "query_type": getattr(c or b, "query_type", None),
                "answerable": getattr(c or b, "answerable", None),
                "baseline_first_relevant_rank": b_rank,
                "candidate_first_relevant_rank": c_rank,
                "baseline_case_status": getattr(b, "case_status", None),
                "candidate_case_status": getattr(c, "case_status", None),
                "baseline_top5": _top5(b) if b else [],
                "candidate_top5": _top5(c) if c else [],
                "verdict": verdict,
            }
        )
    return {
        "label": label,
        "baseline_trial_id": baseline_trial_id,
        "candidate_trial_id": candidate_trial_id,
        "improved": sum(1 for c in cases if c["verdict"] == "improved"),
        "same": sum(1 for c in cases if c["verdict"] == "same"),
        "regressed": sum(1 for c in cases if c["verdict"] == "regressed"),
        "cases": cases,
    }


def _attach_questions(diff: dict[str, Any], dataset_path: str) -> None:
    """補上題目文字（來自 dataset，不含文件全文）。"""
    try:
        with open(dataset_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    q = {str(c["id"]): c.get("question") for c in raw.get("cases", [])}
    for row in diff["cases"]:
        row["question"] = q.get(row["case_id"])


# --------------------------------------------------------------------------
def build_search_report(
    result, search_config: SearchConfig, repo: SearchRepository
) -> dict[str, Any]:
    """組出完整 Optimizer Report（JSON 結構）。"""
    from evaluation.v11 import trial_runner

    base = trial_runner.ROOT
    search = result.search
    locked = bool(search.recommendation_locked)

    cand_by_hash = {c.config_hash: c for c in result.candidates if c.config_hash}
    baseline_hash = search.baseline_config_hash

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "slice": "slice5-optimizer-search",
        # -- 身分 / 可重現性 --
        "search_id": search.search_id,
        "search_name": search.search_name,
        "search_mode": search.search_mode,
        "search_config_hash": search.search_config_hash,
        "search_config": json.loads(search.search_config_json),
        "random_seed": search.random_seed,
        "candidate_budget": search.max_candidates,
        "validation_top_n": search.validation_top_n,
        "status": search.status,
        "dataset_version": search.dataset_version,
        "dataset_hash": search.dataset_hash,
        "corpus_hash": search.corpus_hash,
        "git_commit": search.code_commit,
        "working_tree_dirty": search.working_tree_dirty,
        # -- 生成統計（§9.9）--
        "generation_stats": result.generation.stats(),
        "candidate_manifest": result.generation.manifest(),
        # -- Split（§十二 / §十三）--
        "split_manifest": result.split_manifest.to_dict(),
        "group_leakage_check": {
            "ok": result.split_manifest.leakage_ok,
            "detail": result.split_manifest.leakage_detail,
            "note": "同 group_id 只允許存在於一個 split（§13.1）。",
        },
        "validation_coverage": result.coverage.to_dict() if result.coverage else None,
        # -- Baseline / policy --
        "baseline_config": {
            "config_hash": baseline_hash,
            "chunk_config": search_config.baseline_chunk_config,
            "retrieval_config": search_config.baseline_retrieval_config,
        },
        "hard_constraint_policy": {
            "policy_source": search_config.policy_source,
            "policy_confidence": search_config.policy_confidence,
            "constraints": [c.to_dict() for c in search_config.hard_constraints],
            "disclaimer": (
                "這些門檻來自 Slice 0 baseline 的低可信度建議值（樣本數 n=4~5）。"
                "不宣稱適合所有資料集、不宣稱適合所有機器、不是正式 Production SLA。"
            ),
        },
        "pareto_objectives": [o.to_dict() for o in search_config.pareto_objectives],
        "recommendation_policy": search_config.recommendation_policy.to_dict(),
        "complexity_measure": {
            "version": complexity_mod.COMPLEXITY_VERSION,
            "note": (
                "complexity 僅在所有品質指標完全相同時作為 tie-break，"
                "排在 recommendation_ranking 的所有 metric 之後。"
            ),
        },
        "known_limitations": list(KNOWN_LIMITATIONS),
        "errors": [],
        "notes": list(result.notes),
        "remaining_collections": list(result.remaining_collections),
    }

    # -- Tune / Validation 明細 --
    tune = result.tune
    validation = result.validation
    report["errors"].extend((tune.errors if tune else []) + (validation.errors if validation else []))

    candidate_rows: list[dict[str, Any]] = []
    for c in sorted(result.candidates, key=lambda x: x.ordinal):
        row = c.to_dict()
        cfg = json.loads(c.config_json)
        row["chunk_config"] = cfg.get("chunk_config")
        row["retrieval_config"] = cfg.get("retrieval_config")
        row["assignment"] = cfg.get("assignment")
        if c.config_hash:
            row["complexity"] = complexity_mod.describe(
                cfg.get("chunk_config") or {}, cfg.get("retrieval_config") or {}
            )
        row["tune_metrics"] = _metrics_dict(repo, c.tune_trial_id)
        row["validation_metrics"] = _metrics_dict(repo, c.validation_trial_id)
        if c.constraint_results_json:
            row["constraint_results"] = json.loads(c.constraint_results_json)
        if c.rejection_reasons_json:
            row["rejection_reasons"] = json.loads(c.rejection_reasons_json)
        candidate_rows.append(row)
    report["candidates"] = candidate_rows

    report["tune_ranking"] = [r.to_dict() for r in result.tune_ranking]

    # -- Constraint matrix（§22）--
    matrix: list[dict[str, Any]] = []
    for chash, ce in sorted(result.constraint_evals.items()):
        matrix.append(
            {
                "config_hash": chash,
                "feasible": ce.feasible,
                "pass_count": ce.pass_count,
                "fail_count": ce.fail_count,
                "results": [r.to_dict() for r in ce.results],
            }
        )
    report["constraint_matrix"] = matrix
    report["feasible_config_hashes"] = sorted(
        chash for chash, ce in result.constraint_evals.items() if ce.feasible
    )
    report["infeasible_config_hashes"] = sorted(
        chash for chash, ce in result.constraint_evals.items() if not ce.feasible
    )

    # -- Pareto --
    report["pareto"] = result.pareto.to_dict() if result.pareto else None

    # -- Recommendation --
    rec = result.recommendation
    report["recommendation"] = {
        **(rec.to_dict() if rec else {}),
        "recommendation_locked": locked,
        "recommendation_locked_at": search.recommendation_locked_at,
        "terminology_note": (
            "recommended_candidate 僅代表本次搜尋的建議，"
            "**不是** deployed / active / production candidate；本 Slice 不套用任何設定。"
        ),
    }
    report["no_feasible_candidate_state"] = {
        "is_no_feasible_candidate": search.status
        in (
            SearchStatus.NO_FEASIBLE_CANDIDATE,
            SearchStatus.INSUFFICIENT_VALIDATION_COVERAGE,
        ),
        "status": search.status,
        "recommended_config_hash": search.recommended_config_hash,
        "explanation": (
            rec.reason
            if rec and rec.status != rec_mod.STATUS_OK
            else "有可行 candidate，已選出 recommendation。"
        ),
    }

    # -- Cache / latency（§20.5）--
    cache_rows: list[dict[str, Any]] = []
    for stage_name, stage in (("tune", tune), ("validation", validation)):
        if stage is None:
            continue
        for chash, trial in sorted(stage.trial_by_hash.items()):
            stats = None
            if trial.chunk_cache_stats_json:
                try:
                    stats = json.loads(trial.chunk_cache_stats_json)
                except ValueError:
                    stats = None
            cache_rows.append(
                {
                    "split": stage_name,
                    "config_hash": chash,
                    "trial_name": trial.trial_name,
                    "strategy": trial.strategy,
                    "chunk_config_hash": trial.chunk_config_hash,
                    "chunk_count": trial.chunk_count,
                    "point_count": trial.point_count,
                    "index_build_time_ms": trial.index_build_time_ms,
                    "retrieval_time_ms": trial.retrieval_time_ms,
                    "cleanup_status": trial.cleanup_status,
                    "cache_stats": stats,
                }
            )
    report["cache_and_latency"] = cache_rows

    # -- Baseline vs 最佳 validation candidate 的 per-case diff（lock 前也可揭露）--
    diffs: list[dict[str, Any]] = []
    val_dataset = result.split_manifest.splits["validation"].dataset_path
    baseline_val_trial = None
    if validation is not None and baseline_hash in validation.trial_by_hash:
        baseline_val_trial = validation.trial_by_hash[baseline_hash].trial_id
    top_val = next(
        (
            c
            for c in sorted(result.candidates, key=lambda x: (x.recommendation_rank or 10**6))
            if c.validation_trial_id and c.config_hash != baseline_hash
        ),
        None,
    )
    if baseline_val_trial and top_val:
        d = per_case_diff(
            repo,
            baseline_val_trial,
            top_val.validation_trial_id,
            label=f"validation: baseline vs {top_val.config_hash}",
        )
        if val_dataset:
            _attach_questions(d, val_dataset)
        diffs.append(d)

    # ------------------------------------------------------------------
    # Test 區段 —— **只有 lock 之後才存在**（§13.4）
    # ------------------------------------------------------------------
    if not locked:
        report["test"] = {
            "available": False,
            "reason": (
                "recommendation 尚未 lock，依 §12.4 / §13.3 / §13.4，"
                "test split 未被讀取、未產生任何 test trial 或 test metric，"
                "報告亦不得顯示任何 test 統計。"
            ),
            "test_trial_count": repo.count_test_trials(search.search_id),
        }
        report["c01_same_chunk_multi_identifier"] = {
            "case_id": C01_CASE_ID,
            "observable": False,
            "note": (
                f"{C01_CASE_ID} 屬於 test split。recommendation 未 lock，test 未執行，"
                "因此本次搜尋無法觀察其 first relevant rank。"
            ),
        }
    else:
        test = result.test
        rec_hash = search.recommended_config_hash
        base_test_trial = (
            test.trial_by_hash[baseline_hash].trial_id
            if test and baseline_hash in test.trial_by_hash
            else None
        )
        rec_test_trial = (
            test.trial_by_hash[rec_hash].trial_id
            if test and rec_hash in test.trial_by_hash
            else None
        )
        base_metrics = _metrics_dict(repo, base_test_trial)
        rec_metrics = _metrics_dict(repo, rec_test_trial)
        deltas: dict[str, Any] = {}
        for key, bm in base_metrics.items():
            rm = rec_metrics.get(key)
            if rm and bm.get("value") is not None and rm.get("value") is not None:
                deltas[key] = {
                    "baseline": bm["value"],
                    "recommended": rm["value"],
                    "delta": rm["value"] - bm["value"],
                }
        report["test"] = {
            "available": True,
            "test_trial_count": 2 if rec_hash != baseline_hash else 1,
            "note": (
                "test 只執行 baseline 與已鎖定的 recommendation 兩個 config；"
                "test 結果僅用於誠實驗收與標示 regression，"
                "**不得**用來重新調參或改選 candidate（§12.5）。"
            ),
            "baseline_metrics": base_metrics,
            "recommendation_metrics": rec_metrics,
            "baseline_vs_recommendation_delta": deltas,
        }
        test_dataset = result.split_manifest.splits["test"].dataset_path
        if base_test_trial and rec_test_trial:
            d = per_case_diff(
                repo, base_test_trial, rec_test_trial, label="test: baseline vs recommendation"
            )
            if test_dataset:
                _attach_questions(d, test_dataset)
            diffs.append(d)

        # §十九：C01 誠實記錄
        c01 = {}
        for name, tid in (("baseline", base_test_trial), ("recommended", rec_test_trial)):
            rows = _case_rows(repo, tid)
            row = rows.get(C01_CASE_ID)
            c01[name] = getattr(row, "first_relevant_rank", None)
        unresolved = all(v == 2 for v in c01.values() if v is not None)
        report["c01_same_chunk_multi_identifier"] = {
            "case_id": C01_CASE_ID,
            "observable": True,
            "first_relevant_rank": c01,
            "resolved": bool(c01) and not unresolved,
            "note": (
                "No candidate in the approved search space resolved the same-chunk "
                "multi-identifier intent ambiguity."
                if unresolved
                else "搜尋空間內的設定改變了此題的 first relevant rank，詳見 per-case diff。"
            ),
        }

        # Unanswerable 案例誠實記錄
        unans: list[dict[str, Any]] = []
        for name, tid in (("baseline", base_test_trial), ("recommended", rec_test_trial)):
            for cid, row in sorted(_case_rows(repo, tid).items()):
                if not row.answerable:
                    unans.append(
                        {
                            "trial": name,
                            "case_id": cid,
                            "case_status": row.case_status,
                            "first_relevant_rank": row.first_relevant_rank,
                        }
                    )
        report["unanswerable_cases"] = unans

    report["per_case_diffs"] = diffs
    report["report_paths"] = {
        "tune_run_report": _rel(
            os.path.join("runs", "tune"), base
        ),
    }
    return report


# --------------------------------------------------------------------------
def render_search_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    ap = lines.append

    ap(f"# Optimizer Search Report — {report['search_name']}")
    ap("")
    ap(f"- Search ID：`{report['search_id']}`")
    ap(f"- Status：**{report['status']}**")
    ap(f"- Search Mode：{report['search_mode']}")
    ap(f"- Search Config Hash：`{report['search_config_hash']}`")
    ap(f"- Random Seed：{report['random_seed']}")
    ap(f"- Candidate Budget：{report['candidate_budget']}（validation_top_n={report['validation_top_n']}）")
    ap(f"- Dataset：{report['dataset_version']} / hash `{report['dataset_hash']}`")
    ap(f"- Corpus Hash：`{report['corpus_hash']}`")
    ap(f"- Git Commit：`{report['git_commit']}`（dirty={report['working_tree_dirty']}）")
    ap("")

    g = report["generation_stats"]
    ap("## Candidate 生成")
    ap("")
    ap("| Raw Combination | Invalid | Duplicate | Executable |")
    ap("|---:|---:|---:|---:|")
    ap(
        f"| {g['raw_combination_count']} | {g['invalid_count']} | "
        f"{g['duplicate_count']} | {g['executable_candidate_count']} |"
    )
    ap("")
    for note in g.get("notes", []):
        ap(f"> {note}")
    ap("")

    ap("## Dataset Split")
    ap("")
    ap("| Split | Cases | Groups | Query Types | 缺少期望類型 |")
    ap("|---|---:|---:|---|---|")
    for name, info in report["split_manifest"]["splits"].items():
        ap(
            f"| {name} | {info['case_count']} | {info['group_count']} | "
            f"{info['query_type_counts']} | {info['missing_expected_query_types'] or '—'} |"
        )
    ap("")
    leak = report["group_leakage_check"]
    ap(f"Group Leakage Check：{'✅ 乾淨' if leak['ok'] else '❌ ' + str(leak['detail'])}")
    ap("")

    cov = report.get("validation_coverage")
    if cov:
        ap("## Validation Coverage")
        ap("")
        ap(f"- 足以執行 hard constraints：{'✅ 是' if cov['sufficient'] else '❌ 否'}")
        ap(f"- Status：`{cov['status']}`")
        ap(f"- {cov['message']}")
        if cov["blocking_constraints"]:
            ap("")
            ap("| 受阻 Constraint | Metric | 缺少 Query Type |")
            ap("|---|---|---|")
            for b in cov["blocking_constraints"]:
                ap(f"| {b['constraint_name']} | {b['metric']} | {b['missing_query_types']} |")
        ap("")

    pol = report["hard_constraint_policy"]
    ap("## Hard Constraint Policy")
    ap("")
    ap(f"- policy_source：`{pol['policy_source']}`")
    ap(f"- policy_confidence：`{pol['policy_confidence']}`")
    ap(f"- ⚠️ {pol['disclaimer']}")
    ap("")
    ap("| Constraint | Metric | 判定 | 不可計算政策 | Safety |")
    ap("|---|---|---|---|---|")
    for c in pol["constraints"]:
        target = (
            f"{c['operator']} baseline{c['baseline_delta']:+g}"
            if "baseline_delta" in c
            else f"{c['operator']} {c['value']}"
        )
        ap(
            f"| {c['name']} | {c['metric']}"
            f"{'@' + c['query_type'] if c.get('query_type') else ''} | {target} | "
            f"{c['on_not_computable']} | {'是' if c['safety_critical'] else '否'} |"
        )
    ap("")

    ap("## Constraint Matrix（validation）")
    ap("")
    if not report["constraint_matrix"]:
        ap("_無 candidate 進入 validation constraint 判定。_")
    else:
        ap("| Config Hash | Feasible | Pass | Fail | 失敗原因 |")
        ap("|---|---|---:|---:|---|")
        for row in report["constraint_matrix"]:
            fails = [r for r in row["results"] if r["status"] == "fail"]
            reason = "；".join(
                f"{r['name']}("
                + (r["reason"] or f"{r['observed_value']} !{r['operator']} {r['required_value']}")
                + ")"
                for r in fails
            )
            ap(
                f"| `{row['config_hash'][:12]}` | {'✅' if row['feasible'] else '❌'} | "
                f"{row['pass_count']} | {row['fail_count']} | {reason or '—'} |"
            )
    ap("")

    par = report.get("pareto")
    ap("## Pareto Frontier")
    ap("")
    if not par or not par["points"]:
        ap("_無 feasible candidate，未建立 Pareto frontier。_")
    else:
        ap(f"- Frontier size：{par['frontier_size']}")
        if par["ignored_objectives"]:
            ap(f"- 已忽略 objective：{[o['metric'] for o in par['ignored_objectives']]}")
        ap("")
        ap("| Config Hash | Member | Rank | Dominated By | Dominates |")
        ap("|---|---|---:|---|---|")
        for p in par["points"]:
            ap(
                f"| `{p['config_hash'][:12]}` | {'✅' if p['pareto_member'] else '—'} | "
                f"{p['pareto_rank']} | {len(p['dominated_by'])} | {len(p['dominates'])} |"
            )
    ap("")

    rec = report["recommendation"]
    ap("## Recommendation")
    ap("")
    ap(f"- Status：`{rec.get('status')}`")
    ap(f"- Recommended Config Hash：`{rec.get('recommended_config_hash')}`")
    ap(f"- Recommendation Locked：{'✅' if rec.get('recommendation_locked') else '❌'}"
       f"（{rec.get('recommendation_locked_at') or '未鎖定'}）")
    if rec.get("reason"):
        ap(f"- {rec['reason']}")
    ap(f"- ⚠️ {rec['terminology_note']}")
    if rec.get("closest_infeasible_candidates"):
        ap("")
        ap("### Closest Infeasible Candidates（僅供參考，**不得**視為 recommendation）")
        ap("")
        ap("| Config Hash | Fail | Pass | 失敗 Constraint |")
        ap("|---|---:|---:|---|")
        for r in rec["closest_infeasible_candidates"][:10]:
            ap(
                f"| `{r['config_hash'][:12]}` | {r['fail_count']} | {r['pass_count']} | "
                f"{'、'.join(r['failed_constraints'])} |"
            )
    ap("")

    test = report.get("test") or {}
    ap("## Test Split")
    ap("")
    if not test.get("available"):
        ap(f"🔒 {test.get('reason')}")
        ap("")
        ap(f"- 現存 test trial 數：{test.get('test_trial_count')}（必須為 0）")
    else:
        ap(f"- {test['note']}")
        ap("")
        ap("| Metric | Baseline | Recommended | Δ |")
        ap("|---|---:|---:|---:|")
        for key, d in sorted(test["baseline_vs_recommendation_delta"].items()):
            ap(f"| {key} | {d['baseline']:.4f} | {d['recommended']:.4f} | {d['delta']:+.4f} |")
    ap("")

    c01 = report.get("c01_same_chunk_multi_identifier")
    if c01:
        ap("## C01 —— 同 Chunk 多 Identifier 限制（§十九）")
        ap("")
        ap(f"- Case：`{c01['case_id']}`")
        ap(f"- 可觀察：{'是' if c01['observable'] else '否'}")
        if c01.get("first_relevant_rank"):
            ap(f"- First Relevant Rank：{c01['first_relevant_rank']}")
        ap(f"- {c01['note']}")
        ap("")

    ap("## Per-case Diff")
    ap("")
    for d in report.get("per_case_diffs", []):
        ap(f"### {d['label']}")
        ap("")
        ap(f"improved={d['improved']} / same={d['same']} / regressed={d['regressed']}")
        ap("")
        ap("| Case | Query Type | Baseline Rank | Candidate Rank | 判定 |")
        ap("|---|---|---:|---:|---|")
        for row in d["cases"]:
            ap(
                f"| {row['case_id']} | {row['query_type']} | "
                f"{row['baseline_first_relevant_rank']} | "
                f"{row['candidate_first_relevant_rank']} | {row['verdict']} |"
            )
        ap("")

    ap("## Cache / Latency")
    ap("")
    ap("| Split | Trial | Strategy | Chunks | Points | Build ms | Cleanup |")
    ap("|---|---|---|---:|---:|---:|---|")
    for r in report["cache_and_latency"]:
        ap(
            f"| {r['split']} | {r['trial_name']} | {r['strategy']} | {r['chunk_count']} | "
            f"{r['point_count']} | {r['index_build_time_ms']} | {r['cleanup_status']} |"
        )
    ap("")

    if report["errors"]:
        ap("## Errors")
        ap("")
        for e in report["errors"]:
            ap(f"- {e}")
        ap("")

    ap("## Known Limitations")
    ap("")
    for k in report["known_limitations"]:
        ap(f"- {k}")
    ap("")
    if report.get("notes"):
        ap("## Notes")
        ap("")
        for n in report["notes"]:
            ap(f"- {n}")
        ap("")
    ap(f"- Remaining Collections：{report['remaining_collections'] or '無（已全部清理）'}")
    ap("")
    return "\n".join(lines)


def write_search_report(report: dict[str, Any], output_dir: str) -> tuple[str, str]:
    """寫出 ``optimizer-report.json`` / ``optimizer-report.md``。"""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "optimizer-report.json")
    md_path = os.path.join(output_dir, "optimizer-report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_search_markdown(report))
    return json_path, md_path
