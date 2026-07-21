"""Review Report 產生（Slice 6 §二十）。

輸出 ``<review_output>/<session_id>/review-report.json`` 與 ``review-report.md``。

**不得包含**（§二十）：API Key、``.env`` 內容、完整 Prompt、完整 LLM Response、
文件全文、私人絕對路徑。因此：

- Generator metadata 僅保留白名單欄位（provider / model / request id / token usage /
  response hash），其餘一律丟棄。
- 所有路徑一律轉為 repo-relative（無法轉換時只留檔名）。
- Evidence 只顯示短 Snippet，不顯示全文。
"""
from __future__ import annotations

import json
import os
from typing import Any

from evaluation.v11 import hashing
from evaluation.v11.review.models import (
    CANDIDATE_ACCEPTED,
    CANDIDATE_PROPOSED,
    CANDIDATE_PUBLISHED,
    CANDIDATE_REJECTED,
)
from evaluation.v11.review.service import ReviewService

#: Generator metadata 允許出現在 Report 的欄位（白名單；不含 prompt / response 全文）。
_ALLOWED_GENERATOR_META_KEYS = (
    "generator",
    "provider",
    "model",
    "request_id",
    "response_sha256",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)

KNOWN_LIMITATIONS = [
    "本 Slice 不執行 Active Sampling，也不依 Retrieval 結果挑題。",
    "Candidate Generator 不得取得 Test Metrics／Recommendation；Report 亦不含這些資訊。",
    "Unanswerable 的 Gold Label 由人工負責，未以 LLM 掃描整個 Corpus 自動證明無答案。",
    "發布不會自動重跑 Optimizer Search，也不會 Apply 任何檢索參數。",
]


def _rel_path(path: str | None) -> str | None:
    """轉為 repo-relative 路徑；無法轉換時只留檔名（不外洩私人絕對路徑）。"""
    if not path:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(path), hashing.repo_root())
    except ValueError:  # 跨磁碟機
        return os.path.basename(path)
    if rel.startswith(".."):
        return os.path.basename(path)
    return rel.replace(os.sep, "/")


def _safe_generator_meta(raw_json: str) -> dict[str, Any]:
    try:
        data = json.loads(raw_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in _ALLOWED_GENERATOR_META_KEYS}


def build_report(
    service: ReviewService,
    session_id: str,
    *,
    new_dataset_version: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """組出 Review Report 內容（純資料，不寫檔）。"""
    session = service.get_session(session_id)
    candidates = service.repo.list_candidates(session_id)
    events = service.repo.list_events(session_id)

    status_counts = {
        CANDIDATE_PROPOSED: 0,
        CANDIDATE_ACCEPTED: 0,
        CANDIDATE_REJECTED: 0,
        CANDIDATE_PUBLISHED: 0,
    }
    for c in candidates:
        status_counts[c.status] = status_counts.get(c.status, 0) + 1

    evidence_validation = []
    for c in candidates:
        if c.status in (CANDIDATE_ACCEPTED, CANDIDATE_PUBLISHED):
            result = service.validate_candidate_evidence(c.candidate_id)
            evidence_validation.append(
                {
                    "candidate_id": c.candidate_id,
                    "case_id": c.effective_case_id,
                    "valid": result.valid,
                    "issues": [i.code for i in result.issues],
                    "snippets": result.snippets,  # 只有短 snippet
                }
            )

    coverage_before = service.coverage(session_id, include_accepted=False)
    coverage_after = service.coverage(session_id, include_accepted=True)

    eligibility = None
    if new_dataset_version and output_path:
        eligibility = service.check_publish_eligibility(
            session_id, new_dataset_version=new_dataset_version, output_path=output_path
        ).to_dict()

    audit_summary: dict[str, int] = {}
    for e in events:
        audit_summary[e.event_type] = audit_summary.get(e.event_type, 0) + 1

    return {
        "session": {
            "session_id": session.session_id,
            "session_name": session.session_name,
            "status": session.status,
            "created_at": session.created_at,
            "completed_at": session.completed_at,
        },
        "base_dataset": {
            "path": _rel_path(session.base_dataset_path),
            "version": session.base_dataset_version,
            "hash": session.base_dataset_hash,
        },
        "corpus_hash": session.corpus_hash,
        "canonical_text_version": session.canonical_text_version,
        "loader_version": session.loader_version,
        "generator": {
            "type": session.generator_type,
            "model": session.generator_model,
        },
        "candidate_count": len(candidates),
        "candidate_status_counts": status_counts,
        "revision_count": service.repo.count_revisions(session_id),
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "case_id": c.effective_case_id,
                "status": c.status,
                "current_revision": c.current_revision,
                "query_type": c.proposed_query_type,
                "answerable": c.proposed_answerable,
                "unanswerable_confirmed": c.unanswerable_confirmed,
                "source_filename": c.source_filename,
                "generator_metadata": _safe_generator_meta(c.generator_metadata_json),
            }
            for c in candidates
        ],
        "evidence_validation": evidence_validation,
        "coverage_before": coverage_before.to_dict(),
        "coverage_after": coverage_after.to_dict(),
        "coverage_gaps": [g.code for g in coverage_after.gaps],
        "audit_summary": audit_summary,
        "audit_event_count": len(events),
        "publish_eligibility": eligibility,
        "published_dataset": {
            "path": _rel_path(session.published_dataset_path),
            "version": session.published_dataset_version,
            "hash": session.published_dataset_hash,
        },
        "known_limitations": KNOWN_LIMITATIONS,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    s = report["session"]
    lines = [
        f"# Review Report — {s['session_name']}",
        "",
        f"- Session ID：`{s['session_id']}`",
        f"- Status：**{s['status']}**",
        f"- Base Dataset：`{report['base_dataset']['path']}` "
        f"(version `{report['base_dataset']['version']}`)",
        f"- Base Dataset Hash：`{report['base_dataset']['hash']}`",
        f"- Corpus Hash：`{report['corpus_hash']}`",
        f"- Generator：`{report['generator']['type']}` / `{report['generator']['model']}`",
        "",
        "## Candidates",
        "",
        f"- 總數：{report['candidate_count']}",
    ]
    for status, count in report["candidate_status_counts"].items():
        lines.append(f"- {status}：{count}")
    lines += [
        f"- Revision 總數：{report['revision_count']}",
        "",
        "## Evidence Validation",
        "",
    ]
    if not report["evidence_validation"]:
        lines.append("（無 accepted / published candidate）")
    for ev in report["evidence_validation"]:
        mark = "OK" if ev["valid"] else "FAIL"
        issues = f"（issues：{', '.join(ev['issues'])}）" if ev["issues"] else ""
        lines.append(f"- `{ev['case_id']}`：**{mark}**{issues}")

    lines += ["", "## Coverage", ""]
    before = report["coverage_before"]["summary"]
    after = report["coverage_after"]["summary"]
    lines += [
        f"- Before：{before['total_cases']} cases，splits={before['cases_per_split']}",
        f"- After：{after['total_cases']} cases，splits={after['cases_per_split']}",
        "",
        "### Coverage Gaps",
        "",
    ]
    gaps = report["coverage_after"]["gaps"]
    if not gaps:
        lines.append("（無）")
    for g in gaps:
        lines.append(f"- `{g['code']}`：{g['message']}")

    lines += ["", "## Publish Eligibility", ""]
    elig = report["publish_eligibility"]
    if elig is None:
        lines.append("（未指定目標 dataset version / output path，未檢查）")
    else:
        lines.append(f"- Eligible：**{elig['eligible']}**")
        for e in elig["errors"]:
            lines.append(f"- ERROR：{e}")
        for w in elig["warnings"]:
            lines.append(f"- WARNING：{w}")

    pub = report["published_dataset"]
    lines += [
        "",
        "## Published Dataset",
        "",
        f"- Path：`{pub['path']}`",
        f"- Version：`{pub['version']}`",
        f"- Hash：`{pub['hash']}`",
        "",
        "## Audit Summary",
        "",
    ]
    for event_type, count in sorted(report["audit_summary"].items()):
        lines.append(f"- {event_type}：{count}")

    lines += ["", "## Known Limitations", ""]
    lines.extend(f"- {item}" for item in report["known_limitations"])
    lines.append("")
    return "\n".join(lines)


def write_report(
    service: ReviewService,
    session_id: str,
    review_output_dir: str,
    *,
    new_dataset_version: str | None = None,
    output_path: str | None = None,
) -> tuple[str, str]:
    """寫出 JSON 與 Markdown Report，回傳 ``(json_path, markdown_path)``。"""
    report = build_report(
        service,
        session_id,
        new_dataset_version=new_dataset_version,
        output_path=output_path,
    )
    session_dir = os.path.join(review_output_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)

    json_path = os.path.join(session_dir, "review-report.json")
    md_path = os.path.join(session_dir, "review-report.md")
    with open(json_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_render_markdown(report))
    return json_path, md_path
