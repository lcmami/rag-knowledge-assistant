"""Deployment Report（Slice 7 §二十二）。

可讀 JSON / Markdown，位置為 gitignored ``evaluation/deployments/<operation_id>/``。
**不得**包含 ``.env`` / API key / confirmation 原文 / 文件全文 / embedding vector /
私人絕對路徑 / DB 私人位置（§二十二 排除清單）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from evaluation.v11.deploy.models import DeploymentOperation, DeploymentRecord


def build_report(
    operation: DeploymentOperation,
    *,
    eligibility: Optional[dict[str, Any]] = None,
    build_manifest: Optional[dict[str, Any]] = None,
    readiness: Optional[dict[str, Any]] = None,
    alias_switch: Optional[dict[str, Any]] = None,
    record: Optional[DeploymentRecord] = None,
    audit_summary: Optional[list[dict[str, Any]]] = None,
    rollback_target: Optional[str] = None,
    reconciliation: Optional[dict[str, Any]] = None,
    known_limitations: Optional[list[str]] = None,
) -> dict[str, Any]:
    """組出 redacted deployment report dict（只含安全摘要欄位）。"""
    return {
        "operation_id": operation.operation_id,
        "operation_type": operation.operation_type,
        "status": operation.status,
        "alias": operation.alias_name,
        "previous_target": operation.source_collection,
        "requested_target": operation.target_collection,
        "final_target": (record.physical_collection if record else operation.target_collection),
        "search_id": operation.search_id,
        "candidate_id": operation.candidate_id,
        "config_hash": operation.config_hash,
        "dataset_hash": operation.dataset_hash,
        "corpus_hash": operation.corpus_hash,
        "code_commit": operation.code_commit,
        "eligibility_decision": (eligibility.get("eligible") if eligibility else None),
        "eligibility_rejection_reasons": (eligibility.get("reasons") if eligibility else []),
        "confirmation_valid": operation.confirmation_hash is not None,
        "idempotency_key_hash": _sha(operation.idempotency_key),
        "build_manifest": build_manifest,
        "readiness_report": readiness,
        "alias_switch_result": alias_switch,
        "read_back_result": (alias_switch.get("read_back") if alias_switch else None),
        "rollback_target": rollback_target,
        "audit_summary": audit_summary or [],
        "reconciliation_status": reconciliation,
        "error_code": operation.error_code,
        "known_limitations": known_limitations or _default_limitations(),
    }


def _default_limitations() -> list[str]:
    return [
        "single-machine local admin interface：無 RBAC / 登入 / 多使用者審批；"
        "concurrent guard 為單機 SQLite lock，跨機不保證。",
        "本 Slice 不對正式 factory_knowledge 執行 Apply / Rollback / Alias 初始化。",
    ]


def _sha(text: Optional[str]) -> Optional[str]:
    import hashlib

    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Deployment Report — {report['operation_type']} `{report['operation_id']}`",
        "",
        f"- Status: **{report['status']}**",
        f"- Alias: `{report['alias']}`",
        f"- Previous Target: `{report['previous_target']}`",
        f"- Requested Target: `{report['requested_target']}`",
        f"- Final Target: `{report['final_target']}`",
        f"- Search ID: `{report['search_id']}`",
        f"- Candidate ID: `{report['candidate_id']}`",
        f"- Config Hash: `{report['config_hash']}`",
        f"- Dataset Hash: `{report['dataset_hash']}`",
        f"- Corpus Hash: `{report['corpus_hash']}`",
        f"- Code Commit: `{report['code_commit']}`",
        f"- Eligibility: {report['eligibility_decision']}",
        f"- Eligibility Rejection Reasons: {report['eligibility_rejection_reasons']}",
        f"- Confirmation Valid: {report['confirmation_valid']}",
        f"- Idempotency Key Hash: `{report['idempotency_key_hash']}`",
        f"- Rollback Target: `{report['rollback_target']}`",
        f"- Reconciliation Status: {report['reconciliation_status']}",
        f"- Error Code: `{report['error_code']}`",
        "",
        "## Known Limitations",
    ]
    lines += [f"- {x}" for x in report["known_limitations"]]
    return "\n".join(lines) + "\n"


def write_report(report_dir: str, operation_id: str, report: dict[str, Any]) -> dict[str, str]:
    """寫出 JSON + Markdown 至 ``<report_dir>/<operation_id>/``（gitignored）。回傳兩檔路徑。"""
    out_dir = os.path.join(report_dir, operation_id)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "deployment-report.json")
    md_path = os.path.join(out_dir, "deployment-report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    return {"json": json_path, "markdown": md_path}
