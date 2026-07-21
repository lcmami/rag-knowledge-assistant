"""Review Workflow CLI（Slice 6 §十三）。

CLI 只是 :class:`~evaluation.v11.review.service.ReviewService` 的薄包裝——核心邏輯與
狀態機只有一份實作，CLI 與 Service 共用（§二十三 CLI／Integration）。

用法::

    python -m evaluation.v11.review.cli session create --name <n> --base-dataset <path>
    python -m evaluation.v11.review.cli candidate add --session <id> --question ...
    python -m evaluation.v11.review.cli candidate accept --candidate <id> --split validation ...
    python -m evaluation.v11.review.cli evidence add --candidate <id> --document-sha ...
    python -m evaluation.v11.review.cli coverage show --session <id>
    python -m evaluation.v11.review.cli dataset publish --session <id> --version v --out <path>

**沒有**「接受全部候選」的批次指令（§十三）：每次接受都必須指名單一 candidate。
CLI 不呼叫真實 LLM，除非明確使用 ``candidate generate --generator llm``。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from evaluation.v11 import hashing
from evaluation.v11.review.corpus import load_canonical_corpus
from evaluation.v11.review.generator import FakeCandidateGenerator
from evaluation.v11.review.models import DEFAULT_EDITOR
from evaluation.v11.review.report import write_report
from evaluation.v11.review.repository import ReviewRepository
from evaluation.v11.review.service import ReviewService

#: 預設 corpus（與 Slice 0–5 相同的受追蹤 demo corpus）。
DEFAULT_CORPUS = ("sample_docs", "evaluation/fixtures")
#: Review 產物一律落在 ignored 路徑（§二十一）。
DEFAULT_REVIEW_OUTPUT = "evaluation/reviews"
DEFAULT_REVIEW_DB = "evaluation/reviews/review.review.db"


def _emit(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=False))


def _corpus_base_dir(corpus_paths: tuple[str, ...]) -> str:
    """corpus hash 的相對路徑基準。

    相對路徑一律以 repo root 為基準（與 Slice 0–5 一致）；若使用者指定**絕對**
    corpus 路徑（例如測試的 tmp 目錄，可能位於不同磁碟機），則以其父目錄為基準，
    避免跨磁碟機的 ``os.path.relpath`` 失敗。
    """
    absolute = [p for p in corpus_paths if os.path.isabs(p)]
    if not absolute:
        return hashing.repo_root()
    parents = [os.path.dirname(os.path.abspath(p)) for p in absolute]
    try:
        return os.path.commonpath(parents)
    except ValueError:  # 分散於不同磁碟機
        return parents[0]


def _build_service(args) -> ReviewService:
    corpus_paths = tuple(args.corpus) if args.corpus else DEFAULT_CORPUS
    corpus = load_canonical_corpus(list(corpus_paths), _corpus_base_dir(corpus_paths))
    repo = ReviewRepository(args.db)
    return ReviewService(repo, corpus)


def _make_generator(args, service: ReviewService):
    """建立 Generator。``llm`` 需明確指定，且必須有可用設定。"""
    if args.generator == "fake":
        return FakeCandidateGenerator()
    if args.generator == "llm":
        from app.core.config import get_settings
        from app.services.llm_client import LLMClient
        from evaluation.v11.review.llm_generator import LLMCandidateGenerator

        settings = get_settings()
        if not settings.llm_configured:
            raise SystemExit(
                "LLM 未設定（需 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）；"
                "可改用 --generator fake 或 candidate add 手動新增。"
            )
        return LLMCandidateGenerator(LLMClient(settings))
    raise SystemExit(f"未知 generator：{args.generator!r}")


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------
def cmd_session_create(args) -> int:
    service = _build_service(args)
    session = service.create_session(
        session_name=args.name,
        base_dataset_path=args.base_dataset,
        generator_type=args.generator_type,
        generator_model=args.generator_model,
    )
    _emit(session.to_dict())
    return 0


def cmd_session_show(args) -> int:
    service = _build_service(args)
    session = service.get_session(args.session)
    _emit(session.to_dict())
    return 0


def cmd_session_list(args) -> int:
    service = _build_service(args)
    _emit([s.to_dict() for s in service.list_sessions()])
    return 0


def cmd_session_validate(args) -> int:
    service = _build_service(args)
    result = service.validate_session(
        args.session, new_dataset_version=args.version, output_path=args.out
    )
    _emit(result.to_dict())
    return 0 if result.eligible else 1


def cmd_candidate_generate(args) -> int:
    service = _build_service(args)
    generator = _make_generator(args, service)
    outcomes = service.generate_candidates(
        args.session,
        generator,
        document_sha256=args.document_sha,
        segment_ids=args.segment,
        requested_count=args.count,
        actor=args.actor,
    )
    _emit([o.to_dict() for o in outcomes])
    return 0


def cmd_candidate_add(args) -> int:
    service = _build_service(args)
    outcome = service.add_candidate_manual(
        args.session,
        question=args.question,
        query_type=args.query_type,
        answerable=not args.unanswerable,
        source_document_sha256=args.document_sha,
        source_segment_ids=args.segment,
        suggested_group_id=args.group,
        suggested_split=args.split,
        actor=args.actor,
    )
    _emit(outcome.to_dict())
    return 0


def cmd_candidate_list(args) -> int:
    service = _build_service(args)
    candidates = service.repo.list_candidates(args.session, status=args.status)
    _emit([c.to_dict() for c in candidates])
    return 0


def cmd_candidate_show(args) -> int:
    service = _build_service(args)
    candidate = service.repo.require_candidate(args.candidate)
    revisions = service.repo.list_revisions(args.candidate)
    validation = service.validate_candidate_evidence(args.candidate)
    _emit(
        {
            "candidate": candidate.to_dict(),
            "revisions": [r.to_dict() for r in revisions],
            "evidence_validation": validation.to_dict(),
        }
    )
    return 0


def cmd_candidate_edit(args) -> int:
    service = _build_service(args)
    answerable = None
    if args.answerable is not None:
        answerable = args.answerable == "true"
    outcome = service.edit_candidate(
        args.candidate,
        question=args.question,
        query_type=args.query_type,
        answerable=answerable,
        group_id=args.group,
        split=args.split,
        case_id=args.case_id,
        change_reason=args.reason,
        actor=args.actor,
    )
    _emit(outcome.to_dict())
    return 0


def cmd_candidate_accept(args) -> int:
    service = _build_service(args)
    outcome = service.accept_candidate(
        args.candidate,
        group_id=args.group,
        split=args.split,
        case_id=args.case_id,
        change_reason=args.reason,
        actor=args.actor,
    )
    _emit(outcome.to_dict())
    return 0


def cmd_candidate_reject(args) -> int:
    service = _build_service(args)
    outcome = service.reject_candidate(args.candidate, reason=args.reason, actor=args.actor)
    _emit(outcome.to_dict())
    return 0


def cmd_candidate_confirm_unanswerable(args) -> int:
    service = _build_service(args)
    outcome = service.confirm_unanswerable(args.candidate, actor=args.actor)
    _emit(outcome.to_dict())
    return 0


def cmd_evidence_add(args) -> int:
    service = _build_service(args)
    evidence = {
        "document_sha256": args.document_sha,
        "canonical_text_version": args.canonical_version,
        "loader_version": args.loader_version,
        "filename": args.filename,
        "page": args.page,
        "section": args.section,
        "paragraph_index": args.paragraph_index,
        "offset_scope": args.offset_scope,
        "char_start": args.char_start,
        "char_end": args.char_end,
        "anchor_text": args.anchor or [],
    }
    outcome = service.add_evidence(args.candidate, evidence, actor=args.actor)
    _emit(outcome.to_dict())
    return 0


def cmd_evidence_remove(args) -> int:
    service = _build_service(args)
    outcome = service.remove_evidence(args.candidate, args.index, actor=args.actor)
    _emit(outcome.to_dict())
    return 0


def cmd_evidence_validate(args) -> int:
    service = _build_service(args)
    result = service.validate_candidate_evidence(args.candidate)
    _emit(result.to_dict())
    return 0 if result.valid else 1


def cmd_evidence_find(args) -> int:
    """定位輔助：只回傳候選位置，需人工確認後才可作為 Evidence。"""
    from evaluation.v11.review.evidence_annotation import find_anchor_candidates

    service = _build_service(args)
    matches = find_anchor_candidates(
        service.corpus, args.anchor, document_sha256=args.document_sha
    )
    _emit({"anchor": args.anchor, "candidates": [m.to_dict() for m in matches]})
    return 0


def cmd_coverage_show(args) -> int:
    service = _build_service(args)
    coverage = service.coverage(args.session, include_accepted=not args.base_only)
    _emit(coverage.to_dict())
    return 0


def cmd_dataset_publish(args) -> int:
    service = _build_service(args)
    result = service.publish(
        args.session,
        new_dataset_version=args.version,
        output_path=args.out,
        actor=args.actor,
    )
    _emit(result.to_dict())
    return 0


def cmd_report(args) -> int:
    service = _build_service(args)
    json_path, md_path = write_report(
        service,
        args.session,
        args.review_output,
        new_dataset_version=args.version,
        output_path=args.out,
    )
    _emit({"review_report_json": json_path, "review_report_markdown": md_path})
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review",
        description="v1.1 Slice 6 Human Review 與 Dataset Authoring Workflow",
    )
    parser.add_argument("--db", default=DEFAULT_REVIEW_DB, help="Review SQLite 路徑（可注入）")
    parser.add_argument(
        "--corpus", action="append", help="corpus 目錄（可重複；預設 sample_docs + fixtures）"
    )
    parser.add_argument("--actor", default=DEFAULT_EDITOR, help="操作者（Slice 6 不做真實登入）")
    sub = parser.add_subparsers(dest="group", required=True)

    # --- session ---
    p_session = sub.add_parser("session").add_subparsers(dest="action", required=True)
    p = p_session.add_parser("create")
    p.add_argument("--name", required=True)
    p.add_argument("--base-dataset", required=True)
    p.add_argument("--generator-type")
    p.add_argument("--generator-model")
    p.set_defaults(func=cmd_session_create)

    p = p_session.add_parser("show")
    p.add_argument("--session", required=True)
    p.set_defaults(func=cmd_session_show)

    p = p_session.add_parser("list")
    p.set_defaults(func=cmd_session_list)

    p = p_session.add_parser("validate")
    p.add_argument("--session", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_session_validate)

    # --- candidate ---
    p_cand = sub.add_parser("candidate").add_subparsers(dest="action", required=True)
    p = p_cand.add_parser("generate")
    p.add_argument("--session", required=True)
    p.add_argument("--document-sha", required=True)
    p.add_argument("--segment", action="append")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--generator", default="fake", choices=("fake", "llm"))
    p.set_defaults(func=cmd_candidate_generate)

    p = p_cand.add_parser("add")
    p.add_argument("--session", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--query-type", required=True)
    p.add_argument("--unanswerable", action="store_true")
    p.add_argument("--document-sha")
    p.add_argument("--segment", action="append")
    p.add_argument("--group")
    p.add_argument("--split")
    p.set_defaults(func=cmd_candidate_add)

    p = p_cand.add_parser("list")
    p.add_argument("--session", required=True)
    p.add_argument("--status")
    p.set_defaults(func=cmd_candidate_list)

    p = p_cand.add_parser("show")
    p.add_argument("--candidate", required=True)
    p.set_defaults(func=cmd_candidate_show)

    p = p_cand.add_parser("edit")
    p.add_argument("--candidate", required=True)
    p.add_argument("--question")
    p.add_argument("--query-type")
    p.add_argument("--answerable", choices=("true", "false"))
    p.add_argument("--group")
    p.add_argument("--split")
    p.add_argument("--case-id")
    p.add_argument("--reason")
    p.set_defaults(func=cmd_candidate_edit)

    p = p_cand.add_parser("accept")
    p.add_argument("--candidate", required=True)
    p.add_argument("--group")
    p.add_argument("--split")
    p.add_argument("--case-id")
    p.add_argument("--reason")
    p.set_defaults(func=cmd_candidate_accept)

    p = p_cand.add_parser("reject")
    p.add_argument("--candidate", required=True)
    p.add_argument("--reason")
    p.set_defaults(func=cmd_candidate_reject)

    p = p_cand.add_parser("confirm-unanswerable")
    p.add_argument("--candidate", required=True)
    p.set_defaults(func=cmd_candidate_confirm_unanswerable)

    # --- evidence ---
    p_ev = sub.add_parser("evidence").add_subparsers(dest="action", required=True)
    p = p_ev.add_parser("add")
    p.add_argument("--candidate", required=True)
    p.add_argument("--document-sha", required=True)
    p.add_argument("--canonical-version", required=True)
    p.add_argument("--loader-version")
    p.add_argument("--filename")
    p.add_argument("--page", type=int)
    p.add_argument("--section")
    p.add_argument("--paragraph-index", type=int)
    p.add_argument("--offset-scope", choices=("page", "document"))
    p.add_argument("--char-start", type=int, required=True)
    p.add_argument("--char-end", type=int, required=True)
    p.add_argument("--anchor", action="append")
    p.set_defaults(func=cmd_evidence_add)

    p = p_ev.add_parser("remove")
    p.add_argument("--candidate", required=True)
    p.add_argument("--index", type=int, required=True)
    p.set_defaults(func=cmd_evidence_remove)

    p = p_ev.add_parser("validate")
    p.add_argument("--candidate", required=True)
    p.set_defaults(func=cmd_evidence_validate)

    p = p_ev.add_parser("find")
    p.add_argument("--anchor", required=True)
    p.add_argument("--document-sha")
    p.set_defaults(func=cmd_evidence_find)

    # --- coverage ---
    p_cov = sub.add_parser("coverage").add_subparsers(dest="action", required=True)
    p = p_cov.add_parser("show")
    p.add_argument("--session", required=True)
    p.add_argument("--base-only", action="store_true")
    p.set_defaults(func=cmd_coverage_show)

    # --- dataset ---
    p_ds = sub.add_parser("dataset").add_subparsers(dest="action", required=True)
    p = p_ds.add_parser("publish")
    p.add_argument("--session", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--out", required=True, help="明確輸出路徑；已存在即拒絕")
    p.set_defaults(func=cmd_dataset_publish)

    # --- report ---
    p_rep = sub.add_parser("report").add_subparsers(dest="action", required=True)
    p = p_rep.add_parser("write")
    p.add_argument("--session", required=True)
    p.add_argument("--review-output", default=DEFAULT_REVIEW_OUTPUT)
    p.add_argument("--version")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - CLI 回報明確錯誤，不外洩 traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
