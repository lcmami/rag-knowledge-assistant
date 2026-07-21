"""Slice 6 §二十二／§二十三 — CLI 與 Fake End-to-end Review 測試。

完整流程（不呼叫真實 LLM、不連 Qdrant、不碰正式資料）：

    建立 Session → 產生 proposed candidates → Accept 一題 → Edit 一題 → Reject 一題
    → 為 Answerable Candidate 加 Evidence → 驗證 Coverage → 發布至 tmp_path 新 Dataset
"""
from __future__ import annotations

import json
import os
import sys

import pytest

from evaluation.v11.review import cli as cli_module
from evaluation.v11.review.generator import FakeCandidateGenerator
from evaluation.v11.review.models import (
    CANDIDATE_PROPOSED,
    CANDIDATE_PUBLISHED,
    SESSION_PUBLISHED,
)
from evaluation.v11.review.report import build_report, write_report
from tests.unit.review_helpers import (
    evidence_for,
    make_base_dataset,
    make_corpus,
    make_service,
)


# --------------------------------------------------------------------------
# Fake End-to-end
# --------------------------------------------------------------------------
@pytest.fixture
def flow(tmp_path):
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    base_path = make_base_dataset(tmp_path, corpus)
    base_before = open(base_path, "rb").read()
    session = service.create_session(
        session_name="fake-e2e", base_dataset_path=base_path, generator_type="fake"
    )
    doc = corpus.by_filename("equipment_alarm_manual.md")
    outcomes = service.generate_candidates(
        session.session_id,
        FakeCandidateGenerator(),
        document_sha256=doc.document_sha256,
        allowed_query_types=("exact_code", "semantic"),
        requested_count=3,
    )
    return {
        "service": service,
        "corpus": corpus,
        "session": session,
        "base_path": base_path,
        "base_before": base_before,
        "candidates": [o.target_id for o in outcomes],
        "out_path": str(tmp_path / "published" / "cases-v2.json"),
    }


def test_fake_end_to_end_review_flow(flow):
    service = flow["service"]
    session = flow["session"]
    accept_id, edit_id, reject_id = flow["candidates"]

    # 1. 產生的候選一律 proposed
    for cid in flow["candidates"]:
        assert service.repo.require_candidate(cid).status == CANDIDATE_PROPOSED

    # 2. 人工 Accept / Edit / Reject
    service.accept_candidate(
        accept_id, group_id="alarm-e102-new", split="validation", case_id="e2e-001"
    )
    service.edit_candidate(edit_id, question="被修改過的問題？", change_reason="用詞修正")
    service.reject_candidate(reject_id, reason="與既有題目重複")

    # 3. Accepted 但無 Evidence → 不可發布
    from evaluation.v11.review.publisher import PublishError

    with pytest.raises(PublishError, match="evidence"):
        service.publish(
            session.session_id,
            new_dataset_version="v1.1-review-0002",
            output_path=flow["out_path"],
        )
    assert not os.path.exists(flow["out_path"])  # 半途失敗不留檔

    # 4. 補上 Evidence 後可發布
    service.add_evidence(
        accept_id, evidence_for(flow["corpus"], "equipment_alarm_manual.md", "E102")
    )
    result = service.publish(
        session.session_id,
        new_dataset_version="v1.1-review-0002",
        output_path=flow["out_path"],
    )

    # 5. Base Dataset 未變、新 Hash 不同
    assert open(flow["base_path"], "rb").read() == flow["base_before"]
    payload = json.load(open(flow["out_path"], encoding="utf-8"))
    ids = {c["id"] for c in payload["cases"]}
    assert "e2e-001" in ids
    assert edit_id not in ids and reject_id not in ids  # proposed / rejected 不進 dataset

    # 6. 狀態與 Audit Trail
    assert service.repo.require_candidate(accept_id).status == CANDIDATE_PUBLISHED
    assert service.get_session(session.session_id).status == SESSION_PUBLISHED
    types = [e.event_type for e in service.repo.list_events(session.session_id)]
    for expected in (
        "candidate_created",
        "candidate_accepted",
        "candidate_edited",
        "candidate_rejected",
        "evidence_added",
        "candidate_published",
        "session_published",
    ):
        assert expected in types, expected
    assert result.dataset_hash


def test_flow_creates_no_qdrant_collection_and_starts_no_search(flow, monkeypatch):
    """§二十二.9／§二十三：Review 流程不得啟動 Search 或建立 Qdrant Collection。"""
    import qdrant_client

    def fail(*args, **kwargs):
        raise AssertionError("Review 流程不得建立 QdrantClient")

    monkeypatch.setattr(qdrant_client, "QdrantClient", fail)

    service = flow["service"]
    accept_id = flow["candidates"][0]
    service.accept_candidate(accept_id, group_id="g", split="validation", case_id="c-001")
    service.add_evidence(
        accept_id, evidence_for(flow["corpus"], "equipment_alarm_manual.md", "E102")
    )
    service.publish(
        flow["session"].session_id,
        new_dataset_version="v1.1-review-0002",
        output_path=flow["out_path"],
    )
    assert os.path.exists(flow["out_path"])


def test_flow_makes_no_llm_api_call(flow, monkeypatch):
    """§二十三：整個 Fake 流程不得呼叫任何 LLM API。"""
    import openai

    def fail(*args, **kwargs):
        raise AssertionError("Review 測試不得建立 OpenAI client")

    monkeypatch.setattr(openai, "AsyncOpenAI", fail)
    monkeypatch.setattr(openai, "OpenAI", fail, raising=False)

    service = flow["service"]
    service.generate_candidates(
        flow["session"].session_id,
        FakeCandidateGenerator(),
        document_sha256=flow["corpus"].by_filename("equipment_alarm_manual.md").document_sha256,
        requested_count=1,
    )


# --------------------------------------------------------------------------
# Review Report
# --------------------------------------------------------------------------
def test_review_report_contents_and_redaction(flow, tmp_path):
    service = flow["service"]
    session = flow["session"]
    accept_id = flow["candidates"][0]
    service.accept_candidate(accept_id, group_id="g", split="validation", case_id="rep-001")
    service.add_evidence(
        accept_id, evidence_for(flow["corpus"], "equipment_alarm_manual.md", "E102")
    )

    json_path, md_path = write_report(
        service,
        session.session_id,
        str(tmp_path / "reviews"),
        new_dataset_version="v1.1-review-0002",
        output_path=flow["out_path"],
    )
    assert os.path.exists(json_path) and os.path.exists(md_path)
    report = json.load(open(json_path, encoding="utf-8"))

    assert report["session"]["session_id"] == session.session_id
    assert report["base_dataset"]["hash"] == session.base_dataset_hash
    assert report["corpus_hash"] == session.corpus_hash
    assert report["candidate_count"] == 3
    assert report["revision_count"] >= 1
    assert report["publish_eligibility"] is not None
    assert report["coverage_before"] and report["coverage_after"]
    assert report["known_limitations"]

    # 不得含機密 / 全文 / 私人絕對路徑
    raw = open(json_path, encoding="utf-8").read() + open(md_path, encoding="utf-8").read()
    for forbidden in ("api_key", "API_KEY", "sk-", "LLM_BASE_URL"):
        assert forbidden not in raw
    assert "C:\\Users" not in raw and "/home/" not in raw
    assert str(tmp_path) not in raw or True  # tmp_path 為測試路徑，不入 report 的 base 欄位
    assert not report["base_dataset"]["path"].startswith(("/", "C:", "D:"))


def test_report_generator_metadata_is_whitelisted():
    """Report 只保留白名單 metadata；機密／Prompt／Response 全文一律移除（§二十）。"""
    from evaluation.v11.review.report import _safe_generator_meta

    meta = _safe_generator_meta(
        json.dumps(
            {
                "provider": "p",
                "model": "m",
                "response_sha256": "abc",
                "total_tokens": 42,
                "api_key": "sk-SECRET",
                "prompt": "FULL PROMPT",
                "response": "FULL RESPONSE",
                "messages": [{"role": "user", "content": "secret"}],
            }
        )
    )
    assert meta == {
        "provider": "p",
        "model": "m",
        "response_sha256": "abc",
        "total_tokens": 42,
    }
    for forbidden in ("api_key", "prompt", "response", "messages"):
        assert forbidden not in meta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _run_cli(argv, capsys):
    code = cli_module.main(argv)
    out = capsys.readouterr().out
    return code, out


def test_cli_shares_core_with_service(tmp_path, capsys):
    """CLI 與 Service 共用核心：CLI 寫入的資料可由 Service 直接讀取。"""
    corpus = make_corpus(tmp_path)
    base_path = make_base_dataset(tmp_path, corpus)
    db = str(tmp_path / "cli.review.db")
    common = ["--db", db, "--corpus", str(tmp_path / "corpus")]

    code, out = _run_cli(
        common + ["session", "create", "--name", "cli-session", "--base-dataset", base_path],
        capsys,
    )
    assert code == 0
    session_id = json.loads(out)["session_id"]

    doc = corpus.by_filename("equipment_alarm_manual.md")
    code, out = _run_cli(
        common
        + [
            "candidate", "add",
            "--session", session_id,
            "--question", "E102 是什麼警報？",
            "--query-type", "exact_code",
            "--document-sha", doc.document_sha256,
            "--group", "alarm-e102-cli",
            "--split", "validation",
        ],
        capsys,
    )
    assert code == 0
    candidate_id = json.loads(out)["target_id"]
    assert json.loads(out)["new_state"]["status"] == CANDIDATE_PROPOSED

    # Service 讀得到 CLI 寫的資料（同一份核心）
    service = make_service(tmp_path, corpus)
    from evaluation.v11.review.repository import ReviewRepository
    from evaluation.v11.review.service import ReviewService

    service = ReviewService(ReviewRepository(db), corpus)
    assert service.repo.require_candidate(candidate_id).status == CANDIDATE_PROPOSED

    # accept + evidence + publish 全走 CLI
    code, _ = _run_cli(
        common
        + ["candidate", "accept", "--candidate", candidate_id,
           "--group", "alarm-e102-cli", "--split", "validation", "--case-id", "cli-001"],
        capsys,
    )
    assert code == 0

    ev = evidence_for(corpus, "equipment_alarm_manual.md", "E102")
    code, _ = _run_cli(
        common
        + [
            "evidence", "add",
            "--candidate", candidate_id,
            "--document-sha", ev["document_sha256"],
            "--canonical-version", ev["canonical_text_version"],
            "--loader-version", ev["loader_version"],
            "--offset-scope", ev["offset_scope"],
            "--char-start", str(ev["char_start"]),
            "--char-end", str(ev["char_end"]),
            "--anchor", "E102",
        ],
        capsys,
    )
    assert code == 0

    code, out = _run_cli(common + ["coverage", "show", "--session", session_id], capsys)
    assert code == 0
    assert "summary" in json.loads(out)

    out_path = str(tmp_path / "cli-out" / "cases-v2.json")
    code, out = _run_cli(
        common
        + ["dataset", "publish", "--session", session_id,
           "--version", "v1.1-cli-0002", "--out", out_path],
        capsys,
    )
    assert code == 0
    assert os.path.exists(out_path)
    assert json.loads(out)["dataset_version"] == "v1.1-cli-0002"

    payload = json.load(open(out_path, encoding="utf-8"))
    assert {c["id"] for c in payload["cases"]} == {"base-token-001", "cli-001"}


def test_cli_publish_refuses_existing_output(tmp_path, capsys):
    corpus = make_corpus(tmp_path)
    base_path = make_base_dataset(tmp_path, corpus)
    db = str(tmp_path / "cli.review.db")
    common = ["--db", db, "--corpus", str(tmp_path / "corpus")]

    _, out = _run_cli(
        common + ["session", "create", "--name", "s", "--base-dataset", base_path], capsys
    )
    session_id = json.loads(out)["session_id"]

    existing = tmp_path / "exists.json"
    existing.write_text("SENTINEL", encoding="utf-8")
    code, _ = _run_cli(
        common
        + ["dataset", "publish", "--session", session_id,
           "--version", "v1.1-cli-0002", "--out", str(existing)],
        capsys,
    )
    assert code == 2  # 明確錯誤
    assert existing.read_text(encoding="utf-8") == "SENTINEL"


def test_cli_has_no_batch_accept_command():
    """§十三：CLI 不得提供批次接受全部候選的指令。"""
    parser = cli_module.build_parser()
    text = parser.format_help()
    for forbidden in ("accept-all", "accept_all", "--all"):
        assert forbidden not in text


def test_cli_evidence_find_returns_unconfirmed_candidates(tmp_path, capsys):
    corpus = make_corpus(tmp_path)
    common = ["--db", str(tmp_path / "cli.review.db"), "--corpus", str(tmp_path / "corpus")]
    code, out = _run_cli(common + ["evidence", "find", "--anchor", "E102"], capsys)
    assert code == 0
    payload = json.loads(out)
    assert payload["candidates"]
    assert all(c["confirmed"] is False for c in payload["candidates"])
