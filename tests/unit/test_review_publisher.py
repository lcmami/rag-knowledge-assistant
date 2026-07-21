"""Slice 6 §二十三 — Versioned Dataset Publisher 測試。

Base/Corpus Hash 驗證、拒絕覆寫、Atomic Publish、只發布 Accepted、
舊 Dataset byte-identical、Publish Failure 不留半檔。
"""
from __future__ import annotations

import json
import os

import pytest

from evaluation.v11 import hashing
from evaluation.v11.case_schema import parse_dataset
from evaluation.v11.review.generator import FakeCandidateGenerator
from evaluation.v11.review.models import (
    CANDIDATE_ACCEPTED,
    CANDIDATE_PUBLISHED,
    SESSION_PUBLISHED,
    ReviewStateError,
)
from evaluation.v11.review.publisher import PublishError
from tests.unit.review_helpers import (
    evidence_for,
    make_base_dataset,
    make_corpus,
    make_service,
)

NEW_VERSION = "v1.1-review-0002"


@pytest.fixture
def ready(tmp_path):
    """一個已備妥、可發布的 session（1 個 accepted + evidence 完整的 candidate）。"""
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    base_path = make_base_dataset(tmp_path, corpus)
    session = service.create_session(session_name="s", base_dataset_path=base_path)
    doc = corpus.by_filename("equipment_alarm_manual.md")

    outcome = service.add_candidate_manual(
        session.session_id,
        question="E102 警報代表什麼？",
        query_type="exact_code",
        answerable=True,
        source_document_sha256=doc.document_sha256,
    )
    cid = outcome.target_id
    service.accept_candidate(cid, group_id="alarm-e102-new", split="validation", case_id="new-001")
    service.add_evidence(cid, evidence_for(corpus, "equipment_alarm_manual.md", "E102"))
    out_path = str(tmp_path / "out" / "cases-v2.json")
    return service, corpus, session, cid, base_path, out_path


# --------------------------------------------------------------------------
# 成功路徑
# --------------------------------------------------------------------------
def test_publish_writes_new_dataset_and_updates_states(ready):
    service, _corpus, session, cid, base_path, out_path = ready
    base_before = open(base_path, "rb").read()

    result = service.publish(
        session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
    )

    assert os.path.exists(out_path)
    assert result.dataset_version == NEW_VERSION
    assert result.case_count == 2  # base 1 + accepted 1
    assert result.accepted_candidate_ids == [cid]

    # §十八.14：舊 Dataset byte-for-byte 不變
    assert open(base_path, "rb").read() == base_before
    # §十八.7／§十八.8：狀態轉換
    assert service.repo.require_candidate(cid).status == CANDIDATE_PUBLISHED
    assert service.get_session(session.session_id).status == SESSION_PUBLISHED


def test_published_dataset_is_schema_valid_with_new_hash(ready):
    service, _corpus, session, _cid, base_path, out_path = ready
    result = service.publish(
        session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
    )
    payload = json.load(open(out_path, encoding="utf-8"))
    parsed = parse_dataset(payload)
    assert parsed.dataset_version == NEW_VERSION

    base_hash = hashing.dataset_hash(json.load(open(base_path, encoding="utf-8")))
    assert result.dataset_hash != base_hash  # §二十三：新 Hash 不同
    assert hashing.dataset_hash(payload) == result.dataset_hash


def test_publish_notes_record_provenance(ready):
    """§十八.9：Notes 保存 base version/hash、session id、candidate ids、corpus hash。"""
    service, _corpus, session, cid, _base_path, out_path = ready
    service.publish(session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path)
    notes = json.load(open(out_path, encoding="utf-8"))["notes"]
    assert notes["base_dataset_version"] == session.base_dataset_version
    assert notes["base_dataset_hash"] == session.base_dataset_hash
    assert notes["review_session_id"] == session.session_id
    assert notes["accepted_candidate_ids"] == [cid]
    assert notes["corpus_hash"] == session.corpus_hash
    assert notes["published_at"]


# --------------------------------------------------------------------------
# 只發布 Accepted
# --------------------------------------------------------------------------
def test_only_accepted_candidates_are_published(tmp_path):
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    base_path = make_base_dataset(tmp_path, corpus)
    session = service.create_session(session_name="s", base_dataset_path=base_path)
    doc = corpus.by_filename("equipment_alarm_manual.md")

    accepted = service.add_candidate_manual(
        session.session_id,
        question="E102 是什麼？",
        query_type="exact_code",
        answerable=True,
        source_document_sha256=doc.document_sha256,
    ).target_id
    service.accept_candidate(accepted, group_id="g-acc", split="validation", case_id="acc-001")
    service.add_evidence(accepted, evidence_for(corpus, "equipment_alarm_manual.md", "E102"))

    rejected = service.add_candidate_manual(
        session.session_id,
        question="被拒絕的問題？",
        query_type="semantic",
        answerable=True,
        source_document_sha256=doc.document_sha256,
    ).target_id
    service.reject_candidate(rejected)

    proposed = service.add_candidate_manual(
        session.session_id,
        question="還沒審的問題？",
        query_type="semantic",
        answerable=True,
        source_document_sha256=doc.document_sha256,
    ).target_id

    out_path = str(tmp_path / "cases-new.json")
    service.publish(session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path)

    ids = {c["id"] for c in json.load(open(out_path, encoding="utf-8"))["cases"]}
    assert "acc-001" in ids
    assert rejected not in ids  # §十八.6
    assert proposed not in ids
    # rejected / proposed 狀態不因發布而改變
    assert service.repo.require_candidate(rejected).status == "rejected"
    assert service.repo.require_candidate(proposed).status == "proposed"


def test_accepted_without_evidence_cannot_publish(tmp_path):
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    session = service.create_session(
        session_name="s", base_dataset_path=make_base_dataset(tmp_path, corpus)
    )
    doc = corpus.by_filename("equipment_alarm_manual.md")
    cid = service.add_candidate_manual(
        session.session_id,
        question="沒有 evidence 的問題？",
        query_type="semantic",
        answerable=True,
        source_document_sha256=doc.document_sha256,
    ).target_id
    service.accept_candidate(cid, group_id="g", split="validation")

    out_path = str(tmp_path / "cases-new.json")
    with pytest.raises(PublishError, match="evidence"):
        service.publish(
            session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
        )
    assert not os.path.exists(out_path)  # 失敗不留半檔


def test_no_accepted_candidate_cannot_publish(tmp_path):
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    session = service.create_session(
        session_name="s", base_dataset_path=make_base_dataset(tmp_path, corpus)
    )
    out_path = str(tmp_path / "cases-new.json")
    with pytest.raises(PublishError, match="accepted"):
        service.publish(
            session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
        )


# --------------------------------------------------------------------------
# 拒絕覆寫 / 版本唯一
# --------------------------------------------------------------------------
def test_existing_output_path_is_refused(ready):
    service, _corpus, session, _cid, _base_path, out_path = ready
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("SENTINEL")

    with pytest.raises(PublishError, match="已存在"):
        service.publish(
            session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
        )
    assert open(out_path, encoding="utf-8").read() == "SENTINEL"  # 未被覆寫


def test_reusing_base_dataset_version_is_refused(ready):
    service, _corpus, session, _cid, _base_path, out_path = ready
    with pytest.raises(PublishError, match="必須是新版本"):
        service.publish(
            session.session_id,
            new_dataset_version=session.base_dataset_version,
            output_path=out_path,
        )


# --------------------------------------------------------------------------
# Hash 驗證
# --------------------------------------------------------------------------
def test_changed_base_dataset_hash_blocks_publish(ready):
    """§十九.6：Session 建立後 base dataset 被改動即不得發布。"""
    service, _corpus, session, _cid, base_path, out_path = ready
    payload = json.load(open(base_path, encoding="utf-8"))
    payload["cases"][0]["question"] = "被偷改的問題？"
    with open(base_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    with pytest.raises(PublishError, match="base dataset hash"):
        service.publish(
            session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
        )
    assert not os.path.exists(out_path)


def test_changed_corpus_hash_blocks_publish(tmp_path, ready):
    """§十九.7：Session 建立後 corpus 漂移即不得發布。"""
    service, _corpus, session, _cid, _base_path, out_path = ready

    # 真的改動 corpus（新增一份文件）→ 重新載入後 corpus_hash 不同。
    (tmp_path / "corpus" / "extra_manual.md").write_text("# 新增\n\n新的內容。\n", encoding="utf-8")
    from evaluation.v11.review.corpus import load_canonical_corpus
    from evaluation.v11.review.service import ReviewService

    drifted = load_canonical_corpus([str(tmp_path / "corpus")], str(tmp_path))
    assert drifted.corpus_hash != session.corpus_hash
    drifted_service = ReviewService(service.repo, drifted)

    with pytest.raises(PublishError, match="corpus hash"):
        drifted_service.publish(
            session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
        )
    assert not os.path.exists(out_path)


def test_session_base_and_corpus_hash_are_immutable(ready):
    """§六.1：Base Dataset / Corpus Hash 建立後不得暗中改變。"""
    service, _corpus, session, _cid, _base_path, _out_path = ready
    row = service.repo.require_session(session.session_id)
    original_base, original_corpus = row.base_dataset_hash, row.corpus_hash

    row.base_dataset_hash = "tampered"
    row.corpus_hash = "tampered"
    service.repo.update_session(row)

    reloaded = service.get_session(session.session_id)
    assert reloaded.base_dataset_hash == original_base
    assert reloaded.corpus_hash == original_corpus


def test_group_leakage_blocks_publish(tmp_path):
    """新 case 的 group 與 base 的 group 撞到不同 split → 阻擋。"""
    corpus = make_corpus(tmp_path)
    service = make_service(tmp_path, corpus)
    base_path = make_base_dataset(tmp_path, corpus)
    session = service.create_session(session_name="s", base_dataset_path=base_path)
    doc = corpus.by_filename("api_integration_spec.md")

    cid = service.add_candidate_manual(
        session.session_id,
        question="Token 另一題？",
        query_type="semantic",
        answerable=True,
        source_document_sha256=doc.document_sha256,
    ).target_id
    # base 的 api-token group 在 tune；這裡故意放到 validation
    service.accept_candidate(cid, group_id="api-token", split="validation", case_id="leak-001")
    service.add_evidence(cid, evidence_for(corpus, "api_integration_spec.md", "API Token"))

    out_path = str(tmp_path / "cases-new.json")
    with pytest.raises(PublishError, match="leakage"):
        service.publish(
            session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
        )
    assert not os.path.exists(out_path)


# --------------------------------------------------------------------------
# Atomic / 失敗不留半檔
# --------------------------------------------------------------------------
def test_publish_leaves_no_temp_files(ready):
    service, _corpus, session, _cid, _base_path, out_path = ready
    service.publish(session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path)
    leftovers = [n for n in os.listdir(os.path.dirname(out_path)) if n.startswith(".publish-")]
    assert leftovers == []


def test_failed_write_leaves_no_partial_file(ready, monkeypatch):
    """os.replace 失敗時不得留下 temp 或半個目標檔（§二十三 Publish Failure）。"""
    service, _corpus, session, _cid, _base_path, out_path = ready

    def boom(src, dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        service.publish(
            session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
        )

    assert not os.path.exists(out_path)
    parent = os.path.dirname(out_path)
    assert [n for n in os.listdir(parent) if n.startswith(".publish-")] == []


def test_published_session_cannot_publish_again(ready):
    service, _corpus, session, _cid, _base_path, out_path = ready
    service.publish(session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path)
    with pytest.raises(ReviewStateError):
        service.publish(
            session.session_id,
            new_dataset_version="v1.1-review-0003",
            output_path=str(out_path) + ".other",
        )


def test_cancelled_session_cannot_publish(ready):
    service, _corpus, session, _cid, _base_path, out_path = ready
    service.cancel_session(session.session_id)
    with pytest.raises(ReviewStateError):
        service.publish(
            session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
        )


# --------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------
def test_validate_session_marks_ready_when_eligible(ready):
    service, _corpus, session, _cid, _base_path, out_path = ready
    result = service.validate_session(
        session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
    )
    assert result.eligible, result.errors
    assert service.get_session(session.session_id).status == "ready_to_publish"


def test_coverage_gaps_are_warnings_not_errors(ready):
    service, _corpus, session, _cid, _base_path, out_path = ready
    result = service.check_publish_eligibility(
        session.session_id, new_dataset_version=NEW_VERSION, output_path=out_path
    )
    assert result.eligible
    assert result.warnings  # 小樣本等 gap 仍以 warning 呈現
    assert result.errors == []
