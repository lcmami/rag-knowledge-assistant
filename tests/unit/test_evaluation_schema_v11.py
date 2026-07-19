"""v1.1 Evaluation Schema + Optimizer SQLite 單元測試（Slice 1 §十三 Evaluation Schema）。"""
from __future__ import annotations

import sqlite3

import pytest

from evaluation.v11.case_schema import (
    CaseSchemaError,
    parse_dataset,
)
from evaluation.v11.optimizer_repository import (
    MigrationError,
    OptimizerRepository,
    run_migrations,
)


def _evidence(**over):
    base = {
        "document_sha256": "sha-a",
        "canonical_text_version": "v1",
        "loader_version": "v1",
        "section": "E102 感測器通訊異常",
        "offset_scope": "document",
        "char_start": 0,
        "char_end": 100,
        "anchor_text": ["E102"],
    }
    base.update(over)
    return base


def _case(**over):
    base = {
        "id": "eval-1",
        "question": "E102 是什麼？",
        "query_type": "exact_code",
        "answerable": True,
        "group_id": "g1",
        "split": "tune",
        "expected_evidence": [_evidence()],
    }
    base.update(over)
    return base


def _dataset(cases, dataset_version="v1.1-test"):
    return {"schema_version": "1.1", "dataset_version": dataset_version, "cases": cases}


# --- schema 驗證 ---
def test_valid_answerable_case():
    ds = parse_dataset(_dataset([_case()]))
    assert len(ds.cases) == 1
    assert ds.cases[0].answerable is True


def test_valid_unanswerable_case_needs_no_evidence():
    ds = parse_dataset(
        _dataset([_case(id="u1", query_type="unanswerable", answerable=False, expected_evidence=[])])
    )
    assert ds.cases[0].expected_evidence == []


def test_multiple_evidence_spans():
    c = _case(expected_evidence=[_evidence(), _evidence(section="警報代碼對照表", char_start=200, char_end=300)])
    ds = parse_dataset(_dataset([c]))
    assert len(ds.cases[0].expected_evidence) == 2


def test_duplicate_case_id_rejected():
    with pytest.raises(CaseSchemaError, match="重複的 case id"):
        parse_dataset(_dataset([_case(id="dup"), _case(id="dup")]))


def test_group_split_leakage_rejected():
    with pytest.raises(CaseSchemaError, match="leakage"):
        parse_dataset(
            _dataset(
                [
                    _case(id="a", group_id="g", split="tune"),
                    _case(id="b", group_id="g", split="test"),
                ]
            )
        )


def test_missing_anchor_on_answerable_rejected():
    with pytest.raises(CaseSchemaError, match="anchor_text"):
        parse_dataset(_dataset([_case(expected_evidence=[_evidence(anchor_text=[])])]))


def test_missing_canonical_version_rejected():
    with pytest.raises(CaseSchemaError, match="canonical_text_version"):
        parse_dataset(_dataset([_case(expected_evidence=[_evidence(canonical_text_version="")])]))


def test_missing_document_sha_rejected():
    with pytest.raises(CaseSchemaError, match="document_sha256"):
        parse_dataset(_dataset([_case(expected_evidence=[_evidence(document_sha256="")])]))


def test_illegal_split_rejected():
    with pytest.raises(CaseSchemaError, match="split"):
        parse_dataset(_dataset([_case(split="train")]))


def test_answerable_without_evidence_rejected():
    with pytest.raises(CaseSchemaError, match="至少需一個"):
        parse_dataset(_dataset([_case(expected_evidence=[])]))


def test_empty_dataset_version_rejected():
    with pytest.raises(CaseSchemaError, match="dataset_version"):
        parse_dataset(_dataset([_case()], dataset_version=""))


# --- SQLite import/export round-trip ---
def test_json_sqlite_round_trip(tmp_path):
    ds = parse_dataset(
        _dataset(
            [
                _case(id="a", group_id="g1", split="tune"),
                _case(id="u", query_type="unanswerable", answerable=False, group_id="g2", split="test", expected_evidence=[]),
                _case(id="m", group_id="g3", split="validation",
                      expected_evidence=[_evidence(), _evidence(section="警報代碼對照表", char_start=200, char_end=300, anchor_text=["代碼"])]),
            ]
        )
    )
    repo = OptimizerRepository(str(tmp_path / "optimizer.db"))
    repo.import_dataset(ds)
    exported = repo.export_dataset("v1.1-test")

    def norm(d):
        return {c["id"]: c for c in d.to_dict()["cases"]}

    assert norm(exported) == norm(ds)


def test_does_not_touch_production_metadata_db(tmp_path):
    # Optimizer DB 與正式 registry 完全分離：只建立 evaluation 相關表，無 documents 表
    repo = OptimizerRepository(str(tmp_path / "optimizer.db"))
    conn = sqlite3.connect(str(tmp_path / "optimizer.db"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "evaluation_cases" in tables
    assert "schema_migrations" in tables
    assert "documents" not in tables


# --- migration ---
def test_migration_idempotent(tmp_path):
    db = str(tmp_path / "optimizer.db")
    repo = OptimizerRepository(db)
    assert repo.applied_migrations() == ["0001_initial"]
    # 再建立一次 repo（等於再跑一次 migration）→ 不重複套用
    conn = sqlite3.connect(db, isolation_level=None)
    newly = run_migrations(conn)
    conn.close()
    assert newly == []


def test_migration_failure_rolls_back(tmp_path):
    db = str(tmp_path / "optimizer.db")
    bad = [
        ("0001_ok", ["CREATE TABLE t_ok (id INTEGER)"]),
        ("0002_bad", ["CREATE TABLE t_partial (id INTEGER)", "THIS IS NOT SQL"]),
    ]
    conn = sqlite3.connect(db, isolation_level=None)
    conn.row_factory = sqlite3.Row
    with pytest.raises(MigrationError):
        run_migrations(conn, bad)

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    conn.close()

    # 第一個 migration 成功；第二個失敗整體回滾：t_partial 不存在、0002_bad 未記錄
    assert "t_ok" in tables
    assert "t_partial" not in tables
    assert applied == {"0001_ok"}
