"""Slice 5A §十 —— Dataset Coverage Revision 驗證。

證明新版 dataset 讓 validation 能實際計算全部 safety constraint，同時：

- 原 Slice 1 dataset **完全未變**，且仍會誠實觸發 ``insufficient_validation_coverage``
  （它現在是 Coverage Guard 的迴歸 fixture）。
- Test split 的覆蓋沒有被犧牲。
- 所有 evidence offset 都指向**實際 canonical text**，anchor 真的出現在該段文字中。
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import tempfile

import pytest

from evaluation.v11.case_schema import parse_dataset
from evaluation.v11.hashing import dataset_hash, repo_root
from evaluation.v11.search import split_dataset as sd
from evaluation.v11.search.search_config import SearchConfig

V1_PATH = os.path.join(repo_root(), "evaluation", "datasets", "v1.1", "cases.json")
V2_PATH = os.path.join(repo_root(), "evaluation", "datasets", "v1.1", "cases-v2.json")
CFG_V1 = os.path.join(repo_root(), "evaluation", "configs", "slice5-optimizer-smoke.json")
CFG_V2 = os.path.join(repo_root(), "evaluation", "configs", "slice5-optimizer-smoke-v2.json")

CORPUS = {
    "equipment_maintenance_manual.pdf": ("sample_docs/equipment_maintenance_manual.pdf", "pdf"),
    "equipment_alarm_manual.md": ("sample_docs/equipment_alarm_manual.md", "markdown"),
    "confusable_alarm_d900.md": ("evaluation/fixtures/confusable_alarm_d900.md", "markdown"),
    "mes_operation_sop.md": ("sample_docs/mes_operation_sop.md", "markdown"),
    "api_integration_spec.md": ("sample_docs/api_integration_spec.md", "markdown"),
}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def canonical_docs():
    from app.services.canonical_builder import build_canonical_document
    from app.services.document_loader import load_document

    out = {}
    for name, (rel, dtype) in CORPUS.items():
        with open(os.path.join(repo_root(), rel), "rb") as f:
            raw = f.read()
        sha = hashlib.sha256(raw).hexdigest()
        out[name] = (
            sha,
            build_canonical_document(
                filename=name, document_sha256=sha, document_type=dtype,
                elements=load_document(name, raw),
            ),
        )
    return out


def _covered_text(doc, page, start, end) -> str:
    """與 ``[start, end)`` 有重疊的所有 canonical segment 文字（span 可跨 segment）。"""
    parts = []
    for s in (x.to_dict() for x in doc.segments):
        if s["page"] != page:
            continue
        if s["char_start"] < end and s["char_end"] > start:
            parts.append(s["normalized_text"])
    return "\n".join(parts)


def _splits(path):
    return sd.prepare_split_datasets(path, tempfile.mkdtemp())


# --------------------------------------------------------------------------
# 舊 dataset 保持不變
# --------------------------------------------------------------------------
def test_v1_dataset_is_unchanged():
    """Slice 1 已 Commit 的 dataset 不得被修改或覆寫。"""
    import subprocess

    out = subprocess.run(
        ["git", "status", "--porcelain", "evaluation/datasets/v1.1/cases.json"],
        cwd=repo_root(), capture_output=True, text=True,
    )
    assert out.stdout.strip() == "", f"原 dataset 被改動了：{out.stdout!r}"


def test_v1_and_v2_hashes_differ():
    assert dataset_hash(_load(V1_PATH)) != dataset_hash(_load(V2_PATH))


def test_v2_declares_a_new_dataset_version():
    v1, v2 = _load(V1_PATH), _load(V2_PATH)
    assert v2["dataset_version"] != v1["dataset_version"]
    assert v2["dataset_version"] == "v1.1-coverage-0002"
    assert v2["schema_version"] == v1["schema_version"]


# --------------------------------------------------------------------------
# Schema / 完整性
# --------------------------------------------------------------------------
def test_v2_schema_valid_and_ids_unique():
    ds = parse_dataset(_load(V2_PATH))  # 內含 group leakage 驗證
    ids = [c.id for c in ds.cases]
    assert len(ids) == len(set(ids)) == 16


def test_v2_has_no_group_leakage():
    ds = parse_dataset(_load(V2_PATH))
    by = collections.defaultdict(set)
    for c in ds.cases:
        by[c.group_id].add(c.split)
    assert {g: s for g, s in by.items() if len(s) > 1} == {}


def test_v2_preserves_all_original_cases():
    """原 12 題必須逐字保留（只有 alarm-e101 的 split 改變）。"""
    v1 = {c["id"]: c for c in _load(V1_PATH)["cases"]}
    v2 = {c["id"]: c for c in _load(V2_PATH)["cases"]}
    assert set(v1) <= set(v2)
    for cid, old in v1.items():
        new = v2[cid]
        for key in ("question", "query_type", "answerable", "group_id", "expected_evidence"):
            assert new[key] == old[key], f"{cid} 的 {key} 不得被改動"
        if old["group_id"] == "alarm-e101":
            assert (old["split"], new["split"]) == ("tune", "validation")
        else:
            assert new["split"] == old["split"], f"{cid} split 不應改變"


# --------------------------------------------------------------------------
# Split coverage
# --------------------------------------------------------------------------
def test_v2_tune_coverage():
    info = _splits(V2_PATH).splits["tune"]
    qt = info.query_type_counts
    assert qt.get("exact_code", 0) >= 1
    assert qt.get("semantic", 0) >= 1
    assert any(qt.get(k, 0) >= 1 for k in ("procedure", "cross_section", "confusable"))
    # tune 沒有 unanswerable 是已知且被核准的（tune_prefilter 不因不可計算而淘汰）
    assert qt.get("unanswerable", 0) == 0


def test_v2_validation_coverage_has_all_safety_types():
    info = _splits(V2_PATH).splits["validation"]
    qt = info.query_type_counts
    assert qt.get("exact_code", 0) >= 1
    assert qt.get("unanswerable", 0) >= 1
    assert qt.get("semantic", 0) >= 1
    assert info.case_count == 5 and info.group_count == 5


def test_v2_test_coverage_is_preserved_and_not_weakened():
    v1_info = _splits(V1_PATH).splits["test"]
    v2_info = _splits(V2_PATH).splits["test"]
    qt = v2_info.query_type_counts
    for required in ("exact_code", "unanswerable", "semantic", "confusable"):
        assert qt.get(required, 0) >= 1, f"test 缺少 {required}"
    # 不得因為搬題而讓 test 變弱
    assert v2_info.case_count >= v1_info.case_count
    assert v2_info.group_count >= v1_info.group_count
    assert set(v1_info.group_ids) <= set(v2_info.group_ids), "原 test group 不得被移走"


# --------------------------------------------------------------------------
# Coverage Guard（新舊行為必須不同）
# --------------------------------------------------------------------------
def _verdict(dataset_path, config_path):
    cfg = SearchConfig.from_raw(_load(config_path))
    return sd.assess_validation_coverage(
        cfg.hard_constraints, _splits(dataset_path).splits["validation"]
    )


def test_v1_still_triggers_insufficient_validation_coverage():
    """原 dataset 現在是 Coverage Guard 的迴歸 fixture。"""
    v = _verdict(V1_PATH, CFG_V1)
    assert v.sufficient is False
    assert v.status == sd.STATUS_INSUFFICIENT_VALIDATION_COVERAGE
    assert {b.metric for b in v.blocking} == {
        "exact_code_top1_accuracy", "unanswerable_false_positive_rate",
    }


def test_v2_does_not_trigger_coverage_guard():
    v = _verdict(V2_PATH, CFG_V2)
    assert v.sufficient is True
    assert v.status == "ok"
    assert v.blocking == []


def test_v2_all_four_constraints_predicted_computable_or_unknown():
    v = _verdict(V2_PATH, CFG_V2)
    for f in v.findings:
        assert f.predicted_computable is not False, f"{f.constraint_name} 在 validation 不可計算"


# --------------------------------------------------------------------------
# 遷移的四題
# --------------------------------------------------------------------------
def _case(dataset, case_id):
    return next(c for c in parse_dataset(dataset).cases if c.id == case_id)


@pytest.mark.parametrize("case_id", ["eval-unanswerable-salary", "eval-unanswerable-stock"])
def test_c14_c15_are_valid_unanswerable(case_id):
    c = _case(_load(V2_PATH), case_id)
    assert c.answerable is False
    assert c.query_type == "unanswerable"
    assert c.expected_evidence == []
    assert c.split == "validation"


def test_c07_migrated_with_grounded_evidence(canonical_docs):
    c = _case(_load(V2_PATH), "eval-maint-filter-001")
    assert c.answerable is True and c.split == "tune"
    # 與 eval-maint-cycle-001 同一來源段落 → 必須同 group，避免洩漏
    assert c.group_id == "maint-cycle"
    assert len(c.expected_evidence) == 1
    e = c.expected_evidence[0]
    sha, doc = canonical_docs[e.filename]
    assert e.document_sha256 == sha
    text = _covered_text(doc, e.page, e.char_start, e.char_end)
    assert "空氣過濾網" in text and "250" in text
    for anchor in e.anchor_text:
        assert anchor in text, f"anchor {anchor!r} 不在實際 canonical 文字中"


def test_c10_migrated_with_multi_span_comparison_evidence(canonical_docs):
    c = _case(_load(V2_PATH), "eval-e102-comparison")
    assert c.answerable is True
    assert c.query_type == "confusable"
    # gold 來源即 alarm-e102 group 的來源 → 必須留在 test（group integrity）
    assert c.group_id == "alarm-e102" and c.split == "test"
    assert len(c.expected_evidence) == 2, "比較題必須保留 E102 與其他警報兩個 span"
    files = {e.filename for e in c.expected_evidence}
    assert files == {"equipment_alarm_manual.md", "confusable_alarm_d900.md"}
    for e in c.expected_evidence:
        sha, doc = canonical_docs[e.filename]
        assert e.document_sha256 == sha
        text = _covered_text(doc, e.page, e.char_start, e.char_end)
        assert text, "span 未涵蓋任何 canonical segment"
        for anchor in e.anchor_text:
            assert anchor in text, f"{e.filename} anchor {anchor!r} 不在實際文字中"


def test_every_answerable_evidence_offset_is_grounded(canonical_docs):
    """全 dataset 掃描：不得有任何虛構 offset。"""
    for c in parse_dataset(_load(V2_PATH)).cases:
        if not c.answerable:
            assert not c.expected_evidence
            continue
        assert c.expected_evidence, f"{c.id} 缺少 evidence"
        for e in c.expected_evidence:
            sha, doc = canonical_docs[e.filename]
            assert e.document_sha256 == sha, f"{c.id} corpus sha 不符"
            assert e.canonical_text_version == "v1"
            assert e.loader_version == "v1"
            text = _covered_text(doc, e.page, e.char_start, e.char_end)
            assert text, f"{c.id} span [{e.char_start}:{e.char_end}] 不存在於 canonical text"
            missing = [a for a in e.anchor_text if a not in text]
            assert not missing, f"{c.id} anchor 不在實際文字中：{missing}"


# --------------------------------------------------------------------------
# Config 與 resume 隔離
# --------------------------------------------------------------------------
def test_v2_config_differs_only_in_dataset_and_name():
    a, b = _load(CFG_V1), _load(CFG_V2)
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert diff == ["dataset_path", "search_name"], diff
    # 門檻與信心度不得因為知道前次結果而調整
    assert b["policy_confidence"] == "low"
    assert b["policy_source"] == "slice0_baseline_recommendation"


def test_v2_config_points_at_new_dataset():
    cfg = SearchConfig.from_raw(_load(CFG_V2))
    assert cfg.dataset_path.endswith("cases-v2.json")


def test_v1_and_v2_search_config_hashes_differ():
    """舊 search 不得被誤認為新 dataset 的 search（resume 隔離）。"""
    assert (
        SearchConfig.from_raw(_load(CFG_V1)).compute_hash()
        != SearchConfig.from_raw(_load(CFG_V2)).compute_hash()
    )


def test_old_search_cannot_resume_with_new_dataset(tmp_path):
    """以 v1 建立的 search，改用 v2 config resume 必須被拒絕。"""
    from evaluation.v11.search import search_runner
    from evaluation.v11.search.search_models import SearchResumeError
    from evaluation.v11.search.search_repository import SearchRepository
    from tests.unit.v11_fakes import FakeTrialServiceProvider, ranked

    repo = SearchRepository(str(tmp_path / "s.optimizer.db"))
    prov = FakeTrialServiceProvider(results_for=lambda t, c, q: [ranked(1)])
    out = str(tmp_path / "out")

    a = search_runner.execute_search(
        SearchConfig.from_raw(_load(CFG_V1)), repo, prov, output_dir=out
    )
    with pytest.raises(SearchResumeError) as exc:
        search_runner.execute_search(
            SearchConfig.from_raw(_load(CFG_V2)), repo, prov,
            output_dir=out, search_id=a.search.search_id,
        )
    assert exc.value.code in ("search_config_hash_mismatch", "dataset_hash_mismatch")
