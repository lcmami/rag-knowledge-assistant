"""Slice 4：Metric 使用 Final（admission + reranked + Top-K）結果之回歸測試（§五）。

以 deterministic fake vector scores（**不**鎖定真實模型浮點分數）串接
strategy → metrics RankedResult → evaluate_trial，證明：

- exact-code / confusable metric 讀取的是 strategy reranked Final Top-1 / Final Ranking，
  不是 raw dense / pre-admission 候選。
- 經 identifier_vector_floor admission 才進入的 exact candidate 會被 metric 納入。
- 被 floor 拒絕的候選不影響 metric。
- CX-500 售價 unanswerable：floor=0.4 產生 FP、floor=0.5 依 raw score 被拒。

relevance 由 EvidenceMatcher 的 anchor fallback 決定（chunk text 含 anchor → Level 1；否則 0），
故不需真實向量、可完全 deterministic。
"""
from __future__ import annotations

from evaluation.v11.case_schema import parse_dataset
from evaluation.v11.metric_config import DEFAULT_METRIC_CONFIG
from evaluation.v11.retrieval import (
    Candidate,
    RetrievalConfig,
    RetrievalRequest,
    build_retrieval_strategy,
)
from evaluation.v11.retrieval_metrics import CaseRetrieval, RankedResult, evaluate_trial

ANCHOR = "ANSWERTOKEN"

DENSE = {"strategy": "dense", "top_k": 5, "score_threshold": 0.55,
         "candidate_multiplier": 3, "candidate_cap": 50}
IDA = {"strategy": "identifier_aware", "top_k": 5, "score_threshold": 0.55,
       "candidate_multiplier": 3, "candidate_cap": 50, "identifier_candidate_cap": 20,
       "exact_match_boost": 0.2, "confusable_penalty": 0.15, "identifier_vector_floor": 0.4}


class _Backend:
    def __init__(self, dense=None, identifier=None):
        self._d = list(dense or [])
        self._i = list(identifier or [])

    def dense_search(self, qv, limit):
        return [_clone(c) for c in self._d[:limit]]

    def identifier_search(self, qv, ids, limit):
        return [_clone(c) for c in self._i[:limit]]


# 同文件（sha1）的 provenance span，offset 不與 gold[0:10] 重疊 → 落到 anchor fallback。
_PROV_SAME_DOC = [{"document_sha256": "sha1", "canonical_text_version": "v1", "loader_version": "v1",
                   "section": "other", "offset_scope": "document", "char_start": 100, "char_end": 110}]


def _clone(c):
    return Candidate(document_id=c.document_id, chunk_index=c.chunk_index, text=c.text,
                     vector_score=c.vector_score, identifiers=list(c.identifiers),
                     file_hash=c.file_hash, section=c.section,
                     provenance_spans=list(c.provenance_spans), sources=list(c.sources))


def C(doc_id, ci, score, ids, text="", section="sec", file_hash="sha1", provenance=None):
    return Candidate(document_id=doc_id, chunk_index=ci, text=text, vector_score=score,
                     identifiers=list(ids), section=section, file_hash=file_hash,
                     provenance_spans=list(provenance or []))


def _to_ranked(outcome):
    """mirror real_services.retrieve：strategy 結果 → metrics.RankedResult（score=raw vector）。"""
    return [
        RankedResult(
            rank=r.rank, document_id=r.document_id, document_sha256=r.file_hash,
            filename=r.filename, section=r.section, page=r.page, chunk_index=r.chunk_index,
            score=r.vector_score, text=r.text, provenance_spans=list(r.provenance_spans),
            identifiers=list(r.identifiers), canonical_text_version="v1",
            ranking_score=r.ranking_score, exact_identifier_matches=list(r.exact_identifier_matches),
            conflicting_identifiers=list(r.conflicting_identifiers),
            exact_match_boost=r.exact_match_boost, confusable_penalty=r.confusable_penalty,
            candidate_sources=list(r.candidate_sources), admission_reason=r.admission_reason,
        )
        for r in outcome.results
    ]


def _run(strategy_name, config_raw, question, backend):
    cfg = RetrievalConfig.from_raw(config_raw)
    out = build_retrieval_strategy(strategy_name).retrieve(
        RetrievalRequest(query=question, query_vector=[0.1], collection_name="c", config=cfg),
        backend,
    )
    return _to_ranked(out)


def _dataset(cases):
    return parse_dataset({"schema_version": "1.1", "dataset_version": "v-t", "cases": cases})


def _exact_case(**over):
    base = {
        "id": "c-e102", "question": "E102 warning 如何處理", "query_type": "exact_code",
        "answerable": True, "group_id": "g1", "split": "tune",
        "expected_evidence": [{
            "document_sha256": "sha1", "canonical_text_version": "v1", "loader_version": "v1",
            "section": "sec", "offset_scope": "document", "char_start": 0, "char_end": 10,
            "anchor_text": [ANCHOR]}],
    }
    base.update(over)
    return base


def _metrics(dataset, ranked_by_case):
    crs = [CaseRetrieval(case=c, results=ranked_by_case[c.id], latency_ms=1) for c in dataset.cases]
    ev = evaluate_trial(crs, index_universe=None, metric_config=DEFAULT_METRIC_CONFIG)
    return {m.metric_name: m for m in ev.metrics}


# ---- §五.1 / §五.2：exact-code 與 confusable metric 使用 Final（reranked）結果 ----
def test_exact_and_confusable_metrics_use_reranked_final_results():
    ds = _dataset([_exact_case()])
    # 相同候選：dense top-1 = 錯誤 D900(0.70, 無 anchor→rel0)；正確 E102(0.60, 含 anchor→rel1)。
    dense = [C("wrong", 0, 0.70, ["D900"], text="generic"),
             C("ans", 1, 0.60, ["E102"], text=f"... {ANCHOR} ...", provenance=_PROV_SAME_DOC)]
    ident = [C("ans", 1, 0.60, ["E102"], text=f"... {ANCHOR} ...", provenance=_PROV_SAME_DOC)]

    dense_ranked = _run("dense", DENSE, ds.cases[0].question, _Backend(dense=list(dense)))
    ida_ranked = _run("identifier_aware", IDA, ds.cases[0].question, _Backend(dense=list(dense), identifier=list(ident)))

    # dense final top-1 = wrong(rel0) → exact-code fail；ida rerank 後 ans(rel1) 為 top-1 → success。
    md = _metrics(ds, {"c-e102": dense_ranked})
    mi = _metrics(ds, {"c-e102": ida_ranked})
    assert md["exact_code_top1_accuracy"].value == 0.0
    assert mi["exact_code_top1_accuracy"].value == 1.0
    assert md["exact_code_top1_accuracy"].denominator == 1.0  # 分母 = exact_code answerable 數

    # confusable：dense 的 D900(rel0, conflicting) 排在 frr 之前 → error；ida rerank 後不在 frr 前 → 0。
    assert md["confusable_code_error_count"].value == 1.0
    assert mi["confusable_code_error_count"].value == 0.0


# ---- §五.3：經 floor admission 才進入的 exact candidate 被 metric 納入 ----
def test_floor_admitted_candidate_counted_in_metric():
    ds = _dataset([_exact_case()])
    # 正確 E102 raw=0.50 < threshold 0.55，僅來自 identifier search；靠 floor(0.4) admission。
    ident = [C("ans", 1, 0.50, ["E102"], text=f"{ANCHOR} here", provenance=_PROV_SAME_DOC)]
    ranked = _run("identifier_aware", IDA, ds.cases[0].question, _Backend(dense=[], identifier=list(ident)))
    assert ranked and ranked[0].admission_reason == "exact_identifier_floor"
    m = _metrics(ds, {"c-e102": ranked})
    assert m["exact_code_top1_accuracy"].value == 1.0  # floor-admitted 候選確實被 metric 採計


# ---- §五.4：被 floor 拒絕的候選不影響 metric ----
def test_floor_rejected_candidate_excluded_from_metric():
    ds = _dataset([_exact_case()])
    # 正確 E102 raw=0.30 < floor 0.4 → 拒絕 → 不進 Final results。
    ident = [C("ans", 1, 0.30, ["E102"], text=f"{ANCHOR} here", provenance=_PROV_SAME_DOC)]
    ranked = _run("identifier_aware", IDA, ds.cases[0].question, _Backend(dense=[], identifier=list(ident)))
    assert ranked == []  # 被 floor 拒絕
    m = _metrics(ds, {"c-e102": ranked})
    assert m["exact_code_top1_accuracy"].value == 0.0  # 未被採計（miss）


# ---- §五.5 / §五.6：CX-500 售價 unanswerable —— floor 0.4 產生 FP、0.5 被拒 ----
def _cx_dataset():
    return _dataset([{
        "id": "c-cx", "question": "CX-500 的售價是多少", "query_type": "unanswerable",
        "answerable": False, "group_id": "g", "split": "test", "expected_evidence": [],
    }])


def test_cx500_price_false_positive_at_floor_0_4():
    ds = _cx_dataset()
    # CX500 維護 chunk raw=0.4986（deterministic fake）：< threshold 0.55，>= floor 0.4。
    ident = [C("maint", 0, 0.4986, ["CX500"], text="保養週期說明", provenance=_PROV_SAME_DOC)]
    ranked = _run("identifier_aware", {**IDA, "identifier_vector_floor": 0.4},
                  ds.cases[0].question, _Backend(dense=[], identifier=list(ident)))
    assert ranked and ranked[0].admission_reason == "exact_identifier_floor"
    m = _metrics(ds, {"c-cx": ranked})
    assert m["unanswerable_false_positive_count"].value == 1.0  # FP 出現


def test_cx500_price_rejected_at_floor_0_5():
    ds = _cx_dataset()
    ident = [C("maint", 0, 0.4986, ["CX500"], text="保養週期說明", provenance=_PROV_SAME_DOC)]
    ranked = _run("identifier_aware", {**IDA, "identifier_vector_floor": 0.5},
                  ds.cases[0].question, _Backend(dense=[], identifier=list(ident)))
    assert ranked == []  # raw 0.4986 < floor 0.5 → 拒絕
    m = _metrics(ds, {"c-cx": ranked})
    assert m["unanswerable_false_positive_count"].value == 0.0  # 無 FP
