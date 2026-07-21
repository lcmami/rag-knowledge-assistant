"""Slice 5 search 測試共用工具。

提供最小可用的 search config builder，讓各測試只覆寫自己關心的欄位。
"""
from __future__ import annotations

import json
import os
from typing import Any

from evaluation.v11.search.search_config import SearchConfig

MINIMAL_POLICY: dict[str, Any] = {
    "tune_ranking": [
        {"key": "constraint_pass_count", "direction": "maximize"},
        {"key": "config_hash", "direction": "minimize"},
    ],
    "recommendation_ranking": [
        {"key": "mrr", "direction": "maximize"},
        {"key": "config_complexity", "direction": "minimize"},
        {"key": "config_hash", "direction": "minimize"},
    ],
}


def raw_search_config(**overrides: Any) -> dict[str, Any]:
    """最小合法 search config（dict 形式，方便測試覆寫）。"""
    base: dict[str, Any] = {
        "schema_version": "1.0",
        "search_name": "unit-search",
        "search_mode": "grid",
        "random_seed": 42,
        "max_candidates": 32,
        "validation_top_n": 4,
        "dataset_path": "evaluation/datasets/v1.1/cases.json",
        "corpus_paths": ["sample_docs"],
        "baseline": {
            "chunk_config": {"strategy": "current"},
            "retrieval_config": {"strategy": "dense"},
        },
        "search_space": {"retrieval": {"strategy": ["dense", "identifier_aware"]}},
        "hard_constraints": [],
        "pareto_objectives": [
            {"metric": "mrr", "scope": "aggregate", "direction": "maximize"}
        ],
        "recommendation_policy": dict(MINIMAL_POLICY),
    }
    base.update(overrides)
    return base


def make_search_config(**overrides: Any) -> SearchConfig:
    return SearchConfig.from_raw(raw_search_config(**overrides))


def write_dataset(
    path: str, cases: list[dict[str, Any]], *, version: str = "v1.1-unit-test"
) -> str:
    """寫出一份最小合法 v1.1 dataset（``schema_version`` 必須是 "1.1"）。"""
    payload = {
        "schema_version": "1.1",
        "dataset_version": version,
        "notes": "unit test fixture",
        "cases": cases,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def case(
    case_id: str,
    *,
    split: str,
    query_type: str = "semantic",
    group_id: str | None = None,
    answerable: bool = True,
    question: str = "測試問題",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": case_id,
        "question": question,
        "query_type": query_type,
        "answerable": answerable,
        "group_id": group_id or case_id,
        "split": split,
    }
    if answerable:
        row["expected_evidence"] = [
            {
                "document_sha256": "sha-a",
                "filename": "doc-a.md",
                "canonical_text_version": "v1",
                "loader_version": "v1",
                "page": None,
                "section": "S",
                "offset_scope": "document",
                "char_start": 0,
                "char_end": 10,
                "anchor_text": ["測試"],
            }
        ]
    else:
        row["expected_evidence"] = []
    return row
