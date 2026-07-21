"""Read-only Optimizer Reporting API 測試（Slice 8 §五 / §二十）。

以 dependency override 注入指向 demo fixture 的 read-only store；驗證端點、404、
unavailable 語意、admin 狀態透傳與去識別化。Read API 不依賴 admin write enabled。
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from app.api import optimizer_reporting as rpt
from app.core.config import Settings
from app.dependencies import get_settings
from app.main import create_app
from evaluation.v11.reporting import OptimizerReportingStore
from evaluation.v11.reporting.demo_fixture import build_demo_fixtures


@pytest.fixture()
def fixtures(tmp_path):
    return build_demo_fixtures(str(tmp_path / "demo"))


def _client(store, *, admin_enabled=False):
    os.environ["SKIP_STARTUP_SERVICES"] = "1"
    settings = Settings(
        optimizer_admin_enabled=admin_enabled,
        optimizer_active_alias="factory_knowledge_active",
    )
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[rpt.get_reporting_store] = lambda: store
    return TestClient(app)


@pytest.fixture()
def client(fixtures):
    store = OptimizerReportingStore(fixtures["search"], fixtures["review"], fixtures["deployment"])
    return _client(store)


def test_overview(client):
    with client:
        r = client.get("/api/optimizer/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["is_demo_fixture"] is True
    assert body["search"]["latest_search"]["apply_eligible"] is False
    assert body["deployment"]["admin_write_enabled"] is False


def test_searches_list_and_detail(client):
    with client:
        lst = client.get("/api/optimizer/searches").json()
        assert lst["available"] is True
        sid = lst["searches"][0]["search_id"]
        detail = client.get(f"/api/optimizer/searches/{sid}").json()
    assert detail["found"] is True
    assert detail["search"]["recommendation_locked"] is True


def test_candidates_and_comparison(client):
    with client:
        cands = client.get("/api/optimizer/searches/search-demo/candidates").json()
        cmp = client.get("/api/optimizer/searches/search-demo/comparison").json()
    assert any(c["is_baseline"] for c in cands["candidates"])
    assert cmp["comparison"]["per_case"]["cases"]


def test_reviews_and_coverage(client):
    with client:
        rv = client.get("/api/optimizer/reviews").json()
        detail = client.get("/api/optimizer/reviews/review-demo").json()
        cov = client.get("/api/optimizer/coverage").json()
    assert rv["sessions"]
    assert detail["session"]["counts"]["accepted"] == 1
    assert any("identifier_bearing_unanswerable" in w for w in cov["coverage"]["coverage_warnings"])


def test_deployment_summary(client):
    with client:
        r = client.get("/api/optimizer/deployment/summary").json()
    assert r["alias_initialized"] is False
    assert r["history_count"] == 0
    assert r["admin_write_enabled"] is False


def test_unknown_ids_return_404(client):
    with client:
        assert client.get("/api/optimizer/searches/nope").status_code == 404
        assert client.get("/api/optimizer/searches/nope/candidates").status_code == 404
        assert client.get("/api/optimizer/searches/nope/comparison").status_code == 404
        assert client.get("/api/optimizer/reviews/nope").status_code == 404


def test_read_api_independent_of_admin_enabled(fixtures):
    """Read API 不依賴 admin write enabled（§五.10）：admin 開/關資料相同。"""
    store = OptimizerReportingStore(fixtures["search"], fixtures["review"], fixtures["deployment"])
    c_off = _client(store, admin_enabled=False)
    c_on = _client(store, admin_enabled=True)
    with c_off:
        off = c_off.get("/api/optimizer/searches/search-demo").json()
    with c_on:
        on = c_on.get("/api/optimizer/searches/search-demo").json()
    assert off["search"]["recommended_config_hash"] == on["search"]["recommended_config_hash"]


def test_unavailable_when_not_configured():
    store = OptimizerReportingStore(None, None, None)
    client = _client(store)
    with client:
        ov = client.get("/api/optimizer/overview").json()
        searches = client.get("/api/optimizer/searches").json()
    assert ov["search"]["configured"] is False
    assert searches["available"] is False
    assert searches["searches"] == []


def test_no_path_or_secret_leak(client):
    needles = [".db", "cases-v2.json", "dataset_path", "api_key", "sk-"]
    paths = [
        "/api/optimizer/overview", "/api/optimizer/searches/search-demo",
        "/api/optimizer/searches/search-demo/comparison",
        "/api/optimizer/reviews/review-demo", "/api/optimizer/coverage",
        "/api/optimizer/deployment/summary",
    ]
    with client:
        for p in paths:
            blob = json.dumps(client.get(p).json(), ensure_ascii=False)
            for n in needles:
                assert n not in blob, f"{n} leaked in {p}"
