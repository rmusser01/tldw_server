import json
import csv
import io

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user(monkeypatch, tmp_path):
    async def override_user():
        return User(id=909, username="wluser", email=None, is_active=True)

    base_dir = tmp_path / "test_user_dbs_runs_csv"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("TEST_MODE", "1")

    from fastapi import FastAPI
    from tldw_Server_API.app.core.config import API_V1_PREFIX
    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router
    from tldw_Server_API.app.core.DB_Management.backends.factory import reset_managed_sqlite_backends

    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user
    user_db_path = base_dir / "909" / "Media_DB_v2.db"
    try:
        with TestClient(app) as client:
            yield client
    finally:
        reset_managed_sqlite_backends(sqlite_targets=[str(user_db_path)], mode="hard")
        app.dependency_overrides.clear()


def _seed_run_with_stats(job_id: int, stats: dict) -> int:
    db = WatchlistsDatabase.for_user(909)
    run = db.create_run(job_id, status="finished")
    db.update_run(run.id, status="finished", stats_json=json.dumps(stats))
    return run.id


def test_global_runs_csv_export_headers_and_rows(client_with_user: TestClient):
    c = client_with_user
    # Create minimal source and job
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Feed", "url": "https://example.com/rss.xml", "source_type": "rss"},
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "JobA", "scope": {"sources": [sid]}},
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    stats = {
        "items_found": 5,
        "items_ingested": 3,
        "filters_actions": {"include": 2, "exclude": 1, "flag": 0},
        "filter_tallies": {"kw:foo": 2, "regex:bar": 1},
    }
    run_id = _seed_run_with_stats(jid, stats)

    r = c.get("/api/v1/watchlists/runs/export.csv", params={"scope": "global", "page": 1, "size": 10})
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("text/csv")
    cd = r.headers.get("content-disposition", "")
    assert "watchlists_runs_global_" in cd
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    # Header + at least one row
    assert lines[0].startswith("id,job_id,status,started_at,finished_at,items_found,items_ingested,filters_include,filters_exclude,filters_flag")
    assert any(str(run_id) in ln for ln in lines[1:])


def test_job_runs_csv_export_and_tallies_csv(client_with_user: TestClient):
    c = client_with_user
    # Create another source/job
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "Site", "url": "https://example.com/", "source_type": "site"},
    )
    assert s.status_code == 200, s.text
    sid = s.json()["id"]
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "JobB", "scope": {"sources": [sid]}},
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    stats = {
        "items_found": 9,
        "items_ingested": 4,
        "filters_actions": {"include": 3, "exclude": 2, "flag": 1},
        "filter_tallies": {"kw:bar": 3, "regex:^x": 2, "kw:baz": 1},
    }
    run_id = _seed_run_with_stats(jid, stats)

    # By-job CSV
    r = c.get("/api/v1/watchlists/runs/export.csv", params={"scope": "job", "job_id": jid, "page": 1, "size": 10})
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("text/csv")
    cd = r.headers.get("content-disposition", "")
    assert "watchlists_runs_job_" in cd
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert lines[0].startswith("id,job_id,status,started_at,finished_at,items_found,items_ingested,filters_include,filters_exclude,filters_flag")
    # Expect our run row present
    assert any(str(run_id) in ln for ln in lines[1:])

    # Tallies CSV for that run
    t = c.get(f"/api/v1/watchlists/runs/{run_id}/tallies.csv")
    assert t.status_code == 200, t.text
    assert t.headers.get("content-type", "").startswith("text/csv")
    t_cd = t.headers.get("content-disposition", "")
    assert f"watchlists_run_{run_id}_tallies_" in t_cd
    t_lines = [ln for ln in t.text.splitlines() if ln.strip()]
    # Header + 3 tallies rows
    assert t_lines[0] == "run_id,filter_key,count"
    # Order is not guaranteed; check membership
    assert any("kw:bar" in ln and ",3" in ln for ln in t_lines[1:])
    assert any("regex:^x" in ln and ",2" in ln for ln in t_lines[1:])
    assert any("kw:baz" in ln and ",1" in ln for ln in t_lines[1:])


def test_runs_csv_headers_with_pagination_scope_job(client_with_user: TestClient):
    c = client_with_user
    # Create a source/job
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "FeedP", "url": "https://example.com/p.xml", "source_type": "rss"},
    )
    assert s.status_code == 200, s.text
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "JobP", "scope": {"sources": [s.json()["id"]]}},
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    # Seed two runs to paginate over
    _ = _seed_run_with_stats(jid, {"items_found": 1, "items_ingested": 1})
    _ = _seed_run_with_stats(jid, {"items_found": 2, "items_ingested": 2})

    # Page 1, size 1
    r1 = c.get("/api/v1/watchlists/runs/export.csv", params={"scope": "job", "job_id": jid, "page": 1, "size": 1})
    assert r1.status_code == 200, r1.text
    lines1 = [ln for ln in r1.text.splitlines() if ln.strip()]
    assert lines1[0].startswith("id,job_id,status,started_at,finished_at,items_found,items_ingested,filters_include,filters_exclude,filters_flag")
    assert len(lines1) == 2  # header + 1 row

    # Page 2, size 1
    r2 = c.get("/api/v1/watchlists/runs/export.csv", params={"scope": "job", "job_id": jid, "page": 2, "size": 1})
    assert r2.status_code == 200, r2.text
    lines2 = [ln for ln in r2.text.splitlines() if ln.strip()]
    assert lines2[0] == lines1[0]
    assert len(lines2) == 2


def test_global_runs_csv_export_with_tallies_column(client_with_user: TestClient):
    c = client_with_user
    # Create source/job and seed run with tallies
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "FeedT", "url": "https://example.com/t.xml", "source_type": "rss"},
    )
    assert s.status_code == 200, s.text
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "JobT", "scope": {"sources": [s.json()["id"]]}},
    )
    assert j.status_code == 200, j.text
    jid = j.json()["id"]

    stats = {
        "items_found": 7,
        "items_ingested": 5,
        "filters_actions": {"include": 4, "exclude": 1, "flag": 0},
        "filter_tallies": {"kw:alpha": 3, "regex:beta": 1, "kw:gamma": 1},
    }
    run_id = _seed_run_with_stats(jid, stats)

    r = c.get(
        "/api/v1/watchlists/runs/export.csv",
        params={"scope": "global", "page": 1, "size": 10, "include_tallies": True},
    )
    assert r.status_code == 200, r.text
    rows = list(csv.reader(io.StringIO(r.text)))
    header = rows[0]
    assert header[:10] == [
        "id",
        "job_id",
        "status",
        "started_at",
        "finished_at",
        "items_found",
        "items_ingested",
        "filters_include",
        "filters_exclude",
        "filters_flag",
    ]
    assert header[-1] == "filter_tallies_json"
    # Find our run row and ensure tallies JSON parseable
    run_row = next((row for row in rows[1:] if row and row[0] == str(run_id)), None)
    assert run_row is not None
    assert len(run_row) == len(header)
    tallies = json.loads(run_row[-1])
    assert tallies.get("kw:alpha") == 3


def test_global_runs_csv_export_with_aggregate_tallies_mode(client_with_user: TestClient):
    c = client_with_user
    # Create sources/jobs and seed runs with tallies so aggregation spans multiple jobs/runs.
    s1 = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "AggFeedA", "url": "https://example.com/agg-a.xml", "source_type": "rss"},
    )
    assert s1.status_code == 200, s1.text
    j1 = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "AggJobA", "scope": {"sources": [s1.json()["id"]]}},
    )
    assert j1.status_code == 200, j1.text
    jid1 = j1.json()["id"]

    s2 = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "AggFeedB", "url": "https://example.com/agg-b.xml", "source_type": "rss"},
    )
    assert s2.status_code == 200, s2.text
    j2 = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "AggJobB", "scope": {"sources": [s2.json()["id"]]}},
    )
    assert j2.status_code == 200, j2.text
    jid2 = j2.json()["id"]

    _seed_run_with_stats(
        jid1,
        {
            "items_found": 4,
            "items_ingested": 3,
            "filter_tallies": {"kw:alpha": 2, "kw:beta": 1},
        },
    )
    _seed_run_with_stats(
        jid2,
        {
            "items_found": 6,
            "items_ingested": 5,
            "filter_tallies": {"kw:alpha": 1, "regex:gamma": 4},
        },
    )

    r = c.get(
        "/api/v1/watchlists/runs/export.csv",
        params={
            "scope": "global",
            "include_tallies": True,
            "tallies_mode": "aggregate",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("text/csv")
    assert "watchlists_runs_global_tallies_" in r.headers.get("content-disposition", "")
    assert r.headers.get("x-has-more") == "false"

    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == ["filter_key", "count"]
    # Expect sort by count desc, then key asc.
    assert rows[1] == ["regex:gamma", "4"]
    assert rows[2] == ["kw:alpha", "3"]
    assert rows[3] == ["kw:beta", "1"]


def test_aggregate_tallies_mode_requires_global_scope(client_with_user: TestClient):
    c = client_with_user
    s = c.post(
        "/api/v1/watchlists/sources",
        json={"name": "AggScopeFeed", "url": "https://example.com/agg-scope.xml", "source_type": "rss"},
    )
    assert s.status_code == 200, s.text
    j = c.post(
        "/api/v1/watchlists/jobs",
        json={"name": "AggScopeJob", "scope": {"sources": [s.json()["id"]]}},
    )
    assert j.status_code == 200, j.text

    r = c.get(
        "/api/v1/watchlists/runs/export.csv",
        params={
            "scope": "job",
            "job_id": j.json()["id"],
            "include_tallies": True,
            "tallies_mode": "aggregate",
        },
    )
    assert r.status_code == 400, r.text
    assert r.json().get("detail") == "tallies_aggregation_global_only"
