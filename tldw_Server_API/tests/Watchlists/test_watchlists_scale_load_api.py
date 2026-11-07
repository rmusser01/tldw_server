from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase


pytestmark = [pytest.mark.integration, pytest.mark.performance, pytest.mark.load]


@pytest.fixture()
def client_admin(monkeypatch):
    user_id = 9302

    async def override_user():
        return User(
            id=user_id,
            username="watch-admin",
            email=None,
            role="user",
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=False,
            is_active=True,
        )

    base_dir = Path.cwd() / "Databases" / "test_user_dbs_watchlists_scale_api"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WATCHLISTS_RUNS_REQUIRE_ADMIN", "1")

    user_db_path = DatabasePaths.get_media_db_path(user_id)
    try:
        if user_db_path.exists():
            user_db_path.unlink()
    except Exception:
        _ = None

    from fastapi import FastAPI
    from tldw_Server_API.app.api.v1.endpoints.watchlists import router as watchlists_router
    from tldw_Server_API.app.core.config import API_V1_PREFIX

    app = FastAPI()
    app.include_router(watchlists_router, prefix=f"{API_V1_PREFIX}")
    app.dependency_overrides[get_request_user] = override_user
    # Match mandatory production OpenAPI startup before timing admitted requests.
    app.openapi()
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _seed_scale_dataset(*, user_id: int) -> dict[str, object]:
    db = WatchlistsDatabase.for_user(user_id)
    db.ensure_schema()

    source_count = 300
    source_ids: list[int] = []
    for idx in range(source_count):
        row = db.create_source(
            name=f"ScaleSource{idx}",
            url=f"https://example.com/scale-source-{idx}.xml",
            source_type="rss",
            active=True,
            settings_json=None,
            tags=[],
            group_ids=[],
        )
        source_ids.append(int(row.id))

    job_count = 120
    runs_per_job = 25
    job_ids: list[int] = []
    run_ids: list[int] = []
    for job_idx in range(job_count):
        scope = {"sources": [source_ids[job_idx % source_count]]}
        job = db.create_job(
            name=f"ScaleJob{job_idx}",
            description="scale-load",
            scope_json=json.dumps(scope),
            schedule_expr=None,
            schedule_timezone="UTC",
            active=True,
            max_concurrency=1,
            per_host_delay_ms=50,
            retry_policy_json=json.dumps({}),
            output_prefs_json=json.dumps({}),
            job_filters_json=None,
        )
        job_ids.append(int(job.id))

        for run_idx in range(runs_per_job):
            run = db.create_run(job.id, status="finished")
            stats = {
                "items_found": 8 + (run_idx % 7),
                "items_ingested": 4 + (run_idx % 3),
                "filters_actions": {"include": 3, "exclude": 1, "flag": 1},
                "filter_tallies": {
                    "kw:alpha": 2 + (run_idx % 2),
                    "regex:global": 1,
                    f"kw:job_{job_idx % 10}": 1,
                },
            }
            db.update_run(
                run.id,
                status="finished",
                stats_json=json.dumps(stats),
                error_msg=None,
            )
            run_ids.append(int(run.id))

    return {
        "source_count": source_count,
        "job_count": job_count,
        "run_count": int(job_count * runs_per_job),
        "job_ids": job_ids,
        "run_ids": run_ids,
    }


def test_watchlists_runs_scale_endpoints_within_budget(client_admin: TestClient):
    user_id = 9302
    dataset = _seed_scale_dataset(user_id=user_id)
    run_count = int(dataset["run_count"])  # type: ignore[arg-type]
    first_job_id = int(dataset["job_ids"][0])  # type: ignore[index]
    first_run_id = int(dataset["run_ids"][0])  # type: ignore[index]

    # Runs global listing latency budget
    t0 = time.perf_counter()
    r = client_admin.get("/api/v1/watchlists/runs", params={"page": 1, "size": 200})
    runs_elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    payload = r.json()
    assert int(payload["total"]) >= run_count
    assert len(payload["items"]) <= 200
    assert runs_elapsed < 0.70

    # Runs by job listing latency budget
    t0 = time.perf_counter()
    r = client_admin.get(f"/api/v1/watchlists/jobs/{first_job_id}/runs", params={"page": 1, "size": 200})
    job_runs_elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    payload = r.json()
    assert int(payload["total"]) == 25
    assert len(payload["items"]) <= 200
    assert job_runs_elapsed < 0.55

    # Global CSV export latency budget
    t0 = time.perf_counter()
    r = client_admin.get("/api/v1/watchlists/runs/export.csv", params={"scope": "global", "page": 1, "size": 1000})
    csv_elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0][:3] == ["id", "job_id", "status"]
    assert len(rows) > 1
    assert csv_elapsed < 1.8

    # Aggregate tallies export latency budget
    t0 = time.perf_counter()
    r = client_admin.get(
        "/api/v1/watchlists/runs/export.csv",
        params={"scope": "global", "include_tallies": True, "tallies_mode": "aggregate"},
    )
    agg_elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == ["filter_key", "count"]
    assert any(row and row[0] == "kw:alpha" for row in rows[1:])
    assert agg_elapsed < 4.5

    # Run details latency budget
    t0 = time.perf_counter()
    r = client_admin.get(
        f"/api/v1/watchlists/runs/{first_run_id}/details",
        params={"include_tallies": True, "filtered_sample_max": 50},
    )
    details_elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == first_run_id
    assert isinstance(body.get("stats"), dict)
    assert isinstance(body.get("filter_tallies"), dict)
    assert details_elapsed < 0.65

    # Throughput sanity for repeated runs-list requests
    loops = 10
    t0 = time.perf_counter()
    for _ in range(loops):
        rr = client_admin.get("/api/v1/watchlists/runs", params={"page": 1, "size": 200})
        assert rr.status_code == 200
    throughput = loops / max(time.perf_counter() - t0, 1e-6)
    assert throughput >= 20.0
