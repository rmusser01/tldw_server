"""StudyPack HTTP reads serialize real database timestamps without rewriting rows."""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints import flashcards as endpoint
from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackSummaryResponse
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def pack_api(request, tmp_path):
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgres"
        else None
    )
    db = CharactersRAGDB(tmp_path / "packs.db", client_id="2", backend=backend)
    # Explicit Jobs backend prevents the required-PG runner's global URL from
    # redirecting this independent, real local Jobs store.
    jobs = JobManager(db_path=tmp_path / "jobs.db", backend="sqlite")
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1")
    app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[endpoint.get_job_manager] = lambda: jobs
    app.dependency_overrides[endpoint.get_request_user] = lambda: SimpleNamespace(id=2)
    app.dependency_overrides[endpoint.get_auth_principal] = lambda: AuthPrincipal(
        kind="user", user_id=2, roles=[], permissions=[]
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield db, jobs, client, request.param
    finally:
        app.dependency_overrides.clear()
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


def _seed_pack(db):
    with chacha_operation(independent=True):
        deck = db.add_deck("Synthetic pack deck")
        pack = db.create_study_pack(
            title="Synthetic study pack",
            workspace_id=None,
            deck_id=deck,
            source_bundle_json={"items": [{"source_type": "note", "source_id": "synthetic-note"}]},
            generation_options_json={"deck_mode": "new"},
        )
        return db.get_study_pack(pack)


def _complete_job(jobs, result, *, owner="2"):
    job = jobs.create_job(
        domain="study_packs",
        queue="default",
        job_type="study_pack_generate",
        payload={"title": "Synthetic job"},
        owner_user_id=owner,
    )
    acquired = jobs.acquire_next_job(
        domain="study_packs", queue="default", lease_seconds=30, worker_id="timestamp-fixture"
    )
    assert acquired is not None and int(acquired["id"]) == int(job["id"])
    jobs.complete_job(int(job["id"]), result=result, worker_id="timestamp-fixture", lease_id=str(acquired["lease_id"]))
    return jobs.get_job(int(job["id"]))


@pytest.mark.integration
@pytest.mark.parametrize("route", ["detail", "completed-job"])
def test_http_read_preserves_completed_pack_and_serializes_timestamps(pack_api, route):
    db, jobs, client, kind = pack_api
    before = _seed_pack(db)
    job = _complete_job(jobs, {"pack_id": before["id"], "deck_id": before["deck_id"]})
    for field in ("created_at", "last_modified"):
        assert isinstance(before[field], datetime if kind == "postgres" else str)
    url = (
        f"/api/v1/flashcards/study-packs/{before['id']}"
        if route == "detail"
        else f"/api/v1/flashcards/study-packs/jobs/{job['id']}"
    )
    response = client.get(url)
    with chacha_operation(independent=True):
        assert db.get_study_pack(before["id"]) == before
    assert jobs.get_job(int(job["id"])) == job
    assert response.status_code == 200
    body = response.json()
    if route == "completed-job":
        assert body["job"]["status"] == "completed" and body["error"] is None
        body = body["study_pack"]
    expected = {**before}
    for field in ("created_at", "last_modified"):
        expected[field] = before[field].isoformat() if isinstance(before[field], datetime) else before[field]
    assert body == expected


@pytest.mark.integration
@pytest.mark.parametrize("condition", ["foreign-job", "missing-result", "missing-detail"])
def test_http_read_keeps_existing_missing_and_owner_behavior(pack_api, condition):
    _db, jobs, client, _kind = pack_api
    job = _complete_job(jobs, {}, owner="3" if condition == "foreign-job" else "2")
    url = (
        "/api/v1/flashcards/study-packs/999999"
        if condition == "missing-detail"
        else f"/api/v1/flashcards/study-packs/jobs/{job['id']}"
    )
    response = client.get(url)
    assert response.status_code == (200 if condition == "missing-result" else 404)
    if condition == "missing-result":
        assert response.json()["study_pack"] is None
        assert response.json()["job"]["status"] == "completed"
    assert jobs.get_job(int(job["id"])) == job


def _payload():
    return {
        "id": 7,
        "title": "Synthetic",
        "deck_id": 4,
        "status": "active",
        "deleted": False,
        "client_id": "2",
        "version": 3,
        "source_bundle_json": {"items": [{"source_id": "source-7"}]},
    }


@pytest.mark.parametrize(
    "value,expected",
    [
        (datetime(2026, 9, 17, 8, 20, tzinfo=timezone.utc), "2026-09-17T08:20:00+00:00"),
        (datetime(2026, 9, 17, 8, 20, tzinfo=timezone(timedelta(hours=5, minutes=45))), "2026-09-17T08:20:00+05:45"),
        (datetime(2026, 9, 17, 8, 20), "2026-09-17T08:20:00"),
        ("2026-09-17T08:20:00.123Z", "2026-09-17T08:20:00.123Z"),
        (None, None),
    ],
    ids=["utc", "offset", "naive", "sqlite-string", "null"],
)
def test_schema_preserves_timestamp_and_non_time_values(value, expected):
    payload = _payload()
    baseline = StudyPackSummaryResponse.model_validate(payload).model_dump(mode="json")
    fields = ("created_at", "last_modified")
    result = StudyPackSummaryResponse.model_validate({**payload, **dict.fromkeys(fields, value)}).model_dump(
        mode="json"
    )
    assert {field: result[field] for field in fields} == dict.fromkeys(fields, expected)
    assert {key: val for key, val in result.items() if key not in fields} == {
        key: val for key, val in baseline.items() if key not in fields
    }
    assert StudyPackSummaryResponse.model_validate(result).model_dump(mode="json") == result


@pytest.mark.parametrize("value", [123, {"unexpected": "object"}, [], date(2026, 9, 17)])
def test_schema_rejects_unrelated_timestamp_types(value):
    fields = ("created_at", "last_modified")
    with pytest.raises(ValidationError) as error:
        StudyPackSummaryResponse.model_validate({**_payload(), **dict.fromkeys(fields, value)})
    assert {item["loc"][0] for item in error.value.errors()} == set(fields)
