import os

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.main import app

pytestmark = [pytest.mark.integration, pytest.mark.evaluations]


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch):
    # Open the admin gate; requests authenticate with the real single-user key.
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "false")


def test_admin_idempotency_cleanup_basic():


    client = TestClient(app, headers={"X-API-KEY": get_settings().SINGLE_USER_API_KEY})

    # Trigger cleanup with a short TTL (still safe; DB may be empty)
    r = client.post("/api/v1/evaluations/admin/idempotency/cleanup", params={"ttl_hours": 1})
    assert r.status_code == 200
    j = r.json()

    # Shape assertions
    assert isinstance(j, dict)
    assert "deleted_total" in j
    assert "details" in j
    assert isinstance(j["deleted_total"], int)
    assert isinstance(j["details"], list)
    for entry in j["details"]:
        assert isinstance(entry, dict)
        assert "user_id" in entry and "deleted" in entry
        assert isinstance(entry["user_id"], int)
        assert isinstance(entry["deleted"], int)
