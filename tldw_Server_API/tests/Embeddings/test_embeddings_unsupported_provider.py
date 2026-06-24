import os
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


@pytest.fixture(autouse=True)
def _testing_env():
    previous_testing = os.environ.get("TESTING")
    os.environ["TESTING"] = "true"
    yield
    if previous_testing is None:
        os.environ.pop("TESTING", None)
    else:
        os.environ["TESTING"] = previous_testing
    app.dependency_overrides.pop(get_request_user, None)


@pytest.fixture
def client():
    with TestClient(app) as c:
        c.cookies.set("csrf_token", "csrf")
        c.headers["X-CSRF-Token"] = "csrf"
        c.headers["Authorization"] = "Bearer test-api-key"
        yield c


def _override_user(admin=False):


    async def _f():
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
        return User(id=1, username="u", email="u@x", is_active=True, is_admin=admin)
    return _f


@pytest.mark.unit
def test_unsupported_provider_returns_501(client):
     # Ensure admin bypass does not affect behavior (use non-admin)
    app.dependency_overrides[get_request_user] = _override_user(admin=False)
    # request with an enum-known but not implemented provider (mistral)
    r = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "mistral"},
        json={"input": "hello", "model": "mistral-embed"}
    )
    assert r.status_code == 501
    body = r.json()
    assert "not implemented" in body.get("detail", "").lower()


@pytest.mark.unit
def test_unsupported_provider_is_classified_before_invalid_dimensions(client):
    app.dependency_overrides[get_request_user] = _override_user(admin=False)

    r = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "mistral"},
        json={"input": "hello", "model": "mistral-embed", "dimensions": 5000},
    )

    assert r.status_code == 501
    assert "not implemented" in r.json().get("detail", "").lower()


@pytest.mark.unit
def test_unknown_provider_is_classified_before_invalid_dimensions(client):
    app.dependency_overrides[get_request_user] = _override_user(admin=False)

    r = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "unknown-provider"},
        json={"input": "hello", "model": "some-embed", "dimensions": 5000},
    )

    assert r.status_code == 400
    assert "unknown provider" in r.json().get("detail", "").lower()


@pytest.mark.unit
def test_batch_unsupported_provider_is_classified_before_invalid_dimensions(client):
    app.dependency_overrides[get_request_user] = _override_user(admin=False)

    r = client.post(
        "/api/v1/embeddings/batch",
        json={
            "texts": ["hello"],
            "provider": "mistral",
            "model": "mistral-embed",
            "dimensions": 5000,
        },
    )

    assert r.status_code == 501
    assert "not implemented" in r.json().get("detail", "").lower()


@pytest.mark.unit
def test_batch_unknown_provider_is_classified_before_invalid_dimensions(client):
    app.dependency_overrides[get_request_user] = _override_user(admin=False)

    r = client.post(
        "/api/v1/embeddings/batch",
        json={
            "texts": ["hello"],
            "provider": "unknown-provider",
            "model": "some-embed",
            "dimensions": 5000,
        },
    )

    assert r.status_code == 400
    assert "unknown provider" in r.json().get("detail", "").lower()
