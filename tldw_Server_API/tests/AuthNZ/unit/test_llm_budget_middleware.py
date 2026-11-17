import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = pytest.mark.unit


def _build_app_with_middleware():


    from tldw_Server_API.app.core.AuthNZ.llm_budget_middleware import LLMBudgetMiddleware

    app = FastAPI()

    @app.post("/api/v1/chat/completions")
    def chat_stub():
        return {"ok": True}

    app.add_middleware(LLMBudgetMiddleware)
    return app


# Shared fixture for common middleware dependencies
@pytest.fixture
def mock_middleware_dependencies(monkeypatch):
    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw

    class StubSettings:
        VIRTUAL_KEYS_ENABLED = True
        LLM_BUDGET_ENFORCE = True
        LLM_BUDGET_ENDPOINTS = [
            "/api/v1/chat/completions",
            "/api/v1/embeddings",
        ]

    # Core settings
    monkeypatch.setattr(mw, "get_settings", lambda: StubSettings())

    # Key resolution: return a fixed key/user pair
    async def _fake_resolve_api_key_by_hash(_api_key: str, *, settings=None):
        return {"id": 77, "user_id": 88}

    monkeypatch.setattr(mw, "resolve_api_key_by_hash", _fake_resolve_api_key_by_hash)

    # Helper to set budget status per test
    def set_budget_status(over: bool, day_tokens: int, day_usd: float):
        async def fake_get_key_limits(_key_id: int):
            return {"is_virtual": True}

        class StubAuthGovernor:
            async def check_llm_budget_for_api_key(self, principal, key_id):
                return {
                    "over": over,
                    "reasons": ["day_tokens"] if over else [],
                    "day": {"tokens": day_tokens, "usd": day_usd},
                    "month": {"tokens": 50000, "usd": 50.0},
                    "limits": {
                        "is_virtual": True,
                        "llm_budget_day_tokens": 10000,
                        "llm_budget_day_usd": 5.0,
                        "llm_budget_month_tokens": 300000,
                        "llm_budget_month_usd": 150.0,
                    },
                }

        async def fake_get_auth_governor():
            return StubAuthGovernor()

        monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)
        monkeypatch.setattr(mw, "get_auth_governor", fake_get_auth_governor)

    return set_budget_status


def test_middleware_virtual_over_budget(mock_middleware_dependencies):


     # Over budget scenario
    set_budget = mock_middleware_dependencies
    set_budget(True, 20000, 10.0)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": "x", "messages": []},
    )

    assert r.status_code == 402
    data = r.json()
    assert data.get("error") == "budget_exceeded"
    assert "Virtual key budget exceeded" in data.get("message", "")
    assert isinstance(data.get("details"), dict)
    assert data["details"].get("over") is True


def test_middleware_virtual_under_budget_allows(mock_middleware_dependencies):


     # Under budget scenario
    set_budget = mock_middleware_dependencies
    set_budget(False, 100, 0.1)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": "x", "messages": []},
    )

    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_middleware_endpoint_allowlist_forbidden(monkeypatch, mock_middleware_dependencies):


    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    # Ensure budgets allow
    mock_middleware_dependencies(False, 100, 0.1)

    # Allowlist only 'embeddings' so 'chat.completions' should be forbidden
    async def fake_get_key_limits(_key_id: int):
        return {"is_virtual": True, "llm_allowed_endpoints": ["embeddings"]}

    async def fake_is_key_over_budget(_key_id: int):
        return {"over": False, "reasons": []}

    class StubAuthGovernor:
        async def check_llm_budget_for_api_key(self, principal, key_id):
            return {"over": False, "reasons": []}

    async def fake_get_auth_governor():
        return StubAuthGovernor()

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)
    monkeypatch.setattr(mw, "get_auth_governor", fake_get_auth_governor)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": "m", "messages": []},
    )
    assert r.status_code == 403
    data = r.json()
    assert data.get("error") == "forbidden"
    assert "Endpoint 'chat.completions' not allowed" in data.get("message", "")


def test_middleware_provider_allowlist_forbidden(monkeypatch, mock_middleware_dependencies):


    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    # Ensure budgets allow
    mock_middleware_dependencies(False, 100, 0.1)

    async def fake_get_key_limits(_key_id: int):
        return {"is_virtual": True, "llm_allowed_providers": ["OpenAI", "Anthropic"]}

    async def fake_is_key_over_budget(_key_id: int):
        return {"over": False, "reasons": []}

    class StubAuthGovernor:
        async def check_llm_budget_for_api_key(self, principal, key_id):
            return {"over": False, "reasons": []}

    async def fake_get_auth_governor():
        return StubAuthGovernor()

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)
    monkeypatch.setattr(mw, "get_auth_governor", fake_get_auth_governor)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key", "X-LLM-Provider": "OtherProv"},
        json={"model": "m", "messages": []},
    )
    assert r.status_code == 403
    data = r.json()
    assert data.get("error") == "forbidden"
    assert "Provider 'OtherProv' not allowed" in data.get("message", "")


def test_middleware_model_allowlist_forbidden(monkeypatch, mock_middleware_dependencies):


    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    # Ensure budgets allow
    mock_middleware_dependencies(False, 100, 0.1)

    async def fake_get_key_limits(_key_id: int):
        return {"is_virtual": True, "llm_allowed_models": ["allowed-model"]}

    async def fake_is_key_over_budget(_key_id: int):
        return {"over": False, "reasons": []}

    class StubAuthGovernor:
        async def check_llm_budget_for_api_key(self, principal, key_id):
            return {"over": False, "reasons": []}

    async def fake_get_auth_governor():
        return StubAuthGovernor()

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)
    monkeypatch.setattr(mw, "get_auth_governor", fake_get_auth_governor)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": "forbidden-model", "messages": []},
    )
    assert r.status_code == 403
    data = r.json()
    assert data.get("error") == "forbidden"
    assert "Model 'forbidden-model' not allowed" in data.get("message", "")


def test_middleware_endpoint_allowlist_allowed(monkeypatch, mock_middleware_dependencies):


    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    # Ensure budgets allow
    mock_middleware_dependencies(False, 100, 0.1)

    # Allowlist includes chat.completions, so request should proceed
    async def fake_get_key_limits(_key_id: int):
        return {"is_virtual": True, "llm_allowed_endpoints": ["chat.completions"]}

    async def fake_is_key_over_budget(_key_id: int):
        return {"over": False, "reasons": []}

    class StubAuthGovernor:
        async def check_llm_budget_for_api_key(self, principal, key_id):
            return {"over": False, "reasons": []}

    async def fake_get_auth_governor():
        return StubAuthGovernor()

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)
    monkeypatch.setattr(mw, "get_auth_governor", fake_get_auth_governor)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": "m", "messages": []},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_middleware_provider_allowlist_allowed(monkeypatch, mock_middleware_dependencies):


    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    # Ensure budgets allow
    mock_middleware_dependencies(False, 100, 0.1)

    async def fake_get_key_limits(_key_id: int):
        return {"is_virtual": True, "llm_allowed_providers": ["OpenAI", "Anthropic", "OtherProv"]}

    async def fake_is_key_over_budget(_key_id: int):
        return {"over": False, "reasons": []}

    class StubAuthGovernor:
        async def check_llm_budget_for_api_key(self, principal, key_id):
            return {"over": False, "reasons": []}

    async def fake_get_auth_governor():
        return StubAuthGovernor()

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)
    monkeypatch.setattr(mw, "get_auth_governor", fake_get_auth_governor)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key", "X-LLM-Provider": "OtherProv"},
        json={"model": "m", "messages": []},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_middleware_model_allowlist_allowed(monkeypatch, mock_middleware_dependencies):


    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    # Ensure budgets allow
    mock_middleware_dependencies(False, 100, 0.1)

    async def fake_get_key_limits(_key_id: int):
        return {"is_virtual": True, "llm_allowed_models": ["allowed-model"]}

    async def fake_is_key_over_budget(_key_id: int):
        return {"over": False, "reasons": []}

    class StubAuthGovernor:
        async def check_llm_budget_for_api_key(self, principal, key_id):
            return {"over": False, "reasons": []}

    async def fake_get_auth_governor():
        return StubAuthGovernor()

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)
    monkeypatch.setattr(mw, "get_auth_governor", fake_get_auth_governor)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": "allowed-model", "messages": []},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_settings_cache_invalidation(monkeypatch):


     # Verify middleware caches settings per generation and refreshes on change
    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    from fastapi import FastAPI

    app = FastAPI()
    m = mw.LLMBudgetMiddleware(app)

    state = {"gen": 0, "calls": 0}
    a = object()
    b = object()

    def fake_get_settings():

        state["calls"] += 1
        return a if state["gen"] == 0 else b

    monkeypatch.setattr(mw, "get_settings", fake_get_settings)
    monkeypatch.setattr(mw, "get_settings_generation", lambda: state["gen"])

    s1 = m._get_settings_cached()
    s2 = m._get_settings_cached()
    assert s1 is s2
    assert state["calls"] == 1  # only called once for same generation

    # Bump generation -> should refresh
    state["gen"] = 1
    s3 = m._get_settings_cached()
    assert s3 is b and s3 is not s1
    assert state["calls"] == 2

    # Same generation again -> no additional calls
    s4 = m._get_settings_cached()
    assert s4 is s3
    assert state["calls"] == 2


def test_middleware_budget_check_failure_fails_closed(monkeypatch, mock_middleware_dependencies):


    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    # Ensure middleware path and settings are active and key resolves
    mock_middleware_dependencies(False, 100, 0.1)

    # Force get_key_limits to mark the key as virtual
    async def fake_get_key_limits(_key_id: int):
        return {"is_virtual": True}

    # Make the budget check raise to simulate an internal failure
    class StubAuthGovernor:
        async def check_llm_budget_for_api_key(self, principal, key_id):
            raise RuntimeError("simulated budget failure")

    async def fake_get_auth_governor():
        return StubAuthGovernor()

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)
    monkeypatch.setattr(mw, "get_auth_governor", fake_get_auth_governor)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": "m", "messages": []},
    )

    assert r.status_code == 503
    data = r.json()
    assert data.get("error") == "budget_check_failed"
    assert "Failed to evaluate budget enforcement" in data.get("message", "")
    assert data.get("details", {}).get("reason") == "internal_error"
    assert "simulated budget failure" not in r.text


def test_middleware_key_limits_exception_fails_closed(monkeypatch, mock_middleware_dependencies):


    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    # Ensure middleware path and settings are active and key resolves
    mock_middleware_dependencies(False, 100, 0.1)

    async def fake_get_key_limits(_key_id: int):
        raise RuntimeError("limits unavailable")

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": "m", "messages": []},
    )

    assert r.status_code == 503
    data = r.json()
    assert data.get("error") == "budget_limits_unavailable"
    assert "Failed to load virtual key limits" in data.get("message", "")
    assert data.get("details", {}).get("reason") == "internal_error"
    assert "limits unavailable" not in r.text


def test_middleware_key_limits_missing_fails_closed(monkeypatch, mock_middleware_dependencies):


    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw
    # Ensure middleware path and settings are active and key resolves
    mock_middleware_dependencies(False, 100, 0.1)

    async def fake_get_key_limits(_key_id: int):
        return None

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)

    app = _build_app_with_middleware()
    client = TestClient(app)

    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": "m", "messages": []},
    )

    assert r.status_code == 503
    data = r.json()
    assert data.get("error") == "budget_limits_unavailable"
    assert "Failed to load virtual key limits" in data.get("message", "")


@pytest.mark.parametrize("body_model", [{"name": "allowed-model"}, ["allowed-model"]])
def test_middleware_model_allowlist_refuses_an_unhashable_model_instead_of_skipping(
    monkeypatch, mock_middleware_dependencies, body_model
):
    """A dict allowlist with a dict or list body model raised TypeError, which skipped enforcement (fail open)."""
    import tldw_Server_API.app.core.AuthNZ.llm_budget_middleware as mw

    mock_middleware_dependencies(False, 100, 0.1)

    async def fake_get_key_limits(_key_id: int):
        return {"is_virtual": True, "llm_allowed_models": {"allowed-model": True}}

    class StubAuthGovernor:
        async def check_llm_budget_for_api_key(self, principal, key_id):
            return {"over": False, "reasons": []}

    async def fake_get_auth_governor():
        return StubAuthGovernor()

    monkeypatch.setattr(mw, "get_key_limits", fake_get_key_limits)
    monkeypatch.setattr(mw, "get_auth_governor", fake_get_auth_governor)

    client = TestClient(_build_app_with_middleware())
    r = client.post(
        "/api/v1/chat/completions",
        headers={"X-API-KEY": "dummy-key"},
        json={"model": body_model, "messages": []},
    )

    assert r.status_code == 403
    assert r.json().get("error") == "forbidden"
