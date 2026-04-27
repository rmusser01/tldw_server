from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(str(message))


class _ExplodingLoader:
    async def load_once(self) -> None:
        raise RuntimeError("resource governor backend exploded at /private/tmp/resource-governor.db")


class _ExplodingStore:
    def __init__(self) -> None:
        raise RuntimeError("resource governor store exploded at /private/tmp/resource-governor.db")


class _FakePolicyAdmin:
    async def upsert_policy(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def delete_policy(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


def _assert_sanitized_debug_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.debugs == [expected_message]
    rendered = " ".join(logger_stub.debugs)
    assert "exploded" not in rendered
    assert "/private/" not in rendered


@pytest.mark.asyncio
async def test_upsert_policy_db_loader_init_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor
    logger_stub = _LoggerStub()
    app = SimpleNamespace(
        state=SimpleNamespace(rg_policy_store="db", rg_policy_loader=None),
    )

    from tldw_Server_API.app.core.Resource_Governance import authnz_policy_store

    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "_get_app", lambda: app)
    monkeypatch.setattr(resource_governor, "AuthNZPolicyAdmin", _FakePolicyAdmin)
    monkeypatch.setattr(authnz_policy_store, "AuthNZPolicyStore", _ExplodingStore)
    monkeypatch.setenv("RG_POLICY_STORE", "db")

    response = await resource_governor.upsert_policy(
        "safe.policy",
        resource_governor.PolicyUpsertRequest(payload={"requests": {"rpm": 1}}),
    )

    assert response.status_code == 200
    _assert_sanitized_debug_log(logger_stub, "Policy upsert DB loader init skipped")


@pytest.mark.asyncio
async def test_upsert_policy_refresh_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor
    logger_stub = _LoggerStub()
    app = SimpleNamespace(
        state=SimpleNamespace(rg_policy_store="db", rg_policy_loader=_ExplodingLoader()),
    )

    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "_get_app", lambda: app)
    monkeypatch.setattr(resource_governor, "AuthNZPolicyAdmin", _FakePolicyAdmin)
    monkeypatch.setenv("RG_POLICY_STORE", "db")

    response = await resource_governor.upsert_policy(
        "safe.policy",
        resource_governor.PolicyUpsertRequest(payload={"requests": {"rpm": 1}}),
    )

    assert response.status_code == 200
    _assert_sanitized_debug_log(logger_stub, "Policy upsert refresh skipped")


@pytest.mark.asyncio
async def test_delete_policy_db_loader_init_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor
    logger_stub = _LoggerStub()
    app = SimpleNamespace(
        state=SimpleNamespace(rg_policy_store="db", rg_policy_loader=None),
    )

    from tldw_Server_API.app.core.Resource_Governance import authnz_policy_store

    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "_get_app", lambda: app)
    monkeypatch.setattr(resource_governor, "AuthNZPolicyAdmin", _FakePolicyAdmin)
    monkeypatch.setattr(authnz_policy_store, "AuthNZPolicyStore", _ExplodingStore)
    monkeypatch.setenv("RG_POLICY_STORE", "db")

    response = await resource_governor.delete_policy("safe.policy")

    assert response.status_code == 200
    _assert_sanitized_debug_log(logger_stub, "Policy delete DB loader init skipped")


@pytest.mark.asyncio
async def test_delete_policy_refresh_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor
    logger_stub = _LoggerStub()
    app = SimpleNamespace(
        state=SimpleNamespace(rg_policy_store="db", rg_policy_loader=_ExplodingLoader()),
    )

    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "_get_app", lambda: app)
    monkeypatch.setattr(resource_governor, "AuthNZPolicyAdmin", _FakePolicyAdmin)
    monkeypatch.setenv("RG_POLICY_STORE", "db")

    response = await resource_governor.delete_policy("safe.policy")

    assert response.status_code == 200
    _assert_sanitized_debug_log(logger_stub, "Policy delete refresh skipped")
