from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []
        self.exceptions: list[str] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(str(message))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(str(message))

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.exceptions.append(str(message))


class _ExplodingLoader:
    async def load_once(self) -> None:
        raise RuntimeError("resource governor backend exploded at /private/tmp/resource-governor.db")


class _ExplodingStore:
    def __init__(self) -> None:
        raise RuntimeError("resource governor store exploded at /private/tmp/resource-governor.db")


class _StaticLoader:
    async def load_once(self) -> None:
        return None

    def get_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(version=1, policies={}, tenant={}, source_path="default")


class _MetadataFailingLoader:
    def __init__(self) -> None:
        self.snapshot_calls = 0

    async def load_once(self) -> None:
        return None

    def get_snapshot(self) -> SimpleNamespace:
        self.snapshot_calls += 1
        if self.snapshot_calls == 1:
            raise RuntimeError("metadata update exploded at /private/tmp/resource-governor.db")
        return SimpleNamespace(version=1, policies={}, tenant={}, source_path="default")


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


def _assert_sanitized_warning_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.warnings == [expected_message]
    rendered = " ".join(logger_stub.warnings)
    assert "exploded" not in rendered
    assert "/private/" not in rendered


def _assert_sanitized_exception_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.exceptions == [expected_message]
    rendered = " ".join(logger_stub.exceptions)
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


@pytest.mark.asyncio
async def test_upsert_policy_version_conflict_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor

    class _ConflictPolicyAdmin:
        async def upsert_policy(self, *_args: Any, **_kwargs: Any) -> None:
            raise resource_governor.PolicyVersionConflictError("safe.policy", 2, 3)

    logger_stub = _LoggerStub()
    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "AuthNZPolicyAdmin", _ConflictPolicyAdmin)

    response = await resource_governor.upsert_policy(
        "safe.policy",
        resource_governor.PolicyUpsertRequest(payload={"requests": {"rpm": 1}}, version=2),
    )

    assert response.status_code == 409
    assert response.body
    _assert_sanitized_debug_log(logger_stub, "upsert_policy version conflict")


@pytest.mark.asyncio
async def test_delete_policy_version_conflict_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor

    class _ConflictPolicyAdmin:
        async def delete_policy(self, *_args: Any, **_kwargs: Any) -> bool:
            raise resource_governor.PolicyVersionConflictError("safe.policy", 2, 3)

    logger_stub = _LoggerStub()
    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "AuthNZPolicyAdmin", _ConflictPolicyAdmin)

    response = await resource_governor.delete_policy("safe.policy", version=2)

    assert response.status_code == 409
    assert response.body
    _assert_sanitized_debug_log(logger_stub, "delete_policy version conflict")


def test_resource_governor_lazy_init_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor
    from tldw_Server_API.app.core import Resource_Governance as rg_package

    class _ExplodingGovernor:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("governor init exploded at /private/tmp/resource-governor.db")

    logger_stub = _LoggerStub()
    app = SimpleNamespace(
        state=SimpleNamespace(rg_governor=None, rg_policy_loader=object()),
    )
    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "_get_app", lambda: app)
    monkeypatch.setattr(rg_package, "MemoryResourceGovernor", _ExplodingGovernor)

    result = resource_governor._get_or_init_governor()

    assert result is None
    _assert_sanitized_debug_log(logger_stub, "Resource governor lazy-init skipped")


@pytest.mark.asyncio
async def test_policy_snapshot_db_init_fallback_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor
    from tldw_Server_API.app.core.Resource_Governance import authnz_policy_store
    from tldw_Server_API.app.core.Resource_Governance import policy_loader

    logger_stub = _LoggerStub()
    app = SimpleNamespace(
        state=SimpleNamespace(rg_policy_store="db", rg_policy_loader=None),
    )
    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "_get_app", lambda: app)
    monkeypatch.setattr(authnz_policy_store, "AuthNZPolicyStore", _ExplodingStore)
    monkeypatch.setattr(policy_loader, "default_policy_loader", lambda: _StaticLoader())
    monkeypatch.setenv("RG_POLICY_STORE", "db")
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)

    response = await resource_governor.get_resource_governor_policy()

    assert response.status_code == 200
    _assert_sanitized_warning_log(
        logger_stub,
        "RG policy loader DB init failed; falling back to file store",
    )


@pytest.mark.asyncio
async def test_policy_snapshot_init_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor
    from tldw_Server_API.app.core.Resource_Governance import policy_loader

    logger_stub = _LoggerStub()
    app = SimpleNamespace(
        state=SimpleNamespace(rg_policy_store="file", rg_policy_loader=None),
    )
    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "_get_app", lambda: app)
    monkeypatch.setattr(policy_loader, "default_policy_loader", lambda: _ExplodingLoader())
    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)

    response = await resource_governor.get_resource_governor_policy()

    assert response.status_code == 503
    _assert_sanitized_exception_log(
        logger_stub,
        "Resource governor policy loader init failed",
    )


@pytest.mark.asyncio
async def test_policy_snapshot_metadata_update_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import resource_governor
    from tldw_Server_API.app.core.Resource_Governance import policy_loader

    logger_stub = _LoggerStub()
    app = SimpleNamespace(
        state=SimpleNamespace(rg_policy_store="file", rg_policy_loader=None),
    )
    monkeypatch.setattr(resource_governor, "logger", logger_stub)
    monkeypatch.setattr(resource_governor, "_get_app", lambda: app)
    monkeypatch.setattr(policy_loader, "default_policy_loader", lambda: _MetadataFailingLoader())
    monkeypatch.setenv("RG_POLICY_STORE", "file")
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)

    response = await resource_governor.get_resource_governor_policy()

    assert response.status_code == 200
    _assert_sanitized_exception_log(
        logger_stub,
        "Failed updating app.state RG metadata",
    )
