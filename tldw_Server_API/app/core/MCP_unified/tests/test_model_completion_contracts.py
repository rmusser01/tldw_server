from __future__ import annotations

import dataclasses
import importlib
import inspect
import uuid

import pytest


def _contracts():
    return importlib.import_module("mcp_unified.interfaces.model_completion")


def test_model_invocation_identity_is_minimized_and_immutable() -> None:
    contracts = _contracts()
    execution_id = str(uuid.uuid4())

    identity = contracts.ModelInvocationIdentity(
        user_id=7,
        active_team_id=11,
        active_organization_id=13,
        execution_id=execution_id,
    )

    assert [field.name for field in dataclasses.fields(identity)] == [
        "user_id",
        "active_team_id",
        "active_organization_id",
        "execution_id",
    ]
    assert not hasattr(identity, "request_id")
    assert not hasattr(identity, "client_id")
    assert not hasattr(identity, "metadata")
    assert not hasattr(identity, "credentials")
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.user_id = 9


@pytest.mark.parametrize("field_name", ["user_id", "active_team_id", "active_organization_id"])
@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.0, "1"])
def test_model_invocation_identity_rejects_invalid_numeric_ids(
    field_name: str,
    invalid: object,
) -> None:
    contracts = _contracts()
    values = {
        "user_id": 1,
        "active_team_id": 2,
        "active_organization_id": 3,
        "execution_id": str(uuid.uuid4()),
    }
    values[field_name] = invalid

    with pytest.raises(ValueError, match="positive non-boolean integer"):
        contracts.ModelInvocationIdentity(**values)


def test_model_invocation_identity_allows_absent_active_scope() -> None:
    contracts = _contracts()

    identity = contracts.ModelInvocationIdentity(
        user_id=1,
        active_team_id=None,
        active_organization_id=None,
        execution_id=str(uuid.uuid4()),
    )

    assert identity.active_team_id is None
    assert identity.active_organization_id is None


@pytest.mark.parametrize(
    "invalid",
    [
        "",
        "not-a-uuid",
        uuid.uuid4().hex,
        str(uuid.uuid4()).upper(),
        str(uuid.uuid1()),
        uuid.uuid4(),
    ],
)
def test_model_invocation_identity_requires_canonical_lowercase_uuid4(
    invalid: object,
) -> None:
    contracts = _contracts()

    with pytest.raises(ValueError, match="canonical lowercase UUIDv4"):
        contracts.ModelInvocationIdentity(
            user_id=1,
            active_team_id=None,
            active_organization_id=None,
            execution_id=invalid,
        )


def test_model_completion_capabilities_are_strict_frozen_booleans() -> None:
    contracts = _contracts()
    capabilities = contracts.ModelCompletionCapabilities(
        native_async_cancellation=True,
        response_limit_before_decode=True,
        native_max_output_tokens=True,
        tool_suppression=True,
        automatic_retries_disabled=True,
    )

    assert dataclasses.is_dataclass(capabilities)
    assert all(getattr(capabilities, field.name) is True for field in dataclasses.fields(capabilities))
    with pytest.raises(dataclasses.FrozenInstanceError):
        capabilities.tool_suppression = False

    with pytest.raises(ValueError, match="capability flags must be booleans"):
        contracts.ModelCompletionCapabilities(
            native_async_cancellation=1,
            response_limit_before_decode=True,
            native_max_output_tokens=True,
            tool_suppression=True,
            automatic_retries_disabled=True,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("system_prompt", None),
        ("user_prompt", 1),
        ("max_output_tokens", True),
        ("max_output_tokens", 0),
        ("max_output_chars", -1),
        ("max_output_bytes", 1.5),
        ("max_provider_response_bytes", "100"),
    ],
)
def test_model_completion_request_rejects_invalid_fields(
    field_name: str,
    invalid: object,
) -> None:
    contracts = _contracts()
    values = {
        "system_prompt": "system",
        "user_prompt": "user",
        "max_output_tokens": 32,
        "max_output_chars": 128,
        "max_output_bytes": 512,
        "max_provider_response_bytes": 2048,
    }
    values[field_name] = invalid

    with pytest.raises(ValueError):
        contracts.ModelCompletionRequest(**values)


def test_model_completion_request_and_result_are_immutable_and_minimized() -> None:
    contracts = _contracts()
    request = contracts.ModelCompletionRequest(
        system_prompt="system",
        user_prompt="user",
        max_output_tokens=32,
        max_output_chars=128,
        max_output_bytes=512,
        max_provider_response_bytes=2048,
    )
    result = contracts.ModelCompletionResult(content="answer")

    assert [field.name for field in dataclasses.fields(request)] == [
        "system_prompt",
        "user_prompt",
        "max_output_tokens",
        "max_output_chars",
        "max_output_bytes",
        "max_provider_response_bytes",
    ]
    assert [field.name for field in dataclasses.fields(result)] == ["content"]
    assert not hasattr(request, "provider")
    assert not hasattr(request, "model")
    assert not hasattr(request, "headers")
    assert not hasattr(request, "tools")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.content = "changed"

    with pytest.raises(ValueError, match="non-empty text"):
        contracts.ModelCompletionResult(content=" \t\n")


def test_model_completion_port_settings_are_frozen_operator_values() -> None:
    contracts = _contracts()
    settings = contracts.ModelCompletionPortSettings(
        provider="openai",
        model="gpt-test",
        run_timeout_seconds=60,
        cancellation_cleanup_seconds=5,
    )

    assert settings.provider == "openai"
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.model = "changed"

    for field_name, invalid in (
        ("provider", " "),
        ("model", ""),
        ("run_timeout_seconds", True),
        ("run_timeout_seconds", 0),
        ("cancellation_cleanup_seconds", 1.5),
    ):
        values = dataclasses.asdict(settings)
        values[field_name] = invalid
        with pytest.raises(ValueError):
            contracts.ModelCompletionPortSettings(**values)


def test_model_completion_failure_has_stable_sanitized_shape() -> None:
    contracts = _contracts()
    failure = contracts.ModelCompletionFailure(
        "model_completion_transport_unavailable",
        contracts.ModelFailureDomain.SHARED_INFRASTRUCTURE,
    )

    assert str(failure) == "model_completion_transport_unavailable"
    assert failure.code == "model_completion_transport_unavailable"
    assert failure.domain is contracts.ModelFailureDomain.SHARED_INFRASTRUCTURE
    assert not hasattr(failure, "provider_body")
    assert not hasattr(failure, "cause")

    with pytest.raises(ValueError, match="stable failure code"):
        contracts.ModelCompletionFailure(
            "https://secret.example/?key=value",
            contracts.ModelFailureDomain.REQUEST,
        )
    with pytest.raises(TypeError, match="ModelFailureDomain"):
        contracts.ModelCompletionFailure("valid_code", "request")


def test_model_completion_protocols_expose_only_the_narrow_operations() -> None:
    contracts = _contracts()

    assert {
        name
        for name, value in inspect.getmembers(contracts.ModelCompletionPort)
        if not name.startswith("_") and (inspect.isfunction(value) or isinstance(value, property))
    } == {"capabilities", "complete"}
    assert {
        name
        for name, value in inspect.getmembers(contracts.ManagedModelCompletionPort)
        if not name.startswith("_") and (inspect.isfunction(value) or isinstance(value, property))
    } == {
        "capabilities",
        "complete",
        "is_healthy",
        "shutdown",
        "wait_for_shutdown_completion",
    }


def test_host_model_completion_shim_reexports_package_contracts() -> None:
    package_contracts = _contracts()
    host_contracts = importlib.import_module("tldw_Server_API.app.core.MCP_unified.interfaces.model_completion")

    for name in package_contracts.__all__:
        assert getattr(host_contracts, name) is getattr(package_contracts, name)
