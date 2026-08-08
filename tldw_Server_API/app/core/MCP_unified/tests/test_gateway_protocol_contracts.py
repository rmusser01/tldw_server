"""Public contract tests for the strict MCP stdio protocol layer."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import FrozenInstanceError
from math import inf, nan
from types import MappingProxyType
from typing import Any

import pytest
from mcp_unified.gateway import (
    CURRENT_PROTOCOL_VERSION,
    PREFERRED_LEGACY_PROTOCOL_VERSION,
    PROTOCOL_PROFILES,
    SUPPORTED_LEGACY_PROTOCOL_VERSIONS,
    SUPPORTED_MODERN_PROTOCOL_VERSIONS,
    SUPPORTED_PROTOCOL_VERSIONS,
    GatewayApplicationError,
    GatewayCancellationToken,
    GatewayCoreRuntime,
    GatewayInvalidApplicationResult,
    GatewayLimits,
    GatewayRequestContext,
    GatewayResourceNotFound,
    GatewayResourceTemplateRuntime,
    GatewayResultTooLarge,
    GatewayRuntime,
    GatewayStdioServer,
    GatewayToolExecutionError,
    handle_stdio_line,
)

pytestmark = pytest.mark.unit


class _CoreOnlyRuntime:
    """Small structurally compatible runtime without package-specific aliases."""

    name = "contract-runtime"
    version = "1.0"

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        return []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        return {"ok": True}

    async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        return []

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        return {"contents": []}

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        return []

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        return {"messages": []}


def test_protocol_versions_and_profiles_match_the_approved_matrix() -> None:
    """A changed revision flag must fail before dispatch can drift by version."""

    assert CURRENT_PROTOCOL_VERSION == "2026-07-28"
    assert PREFERRED_LEGACY_PROTOCOL_VERSION == "2025-11-25"
    assert SUPPORTED_PROTOCOL_VERSIONS == (
        "2026-07-28",
        "2025-11-25",
        "2025-06-18",
        "2025-03-26",
        "2024-11-05",
    )
    assert SUPPORTED_MODERN_PROTOCOL_VERSIONS == ("2026-07-28",)
    assert SUPPORTED_LEGACY_PROTOCOL_VERSIONS == (
        "2025-11-25",
        "2025-06-18",
        "2025-03-26",
        "2024-11-05",
    )
    assert isinstance(PROTOCOL_PROFILES, MappingProxyType)

    actual = {
        version: (
            profile.era,
            profile.requires_initialize,
            profile.accepts_batches,
            profile.requires_result_type,
            profile.cache_hints,
            profile.supports_titles,
            profile.supports_icons,
            profile.supports_resource_links,
            profile.structured_content_mode,
            profile.missing_resource_code,
            profile.schema_dialect,
        )
        for version, profile in PROTOCOL_PROFILES.items()
    }
    assert actual == {
        "2026-07-28": (
            "modern", False, False, True, True, True, True, True,
            "any", -32602, "https://json-schema.org/draft/2020-12/schema",
        ),
        "2025-11-25": (
            "legacy", True, False, False, False, True, True, True,
            "object", -32002, "https://json-schema.org/draft/2020-12/schema",
        ),
        "2025-06-18": (
            "legacy", True, False, False, False, True, False, True,
            "object", -32002, "http://json-schema.org/draft-07/schema#",
        ),
        "2025-03-26": (
            "legacy", True, True, False, False, False, False, False,
            "none", -32002, "http://json-schema.org/draft-07/schema#",
        ),
        "2024-11-05": (
            "legacy", True, False, False, False, False, False, False,
            "none", -32002, "http://json-schema.org/draft-07/schema#",
        ),
    }
    with pytest.raises(FrozenInstanceError):
        PROTOCOL_PROFILES[CURRENT_PROTOCOL_VERSION].era = "legacy"  # type: ignore[misc]


def test_gateway_limits_expose_every_approved_default() -> None:
    """A missing or changed resource bound must fail the public defaults contract."""

    assert GatewayLimits() == GatewayLimits(
        max_input_line_bytes=1_048_576,
        max_output_line_bytes=1_048_576,
        max_result_bytes=786_432,
        max_json_depth=64,
        max_in_flight=16,
        default_catalog_page_size=50,
        max_catalog_page_size=100,
        max_catalog_items=10_000,
        max_batch_items=100,
        max_requests_per_minute=600,
        request_burst=32,
        max_schema_bytes=262_144,
        max_schema_depth=32,
        max_schema_subschemas=1_024,
        max_schema_refs=256,
        max_schema_pattern_chars=4_096,
        max_schema_validation_processes=4,
        schema_validation_timeout_seconds=1.0,
        graceful_shutdown_timeout_seconds=5.0,
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "max_input_line_bytes",
        "max_output_line_bytes",
        "max_result_bytes",
        "max_json_depth",
        "max_in_flight",
        "default_catalog_page_size",
        "max_catalog_page_size",
        "max_catalog_items",
        "max_batch_items",
        "max_requests_per_minute",
        "request_burst",
        "max_schema_bytes",
        "max_schema_depth",
        "max_schema_subschemas",
        "max_schema_refs",
        "max_schema_pattern_chars",
        "max_schema_validation_processes",
        "schema_validation_timeout_seconds",
        "graceful_shutdown_timeout_seconds",
    ],
)
def test_gateway_limits_reject_boolean_values(field_name: str) -> None:
    """Boolean values must not pass as numeric resource limits."""

    with pytest.raises(ValueError, match=field_name):
        GatewayLimits(**{field_name: True})


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        ({"max_input_line_bytes": 0}, "max_input_line_bytes"),
        ({"max_output_line_bytes": 16_777_217}, "max_output_line_bytes"),
        ({"max_result_bytes": 0}, "max_result_bytes"),
        ({"max_json_depth": 257}, "max_json_depth"),
        ({"max_in_flight": 1_025}, "max_in_flight"),
        ({"default_catalog_page_size": 1_001}, "default_catalog_page_size"),
        ({"max_catalog_page_size": 0}, "max_catalog_page_size"),
        ({"max_catalog_items": 100_001}, "max_catalog_items"),
        ({"max_batch_items": 1_001}, "max_batch_items"),
        ({"max_requests_per_minute": 60_001}, "max_requests_per_minute"),
        ({"request_burst": 10_001}, "request_burst"),
        ({"max_schema_bytes": 4_194_305}, "max_schema_bytes"),
        ({"max_schema_depth": 129}, "max_schema_depth"),
        ({"max_schema_subschemas": 10_001}, "max_schema_subschemas"),
        ({"max_schema_refs": 4_097}, "max_schema_refs"),
        ({"max_schema_pattern_chars": 65_537}, "max_schema_pattern_chars"),
        ({"max_schema_validation_processes": 33}, "max_schema_validation_processes"),
        ({"schema_validation_timeout_seconds": 0.0}, "schema_validation_timeout_seconds"),
        ({"schema_validation_timeout_seconds": inf}, "schema_validation_timeout_seconds"),
        ({"graceful_shutdown_timeout_seconds": 60.1}, "graceful_shutdown_timeout_seconds"),
        ({"graceful_shutdown_timeout_seconds": nan}, "graceful_shutdown_timeout_seconds"),
    ],
)
def test_gateway_limits_reject_out_of_range_values(
    overrides: dict[str, int | float],
    field_name: str,
) -> None:
    """Each documented lower, upper, and finite-number bound must be enforced."""

    with pytest.raises(ValueError, match=field_name):
        GatewayLimits(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"max_output_line_bytes": 100, "max_result_bytes": 101},
            "max_result_bytes must not exceed max_output_line_bytes",
        ),
        (
            {"default_catalog_page_size": 101, "max_catalog_page_size": 100},
            "default_catalog_page_size must not exceed max_catalog_page_size",
        ),
        (
            {"max_catalog_page_size": 101, "max_catalog_items": 100},
            "max_catalog_items must not be less than max_catalog_page_size",
        ),
        (
            {"max_requests_per_minute": 31, "request_burst": 32},
            "request_burst must not exceed max_requests_per_minute",
        ),
    ],
)
def test_gateway_limits_reject_invalid_cross_field_relationships(
    overrides: dict[str, int],
    message: str,
) -> None:
    """Related bounds must stay internally coherent rather than being clamped."""

    with pytest.raises(ValueError, match=message):
        GatewayLimits(**overrides)


@pytest.mark.asyncio
async def test_cancellation_changes_state_once_bounds_reason_and_wakes_waiter() -> None:
    """The first cancellation must atomically wake waiters with bounded diagnostics."""

    token = GatewayCancellationToken()
    waiter = asyncio.create_task(token.wait())
    await asyncio.sleep(0)

    assert await asyncio.to_thread(token.cancel, "x" * 129) is True
    await asyncio.wait_for(waiter, timeout=1.0)
    assert token.cancelled is True
    assert token.is_cancelled() is True
    assert token.reason == "x" * 128
    assert token.cancel("second") is False
    assert token.reason == "x" * 128
    with pytest.raises(asyncio.CancelledError):
        token.raise_if_cancelled()


@pytest.mark.asyncio
async def test_cancellation_wait_returns_when_cancelled_before_registration() -> None:
    """A waiter registered after cancellation must not sleep forever."""

    token = GatewayCancellationToken()
    assert token.cancel() is True
    await asyncio.wait_for(token.wait(), timeout=1.0)
    assert token.reason is None


def test_request_context_preserves_typed_ids_and_compatibility_defaults() -> None:
    """Integer and string IDs must remain distinct at the runtime boundary."""

    int_context = GatewayRequestContext(request_id=1)
    str_context = GatewayRequestContext(request_id="1")

    assert int_context.request_id == 1
    assert str_context.request_id == "1"
    assert int_context.protocol_version is None
    assert int_context.protocol_era is None
    assert int_context.client_info is None
    assert int_context.client_capabilities == {}
    assert int_context.cancellation is None
    assert int_context.client_capabilities is not str_context.client_capabilities


def test_core_runtime_does_not_require_legacy_module_aliases() -> None:
    """Strict consumers must not implement package-specific module methods."""

    runtime = _CoreOnlyRuntime()

    assert isinstance(runtime, GatewayCoreRuntime)
    assert not isinstance(runtime, GatewayResourceTemplateRuntime)
    assert not hasattr(runtime, "list_modules")
    assert not hasattr(runtime, "get_modules_health")


def test_safe_application_errors_expose_only_approved_public_fields() -> None:
    """Error subclasses must not leak private payloads through their public state."""

    base = GatewayApplicationError(
        "Safe failure",
        reason_code="safe_failure",
    )
    tool = GatewayToolExecutionError(
        "Tool could not complete",
        reason_code="not_implemented",
    )
    missing = GatewayResourceNotFound()
    too_large = GatewayResultTooLarge(limit_bytes=123)
    invalid = GatewayInvalidApplicationResult()

    assert vars(base) == {
        "public_message": "Safe failure",
        "reason_code": "safe_failure",
        "kind": "application",
    }
    assert vars(tool) == {
        "public_message": "Tool could not complete",
        "reason_code": "not_implemented",
        "kind": "tool",
    }
    assert vars(missing) == {
        "public_message": "Resource not found",
        "reason_code": "resource_not_found",
        "kind": "resource",
    }
    assert vars(too_large) == {
        "public_message": "Application result exceeds the configured limit",
        "reason_code": "result_too_large",
        "kind": "application",
        "limit_bytes": 123,
    }
    assert vars(invalid) == {
        "public_message": "Application returned an invalid result",
        "reason_code": "invalid_application_result",
        "kind": "application",
    }


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("",), {"reason_code": "valid"}),
        (("x" * 513,), {"reason_code": "valid"}),
        ((123,), {"reason_code": "valid"}),
        (("Safe",), {"reason_code": "Invalid-Code"}),
        (("Safe",), {"reason_code": "a" * 65}),
        (("Safe",), {"reason_code": 1}),
        (("Safe",), {"reason_code": "valid", "kind": "private"}),
        (("Safe",), {"reason_code": "valid", "kind": []}),
    ],
)
def test_application_error_rejects_unsafe_public_shapes(
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    """Unbounded or unstable error classifications must fail locally."""

    with pytest.raises(ValueError):
        GatewayApplicationError(*args, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit_bytes", [True, 0, -1, 1.5, "1"])
def test_result_too_large_rejects_invalid_limit_metadata(limit_bytes: object) -> None:
    """Only positive integer configured limits may reach error metadata."""

    with pytest.raises(ValueError, match="limit_bytes"):
        GatewayResultTooLarge(limit_bytes=limit_bytes)  # type: ignore[arg-type]


def test_legacy_runtime_and_stdio_signatures_remain_compatible() -> None:
    """Adding strict contracts must not alter existing legacy call surfaces."""

    runtime_members = [
        name
        for name in GatewayRuntime.__dict__
        if not name.startswith("_")
    ]
    assert runtime_members == [
        "list_tools",
        "call_tool",
        "list_resources",
        "read_resource",
        "list_prompts",
        "get_prompt",
        "list_modules",
        "get_modules_health",
    ]
    assert list(inspect.signature(GatewayStdioServer).parameters) == [
        "runtime",
        "path",
        "metadata",
    ]
    stdio_signature = inspect.signature(handle_stdio_line)
    assert list(stdio_signature.parameters) == [
        "runtime",
        "line",
        "path",
        "metadata",
    ]
    assert stdio_signature.parameters["path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert stdio_signature.parameters["path"].default == "stdio://stdin"
    assert stdio_signature.parameters["metadata"].default is None


@pytest.mark.asyncio
async def test_legacy_stdio_keeps_independent_ping_behavior() -> None:
    """Compatibility stdio must still process a single request without lifecycle state."""

    runtime = _CoreOnlyRuntime()
    response = await handle_stdio_line(
        runtime,  # type: ignore[arg-type]
        '{"jsonrpc":"2.0","method":"ping","id":1}\n',
    )

    assert response is not None
    assert json.loads(response) == {
        "jsonrpc": "2.0",
        "result": {"pong": True},
        "id": 1,
    }
