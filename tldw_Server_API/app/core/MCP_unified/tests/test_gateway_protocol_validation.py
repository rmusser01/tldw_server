"""Tests for bounded, process-isolated MCP JSON Schema validation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema.validators import validator_for
from mcp_unified.gateway.protocol_errors import GatewayApplicationError
from mcp_unified.gateway.protocol_limits import GatewayLimits
from mcp_unified.gateway.protocol_profiles import PROTOCOL_PROFILES

pytestmark = pytest.mark.unit

_FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "mcp_protocol"


def _validation_api() -> Any:
    """Import the production API inside tests so fixture tests can run at RED."""

    from mcp_unified.gateway import protocol_validation

    return protocol_validation


def _crash_worker(*_args: object) -> None:
    """Exit abnormally inside a real spawned validation child."""

    os._exit(7)


def _hang_worker(*_args: object) -> None:
    """Keep a real spawned child alive until cancellation or shutdown reaps it."""

    time.sleep(60)


class _TrackingSpawnContext:
    """Delegate to spawn while retaining every real Process handle."""

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self.processes: list[multiprocessing.Process] = []

    def Pipe(self, duplex: bool = True) -> tuple[Any, Any]:  # noqa: N802
        return self._context.Pipe(duplex=duplex)

    def Process(self, *args: object, **kwargs: object) -> multiprocessing.Process:  # noqa: N802
        process = self._context.Process(*args, **kwargs)
        self.processes.append(process)
        return process


class _ReapCheckingSemaphore:
    """Assert each permit is released only after all started children are reaped."""

    def __init__(self, context: _TrackingSpawnContext) -> None:
        self._context = context
        self.acquired = 0
        self.released = 0

    async def acquire(self) -> bool:
        self.acquired += 1
        return True

    def release(self) -> None:
        assert self._context.processes
        assert all(not process.is_alive() for process in self._context.processes)
        assert all(process.exitcode is not None for process in self._context.processes)
        self.released += 1


async def _wait_for_process(context: _TrackingSpawnContext) -> None:
    for _ in range(500):
        if context.processes and context.processes[-1].pid is not None:
            return
        await asyncio.sleep(0.002)
    pytest.fail("validation worker did not start")


def test_pinned_official_schema_manifest_hashes_and_literal_vectors() -> None:
    """Fixture drift or a schema that rejects basic protocol vectors must fail offline."""

    manifest = json.loads((_FIXTURE_ROOT / "manifest.json").read_text("utf-8"))
    assert manifest["upstream"] == {
        "repository": "https://github.com/modelcontextprotocol/modelcontextprotocol",
        "commit": "5f5440bb26a62e2cf3440b92da5a667efa03b267",
        "license": "Apache-2.0",
    }
    assert [entry["revision"] for entry in manifest["fixtures"]] == list(
        PROTOCOL_PROFILES
    )

    request = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    result = {"jsonrpc": "2.0", "id": 1, "result": {}}
    for entry in manifest["fixtures"]:
        fixture_path = _FIXTURE_ROOT / entry["path"]
        raw = fixture_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert entry["url"].endswith(
            f"/{manifest['upstream']['commit']}/schema/{entry['revision']}/schema.json"
        )

        schema = json.loads(raw)
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        validator = validator_class(schema)
        validator.validate(request)
        validator.validate(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema", "instance"),
    [
        ({"type": "object"}, {"value": 1}),
        ({"type": "array"}, [1, 2]),
        ({"type": "string"}, "value"),
        ({"type": "number"}, 1.5),
        ({"type": "boolean"}, True),
        ({"type": "null"}, None),
    ],
)
async def test_draft_2020_12_accepts_every_finite_json_root(
    schema: dict[str, Any],
    instance: object,
) -> None:
    """A validator hard-coded to object roots would reject valid modern output."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    try:
        await manager.validate(
            schema,
            instance,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "schema", "instance", "reason_code"),
    [
        (
            replace(GatewayLimits(), max_schema_bytes=64),
            {"type": "string", "description": "x" * 80},
            "ok",
            "schema_too_large",
        ),
        (
            replace(GatewayLimits(), max_schema_depth=2),
            {"allOf": [{"allOf": [{"type": "string"}]}]},
            "ok",
            "schema_too_deep",
        ),
        (
            replace(GatewayLimits(), max_schema_subschemas=2),
            {"allOf": [{"type": "string"}, {"minLength": 1}]},
            "ok",
            "schema_too_complex",
        ),
        (
            replace(GatewayLimits(), max_schema_refs=1),
            {"$defs": {"x": {}}, "allOf": [{"$ref": "#/$defs/x"}, {"$ref": "#/$defs/x"}]},
            "ok",
            "schema_ref_limit",
        ),
        (
            replace(GatewayLimits(), max_schema_pattern_chars=3),
            {"type": "string", "pattern": "abcd"},
            "ok",
            "schema_pattern_limit",
        ),
        (
            replace(GatewayLimits(), max_schema_pattern_chars=3),
            {"type": "object", "patternProperties": {"abcd": {}}},
            {},
            "schema_pattern_limit",
        ),
        (
            replace(GatewayLimits(), max_json_depth=2),
            {},
            {"one": {"two": {"three": 3}}},
            "instance_too_deep",
        ),
    ],
)
async def test_parent_preflight_rejects_each_literal_complexity_limit(
    limits: GatewayLimits,
    schema: dict[str, Any],
    instance: object,
    reason_code: str,
) -> None:
    """Removing any preflight bound must expose a corresponding failing vector."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager(limits=limits)
    with pytest.raises(GatewayApplicationError) as raised:
        await manager.validate(
            schema,
            instance,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    assert raised.value.reason_code == reason_code
    assert manager.live_process_count == 0
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema", "reason_code"),
    [
        ({"$ref": "https://example.invalid/schema.json"}, "schema_external_ref"),
        ({"$ref": "other-schema.json#/$defs/value"}, "schema_external_ref"),
        ({"$ref": "#/$defs/missing"}, "schema_unresolved_ref"),
        (
            {"$schema": "https://json-schema.org/draft/2019-09/schema"},
            "schema_dialect_unsupported",
        ),
    ],
)
async def test_preflight_rejects_external_unresolved_refs_and_wrong_dialect(
    schema: dict[str, Any],
    reason_code: str,
) -> None:
    """External resolution or dialect fallback would violate the closed validator boundary."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    with pytest.raises(GatewayApplicationError) as raised:
        await manager.validate(
            schema,
            None,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    assert raised.value.reason_code == reason_code
    assert manager.live_process_count == 0
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema", "instance", "reason_code"),
    [
        ({1: {"type": "integer"}}, 1, "schema_not_json"),
        ({}, {1: "value"}, "instance_not_json"),
        ({}, ("not", "a", "json", "array"), "instance_not_json"),
    ],
)
async def test_preflight_rejects_non_string_json_object_keys(
    schema: dict[Any, Any],
    instance: object,
    reason_code: str,
) -> None:
    """Python's permissive JSON encoder must not silently stringify mapping keys."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    with pytest.raises(GatewayApplicationError) as raised:
        await manager.validate(
            schema,
            instance,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    assert raised.value.reason_code == reason_code
    assert manager.live_process_count == 0
    await manager.close()


@pytest.mark.asyncio
async def test_tool_input_and_output_validation_return_safe_literal_failures() -> None:
    """Invalid arguments or structured output must never cross the runtime boundary."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    profile = PROTOCOL_PROFILES["2026-07-28"]
    try:
        for schema, instance, root_mode in [
            (
                {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                },
                {"count": "wrong"},
                "object",
            ),
            ({"type": "array", "items": {"type": "integer"}}, [1, "wrong"], "any"),
        ]:
            with pytest.raises(GatewayApplicationError) as raised:
                await manager.validate(
                    schema,
                    instance,
                    profile=profile,
                    root_mode=root_mode,
                )
            assert raised.value.reason_code == "schema_validation_failed"

        with pytest.raises(GatewayApplicationError) as raised:
            await manager.validate(
                {"type": "array"},
                [],
                profile=profile,
                root_mode="object",
            )
        assert raised.value.reason_code == "schema_root_not_object"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_input_and_output_instance_bytes_use_their_distinct_limits() -> None:
    """Output validation must not inherit the larger request-input byte ceiling."""

    api = _validation_api()
    limits = replace(
        GatewayLimits(),
        max_input_line_bytes=256,
        max_output_line_bytes=256,
        max_result_bytes=100,
    )
    manager = api.GatewaySchemaValidationManager(limits=limits)
    value = "x" * 120
    try:
        await manager.validate(
            {"type": "string"},
            value,
            profile=PROTOCOL_PROFILES["2026-07-28"],
            instance_role="input",
        )
        with pytest.raises(GatewayApplicationError) as raised:
            await manager.validate(
                {"type": "string"},
                value,
                profile=PROTOCOL_PROFILES["2026-07-28"],
                instance_role="output",
            )
        assert raised.value.reason_code == "instance_too_large"
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", list(PROTOCOL_PROFILES))
async def test_each_profile_default_and_declared_dialect_validates_locally(
    version: str,
) -> None:
    """Selecting Draft 2020-12 must not silently break the two Draft 7 profiles."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    profile = PROTOCOL_PROFILES[version]
    try:
        await manager.validate({"type": "integer"}, 1, profile=profile)
        await manager.validate(
            {"$schema": profile.schema_dialect, "type": "integer"},
            1,
            profile=profile,
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_catastrophic_pattern_times_out_and_next_validation_succeeds() -> None:
    """A regex worker timeout must not poison the manager's next permit or worker."""

    api = _validation_api()
    limits = replace(GatewayLimits(), schema_validation_timeout_seconds=1.0)
    manager = api.GatewaySchemaValidationManager(limits=limits)
    profile = PROTOCOL_PROFILES["2026-07-28"]
    try:
        with pytest.raises(GatewayApplicationError) as raised:
            await manager.validate(
                {"type": "string", "pattern": "^(a+)+$"},
                "a" * 80 + "!",
                profile=profile,
            )
        assert raised.value.reason_code == "schema_validation_timeout"
        assert manager.live_process_count == 0

        await manager.validate({"type": "integer"}, 1, profile=profile)
        assert manager.live_process_count == 0
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "invalid", "crash", "timeout"])
async def test_worker_permit_is_released_only_after_real_process_is_reaped(
    outcome: str,
) -> None:
    """Success, failure, crash, and timeout must share reap-before-release cleanup."""

    api = _validation_api()
    context = _TrackingSpawnContext()
    worker_target = {
        "crash": _crash_worker,
        "timeout": _hang_worker,
    }.get(outcome)
    limits = replace(
        GatewayLimits(),
        schema_validation_timeout_seconds=0.15 if outcome == "timeout" else 1.0,
    )
    manager = api.GatewaySchemaValidationManager(
        limits=limits,
        process_context=context,
        _worker_target=worker_target,
    )
    semaphore = _ReapCheckingSemaphore(context)
    manager._semaphore = semaphore

    try:
        if outcome == "success":
            await manager.validate(
                {"type": "integer"},
                1,
                profile=PROTOCOL_PROFILES["2026-07-28"],
            )
        else:
            instance = "wrong" if outcome == "invalid" else 1
            with pytest.raises(GatewayApplicationError):
                await manager.validate(
                    {"type": "integer"},
                    instance,
                    profile=PROTOCOL_PROFILES["2026-07-28"],
                )
        assert semaphore.acquired == 1
        assert semaphore.released == 1
        assert manager.live_process_count == 0
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancellation_reaps_real_child_before_releasing_permit() -> None:
    """Cancelling a validation coroutine must not orphan its spawned worker."""

    api = _validation_api()
    context = _TrackingSpawnContext()
    manager = api.GatewaySchemaValidationManager(
        process_context=context,
        _worker_target=_hang_worker,
    )
    semaphore = _ReapCheckingSemaphore(context)
    manager._semaphore = semaphore

    task = asyncio.create_task(
        manager.validate(
            {"type": "integer"},
            1,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    )
    await _wait_for_process(context)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert semaphore.acquired == 1
    assert semaphore.released == 1
    assert manager.live_process_count == 0
    await manager.close()


@pytest.mark.asyncio
async def test_manager_close_reaps_live_children_releases_permits_and_rejects_work() -> None:
    """Shutdown must synchronously drain real validation children before returning."""

    api = _validation_api()
    context = _TrackingSpawnContext()
    manager = api.GatewaySchemaValidationManager(
        process_context=context,
        _worker_target=_hang_worker,
    )
    semaphore = _ReapCheckingSemaphore(context)
    manager._semaphore = semaphore

    task = asyncio.create_task(
        manager.validate(
            {"type": "integer"},
            1,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    )
    await _wait_for_process(context)
    await manager.close()

    with pytest.raises(GatewayApplicationError) as raised:
        await task
    assert raised.value.reason_code == "schema_validator_closed"
    assert semaphore.acquired == 1
    assert semaphore.released == 1
    assert manager.live_process_count == 0

    with pytest.raises(GatewayApplicationError) as closed:
        await manager.validate(
            {"type": "integer"},
            1,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    assert closed.value.reason_code == "schema_validator_closed"
