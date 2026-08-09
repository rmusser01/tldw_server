"""Tests for bounded, process-isolated MCP JSON Schema validation."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import multiprocessing
import os
import signal
import time
from dataclasses import replace
from functools import partial
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


def _ignore_sigterm_worker(ready: Any, *_args: object) -> None:
    """Stay alive through terminate so cleanup must exercise kill and reap."""

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    time.sleep(60)


def _ignore_sigterm_queue_worker(ready: Any, *_args: object) -> None:
    """Queue readiness from each SIGTERM-ignoring worker."""

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.put(os.getpid())
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


class _PostKillProcess:
    """Wrap a real child and record whether post-kill join receives time."""

    def __init__(self, process: multiprocessing.Process) -> None:
        self._process = process
        self.killed = False
        self.post_kill_join_timeouts: list[float | None] = []
        self.consume_post_kill_wait = False

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode

    def start(self) -> None:
        self._process.start()

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self.killed = True
        self._process.kill()

    def join(self, timeout: float | None = None) -> None:
        if self.killed:
            self.post_kill_join_timeouts.append(timeout)
            if self.consume_post_kill_wait and timeout is not None:
                self.consume_post_kill_wait = False
                time.sleep(timeout)
                self._process.join(0)
                return
        self._process.join(timeout)


class _PostKillTrackingContext(_TrackingSpawnContext):
    """Return real process proxies that expose post-kill scheduling evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.processes: list[Any] = []

    def Process(self, *args: object, **kwargs: object) -> _PostKillProcess:  # noqa: N802
        wrapped = _PostKillProcess(self._context.Process(*args, **kwargs))
        self.processes.append(wrapped)
        return wrapped


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


class _CountingSemaphore:
    """Record releases without assuming unrelated concurrent children are reaped."""

    def __init__(self) -> None:
        self.acquired = 0
        self.released = 0

    async def acquire(self) -> bool:
        self.acquired += 1
        return True

    def release(self) -> None:
        self.released += 1


class _FakeReceiver:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    """Controllable process double for cleanup ordering and deadline tests."""

    def __init__(self, *, clock: list[float] | None = None) -> None:
        self.alive = True
        self.exitcode: int | None = None
        self.fail_reap = False
        self.clock = clock
        self.calls: list[tuple[str, float | None]] = []

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.calls.append(("terminate", None))

    def kill(self) -> None:
        self.calls.append(("kill", None))
        if not self.fail_reap:
            self.alive = False
            self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(("join", timeout))
        if self.clock is not None and timeout is not None:
            self.clock[0] += timeout


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
    assert [entry["revision"] for entry in manifest["fixtures"]] == list(PROTOCOL_PROFILES)

    request = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    result = {"jsonrpc": "2.0", "id": 1, "result": {}}
    for entry in manifest["fixtures"]:
        fixture_path = _FIXTURE_ROOT / entry["path"]
        raw = fixture_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert entry["url"].endswith(f"/{manifest['upstream']['commit']}/schema/{entry['revision']}/schema.json")

        schema = json.loads(raw)
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        validator = validator_class(schema)
        validator.validate(request)
        validator.validate(result)


def test_validate_schema_exposes_the_compile_only_public_signature() -> None:
    """Task 3 needs an explicit compile API with no instance parameter."""

    api = _validation_api()
    signature = inspect.signature(api.GatewaySchemaValidationManager.validate_schema)
    assert list(signature.parameters) == ["self", "schema", "profile", "root_mode"]
    assert signature.parameters["profile"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["root_mode"].default == "any"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema", "version", "root_mode"),
    [
        ({"type": "object", "required": ["value"]}, "2026-07-28", "any"),
        ({"const": 42}, "2026-07-28", "any"),
        (False, "2026-07-28", "any"),
        ({"type": "object", "required": ["legacy"]}, "2025-06-18", "object"),
    ],
)
async def test_validate_schema_compiles_without_fabricating_an_instance(
    schema: dict[str, Any] | bool,
    version: str,
    root_mode: str,
) -> None:
    """Required, const, false, and legacy schemas compile without instance checks."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    try:
        await manager.validate_schema(
            schema,
            profile=PROTOCOL_PROFILES[version],
            root_mode=root_mode,
        )
        assert manager.live_process_count == 0
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_validate_schema_returns_bounded_invalid_schema() -> None:
    """A schema compilation failure must expose only the bounded reason code."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    try:
        with pytest.raises(GatewayApplicationError) as raised:
            await manager.validate_schema(
                {"type": 7},  # type: ignore[dict-item]
                profile=PROTOCOL_PROFILES["2026-07-28"],
            )
        assert raised.value.reason_code == "invalid_schema"
        assert manager.live_process_count == 0
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "schema", "root_mode", "reason_code"),
    [
        (
            replace(GatewayLimits(), max_schema_bytes=64),
            {"description": "x" * 80},
            "any",
            "schema_too_large",
        ),
        (
            replace(GatewayLimits(), max_schema_depth=2),
            {"allOf": [{"allOf": [{"type": "string"}]}]},
            "any",
            "schema_too_deep",
        ),
        (
            replace(GatewayLimits(), max_schema_subschemas=2),
            {"allOf": [{"type": "string"}, {"minLength": 1}]},
            "any",
            "schema_too_complex",
        ),
        (
            replace(GatewayLimits(), max_schema_refs=1),
            {"$defs": {"x": {}}, "allOf": [{"$ref": "#/$defs/x"}, {"$ref": "#/$defs/x"}]},
            "any",
            "schema_ref_limit",
        ),
        (
            replace(GatewayLimits(), max_schema_pattern_chars=3),
            {"pattern": "abcd"},
            "any",
            "schema_pattern_limit",
        ),
        (
            GatewayLimits(),
            {"$ref": "https://example.invalid/schema.json"},
            "any",
            "schema_external_ref",
        ),
        (
            GatewayLimits(),
            {"$ref": "#/$defs/missing"},
            "any",
            "schema_unresolved_ref",
        ),
        (GatewayLimits(), {"type": "array"}, "object", "schema_root_not_object"),
    ],
)
async def test_validate_schema_applies_every_parent_preflight_boundary(
    limits: GatewayLimits,
    schema: dict[str, Any],
    root_mode: str,
    reason_code: str,
) -> None:
    """Compile-only work must fail before spawn on every parent schema boundary."""

    api = _validation_api()
    context = _TrackingSpawnContext()
    manager = api.GatewaySchemaValidationManager(limits=limits, process_context=context)
    with pytest.raises(GatewayApplicationError) as raised:
        await manager.validate_schema(
            schema,
            profile=PROTOCOL_PROFILES["2026-07-28"],
            root_mode=root_mode,
        )
    assert raised.value.reason_code == reason_code
    assert context.processes == []
    await manager.close()


@pytest.mark.asyncio
async def test_validate_schema_only_checks_catastrophic_regex_syntax() -> None:
    """Compilation must not evaluate a valid catastrophic regular expression."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    started = time.monotonic()
    try:
        await manager.validate_schema(
            {"type": "string", "pattern": "^(a+)+$"},
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
        assert time.monotonic() - started < 1.0
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "invalid", "crash", "timeout"])
async def test_validate_schema_reaps_before_releasing_its_process_permit(
    outcome: str,
) -> None:
    """Every compile verdict must reap its real worker before permit release."""

    api = _validation_api()
    context = _TrackingSpawnContext()
    worker_target = {"crash": _crash_worker, "timeout": _hang_worker}.get(outcome)
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
    schema: dict[str, Any] = {"type": 7} if outcome == "invalid" else {"type": "integer"}
    try:
        if outcome == "success":
            await manager.validate_schema(
                schema,
                profile=PROTOCOL_PROFILES["2026-07-28"],
            )
        else:
            with pytest.raises(GatewayApplicationError):
                await manager.validate_schema(
                    schema,
                    profile=PROTOCOL_PROFILES["2026-07-28"],
                )
        assert semaphore.acquired == 1
        assert semaphore.released == 1
        assert manager.live_process_count == 0
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", ["cancel", "close"])
async def test_validate_schema_cancel_and_close_reap_live_workers(
    shutdown: str,
) -> None:
    """Cancellation and close must drain compile-only children before release."""

    api = _validation_api()
    context = _TrackingSpawnContext()
    manager = api.GatewaySchemaValidationManager(
        process_context=context,
        _worker_target=_hang_worker,
    )
    semaphore = _ReapCheckingSemaphore(context)
    manager._semaphore = semaphore
    task = asyncio.create_task(
        manager.validate_schema(
            {"type": "integer"},
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    )
    await _wait_for_process(context)
    if shutdown == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await manager.close()
    else:
        await manager.close()
        with pytest.raises(GatewayApplicationError) as raised:
            await task
        assert raised.value.reason_code == "schema_validator_closed"
    assert semaphore.acquired == 1
    assert semaphore.released == 1
    assert manager.live_process_count == 0


@pytest.mark.asyncio
async def test_validate_still_checks_instances_after_compile_runner_refactor() -> None:
    """A real JSON instance must never be confused with compile-only mode."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    try:
        with pytest.raises(GatewayApplicationError) as raised:
            await manager.validate(
                {"const": None},
                1,
                profile=PROTOCOL_PROFILES["2026-07-28"],
            )
        assert raised.value.reason_code == "schema_validation_failed"
        await manager.validate(
            {"const": None},
            None,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    finally:
        await manager.close()


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


@pytest.mark.asyncio
@pytest.mark.timeout(2)
@pytest.mark.parametrize("target", ["schema", "instance"])
async def test_preflight_rejects_self_referential_json_without_hanging(target: str) -> None:
    """A Python container cycle must be rejected before encoding or worker spawn."""

    api = _validation_api()
    cycle: dict[str, Any] = {}
    cycle["self"] = cycle
    schema: object = cycle if target == "schema" else {}
    instance: object = cycle if target == "instance" else None
    expected = "schema_not_json" if target == "schema" else "instance_not_json"
    manager = api.GatewaySchemaValidationManager()
    with pytest.raises(GatewayApplicationError) as raised:
        await manager.validate(
            schema,  # type: ignore[arg-type]
            instance,  # type: ignore[arg-type]
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    assert raised.value.reason_code == expected
    assert manager.live_process_count == 0
    await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["schema", "instance"])
async def test_preflight_enforces_depth_before_python_json_encoder_limit(target: str) -> None:
    """Deep acyclic JSON must become a bounded gateway error, never RecursionError."""

    api = _validation_api()
    nested: object = None
    for _ in range(1_100):
        nested = [nested]
    limits = replace(GatewayLimits(), max_schema_depth=32, max_json_depth=32)
    manager = api.GatewaySchemaValidationManager(limits=limits)
    expected = "schema_too_deep" if target == "schema" else "instance_too_deep"
    with pytest.raises(GatewayApplicationError) as raised:
        await manager.validate(
            nested if target == "schema" else {},  # type: ignore[arg-type]
            nested if target == "instance" else None,  # type: ignore[arg-type]
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    assert raised.value.reason_code == expected
    await manager.close()


@pytest.mark.asyncio
async def test_parent_rejects_a_forged_profile_dialect_before_worker_spawn() -> None:
    """Only the two accepted dialect literals may select a validator."""

    api = _validation_api()
    context = _TrackingSpawnContext()
    manager = api.GatewaySchemaValidationManager(process_context=context)
    forged = replace(
        PROTOCOL_PROFILES["2026-07-28"],
        schema_dialect="https://json-schema.org/draft/2019-09/schema",
    )
    with pytest.raises(GatewayApplicationError) as raised:
        await manager.validate({}, None, profile=forged)
    assert raised.value.reason_code == "schema_dialect_unsupported"
    assert context.processes == []
    await manager.close()


def test_worker_registry_never_uses_default_url_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker's resolver must fail locally even if parent preflight is bypassed."""

    import urllib.request

    api = _validation_api()
    retrieved: list[object] = []

    def forbidden_retrieval(*args: object, **kwargs: object) -> object:
        retrieved.append((args, kwargs))
        raise AssertionError("network retrieval attempted")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_retrieval)
    receiver, sender = multiprocessing.get_context("spawn").Pipe(duplex=False)
    api._schema_validation_worker(
        sender,
        b'{"$ref":"https://example.invalid/schema.json"}',
        b"null",
        PROTOCOL_PROFILES["2026-07-28"].schema_dialect,
    )
    verdict = json.loads(receiver.recv_bytes())
    receiver.close()
    assert verdict[0] in {"invalid", "internal"}
    assert retrieved == []


@pytest.mark.asyncio
async def test_schema_keywords_inside_instance_payload_keywords_are_not_traversed() -> None:
    """Annotation payloads named like schema keywords must not consume schema limits."""

    api = _validation_api()
    limits = replace(
        GatewayLimits(),
        max_schema_subschemas=2,
        max_schema_refs=1,
        max_schema_pattern_chars=1,
    )
    payload = {
        "$ref": "https://must-not-resolve.invalid",
        "pattern": "x" * 500,
        "patternProperties": {"x" * 500: {"$ref": "remote"}},
    }
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "const": payload,
                "enum": [payload],
                "default": payload,
                "examples": [payload],
            }
        },
    }
    manager = api.GatewaySchemaValidationManager(limits=limits)
    try:
        await manager.validate(
            schema,
            {"payload": payload},
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_embedded_resources_and_changed_base_resolve_without_retrieval() -> None:
    """Relative refs may target submitted embedded resources under their resolved ids."""

    api = _validation_api()
    schema = {
        "$id": "https://example.invalid/root.json",
        "$defs": {
            "child": {
                "$id": "child.json",
                "$anchor": "integer",
                "type": "integer",
            }
        },
        "properties": {
            "first": {"$ref": "child.json"},
            "second": {"$ref": "child.json#integer"},
        },
        "type": "object",
    }
    manager = api.GatewaySchemaValidationManager()
    try:
        await manager.validate(
            schema,
            {"first": 1, "second": 2},
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_close_drains_multiple_saturated_real_workers() -> None:
    """Shutdown must reap every running permit holder under full saturation."""

    api = _validation_api()
    context = _TrackingSpawnContext()
    limits = replace(GatewayLimits(), max_schema_validation_processes=2)
    manager = api.GatewaySchemaValidationManager(
        limits=limits,
        process_context=context,
        _worker_target=_hang_worker,
    )
    tasks = [
        asyncio.create_task(
            manager.validate(
                {"type": "integer"},
                index,
                profile=PROTOCOL_PROFILES["2026-07-28"],
            )
        )
        for index in range(3)
    ]
    for _ in range(500):
        if len(context.processes) == 2 and manager.live_process_count == 2:
            break
        await asyncio.sleep(0.002)
    else:
        pytest.fail("validation workers did not saturate")

    await manager.close()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(outcome, GatewayApplicationError) for outcome in outcomes)
    assert all(not process.is_alive() for process in context.processes)
    assert all(process.exitcode is not None for process in context.processes)
    assert manager.live_process_count == 0


@pytest.mark.asyncio
async def test_close_attempts_later_children_before_reporting_cleanup_failure() -> None:
    """One unreapable child must not prevent cleanup attempts for every sibling."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager()
    semaphore = _CountingSemaphore()
    manager._semaphore = semaphore
    processes = [_FakeProcess(), _FakeProcess(), _FakeProcess()]
    manager._live_processes = set(processes)
    manager._receivers = {process: _FakeReceiver() for process in processes}
    first = tuple(manager._live_processes)[0]
    first.fail_reap = True

    with pytest.raises(RuntimeError, match="could not be reaped"):
        await manager.close()

    assert all(any(call[0] == "terminate" for call in process.calls) for process in processes)
    assert all(any(call[0] == "kill" for call in process.calls) for process in processes)
    assert manager.live_process_count == 1
    assert semaphore.released == 2


@pytest.mark.asyncio
async def test_close_uses_one_global_graceful_shutdown_deadline() -> None:
    """Per-child waits must share, rather than multiply, the configured grace period."""

    api = _validation_api()
    clock = [100.0]
    limits = replace(GatewayLimits(), graceful_shutdown_timeout_seconds=0.12)
    manager = api.GatewaySchemaValidationManager(limits=limits, _clock=lambda: clock[0])
    processes = [_FakeProcess(clock=clock) for _ in range(3)]
    manager._live_processes = set(processes)
    manager._receivers = {process: _FakeReceiver() for process in processes}

    await manager.close()

    assert clock[0] <= 100.12
    assert all(any(call[0] == "terminate" for call in process.calls) for process in processes)
    assert all(any(call[0] == "kill" for call in process.calls) for process in processes)
    assert manager.live_process_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema", "limit_changes", "reason_code"),
    [
        (
            {"definitions": {"legacy": {"pattern": "abcd"}}},
            {"max_schema_pattern_chars": 3},
            "schema_pattern_limit",
        ),
        (
            {"dependencies": {"legacy": {"pattern": "abcd"}}},
            {"max_schema_pattern_chars": 3},
            "schema_pattern_limit",
        ),
        (
            {"$ref": "#/default", "default": {"pattern": "abcd"}},
            {"max_schema_pattern_chars": 3},
            "schema_pattern_limit",
        ),
        (
            {"$ref": "#/default", "default": {"type": "integer"}},
            {"max_schema_subschemas": 1},
            "schema_too_complex",
        ),
        (
            {"$ref": "#/default", "default": {"$ref": "#"}},
            {"max_schema_refs": 1},
            "schema_ref_limit",
        ),
    ],
)
async def test_2020_compatibility_and_referenced_schema_nodes_consume_limits(
    schema: dict[str, Any],
    limit_changes: dict[str, int],
    reason_code: str,
) -> None:
    """Every reachable schema node must consume the matching parent-side bound."""

    api = _validation_api()
    manager = api.GatewaySchemaValidationManager(limits=replace(GatewayLimits(), **limit_changes))
    with pytest.raises(GatewayApplicationError) as raised:
        await manager.validate(
            schema,
            1,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    assert raised.value.reason_code == reason_code
    assert manager.live_process_count == 0
    await manager.close()


@pytest.mark.timeout(2)
def test_reference_cycle_terminates_without_double_counting() -> None:
    """Following local refs must visit each target once while counting each ref once."""

    api = _validation_api()
    schema = {
        "$ref": "#/default",
        "default": {"$ref": "#", "pattern": "x"},
    }
    subschemas, refs, pattern_chars, *_ = api._inspect_schema_keywords(
        schema,
        dialect=PROTOCOL_PROFILES["2026-07-28"].schema_dialect,
    )
    assert (subschemas, len(refs), pattern_chars) == (2, 2, 1)


def test_structural_resource_reached_by_absolute_ref_is_counted_once() -> None:
    """A discovered `$id` resource must retain one visited identity after resolution."""

    api = _validation_api()
    schema = {
        "$id": "https://example.invalid/root.json",
        "$defs": {
            "child": {
                "$id": "child.json",
                "pattern": "x",
            }
        },
        "$ref": "child.json",
    }
    subschemas, refs, pattern_chars, *_ = api._inspect_schema_keywords(
        schema,
        dialect=PROTOCOL_PROFILES["2026-07-28"].schema_dialect,
    )
    assert (subschemas, len(refs), pattern_chars) == (2, 1, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("opaque_id", ["urn:example:child", "custom:opaque-child"])
async def test_fragment_refs_resolve_inside_submitted_opaque_resources(
    opaque_id: str,
) -> None:
    """Opaque resource bases must retain their URI for pointer and anchor fragments."""

    api = _validation_api()
    schema = {
        "$id": "urn:example:root",
        "$defs": {
            "child": {
                "$id": opaque_id,
                "$anchor": "child",
                "$defs": {
                    "value": {
                        "$anchor": "value",
                        "type": "integer",
                    }
                },
                "allOf": [
                    {"$ref": "#/$defs/value"},
                    {"$ref": "#value"},
                ],
            }
        },
        "$ref": f"{opaque_id}#child",
    }
    manager = api.GatewaySchemaValidationManager()
    try:
        await manager.validate(
            schema,
            1,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", ["close", "cancel"])
async def test_sigterm_ignoring_worker_gets_post_kill_reap_budget(
    shutdown: str,
) -> None:
    """Real SIGTERM survivors must be killed and joined before permit release."""

    if not hasattr(signal, "SIGTERM") or not hasattr(signal, "SIGKILL"):
        pytest.skip("POSIX process signals are required")
    api = _validation_api()
    context = _PostKillTrackingContext()
    ready = context._context.Event()
    limits = replace(GatewayLimits(), graceful_shutdown_timeout_seconds=0.3)
    manager = api.GatewaySchemaValidationManager(
        limits=limits,
        process_context=context,
        _worker_target=partial(_ignore_sigterm_worker, ready),
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
    for _ in range(500):
        if ready.is_set():
            break
        await asyncio.sleep(0.002)
    else:
        pytest.fail("SIGTERM-ignoring worker did not become ready")

    if shutdown == "close":
        await manager.close()
        with pytest.raises(GatewayApplicationError) as raised:
            await task
        assert raised.value.reason_code == "schema_validator_closed"
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await manager.close()

    process = context.processes[0]
    assert process.killed
    assert process.exitcode == -signal.SIGKILL
    assert any(timeout is not None and timeout > 0 for timeout in process.post_kill_join_timeouts)
    assert semaphore.released == 1
    assert manager.live_process_count == 0


@pytest.mark.asyncio
async def test_close_kills_and_reaps_every_real_sigterm_survivor() -> None:
    """A shared deadline must retain post-kill scheduling for every live child."""

    if not hasattr(signal, "SIGTERM") or not hasattr(signal, "SIGKILL"):
        pytest.skip("POSIX process signals are required")
    api = _validation_api()
    context = _PostKillTrackingContext()
    ready = context._context.Queue()
    limits = replace(
        GatewayLimits(),
        max_schema_validation_processes=2,
        graceful_shutdown_timeout_seconds=0.4,
    )
    manager = api.GatewaySchemaValidationManager(
        limits=limits,
        process_context=context,
        _worker_target=partial(_ignore_sigterm_queue_worker, ready),
    )
    tasks = [
        asyncio.create_task(
            manager.validate(
                {"type": "integer"},
                index,
                profile=PROTOCOL_PROFILES["2026-07-28"],
            )
        )
        for index in range(2)
    ]
    for _ in range(500):
        if len(context.processes) == 2:
            break
        await asyncio.sleep(0.002)
    else:
        pytest.fail("validation workers did not start")
    ready.get(timeout=5)
    ready.get(timeout=5)

    await manager.close()
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(outcome, GatewayApplicationError) for outcome in outcomes)
    assert all(process.killed for process in context.processes)
    assert all(process.exitcode == -signal.SIGKILL for process in context.processes)
    assert all(
        any(timeout is not None and timeout > 0 for timeout in process.post_kill_join_timeouts)
        for process in context.processes
    )
    assert manager.live_process_count == 0


@pytest.mark.asyncio
async def test_cancellation_preserves_cancelled_error_when_cleanup_reports_failure() -> None:
    """Cleanup diagnostics must not replace an active task cancellation."""

    api = _validation_api()
    context = _TrackingSpawnContext()
    manager = api.GatewaySchemaValidationManager(
        process_context=context,
        _worker_target=_hang_worker,
    )
    task = asyncio.create_task(
        manager.validate(
            {"type": "integer"},
            1,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    )
    await _wait_for_process(context)
    original_cleanup = manager._cleanup_process
    manager._cleanup_process = lambda *_args, **_kwargs: [  # type: ignore[method-assign]
        "injected cleanup failure"
    ]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.live_process_count == 1

    manager._cleanup_process = original_cleanup  # type: ignore[method-assign]
    await manager.close()
    assert manager.live_process_count == 0


@pytest.mark.asyncio
async def test_validation_cleanup_does_not_require_python311_task_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal validation must work when the current task has no `cancelling()` API."""

    api = _validation_api()
    monkeypatch.setattr(api.asyncio, "current_task", lambda: object())
    manager = api.GatewaySchemaValidationManager()
    try:
        await manager.validate(
            {"type": "integer"},
            1,
            profile=PROTOCOL_PROFILES["2026-07-28"],
        )
    finally:
        await manager.close()


@pytest.mark.parametrize("boolean_schema", [True, False])
def test_referenced_boolean_schema_uses_one_canonical_location(
    boolean_schema: bool,
) -> None:
    """A boolean in `$defs` must not be counted again when reached by pointer."""

    api = _validation_api()
    schema = {"$defs": {"target": boolean_schema}, "$ref": "#/$defs/target"}
    subschemas, refs, pattern_chars, *_ = api._inspect_schema_keywords(
        schema,
        dialect=PROTOCOL_PROFILES["2026-07-28"].schema_dialect,
    )
    assert (subschemas, len(refs), pattern_chars) == (2, 1, 0)


def test_unrelated_repeated_boolean_schemas_keep_distinct_locations() -> None:
    """Python's singleton booleans must not collapse independent schema locations."""

    api = _validation_api()
    schema = {"$defs": {"first": True, "second": True, "third": False}}
    subschemas, refs, pattern_chars, *_ = api._inspect_schema_keywords(
        schema,
        dialect=PROTOCOL_PROFILES["2026-07-28"].schema_dialect,
    )
    assert (subschemas, len(refs), pattern_chars) == (4, 0, 0)


@pytest.mark.parametrize("boolean_schema", [True, False])
def test_referenced_boolean_schema_honors_exact_subschema_boundary(
    boolean_schema: bool,
) -> None:
    """Root plus one referenced boolean fits two nodes and exceeds one node."""

    api = _validation_api()
    schema = {"$defs": {"target": boolean_schema}, "$ref": "#/$defs/target"}
    encoded = api._preflight_schema(
        schema,
        profile=PROTOCOL_PROFILES["2026-07-28"],
        limits=replace(GatewayLimits(), max_schema_subschemas=2),
        root_mode="any",
    )
    assert json.loads(encoded) == schema
    with pytest.raises(GatewayApplicationError) as raised:
        api._preflight_schema(
            schema,
            profile=PROTOCOL_PROFILES["2026-07-28"],
            limits=replace(GatewayLimits(), max_schema_subschemas=1),
            root_mode="any",
        )
    assert raised.value.reason_code == "schema_too_complex"


@pytest.mark.parametrize(
    ("descendant", "expected_pattern_chars"),
    [(True, 0), (False, 0), ({"pattern": "x"}, 1)],
)
def test_embedded_resource_descendant_aliases_share_one_physical_location(
    descendant: bool | dict[str, str],
    expected_pattern_chars: int,
) -> None:
    """Containing- and child-resource refs must not recount one physical node."""

    api = _validation_api()
    schema = {
        "$id": "https://example.invalid/root",
        "$ref": "#/properties/embedded/properties/x",
        "properties": {
            "embedded": {
                "$id": "child",
                "$ref": "#/properties/x",
                "properties": {"x": descendant},
            }
        },
    }
    subschemas, refs, pattern_chars, *_ = api._inspect_schema_keywords(
        schema,
        dialect=PROTOCOL_PROFILES["2026-07-28"].schema_dialect,
    )
    assert (subschemas, len(refs), pattern_chars) == (
        3,
        2,
        expected_pattern_chars,
    )


@pytest.mark.parametrize("descendant", [True, False, {"type": "integer"}])
def test_embedded_resource_descendant_aliases_honor_exact_subschema_boundary(
    descendant: bool | dict[str, str],
) -> None:
    """Three physical schemas fit limit three and exceed limit two despite aliases."""

    api = _validation_api()
    schema = {
        "$id": "https://example.invalid/root",
        "$ref": "#/properties/embedded/properties/x",
        "properties": {
            "embedded": {
                "$id": "child",
                "$ref": "#/properties/x",
                "properties": {"x": descendant},
            }
        },
    }
    encoded = api._preflight_schema(
        schema,
        profile=PROTOCOL_PROFILES["2026-07-28"],
        limits=replace(GatewayLimits(), max_schema_subschemas=3),
        root_mode="any",
    )
    assert json.loads(encoded) == schema
    with pytest.raises(GatewayApplicationError) as raised:
        api._preflight_schema(
            schema,
            profile=PROTOCOL_PROFILES["2026-07-28"],
            limits=replace(GatewayLimits(), max_schema_subschemas=2),
            root_mode="any",
        )
    assert raised.value.reason_code == "schema_too_complex"


@pytest.mark.asyncio
async def test_close_uses_shared_wait_phases_for_all_real_children() -> None:
    """One force-wait consumer must not starve a later survivor's reap budget."""

    if not hasattr(signal, "SIGTERM") or not hasattr(signal, "SIGKILL"):
        pytest.skip("POSIX process signals are required")
    api = _validation_api()
    context = _PostKillTrackingContext()
    ready = context._context.Queue()
    limits = replace(
        GatewayLimits(),
        max_schema_validation_processes=2,
        graceful_shutdown_timeout_seconds=0.4,
    )
    manager = api.GatewaySchemaValidationManager(
        limits=limits,
        process_context=context,
        _worker_target=partial(_ignore_sigterm_queue_worker, ready),
    )
    semaphore = _ReapCheckingSemaphore(context)
    manager._semaphore = semaphore
    tasks = [
        asyncio.create_task(
            manager.validate(
                {"type": "integer"},
                index,
                profile=PROTOCOL_PROFILES["2026-07-28"],
            )
        )
        for index in range(2)
    ]
    for _ in range(500):
        if len(context.processes) == 2:
            break
        await asyncio.sleep(0.002)
    else:
        pytest.fail("validation workers did not start")
    ready.get(timeout=5)
    ready.get(timeout=5)
    ordered = tuple(manager._live_processes)
    first, later = ordered
    first.consume_post_kill_wait = True

    try:
        await manager.close()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(outcome, GatewayApplicationError) for outcome in outcomes)
        assert any(timeout is not None and timeout > 0 for timeout in later.post_kill_join_timeouts)
        assert semaphore.released == 2
        assert manager.live_process_count == 0
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(1)
