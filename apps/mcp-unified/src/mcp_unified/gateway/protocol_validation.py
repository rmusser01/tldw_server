"""Bounded JSON Schema validation in disposable spawned worker processes."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any, Literal, TypeAlias

from jsonschema import Draft7Validator, Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .protocol_errors import GatewayApplicationError
from .protocol_limits import GatewayLimits
from .protocol_profiles import GatewayProtocolProfile
from .runtime import GatewayJSONValue

SchemaWorkerVerdict: TypeAlias = tuple[Literal["ok", "invalid", "internal"], str]
SchemaRootMode: TypeAlias = Literal["any", "object"]
SchemaInstanceRole: TypeAlias = Literal["input", "output"]
_SCHEMA_WORKER_MAX_VERDICT_BYTES = 4_096
_SCHEMA_REF_KEYS = frozenset({"$ref", "$dynamicRef", "$recursiveRef"})
_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_DRAFT_7 = "http://json-schema.org/draft-07/schema#"
_WORKER_INVALID_CODES = frozenset({"invalid_schema", "schema_validation_failed"})


def _validation_error(reason_code: str, message: str) -> GatewayApplicationError:
    """Build a bounded safe validation error without input or schema details."""

    return GatewayApplicationError(message, reason_code=reason_code)


def _send_worker_verdict(connection: Connection, verdict: SchemaWorkerVerdict) -> None:
    """Send one small JSON verdict and never expose validator diagnostics."""

    payload = json.dumps(verdict, separators=(",", ":")).encode("utf-8")
    if len(payload) > _SCHEMA_WORKER_MAX_VERDICT_BYTES:
        payload = b'["internal","schema_validation_worker_failed"]'
    try:
        connection.send_bytes(payload)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _schema_validation_worker(
    connection: Connection,
    schema_json: bytes,
    instance_json: bytes,
    dialect: str,
) -> None:
    """Compile and validate one bounded value in a disposable child process."""

    verdict: SchemaWorkerVerdict
    try:
        schema = json.loads(schema_json)
        instance = json.loads(instance_json)
        validator_class = Draft202012Validator if dialect == _DRAFT_2020_12 else Draft7Validator
        validator_class.check_schema(schema)
        validator_class(schema).validate(instance)
        verdict = ("ok", "")
    except SchemaError:
        verdict = ("invalid", "invalid_schema")
    except ValidationError:
        verdict = ("invalid", "schema_validation_failed")
    except Exception:  # noqa: BLE001 - child must reduce every validator failure to a safe verdict
        verdict = ("internal", "schema_validation_worker_failed")
    _send_worker_verdict(connection, verdict)
    connection.close()


def _serialize_json(value: object, *, reason_code: str, message: str) -> bytes:
    """Serialize finite JSON with deterministic, compact UTF-8 encoding."""

    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if any(not isinstance(key, str) for key in current):
                raise _validation_error(reason_code, message)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
        elif current is not None and not isinstance(
            current, (bool, int, float, str)
        ):
            raise _validation_error(reason_code, message)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise _validation_error(reason_code, message) from exc
    return encoded


def _json_container_count_and_depth(value: object) -> tuple[int, int]:
    """Return mapping count and maximum container depth using an iterative walk."""

    mapping_count = 0
    max_depth = 0
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        if isinstance(current, dict):
            mapping_count += 1
            max_depth = max(max_depth, depth)
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            max_depth = max(max_depth, depth)
            stack.extend((child, depth + 1) for child in current)
    return mapping_count, max_depth


def _resolve_json_pointer(schema: object, reference: str, anchors: set[str]) -> bool:
    """Return whether an internal JSON Pointer or anchor resolves locally."""

    if reference in {"", "#"}:
        return True
    fragment = reference[1:]
    if not fragment.startswith("/"):
        return fragment in anchors

    current = schema
    for encoded_token in fragment[1:].split("/"):
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (IndexError, TypeError, ValueError):
                return False
            continue
        return False
    return True


def _preflight_schema(
    schema: object,
    *,
    profile: GatewayProtocolProfile,
    limits: GatewayLimits,
    root_mode: SchemaRootMode,
) -> bytes:
    """Apply bounded structural and reference checks before spawning a worker."""

    schema_json = _serialize_json(
        schema,
        reason_code="schema_not_json",
        message="Schema must be finite JSON",
    )
    if len(schema_json) > limits.max_schema_bytes:
        raise _validation_error("schema_too_large", "Schema exceeds the configured limit")

    mapping_count, max_depth = _json_container_count_and_depth(schema)
    if max_depth > limits.max_schema_depth:
        raise _validation_error("schema_too_deep", "Schema exceeds the configured depth")
    if mapping_count > limits.max_schema_subschemas:
        raise _validation_error(
            "schema_too_complex",
            "Schema exceeds the configured subschema limit",
        )
    if root_mode == "object" and (
        not isinstance(schema, dict) or schema.get("type") != "object"
    ):
        raise _validation_error(
            "schema_root_not_object",
            "Schema must declare an object root",
        )

    if isinstance(schema, dict):
        declared_dialect = schema.get("$schema")
        if declared_dialect is not None and declared_dialect != profile.schema_dialect:
            raise _validation_error(
                "schema_dialect_unsupported",
                "Schema dialect is not supported for this protocol revision",
            )

    refs: list[str] = []
    anchors: set[str] = set()
    pattern_chars = 0
    stack = [schema]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            anchor = current.get("$anchor")
            if isinstance(anchor, str):
                anchors.add(anchor)
            dynamic_anchor = current.get("$dynamicAnchor")
            if isinstance(dynamic_anchor, str):
                anchors.add(dynamic_anchor)
            for key, child in current.items():
                if key in _SCHEMA_REF_KEYS:
                    if not isinstance(child, str) or not child.startswith("#"):
                        raise _validation_error(
                            "schema_external_ref",
                            "Schema references must resolve within the submitted schema",
                        )
                    refs.append(child)
                if key == "pattern" and isinstance(child, str):
                    pattern_chars += len(child)
                if key == "patternProperties" and isinstance(child, dict):
                    pattern_chars += sum(
                        len(pattern)
                        for pattern in child
                        if isinstance(pattern, str)
                    )
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)

    if len(refs) > limits.max_schema_refs:
        raise _validation_error("schema_ref_limit", "Schema exceeds the reference limit")
    if pattern_chars > limits.max_schema_pattern_chars:
        raise _validation_error(
            "schema_pattern_limit",
            "Schema exceeds the pattern-character limit",
        )
    if any(not _resolve_json_pointer(schema, reference, anchors) for reference in refs):
        raise _validation_error(
            "schema_unresolved_ref",
            "Schema contains an unresolved local reference",
        )
    return schema_json


def _preflight_instance(
    instance: object,
    limits: GatewayLimits,
    instance_role: SchemaInstanceRole,
) -> bytes:
    """Serialize an instance and reject excessive recursive depth before spawn."""

    instance_json = _serialize_json(
        instance,
        reason_code="instance_not_json",
        message="Validation instance must be finite JSON",
    )
    limit_bytes = (
        limits.max_input_line_bytes
        if instance_role == "input"
        else limits.max_result_bytes
    )
    if len(instance_json) > limit_bytes:
        raise _validation_error(
            "instance_too_large",
            "Validation instance exceeds the configured limit",
        )
    _, max_depth = _json_container_count_and_depth(instance)
    if max_depth > limits.max_json_depth:
        raise _validation_error(
            "instance_too_deep",
            "Validation instance exceeds the configured depth",
        )
    return instance_json


class GatewaySchemaValidationManager:
    """Own disposable schema workers and their exact concurrency permits."""

    def __init__(
        self,
        limits: GatewayLimits = GatewayLimits(),
        *,
        process_context: Any | None = None,
        _worker_target: Callable[..., None] | None = None,
    ) -> None:
        self._limits = limits
        self._context = process_context or multiprocessing.get_context("spawn")
        self._worker_target = _worker_target or _schema_validation_worker
        self._semaphore = asyncio.Semaphore(limits.max_schema_validation_processes)
        self._live_processes: set[multiprocessing.Process] = set()
        self._receivers: dict[multiprocessing.Process, Connection] = {}
        self._closed = False

    @property
    def live_process_count(self) -> int:
        """Return the number of validation children not yet reaped."""

        return len(self._live_processes)

    async def validate(
        self,
        schema: dict[str, GatewayJSONValue] | bool,
        instance: GatewayJSONValue,
        *,
        profile: GatewayProtocolProfile,
        root_mode: SchemaRootMode = "any",
        instance_role: SchemaInstanceRole = "input",
    ) -> None:
        """Validate one finite JSON value or raise a bounded application error."""

        if self._closed:
            raise _validation_error("schema_validator_closed", "Schema validator is closed")
        if root_mode not in {"any", "object"}:
            raise ValueError("root_mode must be any or object")
        if instance_role not in {"input", "output"}:
            raise ValueError("instance_role must be input or output")

        schema_json = _preflight_schema(
            schema,
            profile=profile,
            limits=self._limits,
            root_mode=root_mode,
        )
        instance_json = _preflight_instance(instance, self._limits, instance_role)

        await self._semaphore.acquire()
        process: multiprocessing.Process | None = None
        receiver: Connection | None = None
        sender: Connection | None = None
        permit_owned = True
        if self._closed:
            self._semaphore.release()
            raise _validation_error("schema_validator_closed", "Schema validator is closed")

        try:
            receiver, sender = self._context.Pipe(duplex=False)
            process = self._context.Process(
                target=self._worker_target,
                args=(sender, schema_json, instance_json, profile.schema_dialect),
                daemon=True,
            )
            process.start()
            self._live_processes.add(process)
            self._receivers[process] = receiver
            permit_owned = False
            sender.close()
            sender = None
            verdict = await self._receive_verdict(process, receiver)
            if verdict[0] == "ok":
                return
            if verdict[0] == "invalid" and verdict[1] in _WORKER_INVALID_CODES:
                raise _validation_error(verdict[1], "Schema validation failed")
            raise _validation_error(
                "schema_validation_worker_failed",
                "Schema validation worker failed",
            )
        finally:
            if sender is not None:
                sender.close()
            if process is not None and process in self._live_processes:
                self._cleanup_process(process)
            elif receiver is not None:
                receiver.close()
            if permit_owned:
                self._semaphore.release()

    async def _receive_verdict(
        self,
        process: multiprocessing.Process,
        receiver: Connection,
    ) -> SchemaWorkerVerdict:
        """Poll one pipe without creating an executor thread that could outlive close."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._limits.schema_validation_timeout_seconds
        while True:
            if self._closed:
                raise _validation_error(
                    "schema_validator_closed",
                    "Schema validator is closed",
                )
            if receiver.poll(0):
                try:
                    payload = receiver.recv_bytes(_SCHEMA_WORKER_MAX_VERDICT_BYTES)
                    decoded = json.loads(payload)
                except (EOFError, OSError, UnicodeDecodeError, json.JSONDecodeError):
                    return ("internal", "schema_validation_worker_failed")
                if (
                    isinstance(decoded, list)
                    and len(decoded) == 2
                    and decoded[0] in {"ok", "invalid", "internal"}
                    and isinstance(decoded[1], str)
                ):
                    return (decoded[0], decoded[1])
                return ("internal", "schema_validation_worker_failed")
            if not process.is_alive():
                return ("internal", "schema_validation_worker_failed")
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise _validation_error(
                    "schema_validation_timeout",
                    "Schema validation timed out",
                )
            await asyncio.sleep(min(0.005, remaining))

    def _cleanup_process(self, process: multiprocessing.Process) -> None:
        """Terminate if needed, kill if needed, reap, then release one permit."""

        if process not in self._live_processes:
            return
        receiver = self._receivers.pop(process)
        try:
            if process.is_alive():
                process.terminate()
            process.join(timeout=min(0.5, self._limits.graceful_shutdown_timeout_seconds))
            if process.is_alive():
                process.kill()
                process.join(timeout=min(0.5, self._limits.graceful_shutdown_timeout_seconds))
            if process.is_alive() or process.exitcode is None:
                raise RuntimeError("schema validation child could not be reaped")
        finally:
            receiver.close()
        self._live_processes.remove(process)
        self._semaphore.release()

    async def close(self) -> None:
        """Reject new work and reap every currently running validation child."""

        if self._closed and not self._live_processes:
            return
        self._closed = True
        for process in tuple(self._live_processes):
            self._cleanup_process(process)


__all__ = [
    "GatewaySchemaValidationManager",
    "SchemaInstanceRole",
    "SchemaRootMode",
    "SchemaWorkerVerdict",
]
