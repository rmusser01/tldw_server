"""Bounded JSON Schema validation in disposable spawned worker processes."""

from __future__ import annotations

import asyncio
import json
import math
import multiprocessing
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any, Literal, TypeAlias
from urllib.parse import unquote, urldefrag, urljoin

from jsonschema import Draft7Validator, Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import NoSuchResource
from referencing.jsonschema import DRAFT7 as REFERENCING_DRAFT7
from referencing.jsonschema import DRAFT202012 as REFERENCING_DRAFT202012

from .protocol_errors import GatewayApplicationError
from .protocol_limits import GatewayLimits
from .protocol_profiles import GatewayProtocolProfile
from .runtime import GatewayJSONValue

SchemaWorkerVerdict: TypeAlias = tuple[Literal["ok", "invalid", "internal"], str]
SchemaRootMode: TypeAlias = Literal["any", "object"]
SchemaInstanceRole: TypeAlias = Literal["input", "output"]
_SCHEMA_WORKER_MAX_VERDICT_BYTES = 4_096
_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_DRAFT_7 = "http://json-schema.org/draft-07/schema#"
_VALIDATORS = {
    _DRAFT_2020_12: Draft202012Validator,
    _DRAFT_7: Draft7Validator,
}
_WORKER_INVALID_CODES = frozenset({"invalid_schema", "schema_validation_failed"})
_SPECIFICATIONS = {
    _DRAFT_2020_12: REFERENCING_DRAFT202012,
    _DRAFT_7: REFERENCING_DRAFT7,
}
_DIALECT_REF_KEYS = {
    _DRAFT_2020_12: frozenset({"$ref", "$dynamicRef"}),
    _DRAFT_7: frozenset({"$ref"}),
}


class _JSONStructureError(ValueError):
    """Classify an unsafe Python structure before recursive JSON encoding."""

    def __init__(self, kind: Literal["cycle", "depth", "key", "type", "number"]):
        super().__init__(kind)
        self.kind = kind


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


def _deny_schema_retrieval(uri: str) -> Any:
    """Fail every registry retrieval without consulting network-capable defaults."""

    raise NoSuchResource(ref=uri)


def _validator_for_dialect(dialect: str) -> type[Draft7Validator] | type[Draft202012Validator]:
    """Select only the two explicitly accepted schema dialects."""

    try:
        return _VALIDATORS[dialect]
    except KeyError as exc:
        raise ValueError("unsupported schema dialect") from exc


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
        validator_class = _validator_for_dialect(dialect)
        registry: Registry[Any] = Registry(retrieve=_deny_schema_retrieval)
        validator_class.check_schema(schema)
        validator_class(schema, registry=registry).validate(instance)
        verdict = ("ok", "")
    except SchemaError:
        verdict = ("invalid", "invalid_schema")
    except ValidationError:
        verdict = ("invalid", "schema_validation_failed")
    except Exception:  # noqa: BLE001 - child must reduce every validator failure to a safe verdict
        verdict = ("internal", "schema_validation_worker_failed")
    _send_worker_verdict(connection, verdict)
    connection.close()


def _validate_json_structure(value: object, *, max_depth: int) -> None:
    """Iteratively reject cycles, excessive depth, and non-JSON Python values."""

    active: set[int] = set()
    stack: list[tuple[object, int, bool]] = [(value, 1, False)]
    while stack:
        current, depth, leaving = stack.pop()
        if leaving:
            active.remove(id(current))
            continue
        if isinstance(current, (dict, list)):
            if depth > max_depth:
                raise _JSONStructureError("depth")
            identity = id(current)
            if identity in active:
                raise _JSONStructureError("cycle")
            active.add(identity)
            stack.append((current, depth, True))
            if isinstance(current, dict):
                if any(not isinstance(key, str) for key in current):
                    raise _JSONStructureError("key")
                children = current.values()
            else:
                children = current
            stack.extend((child, depth + 1, False) for child in children)
            continue
        if current is None or isinstance(current, (bool, int, str)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise _JSONStructureError("number")
            continue
        raise _JSONStructureError("type")


def _serialize_json(value: object, *, reason_code: str, message: str) -> bytes:
    """Serialize finite JSON with deterministic, compact UTF-8 encoding."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeEncodeError) as exc:
        raise _validation_error(reason_code, message) from exc
    return encoded


_UNRESOLVED = object()


def _resolve_fragment(
    resource: object,
    fragment: str,
    anchors: dict[str, object],
) -> object:
    """Return the submitted target of a decoded pointer or anchor."""

    if not fragment:
        return resource
    if not fragment.startswith("/"):
        return anchors.get(unquote(fragment), _UNRESOLVED)

    current = resource
    for encoded_token in fragment[1:].split("/"):
        token = unquote(encoded_token).replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
            continue
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (IndexError, TypeError, ValueError):
                return _UNRESOLVED
            continue
        return _UNRESOLVED
    return current


def _split_reference(base: str, reference: str) -> tuple[str, str]:
    """Resolve a reference exactly like ``referencing.Resolver.lookup``."""

    if reference.startswith("#"):
        return urldefrag(base).url, reference[1:]
    resolved = urldefrag(urljoin(base, reference))
    return resolved.url, resolved.fragment


def _schema_children(node: dict[str, object], dialect: str) -> list[object]:
    """Return only values occupying schema-valued keywords for one dialect."""

    specification = _SPECIFICATIONS[dialect]
    children = list(specification.subresources_of(node))
    if dialect == _DRAFT_2020_12:
        dependencies = node.get("dependencies")
        if isinstance(dependencies, dict):
            children.extend(value for value in dependencies.values() if isinstance(value, (dict, bool)))
    return children


def _inspect_schema_keywords(
    schema: object,
    *,
    dialect: str,
) -> tuple[
    int,
    list[tuple[str, str]],
    int,
    dict[str, object],
    dict[str, dict[str, object]],
]:
    """Inspect schema locations plus every successfully resolved local target."""

    subschema_count = 0
    refs: list[tuple[str, str]] = []
    pattern_chars = 0
    resources: dict[str, object] = {"": schema}
    anchors: dict[str, dict[str, object]] = {"": {}}
    resource_uri_by_identity: dict[int, str] = {id(schema): ""}
    visited: set[tuple[str, int]] = set()
    followed_refs: set[int] = set()
    stack: list[tuple[object, str, str]] = [(schema, "", "")]
    while True:
        while stack:
            current, inherited_base, inherited_resource_uri = stack.pop()
            if not isinstance(current, (dict, bool)):
                continue
            if isinstance(current, bool):
                subschema_count += 1
                continue

            base = inherited_base
            resource_uri = inherited_resource_uri
            identifier = _SPECIFICATIONS[dialect].id_of(current)
            if isinstance(identifier, str):
                base = urljoin(inherited_base, identifier)
                resource_uri = urldefrag(base).url
                resources[resource_uri] = current
                anchors.setdefault(resource_uri, {})
                resource_uri_by_identity[id(current)] = resource_uri
            visit_key = (resource_uri, id(current))
            if visit_key in visited:
                continue
            visited.add(visit_key)
            subschema_count += 1
            for anchor_keyword in ("$anchor", "$dynamicAnchor"):
                anchor = current.get(anchor_keyword)
                if isinstance(anchor, str):
                    anchors.setdefault(resource_uri, {})[anchor] = current
            for ref_keyword in _DIALECT_REF_KEYS[dialect]:
                reference = current.get(ref_keyword)
                if reference is not None:
                    refs.append((reference if isinstance(reference, str) else "\0", base))
            pattern = current.get("pattern")
            if isinstance(pattern, str):
                pattern_chars += len(pattern)
            pattern_properties = current.get("patternProperties")
            if isinstance(pattern_properties, dict):
                pattern_chars += sum(len(key) for key in pattern_properties)
            stack.extend((child, base, resource_uri) for child in _schema_children(current, dialect))

        made_progress = False
        for index, (reference, base) in enumerate(refs):
            if index in followed_refs or reference == "\0":
                continue
            resource_uri, fragment = _split_reference(base, reference)
            resource = resources.get(resource_uri)
            if resource is None:
                continue
            target = _resolve_fragment(
                resource,
                fragment,
                anchors.get(resource_uri, {}),
            )
            if target is _UNRESOLVED:
                continue
            followed_refs.add(index)
            target_resource_uri = resource_uri_by_identity.get(
                id(target),
                resource_uri,
            )
            stack.append((target, target_resource_uri, target_resource_uri))
            made_progress = True
        if not made_progress:
            break
    return subschema_count, refs, pattern_chars, resources, anchors


def _preflight_schema(
    schema: object,
    *,
    profile: GatewayProtocolProfile,
    limits: GatewayLimits,
    root_mode: SchemaRootMode,
) -> bytes:
    """Apply bounded structural and reference checks before spawning a worker."""

    if profile.schema_dialect not in _VALIDATORS:
        raise _validation_error(
            "schema_dialect_unsupported",
            "Schema dialect is not supported for this protocol revision",
        )
    try:
        _validate_json_structure(schema, max_depth=limits.max_schema_depth)
    except _JSONStructureError as exc:
        reason_code = "schema_too_deep" if exc.kind == "depth" else "schema_not_json"
        message = "Schema exceeds the configured depth" if exc.kind == "depth" else "Schema must be finite JSON"
        raise _validation_error(reason_code, message) from exc
    schema_json = _serialize_json(
        schema,
        reason_code="schema_not_json",
        message="Schema must be finite JSON",
    )
    if len(schema_json) > limits.max_schema_bytes:
        raise _validation_error("schema_too_large", "Schema exceeds the configured limit")

    if root_mode == "object" and (not isinstance(schema, dict) or schema.get("type") != "object"):
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

    subschema_count, refs, pattern_chars, resources, anchors = _inspect_schema_keywords(
        json.loads(schema_json),
        dialect=profile.schema_dialect,
    )

    if subschema_count > limits.max_schema_subschemas:
        raise _validation_error(
            "schema_too_complex",
            "Schema exceeds the configured subschema limit",
        )
    if len(refs) > limits.max_schema_refs:
        raise _validation_error("schema_ref_limit", "Schema exceeds the reference limit")
    if pattern_chars > limits.max_schema_pattern_chars:
        raise _validation_error(
            "schema_pattern_limit",
            "Schema exceeds the pattern-character limit",
        )
    for reference, base in refs:
        if reference == "\0":
            raise _validation_error(
                "schema_unresolved_ref",
                "Schema contains an unresolved local reference",
            )
        resource_uri, fragment = _split_reference(base, reference)
        resource = resources.get(resource_uri)
        if resource is None:
            raise _validation_error(
                "schema_external_ref",
                "Schema references must resolve within the submitted schema",
            )
        if (
            _resolve_fragment(
                resource,
                fragment,
                anchors.get(resource_uri, {}),
            )
            is _UNRESOLVED
        ):
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

    try:
        _validate_json_structure(instance, max_depth=limits.max_json_depth)
    except _JSONStructureError as exc:
        reason_code = "instance_too_deep" if exc.kind == "depth" else "instance_not_json"
        message = (
            "Validation instance exceeds the configured depth"
            if exc.kind == "depth"
            else "Validation instance must be finite JSON"
        )
        raise _validation_error(reason_code, message) from exc
    instance_json = _serialize_json(
        instance,
        reason_code="instance_not_json",
        message="Validation instance must be finite JSON",
    )
    limit_bytes = limits.max_input_line_bytes if instance_role == "input" else limits.max_result_bytes
    if len(instance_json) > limit_bytes:
        raise _validation_error(
            "instance_too_large",
            "Validation instance exceeds the configured limit",
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
        _clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limits = limits
        self._context = process_context or multiprocessing.get_context("spawn")
        self._worker_target = _worker_target or _schema_validation_worker
        self._clock = _clock
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
                cleanup_errors = self._cleanup_process(
                    process,
                    deadline=self._clock() + self._limits.graceful_shutdown_timeout_seconds,
                )
                current_task = asyncio.current_task()
                cancellation_active = current_task is not None and current_task.cancelling() > 0
                if cleanup_errors and not cancellation_active:
                    raise RuntimeError("; ".join(cleanup_errors))
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

    def _cleanup_process(
        self,
        process: multiprocessing.Process,
        *,
        deadline: float,
        terminate_first: bool = True,
    ) -> list[str]:
        """Attempt full cleanup and release only after verified process reaping."""

        if process not in self._live_processes:
            return []
        errors: list[str] = []
        receiver = self._receivers.get(process)
        try:
            try:
                alive = process.is_alive()
            except Exception as exc:  # noqa: BLE001 - cleanup must continue for siblings
                errors.append(f"schema validation child status failed: {type(exc).__name__}")
                alive = True
            if alive and terminate_first:
                try:
                    process.terminate()
                except Exception as exc:  # noqa: BLE001 - kill/join must still be attempted
                    errors.append(f"schema validation child terminate failed: {type(exc).__name__}")
            remaining = max(0.0, deadline - self._clock())
            graceful_deadline = self._clock() + remaining / 2
            try:
                process.join(timeout=max(0.0, graceful_deadline - self._clock()))
            except Exception as exc:  # noqa: BLE001 - kill/join must still be attempted
                errors.append(f"schema validation child join failed: {type(exc).__name__}")
            try:
                alive = process.is_alive()
            except Exception as exc:  # noqa: BLE001 - cleanup must remain fail closed
                errors.append(f"schema validation child status failed: {type(exc).__name__}")
                alive = True
            if alive:
                try:
                    process.kill()
                except Exception as exc:  # noqa: BLE001 - final join must still be attempted
                    errors.append(f"schema validation child kill failed: {type(exc).__name__}")
            try:
                process.join(timeout=max(0.0, deadline - self._clock()))
            except Exception as exc:  # noqa: BLE001 - reap status is checked below
                errors.append(f"schema validation child join failed: {type(exc).__name__}")
            try:
                reaped = not process.is_alive() and process.exitcode is not None
            except Exception as exc:  # noqa: BLE001 - cleanup must remain fail closed
                errors.append(f"schema validation child status failed: {type(exc).__name__}")
                reaped = False
            if not reaped:
                errors.append("schema validation child could not be reaped")
        finally:
            if receiver is not None:
                try:
                    receiver.close()
                except Exception as exc:  # noqa: BLE001 - reap accounting still must finish
                    errors.append(f"schema validation receiver close failed: {type(exc).__name__}")
        if reaped:
            self._receivers.pop(process, None)
            self._live_processes.remove(process)
            self._semaphore.release()
        return errors

    async def close(self) -> None:
        """Reject new work and reap every currently running validation child."""

        if self._closed and not self._live_processes:
            return
        self._closed = True
        errors: list[str] = []
        deadline = self._clock() + self._limits.graceful_shutdown_timeout_seconds
        processes = tuple(self._live_processes)
        for process in processes:
            try:
                alive = process.is_alive()
            except Exception as exc:  # noqa: BLE001 - every child must be signaled
                errors.append(f"schema validation child status failed: {type(exc).__name__}")
                alive = True
            if alive:
                try:
                    process.terminate()
                except Exception as exc:  # noqa: BLE001 - siblings must still be signaled
                    errors.append(f"schema validation child terminate failed: {type(exc).__name__}")
        for process in processes:
            errors.extend(
                self._cleanup_process(
                    process,
                    deadline=deadline,
                    terminate_first=False,
                )
            )
        if errors:
            raise RuntimeError("; ".join(errors))


__all__ = [
    "GatewaySchemaValidationManager",
    "SchemaInstanceRole",
    "SchemaRootMode",
    "SchemaWorkerVerdict",
]
