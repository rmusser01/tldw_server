"""Revision-aware lifecycle, dispatch, and admission for strict MCP stdio."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from .protocol_cancellation import GatewayCancellationToken
from .protocol_errors import (
    GatewayApplicationError,
    GatewayInvalidApplicationResult,
    GatewayResultTooLarge,
    GatewayToolExecutionError,
)
from .protocol_limits import GatewayLimits
from .protocol_pagination import GatewayCatalogPaginator
from .protocol_profiles import (
    CURRENT_PROTOCOL_VERSION,
    PREFERRED_LEGACY_PROTOCOL_VERSION,
    PROTOCOL_PROFILES,
    SUPPORTED_LEGACY_PROTOCOL_VERSIONS,
    SUPPORTED_MODERN_PROTOCOL_VERSIONS,
    SUPPORTED_PROTOCOL_VERSIONS,
    GatewayProtocolProfile,
)
from .protocol_projection import (
    _normalize_uri,
    project_application_error,
    project_descriptor,
    project_prompt_result,
    project_resource_result,
    project_tool_result,
)
from .protocol_validation import (
    GatewaySchemaValidationManager,
    _JSONStructureError,
    _validate_json_structure,
)
from .runtime import GatewayCoreRuntime, GatewayJSONValue, GatewayRequestContext

_GatewayProtocolWriter: TypeAlias = Callable[[GatewayJSONValue], Awaitable[None]]
_RequestKey: TypeAlias = tuple[type[str] | type[int], str | int]
_Response: TypeAlias = dict[str, GatewayJSONValue]

_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603
_ADMISSION_REJECTED = -32000
_UNSUPPORTED_PROTOCOL_VERSION = -32022
_SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"
_PROTOCOL_VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
_CLIENT_CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"
_CLIENT_INFO_KEY = "io.modelcontextprotocol/clientInfo"
_RESERVED_CONTEXT_KEYS = frozenset(
    {
        "cancellation",
        "client_capabilities",
        "client_info",
        "method",
        "path",
        "protocol_era",
        "protocol_version",
        "request_id",
        "transport",
    }
)


class _ProtocolFailure(Exception):
    """One already-bounded JSON-RPC failure raised inside dispatch."""

    def __init__(
        self,
        code: int,
        message: str,
        *,
        data: GatewayJSONValue | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(slots=True)
class _PreparedRequest:
    request_id: str | int
    method: str
    params: dict[str, Any]
    profile: GatewayProtocolProfile
    era: Literal["modern", "legacy"]
    client_info: dict[str, Any] | None
    client_capabilities: dict[str, Any]
    request_metadata: dict[str, GatewayJSONValue]
    initialize_result: dict[str, GatewayJSONValue] | None = None


@dataclass(slots=True)
class _ActiveRequest:
    key: _RequestKey
    token: GatewayCancellationToken
    prepared: _PreparedRequest
    task: asyncio.Task[_Response | None] | None = None


@dataclass(frozen=True, slots=True)
class _BatchEntry:
    response: _Response | None = None
    active: _ActiveRequest | None = None


def _error_response(
    request_id: str | int | None,
    code: int,
    message: str,
    *,
    data: GatewayJSONValue | None = None,
) -> _Response:
    error: dict[str, GatewayJSONValue] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _success_response(request_id: str | int, result: GatewayJSONValue) -> _Response:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _request_key(request_id: str | int) -> _RequestKey:
    return (type(request_id), request_id)


def _is_reserved_meta_key(key: str) -> bool:
    if "/" not in key:
        return False
    labels = key.split("/", 1)[0].split(".")
    return len(labels) >= 2 and labels[1] in {"mcp", "modelcontextprotocol"}


def _is_safe_protocol_version(value: str) -> bool:
    """Return whether a date-shaped MCP version is bounded and safe to echo."""

    digits = value.replace("-", "")
    return len(value) == 10 and value[4] == "-" and value[7] == "-" and digits.isascii() and digits.isdigit()


class _OutputLimiter:
    """Serialize and bound application results and final JSON-RPC envelopes."""

    def __init__(self, limits: GatewayLimits) -> None:
        self._limits = limits

    def bound(self, value: GatewayJSONValue) -> GatewayJSONValue | None:
        """Return the original value or the smallest safe fitting error."""

        if self._fits_output_line(value):
            return value

        request_id = self._response_id(value)
        for candidate in self._semantic_error_candidates(value, request_id):
            if self._fits_output_line(candidate):
                return candidate
        for candidate in self._result_too_large_candidates(request_id):
            if self._fits_output_line(candidate):
                return candidate
        for candidate in (
            _error_response(request_id, _INTERNAL_ERROR, "Internal error"),
            _error_response(None, _INTERNAL_ERROR, "Internal error"),
        ):
            if self._fits_output_line(candidate):
                return candidate
        return None

    def ensure_result_size(self, value: object) -> None:
        """Reject a raw application result above its pre-envelope limit."""

        try:
            size = len(self._json_bytes(value))
        except ValueError as exc:
            raise GatewayInvalidApplicationResult() from exc
        if size > self._limits.max_result_bytes:
            raise GatewayResultTooLarge(limit_bytes=self._limits.max_result_bytes)

    def _fits_output_line(self, value: GatewayJSONValue) -> bool:
        try:
            return len(self._json_bytes(value)) + 1 <= self._limits.max_output_line_bytes
        except ValueError:
            return False

    def _semantic_error_candidates(
        self,
        value: GatewayJSONValue,
        request_id: str | int | None,
    ) -> tuple[_Response, ...]:
        if not isinstance(value, dict):
            return ()
        raw_error = value.get("error")
        if not isinstance(raw_error, dict):
            return ()
        code = raw_error.get("code")
        message = raw_error.get("message")
        if (
            isinstance(code, bool)
            or not isinstance(code, int)
            or not isinstance(message, str)
            or not message
            or len(message) > 512
        ):
            return ()

        candidates: list[_Response] = []
        data = raw_error.get("data")
        if data is not None:
            candidates.append(_error_response(None, code, message, data=data))
        candidates.append(_error_response(request_id, code, message))
        if request_id is not None:
            candidates.append(_error_response(None, code, message))
        return tuple(candidates)

    def _result_too_large_candidates(
        self,
        request_id: str | int | None,
    ) -> tuple[_Response, ...]:
        data: dict[str, GatewayJSONValue] = {
            "reasonCode": "result_too_large",
            "kind": "application",
            "limitBytes": self._limits.max_output_line_bytes,
        }
        message = "Application result exceeds the configured limit"
        return (
            _error_response(request_id, -33001, message, data=data),
            _error_response(None, -33001, message, data=data),
            _error_response(request_id, -33001, message),
            _error_response(None, -33001, message),
        )

    def _json_bytes(self, value: object) -> bytes:
        try:
            _validate_json_structure(value, max_depth=self._limits.max_json_depth)
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (
            _JSONStructureError,
            RecursionError,
            TypeError,
            ValueError,
            UnicodeEncodeError,
        ) as exc:
            raise ValueError("value must be finite JSON") from exc

    @staticmethod
    def _response_id(value: GatewayJSONValue) -> str | int | None:
        if not isinstance(value, dict):
            return None
        request_id = value.get("id")
        if isinstance(request_id, (str, int)) and not isinstance(request_id, bool):
            return request_id
        return None


class GatewayProtocolConnection:
    """Own one strict stdio connection's revision and request lifecycle."""

    def __init__(
        self,
        runtime: GatewayCoreRuntime,
        writer: _GatewayProtocolWriter,
        *,
        limits: GatewayLimits = GatewayLimits(),
        metadata: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        _validation_manager: GatewaySchemaValidationManager | None = None,
    ) -> None:
        if not callable(writer):
            raise ValueError("writer must be callable")
        if not callable(clock):
            raise ValueError("clock must be callable")
        runtime_name = getattr(runtime, "name", None)
        runtime_version = getattr(runtime, "version", None)
        if (
            not isinstance(runtime_name, str)
            or not runtime_name
            or len(runtime_name) > 512
            or not isinstance(runtime_version, str)
            or not runtime_version
            or len(runtime_version) > 512
        ):
            raise ValueError("runtime name and version must be bounded strings")
        required_runtime_methods = (
            "list_tools",
            "call_tool",
            "list_resources",
            "read_resource",
            "list_prompts",
            "get_prompt",
        )
        if not all(callable(getattr(runtime, method_name, None)) for method_name in required_runtime_methods):
            raise ValueError("runtime must implement the MCP core methods")

        self._runtime = runtime
        self._writer = writer
        self._limits = limits
        self._clock = clock
        self._metadata = self._clone_metadata(metadata or {})
        self._server_meta: dict[str, GatewayJSONValue] = {
            _SERVER_INFO_KEY: {"name": runtime_name, "version": runtime_version}
        }
        self._output_limiter = _OutputLimiter(limits)
        self._validator = _validation_manager or GatewaySchemaValidationManager(limits)
        self._paginator = GatewayCatalogPaginator(limits)
        self._writer_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._era: Literal["modern", "legacy"] | None = None
        self._legacy_profile: GatewayProtocolProfile | None = None
        self._legacy_initializing: _RequestKey | None = None
        self._legacy_client_info: dict[str, GatewayJSONValue] | None = None
        self._legacy_client_capabilities: dict[str, GatewayJSONValue] = {}
        self._active: dict[_RequestKey, _ActiveRequest] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._task_errors: list[BaseException] = []
        self._closed = False
        self._validator_closed = False
        initial_time = self._read_clock()
        self._rate_tokens = float(limits.request_burst)
        self._rate_updated_at = initial_time

    async def receive(self, payload: GatewayJSONValue) -> None:
        """Validate, admit, schedule, and eventually serialize one line value."""

        if self._closed:
            raise RuntimeError("connection is closed")
        try:
            _validate_json_structure(payload, max_depth=self._limits.max_json_depth)
        except _JSONStructureError:
            await self._write_value(_error_response(None, _INVALID_REQUEST, "Invalid request"))
            return

        if isinstance(payload, list):
            await self._receive_batch(payload)
            return
        await self._receive_single(payload)

    async def wait_for_idle(self) -> None:
        """Wait until every admitted request and validation child is reaped."""

        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
            await asyncio.sleep(0)
        if self._task_errors:
            error = self._task_errors.pop(0)
            self._task_errors.clear()
            raise error

    async def shutdown(self) -> None:
        """Reject new work, cancel tracked requests, and drain bounded cleanup."""

        if self._closed and self._validator_closed:
            return
        self._closed = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._limits.graceful_shutdown_timeout_seconds

        async with self._state_lock:
            active = tuple(self._active.values())
            for request in active:
                request.token.cancel("connection_closed")
                if request.task is not None:
                    request.task.cancel()

        errors: list[BaseException] = []
        remaining = max(0.0, deadline - loop.time())
        try:
            await asyncio.wait_for(self.wait_for_idle(), timeout=remaining)
        except asyncio.TimeoutError:
            pending = tuple(self._tasks)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.wait(
                    pending,
                    timeout=max(0.0, deadline - loop.time()),
                )
            if self._tasks:
                errors.append(RuntimeError("connection shutdown timed out"))
        except asyncio.CancelledError as exc:
            errors.append(exc)
        except Exception as exc:  # noqa: BLE001 - preserve arbitrary writer failure through cleanup
            errors.append(exc)

        if not self._validator_closed:
            remaining = max(0.0, deadline - loop.time())
            if remaining <= 0:
                errors.append(RuntimeError("connection shutdown timed out"))
            else:
                try:
                    await asyncio.wait_for(self._validator.close(), timeout=remaining)
                    self._validator_closed = True
                except asyncio.CancelledError as exc:
                    errors.append(exc)
                except Exception as exc:  # noqa: BLE001 - aggregate arbitrary child-cleanup failure
                    errors.append(exc)
        if errors:
            raise errors[0]

    async def _receive_single(self, payload: GatewayJSONValue) -> None:
        immediate: _Response | None
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("connection is closed")
            prepared, immediate = self._prepare_locked(payload)
            if prepared is not None:
                active, immediate = self._admit_locked(prepared)
                if active is not None:
                    self._start_standalone_locked(active)
        if immediate is not None:
            await self._write_value(immediate)

    async def _receive_batch(self, payload: list[GatewayJSONValue]) -> None:
        immediate: _Response | None = None
        batch_task: asyncio.Task[None] | None = None
        async with self._state_lock:
            if self._closed:
                raise RuntimeError("connection is closed")
            if not payload:
                immediate = _error_response(
                    None,
                    _INVALID_REQUEST,
                    "Invalid request",
                )
            elif (
                len(payload) > self._limits.max_batch_items
                or any(isinstance(item, dict) and item.get("method") == "initialize" for item in payload)
                or self._legacy_profile is None
                or not self._legacy_profile.accepts_batches
            ):
                immediate = self._admission_error(None)
            else:
                entries: list[_BatchEntry] = []
                for item in payload:
                    prepared, response = self._prepare_locked(item)
                    if prepared is None:
                        entries.append(_BatchEntry(response=response))
                        continue
                    active, response = self._admit_locked(prepared)
                    if active is None:
                        entries.append(_BatchEntry(response=response))
                    else:
                        entries.append(_BatchEntry(active=active))
                batch_task = asyncio.create_task(self._run_batch(entries))
                self._track_task(batch_task)
        if immediate is not None:
            await self._write_value(immediate)
        del batch_task

    def _prepare_locked(
        self,
        payload: GatewayJSONValue,
    ) -> tuple[_PreparedRequest | None, _Response | None]:
        if not isinstance(payload, dict):
            return None, _error_response(
                None,
                _INVALID_REQUEST,
                "Invalid request",
            )
        if payload.get("jsonrpc") != "2.0":
            return None, _error_response(
                self._safe_response_id(payload),
                _INVALID_REQUEST,
                "Invalid request",
            )
        method = payload.get("method")
        if not isinstance(method, str) or not method or len(method) > 100:
            return None, _error_response(
                self._safe_response_id(payload),
                _INVALID_REQUEST,
                "Invalid request",
            )

        has_id = "id" in payload
        request_id = payload.get("id")
        if has_id and (request_id is None or isinstance(request_id, bool) or not isinstance(request_id, (str, int))):
            return None, _error_response(
                None,
                _INVALID_REQUEST,
                "Invalid request",
            )
        params_value = payload.get("params", {})
        if not isinstance(params_value, dict):
            if has_id:
                return None, _error_response(
                    request_id,
                    _INVALID_PARAMS,
                    "Invalid params",
                )
            return None, None
        params = dict(params_value)

        if not has_id:
            self._handle_notification_locked(method, params)
            return None, None

        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            return None, _error_response(
                None,
                _INVALID_REQUEST,
                "Invalid request",
            )
        if self._legacy_initializing is not None:
            return None, self._admission_error(request_id)
        modern_marker = self._has_modern_marker(params)
        if method == "initialize":
            if modern_marker or self._era == "modern" or self._legacy_profile is not None:
                return None, self._admission_error(request_id)
            return self._prepare_initialize(request_id, method, params)

        if modern_marker or self._era == "modern" or method == "server/discover":
            if self._era == "legacy":
                return None, self._admission_error(request_id)
            return self._prepare_modern(request_id, method, params)

        if self._legacy_profile is not None:
            return (
                _PreparedRequest(
                    request_id=request_id,
                    method=method,
                    params=params,
                    profile=self._legacy_profile,
                    era="legacy",
                    client_info=(
                        self._clone_object(self._legacy_client_info) if self._legacy_client_info is not None else None
                    ),
                    client_capabilities=self._clone_object(self._legacy_client_capabilities),
                    request_metadata={},
                ),
                None,
            )
        if method == "ping":
            return (
                _PreparedRequest(
                    request_id=request_id,
                    method=method,
                    params=params,
                    profile=PROTOCOL_PROFILES[PREFERRED_LEGACY_PROTOCOL_VERSION],
                    era="legacy",
                    client_info=None,
                    client_capabilities={},
                    request_metadata={},
                ),
                None,
            )
        return None, self._admission_error(request_id)

    def _prepare_modern(
        self,
        request_id: str | int,
        method: str,
        params: dict[str, Any],
    ) -> tuple[_PreparedRequest | None, _Response | None]:
        meta = params.get("_meta")
        if not isinstance(meta, dict):
            return None, _error_response(
                request_id,
                _INVALID_PARAMS,
                "Invalid params",
            )
        version = meta.get(_PROTOCOL_VERSION_KEY)
        capabilities = meta.get(_CLIENT_CAPABILITIES_KEY)
        client_info = meta.get(_CLIENT_INFO_KEY)
        if (
            not isinstance(version, str)
            or not isinstance(capabilities, dict)
            or (client_info is not None and not isinstance(client_info, dict))
        ):
            return None, _error_response(
                request_id,
                _INVALID_PARAMS,
                "Invalid params",
            )
        self._era = "modern"
        if version not in SUPPORTED_MODERN_PROTOCOL_VERSIONS:
            data: dict[str, GatewayJSONValue] = {
                "supported": list(SUPPORTED_PROTOCOL_VERSIONS),
            }
            if _is_safe_protocol_version(version):
                data["requested"] = version
            return None, _error_response(
                request_id,
                _UNSUPPORTED_PROTOCOL_VERSION,
                "Unsupported protocol version",
                data=data,
            )

        try:
            request_metadata = self._vendor_metadata(meta)
            cloned_capabilities = self._clone_object(capabilities)
            cloned_client_info = self._clone_object(client_info) if client_info is not None else None
            self._validate_client_info(cloned_client_info)
        except ValueError:
            return None, _error_response(
                request_id,
                _INVALID_PARAMS,
                "Invalid params",
            )
        params = dict(params)
        params.pop("_meta", None)
        return (
            _PreparedRequest(
                request_id=request_id,
                method=method,
                params=params,
                profile=PROTOCOL_PROFILES[CURRENT_PROTOCOL_VERSION],
                era="modern",
                client_info=cloned_client_info,
                client_capabilities=cloned_capabilities,
                request_metadata=request_metadata,
            ),
            None,
        )

    def _prepare_initialize(
        self,
        request_id: str | int,
        method: str,
        params: dict[str, Any],
    ) -> tuple[_PreparedRequest | None, _Response | None]:
        requested = params.get("protocolVersion")
        capabilities = params.get("capabilities")
        client_info = params.get("clientInfo")
        if not isinstance(requested, str) or not isinstance(capabilities, dict) or not isinstance(client_info, dict):
            return None, _error_response(
                request_id,
                _INVALID_PARAMS,
                "Invalid params",
            )
        negotiated = requested if requested in SUPPORTED_LEGACY_PROTOCOL_VERSIONS else PREFERRED_LEGACY_PROTOCOL_VERSION
        profile = PROTOCOL_PROFILES[negotiated]
        try:
            cloned_capabilities = self._clone_object(capabilities)
            cloned_client_info = self._clone_object(client_info)
            self._validate_client_info(cloned_client_info)
        except ValueError:
            return None, _error_response(
                request_id,
                _INVALID_PARAMS,
                "Invalid params",
            )
        return (
            _PreparedRequest(
                request_id=request_id,
                method=method,
                params=params,
                profile=profile,
                era="legacy",
                client_info=cloned_client_info,
                client_capabilities=cloned_capabilities,
                request_metadata={},
                initialize_result={
                    "protocolVersion": negotiated,
                    "capabilities": self._capabilities(),
                    "serverInfo": self._server_info(),
                },
            ),
            None,
        )

    def _admit_locked(
        self,
        prepared: _PreparedRequest,
    ) -> tuple[_ActiveRequest | None, _Response | None]:
        request_id = prepared.request_id
        key = _request_key(request_id)
        if key in self._active or len(self._active) >= self._limits.max_in_flight:
            return None, self._admission_error(request_id)
        if not self._consume_rate_token():
            return None, self._admission_error(request_id)

        active = _ActiveRequest(
            key=key,
            token=GatewayCancellationToken(),
            prepared=prepared,
        )
        task = asyncio.create_task(self._dispatch_active(active))
        active.task = task
        self._active[key] = active
        if prepared.initialize_result is not None:
            self._legacy_initializing = key
        self._track_task(task)
        return active, None

    def _start_standalone_locked(self, active: _ActiveRequest) -> None:
        if active.task is None:
            raise RuntimeError("request task is missing")
        dispatch_task = active.task

        async def emit() -> None:
            try:
                response = await dispatch_task
                if response is not None:
                    if active.prepared.initialize_result is None:
                        await self._write_value(response, token=active.token)
                    else:
                        await self._write_initialize_response(active, response)
            finally:
                await self._release_active(active)

        output_task = asyncio.create_task(emit())
        self._track_task(output_task)

    async def _run_batch(self, entries: list[_BatchEntry]) -> None:
        actives = [entry.active for entry in entries if entry.active is not None]
        try:
            tasks = [active.task for active in actives if active.task is not None]
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            results_by_key: dict[_RequestKey, _Response | None] = {}
            for active, result in zip(actives, task_results):
                results_by_key[active.key] = result if isinstance(result, dict) else None

            async with self._writer_lock:
                responses: list[_Response] = []
                for entry in entries:
                    if entry.response is not None:
                        responses.append(entry.response)
                    elif entry.active is not None and not entry.active.token.cancelled:
                        response = results_by_key.get(entry.active.key)
                        if response is not None:
                            responses.append(response)
                if responses:
                    bounded = self._output_limiter.bound(responses)
                    if bounded is not None:
                        await self._write_locked(bounded)
        finally:
            for active in actives:
                self._active.pop(active.key, None)

    async def _dispatch_active(self, active: _ActiveRequest) -> _Response | None:
        prepared = active.prepared
        try:
            active.token.raise_if_cancelled()
            result = await self._dispatch(prepared, active.token)
            active.token.raise_if_cancelled()
            return _success_response(prepared.request_id, result)
        except asyncio.CancelledError:
            return None
        except _ProtocolFailure as exc:
            return _error_response(
                prepared.request_id,
                exc.code,
                exc.message,
                data=exc.data,
            )
        except GatewayApplicationError as exc:
            projected = project_application_error(
                exc,
                prepared.profile,
                limits=self._limits,
            )
            return _error_response(
                prepared.request_id,
                int(projected["code"]),
                str(projected["message"]),
                data=projected.get("data"),
            )
        except Exception:  # noqa: BLE001 - runtime internals must become a fixed error
            return _error_response(
                prepared.request_id,
                _INTERNAL_ERROR,
                "Internal error",
            )

    async def _dispatch(
        self,
        prepared: _PreparedRequest,
        token: GatewayCancellationToken,
    ) -> GatewayJSONValue:
        method = prepared.method
        profile = prepared.profile
        context = self._request_context(prepared, token)

        if prepared.initialize_result is not None:
            return prepared.initialize_result
        if method == "server/discover":
            if profile.era != "modern":
                raise _ProtocolFailure(_METHOD_NOT_FOUND, "Method not found")
            result: dict[str, GatewayJSONValue] = {
                "supportedVersions": list(SUPPORTED_PROTOCOL_VERSIONS),
                "capabilities": self._capabilities(),
            }
            return self._complete_result(result, profile, cache=True)
        if method == "ping":
            return self._complete_result({}, profile)
        if method == "tools/list":
            tools = await self._project_tools(profile, context)
            return self._catalog_result("tools/list", "tools", tools, prepared)
        if method == "tools/call":
            return await self._call_tool(prepared, context)
        if method == "resources/list":
            descriptors = await self._runtime.list_resources(context)
            resources = await self._project_catalog(
                "resource",
                descriptors,
                profile,
            )
            return self._catalog_result(
                "resources/list",
                "resources",
                resources,
                prepared,
            )
        if method == "resources/templates/list":
            list_templates = getattr(self._runtime, "list_resource_templates", None)
            if not callable(list_templates):
                raise _ProtocolFailure(_METHOD_NOT_FOUND, "Method not found")
            descriptors = await list_templates(context)
            templates = await self._project_catalog(
                "resource_template",
                descriptors,
                profile,
            )
            return self._catalog_result(
                "resources/templates/list",
                "resourceTemplates",
                templates,
                prepared,
            )
        if method == "resources/read":
            return await self._read_resource(prepared, context)
        if method == "prompts/list":
            prompts = await self._project_prompts(profile, context)
            return self._catalog_result(
                "prompts/list",
                "prompts",
                prompts,
                prepared,
            )
        if method == "prompts/get":
            return await self._get_prompt(prepared, context)
        raise _ProtocolFailure(_METHOD_NOT_FOUND, "Method not found")

    async def _project_tools(
        self,
        profile: GatewayProtocolProfile,
        context: GatewayRequestContext,
    ) -> list[dict[str, GatewayJSONValue]]:
        pairs = await self._load_tools(profile, context)
        return [projected for _, projected in pairs]

    async def _load_tools(
        self,
        profile: GatewayProtocolProfile,
        context: GatewayRequestContext,
    ) -> list[tuple[Mapping[str, object], dict[str, GatewayJSONValue]]]:
        """Load, project, and compile one tool catalog without losing raw schemas."""

        descriptors = await self._runtime.list_tools(context)
        projected = await self._project_catalog("tool", descriptors, profile)
        self._validate_complete_catalog("tools/list", projected, profile)
        for descriptor in projected:
            try:
                await self._validator.validate_schema(
                    descriptor["inputSchema"],  # type: ignore[arg-type]
                    profile=profile,
                    root_mode="object",
                )
                if "outputSchema" in descriptor:
                    await self._validator.validate_schema(
                        descriptor["outputSchema"],  # type: ignore[arg-type]
                        profile=profile,
                        root_mode=("any" if profile.structured_content_mode == "any" else "object"),
                    )
            except GatewayApplicationError as exc:
                raise _ProtocolFailure(
                    _INTERNAL_ERROR,
                    "Internal error",
                ) from exc
        return [(raw, item) for raw, item in zip(descriptors, projected) if isinstance(raw, Mapping)]

    async def _project_prompts(
        self,
        profile: GatewayProtocolProfile,
        context: GatewayRequestContext,
    ) -> list[dict[str, GatewayJSONValue]]:
        descriptors = await self._runtime.list_prompts(context)
        projected = await self._project_catalog("prompt", descriptors, profile)
        self._validate_complete_catalog("prompts/list", projected, profile)
        return projected

    def _validate_complete_catalog(
        self,
        method: Literal["tools/list", "prompts/list"],
        items: list[dict[str, GatewayJSONValue]],
        profile: GatewayProtocolProfile,
    ) -> None:
        """Run Task 2 canonical identity and duplicate checks before lookup."""

        try:
            self._paginator.page(
                method=method,
                profile=profile,
                items=items,
                cursor=None,
            )
        except ValueError as exc:
            raise GatewayInvalidApplicationResult() from exc

    async def _project_catalog(
        self,
        kind: Literal["tool", "resource", "resource_template", "prompt"],
        descriptors: object,
        profile: GatewayProtocolProfile,
    ) -> list[dict[str, GatewayJSONValue]]:
        self._output_limiter.ensure_result_size(descriptors)
        if not isinstance(descriptors, list):
            raise GatewayInvalidApplicationResult()
        if len(descriptors) > self._limits.max_catalog_items:
            raise GatewayInvalidApplicationResult()
        projected: list[dict[str, GatewayJSONValue]] = []
        try:
            for descriptor in descriptors:
                if not isinstance(descriptor, Mapping):
                    raise GatewayInvalidApplicationResult()
                projected.append(
                    project_descriptor(
                        kind,
                        descriptor,
                        profile,
                        reserved_meta=self._server_meta,
                        limits=self._limits,
                    )
                )
        except (GatewayInvalidApplicationResult, ValueError) as exc:
            raise GatewayInvalidApplicationResult() from exc
        return projected

    def _catalog_result(
        self,
        method: str,
        field: str,
        items: list[dict[str, GatewayJSONValue]],
        prepared: _PreparedRequest,
    ) -> dict[str, GatewayJSONValue]:
        cursor = prepared.params.get("cursor")
        if cursor is not None and not isinstance(cursor, str):
            raise _ProtocolFailure(_INVALID_PARAMS, "Invalid params")
        try:
            page = self._paginator.page(
                method=method,
                profile=prepared.profile,
                items=items,
                cursor=cursor,
            )
        except ValueError as exc:
            if cursor is not None:
                raise _ProtocolFailure(_INVALID_PARAMS, "Invalid params") from exc
            raise GatewayInvalidApplicationResult() from exc
        result: dict[str, GatewayJSONValue] = {field: page.items}
        if page.next_cursor is not None:
            result["nextCursor"] = page.next_cursor
        return self._complete_result(result, prepared.profile, cache=True)

    async def _call_tool(
        self,
        prepared: _PreparedRequest,
        context: GatewayRequestContext,
    ) -> dict[str, GatewayJSONValue]:
        name = prepared.params.get("name")
        arguments = prepared.params.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise _ProtocolFailure(_INVALID_PARAMS, "Invalid params")
        tool_pairs = await self._load_tools(prepared.profile, context)
        selected = next(
            ((raw, projected) for raw, projected in tool_pairs if projected["name"] == name),
            None,
        )
        if selected is None:
            raise _ProtocolFailure(_INVALID_PARAMS, "Invalid params")
        raw_descriptor, tool = selected
        try:
            await self._validator.validate(
                tool["inputSchema"],  # type: ignore[arg-type]
                arguments,
                profile=prepared.profile,
                root_mode="object",
                instance_role="input",
            )
        except GatewayApplicationError as exc:
            raise _ProtocolFailure(_INVALID_PARAMS, "Invalid params") from exc

        try:
            raw_result = await self._runtime.call_tool(name, arguments, context)
        except GatewayToolExecutionError as exc:
            return project_tool_result(
                exc,
                prepared.profile,
                reserved_meta=self._server_meta,
                limits=self._limits,
            )
        if isinstance(raw_result, GatewayToolExecutionError):
            return project_tool_result(
                raw_result,
                prepared.profile,
                reserved_meta=self._server_meta,
                limits=self._limits,
            )
        self._output_limiter.ensure_result_size(raw_result)

        output_schema = raw_descriptor.get("outputSchema")
        if output_schema is not None:
            root_mode: Literal["any", "object"] = "any"
            if (
                prepared.profile.structured_content_mode == "object"
                and isinstance(output_schema, dict)
                and output_schema.get("type") == "object"
            ):
                root_mode = "object"
            try:
                await self._validator.validate(
                    output_schema,  # type: ignore[arg-type]
                    raw_result,
                    profile=prepared.profile,
                    root_mode=root_mode,
                    instance_role="output",
                )
            except GatewayApplicationError as exc:
                raise _ProtocolFailure(_INTERNAL_ERROR, "Internal error") from exc
        return project_tool_result(
            raw_result,
            prepared.profile,
            reserved_meta=self._server_meta,
            limits=self._limits,
        )

    async def _read_resource(
        self,
        prepared: _PreparedRequest,
        context: GatewayRequestContext,
    ) -> dict[str, GatewayJSONValue]:
        uri = prepared.params.get("uri")
        try:
            normalized_uri = _normalize_uri(uri)
        except GatewayInvalidApplicationResult as exc:
            raise _ProtocolFailure(_INVALID_PARAMS, "Invalid params") from exc
        result = await self._runtime.read_resource(normalized_uri, context)
        self._output_limiter.ensure_result_size(result)
        return project_resource_result(
            result,
            prepared.profile,
            reserved_meta=self._server_meta,
            limits=self._limits,
        )

    async def _get_prompt(
        self,
        prepared: _PreparedRequest,
        context: GatewayRequestContext,
    ) -> dict[str, GatewayJSONValue]:
        name = prepared.params.get("name")
        arguments = prepared.params.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise _ProtocolFailure(_INVALID_PARAMS, "Invalid params")
        prompts = await self._project_prompts(prepared.profile, context)
        if not any(prompt.get("name") == name for prompt in prompts):
            raise _ProtocolFailure(_INVALID_PARAMS, "Invalid params")
        result = await self._runtime.get_prompt(name, arguments, context)
        self._output_limiter.ensure_result_size(result)
        return project_prompt_result(
            result,
            prepared.profile,
            reserved_meta=self._server_meta,
            limits=self._limits,
        )

    def _request_context(
        self,
        prepared: _PreparedRequest,
        token: GatewayCancellationToken,
    ) -> GatewayRequestContext:
        metadata = self._clone_object(self._metadata)
        metadata.update(self._clone_object(prepared.request_metadata))
        metadata.update({"method": prepared.method, "transport": "stdio"})
        return GatewayRequestContext(
            request_id=prepared.request_id,
            metadata=metadata,
            protocol_version=prepared.profile.version,
            protocol_era=prepared.era,
            client_info=prepared.client_info,
            client_capabilities=prepared.client_capabilities,
            cancellation=token,
        )

    def _complete_result(
        self,
        result: dict[str, GatewayJSONValue],
        profile: GatewayProtocolProfile,
        *,
        cache: bool = False,
    ) -> dict[str, GatewayJSONValue]:
        if profile.era != "modern":
            return result
        result["resultType"] = "complete"
        if cache:
            result["ttlMs"] = 0
            result["cacheScope"] = "private"
        metadata = result.get("_meta")
        safe_meta = dict(metadata) if isinstance(metadata, dict) else {}
        safe_meta[_SERVER_INFO_KEY] = self._server_info()
        result["_meta"] = safe_meta
        return result

    def _capabilities(self) -> dict[str, GatewayJSONValue]:
        return {"tools": {}, "resources": {}, "prompts": {}}

    def _server_info(self) -> dict[str, GatewayJSONValue]:
        value = self._server_meta[_SERVER_INFO_KEY]
        if not isinstance(value, dict):
            raise RuntimeError("server metadata is invalid")
        return dict(value)

    def _handle_notification_locked(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        if method == "notifications/initialized":
            return
        if method != "notifications/cancelled":
            return
        request_id = params.get("requestId")
        if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
            return
        reason = params.get("reason")
        if reason is not None and not isinstance(reason, str):
            return
        active = self._active.get(_request_key(request_id))
        if active is None:
            return
        active.token.cancel(reason)
        if active.task is not None:
            active.task.cancel()

    async def _write_initialize_response(
        self,
        active: _ActiveRequest,
        response: _Response,
    ) -> None:
        """Write initialize success before atomically publishing negotiated state."""

        async with self._writer_lock:
            if active.token.cancelled:
                return
            bounded = self._output_limiter.bound(response)
            if bounded is not response:
                if bounded is not None:
                    await self._write_locked(bounded)
                return
            await self._write_locked(response)
            async with self._state_lock:
                if (
                    not self._closed
                    and self._legacy_initializing == active.key
                    and self._active.get(active.key) is active
                ):
                    self._commit_legacy_initialize_locked(active.prepared)

    def _commit_legacy_initialize_locked(self, prepared: _PreparedRequest) -> None:
        """Publish one successfully written initialize negotiation."""

        self._era = "legacy"
        self._legacy_profile = prepared.profile
        self._legacy_client_info = (
            self._clone_object(prepared.client_info) if prepared.client_info is not None else None
        )
        self._legacy_client_capabilities = self._clone_object(prepared.client_capabilities)
        self._legacy_initializing = None

    async def _release_active(self, active: _ActiveRequest) -> None:
        """Remove one active request and roll back any unfinished initialize."""

        async with self._state_lock:
            self._active.pop(active.key, None)
            if self._legacy_initializing == active.key:
                self._legacy_initializing = None

    async def _write_value(
        self,
        value: GatewayJSONValue,
        *,
        token: GatewayCancellationToken | None = None,
    ) -> None:
        async with self._writer_lock:
            if token is not None and token.cancelled:
                return
            bounded = self._output_limiter.bound(value)
            if bounded is not None:
                await self._write_locked(bounded)

    async def _write_locked(self, value: GatewayJSONValue) -> None:
        await self._writer(value)

    def _clone_metadata(
        self,
        metadata: Mapping[str, Any],
    ) -> dict[str, GatewayJSONValue]:
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        cloned = self._clone_object(dict(metadata))
        return {
            key: value
            for key, value in cloned.items()
            if key not in _RESERVED_CONTEXT_KEYS and not _is_reserved_meta_key(key)
        }

    def _vendor_metadata(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, GatewayJSONValue]:
        cloned = self._clone_object(metadata)
        return {
            key: value
            for key, value in cloned.items()
            if key not in _RESERVED_CONTEXT_KEYS and not _is_reserved_meta_key(key)
        }

    def _clone_object(self, value: object) -> dict[str, GatewayJSONValue]:
        try:
            _validate_json_structure(value, max_depth=self._limits.max_json_depth)
            cloned = json.loads(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (
            _JSONStructureError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError("value must be a finite JSON object") from exc
        if not isinstance(cloned, dict):
            raise ValueError("value must be a finite JSON object")
        return cloned

    def _consume_rate_token(self) -> bool:
        now = max(self._rate_updated_at, self._read_clock())
        elapsed = now - self._rate_updated_at
        refill_rate = self._limits.max_requests_per_minute / 60.0
        self._rate_tokens = min(
            float(self._limits.request_burst),
            self._rate_tokens + elapsed * refill_rate,
        )
        self._rate_updated_at = now
        if self._rate_tokens < 1.0:
            return False
        self._rate_tokens -= 1.0
        return True

    def _read_clock(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("clock must return a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("clock must return a finite number")
        return numeric

    @staticmethod
    def _safe_response_id(payload: Mapping[str, object]) -> str | int | None:
        request_id = payload.get("id")
        if isinstance(request_id, (str, int)) and not isinstance(request_id, bool):
            return request_id
        return None

    def _has_modern_marker(self, params: Mapping[str, object]) -> bool:
        meta = params.get("_meta")
        if not isinstance(meta, Mapping):
            return "_meta" in params
        return (
            _PROTOCOL_VERSION_KEY in meta
            or _CLIENT_CAPABILITIES_KEY in meta
            or _CLIENT_INFO_KEY in meta
            or (self._era != "legacy" and "_meta" in params)
        )

    @staticmethod
    def _validate_client_info(client_info: Mapping[str, object] | None) -> None:
        if client_info is None:
            return
        name = client_info.get("name")
        version = client_info.get("version")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 512
            or not isinstance(version, str)
            or not version
            or len(version) > 512
        ):
            raise ValueError("clientInfo must contain bounded name and version")

    @staticmethod
    def _admission_error(request_id: str | int | None) -> _Response:
        return _error_response(
            request_id,
            _ADMISSION_REJECTED,
            "Request rejected",
        )

    def _track_task(self, task: asyncio.Task[Any]) -> None:
        self._tasks.add(task)

        def finished(completed: asyncio.Task[Any]) -> None:
            self._tasks.discard(completed)
            if completed.cancelled():
                return
            try:
                error = completed.exception()
            except asyncio.CancelledError:
                return
            if error is not None:
                self._task_errors.append(error)

        task.add_done_callback(finished)


__all__ = ["GatewayProtocolConnection"]
