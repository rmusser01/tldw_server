"""Runtime execution stage for prepared MCP tool calls."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger
from mcp_unified.tool_use_reporting.builders import classify_tool_use_exception
from mcp_unified.tool_use_reporting.models import ToolUseStatus

from ..auth.rate_limiter import RateLimitExceeded
from ..execution_outcomes import (
    ExpectedToolFailure,
    ExpectedToolFailureReason,
    get_expected_tool_failure_reason,
)
from ..modules.base import AdmittedModuleOperation
from ..protocol_types import InvalidParamsException, PreparedToolCall, RequestContext
from ..tool_observability import (
    attach_execution_eval_metadata,
    execution_eval_metadata_from_tool_definition,
    sanitize_eval_profile_id,
)
from .dependencies import ToolExecutionDependencies


def _safe_exception_family(exc: BaseException) -> str:
    """Return a bounded inert exception family for observer-only logs."""

    try:
        name = type(exc).__name__
        if (
            type(name) is str
            and 1 <= len(name) <= 64
            and name.isascii()
            and (name[0].isalpha() or name[0] == "_")
            and all(character.isalnum() or character == "_" for character in name)
        ):
            return name
    except asyncio.CancelledError:
        raise
    except BaseException:  # noqa: BLE001 - hostile exceptions cannot replace outcomes.
        return "Exception"
    return "Exception"


class _NoopSpan:
    """Inert span used when the telemetry adapter is unavailable."""

    def set_attribute(self, _key: str, _value: Any) -> None:
        return None


class _BestEffortSpanContext:
    """Isolate a synchronous telemetry context from the tool outcome."""

    def __init__(self, context_manager: Any | None) -> None:
        self._context_manager = context_manager
        self._entered = False

    def __enter__(self) -> Any:
        if self._context_manager is None:
            return _NoopSpan()
        try:
            span = self._context_manager.__enter__()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - telemetry is best effort.
            return _NoopSpan()
        self._entered = True
        return span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        if not self._entered:
            return False
        try:
            if type(exc) is ExpectedToolFailure:
                self._context_manager.__exit__(None, None, None)
            else:
                self._context_manager.__exit__(exc_type, exc, traceback)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - telemetry cannot replace tool outcomes.
            return False
        return False


def _best_effort_span_context(
    telemetry: Any,
    operation_name: str,
    attributes: dict[str, Any],
) -> _BestEffortSpanContext:
    """Create a tool span without allowing ordinary adapter errors to escape."""

    try:
        context_manager = telemetry.trace_context(operation_name, attributes)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - telemetry is best effort.
        context_manager = None
    return _BestEffortSpanContext(context_manager)


def _set_span_attribute(span: Any, key: str, value: Any) -> None:
    """Set one telemetry attribute without changing the tool outcome."""

    try:
        span.set_attribute(key, value)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - telemetry is best effort.
        return None


class ToolExecutionRuntime:
    """Execute validated MCP tool calls and record runtime side effects."""

    def __init__(
        self,
        *,
        dependencies: ToolExecutionDependencies,
        security: Any,
        hooks: Any,
        noncritical_exceptions: tuple[type[BaseException], ...],
        tool_execution_error: str,
        generic_exception_like: Any,
        run_post_tool_hooks: Any | None = None,
    ) -> None:
        """Store dependencies and protocol compatibility callbacks for runtime execution."""

        self.dependencies = dependencies
        self.security = security
        self.hooks = hooks
        self.rate_limiter = dependencies.rate_limiter
        self.metrics = dependencies.metrics
        self.telemetry = dependencies.telemetry
        self.idempotency = dependencies.idempotency
        self.reporter = dependencies.reporter
        self.config_provider = dependencies.config_provider
        self._noncritical_exceptions = noncritical_exceptions
        self._tool_execution_error = tool_execution_error
        self._generic_exception_like = generic_exception_like
        self._run_post_tool_hooks = run_post_tool_hooks

    def sync_from_dependencies(self) -> None:
        """Refresh cached runtime references after protocol test seams mutate."""

        self.rate_limiter = self.dependencies.rate_limiter
        self.metrics = self.dependencies.metrics
        self.telemetry = self.dependencies.telemetry
        self.idempotency = self.dependencies.idempotency
        self.reporter = self.dependencies.reporter
        self.config_provider = self.dependencies.config_provider

    @staticmethod
    def extract_eval_profile_id(context: RequestContext) -> str | None:
        """Extract a non-sensitive profile identifier for execution eval metadata."""

        metadata = getattr(context, "metadata", {})
        if not isinstance(metadata, dict):
            return None
        for key in ("profile_id", "mcp_profile_id", "gateway_profile_id"):
            value = metadata.get(key)
            clean_value = sanitize_eval_profile_id(value)
            if clean_value is not None:
                return clean_value
        return None

    @staticmethod
    def _tool_use_eval_metadata(
        *,
        payload: dict[str, Any] | None = None,
        tool_def: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract safe eval metadata from response payload or tool definition."""

        if isinstance(payload, dict) and isinstance(payload.get("eval"), dict):
            return dict(payload["eval"])
        metadata = (tool_def or {}).get("metadata") if isinstance(tool_def, dict) else None
        if isinstance(metadata, dict) and isinstance(metadata.get("eval"), dict):
            return dict(metadata["eval"])
        return {}

    async def _run_post_hooks(self, **kwargs: Any) -> None:
        """Run post-tool hooks through either a protocol seam or the extracted hook helper."""

        if self._run_post_tool_hooks is not None:
            await self._run_post_tool_hooks(**kwargs)
            return
        await self.hooks.run_post_tool_hooks(**kwargs)

    async def execute_prepared_tool_call(self, prepared: PreparedToolCall) -> dict[str, Any]:
        """Execute a previously prepared tool invocation."""

        tool_name = prepared.tool_name
        tool_args = prepared.tool_args
        module = prepared.module
        module_id = prepared.module_id
        policy = prepared.policy
        is_write = policy.effect == "write"
        normalized_idempotency_key = prepared.normalized_idempotency_key
        idempotency_cache_key = prepared.idempotency_cache_key
        args_hash = prepared.arguments_hash
        context = prepared.context
        execution_start_ts = time.time()
        module_name = module_id or getattr(module, "name", None)
        module_invoked = False
        owner_duration_ms = 0.0

        def _observer_tool_def() -> dict[str, Any] | None:
            return prepared.tool_def

        def _observer_scope_payload() -> dict[str, Any] | None:
            return prepared.scope_payload

        def _expected_failure_payload(
            reason: ExpectedToolFailureReason,
        ) -> dict[str, Any]:
            duration_ms = max(0.0, (time.time() - execution_start_ts) * 1000.0)
            execution_eval = execution_eval_metadata_from_tool_definition(
                tool_name=tool_name,
                tool_def=_observer_tool_def(),
                profile_id=self.extract_eval_profile_id(context),
                duration_ms=duration_ms,
            )
            failure_content = {
                "status": "failed",
                "reason_code": reason.reason_code,
                "message": reason.public_message,
            }
            return {
                "content": [{"type": "json", "json": failure_content}],
                "isError": True,
                "module": module_id or getattr(module, "name", None),
                "tool": tool_name,
                "eval": execution_eval,
            }

        async def _record_prepared_event(
            *,
            status: ToolUseStatus,
            execution_origin: str,
            reason_code: str | None = None,
            payload: dict[str, Any] | None = None,
            idempotency_replay: bool = False,
        ) -> None:
            try:
                if not self.reporter.should_record(context):
                    return
                event = self.reporter.build_event(
                    context=context,
                    requested_tool_name=tool_name,
                    effective_tool_name=tool_name,
                    status=status,
                    execution_origin=execution_origin,
                    duration_ms=self.reporter.duration_ms(execution_start_ts),
                    module_id=module_id or getattr(module, "name", None),
                    tool_def=_observer_tool_def(),
                    payload=payload,
                    tool_args=tool_args,
                    scope_payload=_observer_scope_payload(),
                    is_write=is_write,
                    reason_code=reason_code,
                    idempotency_replay=idempotency_replay,
                )
                await self.reporter.record_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reporting must not affect tool calls.
                logger.warning(
                    "Failed to build or record prepared tool-use event: {}",
                    _safe_exception_family(exc),
                )

        async def _admit_rate_limit() -> None:
            category = policy.rate_limit_category
            key_owner = (
                f"user:{context.user_id}"
                if context.user_id
                else (f"client:{context.client_id}" if context.client_id else "anon")
            )
            rate_key = f"{key_owner}:tool:{tool_name}:cat:{category}"
            try:
                await self.rate_limiter.check_rate_limit(rate_key, category=category)
            except RateLimitExceeded:
                raise
            except asyncio.CancelledError:
                raise
            except self._noncritical_exceptions as exc:
                logger.warning(
                    "MCP tool rate admission unavailable: module={module_id} "
                    "tool={tool_name} error_type={error_type} fail_closed={fail_closed}",
                    module_id=module_id or "unknown",
                    tool_name=tool_name,
                    error_type=_safe_exception_family(exc),
                    fail_closed=policy.rate_limit_fail_closed,
                )
                if policy.rate_limit_fail_closed is True:
                    raise ExpectedToolFailure(
                        ExpectedToolFailureReason.RATE_LIMIT_UNAVAILABLE,
                    ) from None

        async def _run_failure_observers(
            error: Exception,
            *,
            record_module_metrics: bool,
            reason_code: str | None = None,
        ) -> None:
            observer_duration_ms = (
                owner_duration_ms if module_invoked else max(0.0, (time.time() - execution_start_ts) * 1000.0)
            )
            try:
                context.logger.error(  # noqa: TRY400 - the log contains only safe fields.
                    "Tool execution failed: {error_type}",
                    error_type=_safe_exception_family(error),
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - observers cannot replace failures.
                logger.debug(
                    "MCP tool failure log observer failed error_type={error_type}",
                    error_type=_safe_exception_family(exc),
                )
            if isinstance(error, InvalidParamsException):
                try:
                    self.metrics.record_tool_invalid_params(
                        getattr(module, "name", "unknown"),
                        str(tool_name),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - observers cannot replace failures.
                    logger.debug(
                        "MCP tool invalid-params metrics observer failed error_type={error_type}",
                        error_type=_safe_exception_family(exc),
                    )
            if record_module_metrics:
                try:
                    self.metrics.record_module_operation(
                        module=module_name or "unknown",
                        operation="tools_call",
                        duration=observer_duration_ms / 1000.0,
                        success=False,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - observers cannot replace failures.
                    logger.debug(
                        "MCP tool failure metrics observer failed error_type={error_type}",
                        error_type=_safe_exception_family(exc),
                    )
            try:
                self.reporter.audit_tool_event(
                    context,
                    tool_name,
                    module_name,
                    status="failure",
                    duration_ms=observer_duration_ms,
                    arguments_hash=args_hash,
                    error=error,
                    reason_code=reason_code,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - observers cannot replace failures.
                logger.debug(
                    "MCP tool failure audit observer failed error_type={error_type}",
                    error_type=_safe_exception_family(exc),
                )
            try:
                await self._run_post_hooks(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    module_id=module_name,
                    tool_def=_observer_tool_def(),
                    is_write=is_write,
                    arguments_hash=args_hash,
                    context=context,
                    scope_payload=_observer_scope_payload(),
                    status="failure",
                    duration_ms=observer_duration_ms,
                    error=error,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - observers cannot replace failures.
                logger.debug(
                    "MCP tool failure post-hook observer failed error_type={error_type}",
                    error_type=_safe_exception_family(exc),
                )

        async def _record_execution_failure(error: Exception) -> None:
            if module_invoked:
                await _run_failure_observers(
                    error,
                    record_module_metrics=True,
                )
            status, reason_code = classify_tool_use_exception(error)
            await _record_prepared_event(
                status=status,
                execution_origin=(
                    self.reporter.execution_origin_for_failure(status)
                    if status != "error"
                    else ("executed" if module_invoked else "failed_before_execution")
                ),
                reason_code=reason_code,
            )

        async def _complete_expected_failure(
            failure: ExpectedToolFailure,
        ) -> dict[str, Any]:
            reason = get_expected_tool_failure_reason(failure)
            if reason is None:
                sanitized_failure = self._generic_exception_like(
                    failure,
                    self._tool_execution_error,
                )
                await _record_execution_failure(sanitized_failure)
                raise sanitized_failure from None
            await _run_failure_observers(
                failure,
                record_module_metrics=module_invoked,
                reason_code=reason.reason_code,
            )
            await _record_prepared_event(
                status="error",
                execution_origin=("executed" if module_invoked else "failed_before_execution"),
                reason_code=reason.reason_code,
            )
            return _expected_failure_payload(reason)

        try:
            await self.security.verify_prepared_tool_call(
                prepared,
                require_live_binding=True,
            )
            await _admit_rate_limit()
        except RateLimitExceeded as exc:
            status, reason_code = classify_tool_use_exception(exc)
            await _record_prepared_event(
                status=status,
                execution_origin=self.reporter.execution_origin_for_failure(status),
                reason_code=reason_code,
            )
            raise
        except ExpectedToolFailure as failure:
            return await _complete_expected_failure(failure)

        async def _execute_owner() -> dict[str, Any]:
            nonlocal module_invoked, owner_duration_ms
            t0 = time.time()

            with _best_effort_span_context(
                self.telemetry,
                "mcp.tool_call",
                {
                    "mcp.tool": tool_name,
                    "mcp.module": getattr(module, "name", "unknown"),
                    "mcp.user_id": str(context.user_id or ""),
                    "mcp.client_id": str(context.client_id or ""),
                },
            ) as span:
                try:
                    async def _verify_module_admission() -> None:
                        await self.security.verify_prepared_tool_call(
                            prepared,
                            require_live_binding=True,
                        )

                    async def _invoke_admitted_module() -> dict[str, Any]:
                        nonlocal module_invoked
                        execution_args = tool_args
                        if (
                            policy.idempotency.inject_argument
                            and normalized_idempotency_key
                            and isinstance(tool_args, dict)
                        ):
                            execution_args = dict(tool_args)
                            execution_args["idempotencyKey"] = normalized_idempotency_key
                        module_invoked = True
                        return await module.execute_tool(
                            tool_name,
                            execution_args,
                            context,
                        )

                    result = await module.execute_with_circuit_breaker(
                        AdmittedModuleOperation(
                            _verify_module_admission,
                            _invoke_admitted_module,
                        ),
                    )
                    _set_span_attribute(span, "mcp.status", "success")
                except asyncio.CancelledError:
                    raise
                except ExpectedToolFailure as tool_error:
                    _set_span_attribute(span, "mcp.status", "failure")
                    _set_span_attribute(
                        span,
                        "mcp.error_type",
                        _safe_exception_family(tool_error),
                    )
                    raise
                except InvalidParamsException as tool_error:
                    _set_span_attribute(span, "mcp.status", "failure")
                    _set_span_attribute(
                        span,
                        "mcp.error_type",
                        _safe_exception_family(tool_error),
                    )
                    _set_span_attribute(span, "mcp.error_message", "invalid_params")
                    raise
                except (TypeError, ValueError) as tool_error:
                    _set_span_attribute(span, "mcp.status", "failure")
                    _set_span_attribute(
                        span,
                        "mcp.error_type",
                        _safe_exception_family(tool_error),
                    )
                    _set_span_attribute(span, "mcp.error_message", "invalid_params")
                    raise InvalidParamsException(str(tool_error)) from tool_error
                except PermissionError as tool_error:
                    _set_span_attribute(span, "mcp.status", "failure")
                    _set_span_attribute(
                        span,
                        "mcp.error_type",
                        _safe_exception_family(tool_error),
                    )
                    _set_span_attribute(span, "mcp.error_message", "permission_error")
                    raise
                except self._noncritical_exceptions as tool_error:
                    sanitized_tool_error = self._generic_exception_like(
                        tool_error,
                        self._tool_execution_error,
                    )
                    _set_span_attribute(span, "mcp.status", "failure")
                    _set_span_attribute(
                        span,
                        "mcp.error_type",
                        _safe_exception_family(sanitized_tool_error),
                    )
                    _set_span_attribute(
                        span,
                        "mcp.error_message",
                        self._tool_execution_error,
                    )
                    raise sanitized_tool_error from None
                finally:
                    owner_duration_ms = max(0.0, (time.time() - t0) * 1000.0)
                    _set_span_attribute(span, "mcp.duration_ms", owner_duration_ms)

            profile_id = self.extract_eval_profile_id(context)
            result = attach_execution_eval_metadata(
                result,
                tool_name=tool_name,
                tool_def=_observer_tool_def(),
                profile_id=profile_id,
                duration_ms=owner_duration_ms,
            )
            execution_eval = execution_eval_metadata_from_tool_definition(
                tool_name=tool_name,
                tool_def=_observer_tool_def(),
                profile_id=profile_id,
                duration_ms=owner_duration_ms,
            )
            result_eval = self._tool_use_eval_metadata(
                payload=result if isinstance(result, dict) else None,
            )
            for key in (
                "tool_prompt_id",
                "tool_prompt_version",
                "action_family",
                "result_kind",
                "path_filter_used",
                "truncated",
                "reason_code",
            ):
                if key in result_eval:
                    execution_eval[key] = result_eval[key]
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, list):
                content = result
            elif isinstance(result, dict):
                content = [{"type": "json", "json": result}]
            else:
                content = [{"type": "text", "text": str(result)}]
            return {
                "content": content,
                "module": module_name,
                "tool": tool_name,
                "eval": execution_eval,
            }

        async def _run_success_observers() -> None:
            try:
                self.metrics.record_module_operation(
                    module=module_name or "unknown",
                    operation="tools_call",
                    duration=owner_duration_ms / 1000.0,
                    success=True,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - observers cannot replace committed success.
                logger.debug(
                    "MCP tool success metrics observer failed error_type={error_type}",
                    error_type=_safe_exception_family(exc),
                )
            try:
                self.reporter.audit_tool_event(
                    context,
                    tool_name,
                    module_name,
                    status="success",
                    duration_ms=owner_duration_ms,
                    arguments_hash=args_hash,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - observers cannot replace committed success.
                logger.debug(
                    "MCP tool success audit observer failed error_type={error_type}",
                    error_type=_safe_exception_family(exc),
                )
            try:
                await self._run_post_hooks(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    module_id=module_name,
                    tool_def=_observer_tool_def(),
                    is_write=is_write,
                    arguments_hash=args_hash,
                    context=context,
                    scope_payload=_observer_scope_payload(),
                    status="success",
                    duration_ms=owner_duration_ms,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - observers cannot replace committed success.
                logger.debug(
                    "MCP tool success post-hook observer failed error_type={error_type}",
                    error_type=_safe_exception_family(exc),
                )

        if is_write and idempotency_cache_key:
            try:
                if args_hash is None:
                    raise InvalidParamsException("Unable to fingerprint tool arguments for idempotency")
                idempotency_result = await self.idempotency.execute(
                    idempotency_cache_key,
                    args_hash,
                    _execute_owner,
                    policy=policy.idempotency,
                )
                payload = idempotency_result.payload
                from_cache = idempotency_result.from_cache
                if from_cache:
                    await self.security.verify_prepared_tool_call(
                        prepared,
                        require_live_binding=True,
                    )
                try:
                    if from_cache:
                        self.metrics.record_idempotency_hit(
                            module_id or getattr(module, "name", "unknown"), str(tool_name)
                        )
                    else:
                        self.metrics.record_idempotency_miss(
                            module_id or getattr(module, "name", "unknown"), str(tool_name)
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as metrics_exc:  # noqa: BLE001 - metrics cannot replace outcomes.
                    logger.debug(
                        "MCP idempotency metrics skipped after noncritical failure: {error_type}",
                        error_type=_safe_exception_family(metrics_exc),
                    )
            except ExpectedToolFailure as failure:
                return await _complete_expected_failure(failure)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await _record_execution_failure(exc)
                raise
            if not from_cache:
                await _run_success_observers()
            await _record_prepared_event(
                status="success",
                execution_origin="cached" if from_cache else "executed",
                payload=payload if isinstance(payload, dict) else None,
                idempotency_replay=from_cache,
            )
            return payload

        try:
            payload = await _execute_owner()
        except ExpectedToolFailure as failure:
            return await _complete_expected_failure(failure)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _record_execution_failure(exc)
            raise
        await _run_success_observers()
        await _record_prepared_event(
            status="success",
            execution_origin="executed",
            payload=payload if isinstance(payload, dict) else None,
        )
        return payload
