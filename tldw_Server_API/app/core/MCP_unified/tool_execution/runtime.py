"""Runtime execution stage for prepared MCP tool calls."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from loguru import logger
from mcp_unified.tool_use_reporting.builders import classify_tool_use_exception
from mcp_unified.tool_use_reporting.models import ToolUseStatus

from ..auth.rate_limiter import RateLimitExceeded
from ..execution_outcomes import ExpectedToolFailure, ExpectedToolFailureReason
from ..protocol_types import InvalidParamsException, PreparedToolCall, RequestContext
from ..tool_observability import (
    attach_execution_eval_metadata,
    execution_eval_metadata_from_tool_definition,
    sanitize_eval_profile_id,
)
from .dependencies import ToolExecutionDependencies


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

        def _observer_tool_def() -> dict[str, Any] | None:
            return prepared.tool_def

        def _observer_scope_payload() -> dict[str, Any] | None:
            return prepared.scope_payload

        def _expected_failure_payload(
            failure: ExpectedToolFailure,
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
                "reason_code": failure.reason_code,
                "message": failure.public_message,
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
                    exc.__class__.__name__,
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
                    error_type=exc.__class__.__name__,
                    fail_closed=policy.rate_limit_fail_closed,
                )
                if policy.rate_limit_fail_closed is True:
                    raise ExpectedToolFailure(
                        ExpectedToolFailureReason.RATE_LIMIT_UNAVAILABLE,
                    ) from None

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
            return _expected_failure_payload(failure)

        async def _execute_owner() -> dict[str, Any]:
            # Execute tool with circuit breaker (pass context through)
            t0 = time.time()

            try:
                # Trace the tool call with OTEL
                with self.telemetry.trace_context(
                    "mcp.tool_call",
                    {
                        "mcp.tool": tool_name,
                        "mcp.module": getattr(module, "name", "unknown"),
                        "mcp.user_id": str(context.user_id or ""),
                        "mcp.client_id": str(context.client_id or ""),
                    },
                ) as span:
                    try:
                        execution_args = tool_args
                        if (
                            policy.idempotency.inject_argument
                            and normalized_idempotency_key
                            and isinstance(tool_args, dict)
                        ):
                            execution_args = dict(tool_args)
                            execution_args["idempotencyKey"] = normalized_idempotency_key
                        await self.security.verify_prepared_tool_call(
                            prepared,
                            require_live_binding=True,
                        )
                        result = await module.execute_with_circuit_breaker(
                            module.execute_tool,
                            tool_name,
                            execution_args,
                            context,
                        )
                        span.set_attribute("mcp.status", "success")
                    except asyncio.CancelledError:
                        raise
                    except InvalidParamsException as _tool_e:
                        span.set_attribute("mcp.status", "failure")
                        span.set_attribute("mcp.error_type", _tool_e.__class__.__name__)
                        span.set_attribute("mcp.error_message", str(_tool_e)[:200])
                        with contextlib.suppress(self._noncritical_exceptions):
                            self.metrics.record_tool_invalid_params(getattr(module, "name", "unknown"), str(tool_name))
                        raise
                    except (TypeError, ValueError) as _tool_e:
                        # Module argument validators often raise ValueError/TypeError.
                        # Normalize those to INVALID_PARAMS so HTTP callers receive 400.
                        span.set_attribute("mcp.status", "failure")
                        span.set_attribute("mcp.error_type", _tool_e.__class__.__name__)
                        span.set_attribute("mcp.error_message", str(_tool_e)[:200])
                        with contextlib.suppress(self._noncritical_exceptions):
                            self.metrics.record_tool_invalid_params(getattr(module, "name", "unknown"), str(tool_name))
                        raise InvalidParamsException(str(_tool_e)) from _tool_e
                    except PermissionError as _tool_e:
                        span.set_attribute("mcp.status", "failure")
                        span.set_attribute("mcp.error_type", _tool_e.__class__.__name__)
                        span.set_attribute("mcp.error_message", str(_tool_e)[:200])
                        raise
                    except self._noncritical_exceptions as _tool_e:
                        sanitized_tool_error = self._generic_exception_like(
                            _tool_e,
                            self._tool_execution_error,
                        )
                        span.set_attribute("mcp.status", "failure")
                        span.set_attribute("mcp.error_type", sanitized_tool_error.__class__.__name__)
                        span.set_attribute("mcp.error_message", self._tool_execution_error)
                        raise sanitized_tool_error from None
                    finally:
                        span.set_attribute("mcp.duration_ms", max(0.0, (time.time() - t0) * 1000.0))

                # Format result
                duration_ms = max(0.0, (time.time() - t0) * 1000.0)
                profile_id = self.extract_eval_profile_id(context)
                result = attach_execution_eval_metadata(
                    result,
                    tool_name=tool_name,
                    tool_def=_observer_tool_def(),
                    profile_id=profile_id,
                    duration_ms=duration_ms,
                )
                execution_eval = execution_eval_metadata_from_tool_definition(
                    tool_name=tool_name,
                    tool_def=_observer_tool_def(),
                    profile_id=profile_id,
                    duration_ms=duration_ms,
                )
                result_eval = self._tool_use_eval_metadata(payload=result if isinstance(result, dict) else None)
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
                    # Preserve structured tool results as JSON content instead of stringifying.
                    content = [{"type": "json", "json": result}]
                else:
                    content = [{"type": "text", "text": str(result)}]

                module_name = module_id or getattr(module, "name", None)
                # Record module operation metrics
                try:
                    duration = max(0.0, time.time() - t0)
                    self.metrics.record_module_operation(module=module_name or "unknown", operation="tools_call", duration=duration, success=True)
                except self._noncritical_exceptions as metrics_exc:
                    logger.debug(
                        "MCP tool success metrics skipped after noncritical failure: {error_type}",
                        error_type=metrics_exc.__class__.__name__,
                    )
                self.reporter.audit_tool_event(
                    context,
                    tool_name,
                    module_name,
                    status="success",
                    duration_ms=duration_ms,
                    arguments_hash=args_hash,
                )
                response_payload = {
                    "content": content,
                    "module": module_name,
                    "tool": tool_name,
                    "eval": execution_eval,
                }
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
                    duration_ms=duration_ms,
                )
                return response_payload

            except asyncio.CancelledError:
                raise
            except self._noncritical_exceptions as e:
                duration_ms = max(0.0, (time.time() - t0) * 1000.0)
                context.logger.error(  # noqa: TRY400 - structured audit log records sanitized type only.
                    "Tool execution failed: {error_type}",
                    error_type=e.__class__.__name__,
                )
                try:
                    duration = max(0.0, time.time() - t0)
                    self.metrics.record_module_operation(module=getattr(module, "name", "unknown"), operation="tools_call", duration=duration, success=False)
                except self._noncritical_exceptions as metrics_exc:
                    logger.debug(
                        "MCP tool failure metrics skipped after noncritical failure: {error_type}",
                        error_type=metrics_exc.__class__.__name__,
                    )
                self.reporter.audit_tool_event(
                    context,
                    tool_name,
                    module_id or getattr(module, "name", None),
                    status="failure",
                    duration_ms=duration_ms,
                    arguments_hash=args_hash,
                    error=e,
                )
                await self._run_post_hooks(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    module_id=module_id or getattr(module, "name", None),
                    tool_def=_observer_tool_def(),
                    is_write=is_write,
                    arguments_hash=args_hash,
                    context=context,
                    scope_payload=_observer_scope_payload(),
                    status="failure",
                    duration_ms=duration_ms,
                    error=e,
                )
                raise

        if is_write and idempotency_cache_key:
            try:
                ttl = policy.idempotency.ttl_seconds
                max_size = policy.idempotency.max_entries
                if args_hash is None:
                    raise InvalidParamsException("Unable to fingerprint tool arguments for idempotency")
                arguments_bound = await self.idempotency.bind_arguments(
                    idempotency_cache_key,
                    args_hash,
                    ttl=ttl,
                    max_size=max_size,
                )
                if not arguments_bound:
                    raise InvalidParamsException("Idempotency key was already used with different arguments")
                payload, from_cache = await self.idempotency.run(
                    idempotency_cache_key,
                    _execute_owner,
                    ttl=ttl,
                    max_size=max_size,
                    lock_ttl=policy.idempotency.lock_ttl_seconds,
                )
                try:
                    if from_cache:
                        self.metrics.record_idempotency_hit(module_id or getattr(module, "name", "unknown"), str(tool_name))
                    else:
                        self.metrics.record_idempotency_miss(module_id or getattr(module, "name", "unknown"), str(tool_name))
                except self._noncritical_exceptions as metrics_exc:
                    logger.debug(
                        "MCP idempotency metrics skipped after noncritical failure: {error_type}",
                        error_type=metrics_exc.__class__.__name__,
                    )
            except ExpectedToolFailure as failure:
                return _expected_failure_payload(failure)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                status, reason_code = classify_tool_use_exception(exc)
                await _record_prepared_event(
                    status=status,
                    execution_origin=(
                        self.reporter.execution_origin_for_failure(status)
                        if status != "error"
                        else "executed"
                    ),
                    reason_code=reason_code,
                )
                raise
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
            return _expected_failure_payload(failure)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status, reason_code = classify_tool_use_exception(exc)
            await _record_prepared_event(
                status=status,
                execution_origin=(
                    self.reporter.execution_origin_for_failure(status)
                    if status != "error"
                    else "executed"
                ),
                reason_code=reason_code,
            )
            raise
        await _record_prepared_event(
            status="success",
            execution_origin="executed",
            payload=payload if isinstance(payload, dict) else None,
        )
        return payload
