"""
chat_service.py
Lightweight helpers to keep the chat endpoint readable and testable without changing behavior.

These functions encapsulate small, deterministic pieces of logic used by
the /api/v1/chat/completions endpoint so the endpoint can orchestrate at a
higher level. The goal is to reduce duplication and cognitive load while
keeping the wire behavior identical.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import json as _json
import os
import re
import threading
import time
import uuid as _uuid
from collections.abc import AsyncIterator, Awaitable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from fastapi import HTTPException, status
from fastapi.encoders import jsonable_encoder
from loguru import logger
from starlette.responses import StreamingResponse

from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventType,
    MandatoryAuditWriteError,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib_facade import replace_placeholders
from tldw_Server_API.app.core.Character_Chat.constants import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core.Character_Chat.modules.character_utils import (
    map_sender_to_role,
    sanitize_sender_name,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import (
    SYNC_ADAPTER_CALL_POOL,
    DaemonCapacityError,
    await_bounded_sync_call,
    await_owned_worker,
    start_bounded_stream_daemon,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
    ProviderCredentialTerminalError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.Chat.chat_exceptions import get_request_id

# Reuse existing helpers from chat_helpers and prompt templating
from tldw_Server_API.app.core.Chat.chat_helpers import (
    get_or_create_character_context,
    get_or_create_conversation,
)
from tldw_Server_API.app.core.Chat.chat_logging import (
    exception_summary,
    prompt_template_summary,
    text_summary,
)
from tldw_Server_API.app.core.Chat.chat_loop_engine import is_chat_loop_mode_enabled
from tldw_Server_API.app.core.Chat.completion_pipeline import ChatCompletionPipeline
from tldw_Server_API.app.core.Chat.generation_metadata import (
    GENERATION_STATUS_COMPLETE,
    GENERATION_STATUS_LENGTH,
    build_generation_metadata,
)
from tldw_Server_API.app.core.Chat.message_utils import should_persist_message_role
from tldw_Server_API.app.core.Chat.moderation_pipeline import (
    OutputModerationRuntime,
    apply_output_safety_to_choices,
)
from tldw_Server_API.app.core.Chat.persistence_service import (
    build_assistant_message_payload,
    save_assistant_message,
    save_tool_messages,
)
from tldw_Server_API.app.core.Chat.prompt_cost_envelope import (
    build_prompt_cost_envelope,
    estimate_segment_tokens,
)
from tldw_Server_API.app.core.Chat.prompt_cost_guardrails import (
    PromptCostGuardrailDecision,
    evaluate_prompt_cost_guardrails,
    load_prompt_cost_guardrail_config,
)
from tldw_Server_API.app.core.Chat.prompt_template_manager import (
    DEFAULT_RAW_PASSTHROUGH_TEMPLATE,
    apply_template_to_string,
    load_template,
)
from tldw_Server_API.app.core.Chat.request_queue import (
    QueueStreamChannel,
    QueueStreamTerminalError,
    RequestPriority,
    get_request_queue,
)
from tldw_Server_API.app.core.Chat.response_processor import (
    NonStreamChoice,
    collect_non_stream_choices,
    estimate_completion_tokens_from_choices,
    extract_text_from_content,
    inject_assistant_name_into_choices,
    primary_choice,
    validate_structured_choices,
)
from tldw_Server_API.app.core.Chat.run_first_presentation import (
    present_chat_tools,
    tool_names_from_definitions,
)
from tldw_Server_API.app.core.Chat.streaming_pipeline import (
    StreamingPipelineRequest,
    create_chat_streaming_response,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    CHAT_STREAM_INCLUDE_METADATA,
    await_stream_operation_bounded,
    cancel_stream_tasks_bounded,
    create_streaming_response_with_timeout,
    invoke_stream_close_bounded,
    is_trusted_local_stream_frame,
    normalize_provider_stream_error,
    provider_payload_has_structural_error,
    provider_result_contains_error,
    provider_stream_error_allows_replay,
    provider_stream_error_payload,
    sanitized_provider_stream_exception,
    trusted_local_stream_error_frame,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    HEARTBEAT_INTERVAL as CHAT_HEARTBEAT_INTERVAL,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    STREAMING_IDLE_TIMEOUT as CHAT_IDLE_TIMEOUT,
)
from tldw_Server_API.app.core.Chat.tool_auto_exec import execute_assistant_tool_calls
from tldw_Server_API.app.core.Chat.tool_execution_service import (
    UnsupportedMultiChoiceToolAutoexecError,
    ensure_tool_autoexec_supports_request,
)
from tldw_Server_API.app.core.config import (
    load_comprehensive_config,
    resolve_chat_run_first_presentation_variant,
    resolve_chat_run_first_provider_allowlist,
    resolve_chat_run_first_rollout_mode,
    resolve_run_first_cohort_label,
)
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_config_option_names,
    custom_openai_provider_name,
    iter_custom_openai_provider_numbers,
)
from tldw_Server_API.app.core.exceptions import (
    ChatModelNotAvailableError,
    LLMPolicyRefusal,
    LLMQuotaExceededError,
    raise_detached_error,
)
from tldw_Server_API.app.core.LLM_Calls.gateway import (  # noqa: F401 - re-exported; callers move to the gateway (TASK-13950.27)
    infer_provider_from_model_catalog,
    invalidate_model_alias_caches,
    normalize_request_provider_and_model,
    perform_chat_api_call,
    perform_chat_api_call_async,
    resolve_provider_and_model,
    resolve_provider_api_key,
)

# isort: split
from tldw_Server_API.app.core.LLM_Calls.gateway import (  # used by build_call_params_from_request and the known-models helpers
    CHAT_COMPLETIONS_SURFACE,
    _provider_inventory_endpoint,
    _raise_inventory_failure,
)
from tldw_Server_API.app.core.LLM_Calls.llamacpp_request_extensions import (
    resolve_llamacpp_request_extensions,
    resolve_llamacpp_runtime_caps,
)
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import resolve_provider_model_value
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.LLM_Calls.provider_model_inventory import (
    CLOUD_MODEL_PROVIDERS,
    discover_provider_models,
    has_model_listing,
    resolve_provider_model_selection,
)
from tldw_Server_API.app.core.LLM_Calls.provider_readiness import normalize_catalog_provider_for_chat
from tldw_Server_API.app.core.LLM_Calls.sse import is_done_line, sse_data, sse_done
from tldw_Server_API.app.core.LLM_Calls.streaming import wrap_sync_stream
from tldw_Server_API.app.core.LLM_Calls.structured_generation import (
    StructuredGenerationCapabilityError,
    StructuredGenerationError,
    StructuredGenerationParseError,
    StructuredGenerationSchemaError,
    StructuredModeDecision,
    negotiate_structured_response_mode,
    parse_and_validate_structured_output,
)
from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _usage_tokens
from tldw_Server_API.app.core.Moderation.moderation_service import get_moderation_service
from tldw_Server_API.app.core.Moderation.review_service import (
    capture_moderation_review_item,
    is_moderation_review_capture_enabled,
)
from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import get_topic_monitoring_service
from tldw_Server_API.app.core.Persona.conversation_admission import require_current_persona
from tldw_Server_API.app.core.testing import (
    is_truthy as _shared_is_truthy,
)
from tldw_Server_API.app.core.Usage.llm_call_context import call_credential_source
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage, usage_with_service_tier
from tldw_Server_API.app.core.Usage.pricing_catalog import (
    list_provider_models,
)
from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage
from tldw_Server_API.app.core.Utils.cpu_bound_handler import process_large_json_async

_CHAT_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ChatAPIError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    HTTPException,
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    _json.JSONDecodeError,
)

_CHAT_PROVIDER_REFRESH_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ByokResolutionError,
    *_CHAT_NONCRITICAL_EXCEPTIONS,
)

_CHAT_RUN_FIRST_METRIC_EXCEPTIONS: tuple[type[Exception], ...] = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)

PROVIDER_STREAM_FACTORY_TIMEOUT_SECONDS = 60.0
PROVIDER_STREAM_FACTORY_HANDOFF_CLEANUP_SECONDS = 0.1

_LOCAL_CHAT_SIGNAL_PROVENANCE = object()
_LOCAL_CHAT_SIGNAL_KINDS = frozenset(
    {
        "prompt_cost_guardrail",
        "queue_admission",
        "structured_error",
    }
)


def _mark_trusted_local_chat_signal(value: Any, kind: str) -> Any:
    """Attach private identity provenance to a service-created control signal."""

    if kind not in _LOCAL_CHAT_SIGNAL_KINDS:
        raise ValueError("Unsupported local chat signal kind")
    value._chat_local_signal = (_LOCAL_CHAT_SIGNAL_PROVENANCE, kind)
    return value


def trusted_local_chat_signal_kind(value: Any) -> str | None:
    """Return the allowlisted kind for an exactly service-created control signal."""

    signal = getattr(value, "_chat_local_signal", None)
    if not isinstance(signal, tuple) or len(signal) != 2:
        return None
    provenance, kind = signal
    if provenance is not _LOCAL_CHAT_SIGNAL_PROVENANCE:
        return None
    return kind if kind in _LOCAL_CHAT_SIGNAL_KINDS else None


def _close_stream_factory_result_inline(stream: Any) -> None:
    """Best-effort close a provider stream from an already-isolated daemon."""

    close = getattr(stream, "aclose", None)
    if not callable(close):
        close = getattr(stream, "close", None)
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            asyncio.run(result)
    except BaseException as close_error:
        if isinstance(close_error, (KeyboardInterrupt, SystemExit)):
            raise
        logger.debug(
            "Late provider stream close failed error_type={}",
            type(close_error).__name__,
        )


def _close_late_stream_factory_result(stream: Any) -> None:
    """Best-effort close a late factory result on a bounded daemon thread."""

    try:
        start_bounded_stream_daemon(
            lambda: _close_stream_factory_result_inline(stream),
            name="late-provider-stream-close",
        )
    except DaemonCapacityError:
        logger.warning("Late provider stream close skipped: daemon capacity exhausted")


def _observe_factory_cleanup_task(task: asyncio.Task[Any]) -> None:
    """Consume completion from a detached factory-result cleanup task."""

    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.result()


async def _close_accepted_factory_result_after_release(
    stream: Any,
    worker_released: threading.Event,
) -> None:
    """Close one caller-owned result only after its factory lease is released."""

    cleanup_deadline = time.monotonic() + PROVIDER_STREAM_FACTORY_HANDOFF_CLEANUP_SECONDS
    while not worker_released.is_set():
        if time.monotonic() >= cleanup_deadline:
            logger.warning("Accepted provider stream cleanup missed worker handoff deadline")
            return
        await asyncio.sleep(0)
    close = getattr(stream, "aclose", None)
    if not callable(close):
        close = getattr(stream, "close", None)
    if not callable(close):
        return
    try:
        await invoke_stream_close_bounded(
            close,
            max(0.0, cleanup_deadline - time.monotonic()),
        )
    except asyncio.TimeoutError:
        logger.warning("Accepted provider stream cleanup exceeded bounded timeout")
    except _CHAT_NONCRITICAL_EXCEPTIONS as close_error:
        logger.debug(
            "Accepted provider stream cleanup failed error_type={}",
            type(close_error).__name__,
        )


async def _wait_factory_worker_release_bounded(
    worker_released: threading.Event,
    deadline: float,
) -> bool:
    """Wait for post-delivery daemon handoff within the request deadline."""

    while not worker_released.is_set():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0)
    return True


async def _call_stream_factory_bounded(
    factory: Callable[[], Any],
    *,
    timeout: float | None = None,
) -> Any:
    """Invoke a sync provider factory with a hard bound and daemon isolation."""

    wait_timeout = (
        PROVIDER_STREAM_FACTORY_TIMEOUT_SECONDS if timeout is None else float(timeout)
    )
    if wait_timeout <= 0:
        raise asyncio.TimeoutError

    deadline = time.monotonic() + wait_timeout
    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[Any] = loop.create_future()
    decision_lock = threading.Lock()
    decision_made = threading.Event()
    decision: str | None = None
    delivery_observed = threading.Event()
    worker_released = threading.Event()

    def choose_decision(value: str) -> bool:
        nonlocal decision
        with decision_lock:
            if decision is not None:
                return decision == value
            decision = value
            decision_made.set()
            return True

    def deliver(value: Any = None, error: BaseException | None = None) -> None:
        try:
            if result_future.done():
                choose_decision("abandoned")
                return
            if error is not None:
                if isinstance(error, asyncio.CancelledError):
                    error = sanitized_provider_stream_exception(error)
                result_future.set_exception(error)
            else:
                result_future.set_result(value)
        finally:
            delivery_observed.set()

    def wait_for_delivery(value: Any = None) -> None:
        while not delivery_observed.wait(0.05):
            if loop.is_closed():
                choose_decision("abandoned")
                if value is not None:
                    _close_stream_factory_result_inline(value)
                return
        while not decision_made.wait(0.05):
            if loop.is_closed():
                choose_decision("abandoned")
                break
        if decision == "abandoned" and value is not None:
            _close_stream_factory_result_inline(value)

    def worker() -> None:
        try:
            value = factory()
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            try:
                loop.call_soon_threadsafe(deliver, None, error)
            except RuntimeError:
                return
            wait_for_delivery()
        else:
            try:
                loop.call_soon_threadsafe(deliver, value, None)
            except RuntimeError:
                _close_stream_factory_result_inline(value)
                return
            wait_for_delivery(value)

    start_bounded_stream_daemon(
        worker,
        name="provider-stream-factory",
        released_event=worker_released,
    )
    result: Any = None
    accepted = False
    try:
        result = await await_stream_operation_bounded(
            result_future,
            wait_timeout,
        )
        if time.monotonic() >= deadline:
            choose_decision("abandoned")
            raise asyncio.TimeoutError
        accepted = choose_decision("accepted")
        if not accepted:
            raise asyncio.TimeoutError
        while not worker_released.is_set():
            if time.monotonic() >= deadline:
                raise asyncio.TimeoutError
            await asyncio.sleep(0)
        return result
    except asyncio.CancelledError:
        if not accepted:
            choose_decision("abandoned")
        elif result is not None:
            cleanup_task = asyncio.create_task(
                _close_accepted_factory_result_after_release(result, worker_released)
            )
            cleanup_task.add_done_callback(_observe_factory_cleanup_task)
        raise
    except asyncio.TimeoutError:
        if not accepted:
            choose_decision("abandoned")
            if result_future.done():
                await _wait_factory_worker_release_bounded(worker_released, deadline)
        elif result is not None:
            cleanup_task = asyncio.create_task(
                _close_accepted_factory_result_after_release(result, worker_released)
            )
            cleanup_task.add_done_callback(_observe_factory_cleanup_task)
        raise
    except BaseException:
        if not accepted:
            choose_decision("abandoned")
            await _wait_factory_worker_release_bounded(worker_released, deadline)
        raise


def _provider_factory_monotonic() -> float:
    """Return the monotonic clock used by the aggregate provider preflight budget."""

    return time.monotonic()


async def _refresh_provider_params_bounded(
    refresh_provider_params: Callable[[str], Any],
    provider: str,
    remaining_timeout: Callable[[], float],
) -> Any:
    """Invoke and await provider refresh within the existing preflight deadline."""

    refreshed = await _call_stream_factory_bounded(
        partial(refresh_provider_params, provider),
        timeout=remaining_timeout(),
    )
    if not inspect.isawaitable(refreshed):
        return refreshed
    remaining = remaining_timeout()
    if remaining <= 0:
        _close_late_stream_factory_result(refreshed)
        raise asyncio.TimeoutError
    return await await_stream_operation_bounded(refreshed, remaining)


def _refresh_provider_params_bounded_sync(
    refresh_provider_params: Callable[[str], Any],
    provider: str,
    remaining_timeout: Callable[[], float],
    request_loop: asyncio.AbstractEventLoop,
) -> Any:
    """Run queued refresh awaitables on their owning request event loop."""

    if remaining_timeout() <= 0:
        raise TimeoutError("Provider parameter refresh timed out")
    refreshed = refresh_provider_params(provider)
    if not inspect.isawaitable(refreshed):
        return refreshed
    remaining = remaining_timeout()
    if remaining <= 0:
        _close_stream_factory_result_inline(refreshed)
        raise TimeoutError("Provider parameter refresh timed out")

    async def await_on_request_loop() -> Any:
        return await await_stream_operation_bounded(refreshed, remaining)

    try:
        refresh_future = asyncio.run_coroutine_threadsafe(
            await_on_request_loop(),
            request_loop,
        )
    except BaseException:
        _close_stream_factory_result_inline(refreshed)
        raise
    try:
        return refresh_future.result(timeout=remaining)
    except TimeoutError:
        refresh_future.cancel()
        raise TimeoutError("Provider parameter refresh timed out") from None


def _call_stream_factory_from_isolated_worker(
    factory: Callable[[], Any],
    remaining_timeout: Callable[[], float],
) -> Any:
    """Call a provider factory already owned by RequestQueue's daemon worker."""

    if remaining_timeout() <= 0:
        raise TimeoutError("Provider stream factory timed out")
    result = factory()
    if remaining_timeout() <= 0:
        _close_stream_factory_result_inline(result)
        raise TimeoutError("Provider stream factory timed out")
    return result


_config = load_comprehensive_config()
_chat_config: dict[str, str] = {}
if _config and _config.has_section("Chat-Module"):
    _chat_config = dict(_config.items("Chat-Module"))


def _coerce_int(value: str | None, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        return default


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return _shared_is_truthy(value)


def _coerce_int_bounded(
    value: str | None,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    parsed = _coerce_int(value, default)
    return max(minimum, min(maximum, parsed))


_TOOL_ALLOW_CATALOG_TOKEN_RE = re.compile(r"^[A-Za-z0-9._:-]+(?:\*)?$")
_TOOL_ALLOW_CATALOG_DEFAULT = ""
_CHAT_TOOL_CALLS_MIN = 1
_CHAT_TOOL_CALLS_MAX = 20
_CHAT_TOOL_TIMEOUT_MS_MIN = 1000
_CHAT_TOOL_TIMEOUT_MS_MAX = 120_000

_PROVIDER_MODEL_CONFIG_FIELDS: dict[str, tuple[str, str]] = {
    "openai": ("API", "openai_model"),
    "anthropic": ("API", "anthropic_model"),
    "cohere": ("API", "cohere_model"),
    "deepseek": ("API", "deepseek_model"),
    "qwen": ("API", "qwen_model"),
    "google": ("API", "google_model"),
    "groq": ("API", "groq_model"),
    "huggingface": ("API", "huggingface_model"),
    "mistral": ("API", "mistral_model"),
    "bedrock": ("API", "bedrock_model"),
    "openrouter": ("API", "openrouter_model"),
    "moonshot": ("API", "moonshot_model"),
    "zai": ("API", "zai_model"),
    "custom-openai-api": ("API", "custom_openai_api_model"),
    "custom-openai-api-2": ("API", "custom_openai2_api_model"),
    "llama.cpp": ("Local-API", "llama_model"),
    "kobold": ("Local-API", "kobold_model"),
    "ooba": ("Local-API", "ooba_model"),
    "tabbyapi": ("Local-API", "tabby_model"),
    "vllm": ("Local-API", "vllm_model"),
    "ollama": ("Local-API", "ollama_model"),
    "aphrodite": ("Local-API", "aphrodite_model"),
}
_PROVIDER_MODEL_CONFIG_FIELDS.update(
    {
        custom_openai_provider_name(provider_number): (
            "API",
            custom_openai_config_option_names(provider_number, "model")[0],
        )
        for provider_number in iter_custom_openai_provider_numbers(start=3)
    }
)

_DEFAULT_ASSISTANT_SYSTEM_PROMPT = "You are a helpful AI assistant."


def _parse_tool_allow_catalog(value: str | None) -> list[str]:
    """Parse chat tool allow-catalog from comma-separated names/prefixes.

    - Empty string or blank means no tools allowed (empty list).
    - `*` means allow all (returns ``["*"]``).
    - Entries support exact names (``notes.search``) and suffix wildcard prefixes
      (``notes.*``).
    - Invalid entries are ignored.
    """
    raw = (value or "").strip()
    if not raw:
        return []

    if raw == "*":
        return ["*"]

    items: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        entry = token.strip()
        if not entry:
            continue
        if entry == "*":
            return ["*"]
        if "*" in entry[:-1]:
            continue
        if not _TOOL_ALLOW_CATALOG_TOKEN_RE.match(entry):
            continue
        if entry not in seen:
            seen.add(entry)
            items.append(entry)
    return items


def _split_model_list(raw_value: str | None) -> list[str]:
    """Split a model list string into distinct model IDs."""
    raw = str(raw_value or "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item and item.strip()]


def _normalize_conversation_assistant_context(
    conversation: dict[str, Any] | None,
    *,
    default_character_id: int | None = None,
) -> dict[str, Any]:
    """Normalize persisted conversation assistant identity into a small runtime dict."""
    if not conversation:
        return {
            "assistant_kind": None,
            "assistant_id": None,
            "persona_memory_mode": None,
            "is_default_character": False,
        }

    character_id = conversation.get("character_id")
    assistant_id = conversation.get("assistant_id")
    # Mirror the assistant_id fallback used by _resolve_assistant_context_for_chat
    # so moderation scope stays consistent with the runtime character resolution.
    if character_id is None and assistant_id is not None:
        character_id = assistant_id
    assistant_kind = conversation.get("assistant_kind") or ("character" if character_id is not None else None)
    if assistant_id is None and character_id is not None:
        assistant_id = str(character_id)
    is_default_character = _is_default_character_reference(
        character_id if character_id is not None else assistant_id,
        default_character_id=default_character_id,
    )
    if assistant_kind == "character" and is_default_character:
        assistant_kind = "regular"

    return {
        "assistant_kind": assistant_kind,
        "assistant_id": assistant_id,
        "persona_memory_mode": conversation.get("persona_memory_mode"),
        "is_default_character": is_default_character,
    }


def resolve_moderation_chat_type(
    *,
    request_data: Any | None,
    assistant_context: dict[str, Any] | None = None,
    default_character_id: int | None = None,
) -> str:
    """Resolve moderation chat scope, treating the default assistant as regular when that context is available."""
    if isinstance(assistant_context, dict):
        assistant_kind = str(assistant_context.get("assistant_kind") or "").strip().lower()
        if assistant_kind == "character" and not bool(assistant_context.get("is_default_character")):
            return "character"
        if assistant_kind in {"regular", "persona"} or bool(assistant_context.get("is_default_character")):
            return "regular"

    requested_character_id = getattr(request_data, "character_id", None) if request_data is not None else None
    if requested_character_id:
        return "regular" if _is_default_character_reference(
            requested_character_id,
            default_character_id=default_character_id,
        ) else "character"

    return "regular"


async def _resolve_default_character_id(
    chat_db: Any,
    loop: Any,
) -> int | None:
    """Return the persisted default assistant character ID when available."""
    get_character_card_by_name = getattr(chat_db, "get_character_card_by_name", None)
    if not callable(get_character_card_by_name):
        return None

    try:
        default_character = await asyncio.to_thread(get_character_card_by_name,
            DEFAULT_CHARACTER_NAME,
        )
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        return None
    if not isinstance(default_character, dict):
        return None

    try:
        return int(default_character.get("id"))
    except (TypeError, ValueError):
        return None


def _is_default_character_reference(
    character_reference: Any,
    *,
    default_character_id: int | None,
) -> bool:
    """Return whether the provided character reference points at the default assistant."""
    normalized_reference = str(character_reference or "").strip()
    if not normalized_reference:
        return False
    if normalized_reference == DEFAULT_CHARACTER_NAME:
        return True
    if default_character_id is None:
        return False
    try:
        return int(normalized_reference) == int(default_character_id)
    except (TypeError, ValueError):
        return False


async def resolve_input_moderation_chat_type(
    *,
    chat_db: Any,
    request_data: Any,
    loop: Any,
) -> str:
    """Resolve chat_type for input moderation from request fields and saved conversation state."""
    if (getattr(request_data, "tldw_history_selection_v1", None) is not None
            or getattr(request_data, "tldw_history_server_existing_input_v1", None) is not None):
        conversation = await loop.run_in_executor(None, chat_db.get_conversation_by_id, request_data.conversation_id)
        return "character" if conversation and (conversation.get("character_id") or conversation.get("assistant_kind")) else "regular"
    default_character_id = await _resolve_default_character_id(chat_db, loop)
    requested_character_id = getattr(request_data, "character_id", None)
    # Also check legacy persona_id alias so moderation scope is consistent
    # even before the endpoint normalizes persona_id -> character_id.
    if not requested_character_id:
        raw_persona_id = getattr(request_data, "persona_id", None)
        if raw_persona_id is not None:
            persona_id_str = str(raw_persona_id).strip()
            if persona_id_str:
                requested_character_id = persona_id_str
    if requested_character_id:
        return "regular" if _is_default_character_reference(
            requested_character_id,
            default_character_id=default_character_id,
        ) else "character"

    conversation_id = getattr(request_data, "conversation_id", None)
    existing_conversation: dict[str, Any] | None = None
    get_conversation_by_id = getattr(chat_db, "get_conversation_by_id", None)
    if conversation_id and callable(get_conversation_by_id):
        existing_conversation = await asyncio.to_thread(get_conversation_by_id, conversation_id)

    assistant_context = _normalize_conversation_assistant_context(
        existing_conversation,
        default_character_id=default_character_id,
    )
    return resolve_moderation_chat_type(
        request_data=request_data,
        assistant_context=assistant_context,
        default_character_id=default_character_id,
    )


def _build_persona_chat_projection(
    persona_profile: dict[str, Any],
    *,
    fallback_persona_id: str | None = None,
) -> dict[str, Any]:
    """Project a persona profile into the minimal assistant card ordinary chat needs."""
    persona_name = str(persona_profile.get("name") or fallback_persona_id or "Persona").strip() or "Persona"
    system_prompt = str(persona_profile.get("system_prompt") or "").strip() or _DEFAULT_ASSISTANT_SYSTEM_PROMPT
    return {
        "name": persona_name,
        "system_prompt": system_prompt,
        "avatar_url": None,
        "first_message": None,
        "extensions": None,
    }


async def _resolve_assistant_context_for_chat(
    *,
    chat_db: Any,
    request_data: Any,
    loop: Any,
    conversation_id: str | None,
) -> tuple[dict[str, Any] | None, int | None, dict[str, Any] | None, dict[str, Any]]:
    """Recheck current Persona admission before resolving ordinary chat context."""
    existing_conversation: dict[str, Any] | None = None
    get_conversation_by_id = getattr(chat_db, "get_conversation_by_id", None)
    if conversation_id and callable(get_conversation_by_id):
        existing_conversation = await asyncio.to_thread(get_conversation_by_id, conversation_id)

    if existing_conversation:
        persona_profile = await asyncio.to_thread(
            require_current_persona, chat_db,
            owner_id=chat_db.owner_user_id, conversation=existing_conversation,
        )
        if persona_profile is not None:
            assistant_context = _normalize_conversation_assistant_context(existing_conversation)
            return (
                _build_persona_chat_projection(
                    persona_profile, fallback_persona_id=assistant_context["assistant_id"],
                ),
                None,
                existing_conversation,
                assistant_context,
            )

    default_character_id = await _resolve_default_character_id(chat_db, loop)
    assistant_context = _normalize_conversation_assistant_context(
        existing_conversation,
        default_character_id=default_character_id,
    )
    assistant_id = assistant_context.get("assistant_id")

    character_lookup = getattr(request_data, "character_id", None)
    if character_lookup is None and existing_conversation:
        character_lookup = existing_conversation.get("character_id") or assistant_id

    character_card, character_db_id = await get_or_create_character_context(chat_db, character_lookup, loop)
    if character_db_id is not None:
        is_default_character = _is_default_character_reference(
            character_db_id,
            default_character_id=default_character_id,
        )
        assistant_context = {
            "assistant_kind": "regular" if is_default_character else "character",
            "assistant_id": str(character_db_id),
            "persona_memory_mode": assistant_context.get("persona_memory_mode"),
            "is_default_character": is_default_character,
        }
    return character_card, character_db_id, existing_conversation, assistant_context


def _configured_models_for_provider(provider: str) -> tuple[str, ...]:
    """Read current models using the same config/env precedence as discovery."""
    provider_key = (provider or "").strip().lower()
    mapping = _PROVIDER_MODEL_CONFIG_FIELDS.get(provider_key)
    if not mapping:
        return ()

    section, field = mapping
    try:
        raw_value = resolve_provider_model_value(
            provider_key, load_comprehensive_config(), section, field
        )
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        return ()

    return tuple(_split_model_list(raw_value))


def known_models_for_provider_cached(
    provider: str,
    *,
    api_key: str | None = None,
    app_config: dict[str, Any] | None = None,
    credentials_resolved: bool = False,
) -> tuple[str, ...]:
    """Read scoped commercial inventory; retain configured catalogs only locally."""
    provider_key = canonical_provider_name(normalize_catalog_provider_for_chat(provider))
    if not provider_key:
        return ()
    if has_model_listing(provider_key):
        # A context-free prevalidator cannot select the requesting tenant's key.
        if not api_key:
            return ()
        request = {
            "api_key": api_key,
            "app_config": (
                app_config if app_config is not None
                else ({} if credentials_resolved else None)
            ),
            "credentials_resolved": credentials_resolved,
        }
        base_url, discovery_options = _provider_inventory_endpoint(provider_key, request)
        result = discover_provider_models(provider_key, api_key, base_url=base_url, **discovery_options)
        if result.status != "ready":
            _raise_inventory_failure(provider_key, result.status)
        return tuple(result.models)

    known: set[str] = set()
    try:
        for model_name in list_provider_models(provider_key) or []:
            normalized = str(model_name).strip()
            if normalized:
                known.add(normalized)
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        pass

    for model_name in _configured_models_for_provider(provider_key):
        normalized = str(model_name).strip()
        if normalized:
            known.add(normalized)

    return tuple(sorted(known))


def is_model_known_for_provider(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    app_config: dict[str, Any] | None = None,
    credentials_resolved: bool = False,
    endpoint_provenance: str | None = None,
) -> bool | None:
    """Check authoritative commercial request IDs with credential context, else defer."""
    provider_key = canonical_provider_name(normalize_catalog_provider_for_chat(provider))
    model_key = (model or "").strip()
    if not provider_key or not model_key:
        return None
    if provider_key in CLOUD_MODEL_PROVIDERS and not has_model_listing(provider_key):
        return None  # no listing: the provider itself rejects a bad model
    if provider_key in CLOUD_MODEL_PROVIDERS:
        if api_key is None and not credentials_resolved:
            return None
        if not api_key:
            return False
        request = {
            "app_config": app_config if app_config is not None else ({} if credentials_resolved else None),
            "credentials_resolved": credentials_resolved,
            "_endpoint_provenance": endpoint_provenance,
        }
        base_url, discovery_options = _provider_inventory_endpoint(provider_key, request)
        inventory = discover_provider_models(provider_key, api_key, base_url=base_url, **discovery_options)
        selection = resolve_provider_model_selection(
            provider_key, api_key, model_key, base_url=base_url, inventory=inventory, **discovery_options,
        )
        if selection.status not in {"ready", "not_found"}:
            _raise_inventory_failure(provider_key, selection.status)
        return selection.status == "ready"
    known_models = known_models_for_provider_cached(provider_key)
    if not known_models:
        return None
    return model_key.lower() in {str(item).strip().lower() for item in known_models}


_MAX_HISTORY_MESSAGES = max(1, _coerce_int(_chat_config.get("max_history_messages"), 200))

_default_history_limit = 20
if "history_messages_limit" in _chat_config:
    _default_history_limit = max(
        1,
        min(_MAX_HISTORY_MESSAGES, _coerce_int(_chat_config.get("history_messages_limit"), _default_history_limit)),
    )
_env_history_limit = os.getenv("CHAT_HISTORY_LIMIT")
if _env_history_limit:
    with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
        _default_history_limit = max(1, min(_MAX_HISTORY_MESSAGES, int(_env_history_limit)))
DEFAULT_HISTORY_MESSAGE_LIMIT = _default_history_limit

_default_history_order = _chat_config.get("history_messages_order", "desc").strip().lower()
if _default_history_order not in {"asc", "desc"}:
    _default_history_order = "desc"
_env_history_order = os.getenv("CHAT_HISTORY_ORDER")
if _env_history_order:
    _env_history_order_val = _env_history_order.strip().lower()
    if _env_history_order_val in {"asc", "desc"}:
        _default_history_order = _env_history_order_val
DEFAULT_HISTORY_MESSAGE_ORDER = _default_history_order

_inject_assistant_name = _coerce_bool(_chat_config.get("inject_assistant_name"), False)
_env_inject_assistant_name = os.getenv("CHAT_INJECT_ASSISTANT_NAME")
if _env_inject_assistant_name is not None:
    _inject_assistant_name = _coerce_bool(_env_inject_assistant_name, _inject_assistant_name)
INJECT_ASSISTANT_NAME = _inject_assistant_name

_force_normalize_strings = _coerce_bool(_chat_config.get("force_normalize_string_responses"), False)
_env_force_normalize = os.getenv("CHAT_FORCE_NORMALIZE_STRING_RESPONSES")
if _env_force_normalize is not None:
    _force_normalize_strings = _coerce_bool(_env_force_normalize, _force_normalize_strings)
FORCE_NORMALIZE_STRING_RESPONSES = _force_normalize_strings

_chat_auto_execute_tools = _coerce_bool(_chat_config.get("chat_auto_execute_tools"), False)
_env_chat_auto_execute_tools = os.getenv("CHAT_AUTO_EXECUTE_TOOLS")
if _env_chat_auto_execute_tools is not None:
    _chat_auto_execute_tools = _coerce_bool(_env_chat_auto_execute_tools, _chat_auto_execute_tools)
CHAT_AUTO_EXECUTE_TOOLS = _chat_auto_execute_tools

_chat_max_tool_calls = _coerce_int_bounded(
    _chat_config.get("chat_max_tool_calls"),
    3,
    minimum=_CHAT_TOOL_CALLS_MIN,
    maximum=_CHAT_TOOL_CALLS_MAX,
)
_env_chat_max_tool_calls = os.getenv("CHAT_MAX_TOOL_CALLS")
if _env_chat_max_tool_calls is not None:
    _chat_max_tool_calls = _coerce_int_bounded(
        _env_chat_max_tool_calls,
        _chat_max_tool_calls,
        minimum=_CHAT_TOOL_CALLS_MIN,
        maximum=_CHAT_TOOL_CALLS_MAX,
    )
CHAT_MAX_TOOL_CALLS = _chat_max_tool_calls

_chat_tool_timeout_ms = _coerce_int_bounded(
    _chat_config.get("chat_tool_timeout_ms"),
    15_000,
    minimum=_CHAT_TOOL_TIMEOUT_MS_MIN,
    maximum=_CHAT_TOOL_TIMEOUT_MS_MAX,
)
_env_chat_tool_timeout_ms = os.getenv("CHAT_TOOL_TIMEOUT_MS")
if _env_chat_tool_timeout_ms is not None:
    _chat_tool_timeout_ms = _coerce_int_bounded(
        _env_chat_tool_timeout_ms,
        _chat_tool_timeout_ms,
        minimum=_CHAT_TOOL_TIMEOUT_MS_MIN,
        maximum=_CHAT_TOOL_TIMEOUT_MS_MAX,
    )
CHAT_TOOL_TIMEOUT_MS = _chat_tool_timeout_ms

_chat_tool_allow_catalog_raw = _chat_config.get("chat_tool_allow_catalog", _TOOL_ALLOW_CATALOG_DEFAULT)
_env_chat_tool_allow_catalog = os.getenv("CHAT_TOOL_ALLOW_CATALOG")
if _env_chat_tool_allow_catalog is not None:
    _chat_tool_allow_catalog_raw = _env_chat_tool_allow_catalog
CHAT_TOOL_ALLOW_CATALOG = _parse_tool_allow_catalog(_chat_tool_allow_catalog_raw)

_chat_tool_idempotency = _coerce_bool(_chat_config.get("chat_tool_idempotency"), True)
_env_chat_tool_idempotency = os.getenv("CHAT_TOOL_IDEMPOTENCY")
if _env_chat_tool_idempotency is not None:
    _chat_tool_idempotency = _coerce_bool(_env_chat_tool_idempotency, _chat_tool_idempotency)
CHAT_TOOL_IDEMPOTENCY = _chat_tool_idempotency

_chat_tool_auto_continue_once = _coerce_bool(_chat_config.get("chat_tool_auto_continue_once"), False)
_env_chat_tool_auto_continue_once = os.getenv("CHAT_TOOL_AUTO_CONTINUE_ONCE")
if _env_chat_tool_auto_continue_once is not None:
    _chat_tool_auto_continue_once = _coerce_bool(
        _env_chat_tool_auto_continue_once,
        _chat_tool_auto_continue_once,
    )
CHAT_TOOL_AUTO_CONTINUE_ONCE = _chat_tool_auto_continue_once


def should_force_normalize_string_responses() -> bool:
    """Return True when raw-string LLM responses should be wrapped in OpenAI format."""
    raw = os.getenv("CHAT_FORCE_NORMALIZE_STRING_RESPONSES")
    if raw is not None:
        return _coerce_bool(raw, FORCE_NORMALIZE_STRING_RESPONSES)
    return FORCE_NORMALIZE_STRING_RESPONSES


def should_auto_execute_tools() -> bool:
    """Return whether chat should auto-execute assistant tool calls."""
    raw = os.getenv("CHAT_AUTO_EXECUTE_TOOLS")
    if raw is not None:
        return _coerce_bool(raw, CHAT_AUTO_EXECUTE_TOOLS)
    return CHAT_AUTO_EXECUTE_TOOLS


def should_run_legacy_tool_autoexec(cleaned_args: dict[str, Any] | None) -> bool:
    """Gate legacy auto-exec when loop mode is explicitly enabled on a request."""
    if is_chat_loop_mode_enabled(cleaned_args):
        return False
    return should_auto_execute_tools()


def get_chat_max_tool_calls() -> int:
    """Return configured per-response cap for auto-executed tool calls."""
    raw = os.getenv("CHAT_MAX_TOOL_CALLS")
    if raw is not None:
        return _coerce_int_bounded(
            raw,
            CHAT_MAX_TOOL_CALLS,
            minimum=_CHAT_TOOL_CALLS_MIN,
            maximum=_CHAT_TOOL_CALLS_MAX,
        )
    return CHAT_MAX_TOOL_CALLS


def get_chat_tool_timeout_ms() -> int:
    """Return timeout budget in milliseconds for each auto-executed tool call."""
    raw = os.getenv("CHAT_TOOL_TIMEOUT_MS")
    if raw is not None:
        return _coerce_int_bounded(
            raw,
            CHAT_TOOL_TIMEOUT_MS,
            minimum=_CHAT_TOOL_TIMEOUT_MS_MIN,
            maximum=_CHAT_TOOL_TIMEOUT_MS_MAX,
        )
    return CHAT_TOOL_TIMEOUT_MS


def get_chat_tool_allow_catalog() -> list[str]:
    """Return allow-catalog patterns (empty list means no tools allowed)."""
    raw = os.getenv("CHAT_TOOL_ALLOW_CATALOG")
    if raw is not None:
        return _parse_tool_allow_catalog(raw)
    return CHAT_TOOL_ALLOW_CATALOG


def should_attach_tool_idempotency() -> bool:
    """Return whether auto-execution should attach idempotency keys for tools."""
    raw = os.getenv("CHAT_TOOL_IDEMPOTENCY")
    if raw is not None:
        return _coerce_bool(raw, CHAT_TOOL_IDEMPOTENCY)
    return CHAT_TOOL_IDEMPOTENCY


def should_auto_continue_tools_once() -> bool:
    """Return whether chat should run one follow-up assistant turn after tool execution."""
    raw = os.getenv("CHAT_TOOL_AUTO_CONTINUE_ONCE")
    if raw is not None:
        return _coerce_bool(raw, CHAT_TOOL_AUTO_CONTINUE_ONCE)
    return CHAT_TOOL_AUTO_CONTINUE_ONCE


def _resolve_chat_effective_tool_names(cleaned_args: dict[str, Any] | None) -> list[str]:
    """Return the resolved effective tool names for a chat call."""
    if not isinstance(cleaned_args, dict):
        return []

    names = cleaned_args.get("_chat_effective_tool_names")
    if isinstance(names, list) and all(isinstance(name, str) and name.strip() for name in names):
        return [name.strip() for name in names]

    tools = cleaned_args.get("tools")
    if not isinstance(tools, list):
        return []

    names: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        for name in tool_names_from_definitions(tool):
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def _resolve_chat_autoexec_allow_catalog(cleaned_args: dict[str, Any] | None) -> list[str]:
    """Return the allow-catalog derived from the presented chat tool set."""
    if not isinstance(cleaned_args, dict):
        return get_chat_tool_allow_catalog()

    if "_chat_effective_tool_names" in cleaned_args:
        return _resolve_chat_effective_tool_names(cleaned_args)
    derived_names = _resolve_chat_effective_tool_names(cleaned_args)
    if derived_names:
        return derived_names

    return get_chat_tool_allow_catalog()


def _resolve_chat_run_first_metric_context(
    cleaned_args: dict[str, Any] | None,
    *,
    provider: str,
    model: str,
    streaming: bool,
) -> dict[str, Any] | None:
    """Build normalized run-first telemetry labels for one chat execution."""
    if not isinstance(cleaned_args, dict):
        return None

    presentation_variant = cleaned_args.get("_chat_run_first_presentation_variant")
    if not isinstance(presentation_variant, str) or not presentation_variant.strip():
        return None

    eligible = bool(cleaned_args.get("_chat_run_first_eligible"))
    ineligible_reason = cleaned_args.get("_chat_run_first_ineligible_reason")
    if not isinstance(ineligible_reason, str) or not ineligible_reason.strip():
        ineligible_reason = None

    cohort = cleaned_args.get("_chat_run_first_cohort")
    if not isinstance(cohort, str) or not cohort.strip():
        cohort = "gated" if eligible else "control"

    return {
        "presentation_variant": presentation_variant.strip(),
        "cohort": cohort.strip(),
        "provider": str(provider or "").strip() or "unknown",
        "model": str(model or "").strip() or "unknown",
        "streaming": bool(streaming),
        "eligible": eligible,
        "ineligible_reason": ineligible_reason,
    }


def _tool_choice_name(tool_choice: Any) -> str | None:
    """Extract a pinned tool name from an OpenAI-style tool_choice payload."""
    if not isinstance(tool_choice, dict):
        return None

    function_block = tool_choice.get("function")
    if isinstance(function_block, dict):
        raw_name = function_block.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()

    raw_name = tool_choice.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()

    return None


def _tool_path_metric_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop rollout-only labels that tool-path/completion collectors do not accept."""
    if context is None:
        return None

    trimmed = dict(context)
    trimmed.pop("ineligible_reason", None)
    trimmed.pop("_completion_emitted", None)
    return trimmed


def _resolve_run_first_completion_outcome(exc: Exception) -> str:
    """Map post-provider exceptions into the completion-proxy outcome label."""
    if isinstance(exc, HTTPException):
        detail = str(getattr(exc, "detail", "") or "").strip().lower()
        if exc.status_code == status.HTTP_400_BAD_REQUEST and (
            "block" in detail or "moderation" in detail
        ):
            return "blocked"
    return "error"


def _tool_names_from_tool_calls(tool_calls: Any) -> list[str]:
    """Extract normalized tool names from OpenAI-style tool call payloads."""
    if not isinstance(tool_calls, list):
        return []

    names: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_block = tool_call.get("function")
        if not isinstance(function_block, dict):
            continue
        raw_name = function_block.get("name")
        if not isinstance(raw_name, str):
            continue
        name = raw_name.strip()
        if name:
            names.append(name)
    return names


def _emit_chat_run_first_rollout_metrics(
    metrics: Any,
    *,
    cleaned_args: dict[str, Any] | None,
    provider: str,
    model: str,
    streaming: bool,
) -> dict[str, Any] | None:
    """Emit the rollout exposure metric and return the normalized context."""
    context = _resolve_chat_run_first_metric_context(
        cleaned_args,
        provider=provider,
        model=model,
        streaming=streaming,
    )
    if context is None:
        return None

    try:
        metrics.track_run_first_rollout(**context)
    except _CHAT_RUN_FIRST_METRIC_EXCEPTIONS as exc:
        logger.warning("Chat run-first rollout metric emission failed: {}", exc)
    return context


def _emit_chat_run_first_tool_path_metrics(
    metrics: Any,
    *,
    context: dict[str, Any] | None,
    tool_calls: Any,
    function_call: Any = None,
) -> None:
    """Emit first-tool and fallback-after-run metrics for one assistant turn."""
    metric_context = _tool_path_metric_context(context)
    if metric_context is None:
        return

    tool_names = _tool_names_from_tool_calls(tool_calls)
    if not tool_names and isinstance(function_call, dict):
        raw_name = function_call.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            tool_names = [raw_name.strip()]
    if not tool_names:
        return

    first_tool = tool_names[0]
    try:
        metrics.track_run_first_first_tool(**metric_context, first_tool=first_tool)
    except _CHAT_RUN_FIRST_METRIC_EXCEPTIONS as exc:
        logger.warning("Chat run-first first-tool metric emission failed: {}", exc)

    if first_tool != "run":
        return

    fallback_tool = next((name for name in tool_names[1:] if name and name != "run"), None)
    if fallback_tool is None:
        return

    try:
        metrics.track_run_first_fallback_after_run(**metric_context, fallback_tool=fallback_tool)
    except _CHAT_RUN_FIRST_METRIC_EXCEPTIONS as exc:
        logger.warning("Chat run-first fallback metric emission failed: {}", exc)


def _emit_chat_run_first_completion_metric(
    metrics: Any,
    *,
    context: dict[str, Any] | None,
    outcome: str,
) -> None:
    """Emit the completion-proxy metric for one chat execution."""
    metric_context = _tool_path_metric_context(context)
    if context is None or metric_context is None:
        return
    if context.get("_completion_emitted"):
        return
    context["_completion_emitted"] = True
    try:
        metrics.track_run_first_completion_proxy(**metric_context, outcome=outcome)
    except _CHAT_RUN_FIRST_METRIC_EXCEPTIONS as exc:
        logger.warning("Chat run-first completion metric emission failed: {}", exc)


def queue_is_active(queue: Any) -> bool:
    """Return True when the request queue is running and able to process work."""
    try:
        status = queue.is_running
    except AttributeError:
        status = None
    if callable(status):
        try:
            result = status()
            if result is not None:
                return bool(result)
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
    elif status is not None:
        return bool(status)

    fallback_state = getattr(queue, "_running", None)
    if fallback_state is not None:
        return bool(fallback_state)
    # Assume truthy for lightweight test stubs that do not expose state
    return True


def _attach_queue_future_logger(future: asyncio.Future[Any], request_id: str) -> None:
    """Consume queue future exceptions to avoid unhandled warnings in streaming mode."""

    def _consume(fut: asyncio.Future[Any]) -> None:
        try:
            fut.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 - consume arbitrary queue/provider failures
            logger.debug(
                "Queue streaming job {} failed error_type={}",
                request_id,
                type(exc).__name__,
            )

    future.add_done_callback(_consume)


def _schedule_background_task(
    awaitable: Awaitable[Any],
    *,
    task_name: str,
    pending_tasks: list[asyncio.Task[Any]] | None = None,
) -> asyncio.Task[Any] | None:
    """Schedule a background task and observe failures to avoid silent task leaks."""

    try:
        task = asyncio.create_task(awaitable)
    except Exception as exc:  # noqa: BLE001 - scheduling boundary may reject arbitrary awaitables
        logger.debug(
            "Failed to schedule background task {} error_type={}",
            task_name,
            type(exc).__name__,
        )
        return None

    if pending_tasks is not None:
        pending_tasks.append(task)

    def _consume(completed: asyncio.Task[Any]) -> None:
        if pending_tasks is not None:
            with contextlib.suppress(ValueError):
                pending_tasks.remove(completed)
        if completed.cancelled():
            return
        try:
            exc = completed.exception()
        except asyncio.CancelledError:
            return
        except Exception as consume_exc:  # noqa: BLE001 - observe arbitrary task failures
            logger.debug(
                "Background task {} observation failed error_type={}",
                task_name,
                type(consume_exc).__name__,
            )
            return
        if exc is not None:
            logger.debug(
                "Background task {} failed error_type={}",
                task_name,
                type(exc).__name__,
            )

    task.add_done_callback(_consume)
    return task


def resolve_model_context_window(provider: str, model: str) -> Any:
    """Return server-owned context metadata without probing provider tokenizers."""

    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip()
    if not normalized_provider or not normalized_model:
        return None
    try:
        from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
            get_llm_provider_override,
        )

        override = get_llm_provider_override(normalized_provider)
        config = getattr(override, "config", None)
        if isinstance(config, dict):
            model_windows = config.get("model_context_windows")
            if isinstance(model_windows, dict) and normalized_model in model_windows:
                return model_windows[normalized_model]
            if "context_window" in config:
                return config["context_window"]
    except Exception:  # noqa: BLE001 - metadata failure falls back conservatively.
        logger.debug("Chat context-window override metadata unavailable")
    try:
        from tldw_Server_API.app.api.v1.endpoints.llm_providers import (
            get_model_metadata,
        )

        metadata = get_model_metadata(normalized_provider, normalized_model)
        return metadata.get("context_window") if isinstance(metadata, dict) else None
    except Exception:  # noqa: BLE001 - metadata failure falls back conservatively.
        return None


def _is_client_like_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a client/config error (not an upstream failure).

    Used to gate provider fallback: we only fall back on upstream/server errors,
    not on client-side mistakes such as bad credentials or malformed requests.
    """
    name_lower = type(exc).__name__.lower()
    return any(t in name_lower for t in (
        "authentication", "ratelimit", "rate_limit",
        "badrequest", "bad_request", "configuration",
    ))


def merge_api_keys_for_provider(
    provider: str,
    module_keys: dict[str, str | None] | None,
    dynamic_keys: dict[str, str | None],
    requires_key_map: dict[str, bool],
) -> tuple[str | None, str | None]:
    """Merge module-level and dynamic API keys, normalizing empties to None.

    Returns a tuple of (raw_value, normalized_value). The raw value is the
    original string (possibly empty) used to validate presence when a provider
    requires a key. The normalized value is None if empty-string-like.
    """
    def _normalize(value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    raw_dynamic = dynamic_keys.get(provider)
    raw_module = module_keys.get(provider) if module_keys else None

    # Prefer dynamic/runtime keys (env/config) over module-level defaults.
    # If dynamic is explicitly empty/None, fall back to module-level value.
    raw_val = raw_dynamic if raw_dynamic is not None and str(raw_dynamic).strip() != "" else raw_module

    norm_val = _normalize(raw_val)

    # No raise here - the caller enforces requirements using requires_key_map
    return raw_val, norm_val


def build_call_params_from_request(
    request_data: Any,
    target_api_provider: str,
    provider_api_key: str | None,
    templated_llm_payload: list[dict[str, Any]],
    final_system_message: str | None,
    app_config: dict[str, Any] | None = None,
    grammar_record: Mapping[str, Any] | None = None,
    resolved_model: str | None = None,
) -> dict[str, Any]:
    """Construct the cleaned argument dictionary for chat_api_call.

    Mirrors the transformation previously in the endpoint: renames OpenAI-style
    params to the generic names expected by chat_api_call and attaches
    provider/model/messages/system/stream flags. Structured `response_format`
    requests are negotiated here so every caller gets the downgraded outbound
    payload before the provider call is wrapped.
    """
    call_params = request_data.model_dump(
        exclude_none=True,
        exclude={
            "api_provider",
            "messages",
            "character_id",
            "conversation_id",
            "prompt_template_name",
            "stream",
            "save_to_db",
            "tldw_history_selection_v1",
            "tldw_history_server_existing_input_v1",
            "tldw_history_branch",
            "tldw_history_edit_v1",
            "tldw_history_admission_v1",
            "history_message_limit",
            "history_message_order",
            "research_context",
            "slash_command_injection_mode",
            "persona_exemplar_budget_tokens",
            "persona_exemplar_strategy",
            "persona_debug",
            "thinking_budget_tokens",
            "grammar_mode",
            "grammar_id",
            "grammar_inline",
            "grammar_override",
        },
    )
    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
        _is_server_managed_chat_request_field,
    )

    declared_fields = getattr(type(request_data), "model_fields", None)
    call_params = {
        key: value
        for key, value in call_params.items()
        if (
            not _is_server_managed_chat_request_field(key)
            and (not isinstance(declared_fields, Mapping) or key in declared_fields)
        )
    }
    call_model = resolved_model if resolved_model is not None else call_params.get("model")
    if isinstance(call_model, str) and call_model.strip():
        call_params["model"] = call_model.strip()
    else:
        call_params.pop("model", None)

    # Rename keys to match chat_api_call's generic signature
    if "temperature" in call_params:
        call_params["temp"] = call_params.pop("temperature")
    if "top_p" in call_params:
        top_p_value = call_params.pop("top_p")
        # Normalize to a single generic param; provider maps translate as needed
        call_params["topp"] = top_p_value
    if "user" in call_params:
        call_params["user_identifier"] = call_params.pop("user")
    response_format = call_params.get("response_format")
    if isinstance(response_format, dict):
        requested_type = str(response_format.get("type") or "").strip().lower()
        if requested_type == "json_schema":
            decision = negotiate_structured_response_mode(
                provider=target_api_provider,
                requested=response_format,
            )
            call_params["_structured_requested_response_format"] = dict(response_format)
            call_params["response_format"] = decision.response_format

    chat_tools = call_params.get("tools")
    rollout_mode = resolve_chat_run_first_rollout_mode()
    presentation_variant = resolve_chat_run_first_presentation_variant()
    provider_allowlist = resolve_chat_run_first_provider_allowlist()
    run_first_presentation = present_chat_tools(
        tools=chat_tools if isinstance(chat_tools, list) else None,
        allow_catalog=get_chat_tool_allow_catalog(),
        rollout_mode=rollout_mode,
        provider_key=f"{target_api_provider}:{call_params.get('model') or ''}",
        provider_allowlist=provider_allowlist,
        streaming=bool(getattr(request_data, "stream", False)),
        presentation_variant=presentation_variant,
    )
    if run_first_presentation.llm_tools:
        call_params["tools"] = run_first_presentation.llm_tools
    else:
        call_params.pop("tools", None)
    if run_first_presentation.tool_choice is not None:
        call_params["tool_choice"] = run_first_presentation.tool_choice
    else:
        pinned_tool_choice = call_params.get("tool_choice")
        pinned_tool_name = _tool_choice_name(pinned_tool_choice)
        effective_tool_names = set(run_first_presentation.effective_tool_names)
        if pinned_tool_name and pinned_tool_name not in effective_tool_names:
            call_params.pop("tool_choice", None)
    call_params["_chat_effective_tool_names"] = run_first_presentation.effective_tool_names
    call_params["_chat_run_first_eligible"] = run_first_presentation.eligible
    if run_first_presentation.ineligible_reason is not None:
        call_params["_chat_run_first_ineligible_reason"] = run_first_presentation.ineligible_reason
    call_params["_chat_run_first_presentation_variant"] = run_first_presentation.presentation_variant
    call_params["_chat_run_first_cohort"] = resolve_run_first_cohort_label(
        rollout_mode,
        eligible=run_first_presentation.eligible,
        ineligible_reason=run_first_presentation.ineligible_reason,
    )
    if run_first_presentation.prompt_fragment:
        if final_system_message and final_system_message.strip():
            final_system_message = f"{final_system_message.rstrip()}\n\n{run_first_presentation.prompt_fragment}"
        else:
            final_system_message = run_first_presentation.prompt_fragment

    llamacpp_request_fields = {
        "thinking_budget_tokens": getattr(request_data, "thinking_budget_tokens", None),
        "grammar_mode": getattr(request_data, "grammar_mode", None),
        "grammar_id": getattr(request_data, "grammar_id", None),
        "grammar_inline": getattr(request_data, "grammar_inline", None),
        "grammar_override": getattr(request_data, "grammar_override", None),
        "extra_body": getattr(request_data, "extra_body", None),
    }
    call_params["extra_body"] = resolve_llamacpp_request_extensions(
        request_fields=llamacpp_request_fields,
        provider=target_api_provider,
        grammar_record=grammar_record,
        runtime_caps=resolve_llamacpp_runtime_caps(app_config=app_config),
    )["extra_body"]
    for extension in ("extra_body", "extra_headers"):
        if isinstance(call_params.get(extension), dict):
            call_params[extension] = {
                key: value
                for key, value in call_params[extension].items()
                if key.lower().replace("-", "_")
                not in {
                    "tldw_history_selection_v1",
                    "tldw_history_branch",
                    "tldw_history_edit_v1",
                    "tldw_history_admission_v1",
                    "tldw_history_server_existing_input_v1",
                }
            }
    billing_prompt_cache_intent = getattr(request_data, "billing_prompt_cache_intent", None)
    if billing_prompt_cache_intent is not None:
        call_params["billing_prompt_cache_intent"] = billing_prompt_cache_intent

    call_params.update(
        {
            "api_endpoint": target_api_provider,
            "api_key": provider_api_key,
            "messages_payload": templated_llm_payload,
            "system_message": final_system_message,
            "streaming": getattr(request_data, "stream", False),
            "llm_surface": CHAT_COMPLETIONS_SURFACE,
        }
    )
    if app_config is not None:
        call_params["app_config"] = app_config

    # Filter Nones; keep explicit None for system_message only if provided
    cleaned_args = {k: v for k, v in call_params.items() if v is not None}
    if "system_message" not in cleaned_args and final_system_message is not None:
        cleaned_args["system_message"] = final_system_message
    return cleaned_args


def _format_research_context_block(research_context: Any) -> str | None:
    """Convert bounded deep research context into a model-side system block."""
    if research_context is None:
        return None

    if hasattr(research_context, "model_dump"):
        context_data = research_context.model_dump(exclude_none=True)
    elif isinstance(research_context, dict):
        context_data = {k: v for k, v in research_context.items() if v is not None}
    else:
        return None

    run_id = str(context_data.get("run_id") or "").strip()
    query = str(context_data.get("query") or "").strip()
    question = str(context_data.get("question") or "").strip()
    research_url = str(context_data.get("research_url") or "").strip()
    if not run_id and not query and not question:
        return None

    lines = ["Deep Research Context", f"Run ID: {run_id}" if run_id else None, f"Query: {query}" if query else None]
    if question:
        lines.append(f"Question: {question}")

    outline = context_data.get("outline")
    if isinstance(outline, list) and outline:
        lines.append("Outline:")
        for section in outline:
            if hasattr(section, "model_dump"):
                section = section.model_dump(exclude_none=True)
            if isinstance(section, dict):
                title = str(section.get("title") or "").strip()
                if title:
                    lines.append(f"- {title}")

    key_claims = context_data.get("key_claims")
    if isinstance(key_claims, list) and key_claims:
        lines.append("Key Claims:")
        for claim in key_claims:
            if hasattr(claim, "model_dump"):
                claim = claim.model_dump(exclude_none=True)
            if isinstance(claim, dict):
                claim_text = str(claim.get("text") or "").strip()
                if claim_text:
                    lines.append(f"- {claim_text}")

    unresolved = context_data.get("unresolved_questions")
    if isinstance(unresolved, list) and unresolved:
        lines.append("Unresolved Questions:")
        for item in unresolved:
            text = str(item or "").strip()
            if text:
                lines.append(f"- {text}")

    verification_summary = context_data.get("verification_summary")
    if hasattr(verification_summary, "model_dump"):
        verification_summary = verification_summary.model_dump(exclude_none=True)
    if isinstance(verification_summary, dict) and verification_summary:
        unsupported_count = verification_summary.get("unsupported_claim_count")
        if unsupported_count is not None:
            lines.append(f"Unsupported Claims: {unsupported_count}")

    source_trust_summary = context_data.get("source_trust_summary")
    if hasattr(source_trust_summary, "model_dump"):
        source_trust_summary = source_trust_summary.model_dump(exclude_none=True)
    if isinstance(source_trust_summary, dict) and source_trust_summary:
        high_trust_count = source_trust_summary.get("high_trust_count")
        if high_trust_count is not None:
            lines.append(f"High-Trust Sources: {high_trust_count}")

    if research_url:
        lines.append(f"Research URL: {research_url}")

    return "\n".join(line for line in lines if line)


def inject_research_context_into_prompt(
    *,
    final_system_message: str | None,
    templated_llm_payload: list[dict[str, Any]],
    research_context: Any,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Attach bounded research context to the model-facing system prompt only."""
    block = _format_research_context_block(research_context)
    if not block:
        return final_system_message, list(templated_llm_payload)

    if final_system_message and final_system_message.strip():
        return f"{final_system_message.rstrip()}\n\n{block}", list(templated_llm_payload)
    return block, list(templated_llm_payload)


def estimate_tokens_from_json(request_json: str) -> int:
    """Heuristic prompt size for rate limits, quotas, billing and queue admission.

    The chat path's one estimate (``estimate_segment_tokens``: data:image payloads
    redacted, four characters per token rounded up), never below 1.
    """
    try:
        return max(1, estimate_segment_tokens(request_json))
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        return 1


def _estimate_tokens_from_messages(messages: list[dict[str, Any]]) -> int:
    """The same estimate for a message list, taken over its JSON."""
    try:
        return estimate_tokens_from_json(_json.dumps(messages, default=str))
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        return 1


def _extract_text_from_content(content: Any | None) -> str:
    """Delegate response content extraction to the response processor."""
    return extract_text_from_content(content)


_NONSTREAM_REASONING_FIELDS = (
    "reasoning_content",
    "reasoning",
    "reasoning_details",
    "thinking",
    "analysis",
)
_NONSTREAM_REASONING_METADATA_FIELDS = frozenset({"type", "id", "index", "signature"})
_NONSTREAM_REASONING_MAX_NODES = 1024
_NONSTREAM_REASONING_MAX_DEPTH = 8


def _inspect_nonstream_text_output(value: str) -> tuple[bool, bool]:
    """Return semantic-text and rejected-protocol state for provider text."""

    if provider_result_contains_error(value, legacy_error_prefix=True):
        return False, True
    stripped = value.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
    if not stripped:
        return False, False
    if stripped.lower() == "[done]" or is_done_line(stripped):
        return False, True
    first_line = stripped.splitlines()[0].strip().lower()
    if first_line.startswith(("data:", "event:", "id:", "retry:", ":")):
        return False, True
    return True, False


def _inspect_nonstream_reasoning_output(value: Any) -> tuple[bool, bool]:
    """Return semantic-text and rejected-structure state for reasoning output."""

    stack = [(value, 0, True)]
    seen: set[int] = set()
    semantic_text_seen = False
    visited_nodes = 0

    while stack:
        item, depth, semantic_allowed = stack.pop()
        visited_nodes += 1
        if (
            visited_nodes > _NONSTREAM_REASONING_MAX_NODES
            or depth > _NONSTREAM_REASONING_MAX_DEPTH
        ):
            return False, True
        if isinstance(item, str):
            if semantic_allowed:
                has_semantic_text, rejected_protocol = (
                    _inspect_nonstream_text_output(item)
                )
                if rejected_protocol:
                    return False, True
                semantic_text_seen = semantic_text_seen or has_semantic_text
            continue
        if not isinstance(item, (list, dict)):
            continue

        identity = id(item)
        if identity in seen:
            return False, True
        seen.add(identity)

        if isinstance(item, dict):
            if normalize_provider_stream_error(item) is not None:
                return False, True
            stack.extend(
                (
                    nested,
                    depth + 1,
                    semantic_allowed
                    and field not in _NONSTREAM_REASONING_METADATA_FIELDS,
                )
                for field, nested in item.items()
            )
        else:
            stack.extend(
                (nested, depth + 1, semantic_allowed) for nested in item
            )

    return semantic_text_seen, False


def _nonstream_provider_result_is_usable(result: Any) -> bool:
    """Return whether one non-stream provider result has usable Chat semantics."""

    if isinstance(result, str):
        has_semantic_text, rejected_protocol = _inspect_nonstream_text_output(result)
        return has_semantic_text and not rejected_protocol
    if not isinstance(result, dict):
        return False
    if provider_payload_has_structural_error(result):
        return False

    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return False

    for choice in choices:
        if not isinstance(choice, dict):
            return False
        message = choice.get("message")
        if not isinstance(message, dict):
            return False

        content = message.get("content")
        content_is_usable = isinstance(content, str) and bool(content.strip())
        if isinstance(content, list):
            content_is_usable = bool(content)
            for part in content:
                if not isinstance(part, dict):
                    content_is_usable = False
                    break
                part_type = part.get("type")
                if part_type in {"text", "output_text"}:
                    text = part.get("text")
                    if not isinstance(text, str) or not text.strip():
                        content_is_usable = False
                        break
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else None
                    if not isinstance(url, str) or not url.strip():
                        content_is_usable = False
                        break
                else:
                    content_is_usable = False
                    break
        elif content is not None and not isinstance(content, str):
            return False

        tool_calls = message.get("tool_calls")
        tool_calls_are_usable = False
        if tool_calls is not None:
            if not isinstance(tool_calls, list) or not tool_calls:
                return False
            tool_calls_are_usable = True
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    return False
                function = tool_call.get("function")
                if (
                    not isinstance(tool_call.get("id"), str)
                    or not tool_call["id"].strip()
                    or tool_call.get("type") != "function"
                    or not isinstance(function, dict)
                    or not isinstance(function.get("name"), str)
                    or not function["name"].strip()
                    or not isinstance(function.get("arguments"), str)
                    or not function["arguments"].strip()
                ):
                    return False

        function_call = message.get("function_call")
        function_call_is_usable = False
        if function_call is not None:
            if (
                not isinstance(function_call, dict)
                or not isinstance(function_call.get("name"), str)
                or not function_call["name"].strip()
                or not isinstance(function_call.get("arguments"), str)
                or not function_call["arguments"].strip()
            ):
                return False
            function_call_is_usable = True

        refusal = message.get("refusal")
        refusal_is_usable = isinstance(refusal, str) and bool(refusal.strip())
        for field in _NONSTREAM_REASONING_FIELDS:
            if field not in message:
                continue
            _, rejected_structure = (
                _inspect_nonstream_reasoning_output(message[field])
            )
            if rejected_structure:
                return False
        content_filter_is_usable = (
            choice.get("finish_reason") == "content_filter"
            and (content is None or content == "")
            and refusal is None
        )

        if not (
            content_is_usable
            or tool_calls_are_usable
            or function_call_is_usable
            or refusal_is_usable
            or content_filter_is_usable
        ):
            return False

    return True


def _nonstream_reasoning_exhausted_output_limit(result: Any) -> bool:
    """Recognize valid hidden-only output stopped by the provider token limit."""
    if not isinstance(result, dict) or provider_payload_has_structural_error(result):
        return False
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return False
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "length":
        return False
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("content") not in (None, ""):
        return False
    if any(
        message.get(field) is not None
        for field in ("tool_calls", "function_call", "refusal")
    ):
        return False
    reasoning_seen = False
    for field in _NONSTREAM_REASONING_FIELDS:
        if field not in message:
            continue
        has_text, rejected = _inspect_nonstream_reasoning_output(message[field])
        if rejected:
            return False
        reasoning_seen = reasoning_seen or has_text
    return reasoning_seen


def _require_usable_nonstream_provider_result(result: Any) -> None:
    """Raise one bounded, non-replayable error for an invalid provider result."""

    if not _nonstream_provider_result_is_usable(result):
        code = (
            "provider_output_limit"
            if _nonstream_reasoning_exhausted_output_limit(result)
            else "provider_unavailable"
        )
        raise_detached_error(sanitized_provider_stream_exception(code))


def _apply_redaction_to_content(content: Any, moderation: Any, policy: Any) -> Any:
    """Apply redaction to text parts while preserving non-text content when possible."""
    if content is None:
        return content
    if isinstance(content, str):
        return moderation.redact_text(content, policy)
    if isinstance(content, list):
        redacted_parts: list[Any] = []
        for part in content:
            if isinstance(part, str):
                redacted_parts.append(moderation.redact_text(part, policy))
                continue
            if isinstance(part, dict):
                if part.get("type") == "text":
                    new_part = dict(part)
                    new_part["text"] = moderation.redact_text(part.get("text", ""), policy)
                    redacted_parts.append(new_part)
                else:
                    redacted_parts.append(part)
                continue
            try:
                p_type = getattr(part, "type", None)
                if p_type == "text":
                    text_val = getattr(part, "text", "")
                    new_text = moderation.redact_text(text_val, policy)
                    try:
                        updated = part.model_copy(update={"text": new_text})
                        redacted_parts.append(updated)
                    except _CHAT_NONCRITICAL_EXCEPTIONS:
                        redacted_parts.append({"type": "text", "text": new_text})
                else:
                    redacted_parts.append(part)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                redacted_parts.append(part)
        return redacted_parts
    return moderation.redact_text(str(content), policy)


def _wrap_raw_string_response(content: str, model: str | None) -> dict[str, Any]:
    """Wrap raw string responses in an OpenAI-compatible payload."""
    model_name = model or "unknown"
    return {
        "id": f"chatcmpl-{_uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _extract_structured_schema_from_response_format(response_format: Any) -> dict[str, Any] | None:
    """Return the requested JSON Schema payload when `response_format` uses `json_schema`."""

    if not isinstance(response_format, dict):
        return None
    if str(response_format.get("type") or "").strip().lower() != "json_schema":
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return None
    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        return None
    return schema


@dataclass(frozen=True)
class StructuredResponseRequestContext:
    """Resolved structured-output state shared across request execution and validation."""

    requested_response_format: dict[str, Any]
    validation_schema: dict[str, Any]
    decision: StructuredModeDecision


def prepare_structured_response_request(
    *,
    provider: str,
    response_format: Any,
    requested_response_format: Any | None = None,
) -> StructuredResponseRequestContext | None:
    """Resolve structured-output negotiation for the original request payload."""

    candidate_response_format = requested_response_format
    if not isinstance(candidate_response_format, dict):
        candidate_response_format = response_format
    if not isinstance(candidate_response_format, dict):
        return None

    validation_schema = _extract_structured_schema_from_response_format(candidate_response_format)
    if validation_schema is None:
        return None

    decision = negotiate_structured_response_mode(
        provider=provider,
        requested=candidate_response_format,
    )
    return StructuredResponseRequestContext(
        requested_response_format=dict(candidate_response_format),
        validation_schema=validation_schema,
        decision=decision,
    )


def apply_structured_response_request(
    *,
    cleaned_args: dict[str, Any],
    structured_request_context: StructuredResponseRequestContext | None,
) -> None:
    """Apply the negotiated outbound response format to provider call arguments."""

    if structured_request_context is None:
        return
    cleaned_args["response_format"] = structured_request_context.decision.response_format


def validate_structured_response(
    *,
    raw_text: Any,
    structured_request_context: StructuredResponseRequestContext | None,
) -> dict[str, Any] | None:
    """Validate final assistant output against the original requested JSON Schema."""

    if structured_request_context is None:
        return None

    validated_payload = parse_and_validate_structured_output(
        raw_text=raw_text,
        schema=structured_request_context.validation_schema,
    )
    return {
        "validated": True,
        "mode_used": structured_request_context.decision.mode_used,
        "fallback_used": structured_request_context.decision.fallback_used,
        "validated_payload": validated_payload,
    }


def _structured_error_detail(exc: StructuredGenerationError) -> dict[str, Any]:
    """Return a bounded, client-safe structured-output error payload."""

    error_code = getattr(exc, "code", "structured_output_error")
    message_by_code = {
        "structured_output_capability_error": "Requested structured output mode is not supported for this provider.",
        "structured_output_parse_error": "Model output could not be parsed as JSON.",
        "structured_output_schema_error": "Model output did not match the requested JSON schema.",
    }
    detail: dict[str, Any] = {
        "code": error_code,
        "message": message_by_code.get(error_code, "Structured output generation failed."),
    }
    attempts = getattr(exc, "attempts", None)
    if isinstance(attempts, int):
        detail["attempts"] = attempts
    return detail


def build_structured_http_exception(exc: StructuredGenerationError) -> HTTPException:
    """Normalize structured-output failures into a 400 response."""

    structured_http_error = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=_structured_error_detail(exc),
    )
    return _mark_trusted_local_chat_signal(structured_http_error, "structured_error")


def _is_trusted_local_chat_http_exception(exc: BaseException) -> bool:
    """Return whether an HTTP error was created and bounded by this service.

    An LLM quota refusal is trusted: it is the user's limit, not a provider failure,
    and reaches the caller as its own 402 (LLM usage accounting spec §4b).
    A policy refusal is trusted too: the hook's decision, not a provider failure (TASK-13950.13).
    """

    return isinstance(exc, (UnsupportedMultiChoiceToolAutoexecError, LLMQuotaExceededError, LLMPolicyRefusal)) or (
        isinstance(exc, HTTPException)
        and trusted_local_chat_signal_kind(exc) is not None
    )


def _bound_provider_http_exception(exc: BaseException) -> BaseException:
    """Replace untrusted provider HTTP errors with a bounded chat exception."""

    if _is_trusted_local_chat_http_exception(exc):
        return exc
    if isinstance(exc, (SanitizedProviderStreamError, ProviderCredentialTerminalError)):
        return exc
    if isinstance(exc, ByokResolutionError):
        return ProviderCredentialTerminalError(
            getattr(exc, "policy_code", exc.code)
        )
    return sanitized_provider_stream_exception(exc)


def _bound_provider_invocation_exception(exc: BaseException) -> BaseException:
    """Bound untyped adapter failures while preserving trusted chat signals."""

    if isinstance(exc, (SanitizedProviderStreamError, ProviderCredentialTerminalError)):
        return exc
    if _is_trusted_local_chat_http_exception(exc):
        return exc
    if isinstance(exc, ChatAuthenticationError):
        bounded = ChatAuthenticationError(
            message="The selected provider credentials could not be authenticated.",
            status_code=403 if getattr(exc, "status_code", None) == 403 else 401,
        )
        if getattr(exc, "credential_refresh_retry_safe", None) is True:
            bounded.credential_refresh_retry_safe = True
        return bounded
    if isinstance(exc, ChatConfigurationError):
        return ChatConfigurationError(
            message="The selected provider configuration is invalid.",
            error_code=getattr(exc, "error_code", "provider_configuration_invalid"),
        )
    if isinstance(exc, ChatModelNotAvailableError):
        return exc  # its message is built from the request's provider and model, never upstream text
    if isinstance(exc, ChatBadRequestError):
        return ChatBadRequestError(
            provider=getattr(exc, "provider", None),
            message="Invalid provider request.",
        )
    if isinstance(exc, ChatRateLimitError):
        return ChatRateLimitError(
            provider=getattr(exc, "provider", None),
            message="Rate limit exceeded with the chat provider.",
        )
    return sanitized_provider_stream_exception(exc)


def _build_assistant_message_payload(
    *,
    character_card_for_context: dict[str, Any] | None,
    assistant_parent_message_id: str | None,
    content: Any | None,
    tool_calls: Any | None,
    function_call: Any | None,
    generation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for tests/imports that use the old local helper."""

    return build_assistant_message_payload(
        character_card_for_context=character_card_for_context,
        assistant_parent_message_id=assistant_parent_message_id,
        content=content,
        tool_calls=tool_calls,
        function_call=function_call,
        generation_metadata=generation_metadata,
    )


def _nonstream_generation_metadata(
    *,
    llm_response: Any,
    choice: NonStreamChoice | None,
    model: str,
    provider: str,
) -> dict[str, Any]:
    """Describe how a non-streamed reply was produced, for settlement (D7 P2)."""

    raw_choice = choice.choice if choice is not None else None
    finish_reason = raw_choice.get("finish_reason") if isinstance(raw_choice, dict) else None
    usage = llm_response.get("usage") if isinstance(llm_response, dict) else None
    return build_generation_metadata(
        generation_status=(
            GENERATION_STATUS_LENGTH if finish_reason == "length" else GENERATION_STATUS_COMPLETE
        ),
        model_id=model,
        provider=provider,
        finish_reason=finish_reason,
        usage=usage,
    )


async def write_mandatory_moderation_audit(
    *,
    audit_service: Any | None,
    audit_context: Any | None,
    audit_event_type: Any | None,
    action: str,
    result: str,
    metadata: dict[str, Any],
) -> None:
    """Write a moderation audit event and fail closed if persistence is unavailable."""
    if audit_service is None or audit_context is None:
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    try:
        await audit_service.log_event(
            event_type=audit_event_type or AuditEventType.SECURITY_VIOLATION,
            context=audit_context,
            action=action,
            result=result,
            metadata=metadata,
        )
        await audit_service.flush(raise_on_failure=True)
    except MandatoryAuditWriteError:
        raise
    except Exception as exc:
        logger.error(
            "Mandatory moderation audit write failed for {} ({}) error_type={}",
            action,
            result,
            type(exc).__name__,
        )
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable") from exc


def _capture_moderation_review_item_safely(
    *,
    phase: str,
    action: str | None,
    excerpt: str | None,
    category: str | None,
    matched_pattern: str | None,
    effective_policy: Any,
    source_id: str | None,
    user_id: str | None,
    session_id: str | None = None,
) -> None:
    """Best-effort gated capture of sanitized moderation review metadata."""
    if not action or action == "pass" or not is_moderation_review_capture_enabled():
        return
    try:
        policy_snapshot = effective_policy.to_dict() if hasattr(effective_policy, "to_dict") else {}
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        policy_snapshot = {}
    try:
        capture_moderation_review_item(
            phase=phase,
            action=action,
            excerpt=excerpt,
            category=category,
            matched_pattern=matched_pattern,
            effective_policy=policy_snapshot,
            source_type="chat",
            source_id=source_id,
            user_id=user_id,
            session_id=session_id,
        )
    except _CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Moderation review capture failed in chat service: {}", exception_summary(exc))


async def _capture_moderation_review_item_safely_async(**kwargs: Any) -> None:
    action = kwargs.get("action")
    if not action or action == "pass" or not is_moderation_review_capture_enabled():
        return
    await asyncio.to_thread(partial(_capture_moderation_review_item_safely, **kwargs))


async def moderate_input_messages(
    request_data: Any,
    request: Any,
    moderation_service: Any,
    topic_monitoring_service: Any | None,
    metrics: Any,
    audit_service: Any | None,
    audit_context: Any | None,
    client_id: str,
    audit_event_type: Any | None = None,
    supervised_policy_engine: Any | None = None,
    self_monitoring_service: Any | None = None,
    dependent_user_id: str | None = None,
    chat_type: str | None = None,
) -> None:
    """Apply input moderation and redaction to user message text parts in-place.

    - Emits topic monitoring alerts non-blockingly when configured.
    - Tracks moderation metrics and audit events.
    - Optionally overlays guardian supervised policies when engine is provided.
    - Optionally runs self-monitoring checks when service is provided.
    - Raises HTTPException(400) when input is blocked by policy.
    """
    # Determine user id context for policy and telemetry
    req_user_id = None
    try:
        if request is not None and hasattr(request, "state"):
            req_user_id = getattr(request.state, "user_id", None)
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        req_user_id = None

    eff_policy = moderation_service.get_effective_policy(str(req_user_id) if req_user_id is not None else client_id)

    # Guardian policy overlay: merge supervised rules into eff_policy
    _guardian_rule_name_visible = None
    if supervised_policy_engine and dependent_user_id:
        try:
            eff_policy = supervised_policy_engine.build_moderation_policy_overlay(
                dependent_user_id, eff_policy, chat_type=chat_type,
            )
        except _CHAT_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("Guardian policy overlay skipped: {}", exception_summary(e))
        # Guardian notification dispatch: check supervised policies directly
        # and send notification to guardian if a rule triggers.
        # Also stores transparent rule_name for enriched block messages.
        try:
            from tldw_Server_API.app.core.Moderation.supervised_policy import (
                dispatch_guardian_notification,
            )
            _first_text = None
            if request_data and request_data.messages:
                for _m in request_data.messages:
                    if getattr(_m, "role", None) == "user":
                        _mt = getattr(_m, "content", None)
                        if isinstance(_mt, str) and _mt.strip():
                            _first_text = _mt
                            break
            if _first_text:
                _sup_result = supervised_policy_engine.check_text(
                    _first_text, dependent_user_id, phase="input", chat_type=chat_type,
                )
                if _sup_result.rule_name_visible:
                    _guardian_rule_name_visible = _sup_result.rule_name_visible
                if _sup_result.notify_guardian and _sup_result.action != "pass":
                    dispatch_guardian_notification(_sup_result, dependent_user_id)
        except _CHAT_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("Guardian notification dispatch skipped: {}", exception_summary(e))
    conv_id = None
    try:
        conv_id = getattr(request_data, "conversation_id", None)
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        conv_id = None

    async def _moderate_text_in_place(text: str) -> str:
        # Topic monitoring (non-blocking)
        try:
            mon = topic_monitoring_service
            team_ids = None
            org_ids = None
            try:
                if request is not None and hasattr(request, "state"):
                    team_ids = getattr(request.state, "team_ids", None)
                    org_ids = getattr(request.state, "org_ids", None)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                pass
            if mon is not None and text:
                mon.schedule_evaluate_and_alert(
                    user_id=str(req_user_id or client_id) if (req_user_id or client_id) else None,
                    text=text,
                    source="chat.input",
                    scope_type="user",
                    scope_id=str(req_user_id or client_id) if (req_user_id or client_id) else None,
                    team_ids=team_ids,
                    org_ids=org_ids,
                    source_id=str(conv_id) if conv_id is not None else None,
                )
        except _CHAT_NONCRITICAL_EXCEPTIONS as _e:
            logger.debug("Topic monitoring (input) skipped: {}", exception_summary(_e))

        # Self-monitoring check (awareness/notifications)
        if self_monitoring_service and text:
            try:
                _sm_loop = asyncio.get_running_loop()
                sm_result = await _sm_loop.run_in_executor(
                    None,
                    lambda: self_monitoring_service.check_text(
                        text=text,
                        user_id=str(req_user_id or client_id),
                        phase="input",
                        conversation_id=str(conv_id) if conv_id else None,
                        chat_type=chat_type,
                    ),
                )
                if sm_result.action == "block":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=sm_result.block_message or "Blocked by self-monitoring rule",
                    )
                if sm_result.action == "redact" and sm_result.redacted_text is not None:
                    text = sm_result.redacted_text
            except HTTPException:
                raise
            except _CHAT_NONCRITICAL_EXCEPTIONS as e:
                logger.debug("Self-monitoring check skipped: {}", exception_summary(e))

        if not eff_policy.enabled or not eff_policy.input_enabled:
            return text

        resolved_action = None
        sample = None
        redacted = None
        category = None
        matched_pattern = None
        match_span = None
        if hasattr(moderation_service, "evaluate_action_with_match"):
            try:
                eval_res = moderation_service.evaluate_action_with_match(text, eff_policy, "input")
                if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                    resolved_action, redacted, matched_pattern = eval_res[0], eval_res[1], eval_res[2]
                    category = eval_res[3] if len(eval_res) >= 4 else None
                    match_span = eval_res[4] if len(eval_res) >= 5 else None
                else:
                    resolved_action, redacted, matched_pattern = eval_res  # type: ignore
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                resolved_action = None
            if match_span and hasattr(moderation_service, "build_sanitized_snippet"):
                try:
                    sample = moderation_service.build_sanitized_snippet(text, eff_policy, match_span, matched_pattern)
                except _CHAT_NONCRITICAL_EXCEPTIONS:
                    sample = None
        elif hasattr(moderation_service, "evaluate_action"):
            try:
                eval_res = moderation_service.evaluate_action(text, eff_policy, "input")
                if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                    resolved_action, redacted, matched_pattern = eval_res[0], eval_res[1], eval_res[2]
                    category = eval_res[3] if len(eval_res) >= 4 else None
                else:
                    resolved_action, redacted, matched_pattern = eval_res  # type: ignore
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                resolved_action = None
            if resolved_action and resolved_action != "pass" and sample is None:
                try:
                    _, sample = moderation_service.check_text(text, eff_policy, "input")
                except _CHAT_NONCRITICAL_EXCEPTIONS:
                    sample = None
        if not resolved_action:
            flagged, sample = moderation_service.check_text(text, eff_policy, "input")
            if not flagged:
                return text
            resolved_action = eff_policy.input_action
            redacted = (
                moderation_service.redact_text(text, eff_policy)
                if resolved_action == "redact"
                else None
            )

        if resolved_action == "pass":
            return text

        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
            metrics.track_moderation_input(str(req_user_id or client_id), resolved_action, category=(category or "default"))

        await write_mandatory_moderation_audit(
            audit_service=audit_service,
            audit_context=audit_context,
            audit_event_type=audit_event_type,
            action="moderation.input",
            result=("failure" if resolved_action == "block" else "success"),
            metadata={"phase": "input", "action": resolved_action, "pattern": sample},
        )

        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
            await _capture_moderation_review_item_safely_async(
                phase="input",
                action=resolved_action,
                excerpt=sample,
                category=category,
                matched_pattern=matched_pattern,
                effective_policy=eff_policy,
                source_id=str(conv_id) if conv_id is not None else None,
                user_id=str(req_user_id or client_id) if (req_user_id or client_id) else None,
            )

        if resolved_action == "block":
            block_detail = "Input violates moderation policy"
            if _guardian_rule_name_visible:
                block_detail = f"[{_guardian_rule_name_visible}] {block_detail}"
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=block_detail)
        if resolved_action == "redact":
            return redacted if isinstance(redacted, str) else moderation_service.redact_text(text, eff_policy)
        return text

    # Apply moderation across request messages
    # Moderate both "user" messages and "tool" messages (tool call results may contain
    # external content that should be checked for policy violations)
    MODERATED_ROLES = {"user", "tool"}
    _should_moderate = (eff_policy.enabled and eff_policy.input_enabled) or self_monitoring_service
    try:
        if _should_moderate and request_data and request_data.messages:
            for m in request_data.messages:
                if getattr(m, "role", None) not in MODERATED_ROLES:
                    continue
                if isinstance(m.content, str):
                    m.content = await _moderate_text_in_place(m.content)
                elif isinstance(m.content, list):
                    for part in m.content:
                        try_type = getattr(part, "type", None)
                        if try_type == "text":
                            current = getattr(part, "text", None)
                            if isinstance(current, str):
                                part.text = await _moderate_text_in_place(current)
    except HTTPException:
        raise
    except MandatoryAuditWriteError:
        raise
    except _CHAT_NONCRITICAL_EXCEPTIONS as e:
        logger.warning("Moderation input processing error: {}", exception_summary(e))


def _extract_tldw_continuation_spec(request_data: Any) -> tuple[str, str, str | None] | None:
    """Return normalized continuation tuple (anchor_id, mode, assistant_prefill)."""
    raw_spec = getattr(request_data, "tldw_continuation", None)
    if raw_spec is None:
        return None

    if hasattr(raw_spec, "model_dump"):
        try:
            spec_dict = raw_spec.model_dump(exclude_none=True)
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            spec_dict = {}
    elif isinstance(raw_spec, dict):
        spec_dict = raw_spec
    else:
        return None

    from_message_id = str(spec_dict.get("from_message_id") or "").strip()
    mode = str(spec_dict.get("mode") or "").strip().lower()
    assistant_prefill_raw = spec_dict.get("assistant_prefill")
    assistant_prefill = assistant_prefill_raw if isinstance(assistant_prefill_raw, str) else None

    if not from_message_id or mode not in {"branch", "append"}:
        return None
    return from_message_id, mode, assistant_prefill


async def _resolve_tldw_continuation_history(
    *,
    chat_db: Any,
    loop: Any,
    conversation_id: str,
    from_message_id: str,
    mode: str,
    history_limit: int,
    history_order: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve continuation anchor/chain and return history records + metadata."""
    anchor_record = await asyncio.to_thread(chat_db.get_message_by_id, from_message_id)
    if not anchor_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Continuation anchor message was not found.",
        )
    if str(anchor_record.get("conversation_id") or "") != str(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Continuation anchor does not belong to the requested conversation.",
        )

    if mode == "append":
        latest_message = await asyncio.to_thread(chat_db.get_latest_message_for_conversation,
            conversation_id,
        )
        latest_id = str((latest_message or {}).get("id") or "")
        if latest_id != from_message_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Append continuation requires the anchor to be the latest "
                    "message in the conversation."
                ),
            )

    chain_reversed: list[dict[str, Any]] = []
    visited_ids: set[str] = set()
    current_record = anchor_record
    while current_record:
        message_id = str(current_record.get("id") or "")
        if not message_id:
            break
        if message_id in visited_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Detected a cycle while resolving continuation ancestry.",
            )
        visited_ids.add(message_id)
        chain_reversed.append(current_record)

        parent_id_raw = current_record.get("parent_message_id")
        parent_id = parent_id_raw.strip() if isinstance(parent_id_raw, str) else ""
        if not parent_id:
            break
        parent_record = await asyncio.to_thread(chat_db.get_message_by_id, parent_id)
        if not parent_record:
            break
        if str(parent_record.get("conversation_id") or "") != str(conversation_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Continuation parent chain crossed conversation boundary.",
            )
        current_record = parent_record

    ordered_chain = list(reversed(chain_reversed))
    if history_limit > 0:
        ordered_chain = ordered_chain[-history_limit:]
    else:
        ordered_chain = []
    if history_order == "desc":
        ordered_chain = list(reversed(ordered_chain))

    metadata = {
        "applied": True,
        "mode": mode,
        "from_message_id": from_message_id,
    }
    return ordered_chain, metadata


def _is_saved_chat_error_envelope(message: dict[str, Any]) -> bool:
    """Recognize the WebUI's assistant diagnostic envelope, never user text."""
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    if (
        isinstance(content, list)
        and len(content) == 1
        and isinstance(content[0], dict)
        and content[0].get("type") == "text"
    ):
        content = content[0].get("text")
    prefix = "__tldw_error__:"
    if not isinstance(content, str) or not content.startswith(prefix):
        return False
    try:
        payload = _json.loads(content[len(prefix):])
    except (ValueError, TypeError):
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("summary"), str)
        and isinstance(payload.get("hint"), str)
    )


def _retry_content_components(content: Any) -> tuple[str, tuple[tuple[str, bytes, str], ...]]:
    """Compare representable stored text and ordered images, never partial input."""
    from tldw_Server_API.app.core.Utils.image_validation import validate_image_url

    if isinstance(content, str):
        return content, ()
    if not isinstance(content, list):
        raise HTTPException(status_code=409, detail="The failed image turn cannot be matched safely.")
    text = ""
    text_count = 0
    images = []
    for part in content:
        if not isinstance(part, dict):
            raise HTTPException(status_code=409, detail="The failed image turn cannot be matched safely.")
        if part.get("type") == "text" and set(part) <= {"type", "text"} and isinstance(part.get("text"), str):
            text_count += 1
            text = part["text"]
            if text_count > 1:
                raise HTTPException(status_code=409, detail="Retry does not support multiple text segments in one image turn.")
        elif part.get("type") == "image_url" and set(part) <= {"type", "image_url"}:
            image = part.get("image_url")
            if not isinstance(image, dict) or set(image) - {"url", "detail"} or image.get("detail", "auto") not in {"auto", "high", "low"}:
                raise HTTPException(status_code=409, detail="Retry cannot discard image detail or unsupported image options.")
            url = image.get("url")
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise HTTPException(status_code=409, detail="Retry requires a saved inline image.")
            valid, mime, data = validate_image_url(url)
            if not valid or not data or not mime:
                raise HTTPException(status_code=409, detail="A failed-turn image is invalid or unavailable.")
            images.append((mime, data, image.get("detail", "auto")))
        else:
            raise HTTPException(status_code=409, detail="Retry cannot discard unsupported message content.")
    return text, tuple(images)


def _saved_image_text(row: dict[str, Any], extra: Any) -> str:
    """Only unedited server-generated image placeholders represent empty text."""
    text = row.get("content") or ""
    images = row.get("images") or ([row] if row.get("image_data") else [])
    if (str(row.get("sender", "")).lower() == "user" and row.get("version") == 1
            and isinstance(extra, dict) and extra.get("content_placeholder_reason") == "image_attachment"
            and images and text == f"<Image attachment x{len(images)}>"):
        return ""
    return text


def _saved_image_details(extra: Any, image_count: int) -> list[str]:
    """Restore ordered image options, defaulting only legacy rows to auto."""
    details = extra.get("image_details") if isinstance(extra, dict) else None
    if details is None:
        return ["auto"] * image_count
    if (not isinstance(details, list) or len(details) != image_count
            or any(not isinstance(detail, str) or detail not in {"auto", "high", "low"} for detail in details)):
        raise HTTPException(status_code=409, detail="Saved image options are incomplete or unsupported.")
    return details


def _saved_user_content_parts(row: dict[str, Any], extra: Any) -> list[dict[str, Any]]:
    """Reconstruct the literal saved user text and every attachment for reuse."""
    text = _saved_image_text(row, extra)
    parts = [{"type": "text", "text": text}] if text else []
    images = row.get("images") or ([row] if row.get("image_data") else [])
    details = _saved_image_details(extra, len(images))
    for image, detail in zip(images, details):
        data = image.get("image_data")
        if not data:
            raise HTTPException(status_code=409, detail="A saved chat attachment is incomplete.")
        mime = image.get("image_mime_type") or "image/png"
        parts.append({"type": "image_url", "image_url": {
            "url": f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}", "detail": detail
        }})
    return parts


async def _resolve_saved_regeneration(
    chat_db: Any, conversation_id: str, reply_id: str, requested_user: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate the current answered turn, returning its user and legacy reply."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

    tail = await asyncio.to_thread(partial(
        chat_db.get_messages_for_conversation, conversation_id, 2, 0, "DESC", strict_images=True,
    ))
    reply = await asyncio.to_thread(chat_db.get_message_by_id, reply_id)
    if not reply or str(reply.get("conversation_id")) != conversation_id:
        raise HTTPException(status_code=404, detail="The reply to regenerate was not found in this conversation.")
    if (requested_user.get("role") != "user" or not tail
            or str(reply.get("sender", "")).lower() != "assistant"
            or str(tail[0].get("sender", "")).lower() != "assistant"
            or _is_saved_chat_error_envelope({"role": "assistant", "content": tail[0].get("content", "")})):
        raise HTTPException(status_code=409, detail="The answered turn changed. Reload before regenerating.")
    user_id = tail[0].get("parent_message_id")
    if not user_id and len(tail) > 1 and str(tail[1].get("sender", "")).lower() == "user":
        user_id = tail[1]["id"]
    legacy_reply = reply if not reply.get("parent_message_id") and reply["id"] == tail[0]["id"] else None
    if not user_id or (reply.get("parent_message_id") != user_id and legacy_reply is None):
        raise HTTPException(status_code=409, detail="The reply no longer belongs to the current turn. Reload before regenerating.")
    try:
        user = await asyncio.to_thread(partial(chat_db.get_message_by_id, user_id, strict_images=True))
    except CharactersRAGDBError as exc:
        raise HTTPException(status_code=409, detail="The saved chat attachments could not be read completely.") from exc
    if not user or user.get("conversation_id") != conversation_id or str(user.get("sender", "")).lower() != "user":
        raise HTTPException(status_code=409, detail="The saved user turn is unavailable. Reload before regenerating.")
    metadata = await asyncio.to_thread(chat_db.get_message_metadata, user_id)
    parts = _saved_user_content_parts(user, (metadata or {}).get("extra"))
    if _retry_content_components(requested_user.get("content")) != _retry_content_components(parts):
        raise HTTPException(status_code=409, detail="The saved user text or attachments changed. Reload before regenerating.")
    return user, legacy_reply


def _format_selected_history_messages(
    snapshot: Any,
    content: Any,
    character_card: dict[str, Any],
    *,
    user_name: str = "User",
    replace_user_placeholders: bool = True,
) -> list[dict[str, Any]]:
    """Format only detached owner-selected rows; never reload current sources."""
    from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError

    output: list[dict[str, Any]] = []
    source_nodes = {row["id"]: row for row in snapshot.nodes}
    for item in content:
        extra = item.get("extra_metadata") or {}
        source_role = map_sender_to_role(source_nodes[item["id"]]["role"], character_card.get("name"))
        role = extra.get("sender_role") or source_role
        text = _saved_image_text(
            {
                "content": item["message"],
                "images": item["images"],
                "sender": source_nodes[item["id"]]["role"],
                "version": item["_message_version"],
            },
            extra,
        )
        image_details = _saved_image_details(extra, len(item["images"]))
        if text and role != "tool" and (role != "user" or replace_user_placeholders):
            text = replace_placeholders(text, character_card.get("name", "Char"), user_name)
        placeholder = extra.get("content_placeholder_reason")
        if placeholder == "invalid_content" or (
            placeholder in {"tool_calls", "function_call"}
            and (
                role != "assistant"
                or not (item.get("tool_calls") if placeholder == "tool_calls" else extra.get("function_call"))
            )
        ):
            raise HistorySelectionError("unsupported_history_context_content_placeholder")
        if placeholder == "empty_tool_result":
            if role != "tool" or item["images"]:
                raise HistorySelectionError("unsupported_history_context_content_placeholder")
            if item["_message_version"] == 1 and text == "[empty tool result]":
                text = ""
        entry = {"role": role, "content": text}
        if item["images"]:
            entry["content"] = ([{"type": "text", "text": text}] if text else []) + [
                {"type": "image_url", "image_url": {"url": image, "detail": detail}}
                for image, detail in zip(item["images"], image_details)
            ]
        if extra.get("content_placeholder_reason") in ("tool_calls", "function_call"):
            entry["content"] = None
        if item.get("tool_calls") is not None:
            entry["tool_calls"] = item["tool_calls"]
        for key in ("function_call", "tool_call_id"):
            if extra.get(key) is not None:
                entry[key] = extra[key]
        if extra.get("sender_name") or extra.get("tool_name"):
            entry["name"] = extra.get("sender_name") or extra.get("tool_name")
        output.append(entry)
    return output


async def build_context_and_messages(
    chat_db: Any,
    request_data: Any,
    loop: Any,
    metrics: Any,
    default_save_to_db: bool,
    final_conversation_id: str | None,
    save_message_fn: Any,
    runtime_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int | None, str, bool, list[dict[str, Any]], bool]:
    """Resolve character/conversation context, load history, save current messages, and return LLM-ready payload.

    Returns (character_card, character_db_id, final_conversation_id, conversation_created, llm_payload_messages, should_persist)
    """
    request_metadata = getattr(request_data, "metadata", None)
    explicit_failed_retry = isinstance(request_metadata, dict) and request_metadata.get("tldw_retry_failed_turn") is True
    regenerate_reply_id = request_metadata.get("tldw_regenerate_from_message_id") if isinstance(request_metadata, dict) else None
    if regenerate_reply_id is not None:
        if not isinstance(regenerate_reply_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", regenerate_reply_id):
            # 400: the chat endpoint turns statuses outside its allow-list (422) into 500.
            raise HTTPException(status_code=400, detail="A valid saved reply ID is required for regeneration.")
        if (not final_conversation_id or explicit_failed_retry
                or getattr(request_data, "tldw_continuation", None) is not None):
            raise HTTPException(status_code=409, detail="Regeneration requires an existing saved answered turn.")
    from tldw_Server_API.app.core.Chat.history_continuation import (
        continued_reply_prefix,
        parse_continue_reply_id,
    )

    continue_reply_id = parse_continue_reply_id(
        request_data,
        conversation_id=final_conversation_id,
        regenerate_reply_id=regenerate_reply_id,
        explicit_failed_retry=explicit_failed_retry,
    )
    continue_prefix: str | None = None
    reuse_saved_turn = explicit_failed_retry or regenerate_reply_id is not None

    server_existing_input = getattr(request_data, "tldw_history_server_existing_input_v1", None)
    history_selection = getattr(request_data, "tldw_history_selection_v1", None) or server_existing_input
    if history_selection is not None and (explicit_failed_retry or (server_existing_input is not None and (regenerate_reply_id is not None or continue_reply_id is not None))):
        raise HTTPException(409, detail={"status": "unsupported_history_capability", "code": "unsupported_history_legacy_retry"})
    versioned = history_selection is not None
    history_runtime = runtime_state or {}
    if versioned and (not final_conversation_id or not history_runtime.get("history_owner_key")):
        raise HTTPException(409, detail={"code": "missing_authenticated_history_binding"})
    selected_content = ()
    history_snapshot = None
    admission = None
    if versioned:
        from tldw_Server_API.app.core.Chat.history_context import (
            project_history_context,
            selected_unanswered_text_question,
        )
        from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError, HistorySelectionSnapshotV1

        if getattr(request_data, "prompt_template_name", None) not in (None, "", DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name):
            raise HTTPException(409, detail={"status": "unsupported_history_capability", "code": "unsupported_history_context_prompt_template"})
        selection = history_selection.model_dump(mode="json")
        edit_target = getattr(request_data, "tldw_history_edit_v1", None)
        edit_target = edit_target.model_dump(mode="json") if edit_target is not None else None
        owner = history_runtime["history_owner_key"]
        continuation_holder: dict[str, str] = {}

        def accept_history() -> tuple[
            HistorySelectionSnapshotV1,
            tuple[dict[str, Any], ...],
            dict[str, Any] | None,
            dict[str, Any],
            tuple[dict[str, Any] | None, int | None, dict[str, Any]],
        ]:
            with chat_db.transaction() as conn:
                state = chat_db.get_roleplay_resume_state(
                    final_conversation_id,
                    conn=conn,
                    lock_for_update=True,
                    owner_client_id=history_runtime["history_owner_client_id"],
                )
                snap, content = chat_db.validate_history_selection(
                    final_conversation_id,
                    selection,
                    owner_client_id=history_runtime["history_owner_client_id"],
                    owner_key=owner,
                    conn=conn,
                    history_edit=edit_target,
                )
                question = (selected_unanswered_text_question(snap, content, selection)
                    if server_existing_input is not None else None)
                context = project_history_context(state, character_override=getattr(request_data, "character_id", None))
                dictionary_plan = context[2].pop("history_dictionary_plan", None)
                if dictionary_plan is not None:
                    from tldw_Server_API.app.core.Character_Chat.frozen_dictionary_context import (
                        transform_frozen_user_content,
                    )
                    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

                    try:
                        context[2]["history_dictionary_user_content"] = transform_frozen_user_content(
                            ([{"role": "user", "content": question["message"]}] if question is not None
                                else [message.model_dump(exclude_none=True) for message in request_data.messages]), dictionary_plan,
                        )
                    except InputError as exc:
                        raise HistorySelectionError("unsupported_history_context_dictionaries") from exc
                _format_selected_history_messages(snap, content, context[0] or {})
                reused_reply_id = regenerate_reply_id if regenerate_reply_id is not None else continue_reply_id
                unsupported_code = (
                    "unsupported_history_continuation" if continue_reply_id is not None
                    else "unsupported_history_regeneration"
                )
                if server_existing_input is not None:
                    accepted = None
                elif reused_reply_id is not None:
                    if getattr(request_data, "tldw_history_branch", None) is not True:
                        raise HistorySelectionError(unsupported_code)
                    inputs = [
                        message.model_dump(exclude_none=True)
                        for message in request_data.messages
                        if message.role != "system"
                    ]
                    if len(inputs) != 1 or inputs[0].get("role") != "user":
                        raise HistorySelectionError("invalid_regeneration_input")
                    try:
                        accepted = chat_db.reuse_selected_history_admission(
                            final_conversation_id,
                            selection,
                            reused_reply_id,
                            owner_client_id=history_runtime["history_owner_client_id"],
                            owner_key=owner,
                            conn=conn,
                        )
                    except HistorySelectionError as exc:
                        if exc.code == "unsupported_history_regeneration":
                            raise HistorySelectionError(unsupported_code) from exc
                        raise
                    original = next(item for item in content if item["id"] == accepted["input_message_id"])
                    extra = original.get("extra_metadata") or {}
                    if (
                        original.get("tool_calls") is not None
                        or extra.get("function_call") is not None
                        or extra.get("tool_call_id") is not None
                    ):
                        raise HistorySelectionError(unsupported_code)
                    text = _saved_image_text(
                        {
                            "sender": "user",
                            "version": original["_message_version"],
                            "content": original["message"],
                            "images": original["images"],
                        },
                        extra,
                    )
                    details = _saved_image_details(extra, len(original["images"]))
                    parts = ([{"type": "text", "text": text}] if text else []) + [
                        {"type": "image_url", "image_url": {"url": image, "detail": detail}}
                        for image, detail in zip(original["images"], details)
                    ]
                    if (
                        _retry_content_components(inputs[0].get("content")) != _retry_content_components(parts)
                        or any(
                            inputs[0].get(key) is not None for key in ("tool_calls", "function_call", "tool_call_id")
                        )
                        or inputs[0].get("name") not in (None, extra.get("sender_name"))
                    ):
                        raise HistorySelectionError("regeneration_input_changed")
                    if continue_reply_id is not None:
                        # Continue keeps the saved reply as the final context message.
                        continuation_holder["prefix"] = continued_reply_prefix(
                            next(item for item in content if item["id"] == continue_reply_id)
                        )
                    else:
                        content = tuple(item for item in content if item["id"] != regenerate_reply_id)
                else:
                    accepted = chat_db.append_selected_history_inputs(
                        final_conversation_id,
                        selection,
                        history_runtime["history_inputs"],
                        owner_client_id=history_runtime["history_owner_client_id"],
                        owner_key=owner,
                        conn=conn,
                        history_branch=getattr(request_data, "tldw_history_branch", None),
                        history_edit=edit_target,
                    )
                return snap, content, accepted, state["conversation"], context

        try:
            history_snapshot, selected_content, admission, existing_conversation, context = await asyncio.to_thread(accept_history)
        except HistorySelectionError as exc:
            failure = (
                "unsupported_history_capability"
                if (exc.code.startswith("unsupported_history_context")
                    or exc.code in {"unsupported_history_regeneration", "unsupported_history_continuation"})
                else "stale_selection"
            )
            raise HTTPException(409, detail={"status": failure, "code": exc.code, **exc.details}) from exc
        continue_prefix = continuation_holder.get("prefix")
        character_card, character_db_id, assistant_context = context
        for field, value in assistant_context.pop("history_sampling", {}).items():
            if field not in request_data.model_fields_set:
                setattr(request_data, field, value)
    else:
        # Assistant context
        (
            character_card,
            character_db_id,
            existing_conversation,
            assistant_context,
        ) = await _resolve_assistant_context_for_chat(
            chat_db=chat_db,
            request_data=request_data,
            loop=loop,
            conversation_id=final_conversation_id,
        )
    if character_card:
        logger.debug(
            "Loaded assistant context name={} system_prompt_summary={}",
            character_card.get("name"),
            text_summary(character_card.get("system_prompt")),
        )

    if character_card and character_db_id is not None:
        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
            metrics.track_character_access(character_id=str(request_data.character_id or "default"), cache_hit=False)

    if not character_card:
        logger.warning("No character context found; proceeding with ephemeral default context.")
        character_card = {"name": DEFAULT_CHARACTER_NAME, "system_prompt": _DEFAULT_ASSISTANT_SYSTEM_PROMPT}
        character_db_id = None
        assistant_context = {
            "assistant_kind": None,
            "assistant_id": None,
            "persona_memory_mode": assistant_context.get("persona_memory_mode"),
        }

    # Persistence decision
    requested = getattr(request_data, "save_to_db", None)
    should_persist: bool = bool(requested) if (requested is not None) else bool(default_save_to_db)

    # Conversation resolution
    client_id_from_db = getattr(chat_db, "client_id", None)
    conversation_created = False
    conv_id = final_conversation_id
    is_existing_persona_conversation = bool(
        existing_conversation
        and assistant_context.get("assistant_kind") == "persona"
        and assistant_context.get("assistant_id")
    )
    # Workspace chats can intentionally have no saved assistant identity. An
    # implicit default character supplies prompt context, not a request to fork.
    is_owned_neutral_conversation = bool(
        existing_conversation
        and client_id_from_db is not None
        and existing_conversation.get("client_id") == client_id_from_db
        and getattr(request_data, "character_id", None) is None
        and existing_conversation.get("character_id") is None
        and existing_conversation.get("assistant_kind") is None
        and existing_conversation.get("assistant_id") is None
    )
    from tldw_Server_API.app.core.Buddy.publication import current_buddy_publication

    publication = current_buddy_publication.get()
    # A Buddy acceptance already captured one exact existing conversation. Its
    # transaction fence rechecks owner/revision; default-character resolution
    # must not fork a neutral conversation into a new target.
    is_accepted_buddy_conversation = bool(
        publication is not None
        and existing_conversation
        and conv_id == publication.turn["conversation_id"]
        and str(client_id_from_db) == publication.repository.user_id
    )
    # Ensure a valid assistant identity is present before attempting persistence
    if should_persist and character_db_id is None and not (
        is_existing_persona_conversation or is_accepted_buddy_conversation or is_owned_neutral_conversation or versioned
    ):
        logger.warning(
            'Persistence requested but no compatible assistant identity is available; disabling persistence for conversation {}.',
            final_conversation_id or "<new>",
        )
        should_persist = False

    if regenerate_reply_id or continue_reply_id:
        if (not should_persist or not existing_conversation
                or existing_conversation.get("client_id") != client_id_from_db
                or not (is_existing_persona_conversation or is_accepted_buddy_conversation or is_owned_neutral_conversation or versioned
                        or existing_conversation.get("character_id") == character_db_id)):
            raise HTTPException(status_code=409, detail="Regeneration requires the existing saved assistant and conversation.")
        # This operation can only reuse its existing target; it must never fork.
    elif should_persist:
        if (is_existing_persona_conversation or is_accepted_buddy_conversation or is_owned_neutral_conversation or versioned) and conv_id:
            conversation_created = False
        else:
            conv_id, conversation_created = await get_or_create_conversation(
                chat_db,
                conv_id,
                character_db_id,
                character_card.get("name", "Chat"),
                client_id_from_db,
                loop,
            )
    else:
        if not conv_id:
            conv_id = str(_uuid.uuid4())
        conversation_created = False

    if conv_id:
        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
            metrics.track_conversation(conv_id, conversation_created)
    if not conv_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to establish conversation context.")

    continuation_spec = _extract_tldw_continuation_spec(request_data)
    continuation_metadata: dict[str, Any] | None = None
    assistant_parent_message_id: str | None = None
    assistant_prefill: str | None = None
    if continuation_spec and not versioned:
        assistant_parent_message_id, _mode, assistant_prefill = continuation_spec
        if conversation_created:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="conversation_id is required for continuation and must reference an existing conversation.",
            )

    # History loading (configurable limit/order; filter missing roles, normalize assistant names)
    requested_history_limit = getattr(request_data, "history_message_limit", None)
    if requested_history_limit is None:
        history_limit = DEFAULT_HISTORY_MESSAGE_LIMIT
    else:
        try:
            history_limit = int(requested_history_limit)
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            history_limit = DEFAULT_HISTORY_MESSAGE_LIMIT
        history_limit = max(0, min(_MAX_HISTORY_MESSAGES, history_limit))

    requested_history_order = getattr(request_data, "history_message_order", None)
    if requested_history_order:
        history_order = str(requested_history_order).strip().lower()
        if history_order not in {"asc", "desc"}:
            history_order = DEFAULT_HISTORY_MESSAGE_ORDER
    else:
        history_order = DEFAULT_HISTORY_MESSAGE_ORDER
    db_order = "ASC" if history_order == "asc" else "DESC"

    historical_msgs: list[dict[str, Any]] = []
    historical_ids: list[str | None] = []
    retry_history_content: dict[int, Any] = {}
    raw_hist: list[dict[str, Any]] = []
    if conv_id and (not conversation_created) and not versioned:
        if continuation_spec:
            from_message_id, mode, assistant_prefill = continuation_spec
            resolved_hist, continuation_metadata = await _resolve_tldw_continuation_history(
                chat_db=chat_db,
                loop=loop,
                conversation_id=conv_id,
                from_message_id=from_message_id,
                mode=mode,
                history_limit=(history_limit if history_limit > 0 else _MAX_HISTORY_MESSAGES),
                history_order=history_order,
            )
            if history_limit > 0:
                raw_hist = resolved_hist
            continuation_metadata["anchor_message_id"] = from_message_id
        elif history_limit > 0:
            raw_hist = await asyncio.to_thread(partial(chat_db.get_messages_for_conversation,
                    conv_id, history_limit, 0, db_order,
                    **({"strict_images": True} if reuse_saved_turn else {})),
            )
        for db_msg in raw_hist:
            sender_val = str(db_msg.get("sender", "") or "")
            metadata = None
            try:
                metadata = await asyncio.to_thread(chat_db.get_message_metadata, db_msg.get("id"))
            except _CHAT_NONCRITICAL_EXCEPTIONS as meta_err:
                logger.debug("Metadata lookup failed for message {}: {}", db_msg.get("id"), meta_err)

            tool_calls_meta = None
            function_call_meta = None
            content_placeholder_reason = None
            tool_call_id_meta = None
            stored_role = None
            stored_name = None
            if metadata:
                tool_calls_meta = metadata.get("tool_calls")
                extra_meta = metadata.get("extra") or {}
                if isinstance(extra_meta, dict):
                    function_call_meta = extra_meta.get("function_call")
                    content_placeholder_reason = extra_meta.get("content_placeholder_reason")
                    tool_call_id_meta = extra_meta.get("tool_call_id")
                    stored_role = extra_meta.get("sender_role")
                    stored_name = extra_meta.get("sender_name")

            role_initial = map_sender_to_role(sender_val, character_card.get("name") if character_card else None)
            role = stored_role if stored_role in {"user", "assistant", "system", "tool"} else role_initial
            if not should_persist_message_role(role):
                continue
            char_name_hist = character_card.get("name", "Char") if character_card else "Char"
            text_content = _saved_image_text(db_msg, metadata.get("extra") if isinstance(metadata, dict) else None)
            raw_user_text = text_content
            # Normal/persona Retry compares and sends the literal saved user input.
            preserve_literal_retry = reuse_saved_turn and role == "user" and (is_owned_neutral_conversation or is_existing_persona_conversation)
            if text_content and role != "tool" and not preserve_literal_retry:
                text_content = replace_placeholders(text_content, char_name_hist, "User")
            msg_parts = []
            if text_content:
                msg_parts.append({"type": "text", "text": text_content})
            raw_images = db_msg.get("images") or []
            if (not raw_images) and db_msg.get("image_data") and db_msg.get("image_mime_type"):
                raw_images = [{
                    "position": 0,
                    "image_data": db_msg.get("image_data"),
                    "image_mime_type": db_msg.get("image_mime_type"),
                }]

            image_details = _saved_image_details(metadata.get("extra") if isinstance(metadata, dict) else None, len(raw_images))
            for image_entry, image_detail in zip(raw_images, image_details):
                try:
                    img_bytes = image_entry.get("image_data")
                    if isinstance(img_bytes, memoryview):
                        img_bytes = img_bytes.tobytes()
                    if not img_bytes:
                        if reuse_saved_turn:
                            raise HTTPException(status_code=409, detail="A saved chat attachment is incomplete. Reload after repairing the source message.")
                        continue
                    img_mime = image_entry.get("image_mime_type") or db_msg.get("image_mime_type") or "image/png"
                    b64_img = await loop.run_in_executor(None, base64.b64encode, img_bytes)
                    image_part = {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_mime};base64,{b64_img.decode('utf-8')}", "detail": image_detail}
                    }
                    if reuse_saved_turn:
                        _retry_content_components([image_part])
                    msg_parts.append(image_part)
                except _CHAT_NONCRITICAL_EXCEPTIONS as e:
                    if reuse_saved_turn:
                        raise HTTPException(status_code=409, detail="A saved chat attachment could not be read completely.") from e
                    logger.warning(
                        "Error encoding DB image for history msg_id={} error={}",
                        db_msg.get("id"),
                        exception_summary(e),
                    )
            if msg_parts:
                hist_entry = {"role": role}
                if role == "tool" and len(msg_parts) == 1 and msg_parts[0].get("type") == "text":
                    hist_entry["content"] = msg_parts[0].get("text", "")
                else:
                    hist_entry["content"] = msg_parts
                if role == "assistant":
                    name_source = stored_name or (character_card.get("name") if character_card else None)
                    if name_source:
                        name = sanitize_sender_name(name_source)
                        if name:
                            hist_entry["name"] = name

                if tool_calls_meta is not None:
                    hist_entry["tool_calls"] = tool_calls_meta
                if function_call_meta and not hist_entry.get("tool_calls"):
                    hist_entry["function_call"] = function_call_meta
                if content_placeholder_reason in {"tool_calls", "function_call"} or (
                    content_placeholder_reason == "empty_tool_result"
                    and role == "tool"
                    and db_msg.get("version") == 1
                    and db_msg.get("content") == "[empty tool result]"
                ):
                    hist_entry["content"] = ""
                if role == "tool" and tool_call_id_meta:
                    hist_entry["tool_call_id"] = tool_call_id_meta
                if role == "tool" and not tool_call_id_meta:
                    logger.debug(
                        "Skipping tool message {} without tool_call_id metadata.",
                        db_msg.get("id"),
                    )
                    continue
                if reuse_saved_turn and role == "user":
                    retry_history_content[id(hist_entry)] = ([{"type": "text", "text": raw_user_text}] if raw_user_text else []) + [part for part in msg_parts if part.get("type") == "image_url"]
                historical_msgs.append(hist_entry)
                historical_ids.append(db_msg.get("id"))
        logger.info(f"Loaded {len(historical_msgs)} historical messages for conv_id '{conv_id}'.")

    # Keep failure diagnostics in saved history, but never send them as model
    # replies. Only a matching retry of the trailing failure can reuse its user
    # turn; ordinary repeated user messages retain their existing semantics.
    history_ended_with_error = bool(
        historical_msgs
        and _is_saved_chat_error_envelope(historical_msgs[0 if history_order == "desc" else -1])
    )
    visible_history = [(message, message_id) for message, message_id in zip(historical_msgs, historical_ids)
                       if not _is_saved_chat_error_envelope(message)]
    historical_msgs = [message for message, _ in visible_history]
    historical_ids = [message_id for _, message_id in visible_history]

    # Process current turn messages (persist if needed)
    request_messages: list[dict[str, Any]] = []
    for msg_model in request_data.messages:
        if not should_persist_message_role(msg_model.role):
            continue
        msg_dict = msg_model.model_dump(exclude_none=True)
        raw_name = msg_dict.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            sanitized = sanitize_sender_name(raw_name)
            if sanitized:
                msg_dict["name"] = sanitized
            else:
                msg_dict.pop("name", None)
        request_messages.append(msg_dict)

    metadata = getattr(request_data, "metadata", None)
    client_message_id = metadata.get("tldw_client_message_id") if isinstance(metadata, dict) else None
    if not isinstance(client_message_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", client_message_id):
        client_message_id = None
    explicit_failed_retry = isinstance(metadata, dict) and metadata.get("tldw_retry_failed_turn") is True
    retry_user_message_id: str | None = None
    saved_retry_system_messages: list[dict[str, Any]] = []
    legacy_regenerated_reply: dict[str, Any] | None = None
    if regenerate_reply_id and not versioned:
        if not request_messages:
            raise HTTPException(status_code=409, detail="Regeneration requires the saved user turn.")
        if any(message.get("role") != "system" for message in request_messages[:-1]):
            raise HTTPException(status_code=409, detail="Regeneration uses saved history. Send only the original user turn and current system instructions.")
        # System instructions still participate in prompt construction through
        # request_data. Neither they nor a supplied history prefix are new turns.
        request_messages = request_messages[-1:]
        regenerated_user, legacy_regenerated_reply = await _resolve_saved_regeneration(
            chat_db, conv_id, regenerate_reply_id, request_messages[-1],
        )
        retry_user_message_id = str(regenerated_user["id"])
        # Prior variants are alternatives to the answer being generated, not
        # preceding conversation context. Retain earlier turns unchanged.
        variant_ids = {row["id"] for row in raw_hist if
                       row.get("parent_message_id") == retry_user_message_id
                       or row["id"] == regenerate_reply_id}
        retained = [(message, message_id) for message, message_id in zip(historical_msgs, historical_ids)
                    if message_id not in variant_ids]
        historical_msgs = [message for message, _ in retained]
        historical_ids = [message_id for _, message_id in retained]
    if explicit_failed_retry and should_persist and not conversation_created and request_messages:
        # Read the actual tail independently of the requested context window.
        # Retry is an explicit operation; equal text on an ordinary send is new.
        tail_rows = await asyncio.to_thread(partial(chat_db.get_messages_for_conversation, conv_id, 2, 0, "DESC", strict_images=True))
        if (any(message.get("role") == "system" for message in request_messages)
                or any(str(row.get("sender", "")).lower() == "system" for row in tail_rows)):
            tail_ids, saved_retry_system_messages = await asyncio.to_thread(chat_db.get_retry_message_context, conv_id)
            tail_rows = [await asyncio.to_thread(partial(chat_db.get_message_by_id, message_id, strict_images=True))
                         for message_id in tail_ids]
            if any(row is None for row in tail_rows):
                raise HTTPException(status_code=409, detail="The failed turn changed. Reload before retrying.")
        tail_metadata = {row["id"]: await asyncio.to_thread(chat_db.get_message_metadata, row["id"]) for row in tail_rows}
        requested_user = request_messages[-1]
        if requested_user.get("role") != "user":
            raise HTTPException(status_code=409, detail="The failed turn changed. Reload the conversation before retrying.")

        def matches_saved_user(row: dict[str, Any]) -> bool:
            if str(row.get("sender", "")).lower() != "user":
                return False
            stored_metadata = tail_metadata.get(row["id"]) or {}
            parts = _saved_user_content_parts(row, stored_metadata.get("extra"))
            return _retry_content_components(requested_user.get("content")) == _retry_content_components(parts)

        if tail_rows:
            tail = tail_rows[0]
            if str(tail.get("sender", "")).lower() == "user":
                if not matches_saved_user(tail):
                    raise HTTPException(status_code=409, detail="The unanswered turn changed. Reload the conversation before retrying.")
                retry_user_message_id = str(tail["id"])
            elif len(tail_rows) > 1 and matches_saved_user(tail_rows[1]):
                if _is_saved_chat_error_envelope({"role": "assistant", "content": tail.get("content", "")}):
                    retry_user_message_id = str(tail_rows[1]["id"])
                else:
                    saved_extra = (tail_metadata.get(tail_rows[1]["id"]) or {}).get("extra")
                    saved_client_id = saved_extra.get("client_message_id") if isinstance(saved_extra, dict) else None
                    saved_identity_is_valid = (
                        isinstance(saved_client_id, str)
                        and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", saved_client_id)
                    )
                    if not (client_message_id and saved_identity_is_valid and saved_client_id != client_message_id):
                        raise HTTPException(status_code=409, detail="This turn already has an answer. Reload the conversation before retrying.")

    retry_system_messages: list[dict[str, Any]] = []
    if explicit_failed_retry and retry_user_message_id:
        # Instructions are the requested current block, even when the client
        # also supplies overlapping history. A matched user suffix must not
        # trim a deliberately changed system prefix from persistence.
        retry_system_messages = [message for message in request_messages if message.get("role") == "system"]
        request_messages = [message for message in request_messages if message.get("role") != "system"]

    # If the client included history with conversation_id, trim overlaps against DB history
    overlap_cut = 0
    if conv_id and historical_msgs and request_messages and not versioned:
        has_non_user_role = any(
            msg.get("role") in {"assistant", "tool"} for msg in request_messages
        )
        all_user_roles = all(msg.get("role") == "user" for msg in request_messages)
        def _normalize_content(value: Any) -> Any:
            if isinstance(value, str):
                return [{"type": "text", "text": value}]
            if isinstance(value, list):
                return value
            return value

        def _msg_sig(msg: dict[str, Any]) -> str:
            payload = {
                "role": msg.get("role"),
                "content": (_retry_content_components(retry_history_content.get(id(msg), msg.get("content"))) if reuse_saved_turn and msg.get("role") == "user" else _normalize_content(msg.get("content"))),
                "tool_calls": msg.get("tool_calls"),
                "function_call": msg.get("function_call"),
                "tool_call_id": msg.get("tool_call_id"),
            }
            try:
                return _json.dumps(payload, sort_keys=True, default=str)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                return str(payload)

        if has_non_user_role:
            # Overlap detection assumes chronological ordering for comparison.
            # If history is requested newest-first, reverse a copy for overlap checks
            # while preserving the requested order in the final payload.
            hist_for_overlap = (
                list(reversed(historical_msgs)) if history_order == "desc" else historical_msgs
            )
            hist_sigs = [_msg_sig(m) for m in hist_for_overlap]
            req_sigs = [_msg_sig(m) for m in request_messages]
            # An ordinary final user is a new turn, even when its text equals
            # an interrupted saved user. Only explicit Retry may reuse it.
            if request_messages[-1].get("role") == "user" and not retry_user_message_id:
                req_sigs = req_sigs[:-1]
            max_k = min(len(hist_sigs), len(req_sigs))
            overlap_start = None
            overlap_k = 0
            for k in range(max_k, 0, -1):
                tail = hist_sigs[-k:]
                for i in range(0, len(req_sigs) - k + 1):
                    if req_sigs[i:i + k] == tail:
                        overlap_start = i
                        overlap_k = k
                        break
                if overlap_k:
                    break
            if not overlap_k:
                # A restored client can lag behind an interrupted canonical
                # turn. Trim only its exact historical PREFIX, including an
                # assistant, proven within the loaded history window. Later
                # canonical rows remain model context; do not replay them.
                for k in range(max_k, 0, -1):
                    if not any(message.get("role") == "assistant" for message in request_messages[:k]):
                        continue
                    if any(hist_sigs[i:i + k] == req_sigs[:k] for i in range(len(hist_sigs) - k + 1)):
                        overlap_start = 0
                        overlap_k = k
                        break
            if overlap_k and overlap_start is not None:
                overlap_cut = overlap_start + overlap_k
                if overlap_cut > 0:
                    logger.debug(
                        'Trimmed {} request messages overlapping history for conv_id {}.',
                        overlap_cut,
                        conv_id,
                    )
        else:
            try:
                user_only_trim = str(os.getenv("CHAT_TRIM_USER_ONLY_OVERLAP", "false")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                user_only_trim = False
            retrying_failed_turn = (history_ended_with_error or retry_user_message_id is not None) and len(request_messages) == 1
            if (user_only_trim or retrying_failed_turn) and all_user_roles:
                hist_for_overlap = (
                    list(reversed(historical_msgs)) if history_order == "desc" else historical_msgs
                )
                hist_sigs = [_msg_sig(m) for m in hist_for_overlap]
                req_sigs = [_msg_sig(m) for m in request_messages]
                if req_sigs and len(req_sigs) <= len(hist_sigs):
                    if hist_sigs[-len(req_sigs):] == req_sigs:
                        overlap_cut = len(req_sigs)
                        logger.debug(
                            'Trimmed {} user-only request messages overlapping history for conv_id {}.',
                            overlap_cut,
                            conv_id,
                        )

    if versioned:
        if regenerate_reply_id is not None or continue_reply_id is not None:
            overlap_cut = len(request_messages)
        formatted_history = _format_selected_history_messages(history_snapshot, selected_content, character_card)
        if server_existing_input is not None:
            existing_dictionary_content = assistant_context.pop("history_dictionary_user_content", [])
            if existing_dictionary_content:
                formatted_history[-1]["content"] = existing_dictionary_content[0]
        historical_msgs.extend(formatted_history)
        assistant_parent_message_id = (selected_content[-1]["id"] if server_existing_input is not None
            else admission["input_message_id"])
        runtime_state["assistant_parent_message_id"] = assistant_parent_message_id
        if server_existing_input is not None:
            runtime_state["tldw_history_server_existing_input_v1"] = history_selection.model_dump(mode="json")
        else:
            runtime_state["tldw_history_admission_v1"] = admission
        if continue_prefix is not None:
            runtime_state["history_continue_prefix"] = continue_prefix

    if retry_user_message_id and any(message.get("role") == "user" for message in request_messages[overlap_cut:-1]):
        raise HTTPException(status_code=409, detail="This retry contains extra unsaved user messages. Resolve the local conversation before retrying.")
    if retry_user_message_id and client_message_id:
        saved_metadata = await asyncio.to_thread(chat_db.get_message_metadata, retry_user_message_id)
        saved_extra = (saved_metadata or {}).get("extra")
        saved_client_id = saved_extra.get("client_message_id") if isinstance(saved_extra, dict) else None
        if saved_client_id and saved_client_id != client_message_id:
            raise HTTPException(status_code=409, detail="The failed turn identity changed. Reload the conversation before retrying.")

    if legacy_regenerated_reply is not None:
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError

        try:
            await asyncio.to_thread(
                chat_db.update_message, legacy_regenerated_reply["id"],
                {"parent_message_id": retry_user_message_id}, legacy_regenerated_reply["version"],
            )
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail="The saved reply changed. Reload before regenerating.") from exc

    persisted_user_message_id: str | None = assistant_parent_message_id if versioned else retry_user_message_id
    system_block_id = str(_uuid.uuid4())
    if (should_persist and retry_system_messages
            and _extract_system_messages(retry_system_messages) != _extract_system_messages(saved_retry_system_messages)):
        import sqlite3

        from tldw_Server_API.app.core.Buddy.publication import current_buddy_publication
        from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

        publication = current_buddy_publication.get()

        def persist_retry_instructions() -> None:
            """Keep publication receipts and the complete block in one transaction."""
            with chat_db.transaction() as conn:
                if publication is not None:
                    publication.repository.assert_publication(conn, publication.turn, conv_id)
                message_ids = chat_db.add_retry_system_instruction_block(
                    conv_id, retry_system_messages, system_block_id,
                )
                if publication is not None:
                    for message_id in message_ids:
                        publication.repository.record_message(conn, publication.turn["id"], "system", message_id)

        try:
            await asyncio.to_thread(persist_retry_instructions)
        except (CharactersRAGDBError, sqlite3.Error, BackendDatabaseError) as exc:
            raise HTTPException(
                status_code=503,
                detail="Unable to save updated retry instructions. Try again.",
            ) from exc
    current_turn: list[dict[str, Any]] = list(retry_system_messages)
    dictionary_user_content = iter(assistant_context.pop("history_dictionary_user_content", []))
    for index, msg_dict in enumerate(request_messages[overlap_cut:], start=overlap_cut):
        role = msg_dict.get("role")
        msg_for_db = msg_dict.copy()
        if role == "system":
            msg_for_db["system_instruction_block_id"] = system_block_id
        if role == "assistant" and character_card:
            # Persist assistant sender as sanitized character name
            msg_for_db["name"] = sanitize_sender_name(character_card.get("name", "Assistant"))
        reused_retry = retry_user_message_id is not None and index == len(request_messages) - 1
        if should_persist and not reused_retry and not versioned:
            if role == "user" and index == len(request_messages) - 1 and client_message_id:
                msg_for_db["client_message_id"] = client_message_id
            saved_message_id = await save_message_fn(chat_db, conv_id, msg_for_db, use_transaction=True)
            if role == "user":
                persisted_user_message_id = str(saved_message_id) if saved_message_id else None
        msg_for_llm = msg_dict.copy()
        if versioned and role == "user":
            msg_for_llm["content"] = next(dictionary_user_content, msg_for_llm.get("content"))
        if role == "assistant" and character_card and character_card.get("name"):
            name = sanitize_sender_name(character_card.get("name"))
            if name:
                msg_for_llm["name"] = name
        if not reused_retry or retry_user_message_id not in historical_ids:
            current_turn.append(msg_for_llm)

    if retry_user_message_id is not None and retry_user_message_id in historical_ids:
        # Overlap and identity checks above have accepted this exact saved turn.
        # Keep its validated content, but make it current even with DESC history.
        retry_index = historical_ids.index(retry_user_message_id)
        current_turn.append(historical_msgs.pop(retry_index))

    if continuation_spec and assistant_prefill and not versioned:
        prefill_payload: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_prefill,
        }
        if character_card and character_card.get("name"):
            prefill_name = sanitize_sender_name(character_card.get("name"))
            if prefill_name:
                prefill_payload["name"] = prefill_name
        current_turn.append(prefill_payload)
        if continuation_metadata is not None:
            continuation_metadata["assistant_prefill"] = assistant_prefill
            continuation_metadata["assistant_prefill_applied"] = True

    if runtime_state is not None:
        runtime_state["user_message_id"] = persisted_user_message_id
        if persisted_user_message_id:
            runtime_state["assistant_parent_message_id"] = persisted_user_message_id
        runtime_state["assistant_context"] = dict(assistant_context)
        if continuation_spec and continuation_metadata is not None:
            runtime_state["tldw_continuation"] = continuation_metadata
            if assistant_parent_message_id:
                runtime_state["assistant_parent_message_id"] = assistant_parent_message_id

    # Preserve requested history ordering (DB fetch already honors ASC/DESC).
    historical_msgs_for_payload = historical_msgs

    llm_payload_messages = historical_msgs_for_payload + current_turn
    if versioned and assistant_context.get("history_world_book_plan") is not None:
        from tldw_Server_API.app.core.Character_Chat.world_book_prompt_context import (
            build_frozen_world_book_prompt_context,
        )

        character_card["_history_world_book_context"] = build_frozen_world_book_prompt_context(
            llm_payload_messages, assistant_context["history_world_book_plan"],
        )

    return character_card, character_db_id, conv_id, conversation_created, llm_payload_messages, should_persist


def _extract_system_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Extract system message text content from OpenAI-style message payloads."""
    system_messages: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text_val = content.strip()
            if text_val:
                system_messages.append(text_val)
            continue
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_val = part.get("text")
                    if isinstance(text_val, str) and text_val.strip():
                        text_parts.append(text_val)
            combined = "\n".join(text_parts).strip()
            if combined:
                system_messages.append(combined)
    return system_messages


def _extract_system_messages_from_request(messages: list[Any]) -> list[str]:
    """Extract system messages from request model objects."""
    system_messages: list[str] = []
    for msg in messages or []:
        try:
            if getattr(msg, "role", None) != "system":
                continue
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                text_val = content.strip()
                if text_val:
                    system_messages.append(text_val)
                continue
            if isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    try_type = getattr(part, "type", None)
                    if try_type is None and isinstance(part, dict):
                        try_type = part.get("type")
                    if try_type == "text":
                        text_val = getattr(part, "text", None) if not isinstance(part, dict) else part.get("text")
                        if isinstance(text_val, str) and text_val.strip():
                            text_parts.append(text_val)
                combined = "\n".join(text_parts).strip()
                if combined:
                    system_messages.append(combined)
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            continue
    return system_messages


def apply_prompt_templating(
    request_data: Any,
    character_card: dict[str, Any],
    llm_payload_messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Compute final system message and apply content templating to payload messages.

    Returns (final_system_message, templated_llm_payload)
    """
    active_template = (DEFAULT_RAW_PASSTHROUGH_TEMPLATE if (getattr(request_data, "tldw_history_selection_v1", None) is not None
            or getattr(request_data, "tldw_history_server_existing_input_v1", None) is not None)
        else load_template(getattr(request_data, "prompt_template_name", None) or DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name))
    template_data: dict[str, Any] = {}
    if character_card:
        template_data.update({k: v for k, v in character_card.items() if isinstance(v, (str, int, float))})
        template_data["char_name"] = character_card.get("name", "Character")
        template_data["character_system_prompt"] = character_card.get("system_prompt", "")

    system_msgs_from_request = _extract_system_messages_from_request(getattr(request_data, "messages", []))
    sys_msg_from_req = system_msgs_from_request[0] if system_msgs_from_request else None
    template_data["original_system_message_from_request"] = sys_msg_from_req or ""
    template_data["system_messages_from_request"] = "\n\n".join(system_msgs_from_request) if system_msgs_from_request else ""
    system_msgs_from_payload = _extract_system_messages(llm_payload_messages)
    payload_system_message = system_msgs_from_payload[-1] if system_msgs_from_payload else None
    template_data["system_messages_from_payload"] = "\n\n".join(system_msgs_from_payload) if system_msgs_from_payload else ""
    template_data["system_messages_combined"] = (
        template_data["system_messages_from_request"]
        or (payload_system_message or "")
    )

    final_system_message: str | None = None
    logger.debug(
        "Prompt templating inputs {}",
        prompt_template_summary(
            template_name=getattr(active_template, "name", None),
            system_message=sys_msg_from_req,
            payload_system_messages=system_msgs_from_payload,
            request_system_messages=system_msgs_from_request,
            character_name=character_card.get("name") if character_card else None,
        ),
    )
    if active_template and active_template.system_message_template:
        final_system_message = apply_template_to_string(active_template.system_message_template, template_data)
        if not final_system_message and payload_system_message:
            final_system_message = payload_system_message
            logger.debug("Template empty, using payload system message summary {}", text_summary(final_system_message))
        if not final_system_message and character_card and character_card.get("system_prompt"):
            final_system_message = character_card.get("system_prompt")
            logger.debug("Template empty, using character system prompt summary {}", text_summary(final_system_message))
    elif sys_msg_from_req:
        final_system_message = sys_msg_from_req
    elif payload_system_message:
        final_system_message = payload_system_message
    elif character_card and character_card.get("system_prompt"):
        final_system_message = character_card.get("system_prompt")
        logger.debug("Using character system prompt summary {}", text_summary(final_system_message))

    if (getattr(request_data, "tldw_history_selection_v1", None) is not None
            or getattr(request_data, "tldw_history_server_existing_input_v1", None) is not None):
        lore = character_card.get("_history_world_book_context")
        if lore is not None and lore.text:
            final_system_message = "\n\n".join(part for part in (final_system_message, lore.text) if part)

    logger.debug("Final system message summary {}", text_summary(final_system_message))

    if final_system_message:
        llm_payload_messages = [m for m in llm_payload_messages if m.get("role") != "system"]

    templated_llm_payload: list[dict[str, Any]] = []
    if active_template:
        for msg in llm_payload_messages:
            templated_msg_content = msg.get("content")
            role = msg.get("role")
            content_template_str = None
            if role == "user" and active_template.user_message_content_template:
                content_template_str = active_template.user_message_content_template
            elif role == "assistant" and active_template.assistant_message_content_template:
                content_template_str = active_template.assistant_message_content_template
            if content_template_str:
                new_content_parts = []
                msg_template_data = template_data.copy()
                if isinstance(templated_msg_content, str):
                    msg_template_data["message_content"] = templated_msg_content
                    new_text = apply_template_to_string(content_template_str, msg_template_data)
                    new_content_parts.append({"type": "text", "text": new_text or templated_msg_content})
                elif isinstance(templated_msg_content, list):
                    for part in templated_msg_content:
                        if part.get("type") == "text":
                            msg_template_data["message_content"] = part.get("text", "")
                            new_text_part = apply_template_to_string(content_template_str, msg_template_data)
                            new_content_parts.append({"type": "text", "text": new_text_part or part.get("text", "")})
                        else:
                            new_content_parts.append(part)
                templated_llm_payload.append({**msg, "content": new_content_parts or templated_msg_content})
            else:
                templated_llm_payload.append(msg)
    else:
        templated_llm_payload = llm_payload_messages

    return final_system_message, templated_llm_payload


async def _maybe_refund_streaming_rg(
    rg_refund_cb: Callable[..., Any] | None,
    *,
    cancelled: bool = False,
    error: bool = True,
) -> None:
    """Best-effort RG refund hook for streaming failures before stream setup."""
    if not callable(rg_refund_cb):
        return
    try:
        res = rg_refund_cb(cancelled=cancelled, error=error)
        if hasattr(res, "__await__"):
            await res  # type: ignore[misc]
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        pass


def _evaluate_chat_prompt_cost_guardrails(
    *,
    cleaned_args: Mapping[str, Any],
    templated_llm_payload: list[dict[str, Any]],
    selected_provider: str,
    model: str,
    streaming: bool,
    continuation_metadata: Mapping[str, Any] | None = None,
) -> PromptCostGuardrailDecision | None:
    """Evaluate prompt-cost guardrails and fail open on diagnostic errors."""
    try:
        config = load_prompt_cost_guardrail_config()
        if not config.enabled:
            return None
        envelope_messages = list(templated_llm_payload)
        system_message = cleaned_args.get("system_message")
        if system_message is not None and (
            not isinstance(system_message, str) or system_message.strip()
        ):
            envelope_messages = [
                {"role": "system", "content": system_message},
                *envelope_messages,
            ]
        envelope = build_prompt_cost_envelope(envelope_messages)
        decision = evaluate_prompt_cost_guardrails(
            envelope,
            request_options=cleaned_args,
            previous_fingerprints=_extract_previous_prompt_guardrail_fingerprints(
                cleaned_args=cleaned_args,
                continuation_metadata=continuation_metadata,
            ),
            config=config,
        )
        if decision.action == "allow":
            return None

        metadata = decision.to_response_metadata()
        log_metadata = {
            "provider": selected_provider,
            "model": model,
            "streaming": streaming,
            "decision": metadata,
        }
        if decision.action == "block":
            logger.warning("Chat prompt cost guardrail blocked request: {}", log_metadata)
            guardrail_error = HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "code": "prompt_cost_guardrail_block",
                    "type": "prompt_cost_guardrail_block",
                    "message": "Prompt cost guardrail blocked request before provider dispatch.",
                    "prompt_guardrails": metadata,
                },
            )
            raise _mark_trusted_local_chat_signal(
                guardrail_error,
                "prompt_cost_guardrail",
            )
        logger.warning("Chat prompt cost guardrail warnings: {}", log_metadata)
        return decision
    except HTTPException:
        raise
    except _CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Prompt cost guardrail evaluation skipped due to {}",
            type(exc).__name__,
        )
        return None


def _extract_previous_prompt_guardrail_fingerprints(
    *,
    cleaned_args: Mapping[str, Any],
    continuation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str] | None:
    candidates = (
        cleaned_args.get("prompt_guardrail_previous_fingerprints"),
        cleaned_args.get("previous_prompt_fingerprints"),
        cleaned_args.get("tldw_previous_prompt_fingerprints"),
        (
            continuation_metadata.get("prompt_guardrail_fingerprints")
            if isinstance(continuation_metadata, Mapping)
            else None
        ),
    )
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        sanitized = {
            str(key): str(value)
            for key, value in candidate.items()
            if isinstance(key, str) and isinstance(value, str) and value
        }
        if sanitized:
            return sanitized
    return None


async def execute_streaming_call(
    *,
    current_loop: Any,
    cleaned_args: dict[str, Any],
    selected_provider: str,
    provider: str,
    model: str,
    request_json: str,
    request: Any,
    metrics: Any,
    provider_manager: Any,
    templated_llm_payload: list[dict[str, Any]],
    should_persist: bool,
    final_conversation_id: str,
    character_card_for_context: dict[str, Any] | None,
    chat_db: Any,
    save_message_fn: Callable[..., Any],
    system_message_id: str | None = None,
    user_message_id: str | None = None,
    audit_service: Any | None,
    audit_context: Any | None,
    client_id: str,
    queue_execution_enabled: bool,
    enable_provider_fallback: bool,
    llm_call_func: Callable[[], Any],
    refresh_provider_params: Callable[[str], Any],
    structured_request_context: StructuredResponseRequestContext | None = None,
    moderation_getter: Callable[[], Any] | None = None,
    rg_commit_cb: Callable[[int], Any] | None = None,
    rg_refund_cb: Callable[..., Any] | None = None,
    on_success: Callable[[str], Awaitable[bool | None]] | None = None,
    on_provider_output: Callable[[str], Awaitable[bool | None]] | None = None,
    self_monitoring_service: Any | None = None,
    on_stream_full_reply: Callable[[str], Awaitable[None] | None] | None = None,
    assistant_parent_message_id: str | None = None,
    continuation_metadata: dict[str, Any] | None = None,
    queue_request_id: str | None = None,
    provider_factory_timeout: float | None = None,
    history_persistence_ack: bool = False,
    forward_usage_chunk: bool = False,
) -> StreamingResponse:
    """Execute a streaming LLM call with queue, failover, moderation, and persistence.

    Returns a StreamingResponse that yields SSE chunks and handles:
    - provider call invocation and fallback
    - output moderation (chunk-wise)
    - saving final assistant message to DB
    - usage logging and audit success
    """
    llm_start_time = time.time()
    usage_credential_source = call_credential_source(cleaned_args)  # whose key the reply ran on, for the usage row
    factory_budget_seconds = (
        PROVIDER_STREAM_FACTORY_TIMEOUT_SECONDS
        if provider_factory_timeout is None
        else max(0.0, float(provider_factory_timeout))
    )
    provider_factory_deadline: float | None = None

    def _remaining_provider_factory_timeout() -> float:
        nonlocal provider_factory_deadline
        now = _provider_factory_monotonic()
        if provider_factory_deadline is None:
            provider_factory_deadline = now + factory_budget_seconds
        return max(0.0, provider_factory_deadline - now)

    normalized_continuation_metadata = (
        dict(continuation_metadata)
        if isinstance(continuation_metadata, dict) and continuation_metadata
        else None
    )
    run_first_metric_context = _emit_chat_run_first_rollout_metrics(
        metrics,
        cleaned_args=cleaned_args,
        provider=selected_provider,
        model=model,
        streaming=True,
    )
    stream_metrics_recorded = False
    stream_failure_recorded = False
    raw_stream_iter: AsyncIterator[str] | Iterator[str] | None = None
    queue_for_exec = None
    queue_future: asyncio.Future[Any] | None = None
    queue_enabled = False

    def _public_stream_error_payload(value: Any) -> dict[str, Any]:
        if isinstance(value, UnsupportedMultiChoiceToolAutoexecError):
            payload: dict[str, Any] = {
                "error": {
                    "code": value.error_code,
                    "type": value.error_code,
                    "message": value.public_message,
                }
            }
        else:
            payload = provider_stream_error_payload(value)
        if CHAT_STREAM_INCLUDE_METADATA and final_conversation_id:
            payload["conversation_id"] = final_conversation_id
            payload["tldw_conversation_id"] = final_conversation_id
            if user_message_id:
                payload["tldw_user_message_id"] = user_message_id
            if system_message_id:
                payload["tldw_system_message_id"] = system_message_id
            if normalized_continuation_metadata:
                payload["tldw_continuation"] = normalized_continuation_metadata
        return payload

    async def _terminal_stream_error(payload: dict[str, Any]) -> AsyncIterator[str]:
        yield sse_data(payload)
        yield sse_done()

    try:
        try:
            if structured_request_context is None:
                structured_request_context = prepare_structured_response_request(
                    provider=selected_provider,
                    response_format=cleaned_args.get("response_format"),
                    requested_response_format=cleaned_args.get("_structured_requested_response_format"),
                )
            apply_structured_response_request(
                cleaned_args=cleaned_args,
                structured_request_context=structured_request_context,
            )
            ensure_tool_autoexec_supports_request(
                cleaned_args=cleaned_args,
                should_run_tool_autoexec=should_run_legacy_tool_autoexec,
            )
            _evaluate_chat_prompt_cost_guardrails(
                cleaned_args=cleaned_args,
                templated_llm_payload=templated_llm_payload,
                selected_provider=selected_provider,
                model=model,
                streaming=True,
                continuation_metadata=normalized_continuation_metadata,
            )
        except (
            StructuredGenerationCapabilityError,
            StructuredGenerationParseError,
            StructuredGenerationSchemaError,
        ) as structured_exc:
            raise_detached_error(build_structured_http_exception(structured_exc))
        try:
            queue_for_exec = get_request_queue()
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            queue_for_exec = None
        queue_enabled = (
            queue_execution_enabled
            and queue_for_exec is not None
            and queue_is_active(queue_for_exec)
        )

        if queue_enabled:
            # Submit streaming job to the queue and bridge chunks via channel
            # Bounded per-request channel for queued streaming; size is configurable via config.txt
            try:
                _maxsz_raw = os.getenv("CHAT_STREAM_CHANNEL_MAXSIZE")
                if not _maxsz_raw:
                    from tldw_Server_API.app.core.config import load_comprehensive_config
                    _cp = load_comprehensive_config()
                    _maxsz_raw = _cp.get('Chat-Module', 'chat_stream_channel_maxsize', fallback='100') if _cp else '100'
                stream_channel_maxsize = int(str(_maxsz_raw))
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                stream_channel_maxsize = 100
            stream_channel: asyncio.Queue = QueueStreamChannel(
                maxsize=stream_channel_maxsize
            )
            est_tokens_for_queue = estimate_tokens_from_json(request_json)

            def _refresh_params_sync(target_provider: str) -> tuple[dict[str, Any], str | None]:
                """Resolve provider params inside a sync queue processor."""
                return _refresh_provider_params_bounded_sync(
                    refresh_provider_params,
                    target_provider,
                    _remaining_provider_factory_timeout,
                    current_loop,
                )

            def _queued_processor():
                nonlocal selected_provider, model, llm_call_func, stream_failure_recorded, structured_request_context
                nonlocal usage_credential_source
                local_start = time.time()
                try:
                    result = _call_stream_factory_from_isolated_worker(
                        llm_call_func,
                        _remaining_provider_factory_timeout,
                    )
                    if not hasattr(result, "__aiter__") and hasattr(result, "__iter__"):
                        result = wrap_sync_stream(result)
                    if selected_provider != provider:
                        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                            metrics.track_provider_fallback_success(
                                requested_provider=provider,
                                selected_provider=selected_provider,
                                streaming=True,
                                queued=True,
                            )
                    return result
                except (LLMQuotaExceededError, LLMPolicyRefusal):
                    raise  # the user's limit or a policy refusal, not the provider's health: no failure, no fallback
                except Exception as proc_error:  # noqa: BLE001 - adapter trust boundary
                    latency = time.time() - local_start
                    metrics.track_llm_call(
                        selected_provider,
                        model,
                        latency,
                        success=False,
                        error_type=type(proc_error).__name__,
                    )
                    stream_failure_recorded = True
                    if provider_manager:
                        provider_manager.record_failure(
                            selected_provider,
                            sanitized_provider_stream_exception(proc_error),
                        )
                        if (
                            enable_provider_fallback
                            and isinstance(proc_error, (ChatProviderError, ChatAPIError))
                            and not _is_client_like_error(proc_error)
                            and provider_stream_error_allows_replay(proc_error)
                        ):
                            fallback_provider = provider_manager.get_available_provider(exclude=[selected_provider])
                            if fallback_provider:
                                logger.warning(
                                    f"Trying fallback provider {fallback_provider} after {selected_provider} failed (queued)"
                                )
                                try:
                                    refreshed_args, refreshed_model = _refresh_params_sync(fallback_provider)
                                    if not isinstance(refreshed_args, dict) or "messages_payload" not in refreshed_args:
                                        raise ValueError(
                                            f"Invalid refreshed params for {fallback_provider}: missing required fields"
                                        )
                                    try:
                                        structured_request_context = prepare_structured_response_request(
                                            provider=fallback_provider,
                                            response_format=(
                                                refreshed_args.get("response_format")
                                            ),
                                            requested_response_format=(
                                                structured_request_context.requested_response_format
                                                if structured_request_context is not None
                                                else refreshed_args.get("_structured_requested_response_format")
                                            ),
                                        )
                                    except (
                                        StructuredGenerationCapabilityError,
                                        StructuredGenerationParseError,
                                        StructuredGenerationSchemaError,
                                    ) as structured_exc:
                                        raise_detached_error(
                                            build_structured_http_exception(structured_exc)
                                        )
                                    apply_structured_response_request(
                                        cleaned_args=refreshed_args,
                                        structured_request_context=structured_request_context,
                                    )
                                except _CHAT_PROVIDER_REFRESH_EXCEPTIONS as refresh_error:
                                    if not isinstance(refresh_error, ByokResolutionError):
                                        provider_manager.record_failure(
                                            fallback_provider,
                                            sanitized_provider_stream_exception(refresh_error),
                                        )
                                    raise_detached_error(
                                        _bound_provider_http_exception(refresh_error)
                                    )
                                try:
                                    ensure_tool_autoexec_supports_request(
                                        cleaned_args=refreshed_args,
                                        should_run_tool_autoexec=should_run_legacy_tool_autoexec,
                                    )
                                except UnsupportedMultiChoiceToolAutoexecError as guard_error:
                                    return iter(
                                        [
                                            trusted_local_stream_error_frame(
                                                guard_error.error_code
                                            )
                                        ]
                                    )
                                model = refreshed_model or model
                                def llm_call_func_fb():
                                    return perform_chat_api_call(**refreshed_args)
                                try:
                                    result = _call_stream_factory_from_isolated_worker(
                                        llm_call_func_fb,
                                        _remaining_provider_factory_timeout,
                                    )
                                    if not hasattr(result, "__aiter__") and hasattr(result, "__iter__"):
                                        result = wrap_sync_stream(result)
                                    selected_provider = fallback_provider
                                    usage_credential_source = call_credential_source(refreshed_args)
                                    # Update run-first metric context to reflect fallback provider/model
                                    if run_first_metric_context is not None:
                                        run_first_metric_context["provider"] = str(fallback_provider or "").strip() or "unknown"
                                        run_first_metric_context["model"] = str(model or "").strip() or "unknown"
                                    llm_call_func = llm_call_func_fb
                                    with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                                        metrics.track_provider_fallback_success(
                                            requested_provider=provider,
                                            selected_provider=fallback_provider,
                                            streaming=True,
                                            queued=True,
                                        )
                                    return result
                                except LLMQuotaExceededError:
                                    raise  # the user's limit, not the fallback provider's health (TASK-13955)
                                except _CHAT_NONCRITICAL_EXCEPTIONS as fallback_error:
                                    provider_manager.record_failure(
                                        fallback_provider,
                                        sanitized_provider_stream_exception(fallback_error),
                                    )
                                    raise_detached_error(
                                        _bound_provider_invocation_exception(fallback_error)
                                    )
                    raise_detached_error(_bound_provider_invocation_exception(proc_error))

            queue_request_id = queue_request_id or get_request_id() or "unknown"
            try:
                queue_future = await queue_for_exec.enqueue(
                    request_id=queue_request_id,
                    request_data={"endpoint": "/api/v1/chat/completions", "mode": "stream"},
                    client_id=str(client_id),
                    priority=RequestPriority.HIGH,
                    estimated_tokens=est_tokens_for_queue,
                    processor=_queued_processor,
                    processor_args=(),
                    processor_kwargs={},
                    streaming=True,
                    stream_channel=stream_channel,
                    stream_factory_timeout=factory_budget_seconds,
                )
                _attach_queue_future_logger(queue_future, queue_request_id)
            except (ValueError, TimeoutError) as admission_error:
                with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                    metrics.track_rate_limit(str(client_id))
                detail = str(admission_error) or "Service busy. Please retry."
                status_code = (
                    status.HTTP_429_TOO_MANY_REQUESTS
                    if "rate limit" in detail.lower()
                    else status.HTTP_503_SERVICE_UNAVAILABLE
                )
                queue_exc = HTTPException(status_code=status_code, detail=detail)
                raise _mark_trusted_local_chat_signal(
                    queue_exc,
                    "queue_admission",
                ) from admission_error

            async def _channel_stream():
                graceful_end = False
                try:
                    while True:
                        try:
                            # Add timeout to prevent indefinite hang if producer crashes
                            # without sending the sentinel None value
                            item = await asyncio.wait_for(
                                stream_channel.get(),
                                timeout=float(CHAT_IDLE_TIMEOUT)
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Stream channel get timed out after {}s - "
                                "producer may have crashed without sending sentinel",
                                CHAT_IDLE_TIMEOUT
                            )
                            try:
                                if queue_future is not None and not queue_future.done():
                                    queue_future.cancel()
                            except _CHAT_NONCRITICAL_EXCEPTIONS:
                                pass
                            try:
                                while True:
                                    stream_channel.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            yield sse_data(_public_stream_error_payload('provider_unavailable'))
                            break
                        if item is None:
                            graceful_end = True
                            break
                        if isinstance(item, QueueStreamTerminalError):
                            yield (
                                sse_data(_public_stream_error_payload(item.code))
                            )
                            yield sse_done()
                            break
                        yield item
                except asyncio.CancelledError:
                    logger.info(
                        "Queued stream consumer cancelled for request {}; cancelling queued job",
                        queue_request_id,
                    )
                    raise
                finally:
                    # Pragmatic disconnect handling: cancel queued producer immediately so
                    # workers do not remain blocked waiting on a full channel.
                    if not graceful_end:
                        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                            if queue_future is not None and not queue_future.done():
                                queue_future.cancel()
                        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                            while True:
                                try:
                                    stream_channel.get_nowait()
                                except asyncio.QueueEmpty:
                                    break

            raw_stream_iter = _channel_stream()
        else:
            # Execute provided LLM call function in a worker to avoid blocking the loop.
            # llm_call_func is a sync callable (partial of perform_chat_api_call or a mock).
            raw_stream_iter = await _call_stream_factory_bounded(
                llm_call_func,
                timeout=_remaining_provider_factory_timeout(),
            )
            if selected_provider != provider:
                with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                    metrics.track_provider_fallback_success(
                        requested_provider=provider,
                        selected_provider=selected_provider,
                        streaming=True,
                        queued=False,
                    )
    except asyncio.CancelledError:
        raise
    except HTTPException as he:
        _emit_chat_run_first_completion_metric(
            metrics,
            context=run_first_metric_context,
            outcome="error",
        )
        local_signal_kind = trusted_local_chat_signal_kind(he)
        if local_signal_kind == "queue_admission":
            raise
        metrics.track_llm_call(
            selected_provider,
            model,
            time.time() - llm_start_time,
            success=False,
            error_type=type(he).__name__,
        )
        error_payload = _public_stream_error_payload(he)
        if (
            he.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            and local_signal_kind == "prompt_cost_guardrail"
            and isinstance(he.detail, dict)
            and he.detail.get("code") == "prompt_cost_guardrail_block"
            and he.detail.get("type") == "prompt_cost_guardrail_block"
        ):
            error_payload["error"] = {
                "code": "prompt_cost_guardrail_block",
                "type": "prompt_cost_guardrail_block",
                "message": "Prompt cost guardrail blocked request before provider dispatch.",
            }
        await _maybe_refund_streaming_rg(rg_refund_cb, cancelled=False, error=True)
        if isinstance(he, (LLMQuotaExceededError, LLMPolicyRefusal)):
            raise  # quota (§4b) or policy refusal: the caller gets its 402/400, not a provider-style error frame
        response = StreamingResponse(
            _terminal_stream_error(error_payload),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )
        if (
            he.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            and local_signal_kind == "prompt_cost_guardrail"
        ):
            _mark_trusted_local_chat_signal(response, "prompt_cost_guardrail")
        return response
    except _CHAT_NONCRITICAL_EXCEPTIONS as e:
        metrics.track_llm_call(
            selected_provider,
            model,
            time.time() - llm_start_time,
            success=False,
            error_type=type(e).__name__,
        )
        _fallback_stream_ok = False
        if provider_manager and not queue_enabled:
            provider_manager.record_failure(
                selected_provider,
                sanitized_provider_stream_exception(e),
            )
            # Only fallback on upstream/server errors; skip fallback for client/config errors
            if (
                enable_provider_fallback
                and isinstance(e, (ChatProviderError, ChatAPIError))
                and not _is_client_like_error(e)
                and provider_stream_error_allows_replay(e)
            ):
                fallback_provider = provider_manager.get_available_provider(exclude=[selected_provider])
                if fallback_provider:
                    logger.warning(f"Trying fallback provider {fallback_provider} after {selected_provider} failed")
                    try:
                        refreshed_args, refreshed_model = await _refresh_provider_params_bounded(
                            refresh_provider_params,
                            fallback_provider,
                            _remaining_provider_factory_timeout,
                        )
                        # Validate refreshed params have required fields before proceeding
                        if not isinstance(refreshed_args, dict) or "messages_payload" not in refreshed_args:
                            raise ValueError(f"Invalid refreshed params for {fallback_provider}: missing required fields")
                        try:
                            structured_request_context = prepare_structured_response_request(
                                provider=fallback_provider,
                                response_format=refreshed_args.get("response_format"),
                                requested_response_format=(
                                    structured_request_context.requested_response_format
                                    if structured_request_context is not None
                                    else refreshed_args.get("_structured_requested_response_format")
                                ),
                            )
                        except (
                            StructuredGenerationCapabilityError,
                            StructuredGenerationParseError,
                            StructuredGenerationSchemaError,
                        ) as structured_exc:
                            raise_detached_error(
                                build_structured_http_exception(structured_exc)
                            )
                        apply_structured_response_request(
                            cleaned_args=refreshed_args,
                            structured_request_context=structured_request_context,
                        )
                    except _CHAT_NONCRITICAL_EXCEPTIONS as refresh_error:
                        provider_manager.record_failure(
                            fallback_provider,
                            sanitized_provider_stream_exception(refresh_error),
                        )
                        raise
                    try:
                        ensure_tool_autoexec_supports_request(
                            cleaned_args=refreshed_args,
                            should_run_tool_autoexec=should_run_legacy_tool_autoexec,
                        )
                    except HTTPException as guard_error:
                        _emit_chat_run_first_completion_metric(
                            metrics,
                            context=run_first_metric_context,
                            outcome="error",
                        )
                        await _maybe_refund_streaming_rg(rg_refund_cb, cancelled=False, error=True)
                        return StreamingResponse(
                            _terminal_stream_error(
                                _public_stream_error_payload(guard_error)
                            ),
                            media_type="text/event-stream",
                            headers={
                                "Cache-Control": "no-cache, no-transform",
                                "X-Accel-Buffering": "no",
                            },
                        )
                    cleaned_args = refreshed_args
                    usage_credential_source = call_credential_source(cleaned_args)
                    model = refreshed_model or model
                    fallback_start_time = time.time()
                    def llm_call_func_fb():
                        return perform_chat_api_call(**cleaned_args)
                    try:
                        raw_stream_iter = await _call_stream_factory_bounded(
                            llm_call_func_fb,
                            timeout=_remaining_provider_factory_timeout(),
                        )
                        fallback_latency = time.time() - fallback_start_time
                        provider_manager.record_success(fallback_provider, fallback_latency)
                        metrics.track_llm_call(fallback_provider, model, fallback_latency, success=True)
                        selected_provider = fallback_provider
                        run_first_metric_context = _emit_chat_run_first_rollout_metrics(
                            metrics,
                            cleaned_args=cleaned_args,
                            provider=selected_provider,
                            model=model,
                            streaming=True,
                        )
                        llm_call_func = llm_call_func_fb
                        _fallback_stream_ok = True
                        # Explicit telemetry for direct (non-queued) streaming fallback success
                        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                            metrics.track_provider_fallback_success(
                                requested_provider=provider,
                                selected_provider=fallback_provider,
                                streaming=True,
                                queued=False,
                            )
                    except LLMQuotaExceededError:
                        raise  # the user's limit, not the fallback provider's health (TASK-13955)
                    except _CHAT_NONCRITICAL_EXCEPTIONS as fallback_error:
                        provider_manager.record_failure(
                            fallback_provider,
                            sanitized_provider_stream_exception(fallback_error),
                        )
                        raise

        if not _fallback_stream_ok:
            _emit_chat_run_first_completion_metric(
                metrics,
                context=run_first_metric_context,
                outcome="error",
            )

            error_payload = _public_stream_error_payload(e)
            await _maybe_refund_streaming_rg(rg_refund_cb, cancelled=False, error=True)
            return StreamingResponse(
                _terminal_stream_error(error_payload),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                },
            )

    if not (hasattr(raw_stream_iter, "__aiter__") or hasattr(raw_stream_iter, "__iter__")):
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Provider did not return a valid stream.")

    stream_mod_state = {
        "block_logged": False,
        "redact_logged": False,
        "block_capture_logged": False,
        "redact_capture_logged": False,
        "warn_capture_logged": False,
    }
    provider_output_recorded = False
    provider_output_lock = asyncio.Lock()
    clean_success_recorded = False
    clean_success_lock = asyncio.Lock()

    async def _record_provider_output_once() -> bool:
        """Record validated provider output once, retaining a clean-end retry."""

        nonlocal provider_output_recorded
        if not callable(on_provider_output):
            return False
        async with provider_output_lock:
            if provider_output_recorded:
                return True
            try:
                persisted = await on_provider_output(selected_provider)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                return False
            if persisted is not False:
                provider_output_recorded = True
            return provider_output_recorded

    async def _record_clean_success_once() -> bool:
        """Invoke the clean-completion callback at most once per stream."""

        nonlocal clean_success_recorded
        if not callable(on_success):
            return False
        async with clean_success_lock:
            if clean_success_recorded:
                return True
            try:
                persisted = await on_success(selected_provider)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                return False
            if persisted is not False:
                clean_success_recorded = True
            return clean_success_recorded

    loop_compat_enabled = False
    loop_compat_run_id = str(final_conversation_id or f"run_{_uuid.uuid4().hex[:12]}")
    try:
        if request is not None:
            loop_compat_enabled = _shared_is_truthy(request.headers.get("X-TLDW-Loop-Compat"))
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        loop_compat_enabled = False

    async def _apply_stream_output_policies(full_reply: str) -> tuple[str | None, bool]:
        """Run self-monitoring and output moderation on a reply before it is saved.

        Shared by complete and partial settlement so both store the same text.

        Returns:
            The text to save (None when blocked) and whether the reply was blocked.
        """
        full_reply_to_save: str | None = full_reply
        post_stream_blocked = False

        # Self-monitoring output check (streaming)
        if self_monitoring_service and full_reply:
            try:
                _sm_uid_s = None
                try:
                    if request is not None and hasattr(request, "state"):
                        _sm_uid_s = getattr(request.state, "user_id", None)
                except _CHAT_NONCRITICAL_EXCEPTIONS:
                    _sm_uid_s = None
                _sm_user_s = str(_sm_uid_s or client_id)
                _sm_loop_s = asyncio.get_running_loop()
                sm_result_s = await _sm_loop_s.run_in_executor(
                    None,
                    lambda: self_monitoring_service.check_text(
                        text=full_reply,
                        user_id=_sm_user_s,
                        phase="output",
                        conversation_id=str(final_conversation_id) if final_conversation_id else None,
                    ),
                )
                if sm_result_s.action == "block":
                    post_stream_blocked = True
                    full_reply_to_save = None
                elif sm_result_s.action == "redact" and sm_result_s.redacted_text is not None:
                    full_reply_to_save = sm_result_s.redacted_text
            except _CHAT_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(
                    "Self-monitoring output check (streaming) skipped error_type={}",
                    type(e).__name__,
                )

        try:
            _get_mod = moderation_getter or get_moderation_service
            moderation = _get_mod()
            req_user_id = None
            try:
                if request is not None and hasattr(request, "state"):
                    req_user_id = getattr(request.state, "user_id", None)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                req_user_id = None
            eff_policy = moderation.get_effective_policy(str(req_user_id) if req_user_id is not None else client_id)
            if full_reply and eff_policy.enabled and eff_policy.output_enabled:
                action = None
                redacted = None
                sample = None
                category = None
                matched_pattern = None
                match_span = None
                if hasattr(moderation, "evaluate_action_with_match"):
                    eval_res = moderation.evaluate_action_with_match(full_reply, eff_policy, "output")
                    if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                        action, redacted, matched_pattern = eval_res[0], eval_res[1], eval_res[2]
                        category = eval_res[3] if len(eval_res) >= 4 else None
                        match_span = eval_res[4] if len(eval_res) >= 5 else None
                    else:
                        action, redacted, matched_pattern = eval_res  # type: ignore
                    if match_span and hasattr(moderation, "build_sanitized_snippet"):
                        try:
                            sample = moderation.build_sanitized_snippet(full_reply, eff_policy, match_span, matched_pattern)
                        except _CHAT_NONCRITICAL_EXCEPTIONS:
                            sample = None
                elif hasattr(moderation, "evaluate_action"):
                    eval_res = moderation.evaluate_action(full_reply, eff_policy, "output")
                    if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                        action, redacted, matched_pattern = eval_res[0], eval_res[1], eval_res[2]
                        category = eval_res[3] if len(eval_res) >= 4 else None
                    else:
                        action, redacted, matched_pattern = eval_res  # type: ignore
                if action and action != "pass" and sample is None:
                    try:
                        _, sample = moderation.check_text(full_reply, eff_policy, "output")
                    except _CHAT_NONCRITICAL_EXCEPTIONS:
                        sample = None
                if action is None:
                    flagged, sample = moderation.check_text(full_reply, eff_policy, "output")
                    if flagged:
                        action = eff_policy.output_action
                        if action == "redact":
                            moderation.redact_text(full_reply, eff_policy)

                    if action == "block":
                        if not stream_mod_state["block_logged"]:
                            with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                                metrics.track_moderation_stream_block(
                                    str(req_user_id or client_id),
                                    category=(category or "default"),
                                )
                            try:
                                if audit_service and audit_context:
                                    _schedule_background_task(
                                        audit_service.log_event(
                                            event_type=AuditEventType.SECURITY_VIOLATION,
                                            context=audit_context,
                                            action="moderation.output",
                                            result="failure",
                                            metadata={
                                                "phase": "output",
                                                "streaming": True,
                                                "action": "block",
                                                "pattern": sample,
                                            },
                                        ),
                                        task_name="chat.stream.save.output.block.audit",
                                    )
                            except _CHAT_NONCRITICAL_EXCEPTIONS:
                                pass
                            stream_mod_state["block_logged"] = True
                        if not stream_mod_state["block_capture_logged"]:
                            await _capture_moderation_review_item_safely_async(
                                phase="output",
                                action="block",
                                excerpt=sample,
                                category=category,
                                matched_pattern=matched_pattern,
                                effective_policy=eff_policy,
                                source_id=str(final_conversation_id) if final_conversation_id else None,
                                user_id=str(req_user_id or client_id) if (req_user_id or client_id) else None,
                            )
                            stream_mod_state["block_capture_logged"] = True
                        post_stream_blocked = True
                        full_reply_to_save = None
                    elif action == "redact":
                        if not stream_mod_state["redact_logged"]:
                            with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                                metrics.track_moderation_output(
                                    str(req_user_id or client_id),
                                    "redact",
                                    streaming=True,
                                    category=(category or "default"),
                                )
                            try:
                                if audit_service and audit_context:
                                    _schedule_background_task(
                                        audit_service.log_event(
                                            event_type=AuditEventType.SECURITY_VIOLATION,
                                            context=audit_context,
                                            action="moderation.output",
                                            result="success",
                                            metadata={
                                                "phase": "output",
                                                "streaming": True,
                                                "action": "redact",
                                                "pattern": sample,
                                            },
                                        ),
                                        task_name="chat.stream.save.output.redact.audit",
                                    )
                            except _CHAT_NONCRITICAL_EXCEPTIONS:
                                pass
                            stream_mod_state["redact_logged"] = True
                        if not stream_mod_state["redact_capture_logged"]:
                            await _capture_moderation_review_item_safely_async(
                                phase="output",
                                action="redact",
                                excerpt=sample,
                                category=category,
                                matched_pattern=matched_pattern,
                                effective_policy=eff_policy,
                                source_id=str(final_conversation_id) if final_conversation_id else None,
                                user_id=str(req_user_id or client_id) if (req_user_id or client_id) else None,
                            )
                            stream_mod_state["redact_capture_logged"] = True
                        full_reply_to_save = moderation.redact_text(full_reply, eff_policy)
                    elif action == "warn":
                        if not stream_mod_state["warn_capture_logged"]:
                            await _capture_moderation_review_item_safely_async(
                                phase="output",
                                action="warn",
                                excerpt=sample,
                                category=category,
                                matched_pattern=matched_pattern,
                                effective_policy=eff_policy,
                                source_id=str(final_conversation_id) if final_conversation_id else None,
                                user_id=str(req_user_id or client_id) if (req_user_id or client_id) else None,
                            )
                            stream_mod_state["warn_capture_logged"] = True
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass

        return full_reply_to_save, post_stream_blocked

    def _stream_generation_metadata(generation: dict[str, Any] | None) -> dict[str, Any]:
        """Combine what the stream observed with the model and provider that produced it."""
        observed = generation if isinstance(generation, dict) else {}
        return build_generation_metadata(
            generation_status=observed.get("generation_status", GENERATION_STATUS_COMPLETE),
            model_id=model,
            provider=selected_provider,
            finish_reason=observed.get("finish_reason"),
            usage=observed.get("usage"),
        )

    # Partial replies are kept only for turns admitted through native history
    # selection, where the server owns settlement. Other save_to_db clients
    # retry an unanswered turn with tldw_retry_failed_turn, and that contract
    # needs the failed turn to have no saved reply.
    keep_partial_replies = bool(should_persist and history_persistence_ack)

    async def partial_save_callback(full_reply: str, *, generation: dict[str, Any]) -> str | None:
        """Keep an unfinished reply after a disconnect, stop or provider failure (D7 S3-1).

        Only the text is kept: partial tool calls are incomplete, and tool
        execution, success accounting and audit success belong to complete
        replies. A reply blocked by output policy is never saved.
        """
        if not keep_partial_replies or not final_conversation_id:
            return None
        if stream_mod_state.get("pending_stop_error") is not None:
            return None
        full_reply_to_save, post_stream_blocked = await _apply_stream_output_policies(full_reply)
        if post_stream_blocked or not isinstance(full_reply_to_save, str) or not full_reply_to_save.strip():
            return None
        message_payload = build_assistant_message_payload(
            character_card_for_context=character_card_for_context,
            assistant_parent_message_id=assistant_parent_message_id,
            content=full_reply_to_save,
            tool_calls=None,
            function_call=None,
            generation_metadata=_stream_generation_metadata(generation),
        )
        return await save_assistant_message(
            chat_db=chat_db,
            conversation_id=final_conversation_id,
            save_message_fn=save_message_fn,
            payload=message_payload,
        )

    async def save_callback(
        full_reply: str,
        tool_calls: list[dict[str, Any]] | None,
        function_call: dict[str, Any] | None,
        *,
        generation: dict[str, Any] | None = None,
        provider_usage: dict[str, Any] | None = None,
    ):
        nonlocal stream_metrics_recorded
        saved_message_id: str | None = None
        tool_execution_payload: list[dict[str, Any]] | None = None
        structured_events: list[dict[str, Any]] = []
        full_reply_to_save, post_stream_blocked = await _apply_stream_output_policies(full_reply)

        if structured_request_context is not None:
            try:
                structured_metadata = validate_structured_response(
                    raw_text=full_reply_to_save,
                    structured_request_context=structured_request_context,
                )
                structured_events.append(
                    {
                        "event": "structured_result",
                        "data": structured_metadata,
                    }
                )
            except (
                StructuredGenerationCapabilityError,
                StructuredGenerationParseError,
                StructuredGenerationSchemaError,
            ) as structured_exc:
                structured_events.append(
                    {
                        "event": "structured_error",
                        "data": _structured_error_detail(structured_exc),
                    }
                )

        if callable(on_stream_full_reply):
            try:
                maybe_result = on_stream_full_reply(str(full_reply or ""))
                if hasattr(maybe_result, "__await__"):
                    await maybe_result  # type: ignore[misc]
            except _CHAT_NONCRITICAL_EXCEPTIONS as stream_callback_err:
                logger.debug(
                    "on_stream_full_reply callback skipped error_type={}",
                    type(stream_callback_err).__name__,
                )

        if not stream_metrics_recorded:
            try:
                latency = time.time() - llm_start_time
                metrics.track_llm_call(selected_provider, model, latency, success=True)
                if provider_manager:
                    provider_manager.record_success(selected_provider, latency)
                stream_metrics_recorded = True
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                pass

        _emit_chat_run_first_tool_path_metrics(
            metrics,
            context=run_first_metric_context,
            tool_calls=tool_calls,
            function_call=function_call,
        )
        _emit_chat_run_first_completion_metric(
            metrics,
            context=run_first_metric_context,
            outcome="blocked" if post_stream_blocked else "success",
        )

        if should_persist and final_conversation_id and not post_stream_blocked and (
            full_reply_to_save or tool_calls or function_call
        ):
            message_payload = build_assistant_message_payload(
                character_card_for_context=character_card_for_context,
                assistant_parent_message_id=assistant_parent_message_id,
                content=full_reply_to_save,
                tool_calls=tool_calls,
                function_call=function_call,
                generation_metadata=_stream_generation_metadata(generation),
            )
            saved_message_id = await save_assistant_message(
                chat_db=chat_db,
                conversation_id=final_conversation_id,
                save_message_fn=save_message_fn,
                payload=message_payload,
            )

        if (
            not post_stream_blocked
            and should_run_legacy_tool_autoexec(cleaned_args)
            and isinstance(tool_calls, list)
            and tool_calls
        ):
            try:
                req_user_id = None
                try:
                    if request is not None and hasattr(request, "state"):
                        req_user_id = getattr(request.state, "user_id", None)
                except _CHAT_NONCRITICAL_EXCEPTIONS:
                    req_user_id = None

                autoexec_result = await execute_assistant_tool_calls(
                    tool_calls=tool_calls,
                    user_id=str(req_user_id) if req_user_id is not None else None,
                    client_id=client_id,
                    max_tool_calls=get_chat_max_tool_calls(),
                    timeout_ms=get_chat_tool_timeout_ms(),
                    allow_catalog=_resolve_chat_autoexec_allow_catalog(cleaned_args),
                    attach_idempotency=should_attach_tool_idempotency(),
                    idempotency_seed=str(final_conversation_id) if final_conversation_id else None,
                )
                tool_execution_payload = autoexec_result.event_payload().get("tool_results", [])

                if should_persist and final_conversation_id:
                    await save_tool_messages(
                        chat_db=chat_db,
                        conversation_id=final_conversation_id,
                        save_message_fn=save_message_fn,
                        tool_messages=autoexec_result.tool_messages(),
                    )
            except _CHAT_NONCRITICAL_EXCEPTIONS as autoexec_err:
                logger.warning(
                    "Streaming chat tool auto-execution skipped error_type={}",
                    type(autoexec_err).__name__,
                )
        # Usage logging after the stream completes: from the provider's usage chunk, else an estimate
        total_tokens = 0
        try:
            reported_prompt, reported_completion = _usage_tokens(provider_usage) if provider_usage else (0, 0)
            reported = (reported_prompt, reported_completion) != (0, 0)
            # A side the block does not count (a prompt-only block, say) comes from the estimate.
            prompt_tokens = reported_prompt
            if not prompt_tokens:
                try:
                    prompt_tokens = _estimate_tokens_from_messages(templated_llm_payload)
                except _CHAT_NONCRITICAL_EXCEPTIONS:
                    prompt_tokens = 0
            completion_tokens = reported_completion or max(0, len(full_reply or "") // 4)
            user_id = None
            api_key_id = None
            try:
                if request is not None and hasattr(request, "state"):
                    user_id = getattr(request.state, "user_id", None)
                    api_key_id = getattr(request.state, "api_key_id", None)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                pass
            latency_ms = int((time.time() - llm_start_time) * 1000)
            total_tokens = int(prompt_tokens + completion_tokens)
            await log_llm_usage(
                user_id=user_id,
                key_id=api_key_id,
                request=request,
                endpoint=(f"{request.method}:{request.url.path}" if request else "POST:/api/v1/chat/completions"),
                operation="chat",
                provider=selected_provider,
                model=model,
                status=200,
                latency_ms=latency_ms,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=total_tokens,
                request_id=(request.headers.get("X-Request-ID") if request else None) or (get_request_id() or None),
                conversation_id=(str(final_conversation_id) if final_conversation_id is not None else None),
                usage_metadata=provider_usage if reported else None,
                estimated=None if reported_prompt and reported_completion else True,
                estimate_source=None if reported else "stream_estimate",
                credential_source=usage_credential_source,
            )
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        # Commit the recorded tokens to the Resource Governor, if provided
        try:
            if callable(rg_commit_cb):
                # rg_commit_cb may be async or sync; call accordingly
                res = rg_commit_cb(total_tokens)
                if hasattr(res, "__await__"):
                    await res  # type: ignore[misc]
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        # Audit success
        try:
            if audit_service and audit_context:
                await audit_service.log_event(
                    event_type=AuditEventType.API_RESPONSE,
                    context=audit_context,
                    action="chat_completion_success",
                    result="success",
                    metadata={
                        "conversation_id": final_conversation_id,
                        "provider": selected_provider,
                        "model": model,
                        "streaming": True,
                    },
                )
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        # Retry any failed first-output accounting at the last clean boundary,
        # then notify consumers that the complete stream succeeded.
        try:
            await _record_provider_output_once()
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            await _record_clean_success_once()
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        events: list[dict[str, Any]] = list(structured_events)
        if tool_execution_payload is not None:
            events.append(
                {
                    "event": "tool_results",
                    "data": {"tool_results": tool_execution_payload},
                }
            )
            if loop_compat_enabled:
                seq = 1
                events.append(
                    {
                        "event": "run_started",
                        "data": {"run_id": loop_compat_run_id, "seq": seq},
                    }
                )
                seq += 1
                if isinstance(tool_calls, list) and tool_calls:
                    events.append(
                        {
                            "event": "tool_proposed",
                            "data": {"run_id": loop_compat_run_id, "seq": seq, "tool_calls": tool_calls},
                        }
                    )
                    seq += 1
                events.append(
                    {
                        "event": "tool_finished",
                        "data": {"run_id": loop_compat_run_id, "seq": seq, "tool_results": tool_execution_payload},
                    }
                )
                seq += 1
                events.append(
                    {
                        "event": "run_complete",
                        "data": {"run_id": loop_compat_run_id, "seq": seq},
                    }
                )
        elif loop_compat_enabled:
            events.extend(
                [
                    {
                        "event": "run_started",
                        "data": {"run_id": loop_compat_run_id, "seq": 1},
                    },
                    {
                        "event": "run_complete",
                        "data": {"run_id": loop_compat_run_id, "seq": 2},
                    },
                ]
            )

        if events:
            return {
                "saved_message_id": saved_message_id,
                "events": events,
            }
        return saved_message_id

    async def tracked_streaming_generator():
        async with metrics.track_streaming(final_conversation_id) as stream_tracker:
            _get_mod = moderation_getter or get_moderation_service
            moderation = _get_mod()
            req_user_id = None
            try:
                if request is not None and hasattr(request, "state"):
                    req_user_id = getattr(request.state, "user_id", None)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                req_user_id = None
            eff_policy = moderation.get_effective_policy(str(req_user_id) if req_user_id is not None else client_id)

            from tldw_Server_API.app.core.Chat.streaming_utils import StopStreamWithError

            mandatory_moderation_audit_tasks: list[asyncio.Task[Any]] = []
            stream_id = _uuid.uuid4().hex
            chunk_seq = 0

            stream_holdback = ""
            if CHAT_STREAM_INCLUDE_METADATA and final_conversation_id:
                metadata_payload = {
                    "event": "tldw_metadata",
                    "conversation_id": final_conversation_id,
                    "tldw_conversation_id": final_conversation_id,
                }
                if user_message_id:
                    metadata_payload["tldw_user_message_id"] = user_message_id
                if system_message_id:
                    metadata_payload["tldw_system_message_id"] = system_message_id
                if normalized_continuation_metadata:
                    metadata_payload["tldw_continuation"] = normalized_continuation_metadata
                yield sse_data(metadata_payload)
            try:
                stream_buffer_limit = int(os.getenv("MODERATION_STREAM_BUFFER_CHARS", "1024"))
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                stream_buffer_limit = 1024
            if stream_buffer_limit < 0:
                stream_buffer_limit = 0

            def _emit_with_holdback(text: str) -> str:
                nonlocal stream_holdback
                if stream_buffer_limit <= 0:
                    stream_holdback = ""
                    return text
                if not text:
                    return ""
                if len(text) <= stream_buffer_limit:
                    stream_holdback = text
                    return ""
                emit = text[:-stream_buffer_limit]
                stream_holdback = text[-stream_buffer_limit:]
                return emit

            def _flush_holdback() -> str:
                nonlocal stream_holdback
                if not stream_holdback:
                    return ""
                out = stream_holdback
                stream_holdback = ""
                return out

            async def _await_mandatory_moderation_audits() -> None:
                pending_tasks = list(mandatory_moderation_audit_tasks)
                mandatory_moderation_audit_tasks.clear()
                failures: list[BaseException] = []
                if pending_tasks:
                    results = await asyncio.gather(
                        *pending_tasks,
                        return_exceptions=True,
                    )
                    failures = [result for result in results if isinstance(result, BaseException)]
                if failures:
                    for failure in failures:
                        logger.error(
                            "Mandatory moderation audit task failed for {} error_type={}",
                            final_conversation_id,
                            type(failure).__name__,
                        )
                    raise StopStreamWithError(
                        message="Mandatory audit persistence unavailable",
                        error_type="audit_persistence_failure",
                    ) from failures[0]
                pending_stopper = stream_mod_state.pop("pending_stop_error", None)
                if pending_stopper is not None:
                    raise pending_stopper

            def _out_transform(s: str) -> str:
                nonlocal chunk_seq, stream_holdback
                if stream_mod_state.get("pending_stop_error") is not None:
                    return ""
                try:
                    mon = None
                    try:
                        mon = get_topic_monitoring_service()
                    except _CHAT_NONCRITICAL_EXCEPTIONS:
                        mon = None
                    team_ids = None
                    org_ids = None
                    try:
                        if request is not None and hasattr(request, "state"):
                            team_ids = getattr(request.state, "team_ids", None)
                            org_ids = getattr(request.state, "org_ids", None)
                    except _CHAT_NONCRITICAL_EXCEPTIONS:
                        pass
                    if mon is not None and s:
                        chunk_seq += 1
                        chunk_id = f"{stream_id}:{chunk_seq}"
                        mon.schedule_evaluate_and_alert(
                            user_id=str(req_user_id or client_id) if (req_user_id or client_id) else None,
                            text=s,
                            source="chat.output",
                            scope_type="user",
                            scope_id=str(req_user_id or client_id) if (req_user_id or client_id) else None,
                            team_ids=team_ids,
                            org_ids=org_ids,
                            source_id=stream_id,
                            chunk_id=chunk_id,
                            chunk_seq=chunk_seq,
                        )
                except _CHAT_NONCRITICAL_EXCEPTIONS as _e:
                    logger.debug("Topic monitoring (stream chunk) skipped: {}", exception_summary(_e))
                if not eff_policy.enabled or not eff_policy.output_enabled:
                    if stream_holdback:
                        out = f"{stream_holdback}{s}"
                        _flush_holdback()
                        return out
                    return s
                combined = f"{stream_holdback}{s}" if stream_holdback else s
                resolved_action = None
                matched_pattern = None
                sample = None
                redacted_combined = None
                out_category = None
                match_span = None
                if hasattr(moderation, "evaluate_action_with_match"):
                    try:
                        eval_res = moderation.evaluate_action_with_match(combined, eff_policy, "output")
                        if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                            resolved_action, redacted_combined, matched_pattern = eval_res[0], eval_res[1], eval_res[2]
                            out_category = eval_res[3] if len(eval_res) >= 4 else None
                            match_span = eval_res[4] if len(eval_res) >= 5 else None
                        else:
                            resolved_action, redacted_combined, matched_pattern = eval_res  # type: ignore
                    except _CHAT_NONCRITICAL_EXCEPTIONS:
                        resolved_action = None
                    if match_span and hasattr(moderation, "build_sanitized_snippet"):
                        try:
                            sample = moderation.build_sanitized_snippet(combined, eff_policy, match_span, matched_pattern)
                        except _CHAT_NONCRITICAL_EXCEPTIONS:
                            sample = None
                elif hasattr(moderation, "evaluate_action"):
                    try:
                        eval_res = moderation.evaluate_action(combined, eff_policy, "output")
                        if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                            resolved_action, redacted_combined, matched_pattern = eval_res[0], eval_res[1], eval_res[2]
                            out_category = eval_res[3] if len(eval_res) >= 4 else None
                        else:
                            resolved_action, redacted_combined, matched_pattern = eval_res  # type: ignore
                    except _CHAT_NONCRITICAL_EXCEPTIONS:
                        resolved_action = None
                if resolved_action and resolved_action != "pass" and sample is None:
                    try:
                        _, sample = moderation.check_text(combined, eff_policy, "output")
                    except _CHAT_NONCRITICAL_EXCEPTIONS:
                        sample = None
                if not resolved_action:
                    flagged, sample = moderation.check_text(combined, eff_policy, "output")
                    if not flagged:
                        return _emit_with_holdback(combined)
                    resolved_action = eff_policy.output_action
                    redacted_combined = moderation.redact_text(combined, eff_policy) if resolved_action == "redact" else None
                if resolved_action == "block":
                    if not stream_mod_state["block_logged"]:
                        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                            metrics.track_moderation_stream_block(str(req_user_id or client_id), category=(out_category or "default"))
                        mandatory_moderation_audit_tasks.append(
                            asyncio.create_task(
                                write_mandatory_moderation_audit(
                                    audit_service=audit_service,
                                    audit_context=audit_context,
                                    audit_event_type=AuditEventType.SECURITY_VIOLATION,
                                    action="moderation.output",
                                    result="failure",
                                    metadata={
                                        "phase": "output",
                                        "streaming": True,
                                        "action": "block",
                                        "pattern": sample,
                                    },
                                )
                            )
                        )
                        stream_mod_state["block_logged"] = True
                    stream_holdback = ""
                    stream_mod_state["pending_stop_error"] = StopStreamWithError(
                        message="Output violates moderation policy",
                        error_type="output_moderation_block",
                    )
                    return ""
                if resolved_action == "redact":
                    if not stream_mod_state["redact_logged"]:
                        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                            metrics.track_moderation_output(str(req_user_id or client_id), "redact", streaming=True, category=(out_category or "default"))
                        mandatory_moderation_audit_tasks.append(
                            asyncio.create_task(
                                write_mandatory_moderation_audit(
                                    audit_service=audit_service,
                                    audit_context=audit_context,
                                    audit_event_type=AuditEventType.SECURITY_VIOLATION,
                                    action="moderation.output",
                                    result="success",
                                    metadata={
                                        "phase": "output",
                                        "streaming": True,
                                        "action": "redact",
                                        "pattern": sample,
                                    },
                                )
                            )
                        )
                        stream_mod_state["redact_logged"] = True
                    redacted_out = (
                        redacted_combined
                        if isinstance(redacted_combined, str)
                        else moderation.redact_text(combined, eff_policy)
                    )
                    return _emit_with_holdback(redacted_out)
                return _emit_with_holdback(combined)

            # Allow streaming handler to flush any held-back tail at stream end
            with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                _out_transform.flush = _flush_holdback

            async def _finalize_stream(*, success: bool, cancelled: bool, error: bool) -> None:
                nonlocal stream_failure_recorded, stream_metrics_recorded
                if not success and not stream_failure_recorded and not stream_metrics_recorded:
                    try:
                        latency = time.time() - llm_start_time
                        if cancelled:
                            error_type = "stream_cancelled"
                        elif error:
                            error_type = "stream_error"
                        else:
                            error_type = "stream_incomplete"
                        metrics.track_llm_call(
                            selected_provider,
                            model,
                            latency,
                            success=False,
                            error_type=error_type,
                        )
                        if provider_manager and error and not cancelled:
                            provider_manager.record_failure(selected_provider, RuntimeError(error_type))
                        stream_failure_recorded = True
                    except _CHAT_NONCRITICAL_EXCEPTIONS:
                        pass
                if success:
                    return
                if callable(rg_refund_cb):
                    res = rg_refund_cb(cancelled=cancelled, error=error)
                    if hasattr(res, "__await__"):
                        await res  # type: ignore[misc]

            generator = get_chat_completion_pipeline().execute_streaming(
                request=StreamingPipelineRequest(
                    stream=raw_stream_iter,  # type: ignore[arg-type]
                    conversation_id=final_conversation_id,
                    model_name=model,
                    save_callback=save_callback,
                    partial_save_callback=partial_save_callback if keep_partial_replies else None,
                    finalize_callback=_finalize_stream,
                    on_first_output=_record_provider_output_once,
                    idle_timeout=CHAT_IDLE_TIMEOUT,
                    heartbeat_interval=CHAT_HEARTBEAT_INTERVAL,
                    text_transform=_out_transform,
                    before_success_callback=_await_mandatory_moderation_audits,
                    system_message_id=system_message_id,
                    user_message_id=user_message_id,
                    continuation_metadata=normalized_continuation_metadata,
                    history_persistence_ack=history_persistence_ack or None,
                    forward_usage_chunk=forward_usage_chunk,
                ),
                stream_factory=create_streaming_response_with_timeout,
            )
            try:
                async for chunk in generator:
                    if "heartbeat" in chunk:
                        stream_tracker.add_heartbeat()
                    else:
                        stream_tracker.add_chunk()
                    yield chunk
            finally:
                if mandatory_moderation_audit_tasks:
                    await asyncio.gather(*mandatory_moderation_audit_tasks, return_exceptions=True)

    streaming_generator = tracked_streaming_generator()

    # Feature-flagged: route through unified SSE abstraction for pilot
    try:
        use_unified = _shared_is_truthy(os.getenv("STREAMS_UNIFIED", "0"))
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        use_unified = False

    if use_unified:
        # Use SSEStream to standardize lifecycle + metrics; forward lines from the
        # existing tracked generator, filtering provider [DONE] and emitting our own.
        from tldw_Server_API.app.core.Streaming.streams import SSEStream

        sse_stream = SSEStream(
            labels={"component": "chat", "endpoint": "chat_completions_stream"},
            provider_control_passthru=True,
        )
        done_seen = False

        async def _produce():
            nonlocal done_seen
            try:
                async for ln in streaming_generator:
                    if not ln:
                        continue
                    if is_done_line(ln):
                        # Suppress provider DONE; emit unified DONE immediately and stop producing
                        if not done_seen:
                            await sse_stream.done()
                            done_seen = True
                        break
                    if is_trusted_local_stream_frame(ln):
                        # This marker is created only after raw provider frames pass
                        # through the adapter sanitizer. Preserve its identity across
                        # the unified queue; ASGI encoding removes it before delivery.
                        await sse_stream._enqueue(ln)
                    else:
                        await sse_stream.send_raw_sse_line(ln)
                if not done_seen:
                    await sse_stream.done()
            except asyncio.CancelledError:
                raise
            except _CHAT_NONCRITICAL_EXCEPTIONS as e:
                # As a safeguard; tracked_streaming_generator typically yields error frames itself
                normalized_error = normalize_provider_stream_error(e)
                code = normalized_error.code if normalized_error else "provider_unavailable"
                await sse_stream.error(
                    code,
                    provider_stream_error_payload(code)["error"]["message"],
                    force=True,
                )

        async def _gen():
            prod = asyncio.create_task(_produce())
            try:
                async for line in sse_stream.iter_sse():
                    yield line
            finally:
                await cancel_stream_tasks_bounded([prod])

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    # Legacy path: return the tracked generator directly
    return StreamingResponse(
        streaming_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _execute_non_stream_call_impl(
    *,
    current_loop: Any,
    cleaned_args: dict[str, Any],
    selected_provider: str,
    provider: str,
    model: str,
    request_json: str,
    request: Any,
    metrics: Any,
    provider_manager: Any,
    templated_llm_payload: list[dict[str, Any]],
    should_persist: bool,
    final_conversation_id: str,
    character_card_for_context: dict[str, Any] | None,
    chat_db: Any,
    save_message_fn: Callable[..., Any],
    system_message_id: str | None = None,
    user_message_id: str | None = None,
    audit_service: Any | None,
    audit_context: Any | None,
    client_id: str,
    queue_execution_enabled: bool,
    enable_provider_fallback: bool,
    llm_call_func: Callable[[], Any],
    refresh_provider_params: Callable[[str], Any],
    structured_request_context: StructuredResponseRequestContext | None = None,
    moderation_getter: Callable[[], Any] | None = None,
    on_success: Callable[[str], Awaitable[bool | None]] | None = None,
    self_monitoring_service: Any | None = None,
    assistant_parent_message_id: str | None = None,
    continuation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a non-streaming LLM call with queue, failover, moderation, and persistence.

    Returns the encoded payload (dict) ready to be wrapped by JSONResponse.
    """
    llm_start_time = time.time()
    structured_metadata: dict[str, Any] | None = None
    normalized_continuation_metadata = (
        dict(continuation_metadata)
        if isinstance(continuation_metadata, dict) and continuation_metadata
        else None
    )
    run_first_metric_context = _emit_chat_run_first_rollout_metrics(
        metrics,
        cleaned_args=cleaned_args,
        provider=selected_provider,
        model=model,
        streaming=False,
    )
    llm_response = None
    metrics_recorded = False
    queue_failure_recorded = False
    queue_enabled = False
    provider_use_recorded = False
    provider_use_lock = asyncio.Lock()
    provider_failure_type: str | None = None
    pending_assistant_payload: dict[str, Any] | None = None
    pending_tool_messages: list[dict[str, Any]] = []
    prompt_guardrail_decision: PromptCostGuardrailDecision | None = None

    async def _record_provider_use(provider_name: str) -> bool:
        """Best-effort, once-only accounting immediately after provider success."""

        nonlocal provider_use_recorded
        if not callable(on_success):
            return False
        async with provider_use_lock:
            if provider_use_recorded:
                return True
            try:
                persisted = await on_success(provider_name)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                return False
            if persisted is False:
                return False
            provider_use_recorded = True
            return True

    async def _record_dict_response_usage(
        *,
        response: dict[str, Any],
        messages: list[dict[str, Any]],
        content: Any,
        provider_name: str,
        model_name: str,
        started_at: float,
    ) -> None:
        """Record token and durable usage for one validated dict response."""

        choices = response.get("choices")
        usage = usage_with_service_tier(response)  # OpenAI reports service_tier beside usage
        if usage:
            try:
                normalized_usage = normalize_llm_usage(
                    provider=provider_name,
                    usage=usage if isinstance(usage, dict) else None,
                    choices=choices if isinstance(choices, list) else None,
                )
                prompt_tokens = normalized_usage.input_tokens
                completion_tokens = normalized_usage.output_tokens
                metrics.track_tokens(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=model_name,
                    provider=provider_name,
                )
                user_id = None
                api_key_id = None
                try:
                    if request is not None and hasattr(request, "state"):
                        user_id = getattr(request.state, "user_id", None)
                        api_key_id = getattr(request.state, "api_key_id", None)
                except _CHAT_NONCRITICAL_EXCEPTIONS:
                    pass
                await log_llm_usage(
                    user_id=user_id,
                    key_id=api_key_id,
                    request=request,
                    endpoint=(
                        f"{request.method}:{request.url.path}"
                        if request
                        else "POST:/api/v1/chat/completions"
                    ),
                    operation="chat",
                    provider=provider_name,
                    model=model_name,
                    status=200,
                    latency_ms=int((time.time() - started_at) * 1000),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=normalized_usage.total_tokens,
                    request_id=(
                        request.headers.get("X-Request-ID") if request else None
                    )
                    or (get_request_id() or None),
                    conversation_id=(
                        str(final_conversation_id)
                        if final_conversation_id is not None
                        else None
                    ),
                    usage_metadata=usage if isinstance(usage, dict) else None,
                    choice_count=normalized_usage.choice_count,
                    estimate_source=normalized_usage.estimate_source,
                    credential_source=call_credential_source(cleaned_args),
                )
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                pass
            return

        try:
            try:
                prompt_tokens = _estimate_tokens_from_messages(messages)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                prompt_tokens = 0
            response_choices = collect_non_stream_choices(response)
            completion_tokens = (
                estimate_completion_tokens_from_choices(response_choices)
                if response_choices
                else max(0, len(_extract_text_from_content(content)) // 4)
            )
            metrics.track_tokens(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                model=model_name,
                provider=provider_name,
            )
            user_id = None
            api_key_id = None
            try:
                if request is not None and hasattr(request, "state"):
                    user_id = getattr(request.state, "user_id", None)
                    api_key_id = getattr(request.state, "api_key_id", None)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                pass
            await log_llm_usage(
                user_id=user_id,
                key_id=api_key_id,
                request=request,
                endpoint=(
                    f"{request.method}:{request.url.path}"
                    if request
                    else "POST:/api/v1/chat/completions"
                ),
                operation="chat",
                provider=provider_name,
                model=model_name,
                status=200,
                latency_ms=int((time.time() - started_at) * 1000),
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(prompt_tokens + completion_tokens),
                request_id=(request.headers.get("X-Request-ID") if request else None)
                or (get_request_id() or None),
                conversation_id=(
                    str(final_conversation_id)
                    if final_conversation_id is not None
                    else None
                ),
                estimated=True,
                estimate_source="missing_usage",
                credential_source=call_credential_source(cleaned_args),
            )
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass

    try:
        try:
            if structured_request_context is None:
                structured_request_context = prepare_structured_response_request(
                    provider=selected_provider,
                    response_format=cleaned_args.get("response_format"),
                    requested_response_format=cleaned_args.get("_structured_requested_response_format"),
                )
            apply_structured_response_request(
                cleaned_args=cleaned_args,
                structured_request_context=structured_request_context,
            )
            ensure_tool_autoexec_supports_request(
                cleaned_args=cleaned_args,
                should_run_tool_autoexec=should_run_legacy_tool_autoexec,
            )
            prompt_guardrail_decision = _evaluate_chat_prompt_cost_guardrails(
                cleaned_args=cleaned_args,
                templated_llm_payload=templated_llm_payload,
                selected_provider=selected_provider,
                model=model,
                streaming=False,
                continuation_metadata=normalized_continuation_metadata,
            )
        except (
            StructuredGenerationCapabilityError,
            StructuredGenerationParseError,
            StructuredGenerationSchemaError,
        ) as structured_exc:
            raise_detached_error(build_structured_http_exception(structured_exc))
        queue_for_exec = None
        try:
            queue_for_exec = get_request_queue()
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            queue_for_exec = None
        queue_enabled = (
            queue_execution_enabled
            and queue_for_exec is not None
            and queue_is_active(queue_for_exec)
        )

        async def _record_provider_use_after_cancel(result: Any) -> None:
            """Record only usable late output before cancellation closes its runtime."""

            if _nonstream_provider_result_is_usable(result):
                await _record_provider_use(selected_provider)

        if queue_enabled:
            est_tokens_for_queue = estimate_tokens_from_json(request_json)
            def _queued_processor():
                nonlocal queue_failure_recorded
                local_start = time.time()
                try:
                    result = llm_call_func()
                    _require_usable_nonstream_provider_result(result)
                    latency = time.time() - local_start
                    metrics.track_llm_call(selected_provider, model, latency, success=True)
                    if provider_manager:
                        provider_manager.record_success(selected_provider, latency)
                    if selected_provider != provider:
                        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                            metrics.track_provider_fallback_success(
                                requested_provider=provider,
                                selected_provider=selected_provider,
                                streaming=False,
                                queued=True,
                            )
                    return result
                except (LLMQuotaExceededError, LLMPolicyRefusal):
                    raise  # the user's limit or a policy refusal, not the provider's health: no failure, no fallback
                except _CHAT_NONCRITICAL_EXCEPTIONS as proc_error:
                    latency = time.time() - local_start
                    metrics.track_llm_call(
                        selected_provider,
                        model,
                        latency,
                        success=False,
                        error_type=type(proc_error).__name__,
                    )
                    if provider_manager:
                        provider_manager.record_failure(
                            selected_provider,
                            sanitized_provider_stream_exception(proc_error),
                        )
                    queue_failure_recorded = True
                    bounded_error = _bound_provider_invocation_exception(proc_error)
                    raise_detached_error(bounded_error)

            try:
                fut = await queue_for_exec.enqueue(
                    request_id=(get_request_id() or "unknown"),
                    request_data={"endpoint": "/api/v1/chat/completions", "mode": "non-stream"},
                    client_id=str(client_id),
                    priority=RequestPriority.NORMAL,
                    estimated_tokens=est_tokens_for_queue,
                    processor=_queued_processor,
                    processor_args=(),
                    processor_kwargs={},
                    streaming=False,
                    stream_channel=None,
                )
            except (ValueError, TimeoutError) as admission_error:
                with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                    metrics.track_rate_limit(str(client_id))
                detail = str(admission_error) or "Service busy. Please retry."
                status_code = (
                    status.HTTP_429_TOO_MANY_REQUESTS
                    if "rate limit" in detail.lower()
                    else status.HTTP_503_SERVICE_UNAVAILABLE
                )
                queue_exc = HTTPException(status_code=status_code, detail=detail)
                raise _mark_trusted_local_chat_signal(
                    queue_exc,
                    "queue_admission",
                ) from admission_error
            try:
                llm_response = await await_owned_worker(
                    fut,
                    on_cancel_result=_record_provider_use_after_cancel,
                )
                _require_usable_nonstream_provider_result(llm_response)
                await await_owned_worker(
                    _record_provider_use(selected_provider)
                )
            except Exception as future_error:  # noqa: BLE001 - queue/provider trust boundary
                bounded_error = _bound_provider_invocation_exception(future_error)
                raise_detached_error(bounded_error)
            metrics_recorded = True
        else:
            # Execute provided LLM call function in a worker to avoid blocking the loop.
            # llm_call_func is a sync callable (partial of perform_chat_api_call or a mock).
            def _invoke_non_stream_provider() -> Any:
                try:
                    return llm_call_func()
                except asyncio.CancelledError as provider_cancel:
                    raise_detached_error(
                        sanitized_provider_stream_exception(provider_cancel)
                    )

            try:
                llm_response = await await_owned_worker(
                    await_bounded_sync_call(
                        _invoke_non_stream_provider,
                        pool=SYNC_ADAPTER_CALL_POOL,
                        exhaustion_message="Provider adapter capacity is exhausted",
                    ),
                    on_cancel_result=_record_provider_use_after_cancel,
                )
                _require_usable_nonstream_provider_result(llm_response)
                await await_owned_worker(
                    _record_provider_use(selected_provider)
                )
            except Exception as provider_error:  # noqa: BLE001 - adapter trust boundary
                provider_failure_type = type(provider_error).__name__
                bounded_error = _bound_provider_invocation_exception(provider_error)
                raise_detached_error(bounded_error)
        llm_latency = time.time() - llm_start_time
        if not metrics_recorded:
            metrics.track_llm_call(selected_provider, model, llm_latency, success=True)
            if provider_manager:
                provider_manager.record_success(selected_provider, llm_latency)
            if selected_provider != provider:
                with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                    metrics.track_provider_fallback_success(
                        requested_provider=provider,
                        selected_provider=selected_provider,
                        streaming=False,
                        queued=False,
                    )
    except HTTPException as he:
        _emit_chat_run_first_completion_metric(
            metrics,
            context=run_first_metric_context,
            outcome="error",
        )
        if _is_trusted_local_chat_http_exception(he):
            raise
        raise_detached_error(sanitized_provider_stream_exception(he))
    except _CHAT_NONCRITICAL_EXCEPTIONS as e:
        llm_latency = time.time() - llm_start_time
        if not queue_failure_recorded:
            metrics.track_llm_call(
                selected_provider,
                model,
                llm_latency,
                success=False,
                error_type=provider_failure_type or type(e).__name__,
            )
            if provider_manager:
                provider_manager.record_failure(
                    selected_provider,
                    sanitized_provider_stream_exception(e),
                )

        if provider_manager:
            if (
                enable_provider_fallback
                and isinstance(e, (ChatProviderError, ChatAPIError))
                and not _is_client_like_error(e)
                and provider_stream_error_allows_replay(e)
            ):
                fallback_provider = provider_manager.get_available_provider(exclude=[selected_provider])
                if fallback_provider:
                    logger.warning(f"Trying fallback provider {fallback_provider} after {selected_provider} failed")
                    try:
                        refreshed = refresh_provider_params(fallback_provider)
                        if hasattr(refreshed, "__await__"):
                            refreshed_args, refreshed_model = await refreshed  # type: ignore[misc]
                        else:
                            refreshed_args, refreshed_model = refreshed
                        # Validate refreshed params have required fields before proceeding
                        if not isinstance(refreshed_args, dict) or "messages_payload" not in refreshed_args:
                            raise ValueError(
                                f"Invalid refreshed params for {fallback_provider}: missing required fields"
                            )
                        try:
                            structured_request_context = prepare_structured_response_request(
                                provider=fallback_provider,
                                response_format=refreshed_args.get("response_format"),
                                requested_response_format=(
                                    structured_request_context.requested_response_format
                                    if structured_request_context is not None
                                    else refreshed_args.get("_structured_requested_response_format")
                                ),
                            )
                        except (
                            StructuredGenerationCapabilityError,
                            StructuredGenerationParseError,
                            StructuredGenerationSchemaError,
                        ) as structured_exc:
                            raise_detached_error(
                                build_structured_http_exception(structured_exc)
                            )
                        apply_structured_response_request(
                            cleaned_args=refreshed_args,
                            structured_request_context=structured_request_context,
                        )
                    except _CHAT_PROVIDER_REFRESH_EXCEPTIONS as refresh_error:
                        if not isinstance(refresh_error, ByokResolutionError):
                            provider_manager.record_failure(
                                fallback_provider,
                                sanitized_provider_stream_exception(refresh_error),
                            )
                        _emit_chat_run_first_completion_metric(
                            metrics,
                            context=run_first_metric_context,
                            outcome="error",
                        )
                        bounded_error = _bound_provider_http_exception(refresh_error)
                        raise_detached_error(bounded_error)
                    try:
                        ensure_tool_autoexec_supports_request(
                            cleaned_args=refreshed_args,
                            should_run_tool_autoexec=should_run_legacy_tool_autoexec,
                        )
                    except HTTPException:
                        _emit_chat_run_first_completion_metric(
                            metrics,
                            context=run_first_metric_context,
                            outcome="error",
                        )
                        raise
                    cleaned_args = refreshed_args
                    model = refreshed_model or model
                    fallback_start_time = time.time()

                    async def _record_fallback_use_after_cancel(result: Any) -> None:
                        if _nonstream_provider_result_is_usable(result):
                            await _record_provider_use(fallback_provider)

                    try:
                        llm_response = await await_owned_worker(
                            perform_chat_api_call_async(**cleaned_args),
                            on_cancel_result=_record_fallback_use_after_cancel,
                        )
                        _require_usable_nonstream_provider_result(llm_response)
                        await await_owned_worker(
                            _record_provider_use(fallback_provider)
                        )
                        fallback_latency = time.time() - fallback_start_time
                        provider_manager.record_success(fallback_provider, fallback_latency)
                        metrics.track_llm_call(fallback_provider, model, fallback_latency, success=True)
                        selected_provider = fallback_provider
                        run_first_metric_context = _emit_chat_run_first_rollout_metrics(
                            metrics,
                            cleaned_args=cleaned_args,
                            provider=selected_provider,
                            model=model,
                            streaming=False,
                        )
                        metrics_recorded = True
                        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
                            metrics.track_provider_fallback_success(
                                requested_provider=provider,
                                selected_provider=fallback_provider,
                                streaming=False,
                                queued=False,
                            )
                    except LLMQuotaExceededError:
                        _emit_chat_run_first_completion_metric(
                            metrics,
                            context=run_first_metric_context,
                            outcome="error",
                        )
                        raise  # the user's limit, not the fallback provider's health (TASK-13955)
                    except Exception as fallback_error:  # noqa: BLE001 - adapter trust boundary
                        fallback_latency = time.time() - fallback_start_time
                        bounded_error = _bound_provider_invocation_exception(
                            fallback_error
                        )
                        metrics.track_llm_call(
                            fallback_provider,
                            model,
                            fallback_latency,
                            success=False,
                            error_type=type(bounded_error).__name__,
                        )
                        provider_manager.record_failure(
                            fallback_provider,
                            sanitized_provider_stream_exception(bounded_error),
                        )
                        _emit_chat_run_first_completion_metric(
                            metrics,
                            context=run_first_metric_context,
                            outcome="error",
                        )
                        raise_detached_error(bounded_error)
                else:
                    _emit_chat_run_first_completion_metric(
                        metrics,
                        context=run_first_metric_context,
                        outcome="error",
                    )
                    raise
            else:
                _emit_chat_run_first_completion_metric(
                    metrics,
                    context=run_first_metric_context,
                    outcome="error",
                )
                raise
        else:
            _emit_chat_run_first_completion_metric(
                metrics,
                context=run_first_metric_context,
                outcome="error",
            )
            raise

    if isinstance(llm_response, str) and should_force_normalize_string_responses():
        llm_response = _wrap_raw_string_response(llm_response, model)

    processed_choices: list[NonStreamChoice] = collect_non_stream_choices(llm_response)
    first_choice = primary_choice(processed_choices)
    content_to_save: Any | None = first_choice.content if first_choice else None
    tool_calls_to_save: Any | None = first_choice.tool_calls if first_choice else None
    function_call_to_save: Any | None = first_choice.function_call if first_choice else None
    first_turn_tool_calls: Any | None = tool_calls_to_save
    first_turn_function_call: Any | None = function_call_to_save
    if llm_response and isinstance(llm_response, dict):
        await _record_dict_response_usage(
            response=llm_response,
            messages=templated_llm_payload,
            content=content_to_save,
            provider_name=selected_provider,
            model_name=model,
            started_at=llm_start_time,
        )
    elif isinstance(llm_response, str):
        content_to_save = llm_response
        await _record_dict_response_usage(
            response={},
            messages=templated_llm_payload,
            content=content_to_save,
            provider_name=selected_provider,
            model_name=model,
            started_at=llm_start_time,
        )
    elif llm_response is None:
        _emit_chat_run_first_completion_metric(
            metrics,
            context=run_first_metric_context,
            outcome="error",
        )
        raise ChatProviderError(provider=provider, message="Provider unavailable or returned no response", status_code=502)

    # Cache content text for moderation/usage when content is non-string
    content_text_for_usage = _extract_text_from_content(content_to_save)

    def _refresh_primary_choice_state() -> None:
        nonlocal first_choice, content_to_save, tool_calls_to_save, function_call_to_save, content_text_for_usage
        first_choice = primary_choice(processed_choices)
        if first_choice is not None:
            content_to_save = first_choice.content
            tool_calls_to_save = first_choice.tool_calls
            function_call_to_save = first_choice.function_call
            content_text_for_usage = first_choice.content_text
        else:
            content_text_for_usage = _extract_text_from_content(content_to_save)

    async def _apply_output_safety_to_current_choices() -> None:
        nonlocal content_to_save, content_text_for_usage

        result = await apply_output_safety_to_choices(
            choices=processed_choices,
            fallback_content=content_to_save,
            fallback_content_text=content_text_for_usage,
            runtime=OutputModerationRuntime(
                request=request,
                client_id=client_id,
                conversation_id=final_conversation_id,
                metrics=metrics,
                audit_service=audit_service,
                audit_context=audit_context,
                moderation_getter=moderation_getter,
                self_monitoring_service=self_monitoring_service,
                topic_monitoring_getter=get_topic_monitoring_service,
                review_item_callback=_capture_moderation_review_item_safely_async,
                completion_metric_callback=lambda outcome: _emit_chat_run_first_completion_metric(
                    metrics,
                    context=run_first_metric_context,
                    outcome=outcome,
                ),
            ),
        )
        content_to_save = result.content_to_save
        content_text_for_usage = result.content_text_for_usage
        _refresh_primary_choice_state()

    await _apply_output_safety_to_current_choices()

    assistant_message_id: str | None = None
    tool_execution_payload: list[dict[str, Any]] | None = None
    tool_auto_continue_meta: dict[str, bool] | None = None
    defer_structured_persistence = bool(
        structured_request_context is not None
        and should_run_legacy_tool_autoexec(cleaned_args)
        and isinstance(tool_calls_to_save, list)
        and tool_calls_to_save
        and should_auto_continue_tools_once()
    )
    if structured_request_context is not None and not defer_structured_persistence:
        try:
            structured_metadata = validate_structured_choices(
                choices=processed_choices,
                structured_request_context=structured_request_context,
                validate_structured_response=validate_structured_response,
                fallback_content=content_to_save,
            )
        except (
            StructuredGenerationCapabilityError,
            StructuredGenerationParseError,
            StructuredGenerationSchemaError,
        ) as structured_exc:
            _emit_chat_run_first_completion_metric(
                metrics,
                context=run_first_metric_context,
                outcome="error",
            )
            raise_detached_error(build_structured_http_exception(structured_exc))

    should_save_response = (
        should_persist
        and final_conversation_id
        and (content_to_save or tool_calls_to_save or function_call_to_save)
    )
    if should_save_response and not defer_structured_persistence:
        message_payload = _build_assistant_message_payload(
            character_card_for_context=character_card_for_context,
            assistant_parent_message_id=assistant_parent_message_id,
            content=content_to_save,
            tool_calls=tool_calls_to_save,
            function_call=function_call_to_save,
            generation_metadata=_nonstream_generation_metadata(
                llm_response=llm_response,
                choice=first_choice,
                model=model,
                provider=selected_provider,
            ),
        )
        assistant_message_id = await save_assistant_message(
            chat_db=chat_db,
            conversation_id=final_conversation_id,
            save_message_fn=save_message_fn,
            payload=message_payload,
        )
    elif should_save_response and defer_structured_persistence:
        pending_assistant_payload = _build_assistant_message_payload(
            character_card_for_context=character_card_for_context,
            assistant_parent_message_id=assistant_parent_message_id,
            content=content_to_save,
            tool_calls=tool_calls_to_save,
            function_call=function_call_to_save,
            generation_metadata=_nonstream_generation_metadata(
                llm_response=llm_response,
                choice=first_choice,
                model=model,
                provider=selected_provider,
            ),
        )

    if should_run_legacy_tool_autoexec(cleaned_args) and isinstance(tool_calls_to_save, list) and tool_calls_to_save:
        try:
            req_user_id = None
            try:
                if request is not None and hasattr(request, "state"):
                    req_user_id = getattr(request.state, "user_id", None)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                req_user_id = None

            autoexec_result = await execute_assistant_tool_calls(
                tool_calls=tool_calls_to_save,
                user_id=str(req_user_id) if req_user_id is not None else None,
                client_id=client_id,
                max_tool_calls=get_chat_max_tool_calls(),
                timeout_ms=get_chat_tool_timeout_ms(),
                allow_catalog=_resolve_chat_autoexec_allow_catalog(cleaned_args),
                attach_idempotency=should_attach_tool_idempotency(),
                idempotency_seed=str(final_conversation_id) if final_conversation_id else None,
            )
            tool_execution_payload = autoexec_result.event_payload().get("tool_results", [])
            tool_messages = autoexec_result.tool_messages()

            failed_results = len([r for r in autoexec_result.results if not r.ok])
            timed_out_results = len([r for r in autoexec_result.results if r.timed_out])
            logger.info(
                "Chat tool auto-exec summary (conv={} requested={} attempts={} executed={} failed={} timed_out={} truncated={})",
                final_conversation_id,
                autoexec_result.requested_calls,
                autoexec_result.execution_attempts,
                autoexec_result.executed_calls,
                failed_results,
                timed_out_results,
                autoexec_result.truncated,
            )

            if should_persist and final_conversation_id:
                if defer_structured_persistence:
                    pending_tool_messages = list(tool_messages)
                else:
                    await save_tool_messages(
                        chat_db=chat_db,
                        conversation_id=final_conversation_id,
                        save_message_fn=save_message_fn,
                        tool_messages=list(tool_messages),
                    )

            if should_auto_continue_tools_once() and tool_messages:
                tool_auto_continue_meta = {"attempted": True, "succeeded": False}
                try:
                    continuation_messages: list[dict[str, Any]] = list(templated_llm_payload)
                    assistant_context_message: dict[str, Any] = {"role": "assistant"}
                    if content_to_save is not None:
                        assistant_context_message["content"] = content_to_save
                    if tool_calls_to_save is not None:
                        assistant_context_message["tool_calls"] = tool_calls_to_save
                    if function_call_to_save is not None:
                        assistant_context_message["function_call"] = function_call_to_save
                    if len(assistant_context_message) > 1:
                        continuation_messages.append(assistant_context_message)
                    continuation_messages.extend(tool_messages)

                    continuation_args = dict(cleaned_args)
                    continuation_args["messages_payload"] = continuation_messages
                    continuation_args["streaming"] = False
                    continuation_start_time = time.time()
                    try:
                        continuation_response = await await_owned_worker(
                            perform_chat_api_call_async(**continuation_args),
                            on_cancel_result=_record_provider_use_after_cancel,
                        )
                        _require_usable_nonstream_provider_result(
                            continuation_response
                        )
                    except Exception as continuation_error:  # noqa: BLE001 - adapter trust boundary
                        continuation_latency = time.time() - continuation_start_time
                        bounded_error = _bound_provider_invocation_exception(
                            continuation_error
                        )
                        metrics.track_llm_call(
                            selected_provider,
                            model,
                            continuation_latency,
                            success=False,
                            error_type=type(bounded_error).__name__,
                        )
                        if provider_manager and not isinstance(continuation_error, LLMPolicyRefusal):
                            provider_manager.record_failure(
                                selected_provider,
                                sanitized_provider_stream_exception(bounded_error),
                            )
                        raise_detached_error(bounded_error)
                    await await_owned_worker(
                        _record_provider_use(selected_provider)
                    )
                    continuation_latency = time.time() - continuation_start_time
                    metrics.track_llm_call(
                        selected_provider,
                        model,
                        continuation_latency,
                        success=True,
                    )
                    if provider_manager:
                        provider_manager.record_success(
                            selected_provider,
                            continuation_latency,
                        )
                    if isinstance(continuation_response, str) and should_force_normalize_string_responses():
                        continuation_response = _wrap_raw_string_response(continuation_response, model)

                    continuation_content: Any | None = None
                    if continuation_response and isinstance(continuation_response, dict):
                        continuation_choices = continuation_response.get("choices")
                        if continuation_choices and isinstance(continuation_choices, list):
                            continuation_message = continuation_choices[0].get("message") or {}
                            if isinstance(continuation_message, dict):
                                continuation_content = continuation_message.get("content")
                    elif isinstance(continuation_response, str):
                        continuation_content = continuation_response

                    if isinstance(continuation_response, dict):
                        await _record_dict_response_usage(
                            response=continuation_response,
                            messages=continuation_messages,
                            content=continuation_content,
                            provider_name=selected_provider,
                            model_name=model,
                            started_at=continuation_start_time,
                        )
                    elif isinstance(continuation_response, str):
                        await _record_dict_response_usage(
                            response={},
                            messages=continuation_messages,
                            content=continuation_content,
                            provider_name=selected_provider,
                            model_name=model,
                            started_at=continuation_start_time,
                        )

                    if continuation_response is not None:
                        llm_response = continuation_response
                        processed_choices = collect_non_stream_choices(llm_response)
                        first_choice = primary_choice(processed_choices)
                        content_to_save = first_choice.content if first_choice else None
                        tool_calls_to_save = first_choice.tool_calls if first_choice else None
                        function_call_to_save = first_choice.function_call if first_choice else None
                        if first_choice is None and isinstance(llm_response, str):
                            content_to_save = llm_response
                        content_text_for_usage = _extract_text_from_content(content_to_save)
                        await _apply_output_safety_to_current_choices()
                        if not processed_choices and isinstance(llm_response, str):
                            llm_response = content_to_save
                        try:
                            continuation_structured_metadata = validate_structured_choices(
                                choices=processed_choices,
                                structured_request_context=structured_request_context,
                                validate_structured_response=validate_structured_response,
                                fallback_content=content_to_save,
                            )
                        except (
                            StructuredGenerationCapabilityError,
                            StructuredGenerationParseError,
                            StructuredGenerationSchemaError,
                        ) as structured_exc:
                            _emit_chat_run_first_completion_metric(
                                metrics,
                                context=run_first_metric_context,
                                outcome="error",
                            )
                            raise_detached_error(
                                build_structured_http_exception(structured_exc)
                            )
                        if continuation_structured_metadata is not None:
                            structured_metadata = continuation_structured_metadata

                        if (
                            should_persist
                            and final_conversation_id
                            and (content_to_save or tool_calls_to_save or function_call_to_save)
                        ):
                            if pending_assistant_payload is not None:
                                assistant_message_id = await save_assistant_message(
                                    chat_db=chat_db,
                                    conversation_id=final_conversation_id,
                                    save_message_fn=save_message_fn,
                                    payload=pending_assistant_payload,
                                )
                                pending_assistant_payload = None
                            if pending_tool_messages:
                                await save_tool_messages(
                                    chat_db=chat_db,
                                    conversation_id=final_conversation_id,
                                    save_message_fn=save_message_fn,
                                    tool_messages=pending_tool_messages,
                                )
                                pending_tool_messages = []
                            continuation_payload = _build_assistant_message_payload(
                                character_card_for_context=character_card_for_context,
                                assistant_parent_message_id=assistant_parent_message_id,
                                content=content_to_save,
                                tool_calls=tool_calls_to_save,
                                function_call=function_call_to_save,
                                generation_metadata=_nonstream_generation_metadata(
                                    llm_response=llm_response,
                                    choice=first_choice,
                                    model=model,
                                    provider=selected_provider,
                                ),
                            )
                            continuation_message_id = await save_assistant_message(
                                chat_db=chat_db,
                                conversation_id=final_conversation_id,
                                save_message_fn=save_message_fn,
                                payload=continuation_payload,
                            )
                            if continuation_message_id:
                                assistant_message_id = continuation_message_id
                        tool_auto_continue_meta["succeeded"] = True
                except HTTPException:
                    raise
                except _CHAT_NONCRITICAL_EXCEPTIONS as continue_err:
                    logger.warning(
                        "Chat tool auto-continue skipped due to error: {}",
                        exception_summary(continue_err),
                    )
        except HTTPException:
            raise
        except _CHAT_NONCRITICAL_EXCEPTIONS as autoexec_err:
            logger.warning(
                "Chat tool auto-execution skipped due to error: {}",
                exception_summary(autoexec_err),
            )

    if structured_request_context is not None and structured_metadata is None:
        try:
            structured_metadata = validate_structured_choices(
                choices=processed_choices,
                structured_request_context=structured_request_context,
                validate_structured_response=validate_structured_response,
                fallback_content=content_to_save,
            )
        except (
            StructuredGenerationCapabilityError,
            StructuredGenerationParseError,
            StructuredGenerationSchemaError,
        ) as structured_exc:
            _emit_chat_run_first_completion_metric(
                metrics,
                context=run_first_metric_context,
                outcome="error",
            )
            raise_detached_error(build_structured_http_exception(structured_exc))

    if pending_assistant_payload is not None and should_persist and final_conversation_id:
        assistant_message_id = await save_assistant_message(
            chat_db=chat_db,
            conversation_id=final_conversation_id,
            save_message_fn=save_message_fn,
            payload=pending_assistant_payload,
        )
        pending_assistant_payload = None
    if pending_tool_messages and should_persist and final_conversation_id:
        await save_tool_messages(
            chat_db=chat_db,
            conversation_id=final_conversation_id,
            save_message_fn=save_message_fn,
            tool_messages=pending_tool_messages,
        )
        pending_tool_messages = []

    _emit_chat_run_first_tool_path_metrics(
        metrics,
        context=run_first_metric_context,
        tool_calls=first_turn_tool_calls,
        function_call=first_turn_function_call,
    )
    _emit_chat_run_first_completion_metric(
        metrics,
        context=run_first_metric_context,
        outcome="success",
    )

    if INJECT_ASSISTANT_NAME and isinstance(llm_response, dict):
        try:
            asst_name = sanitize_sender_name(
                character_card_for_context.get("name") if character_card_for_context else None
            )
            inject_assistant_name_into_choices(collect_non_stream_choices(llm_response), asst_name)
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass

    # Encode payload (large responses via CPU-bound handler)
    if llm_response and isinstance(llm_response, dict) and len(str(llm_response)) > 10000:
        encoded_json = await process_large_json_async(llm_response)
        encoded_payload = _json.loads(encoded_json)
    else:
        encoded_payload = await current_loop.run_in_executor(None, jsonable_encoder, llm_response)

    if isinstance(encoded_payload, dict):
        if INJECT_ASSISTANT_NAME:
            try:
                asst_name = sanitize_sender_name(
                    character_card_for_context.get("name") if character_card_for_context else None
                )
                inject_assistant_name_into_choices(collect_non_stream_choices(encoded_payload), asst_name)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                pass
        encoded_payload["tldw_conversation_id"] = final_conversation_id
        if user_message_id:
            encoded_payload["tldw_user_message_id"] = user_message_id
        if assistant_message_id:
            encoded_payload["tldw_message_id"] = assistant_message_id
        if system_message_id:
            encoded_payload["tldw_system_message_id"] = system_message_id
        if tool_execution_payload is not None:
            encoded_payload["tldw_tool_results"] = tool_execution_payload
        if tool_auto_continue_meta is not None:
            encoded_payload["tldw_tool_auto_continue"] = tool_auto_continue_meta
        if normalized_continuation_metadata is not None:
            encoded_payload["tldw_continuation"] = normalized_continuation_metadata
        if structured_metadata is not None:
            encoded_payload["tldw_structured"] = structured_metadata
        if prompt_guardrail_decision is not None:
            encoded_payload["tldw_prompt_guardrails"] = (
                prompt_guardrail_decision.to_response_metadata()
            )

    # Audit success
    if audit_service and audit_context:
        with contextlib.suppress(_CHAT_NONCRITICAL_EXCEPTIONS):
            await audit_service.log_event(
                event_type=AuditEventType.API_RESPONSE,
                context=audit_context,
                action="chat_completion_success",
                result="success",
                metadata={
                    "conversation_id": final_conversation_id,
                    "provider": selected_provider,
                    "model": model,
                    "streaming": False,
                },
            )

    # Retry best-effort accounting if an earlier touch failed transiently.
    await _record_provider_use(selected_provider)

    return encoded_payload


def get_chat_completion_pipeline() -> ChatCompletionPipeline:
    return ChatCompletionPipeline(
        non_stream_executor=_execute_non_stream_call_impl,
        streaming_executor=create_chat_streaming_response,
    )


async def execute_non_stream_call(**kwargs: Any) -> dict[str, Any]:
    return await get_chat_completion_pipeline().execute_non_stream(**kwargs)
