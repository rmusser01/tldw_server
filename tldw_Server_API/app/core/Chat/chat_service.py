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
    first_user_message_text,
    get_or_create_character_context,
    get_or_create_conversation,
)
from tldw_Server_API.app.core.Chat.chat_logging import (
    exception_summary,
    prompt_template_summary,
    text_summary,
)
from tldw_Server_API.app.core.Chat.chat_loop_engine import is_chat_loop_mode_enabled
from tldw_Server_API.app.core.Chat.generation_metadata import (
    GENERATION_STATUS_COMPLETE,
    GENERATION_STATUS_LENGTH,
    build_generation_metadata,
    moonshot_reasoning_event,
)
from tldw_Server_API.app.core.Chat.history_selection import (
    REPLY_ACTION_REFUSED_QUESTION,
    REPLY_ACTION_REFUSED_REQUEST,
    HistorySelectionError,
    HistorySelectionSnapshotV1,
    reply_action_refusal,
)
from tldw_Server_API.app.core.Chat.knowledge_save import strip_reasoning_blocks
from tldw_Server_API.app.core.Chat.message_utils import should_persist_message_role
from tldw_Server_API.app.core.Chat.moderation_pipeline import (
    OutputModerationRuntime,
    apply_output_safety_to_choices,
    prepare_saved_stream_reply,
)
from tldw_Server_API.app.core.Chat.persistence_service import (
    NativeStreamPersistence,
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
from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError

# isort: split
from tldw_Server_API.app.core.LLM_Calls.gateway import (  # used by build_call_params_from_request and the known-models helpers
    CHAT_COMPLETIONS_SURFACE,
    _provider_inventory_endpoint,
    _raise_inventory_failure,
    official_moonshot_chat_base_url,
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
from tldw_Server_API.app.core.LLM_Calls.provider_results import MOONSHOT_REASONING_KEY, bounded_moonshot_reasoning
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
from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _approx_tokens, _StreamUsage, _usage_tokens
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
from tldw_Server_API.app.core.Usage.llm_accounting import submit_accounting
from tldw_Server_API.app.core.Usage.llm_call_context import call_credential_source
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage, usage_with_service_tier
from tldw_Server_API.app.core.Usage.pricing_catalog import (
    list_provider_models,
)
from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage
from tldw_Server_API.app.core.Utils.cpu_bound_handler import process_large_json_async
from tldw_Server_API.app.core.Chat import call_state as _call_state
from tldw_Server_API.app.core.Chat import completion_pipeline as _completion_pipeline
import functools
from tldw_Server_API.app.core.Chat import streaming_pipeline as _streaming_pipeline
from tldw_Server_API.app.core.Chat.history_context import build_context_and_messages  # noqa: F401 - re-export (TASK-13950.25)

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


def _tool_continuation_assistant_message(
    choice: NonStreamChoice | None,
    provider: str,
    *,
    content: Any,
    tool_calls: list[dict[str, Any]] | None,
    function_call: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the assistant turn that precedes the executed tool results."""
    message: dict[str, Any] = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if function_call is not None:
        message["function_call"] = function_call
    if provider == "moonshot" and choice is not None:
        reasoning = choice.message.get("reasoning_content")
        if isinstance(reasoning, str):
            message["reasoning_content"] = reasoning
    return message


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
        from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
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
        from tldw_Server_API.app.core.LLM_Calls.model_metadata import (
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
    from tldw_Server_API.app.core.Chat.chat_request_schemas import (
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
    request_metadata = getattr(request_data, "metadata", None)
    if isinstance(request_metadata, dict) and request_metadata.get("tldw_moonshot_reasoning_history_v1") is True:
        call_params["_chat_moonshot_reasoning_history"] = True

    call_model = resolved_model if resolved_model is not None else call_params.get("model")
    if isinstance(call_model, str) and call_model.strip():
        call_params["model"] = call_model.strip()
    else:
        call_params.pop("model", None)

    if call_params.get("max_completion_tokens") is not None:
        call_params.pop("max_tokens", None)

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
    moonshot_reasoning: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for tests/imports that use the old local helper."""

    return build_assistant_message_payload(
        character_card_for_context=character_card_for_context,
        assistant_parent_message_id=assistant_parent_message_id,
        content=content,
        tool_calls=tool_calls,
        function_call=function_call,
        generation_metadata=generation_metadata,
        moonshot_reasoning=moonshot_reasoning,
    )


def _stream_generation_metadata(
    generation: dict[str, Any] | None, *, model: str, provider: str
) -> dict[str, Any]:
    """Combine what the stream observed with the model and provider that produced it."""
    observed = generation if isinstance(generation, dict) else {}
    return build_generation_metadata(
        generation_status=observed.get("generation_status", GENERATION_STATUS_COMPLETE),
        model_id=model,
        provider=provider,
        finish_reason=observed.get("finish_reason"),
        usage=observed.get("usage"),
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


def _history_for_request(
    snapshot: Any,
    content: Any,
    character_card: dict[str, Any],
    continue_prefix: str | None,
) -> list[dict[str, Any]]:
    """Format the selected rows for this request.

    On Continue the reply being continued is the final context message, and some
    providers (for example Anthropic) refuse a final message that ends in whitespace,
    so it goes out without its trailing whitespace.
    """
    formatted = _format_selected_history_messages(snapshot, content, character_card)
    if continue_prefix is not None:
        formatted[-1]["content"] = formatted[-1]["content"].rstrip()
    return formatted


def _attach_saved_moonshot_reasoning(message: dict[str, Any], extra: Mapping[str, Any]) -> None:
    """Carry approved reasoning only from stored Moonshot assistant turns."""
    if message["role"] != "assistant" or extra.get("provider") != "moonshot":
        return
    reasoning = bounded_moonshot_reasoning(extra.get(MOONSHOT_REASONING_KEY))
    if reasoning is not None:
        message[MOONSHOT_REASONING_KEY] = reasoning


def _format_selected_history_messages(
    snapshot: Any,
    content: Any,
    character_card: dict[str, Any],
    *,
    user_name: str = "User",
    replace_user_placeholders: bool = True,
) -> list[dict[str, Any]]:
    """Format only detached owner-selected rows; never reload current sources."""
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
        if role == "assistant":
            answer = strip_reasoning_blocks(text)
            if not answer.strip() and answer != text and not item["images"]:
                if not (item.get("tool_calls") or extra.get("function_call")):
                    continue
                text = None
            else:
                text = answer
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
        _attach_saved_moonshot_reasoning(entry, extra)
        if item.get("tool_calls") is not None:
            entry["tool_calls"] = item["tool_calls"]
        for key in ("function_call", "tool_call_id"):
            if extra.get(key) is not None:
                entry[key] = extra[key]
        if extra.get("sender_name") or extra.get("tool_name"):
            entry["name"] = extra.get("sender_name") or extra.get("tool_name")
        output.append(entry)
    return output


def _normalize_history_overlap_content(value: Any, *, assistant: bool) -> Any:
    """Normalize text parts for comparing the saved and submitted history."""
    if isinstance(value, str):
        value = [{"type": "text", "text": value}]
    if assistant and isinstance(value, list):
        return [
            {**part, "text": strip_reasoning_blocks(part["text"])}
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            else part
            for part in value
        ]
    return value


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


async def _record_stream_usage(
    *,
    request: Any,
    provider: str,
    model: str,
    templated_llm_payload: list[dict[str, Any]],
    conversation_id: str | None,
    llm_start_time: float,
    credential_source: str | None,
    stream_usage: _StreamUsage | None,
    full_reply: str = "",
    rg_commit_cb: Callable[[int], Any] | None,
) -> None:
    """Write the one usage row of a /chat/completions stream and commit its tokens; never raises.

    The row comes from the provider's usage chunk when one arrived, else from the prompt estimate and what
    the stream received: content, reasoning and tool-call arguments, as the provider-level recorder counts
    them. A side the chunk does not count comes from the estimate. The same tokens go to the Resource
    Governor and billing through ``rg_commit_cb``, however the stream ended (TASK-13953).
    """
    total_tokens = 0
    try:
        provider_usage = stream_usage.usage if stream_usage is not None else None
        reported_prompt, reported_completion = _usage_tokens(provider_usage) if provider_usage else (0, 0)
        reported = (reported_prompt, reported_completion) != (0, 0)
        prompt_tokens = reported_prompt
        if not prompt_tokens:
            try:
                prompt_tokens = _estimate_tokens_from_messages(templated_llm_payload)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                prompt_tokens = 0
        received_chars = stream_usage.chars if stream_usage is not None else len(full_reply or "")
        completion_tokens = reported_completion or _approx_tokens(received_chars)
        user_id = None
        api_key_id = None
        try:
            if request is not None and hasattr(request, "state"):
                user_id = getattr(request.state, "user_id", None)
                api_key_id = getattr(request.state, "api_key_id", None)
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        total_tokens = int(prompt_tokens + completion_tokens)
        await log_llm_usage(
            user_id=user_id,
            key_id=api_key_id,
            request=request,
            endpoint=(f"{request.method}:{request.url.path}" if request else "POST:/api/v1/chat/completions"),
            operation="chat",
            provider=provider,
            model=model,
            status=200,
            latency_ms=int((time.time() - llm_start_time) * 1000),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=total_tokens,
            request_id=(request.headers.get("X-Request-ID") if request else None) or (get_request_id() or None),
            conversation_id=(str(conversation_id) if conversation_id is not None else None),
            usage_metadata=provider_usage if reported else None,
            estimated=None if reported and (prompt_tokens, completion_tokens) == (reported_prompt, reported_completion) else True,
            estimate_source=None if reported else "stream_estimate",
            credential_source=credential_source,
        )
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        pass
    finally:
        # Commit the recorded tokens to the Resource Governor and billing whatever the write did: a commit skipped
        # on a failed write would be a quota bypass
        try:
            if callable(rg_commit_cb):
                res = rg_commit_cb(total_tokens)  # may be async or sync
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
        from tldw_Server_API.app.core.LLM_Calls.commercial_model_defaults import apply_commercial_output_default

        request_options = dict(cleaned_args, model=model, messages=templated_llm_payload, stream=streaming)
        apply_commercial_output_default(selected_provider, request_options)
        decision = evaluate_prompt_cost_guardrails(
            envelope,
            request_options=request_options,
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


def _check_fallback_prompt_cost(
    args: dict[str, Any], provider: str, model: str, *, streaming: bool
) -> None:
    """Recheck the redirected prompt against the fallback provider's output budget."""
    _evaluate_chat_prompt_cost_guardrails(
        cleaned_args=args,
        templated_llm_payload=args["messages_payload"],
        selected_provider=provider,
        model=model,
        streaming=streaming,
    )


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
    native_persistence: NativeStreamPersistence | None = None,
    forward_usage_chunk: bool = False,
) -> StreamingResponse:
    """Execute a streaming LLM call with queue, failover, moderation, and persistence.

    Returns a StreamingResponse that yields SSE chunks and handles:
    - provider call invocation and fallback
    - output moderation (chunk-wise)
    - saving final assistant message to DB
    - usage logging and audit success
    """
    return await _streaming_pipeline.run_streaming_call(_call_state.StreamCall.from_call_locals(locals()))


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
    return await _completion_pipeline.run_non_stream_call(_call_state.NonStreamCall.from_call_locals(locals()))


async def execute_non_stream_call(**kwargs: Any) -> dict[str, Any]:
    return await _execute_non_stream_call_impl(**kwargs)
