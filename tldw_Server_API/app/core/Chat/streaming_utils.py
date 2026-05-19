# streaming_utils.py
# Description: Utilities for handling streaming responses safely
#
# Imports
import asyncio
import concurrent.futures
import contextlib
import json

#######################################################################################################################
#
# Constants:
# Load configuration values
import os
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import datetime, timezone
from typing import Any, Optional, Union

from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.testing import is_truthy

_STREAMING_NONCRITICAL_EXCEPTIONS = (
    ChatAPIError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    json.JSONDecodeError,
)

_config = load_comprehensive_config()
# ConfigParser uses sections, check if Chat-Module section exists
_chat_config = {}
if _config and _config.has_section('Chat-Module'):
    _chat_config = dict(_config.items('Chat-Module'))


def _parse_int(value: Any, default: int, *, min_value: Optional[int] = None) -> int:
    try:
        if value is None:
            return default
        parsed = int(str(value).strip())
    except _STREAMING_NONCRITICAL_EXCEPTIONS:
        return default
    if min_value is not None and parsed < min_value:
        return min_value
    return parsed

# Timeout for idle connections (seconds)
STREAMING_IDLE_TIMEOUT = _parse_int(
    os.getenv('STREAMING_IDLE_TIMEOUT_SECONDS') or
    _chat_config.get('streaming_idle_timeout_seconds', 300),
    300,
    min_value=1,
)  # Default 5 minutes

# Heartbeat interval for long-running streams (seconds)
HEARTBEAT_INTERVAL = _parse_int(
    os.getenv('STREAMING_HEARTBEAT_INTERVAL_SECONDS') or
    _chat_config.get('streaming_heartbeat_interval_seconds', 30),
    30,
    min_value=0,
)

# Maximum response size in bytes (default 10MB) - configurable via env or config
MAX_RESPONSE_SIZE_BYTES = _parse_int(
    os.getenv('STREAMING_MAX_RESPONSE_SIZE_BYTES') or
    _chat_config.get('streaming_max_response_size_bytes', 10 * 1024 * 1024),
    10 * 1024 * 1024,
    min_value=1,
)

# Tool call accumulator max index to prevent memory exhaustion - configurable
MAX_TOOL_CALL_INDEX = _parse_int(
    os.getenv('STREAMING_MAX_TOOL_CALL_INDEX') or
    _chat_config.get('streaming_max_tool_call_index', 1000),
    1000,
    min_value=0,
)

# Maximum length for accumulated tool call arguments (in characters)
# This prevents OOM attacks from malicious streams with unbounded arguments
MAX_TOOL_ARGUMENT_LENGTH = _parse_int(
    os.getenv('STREAMING_MAX_TOOL_ARGUMENT_LENGTH') or
    _chat_config.get('streaming_max_tool_argument_length', 50_000),
    50_000,
    min_value=0,
)

# Maximum number of items in the full_response list to prevent unbounded growth
MAX_RESPONSE_LIST_LENGTH = _parse_int(
    os.getenv('STREAMING_MAX_RESPONSE_LIST_LENGTH') or
    _chat_config.get('streaming_max_response_list_length', 100_000),
    100_000,
    min_value=1,
)

# Offload sync iterators to a background thread to avoid blocking the event loop
try:
    STREAMING_SYNC_BRIDGE_ENABLED = is_truthy(str(
        os.getenv('STREAMING_SYNC_BRIDGE_ENABLED') or
        _chat_config.get('streaming_sync_bridge_enabled', 'true')
    ).lower())
except (ValueError, TypeError) as exc:
    logger.debug(f"Failed to parse STREAMING_SYNC_BRIDGE_ENABLED, using default: {exc}")
    STREAMING_SYNC_BRIDGE_ENABLED = True

try:
    STREAMING_SYNC_BRIDGE_MAX_QUEUE = int(
        os.getenv('STREAMING_SYNC_BRIDGE_MAX_QUEUE') or
        _chat_config.get('streaming_sync_bridge_max_queue', 32)
    )
except (ValueError, TypeError) as exc:
    logger.debug(f"Failed to parse STREAMING_SYNC_BRIDGE_MAX_QUEUE, using default: {exc}")
    STREAMING_SYNC_BRIDGE_MAX_QUEUE = 32
if STREAMING_SYNC_BRIDGE_MAX_QUEUE <= 0:
    STREAMING_SYNC_BRIDGE_MAX_QUEUE = 32

try:
    _include_meta_raw = (
        os.getenv("CHAT_STREAM_INCLUDE_METADATA")
        or _chat_config.get("chat_stream_include_metadata")
        or "true"
    )
    CHAT_STREAM_INCLUDE_METADATA = is_truthy(_include_meta_raw)
except (ValueError, TypeError) as exc:
    logger.debug(f"Failed to parse CHAT_STREAM_INCLUDE_METADATA, using default: {exc}")
    CHAT_STREAM_INCLUDE_METADATA = True

#######################################################################################################################
#
# Functions:

def _extract_text_from_upstream_sse(chunk_str: str) -> tuple[Optional[str], Optional[dict[str, Any]], bool]:
    """
    Normalize provider-emitted SSE frames to plain text content.

    Accepts a string that may be:
      - a raw text fragment (returns it as text_content)
      - an SSE line like "data: {...}" (extracts JSON and returns delta.content if present)
      - an SSE DONE line "data: [DONE]" (signals completion via is_done=True)

    Returns: (text_content, error_payload, is_done)
      - text_content: extracted textual delta (or original text) or None
      - error_payload: if upstream provided an error object, return it for direct emission
      - is_done: True if upstream indicated [DONE]
    """
    if not chunk_str:
        return None, None, False

    # Normalize common invisible prefixes (BOM, zero-width spaces) and trim whitespace
    s = chunk_str.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()

    # Ignore comment/heartbeat/event-only lines from upstream
    if s.startswith(":") or s.startswith("event:"):
        return None, None, False

    # If any 'data:' line exists, try to parse; some providers send 'event:' + 'data:' pairs or multiple frames
    if s.startswith("data:") or ("\ndata:" in s or s.startswith("event:") or "data:" in s):
        saw_done = False
        first_error = None
        # Process by lines to handle possible multi-line chunks
        for line in s.splitlines():
            ls = line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
            if not ls:
                continue
            if ls.startswith(":") or ls.startswith("event:"):
                # Skip comment or event name lines
                continue
            if not ls.startswith("data:"):
                continue
            payload_str = ls[len("data:"):].strip()
            if payload_str == "[DONE]":
                saw_done = True
                continue
            try:
                data = json.loads(payload_str)
            except _STREAMING_NONCRITICAL_EXCEPTIONS:
                # Try next line if present
                continue

            if isinstance(data, dict) and "error" in data and first_error is None:
                try:
                    _ = json.dumps({"error": data.get("error")})
                    first_error = {"error": data.get("error")}
                except _STREAMING_NONCRITICAL_EXCEPTIONS:
                    first_error = {"error": {"message": "Upstream error (unparseable)", "type": "stream_error"}}
                continue

            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0] or {}
                    delta = first.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        return str(content), None, False
                    # Fallback to message.content (non-stream case)
                    message = first.get("message") or {}
                    msg_content = message.get("content")
                    if msg_content:
                        return str(msg_content), None, False
        # If no content found but DONE or error encountered, reflect that
        if first_error is not None:
            return None, first_error, False
        if saw_done:
            return None, None, True
        return None, None, False

    # Not an SSE frame; treat as plain text chunk
    return chunk_str, None, False


async def _async_iter_sync_stream(
    stream: Iterator[Any],
    *,
    queue_maxsize: int = STREAMING_SYNC_BRIDGE_MAX_QUEUE,
) -> AsyncIterator[Any]:
    """Bridge a sync iterator onto the event loop without blocking it.

    Spawns a daemon thread to consume the sync iterator, passing chunks
    through an asyncio.Queue for non-blocking async consumption.

    Args:
        stream: A synchronous iterator to bridge.
        queue_maxsize: Maximum queue depth for backpressure (default: 32).

    Yields:
        Items from the sync iterator, now available asynchronously.

    Raises:
        Exception: Re-raises any exception that occurred in the sync iterator.
    """
    loop = asyncio.get_running_loop()
    maxsize = max(int(queue_maxsize or 0), 1)
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=maxsize)
    stop_event = threading.Event()

    def _queue_put(item: tuple[str, Any]) -> None:
        if loop.is_closed():
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        except (RuntimeError, asyncio.InvalidStateError) as exc:
            logger.debug(f"Failed to schedule sync stream enqueue: {exc}")
            return
        while True:
            try:
                fut.result(timeout=1.0)
            except (concurrent.futures.TimeoutError, TimeoutError):
                if stop_event.is_set() or loop.is_closed():
                    try:
                        fut.cancel()
                    except (RuntimeError, asyncio.InvalidStateError) as cancel_err:
                        logger.debug(f"Failed to cancel sync stream enqueue: {cancel_err}")
                    return
            except (RuntimeError, concurrent.futures.CancelledError) as exc:
                logger.debug(f"Failed to enqueue sync stream chunk: {exc}")
                return
            else:
                return

    def _worker() -> None:
        try:
            for chunk in stream:
                if stop_event.is_set():
                    break
                _queue_put(("data", chunk))
        except _STREAMING_NONCRITICAL_EXCEPTIONS as exc:
            _queue_put(("error", exc))
        finally:
            _queue_put(("done", None))

    thread = threading.Thread(target=_worker, name="sync-stream-bridge", daemon=True)
    thread.start()

    try:
        while True:
            kind, payload = await queue.get()
            if kind == "data":
                yield payload
            elif kind == "error":
                raise payload
            elif kind == "done":
                break
    finally:
        stop_event.set()
        # Give worker thread a brief moment to notice stop_event before closing.
        thread.join(timeout=0.5)
        try:
            if hasattr(stream, "close") and callable(stream.close):  # type: ignore[attr-defined]
                stream.close()  # type: ignore[attr-defined]
        except _STREAMING_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                f"Exception while closing bridged sync stream ({type(stream).__name__}): {exc}"
            )

class StreamingResponseHandler:
    """
    Handles streaming responses with proper error handling, cleanup, and timeouts.

    This class is designed to be thread-safe for concurrent state access through
    the use of an asyncio.Lock for state modifications.
    """

    def __init__(
        self,
        conversation_id: str,
        model_name: str,
        idle_timeout: int = STREAMING_IDLE_TIMEOUT,
        heartbeat_interval: int = HEARTBEAT_INTERVAL,
        max_response_size: int = MAX_RESPONSE_SIZE_BYTES,
        text_transform: Optional[callable] = None,
    ):
        """
        Initialize the streaming response handler.

        Args:
            conversation_id: ID of the conversation
            model_name: Name of the model being used
            idle_timeout: Timeout for idle connections in seconds
            heartbeat_interval: Interval for sending heartbeat messages
            max_response_size: Maximum response size in bytes
        """
        self.conversation_id = conversation_id
        self.model_name = model_name
        self.idle_timeout = idle_timeout
        self.heartbeat_interval = heartbeat_interval
        self.max_response_size = max_response_size
        self.last_activity = time.time()
        self.is_cancelled = False
        self.full_response: list[str] = []
        self.response_size = 0
        self.error_occurred = False
        # Optional transform to apply to textual deltas before emission (e.g., moderation redaction)
        self.text_transform = text_transform
        # Track whether a terminal [DONE] was already sent (directly or via transform-combined payload)
        self.done_sent = False
        # Track upstream DONE so we can defer the terminal sentinel until after metadata.
        self.upstream_done_received = False
        # Accumulate tool/function call deltas for persistence once the stream completes
        self.tool_call_accumulator: dict[int, dict[str, Any]] = {}
        self.tool_call_order: list[int] = []
        self.function_call_accumulator: Optional[dict[str, Any]] = None
        self.saved_message_id: Optional[str] = None
        self.system_message_id: Optional[str] = None
        self.continuation_metadata: Optional[dict[str, Any]] = None
        # Lock for thread-safe state modifications
        self._state_lock = asyncio.Lock()

    def _attach_stream_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return payload
        if not CHAT_STREAM_INCLUDE_METADATA:
            return payload
        if self.conversation_id:
            payload.setdefault("conversation_id", self.conversation_id)
            payload.setdefault("tldw_conversation_id", self.conversation_id)
        if self.system_message_id:
            payload.setdefault("tldw_system_message_id", self.system_message_id)
        if self.saved_message_id:
            payload.setdefault("tldw_message_id", self.saved_message_id)
        if self.continuation_metadata:
            payload.setdefault("tldw_continuation", self.continuation_metadata)
        return payload

    def _parse_save_callback_result(self, save_result: Any) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Normalize save-callback return payload.

        Supported return values:
        - `"message_id"` string (legacy)
        - `{"saved_message_id": "...", "events": [{"event": "...", "data": {...}}]}`
        """
        saved_message_id: Optional[str] = None
        extra_events: list[dict[str, Any]] = []

        if isinstance(save_result, str):
            normalized = save_result.strip()
            if normalized:
                saved_message_id = normalized
            return saved_message_id, extra_events

        if not isinstance(save_result, dict):
            return saved_message_id, extra_events

        raw_id = save_result.get("saved_message_id")
        if raw_id is None:
            raw_id = save_result.get("message_id")
        if isinstance(raw_id, str):
            normalized = raw_id.strip()
            if normalized:
                saved_message_id = normalized

        raw_events = save_result.get("events")
        if raw_events is None:
            raw_events = save_result.get("extra_events")
        if raw_events is None:
            raw_events = save_result.get("loop_events")
        if not isinstance(raw_events, list):
            return saved_message_id, extra_events

        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            raw_name = raw_event.get("event")
            event_name = raw_name.strip() if isinstance(raw_name, str) else ""
            if not event_name:
                continue
            if "data" not in raw_event:
                continue
            extra_events.append({"event": event_name, "data": raw_event.get("data")})

        return saved_message_id, extra_events

    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = time.time()

    def is_timed_out(self) -> bool:
        """Check if the stream has timed out due to inactivity."""
        return (time.time() - self.last_activity) > self.idle_timeout

    def cancel(self):
        """Mark the stream as cancelled."""
        self.is_cancelled = True
        logger.info(f"Stream cancelled for conversation {self.conversation_id}")

    def _accumulate_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Merge incremental tool call deltas into a final structure.

        This method includes bounds checking to prevent memory exhaustion from
        malformed tool call indices.
        """
        if not isinstance(tool_calls, list):
            return
        for idx, entry in enumerate(tool_calls):
            if not isinstance(entry, dict):
                continue
            call_index = entry.get("index")
            if call_index is None:
                call_index = idx
            try:
                call_index = int(call_index)
            except _STREAMING_NONCRITICAL_EXCEPTIONS:
                call_index = idx

            # Bounds check to prevent memory exhaustion
            if call_index < 0 or call_index > MAX_TOOL_CALL_INDEX:
                logger.warning(
                    f"Tool call index {call_index} out of bounds (0-{MAX_TOOL_CALL_INDEX}), skipping"
                )
                continue

            if call_index not in self.tool_call_accumulator:
                self.tool_call_accumulator[call_index] = {
                    "id": None,
                    "type": None,
                    "function": {"name": None, "arguments": ""},
                }
                self.tool_call_order.append(call_index)
            accumulator = self.tool_call_accumulator[call_index]
            if entry.get("id"):
                accumulator["id"] = entry["id"]
            if entry.get("type"):
                accumulator["type"] = entry["type"]
            function_delta = entry.get("function") or {}
            if function_delta.get("name"):
                accumulator["function"]["name"] = function_delta["name"]
            if function_delta.get("arguments"):
                new_args = function_delta["arguments"]
                current_len = len(accumulator["function"]["arguments"])
                # Enforce bounds on accumulated argument length to prevent OOM
                if current_len + len(new_args) > MAX_TOOL_ARGUMENT_LENGTH:
                    logger.warning(
                        f"Tool call arguments exceeded max length ({MAX_TOOL_ARGUMENT_LENGTH}), truncating"
                    )
                    # Truncate to fit within bounds
                    remaining = MAX_TOOL_ARGUMENT_LENGTH - current_len
                    if remaining > 0:
                        accumulator["function"]["arguments"] += new_args[:remaining]
                    # Skip further argument accumulation for this tool call
                else:
                    accumulator["function"]["arguments"] += new_args

    def _accumulate_function_call(self, function_delta: dict[str, Any]) -> None:
        """Merge incremental function call deltas into a final structure."""
        if not isinstance(function_delta, dict):
            return
        if self.function_call_accumulator is None:
            self.function_call_accumulator = {"name": None, "arguments": ""}
        if function_delta.get("name"):
            self.function_call_accumulator["name"] = function_delta["name"]
        if function_delta.get("arguments"):
            new_args = function_delta["arguments"]
            current_len = len(self.function_call_accumulator["arguments"])
            # Enforce bounds on accumulated argument length to prevent OOM
            if current_len + len(new_args) > MAX_TOOL_ARGUMENT_LENGTH:
                logger.warning(
                    f"Function call arguments exceeded max length ({MAX_TOOL_ARGUMENT_LENGTH}), truncating"
                )
                remaining = MAX_TOOL_ARGUMENT_LENGTH - current_len
                if remaining > 0:
                    self.function_call_accumulator["arguments"] += new_args[:remaining]
            else:
                self.function_call_accumulator["arguments"] += new_args

    def get_accumulated_tool_calls(self) -> Optional[list[dict[str, Any]]]:
        """Return the finalized list of tool calls, if any were streamed."""
        if not self.tool_call_accumulator:
            return None
        ordered_indices = sorted(set(self.tool_call_order))
        results: list[dict[str, Any]] = []
        for index in ordered_indices:
            data = self.tool_call_accumulator.get(index)
            if not data:
                continue
            function_block = data.get("function") or {}
            results.append(
                {
                    "id": data.get("id"),
                    "type": data.get("type"),
                    "function": {
                        "name": function_block.get("name"),
                        "arguments": function_block.get("arguments", ""),
                    },
                }
            )
        return results or None

    def get_accumulated_function_call(self) -> Optional[dict[str, Any]]:
        """Return the finalized function call payload, if one was streamed."""
        if not self.function_call_accumulator:
            return None
        name = self.function_call_accumulator.get("name")
        arguments = self.function_call_accumulator.get("arguments", "")
        if not name and not arguments:
            return None
        return {"name": name, "arguments": arguments}

    def has_accumulated_output(self) -> bool:
        """Return True when any text, tool calls, or function calls were gathered."""
        return bool(
            self.full_response
            or self.tool_call_accumulator
            or self.function_call_accumulator
        )

    async def heartbeat_generator(self) -> AsyncIterator[str]:
        """
        Generate heartbeat messages to keep the connection alive.

        Yields:
            SSE heartbeat messages
        """
        while not self.is_cancelled and not self.error_occurred:
            await asyncio.sleep(self.heartbeat_interval)
            if self.is_timed_out():
                logger.warning(f"Stream timeout for conversation {self.conversation_id}")
                self.cancel()
                yield f"data: {json.dumps({'error': {'message': 'Stream timeout - no activity'}})}\n\n"
                break
            yield f": heartbeat {datetime.now(timezone.utc).isoformat()}\n\n"

    async def safe_stream_generator(
        self,
        stream: Union[Iterator, AsyncIterator],
        save_callback: Optional[callable] = None,
        finalize_callback: Optional[callable] = None,
        before_success_callback: Optional[Callable[[], Any]] = None,
    ) -> AsyncIterator[str]:
        """
        Safely generate streaming responses with error handling and cleanup.

        Args:
            stream: The stream to process (sync or async iterator)
            save_callback: Optional callback to save the full response
            finalize_callback: Optional callback invoked on error/cancel to finalize state

        Yields:
            SSE formatted messages
        """
        try:
            # Send initial metadata
            start_payload = {
                "conversation_id": self.conversation_id,
                "model": self.model_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._attach_stream_metadata(start_payload)
            yield f"event: stream_start\ndata: {json.dumps(start_payload)}\n\n"
            self.update_activity()

            def iter_logical_lines(raw_chunk: str) -> list[str]:
                return raw_chunk.splitlines() if ("\n" in raw_chunk or raw_chunk.count("data:") > 1) else [raw_chunk]

            def append_content(text_piece: str) -> bool:
                if not text_piece:
                    return True
                chunk_size = len(text_piece.encode("utf-8"))
                if self.response_size + chunk_size > self.max_response_size:
                    return False
                # Also check list length to prevent unbounded item count
                if len(self.full_response) >= MAX_RESPONSE_LIST_LENGTH:
                    logger.warning(
                        f"Response list length exceeded max ({MAX_RESPONSE_LIST_LENGTH}) for {self.conversation_id}"
                    )
                    return False
                self.full_response.append(text_piece)
                self.response_size += chunk_size
                return True

            def process_line(raw_line: str) -> tuple[list[str], bool]:
                outputs: list[str] = []
                stripped_leading = raw_line.lstrip("\ufeff\u200b\u200c\u200d\u2060")
                candidate = stripped_leading.strip()
                if not candidate and not stripped_leading:
                    return outputs, False
                if candidate.startswith(":") or candidate.startswith("event:"):
                    return outputs, False
                if candidate.startswith("data:"):
                    payload_str = candidate[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        # Defer terminal DONE until after stream_end metadata is emitted.
                        self.upstream_done_received = True
                        self.update_activity()
                        return outputs, True
                    try:
                        data = json.loads(payload_str)
                    except _STREAMING_NONCRITICAL_EXCEPTIONS:
                        outputs.append(f"data: {payload_str}\n\n")
                        self.update_activity()
                        return outputs, False
                    if isinstance(data, dict) and "error" in data:
                        try:
                            err_payload = {"error": data.get("error")}
                            self._attach_stream_metadata(err_payload)
                            outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                        except _STREAMING_NONCRITICAL_EXCEPTIONS:
                            err_payload = {"error": {"message": "Upstream error"}}
                            self._attach_stream_metadata(err_payload)
                            outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                        self.error_occurred = True
                        return outputs, True
                    if isinstance(data, dict):
                        choices = data.get("choices")
                        if isinstance(choices, list) and choices:
                            for choice in choices:
                                delta = choice.get("delta")
                                # Be tolerant: providers/tests may send a plain string delta
                                if isinstance(delta, str):
                                    delta = {"content": delta}
                                # Guard against unexpected delta types
                                if not isinstance(delta, dict):
                                    delta = {}

                                tool_calls_delta = delta.get("tool_calls")
                                if tool_calls_delta:
                                    self._accumulate_tool_calls(tool_calls_delta)
                                function_call_delta = delta.get("function_call")
                                if function_call_delta:
                                    self._accumulate_function_call(function_call_delta)
                                if "content" in delta and delta["content"] is not None:
                                    text_piece = str(delta["content"])
                                    try:
                                        if self.text_transform:
                                            text_piece = self.text_transform(text_piece)
                                    except StopStreamWithError as stopper:
                                        err_payload = {
                                            "error": {
                                                "message": str(stopper) or "Stream blocked by policy",
                                                "type": stopper.error_type,
                                            }
                                        }
                                        self._attach_stream_metadata(err_payload)
                                        outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                                        self.error_occurred = True
                                        return outputs, True
                                    except StopIteration:
                                        return outputs, True
                                    except _STREAMING_NONCRITICAL_EXCEPTIONS as transform_err:
                                        logger.debug(f"text_transform error ignored: {transform_err}")
                                    if text_piece and not append_content(text_piece):
                                        err_payload = {"error": {"message": "Response size limit exceeded"}}
                                        self._attach_stream_metadata(err_payload)
                                        outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                                        self.error_occurred = True
                                        return outputs, True
                                    delta["content"] = text_piece
                            self._attach_stream_metadata(data)
                            outputs.append(f"data: {json.dumps(data)}\n\n")
                            self.update_activity()
                            return outputs, False
                    if isinstance(data, dict):
                        self._attach_stream_metadata(data)
                    outputs.append(f"data: {json.dumps(data)}\n\n")
                    self.update_activity()
                    return outputs, False
                # Non-SSE chunk: preserve spaces (avoid stripping)
                text_piece = stripped_leading
                with contextlib.suppress(_STREAMING_NONCRITICAL_EXCEPTIONS):
                    text_piece = str(text_piece)
                try:
                    if self.text_transform:
                        text_piece = self.text_transform(text_piece)
                except StopStreamWithError as stopper:
                    err_payload = {
                        "error": {
                            "message": str(stopper) or "Stream blocked by policy",
                            "type": stopper.error_type,
                        }
                    }
                    self._attach_stream_metadata(err_payload)
                    outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                    self.error_occurred = True
                    return outputs, True
                except StopIteration:
                    return outputs, True
                except _STREAMING_NONCRITICAL_EXCEPTIONS as transform_err:
                    logger.debug(f"text_transform error ignored: {transform_err}")
                if text_piece and not append_content(text_piece):
                    err_payload = {"error": {"message": "Response size limit exceeded"}}
                    self._attach_stream_metadata(err_payload)
                    outputs.append(f"data: {json.dumps(err_payload)}\n\n")
                    self.error_occurred = True
                    return outputs, True
                if text_piece:
                    content_payload = {"choices": [{"delta": {"content": text_piece}}]}
                    self._attach_stream_metadata(content_payload)
                    outputs.append(f"data: {json.dumps(content_payload)}\n\n")
                    self.update_activity()
                return outputs, False

            # Process the stream
            async_stream = stream if hasattr(stream, '__aiter__') else None
            if async_stream is None and STREAMING_SYNC_BRIDGE_ENABLED:
                async_stream = _async_iter_sync_stream(stream)

            if async_stream is not None:
                # Async iterator (native or bridged from sync)
                async for chunk in async_stream:
                    if self.is_cancelled:
                        logger.info(f"Stream processing cancelled for {self.conversation_id}")
                        break

                    if self.is_timed_out():
                        logger.warning(f"Stream timeout during processing for {self.conversation_id}")
                        yield f"data: {json.dumps({'error': {'message': 'Stream timeout'}})}\n\n"
                        break

                    try:
                        raw_str = chunk.decode('utf-8', errors='replace') if isinstance(chunk, bytes) else str(chunk)
                        stop_stream = False
                        for logical_line in iter_logical_lines(raw_str):
                            outputs, should_stop = process_line(logical_line)
                            for out in outputs:
                                yield out
                            if should_stop:
                                stop_stream = True
                                break
                        if stop_stream:
                            break
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Error processing stream chunk for {self.conversation_id}: {e}")
                        self.error_occurred = True
                        err_payload = {"error": {"message": f"Error processing chunk: {str(e)}"}}
                        self._attach_stream_metadata(err_payload)
                        yield f"data: {json.dumps(err_payload)}\n\n"
                        break
            else:
                # Sync iterator (legacy, blocks event loop) - deprecated path
                logger.warning(
                    f"Using blocking sync iterator for {self.conversation_id}. "
                    "Set STREAMING_SYNC_BRIDGE_ENABLED=true for non-blocking behavior."
                )
                def sync_iterator():
                    try:
                        for chunk in stream:
                            if self.is_cancelled:
                                break
                            yield chunk
                    except StopIteration:
                        pass

                for chunk in sync_iterator():
                    if self.is_cancelled:
                        break

                    if self.is_timed_out():
                        logger.warning(f"Stream timeout during sync processing for {self.conversation_id}")
                        yield f"data: {json.dumps({'error': {'message': 'Stream timeout'}})}\n\n"
                        break

                    try:
                        raw_str = chunk.decode('utf-8', errors='replace') if isinstance(chunk, bytes) else str(chunk)
                        stop_stream = False
                        for logical_line in iter_logical_lines(raw_str):
                            outputs, should_stop = process_line(logical_line)
                            for out in outputs:
                                yield out
                            if should_stop:
                                stop_stream = True
                                break
                        if stop_stream:
                            break
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Error processing sync stream chunk for {self.conversation_id}: {e}")
                        self.error_occurred = True
                        err_payload = {"error": {"message": f"Error processing chunk: {str(e)}"}}
                        self._attach_stream_metadata(err_payload)
                        yield f"data: {json.dumps(err_payload)}\n\n"
                        break

        except asyncio.CancelledError:
            # Client disconnected
            logger.info(f"Client disconnected from stream for {self.conversation_id}")
            self.cancel()
        except GeneratorExit:
            # Generator is being closed; do not yield anything here
            logger.info(f"Stream generator closed for {self.conversation_id}")
            self.cancel()
            # Re-raise to ensure proper generator closure semantics
            raise
        except _STREAMING_NONCRITICAL_EXCEPTIONS as e:
            # Unexpected error
            logger.error(f"Unexpected error in stream for {self.conversation_id}: {e}", exc_info=True)
            # Best-effort: flush any buffered tail before emitting the error frame.
            # This preserves earlier valid chunks when the upstream fails mid-stream.
            if self.text_transform:
                flush_fn = getattr(self.text_transform, "flush", None)
                if callable(flush_fn):
                    try:
                        flush_text = flush_fn()
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as flush_err:
                        logger.debug(f"text_transform flush on error ignored: {flush_err}")
                        flush_text = None
                    if flush_text:
                        try:
                            flush_text = str(flush_text)
                        except _STREAMING_NONCRITICAL_EXCEPTIONS:
                            flush_text = ""
                        if flush_text:
                            if append_content(flush_text):
                                content_payload = {"choices": [{"delta": {"content": flush_text}}]}
                                self._attach_stream_metadata(content_payload)
                                yield f"data: {json.dumps(content_payload)}\n\n"
                                self.update_activity()
                            else:
                                size_err = {"error": {"message": "Response size limit exceeded"}}
                                self._attach_stream_metadata(size_err)
                                self.error_occurred = True
                                yield f"data: {json.dumps(size_err)}\n\n"
                                return
            self.error_occurred = True
            err_payload = {"error": {"message": f"Stream error: {str(e)}"}}
            self._attach_stream_metadata(err_payload)
            yield f"data: {json.dumps(err_payload)}\n\n"

        finally:
            # Cleanup and final message
            try:
                # Always attempt to close the upstream stream first
                try:
                    if hasattr(stream, "aclose") and callable(stream.aclose):
                        # Async generator
                        await stream.aclose()  # type: ignore[attr-defined]
                    elif hasattr(stream, "close") and callable(stream.close):
                        # Sync generator
                        stream.close()  # type: ignore[attr-defined]
                except _STREAMING_NONCRITICAL_EXCEPTIONS as cleanup_err:
                    # Log cleanup errors for debugging, but don't propagate
                    logger.debug(
                        f"Stream cleanup warning for {self.conversation_id}: {cleanup_err}"
                    )

                # If cancelled (e.g., client disconnect or generator close), do not yield or await further
                if self.is_cancelled:
                    if finalize_callback and (self.is_cancelled or self.error_occurred):
                        try:
                            maybe_result = finalize_callback(
                                success=False,
                                cancelled=True,
                                error=self.error_occurred,
                            )
                            if hasattr(maybe_result, "__await__"):
                                await maybe_result
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as finalize_err:
                            logger.debug(f"Finalize callback error after cancel: {finalize_err}")
                    return  # noqa: B012

                # Flush any pending tail from text_transform (e.g., moderation holdback)
                if not self.error_occurred and self.text_transform:
                    flush_fn = getattr(self.text_transform, "flush", None)
                    if callable(flush_fn):
                        try:
                            flush_text = flush_fn()
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as flush_err:
                            logger.debug(f"text_transform flush error ignored: {flush_err}")
                            flush_text = None
                        if flush_text:
                            try:
                                flush_text = str(flush_text)
                            except _STREAMING_NONCRITICAL_EXCEPTIONS:
                                flush_text = ""
                            if flush_text:
                                if not append_content(flush_text):
                                    err_payload = {"error": {"message": "Response size limit exceeded"}}
                                    self._attach_stream_metadata(err_payload)
                                    yield f"data: {json.dumps(err_payload)}\n\n"
                                    self.error_occurred = True
                                else:
                                    content_payload = {"choices": [{"delta": {"content": flush_text}}]}
                                    self._attach_stream_metadata(content_payload)
                                    yield f"data: {json.dumps(content_payload)}\n\n"
                                    self.update_activity()

                # Save the full response/tool calls if callback provided (only when not cancelled)
                has_output = self.has_accumulated_output()
                if (
                    not self.is_cancelled
                    and not self.error_occurred
                ):
                    if before_success_callback and not self.is_cancelled:
                        try:
                            maybe_before_success = before_success_callback()
                            if hasattr(maybe_before_success, "__await__"):
                                await maybe_before_success
                        except StopStreamWithError as stopper:
                            err_payload = {
                                "error": {
                                    "message": str(stopper) or "Stream blocked by policy",
                                    "type": stopper.error_type,
                                }
                            }
                            self._attach_stream_metadata(err_payload)
                            self.error_occurred = True
                            yield f"data: {json.dumps(err_payload)}\n\n"
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as before_success_err:
                            logger.error(
                                f"Before-success callback error for {self.conversation_id}: {before_success_err}"
                            )
                            self.error_occurred = True
                            err_payload = {
                                "error": {
                                    "message": f"Stream error: {str(before_success_err)}",
                                    "type": "stream_error",
                                }
                            }
                            self._attach_stream_metadata(err_payload)
                            yield f"data: {json.dumps(err_payload)}\n\n"

                if (
                    not self.is_cancelled
                    and save_callback
                    and not self.error_occurred
                    and has_output
                ):
                    full_text = "".join(self.full_response)
                    aggregated_tool_calls = self.get_accumulated_tool_calls()
                    aggregated_function_call = self.get_accumulated_function_call()
                    extra_events: list[dict[str, Any]] = []
                    try:
                        # Support flexible callback signatures (text only or extended)
                        maybe_result = None
                        save_result = None
                        try:
                            maybe_result = save_callback(
                                full_text,
                                aggregated_tool_calls,
                                aggregated_function_call,
                            )
                        except TypeError:
                            maybe_result = save_callback(full_text)
                        if hasattr(maybe_result, "__await__"):
                            save_result = await maybe_result
                        else:
                            save_result = maybe_result
                        parsed_message_id, parsed_events = self._parse_save_callback_result(save_result)
                        if parsed_message_id:
                            self.saved_message_id = parsed_message_id
                        extra_events = parsed_events
                        logger.info(
                            'Saved streaming response for {} (text_len={}, tool_calls={}, function_call={}, events={})',
                            self.conversation_id,
                            len(full_text),
                            len(aggregated_tool_calls or []),
                            "yes" if aggregated_function_call else "no",
                            len(extra_events),
                        )
                    except Exception as e:
                        logger.error(f"Failed to save streaming response for {self.conversation_id}: {e}")
                        extra_events = []

                    for event_entry in extra_events:
                        event_name = str(event_entry.get("event") or "").strip()
                        if not event_name:
                            continue
                        payload_obj = event_entry.get("data")
                        if payload_obj is None:
                            continue
                        try:
                            if isinstance(payload_obj, dict):
                                payload = dict(payload_obj)
                                self._attach_stream_metadata(payload)
                                payload_json = json.dumps(payload)
                            else:
                                payload_json = json.dumps(payload_obj, default=str)
                            yield f"event: {event_name}\ndata: {payload_json}\n\n"
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as event_err:
                            logger.debug(
                                f"Skipping extra stream event {event_name} for {self.conversation_id}: {event_err}"
                            )

                # Send completion marker(s) after save so metadata includes IDs.
                if not self.error_occurred:
                    done_payload = {
                        "id": f"chatcmpl-{datetime.now(timezone.utc).timestamp()}",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "model": self.model_name,
                        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                    }
                    self._attach_stream_metadata(done_payload)
                    yield f"data: {json.dumps(done_payload)}\n\n"

                if finalize_callback and self.error_occurred:
                    try:
                        maybe_result = finalize_callback(
                            success=False,
                            cancelled=False,
                            error=True,
                        )
                        if hasattr(maybe_result, "__await__"):
                            await maybe_result
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as finalize_err:
                        logger.debug(f"Finalize callback error after stream error: {finalize_err}")

                # Send stream end event (only when not cancelled)
                if not self.is_cancelled:
                    end_payload = {
                        "conversation_id": self.conversation_id,
                        "success": not self.error_occurred,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self._attach_stream_metadata(end_payload)
                    yield f"event: stream_end\ndata: {json.dumps(end_payload)}\n\n"
                # Ensure final [DONE] sentinel for client compatibility (unless already sent).
                # If upstream already sent [DONE], defer emission until after stream_end.
                if self.upstream_done_received and not self.done_sent or not self.done_sent:
                    yield "data: [DONE]\n\n"
                    self.done_sent = True
                self.upstream_done_received = False

            except _STREAMING_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error in stream cleanup for {self.conversation_id}: {e}")


async def create_streaming_response_with_timeout(
    stream: Union[Iterator, AsyncIterator],
    conversation_id: str,
    model_name: str,
    save_callback: Optional[callable] = None,
    finalize_callback: Optional[callable] = None,
    before_success_callback: Optional[Callable[[], Any]] = None,
    idle_timeout: int = STREAMING_IDLE_TIMEOUT,
    heartbeat_interval: int = HEARTBEAT_INTERVAL,
    text_transform: Optional[callable] = None,
    system_message_id: Optional[str] = None,
    continuation_metadata: Optional[dict[str, Any]] = None,
) -> AsyncIterator[str]:
    """
    Create a streaming response with timeout and error handling.

    Args:
        stream: The stream to process
        conversation_id: ID of the conversation
        model_name: Name of the model
        save_callback: Optional callback to save the response
        finalize_callback: Optional callback invoked on error/cancel to finalize state
        idle_timeout: Timeout for idle connections
        heartbeat_interval: Interval for heartbeat messages
        system_message_id: Optional system message ID to echo in stream_end payload
        continuation_metadata: Optional continuation metadata to attach to stream payloads

    Yields:
        SSE formatted messages
    """
    handler = StreamingResponseHandler(
        conversation_id=conversation_id,
        model_name=model_name,
        idle_timeout=idle_timeout,
        heartbeat_interval=heartbeat_interval,
        text_transform=text_transform,
    )
    handler.system_message_id = system_message_id
    if isinstance(continuation_metadata, dict) and continuation_metadata:
        handler.continuation_metadata = dict(continuation_metadata)

    # Create tasks for streaming and optional heartbeat using persistent generator instances
    async def stream_with_heartbeat():
        stream_gen = handler.safe_stream_generator(
            stream,
            save_callback,
            finalize_callback,
            before_success_callback,
        )
        heartbeats_enabled = isinstance(heartbeat_interval, (int, float)) and heartbeat_interval > 0
        heartbeat_gen = handler.heartbeat_generator() if heartbeats_enabled else None

        stream_task: Optional[asyncio.Task] = asyncio.create_task(stream_gen.__anext__())
        heartbeat_task: Optional[asyncio.Task] = (
            asyncio.create_task(heartbeat_gen.__anext__()) if heartbeats_enabled and heartbeat_gen is not None else None
        )

        try:
            while not handler.is_cancelled and (stream_task is not None or heartbeat_task is not None):
                if handler.error_occurred and heartbeat_task is not None:
                    if not heartbeat_task.done():
                        heartbeat_task.cancel()
                    heartbeat_task = None
                wait_set = {t for t in (stream_task, heartbeat_task) if t is not None}
                if not wait_set:
                    break
                done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                should_exit = False
                for task in done:
                    try:
                        result = task.result()
                        if task is stream_task:
                            # Stream chunk
                            if result is not None:
                                yield result
                            # Schedule next chunk
                            stream_task = asyncio.create_task(stream_gen.__anext__())
                        elif heartbeat_task is not None and task is heartbeat_task:
                            # Heartbeat
                            if result is not None:
                                yield result
                            # Schedule next heartbeat
                            if heartbeats_enabled and heartbeat_gen is not None:
                                heartbeat_task = asyncio.create_task(heartbeat_gen.__anext__())
                    except StopAsyncIteration:
                        # A generator ended naturally; exit the loop without flagging cancel
                        should_exit = True
                    except asyncio.CancelledError:
                        # Task was cancelled (likely due to shutdown); exit loop
                        should_exit = True
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Error in streaming task: {e}")
                        handler.error_occurred = True
                        should_exit = True

                # Do not cancel pending tasks on normal loop progression; keep them running

                if should_exit:
                    # Also cancel the latest scheduled tasks in case we created replacements
                    for t in (stream_task, heartbeat_task):
                        if t is not None and not t.done():
                            t.cancel()
                    gather_targets = tuple(filter(None, (stream_task, heartbeat_task)))
                    if gather_targets:
                        try:
                            await asyncio.gather(*gather_targets, return_exceptions=True)
                        except _STREAMING_NONCRITICAL_EXCEPTIONS as gather_err:
                            logger.debug(f"Task gather cleanup: {gather_err}")
                    # As a safety net, emit a final [DONE] only if it hasn't been sent yet
                    try:
                        if not handler.done_sent and not handler.is_cancelled:
                            yield "data: [DONE]\n\n"
                            handler.done_sent = True
                    except _STREAMING_NONCRITICAL_EXCEPTIONS as done_err:
                        logger.debug(f"Final DONE emission error: {done_err}")
                    break
        finally:
            # Ensure any pending tasks are cancelled and awaited exactly once
            remaining_tasks = [t for t in (stream_task, heartbeat_task) if t is not None]
            for task in remaining_tasks:
                if not task.done():
                    task.cancel()
            if remaining_tasks:
                try:
                    await asyncio.gather(*remaining_tasks, return_exceptions=True)
                except _STREAMING_NONCRITICAL_EXCEPTIONS as final_gather_err:
                    logger.debug(f"Final task cleanup: {final_gather_err}")
            # Ensure generators are properly closed; avoid yielding here
            try:
                await stream_gen.aclose()
            except _STREAMING_NONCRITICAL_EXCEPTIONS as stream_close_err:
                logger.debug(f"Stream generator close: {stream_close_err}")
            if heartbeat_gen is not None:
                try:
                    await heartbeat_gen.aclose()
                except _STREAMING_NONCRITICAL_EXCEPTIONS as heartbeat_close_err:
                    logger.debug(f"Heartbeat generator close: {heartbeat_close_err}")

    async for message in stream_with_heartbeat():
        yield message


class StopStreamWithError(Exception):
    """Signal the streaming handler to stop after emitting an SSE error payload."""
    def __init__(self, message: str = "Stream blocked by policy", error_type: str = "stream_error"):
        super().__init__(message)
        self.error_type = error_type


#
# End of streaming_utils.py
#######################################################################################################################
