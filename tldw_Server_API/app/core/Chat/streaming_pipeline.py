# streaming_pipeline.py
# Description: Thin streaming response assembly wrapper for Chat completions.
"""
Compatibility wrapper for constructing Chat streaming responses.

The streaming implementation remains in ``streaming_utils``. This module gives
``chat_service`` a narrower assembly boundary without changing SSE behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class StreamingPipelineRequest:
    """Inputs needed to construct the Chat SSE stream."""

    stream: Any
    conversation_id: str
    model_name: str
    save_callback: Callable[..., Any] | None = None
    finalize_callback: Callable[..., Any] | None = None
    on_first_output: Callable[..., Any] | None = None
    idle_timeout: float | None = None
    heartbeat_interval: float | None = None
    text_transform: Callable[[str], str] | None = None
    before_success_callback: Callable[..., Any] | None = None
    system_message_id: str | None = None
    continuation_metadata: dict[str, Any] | None = None


def create_chat_streaming_response(
    *,
    request: StreamingPipelineRequest,
    stream_factory: Callable[..., Any],
) -> Any:
    """Create a streaming response through the injected stream factory."""
    factory_kwargs: dict[str, Any] = {
        "stream": request.stream,
        "conversation_id": request.conversation_id,
        "model_name": request.model_name,
    }
    optional_kwargs = {
        "save_callback": request.save_callback,
        "finalize_callback": request.finalize_callback,
        "on_first_output": request.on_first_output,
        "idle_timeout": request.idle_timeout,
        "heartbeat_interval": request.heartbeat_interval,
        "text_transform": request.text_transform,
        "before_success_callback": request.before_success_callback,
        "system_message_id": request.system_message_id,
        "continuation_metadata": request.continuation_metadata,
    }
    factory_kwargs.update(
        {
            key: value
            for key, value in optional_kwargs.items()
            if value is not None
        }
    )
    return stream_factory(**factory_kwargs)


__all__ = [
    "StreamingPipelineRequest",
    "create_chat_streaming_response",
]
