"""Branch prompt runner seam for chat macro execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .context_snapshot import MacroContextSnapshot


class BranchPromptResult(BaseModel):
    """Normalized result from one branch prompt execution."""

    model_config = ConfigDict(extra="forbid")

    text: str = ""
    status: str = "completed"
    citations: list[Any] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    acp_child_session_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class BranchPromptRunner(Protocol):
    """Protocol for fakeable branch prompt execution."""

    async def run_branch(
        self,
        *,
        prompt: str,
        snapshot: MacroContextSnapshot,
        model_selection: dict[str, Any],
    ) -> BranchPromptResult:
        """Run one branch prompt against a stable context snapshot."""


ChatCall = Callable[..., Awaitable[Any]]

_SYSTEM_PROMPT = (
    "You are executing one analysis branch of a chat macro. Use only the "
    "provided conversation snapshot, answer the branch task directly, do not "
    "request tools or side effects, and keep the response self-contained."
)


class ChatMacroLLMBranchRunner:
    """Execute chat-native macro branches through the canonical LLM service."""

    def __init__(self, *, chat_call: ChatCall | None = None, max_tokens: int = 1000) -> None:
        self._chat_call = chat_call
        self._max_tokens = max(1, min(int(max_tokens), 4000))

    async def run_branch(
        self,
        *,
        prompt: str,
        snapshot: MacroContextSnapshot,
        model_selection: dict[str, Any],
    ) -> BranchPromptResult:
        chat_call = self._chat_call
        if chat_call is None:
            from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

            chat_call = perform_chat_api_call_async

        call_kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": _branch_user_prompt(snapshot, prompt)}],
            "system_message": _SYSTEM_PROMPT,
            "max_tokens": self._max_tokens,
            "temperature": 0.2,
            "stream": False,
        }
        provider = model_selection.get("api_provider") or model_selection.get("provider")
        model = model_selection.get("model")
        if provider:
            call_kwargs["api_provider"] = str(provider)
        if model:
            call_kwargs["model"] = str(model)

        response = await chat_call(**call_kwargs)
        text = _extract_response_text(response)
        if not text:
            return BranchPromptResult(
                status="failed",
                error_code="empty_completion",
                error_message="The model returned an empty branch completion.",
            )
        usage = response.get("usage") if isinstance(response, dict) else None
        return BranchPromptResult(
            text=text,
            usage=dict(usage) if isinstance(usage, dict) else {},
        )


def _branch_user_prompt(snapshot: MacroContextSnapshot, branch_prompt: str) -> str:
    transcript = "\n".join(
        f"{str(message.get('role') or 'unknown')}: {str(message.get('excerpt') or '')}"
        for message in snapshot.messages
        if message.get("excerpt")
    )
    if not transcript:
        transcript = "(No conversation messages were captured.)"
    return f"Conversation snapshot:\n{transcript}\n\nBranch task:\n{branch_prompt.strip()}"


def _extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"].strip()
    for key in ("content", "text"):
        if isinstance(response.get(key), str):
            return response[key].strip()
    return ""
