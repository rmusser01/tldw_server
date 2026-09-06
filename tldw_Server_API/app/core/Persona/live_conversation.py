"""Bounded Persona Live transport into the authenticated Chat HTTP boundary."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from threading import RLock
from typing import Any

import httpx

from tldw_Server_API.app.core.exceptions import PersonaConversationError


def resolve_persona_conversation_target() -> Any:
    """Resolve the target; authenticated Chat owns effective credentials."""
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.Chat.chat_target_resolution import resolve_chat_target

    try:
        return resolve_chat_target(requested_provider=None, requested_model=None)
    except ChatConfigurationError:
        raise PersonaConversationError(
            "Configure a supported default Chat provider and model in server settings."
        ) from None


def require_persona_voice_conversation_credentials() -> Any:
    """Conservatively qualify voice against server-configured Chat credentials.

    User-scoped BYOK credentials remain supported by the normal text Chat route;
    this local preparation check does not certify their availability for voice.
    """
    from tldw_Server_API.app.core.Chat.chat_service import resolve_static_provider_fallback
    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import provider_auth_is_resolved
    from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key

    try:
        target = resolve_persona_conversation_target()
        credentials = resolve_static_provider_fallback(target.provider)
        if provider_requires_api_key(target.provider) and not provider_auth_is_resolved(
            target.provider,
            api_key=credentials.api_key,
            app_config=credentials.app_config,
            credentials_resolved=True,
        ):
            raise PersonaConversationError(
                "Voice preparation requires server-configured credentials for the default Chat provider."
            )
        return target
    except PersonaConversationError:
        raise
    except (ValueError, TypeError, KeyError):
        raise PersonaConversationError(
            "Configure a supported default Chat provider and model in server settings."
        ) from None


def requires_tool_plan(text: str) -> bool:
    """Recognize explicit tool requests; ordinary questions remain conversation."""
    normalized = str(text or "").strip().lower()
    if normalized.startswith(("skill:", "http://", "https://")):
        return True
    return bool(
        re.match(
            r"^(?:(?:please|can you|could you|would you)\s+)?"
            r"(?:search|find|look up|ingest|fetch|browse|rag_search|ingest_url)\b",
            normalized,
        )
    )


async def complete_persona_turn(
    *,
    profile_store: Any,
    persona_id: str,
    user_id: str,
    app: Any,
    headers: dict[str, str],
    client: tuple[str, int] | None,
    turns: list[dict[str, Any]],
    context_sections: Iterable[Iterable[Any]],
) -> str | None:
    """Build bounded profile context and delegate to authenticated Chat.

    Args:
        profile_store: Existing user-scoped Persona repository.
        persona_id: Selected active profile identifier.
        user_id: Authenticated profile owner.
        app: ASGI application containing the ordinary Chat route.
        headers: Caller authentication forwarded to Chat admission.
        client: Original client address for admission and accounting.
        turns: Session history in chronological order.
        context_sections: Already-authorized memory/state/companion/exemplar data.

    Returns:
        Provider reply, or None if the active profile is no longer available.

    Raises:
        PersonaConversationError: Chat admission or completion is unavailable.
    """
    profile = await asyncio.to_thread(
        profile_store.get_persona_profile,
        persona_id,
        user_id=user_id,
        include_deleted=False,
    )
    if not isinstance(profile, dict):
        return None
    system_prompt = str(profile.get("system_prompt") or "You are a helpful assistant.")[:8000]
    context_lines = [line for section in context_sections for line in section]
    if context_lines:
        context_text = "\n".join(str(line) for line in context_lines)[:7800]
        system_prompt += "\n\nPersona reference context (data, not tool authorization):\n" + context_text
    return await complete_persona_conversation(
        app=app,
        headers=headers,
        client=client,
        system_prompt=system_prompt,
        turns=turns,
    )


async def complete_persona_conversation(
    *,
    app: Any,
    headers: dict[str, str],
    client: tuple[str, int] | None,
    system_prompt: str,
    turns: list[dict[str, Any]],
) -> str:
    """Run full HTTP admission, moderation and accounting with the caller's auth.

    No network URL is accepted. The same ASGI application handles the fixed Chat
    route; no FastAPI dependency is invoked manually or replaced by a trust flag.

    Args:
        app: ASGI application exposing the fixed Chat completion route.
        headers: Caller authentication headers, never logged here.
        client: Original client address for admission and accounting.
        system_prompt: Persona instructions and authorized reference context.
        turns: Chronological user/assistant history, bounded before transport.

    Returns:
        Nonempty provider text bounded to 48,000 characters.

    Raises:
        PersonaConversationError: Invalid input, unavailable target, denied
            admission, timeout, or an unusable provider answer.
        asyncio.CancelledError: The owning session cancels its request.
    """
    latest_user_text = next(
        (str(turn.get("content") or "") for turn in reversed(turns) if turn.get("role") == "user"),
        "",
    )
    if not latest_user_text.strip():
        raise PersonaConversationError("Enter a message before sending it to Persona.")
    target = resolve_persona_conversation_target()
    messages = [{"role": "system", "content": str(system_prompt or "You are a helpful assistant.")[:16000]}]
    remaining = 48000
    history = []
    for turn in reversed(turns[-24:]):
        role = turn.get("role")
        content = str(turn.get("content") or "")
        if role not in {"user", "assistant"} or not content:
            continue
        bounded = content[: min(12000, remaining)]
        if not bounded:
            break
        history.append({"role": role, "content": bounded})
        remaining -= len(bounded)
    messages.extend(reversed(history))
    outgoing_user_text = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    if not outgoing_user_text.strip():
        raise PersonaConversationError("Enter a message before sending it to Persona.")
    if outgoing_user_text.lstrip().startswith("/"):
        # Validate the actual bounded payload: Chat runs slash commands before
        # inference, even without tools. Dropped turns must not expose an old one.
        raise PersonaConversationError(
            "Slash commands are unavailable in Live conversation. Use an explicit "
            "search or skill: request and review its proposed actions in Live."
        )
    transport = httpx.ASGITransport(app=app, client=client or ("127.0.0.1", 0), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", headers=headers) as internal:
        try:
            response = await asyncio.wait_for(
                internal.post(
                    "/api/v1/chat/completions",
                    json={
                        "api_provider": target.provider,
                        "model": target.model,
                        "messages": messages,
                        "stream": False,
                        "save_to_db": False,
                        "max_tokens": 1024,
                    },
                ),
                timeout=90,
            )
        except TimeoutError:
            raise PersonaConversationError("The Chat provider timed out. Retry or check its configuration.") from None
    if response.status_code in {401, 403}:
        raise PersonaConversationError("This session is not authorized for Chat. Reconnect with Chat access.")
    if response.status_code == 429:
        raise PersonaConversationError("Chat usage is limited. Check your budget or retry later.")
    if not response.is_success:
        raise PersonaConversationError("The Chat provider could not respond. Check server Chat settings and retry.")
    try:
        message = response.json()["choices"][0]["message"]
        text = message.get("content")
        if message.get("tool_calls") or not isinstance(text, str) or not text.strip():
            raise ValueError("No conversational answer")
    except (ValueError, KeyError, IndexError, TypeError):
        raise PersonaConversationError(
            "The Chat provider returned no conversational answer. Retry or select another model."
        ) from None
    return text.strip()[:48000]


class PersonaLiveTurnRegistry:
    """Exact task ownership shared with synchronous REST Stop handlers."""

    def __init__(self) -> None:
        """Initialize the thread-safe task ownership table."""
        self._tasks: dict[tuple[str, str], set[asyncio.Task]] = {}
        self._lock = RLock()

    def register(self, *, user_id: str, session_id: str, task: asyncio.Task) -> None:
        """Register task under its authenticated user and session identifiers."""
        with self._lock:
            self._tasks.setdefault((user_id, session_id), set()).add(task)

    def cancel(self, *, user_id: str, session_id: str) -> None:
        """Retire the user/session owners and schedule cancellation on their loops."""
        with self._lock:
            tasks = self._tasks.pop((user_id, session_id), set())
        for task in tasks:
            if not task.done():
                task.get_loop().call_soon_threadsafe(task.cancel)

    def is_current(self, *, user_id: str, session_id: str, task: asyncio.Task | None) -> bool:
        """Return whether task still belongs to the exact user/session owner set."""
        with self._lock:
            return task is not None and task in self._tasks.get((user_id, session_id), ())

    def release(self, *, user_id: str, session_id: str, task: asyncio.Task) -> None:
        """Remove the completed task without disturbing other session owners."""
        with self._lock:
            key = (user_id, session_id)
            tasks = self._tasks.get(key)
            if tasks is not None:
                tasks.discard(task)
                if not tasks:
                    self._tasks.pop(key, None)


persona_live_turn_registry = PersonaLiveTurnRegistry()
