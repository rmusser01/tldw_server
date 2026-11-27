"""Moderation hooks for the LLM gateway (TASK-13950.27 PR C).

``register_moderation_gateway_hooks()`` adds one pre-call and one post-call hook to
``core.LLM_Calls.gateway``. They moderate the input messages and the model output of every
surface that is not skipped (see ``resolve_policy``). The text functions are public so surfaces
without a gateway call (Tasks 5, 9 and 10) moderate the same way.

Rules this module keeps:

- Hooks never log the request (it carries ``api_key``); a log line carries phase, action,
  surface, provider and the integer user id, never message text.
- A post hook never raises. An evaluator failure on output is treated as a block (fail closed).
- When moderation changes a chunk or a response, ``provider_response`` is dropped from it,
  because it carries the original text.
- Not moderated (documented in Task 11): non-text content parts, tool-call arguments,
  ``reasoning_content``, ``system`` messages and anything on a chunk without ``choices``.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls.gateway import GatewayCall, add_policy_hooks
from tldw_Server_API.app.core.LLM_Calls.sse import sse_data, sse_done
from tldw_Server_API.app.core.Moderation.moderation_service import get_moderation_service
from tldw_Server_API.app.core.Moderation.surfaces import (
    CALLER_MODERATED_SURFACES,
    EXEMPT_SURFACES,
    UNCLASSIFIED_SURFACE,
)

INPUT_REFUSAL = "Input violates moderation policy"
OUTPUT_REFUSAL = "Output violates moderation policy"
INPUT_UNCHECKED_REFUSAL = "Input could not be checked against moderation policy"
_MODERATED_ROLES = frozenset({"user", "tool"})  # chat's MODERATED_ROLES
_DEFAULT_BUFFER_CHARS = 1024
_SKIP = object()  # stored on the call: resolved, nothing to moderate
_UNRESOLVED = object()
_IN_HOOK: contextvars.ContextVar[bool] = contextvars.ContextVar("moderation_gateway_hook_active", default=False)
# Chunk keys that are metadata only: a chunk with just these and an empty delta carries nothing.
_CHUNK_METADATA_KEYS = frozenset({"choices", "id", "object", "created", "model", "system_fingerprint", "provider_response"})


@contextlib.contextmanager
def _evaluating() -> Iterator[None]:
    """Mark this context as inside a hook, so a gateway call made while evaluating is not moderated."""
    token = _IN_HOOK.set(True)
    try:
        yield
    finally:
        _IN_HOOK.reset(token)


def _log_action(phase: str, action: str, surface: str | None, provider: str | None, user_id: int | None) -> None:
    if action != "pass":
        logger.warning(
            "moderation {} {} surface={} provider={} user_id={}", phase, action, surface, provider, user_id
        )


# --------------------------------------------------------------------------- config


@functools.lru_cache(maxsize=1)
def _stream_buffer_chars() -> int:
    """``[Moderation] stream_buffer_chars``, read once per process; a bad value warns and uses 1024."""
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config
        from tldw_Server_API.app.core.config_sections.moderation import load_moderation_surface_config

        return load_moderation_surface_config(load_comprehensive_config()).stream_buffer_chars
    except Exception as exc:  # a bad value must not stop moderation; the service logs the rest
        logger.warning("Invalid [Moderation] stream_buffer_chars ({}); using {}", type(exc).__name__, _DEFAULT_BUFFER_CHARS)
        return _DEFAULT_BUFFER_CHARS


# --------------------------------------------------------------------------- guardian

# ponytail: one SupervisedPolicyEngine per user id, never evicted (it holds a compiled-rule cache with
# its own 60 s TTL); the ceiling is an unbounded number of distinct users, and a user whose Guardian DB is
# deleted keeps a stale engine. Upgrade path: an LRU with a size cap.
_GUARDIAN_ENGINES: dict[int, Any] = {}
_GUARDIAN_LOCK = threading.Lock()


def _guardian_engine(user_id: int) -> Any | None:
    """The user's cached engine, or None when Guardian is off or the user has no Guardian DB (none is created)."""
    from tldw_Server_API.app.core.feature_flags import is_guardian_enabled

    if not is_guardian_enabled():
        return None
    with _GUARDIAN_LOCK:
        engine = _GUARDIAN_ENGINES.get(user_id)
    if engine is not None:
        return engine
    try:
        from tldw_Server_API.app.core.DB_Management.guardian_db_resolver import resolve_existing_guardian_db_for_user_id
        from tldw_Server_API.app.core.Moderation.supervised_policy import get_supervised_policy_engine

        engine = get_supervised_policy_engine(resolve_existing_guardian_db_for_user_id(user_id))
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("Guardian overlay unavailable: {}", type(exc).__name__)
        return None
    with _GUARDIAN_LOCK:
        return _GUARDIAN_ENGINES.setdefault(user_id, engine)


def _with_guardian(policy: Any, user_id: int | None) -> Any:
    if user_id is None:
        return policy
    engine = _guardian_engine(user_id)
    if engine is None:
        return policy
    try:
        return engine.build_moderation_policy_overlay(str(user_id), policy, chat_type="regular")
    except Exception as exc:  # an engine failure keeps the base policy
        logger.debug("Guardian policy overlay skipped: {}", type(exc).__name__)
        return policy


# --------------------------------------------------------------------------- policy


_WARNED_UNSURFACED: set[str] = set()


def _warn_unsurfaced(provider: str | None) -> None:
    key = str(provider)
    with _GUARDIAN_LOCK:
        first = key not in _WARNED_UNSURFACED
        _WARNED_UNSURFACED.add(key)
    if first:
        logger.warning("unsurfaced LLM call provider={}", provider)


def resolve_policy(call_or_surface: GatewayCall | str | None, user_id: int | None = None) -> Any | None:
    """The policy to apply, or ``None`` for a skip.

    Skips: a caller-moderated or exempt surface, re-entry from a hook, and a disabled policy. A
    ``None`` surface is moderated as ``unclassified`` (the backstop), with one warning per provider. The policy user is ``str(user_id)``; ``None`` means the
    default policy.
    """
    surface = call_or_surface.surface if isinstance(call_or_surface, GatewayCall) else call_or_surface
    if isinstance(call_or_surface, GatewayCall) and user_id is None:
        user_id = call_or_surface.user_id
    if surface in CALLER_MODERATED_SURFACES or surface in EXEMPT_SURFACES or _IN_HOOK.get():
        return None
    if surface is None:
        surface = UNCLASSIFIED_SURFACE
        if isinstance(call_or_surface, GatewayCall):
            _warn_unsurfaced(call_or_surface.provider)
    policy = get_moderation_service().get_effective_policy(None if user_id is None else str(user_id), surface=surface)
    policy = _with_guardian(policy, user_id)
    return policy if policy.enabled else None


# --------------------------------------------------------------------------- text functions


def _moderate_text(text: str, policy: Any, phase: str) -> tuple[str, str]:
    """(text, action) for one string. A redact returns the redacted text; a block returns ``text`` unchanged."""
    service = get_moderation_service()
    with _evaluating():
        action, redacted, _pattern, _category = service.evaluate_action(text, policy, phase)
        if action == "redact":
            redacted = redacted if isinstance(redacted, str) else service.redact_text(text, policy, phase)
            return redacted, action
    return text, action


def _moderate_content(content: Any, policy: Any, note: Callable[[str], None]) -> Any:
    """``content`` moderated as input: a string, or the text parts of a list. Same object when unchanged."""
    if isinstance(content, str):
        new, action = _moderate_text(content, policy, "input")
        note(action)
        return new
    if not isinstance(content, list):
        return content
    parts: list[Any] | None = None
    for i, part in enumerate(content):
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            new, action = _moderate_text(part["text"], policy, "input")
            note(action)
            if new != part["text"]:
                parts = list(content) if parts is None else parts
                parts[i] = {**part, "text": new}
    return content if parts is None else parts


@contextlib.contextmanager
def _input_check_refusal(surface: str | None, provider: str | None, user_id: int | None) -> Iterator[None]:
    """A crash while checking input refuses the call (fail closed) as a policy refusal, never as a provider failure."""
    try:
        yield
    except LLMPolicyRefusal:
        raise
    except Exception as exc:
        logger.warning(
            "moderation input check failed ({}) surface={} provider={} user_id={}",
            type(exc).__name__, surface, provider, user_id,
        )
        raise LLMPolicyRefusal(INPUT_UNCHECKED_REFUSAL) from exc


def _moderate_input(
    messages: list[dict], policy: Any, surface: str | None, provider: str | None, user_id: int | None
) -> list[dict]:
    if not policy.enabled or not policy.input_enabled:
        return messages
    actions: list[str] = []

    def note(action: str) -> None:
        actions.append(action)
        _log_action("input", action, surface, provider, user_id)
        if action == "block":
            raise LLMPolicyRefusal(INPUT_REFUSAL)

    out: list[dict] | None = None
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in _MODERATED_ROLES:
            continue
        content = message.get("content")
        new = _moderate_content(content, policy, note)
        if new is not content:
            out = list(messages) if out is None else out
            out[i] = {**message, "content": new}
    return messages if out is None else out


def moderate_input_messages_for_surface(
    messages: list[dict], *, surface: str | None, user_id: int | None, policy: Any = None
) -> list[dict]:
    """Input moderation for one surface: the same list when nothing changes, a new list on redact,
    ``LLMPolicyRefusal`` on block. Only ``user`` and ``tool`` text is moderated, as in chat."""
    with _input_check_refusal(surface, None, user_id):
        policy = resolve_policy(surface, user_id) if policy is None else policy
        if policy is None:
            return messages
        return _moderate_input(messages, policy, surface, None, user_id)


def _moderate_output(
    text: str, policy: Any, surface: str | None, provider: str | None, user_id: int | None
) -> tuple[str, str]:
    if not policy.enabled or not policy.output_enabled:
        return text, "pass"
    try:
        new, action = _moderate_text(text, policy, "output")
    except Exception as exc:  # fail closed: an evaluator failure must not release unchecked text
        logger.warning("moderation output evaluator failed ({}); blocking surface={}", type(exc).__name__, surface)
        new, action = text, "block"
    _log_action("output", action, surface, provider, user_id)
    return (OUTPUT_REFUSAL if action == "block" else new), action


def moderate_output_text_for_surface(
    text: str, *, surface: str | None, user_id: int | None, policy: Any = None
) -> tuple[str, str]:
    """Output moderation for one string: ``(text, action)`` with action ``pass|warn|redact|block``.
    On block the text is the refusal text."""
    policy = resolve_policy(surface, user_id) if policy is None else policy
    if policy is None:
        return text, "pass"
    return _moderate_output(text, policy, surface, None, user_id)


# --------------------------------------------------------------------------- streaming


class StreamModerator:
    """Moderates one choice's streamed text, holding back the last ``holdback`` characters so a match
    that spans chunks is seen whole before any of it is released."""

    def __init__(self, policy: Any, *, holdback: int | None = None, on_action: Callable[[str], None] | None = None):
        self._policy = policy
        self._holdback = max(0, _stream_buffer_chars() if holdback is None else holdback)
        self._held = ""
        self._on_action = on_action
        self._seen: set[str] = set()

    @property
    def held(self) -> str:
        return self._held

    def feed(self, text: str) -> tuple[str, bool]:
        """``(text to emit now, blocked)``. After a block nothing is held."""
        full = self._held + text
        try:
            full, action = _moderate_text(full, self._policy, "output") if self._policy.output_enabled else (full, "pass")
        except Exception as exc:  # fail closed
            logger.warning("moderation output evaluator failed ({}); blocking stream", type(exc).__name__)
            action = "block"
        if action != "pass" and action not in self._seen:
            self._seen.add(action)
            if self._on_action is not None:
                self._on_action(action)
        if action == "block":
            self._held = ""
            return "", True
        cut = len(full) - self._holdback
        self._held = full[cut:] if cut > 0 else full
        return (full[:cut] if cut > 0 else ""), False

    def flush(self) -> str:
        """The held text. It was evaluated with the text after it, so it is released as is."""
        held, self._held = self._held, ""
        return held


def _data_payload(item: str) -> str | None:
    """The joined ``data:`` lines of one SSE item, or None when it has none."""
    lines = [line[5:][1:] if line[5:].startswith(" ") else line[5:] for line in item.splitlines() if line.startswith("data:")]
    return "\n".join(lines) if lines else None


def _content_frame(index: int, text: str) -> str:
    return sse_data({"choices": [{"index": index, "delta": {"content": text}}]})


def _choice_index(choice: dict) -> int:
    index = choice.get("index", 0)
    return index if isinstance(index, int) and not isinstance(index, bool) else 0


class _StreamState:
    """The stream wrapper's logic, shared by the sync and async wrappers. ``process`` maps one source
    item to the items to yield, ``finish`` the items to yield when the source ends."""

    def __init__(self, policy: Any, surface: str | None, provider: str | None, user_id: int | None):
        self._policy = policy
        self._moderators: dict[int, StreamModerator] = {}
        self._context = (surface, provider, user_id)
        self.stopped = False  # a block or a [DONE] ended the stream
        self.blocked = False

    def _moderator(self, index: int) -> StreamModerator:
        if index not in self._moderators:
            surface, provider, user_id = self._context
            self._moderators[index] = StreamModerator(
                self._policy, on_action=lambda action: _log_action("output", action, surface, provider, user_id)
            )
        return self._moderators[index]

    def _flush_all(self) -> list[str]:
        return [_content_frame(i, held) for i, m in sorted(self._moderators.items()) if (held := m.flush())]

    def finish(self) -> list[str]:
        return [] if self.stopped else self._flush_all()

    def _block(self, index: int) -> list[str]:
        self.stopped = self.blocked = True
        finish = {"choices": [{"index": index, "delta": {}, "finish_reason": "content_filter"}]}
        return [_content_frame(index, OUTPUT_REFUSAL), sse_data(finish), sse_done()]

    def process(self, item: Any) -> list[Any]:
        data = _data_payload(item) if isinstance(item, str) and not self.stopped else None
        if data is None:
            return [item]
        if data.strip() == "[DONE]":
            frames: list[Any] = self._flush_all()
            self.stopped = True
            return [*frames, item]
        try:
            chunk = json.loads(data)
        except ValueError:
            return [item]
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if not isinstance(choices, list) or not any(isinstance(c, dict) for c in choices):
            return [item]
        return self._process_chunk(item, chunk, choices)

    def _process_chunk(self, item: str, chunk: dict, choices: list) -> list[Any]:
        pre: list[str] = []  # held text released before a finish_reason chunk
        new_choices: list[Any] = []
        changed = False
        for choice in choices:
            if not isinstance(choice, dict):
                new_choices.append(choice)
                continue
            index = _choice_index(choice)
            delta = choice.get("delta")
            text = delta.get("content") if isinstance(delta, dict) else None
            text = text if isinstance(text, str) and text else None
            moderator = self._moderator(index) if text else self._moderators.get(index)
            emit = text
            if text:
                emit, blocked = moderator.feed(text)
                if blocked:
                    return self._block(index)
            if choice.get("finish_reason") and moderator is not None:
                held = moderator.flush()
                if text:
                    emit = (emit or "") + held
                elif held:
                    pre.append(_content_frame(index, held))
            # provider_response can carry other parts of the same event, so a content chunk that has one is rebuilt
            if text is None or (emit == text and not moderator.held and "provider_response" not in chunk):
                new_choices.append(choice)
                continue
            changed = True
            new_delta = {k: v for k, v in delta.items() if k != "content"}
            if emit:
                new_delta["content"] = emit
            # logprobs carry the original tokens, like provider_response
            new_choices.append({**{k: v for k, v in choice.items() if k != "logprobs"}, "delta": new_delta})
        if not changed:
            return [*pre, item]
        return [*pre, *_rebuilt(item, chunk, new_choices)]


def _rebuilt(item: str, chunk: dict, choices: list) -> list[str]:
    """The changed chunk without ``provider_response``, or nothing when it only carried held text.
    Non-``data:`` lines of the item (``event:``) are kept; the first ``data:`` line is replaced."""
    rebuilt = {k: v for k, v in chunk.items() if k != "provider_response"}
    rebuilt["choices"] = choices
    carries = any(k not in _CHUNK_METADATA_KEYS for k in rebuilt) or any(
        not isinstance(c, dict) or set(c) - {"index", "delta"} or c.get("delta") for c in choices
    )
    if not carries:
        return []
    lines = item.splitlines()
    if all(line.startswith("data:") for line in lines if line):
        return [sse_data(rebuilt)]
    new_data = sse_data(rebuilt).strip()
    out: list[str] = []
    for line in lines:
        if not line.startswith("data:"):
            out.append(line)
        elif new_data:
            out.append(new_data)
            new_data = ""
    return ["\n".join(out) + "\n\n"]


def _close(source: Any) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


class _SyncStream:
    """The sync wrapper. A generator's ``finally`` never runs if it is closed before its first ``next()``,
    so ``close()`` closes the source itself."""

    def __init__(self, source: Any, state: _StreamState) -> None:
        self._source = source
        self._gen = _sync_stream(source, state)

    def __iter__(self) -> _SyncStream:
        return self

    def __next__(self) -> Any:
        return next(self._gen)

    def close(self) -> None:
        self._gen.close()
        _close(self._source)


def _sync_stream(source: Any, state: _StreamState) -> Iterator[Any]:
    try:
        for item in source:
            yield from state.process(item)
            if state.blocked:
                return
        yield from state.finish()
    finally:
        _close(source)


async def _async_stream(source: Any, state: _StreamState) -> AsyncIterator[Any]:
    try:
        async for item in source:
            for out in state.process(item):
                yield out
            if state.blocked:
                return
        for out in state.finish():
            yield out
    finally:
        aclose = getattr(source, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await aclose()
        else:
            _close(source)


# --------------------------------------------------------------------------- Anthropic format
# For /api/v1/messages' native providers (raw HTTP, no gateway call). They share the policy
# resolution, fail-closed rules and logging of the hooks above.

_THINKING_BLOCKS = frozenset({"thinking", "redacted_thinking"})
# deltas that can carry text the moderator already changed: dropped from a message once any text was changed or held
_THINKING_DELTAS = frozenset({"thinking_delta", "signature_delta", "citations_delta"})


def _moderate_anthropic_content(content: Any, policy: Any, note: Callable[[str], None]) -> Any:
    """A user message's content: a string, text blocks, and the text inside ``tool_result`` blocks."""
    if isinstance(content, str):
        return _moderate_content(content, policy, note)
    if not isinstance(content, list):
        return content
    blocks: list[Any] | None = None
    for i, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            new = _moderate_content([block], policy, note)[0]
        elif kind == "tool_result" and isinstance(block.get("content"), (str, list)):
            inner = _moderate_anthropic_content(block["content"], policy, note)
            new = block if inner is block["content"] else {**block, "content": inner}
        else:
            continue
        if new is not block:
            blocks = list(content) if blocks is None else blocks
            blocks[i] = new
    return content if blocks is None else blocks


def moderate_anthropic_request(payload: dict, *, surface: str | None, user_id: int | None) -> dict:
    """Input moderation for an Anthropic Messages payload: the same dict when nothing changes, a new one
    on redact, ``LLMPolicyRefusal`` on block. Moderates ``user`` text and ``tool_result`` text; the
    top-level ``system``, images, ``tool_use`` and assistant messages are left alone."""
    with _input_check_refusal(surface, None, user_id):
        policy = resolve_policy(surface, user_id)
        messages = payload.get("messages")
        if policy is None or not policy.input_enabled or not isinstance(messages, list):
            return payload

        def note(action: str) -> None:
            _log_action("input", action, surface, None, user_id)
            if action == "block":
                raise LLMPolicyRefusal(INPUT_REFUSAL)

        out: list[Any] | None = None
        for i, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            new = _moderate_anthropic_content(content, policy, note)
            if new is not content:
                out = list(messages) if out is None else out
                out[i] = {**message, "content": new}
        return payload if out is None else {**payload, "messages": out}


def _refused_message(resp: dict) -> dict:
    """The refusal in place of ``resp``: nothing of the original survives but its ids and usage."""
    kept = {k: v for k, v in resp.items() if k in ("id", "type", "role", "model", "usage")}
    return {
        **kept,
        "content": [{"type": "text", "text": OUTPUT_REFUSAL}],
        "stop_reason": "refusal",
        "stop_sequence": None,
    }


def moderate_anthropic_response(resp: dict, *, surface: str | None, user_id: int | None) -> dict:
    """Output moderation for an Anthropic Messages response: the same dict when nothing changes. A block
    returns the refusal text with ``stop_reason: "refusal"``; a redact replaces the text blocks' text. Any
    change drops ``thinking`` blocks and the changed block's ``citations``, which carry the original text.
    Usage is the provider's. Never raises: a failure is a block."""
    try:
        policy = resolve_policy(surface, user_id)
    except Exception as exc:  # fail closed: moderation is on for all we know, and nothing was checked
        logger.warning("moderation policy resolution failed ({}) surface={}", type(exc).__name__, surface)
        return _refused_message(resp)
    content = resp.get("content")
    if policy is None or not policy.output_enabled or not isinstance(content, list):
        return resp
    blocks: list[Any] = []
    changed = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            new, action = _moderate_output(block["text"], policy, surface, None, user_id)
            if action == "block":
                return _refused_message(resp)
            if action == "redact":
                changed = True
                block = {**{k: v for k, v in block.items() if k != "citations"}, "text": new}
        blocks.append(block)
    if not changed:
        return resp
    return {**resp, "content": [b for b in blocks if not (isinstance(b, dict) and b.get("type") in _THINKING_BLOCKS)]}


def _anthropic_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def _with_data(item: str, data: dict) -> str:
    """``item`` with its first ``data:`` line replaced; every other line (``event:``) is kept."""
    new_data = "data: " + json.dumps(data, separators=(",", ":"))
    out: list[str] = []
    for line in item.strip("\n").splitlines():
        if not line.startswith("data:"):
            out.append(line)
        elif new_data:
            out.append(new_data)
            new_data = ""
    return "\n".join(out) + "\n\n"


def _text_delta(index: int, text: str) -> str:
    return _anthropic_frame(
        "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}
    )


class _AnthropicStreamState:
    """Logic of the Anthropic stream wrapper, shared by the sync and async wrappers. ``process`` maps one
    source frame to the items to yield, ``finish`` the items to yield when the source ends."""

    def __init__(self, policy: Any, surface: str | None, user_id: int | None):
        self._policy = policy
        self._moderators: dict[int, StreamModerator] = {}
        self._open: set[int] = set()
        self._context = (surface, user_id)
        # Set whenever text is held or rewritten, which is nearly every text frame while the holdback is above 0.
        # That is why later thinking/citations deltas are dropped even when nothing was redacted, and why text
        # frames are rarely the same object. The OpenAI path does the same.
        self.changed = False
        self.stopped = False
        self.blocked = False
        self._used: set[int] = set()  # every block index seen
        self._start_usage: dict = {}  # message_start's usage, kept for the injected block message_delta
        self._seen_output_tokens = 0  # the provider's own last output_tokens from a message_delta
        self._chars = 0  # text characters the provider produced so far, held text included

    def _moderator(self, index: int) -> StreamModerator:
        if index not in self._moderators:
            surface, user_id = self._context
            self._moderators[index] = StreamModerator(
                self._policy, on_action=lambda action: _log_action("output", action, surface, None, user_id)
            )
        return self._moderators[index]

    def _flush(self, only: int | None = None) -> list[str]:
        items = self._moderators.items() if only is None else [(only, self._moderators[only])] if only in self._moderators else []
        return [_text_delta(i, held) for i, m in sorted(items) if (held := m.flush())]

    def finish(self) -> list[str]:
        return [] if self.stopped else self._flush()

    def block(self, trigger: int | None = None) -> list[str]:
        """The refusal, a closed lifecycle for every open block, then the end of the message. The refusal goes
        in the block that triggered the block; with none open it goes in a block index nothing has used."""
        self.stopped = self.blocked = True
        index = trigger if trigger in self._open else max(self._used, default=-1) + 1
        frames = [] if index in self._open else [_anthropic_frame(
            "content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})]
        frames.append(_text_delta(index, OUTPUT_REFUSAL))
        frames += [_anthropic_frame("content_block_stop", {"type": "content_block_stop", "index": i}) for i in sorted(self._open | {index})]
        # The source is closed, so its own message_delta never arrives: estimate what it produced (4 chars a token).
        usage = {**self._start_usage, "output_tokens": max(1, self._chars // 4, self._seen_output_tokens)}
        frames.append(_anthropic_frame("message_delta", {
            "type": "message_delta", "delta": {"stop_reason": "refusal", "stop_sequence": None}, "usage": usage}))
        frames.append(_anthropic_frame("message_stop", {"type": "message_stop"}))
        self._open.clear()
        return frames

    def process(self, item: Any) -> list[Any]:
        data = _data_payload(item) if isinstance(item, str) and not self.stopped else None
        if data is None:
            return [item]
        try:
            event = json.loads(data)
        except ValueError:
            return [item]
        kind = event.get("type") if isinstance(event, dict) else None
        index = event.get("index") if isinstance(event, dict) else None
        index = index if isinstance(index, int) and not isinstance(index, bool) else 0
        if kind == "message_start":
            return self._process_message_start(item, event)
        if kind == "content_block_start":
            self._open.add(index)
            self._used.add(index)
            block = event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"]:
                return self._process_text(item, event, index, block["text"], start=True)
        elif kind == "content_block_stop":
            pre = self._flush(index)
            self._open.discard(index)
            return [*pre, item]
        elif kind in ("message_delta", "message_stop"):
            usage = event.get("usage")
            tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
            if isinstance(tokens, int) and not isinstance(tokens, bool):
                self._seen_output_tokens = max(self._seen_output_tokens, tokens)
            return [*self._flush(), item]
        elif kind == "content_block_delta" and isinstance(event.get("delta"), dict):
            delta = event["delta"]
            if delta.get("type") in _THINKING_DELTAS:
                return [] if self.changed else [item]
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                return self._process_text(item, event, index, delta["text"])
        return [item]

    def _process_message_start(self, item: str, event: dict) -> list[Any]:
        """Keep the start usage; moderate any text blocks the start frame itself carries (real providers send none)."""
        message = event.get("message")
        if not isinstance(message, dict):
            return [item]
        if isinstance(message.get("usage"), dict):
            self._start_usage = dict(message["usage"])
        content = message.get("content")
        if not isinstance(content, list):
            return [item]
        blocks: list[Any] | None = None
        for i, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"]:
                self._chars += len(block["text"])
                try:  # one-shot: this text has no later chunk, so nothing is held
                    new, action = _moderate_text(block["text"], self._policy, "output") if self._policy.output_enabled else (block["text"], "pass")
                except Exception as exc:  # fail closed
                    logger.warning("moderation output evaluator failed ({}); blocking stream", type(exc).__name__)
                    action = "block"
                if action == "block":
                    return self.block()
                if action == "redact":
                    blocks = list(content) if blocks is None else blocks
                    blocks[i] = {**{k: v for k, v in block.items() if k != "citations"}, "text": new}
        if blocks is None:
            return [item]
        self.changed = True
        return [_with_data(item, {**event, "message": {**message, "content": blocks}})]

    def _process_text(self, item: str, event: dict, index: int, text: str, start: bool = False) -> list[Any]:
        self._chars += len(text)
        moderator = self._moderator(index)
        emit, blocked = moderator.feed(text)
        if blocked:
            return self.block(index)
        if emit == text and not moderator.held:
            return [item]
        self.changed = True
        if start:  # the held text follows as a delta once the block is open
            return [_with_data(item, {**event, "content_block": {**event["content_block"], "text": emit}})]
        if not emit:
            return []
        return [_with_data(item, {**event, "delta": {**event["delta"], "text": emit}})]


class _RefusedState:
    """A state that ignores its source and yields the refusal frames, then ends."""

    def __init__(self, frames: list[str]) -> None:
        self._frames = frames
        self.stopped = self.blocked = True

    def process(self, item: Any) -> list[str]:
        frames, self._frames = self._frames, []
        return frames

    def finish(self) -> list[str]:
        return self.process(None)


class _AnthropicAsyncStream:
    """The async wrapper. It keeps the source's ``defer_usage_until_terminal`` flag, which the
    accounting wrappers read from the stream they are given."""

    def __init__(self, source: Any, state: _AnthropicStreamState) -> None:
        self.defer_usage_until_terminal = getattr(source, "defer_usage_until_terminal", False)
        self._source = source
        self._gen = _async_stream(source, state)  # type: ignore[arg-type]

    def __aiter__(self) -> _AnthropicAsyncStream:
        return self

    async def __anext__(self) -> Any:
        return await self._gen.__anext__()

    async def aclose(self) -> None:
        await self._gen.aclose()
        aclose = getattr(self._source, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await aclose()


def wrap_anthropic_stream(iterator: Any, *, surface: str | None, user_id: int | None) -> Any:
    """Moderate an Anthropic SSE stream (one frame per item, ``event:`` + ``data:`` lines). The same
    iterator when moderation is off. Text deltas are moderated per content block; an unchanged frame is
    yielded as the same object. On block the stream ends with a refusal delta, ``content_block_stop``
    for every open block, ``message_delta`` (``stop_reason: "refusal"``) and ``message_stop``."""
    try:
        policy = resolve_policy(surface, user_id)
    except Exception as exc:  # fail closed: nothing was checked, so the source is closed and only the refusal is sent
        logger.warning("moderation policy resolution failed ({}) surface={}", type(exc).__name__, surface)
        refusal = _AnthropicStreamState(None, surface, user_id).block()
        return _AnthropicAsyncStream(iterator, _RefusedState(refusal)) if hasattr(iterator, "__aiter__") else _SyncStream(
            iterator, _RefusedState(refusal)  # type: ignore[arg-type]
        )
    if policy is None or not policy.output_enabled:
        return iterator
    state = _AnthropicStreamState(policy, surface, user_id)
    if hasattr(iterator, "__aiter__"):
        return _AnthropicAsyncStream(iterator, state)
    return _SyncStream(iterator, state)  # type: ignore[arg-type]  # same process/finish/blocked shape


# --------------------------------------------------------------------------- hooks


def _policy_of(call: GatewayCall) -> Any | None:
    stored = getattr(call, "_moderation_policy", _UNRESOLVED)
    if stored is _UNRESOLVED:
        stored = resolve_policy(call) or _SKIP
        call._moderation_policy = stored  # type: ignore[attr-defined]
    return None if stored is _SKIP else stored


def pre_call(call: GatewayCall) -> None:
    """Resolve the policy once, store it on the call, then moderate the request's messages."""
    with _input_check_refusal(call.surface, call.provider, call.user_id):
        policy = _policy_of(call)
        messages = call.request.get("messages")
        if policy is None or not isinstance(messages, list):
            return
        moderated = _moderate_input(messages, policy, call.surface, call.provider, call.user_id)
        if moderated is not messages:
            call.request["messages"] = moderated


def _moderate_result(call: GatewayCall, result: dict, policy: Any) -> dict:
    choices = result.get("choices")
    if not isinstance(choices, list):
        return result
    out: list[Any] = []
    changed = False
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            out.append(choice)
            continue
        new, action = _moderate_output(content, policy, call.surface, call.provider, call.user_id)
        if action not in ("redact", "block"):
            out.append(choice)
            continue
        changed = True
        new_message = {**message, "content": new}
        new_choice = {k: v for k, v in choice.items() if k not in ("provider_response", "logprobs")}
        if action == "block":  # nothing of the refused message survives: reasoning, tool calls, unknown fields
            new_message = {"role": message.get("role", "assistant"), "content": OUTPUT_REFUSAL}
            new_choice["finish_reason"] = "content_filter"
        out.append({**new_choice, "message": new_message})
    if not changed:
        return result
    return {**{k: v for k, v in result.items() if k != "provider_response"}, "choices": out}


def _refused(result: Any) -> Any:
    """The refusal in place of ``result``, whose text was never checked; usage is kept for a dict."""
    if isinstance(result, dict):
        kept = {k: v for k, v in result.items() if k in ("id", "object", "created", "model", "usage")}
        choice = {"index": 0, "message": {"role": "assistant", "content": OUTPUT_REFUSAL}, "finish_reason": "content_filter"}
        return {**kept, "choices": [choice]}
    state = _StreamState(None, None, None, None)
    if hasattr(result, "__aiter__"):
        async def refused_async() -> AsyncIterator[Any]:
            for frame in state._block(0):
                yield frame
            aclose = getattr(result, "aclose", None)
            if callable(aclose):
                with contextlib.suppress(Exception):
                    await aclose()

        return refused_async()
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
        def refused_sync() -> Iterator[Any]:
            try:
                yield from state._block(0)
            finally:
                _close(result)

        return refused_sync()
    return result


def post_call(call: GatewayCall, result: Any) -> Any:
    """Moderate the provider's answer: a dict (non-stream) or a sync/async stream iterator. Never raises."""
    try:
        policy = _policy_of(call)
    except Exception as exc:  # fail closed: moderation is on for all we know, and nothing was checked
        logger.warning("moderation policy resolution failed ({}) surface={}", type(exc).__name__, call.surface)
        return _refused(result)
    try:
        if policy is None or not policy.output_enabled:
            return result
        if isinstance(result, dict):
            return _moderate_result(call, result, policy)
        state = _StreamState(policy, call.surface, call.provider, call.user_id)
        if hasattr(result, "__aiter__"):
            return _async_stream(result, state)
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            return _SyncStream(result, state)
    except Exception as exc:
        logger.warning("moderation post hook failed ({}) surface={}", type(exc).__name__, call.surface)
    return result


# --------------------------------------------------------------------------- registration

_REGISTRATION: dict[str, Callable[[], None] | None] = {"remove": None}
_REGISTRATION_LOCK = threading.Lock()


def _unregister() -> None:
    with _REGISTRATION_LOCK:
        remove = _REGISTRATION["remove"]
        _REGISTRATION["remove"] = None
        if remove is not None:
            remove()


def register_moderation_gateway_hooks() -> Callable[[], None]:
    """Register the hooks once; a second call while registered adds nothing. The returned remover
    unregisters them and resets the flag, so a later call registers again."""
    with _REGISTRATION_LOCK:
        if _REGISTRATION["remove"] is None:
            _REGISTRATION["remove"] = add_policy_hooks(pre_call=pre_call, post_call=post_call)
    return _unregister
