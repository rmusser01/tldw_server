from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator, Iterable
from typing import Any

from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.cache_intents import (
    apply_billing_prompt_cache_intent,
    attach_cache_intent_metadata,
)
from tldw_Server_API.app.core.LLM_Calls.payload_utils import merge_extra_body, merge_extra_headers
from tldw_Server_API.app.core.LLM_Calls.sse import (
    finalize_stream,
    is_done_line,
    normalize_provider_line,
    sse_data,
    sse_done,
)
from tldw_Server_API.app.core.testing import is_truthy

from .base import ChatProvider

_ANTHROPIC_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AssertionError,
    AttributeError,
    ConnectionError,
    EOFError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)

try:
    import httpx as _httpx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _httpx = None  # type: ignore
else:
    _ANTHROPIC_NONCRITICAL_EXCEPTIONS = _ANTHROPIC_NONCRITICAL_EXCEPTIONS + (
        _httpx.DecodingError,
        _httpx.HTTPError,
        _httpx.ProtocolError,
        _httpx.TimeoutException,
        _httpx.TransportError,
    )


def _prefer_httpx_in_tests() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST"))
from tldw_Server_API.app.core.http_client import (
    create_client as _hc_create_client,
)

http_client_factory = _hc_create_client


class AnthropicAdapter(ChatProvider):
    name = "anthropic"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 60,
            "max_output_tokens_default": 8192,
        }

    def _use_native_http(self) -> bool:
        import os
        if os.getenv("PYTEST_CURRENT_TEST"):
            return True
        v = (os.getenv("LLM_ADAPTERS_NATIVE_HTTP_ANTHROPIC") or "").strip().lower()
        if v in {"0", "false", "no", "off"}:
            return False
        if is_truthy(v):
            return True
        return True

    def _anthropic_base_url(self) -> str:
        import os
        return os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")

    def _resolve_base_url(self, request: dict[str, Any]) -> str:
        """Resolve API base URL with precedence: app_config -> env -> default."""
        override = (request or {}).get("base_url")
        if isinstance(override, str) and override.strip():
            return override.strip()
        try:
            cfg = (request or {}).get("app_config") or {}
            anth_cfg = cfg.get("anthropic_api") or {}
            base = anth_cfg.get("api_base_url")
            if isinstance(base, str) and base.strip():
                return base.strip()
        except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
            pass
        return self._anthropic_base_url()

    def _resolve_timeout(self, request: dict[str, Any], fallback: float | None) -> float:
        """Resolve request timeout seconds from request/app_config, else fallback/capability default."""
        try:
            cfg = (request or {}).get("app_config") or {}
            anth_cfg = cfg.get("anthropic_api") or {}
            t = anth_cfg.get("api_timeout")
            if t is not None:
                # Accept int/float/str that can be cast to float
                try:
                    return float(t)
                except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                    pass
        except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
            pass
        if fallback is not None:
            return float(fallback)
        # Use adapter capability default
        try:
            return float(self.capabilities().get("default_timeout_seconds", 60))
        except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
            return 60.0

    def _headers(self, api_key: str | None) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key or "",
            "anthropic-version": "2023-06-01",
        }

    @staticmethod
    def _to_anthropic_messages(messages: list[dict[str, Any]], system: str | None) -> dict[str, Any]:
        # Anthropic expects a list of {role, content}; include system separately
        out = {"messages": messages}
        if system:
            out["system"] = system
        return out

    def _parse_data_url_for_multimodal(self, url: str) -> tuple[str, str] | None:
        try:
            if not isinstance(url, str) or not url.startswith("data:"):
                return None
            # Format: data:<mime>;base64,<data>
            head, b64 = url.split(",", 1)
            mime = head[5:]  # strip 'data:'
            if ";base64" in mime:
                mime = mime.replace(";base64", "").strip()
            return mime, b64
        except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
            return None

    def _anthropic_image_source_from_part(self, image_url: dict[str, Any]) -> dict[str, Any] | None:
        url_str = (image_url or {}).get("url")
        if not url_str:
            return None
        parsed = self._parse_data_url_for_multimodal(url_str)
        if parsed:
            mime_type, b64 = parsed
            return {"type": "base64", "media_type": mime_type, "data": b64}
        if isinstance(url_str, str) and url_str.startswith(("http://", "https://")):
            return {"type": "url", "url": url_str}
        return None

    def _build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        raw_messages = request.get("messages") or []
        system_message = request.get("system_message")

        # Convert OpenAI-style messages to Anthropic messages format
        messages: list[dict[str, Any]] = []
        tool_result_counter = 0
        tool_use_counter = 0

        def _tool_result_content(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                text_parts: list[str] = []
                for item in value:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif isinstance(item, str):
                        text_parts.append(item)
                if text_parts:
                    return "".join(text_parts)
            try:
                return json.dumps(value, ensure_ascii=True)
            except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                return str(value)

        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "tool":
                tool_use_id = msg.get("tool_call_id") or msg.get("tool_use_id") or msg.get("id")
                if not tool_use_id:
                    tool_use_id = f"tool_result_{tool_result_counter}"
                    tool_result_counter += 1
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": _tool_result_content(msg.get("content")),
                }
                if isinstance(msg.get("is_error"), bool):
                    block["is_error"] = msg.get("is_error")
                messages.append({"role": "user", "content": [block]})
                continue
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content")
            parts: list[dict[str, Any]] = []
            if isinstance(content, str):
                parts.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    pt = p.get("type")
                    if pt == "text":
                        parts.append({"type": "text", "text": p.get("text", "")})
                    elif pt == "image_url":
                        src = self._anthropic_image_source_from_part(p.get("image_url", {}))
                        if src:
                            parts.append({"type": "image", "source": src})
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        func = tc.get("function") or {}
                        name = func.get("name") or tc.get("name")
                        if not isinstance(name, str) or not name.strip():
                            continue
                        args = func.get("arguments")
                        input_obj: Any = {}
                        if isinstance(args, str):
                            try:
                                input_obj = json.loads(args)
                            except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                                input_obj = args
                        elif args is not None:
                            input_obj = args
                        tool_id = tc.get("id") or f"tool_{tool_use_counter}"
                        tool_use_counter += 1
                        parts.append({"type": "tool_use", "id": tool_id, "name": name, "input": input_obj})
            if parts:
                messages.append({"role": role, "content": parts})

        payload = {
            "model": request.get("model"),
            "messages": messages,
            "max_tokens": request.get("max_tokens") or 1024,
        }
        if system_message:
            payload["system"] = system_message
        if request.get("temperature") is not None:
            payload["temperature"] = request.get("temperature")
        if request.get("top_p") is not None:
            payload["top_p"] = request.get("top_p")
        if request.get("top_k") is not None:
            payload["top_k"] = request.get("top_k")
        stop_val = request.get("stop")
        if stop_val is not None:
            if isinstance(stop_val, (list, tuple)):
                payload["stop_sequences"] = list(stop_val)
            else:
                payload["stop_sequences"] = [stop_val]
        # Tools mapping (OpenAI-style → Anthropic)
        tool_choice = request.get("tool_choice")
        tools = request.get("tools")
        if tool_choice == "none":
            # Honor explicit none by omitting tools entirely
            tools = None
        if isinstance(tools, list) and tools:
            converted: list[dict[str, Any]] = []
            for t in tools:
                try:
                    if isinstance(t, dict) and (t.get("type") == "function") and isinstance(t.get("function"), dict):
                        fn = t["function"]
                        # Require a non-empty string function name; otherwise skip as malformed.
                        name_raw = fn.get("name")
                        if not isinstance(name_raw, str):
                            continue
                        name = name_raw.strip()
                        if not name:
                            continue
                        desc_val = fn.get("description")
                        desc = str(desc_val) if isinstance(desc_val, (str, int, float)) else (desc_val or "")
                        schema = fn.get("parameters") or {}
                        if not isinstance(schema, dict):
                            schema = {}
                        converted.append({
                            "name": name,
                            "description": desc,
                            "input_schema": schema,
                        })
                except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                    continue
            # Only include tools if at least one valid entry exists.
            # Valid means function name is a non-empty string; malformed entries are skipped.
            # This ensures tests like malformed-tools expect 'tools' to be omitted entirely.
            # Filter again defensively in case prior logic added any invalid entries.
            converted = [
                t for t in converted
                if isinstance(t.get("name"), str) and t.get("name", "").strip()
            ]
            if converted:
                payload["tools"] = converted
        # tool_choice mapping (force a specific tool when requested)
        if isinstance(tool_choice, dict):
            try:
                if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
                    name = tool_choice["function"].get("name")
                    if name:
                        payload["tool_choice"] = {"type": "tool", "name": str(name)}
            except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                pass
        return payload

    @staticmethod
    def _normalize_to_openai_shape(data: dict[str, Any]) -> dict[str, Any]:
        # Best-effort shaping of Anthropic "message" into OpenAI-like chat completion
        if not (isinstance(data, dict) and data.get("type") == "message"):
            return data
        parts = data.get("content") or []
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        if isinstance(parts, list):
            for p in parts:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    text_parts.append(p.get("text", ""))
                elif p.get("type") == "tool_use":
                    tool_id = p.get("id") or f"anthropic_tool_{len(tool_calls)}"
                    name = p.get("name") or ""
                    try:
                        args = __import__("json").dumps(p.get("input", {}))
                    except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                        args = str(p.get("input"))
                    tool_calls.append({
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
        message_payload: dict[str, Any] = {"role": "assistant", "content": None}
        content_text = "\n".join([t for t in text_parts if t]).strip()
        if content_text:
            message_payload["content"] = content_text
        if tool_calls:
            message_payload["tool_calls"] = tool_calls
        finish_reason_map = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop", "tool_use": "tool_calls"}
        shaped = {
            "id": data.get("id"),
            "object": "chat.completion",
            "model": data.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": message_payload,
                    "finish_reason": finish_reason_map.get(data.get("stop_reason"), data.get("stop_reason")),
                }
            ],
        }
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            shaped["usage"] = {
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
                "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
            }
        shaped["provider_response"] = data
        return shaped

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = validate_payload(self.name, request or {})
        if _prefer_httpx_in_tests() or os.getenv("PYTEST_CURRENT_TEST") or self._use_native_http():
            api_key = request.get("api_key")
            url = f"{self._resolve_base_url(request).rstrip('/')}/messages"
            headers = self._headers(api_key)
            payload = self._build_payload(request)
            payload["stream"] = False
            payload, cache_intent_diagnostic = apply_billing_prompt_cache_intent(self.name, payload, request)
            payload = merge_extra_body(payload, request)
            headers = merge_extra_headers(headers, request)
            try:
                resolved_timeout = self._resolve_timeout(request, timeout)
                with http_client_factory(timeout=resolved_timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return attach_cache_intent_metadata(
                        self._normalize_to_openai_shape(data),
                        cache_intent_diagnostic,
                    )
            except _ANTHROPIC_NONCRITICAL_EXCEPTIONS as e:
                raise self.normalize_error(e) from e
        # If native HTTP is explicitly disabled, raise a clear error rather than
        # delegating to legacy paths to avoid recursion and mixed behaviors.
        raise RuntimeError("AnthropicAdapter native HTTP disabled by configuration")

    def _tool_delta_chunk(
        self,
        tool_index: int,
        tool_id: str,
        tool_name: str | None,
        arguments: str,
        provider_response: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": tool_index,
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": tool_name or "", "arguments": arguments},
                    }]
                },
            }]
        }
        if provider_response is not None:
            payload["provider_response"] = provider_response
        return sse_data(payload)

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = validate_payload(self.name, request or {})
        if _prefer_httpx_in_tests() or os.getenv("PYTEST_CURRENT_TEST") or self._use_native_http():
            api_key = request.get("api_key")
            url = f"{self._resolve_base_url(request).rstrip('/')}/messages"
            headers = self._headers(api_key)
            payload = self._build_payload(request)
            payload["stream"] = True
            payload, _cache_intent_diagnostic = apply_billing_prompt_cache_intent(self.name, payload, request)
            payload = merge_extra_body(payload, request)
            headers = merge_extra_headers(headers, request)
            try:
                resolved_timeout = self._resolve_timeout(request, timeout)
                with http_client_factory(timeout=resolved_timeout) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as resp:
                        resp.raise_for_status()
                        tool_states: dict[int, dict[str, Any]] = {}
                        tool_counter = 0
                        done_sent = False
                        for raw in resp.iter_lines():
                            if not raw:
                                continue
                            try:
                                line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                            except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                                line = str(raw)
                            if is_done_line(line):
                                if not done_sent:
                                    done_sent = True
                                    yield sse_done()
                                continue
                            ls = line.strip()
                            if not ls or not ls.startswith("data:"):
                                # Drop provider control lines/comments by default
                                normalized = normalize_provider_line(ls)
                                if normalized is not None:
                                    yield normalized
                                continue
                            event_data = ls[len("data:"):].strip()
                            if not event_data:
                                continue
                            try:
                                ev = __import__("json").loads(event_data)
                            except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                                continue
                            ev_type = ev.get("type")
                            if ev_type == "content_block_start":
                                cb = ev.get("content_block", {})
                                if cb.get("type") == "tool_use":
                                    idx = int(ev.get("index", 0))
                                    tool_id = cb.get("id") or f"anthropic_tool_{tool_counter}"
                                    tool_name = cb.get("name")
                                    initial_input = cb.get("input")
                                    buf = ""
                                    if initial_input is not None:
                                        try:
                                            buf = __import__("json").dumps(initial_input)
                                        except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                                            buf = str(initial_input)
                                    tool_states[idx] = {"id": tool_id, "name": tool_name, "buffer": buf, "position": tool_counter}
                                    tool_counter += 1
                                    yield self._tool_delta_chunk(
                                        tool_states[idx]["position"],
                                        tool_id,
                                        tool_name,
                                        buf,
                                        provider_response=ev,
                                    )
                            elif ev_type == "content_block_delta":
                                delta = ev.get("delta", {})
                                idx = int(ev.get("index", 0))
                                dt = delta.get("type")
                                if dt == "text_delta" and "text" in delta:
                                    text = delta.get("text") or ""
                                    yield sse_data({
                                        "choices": [{"delta": {"content": text}}],
                                        "provider_response": ev,
                                    })
                                elif dt == "input_json_delta" and idx in tool_states:
                                    partial = delta.get("partial_json", "")
                                    if partial:
                                        st = tool_states[idx]
                                        st["buffer"] += partial
                                        yield self._tool_delta_chunk(
                                            st["position"],
                                            st["id"],
                                            st["name"],
                                            st["buffer"],
                                            provider_response=ev,
                                        )
                                elif dt == "tool_use_delta" and idx in tool_states:
                                    st = tool_states[idx]
                                    if "name" in delta and delta["name"]:
                                        st["name"] = delta["name"]
                                    if "input" in delta and delta["input"] is not None:
                                        try:
                                            st["buffer"] = __import__("json").dumps(delta["input"])
                                        except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                                            st["buffer"] = str(delta["input"])
                                    yield self._tool_delta_chunk(
                                        st["position"],
                                        st["id"],
                                        st["name"],
                                        st["buffer"],
                                        provider_response=ev,
                                    )
                            elif ev_type == "message_delta":
                                stop_reason = (ev.get("delta") or {}).get("stop_reason")
                                if stop_reason:
                                    fr_map = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop", "tool_use": "tool_calls"}
                                    finish_reason = fr_map.get(stop_reason, stop_reason)
                                    yield sse_data({
                                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                                        "provider_response": ev,
                                    })
                        yield from finalize_stream(response=resp, done_already=done_sent)
                return
            except _ANTHROPIC_NONCRITICAL_EXCEPTIONS as e:
                raise self.normalize_error(e) from e
        # If native HTTP is explicitly disabled, raise a clear error rather than
        # delegating to legacy paths to avoid recursion and mixed behaviors.
        raise RuntimeError("AnthropicAdapter native HTTP disabled by configuration")

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        gen = self.stream(request, timeout=timeout)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        stop_event = threading.Event()

        def _worker() -> None:
            try:
                for item in gen:
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                try:
                    if hasattr(gen, "close"):
                        gen.close()
                except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                    pass
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop_event.set()

    def normalize_error(self, exc: Exception):  # type: ignore[override]
        from tldw_Server_API.app.core.LLM_Calls.error_utils import (
            get_http_error_text,
            get_http_status_from_exception,
            is_http_status_error,
            log_http_400_body,
        )
        if is_http_status_error(exc):
            from tldw_Server_API.app.core.Chat.Chat_Deps import (
                ChatAPIError,
                ChatAuthenticationError,
                ChatBadRequestError,
                ChatProviderError,
                ChatRateLimitError,
            )
            resp = getattr(exc, "response", None)
            status = get_http_status_from_exception(exc)
            body = None
            try:
                body = resp.json()
            except _ANTHROPIC_NONCRITICAL_EXCEPTIONS:
                body = None
            log_http_400_body(self.name, exc, body)
            detail = None
            # Anthropic returns {"error": {"type": "...", "message": "..."}}
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                eobj = body["error"]
                msg = (eobj.get("message") or "").strip()
                typ = (eobj.get("type") or "").strip()
                detail = (f"{typ} {msg}" if typ else msg) or str(exc)
            else:
                detail = get_http_error_text(exc)
            if status in (400, 404, 422):
                return ChatBadRequestError(provider=self.name, message=str(detail))
            if status in (401, 403):
                return ChatAuthenticationError(provider=self.name, message=str(detail))
            if status == 429:
                return ChatRateLimitError(provider=self.name, message=str(detail))
            if status and 500 <= status < 600:
                return ChatProviderError(provider=self.name, message=str(detail), status_code=status)
            return ChatAPIError(provider=self.name, message=str(detail), status_code=status or 500)
        return super().normalize_error(exc)
