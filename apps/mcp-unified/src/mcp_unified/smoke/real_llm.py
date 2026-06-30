"""Env-gated real LLM calls for MCP smoke UAT scenarios."""

from __future__ import annotations

import json
from typing import Any

import httpx

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TIMEOUT_SECONDS = 15.0
_MAX_RESPONSE_BYTES = 64 * 1024


async def call_openai_compatible(
    *,
    api_key: str,
    prompt: str,
    base_url: object | None = None,
    model: object | None = None,
    http_client: httpx.AsyncClient | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Call an OpenAI-compatible chat endpoint and return redaction-safe metadata."""

    if not api_key:
        raise ValueError("api_key is required")
    endpoint_root = str(base_url or _DEFAULT_BASE_URL).rstrip("/")
    model_name = str(model or _DEFAULT_MODEL)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0,
        "max_tokens": 80,
    }
    client = http_client or httpx.AsyncClient(timeout=timeout)
    should_close = http_client is None
    try:
        async with client.stream(
            "POST",
            f"{endpoint_root}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise ValueError("real LLM response exceeded size bound")
            data = json.loads(bytes(body))
    finally:
        if should_close:
            await client.aclose()

    choice_text, finish_reason, choice_count = _extract_choice_metadata(data)
    return {
        "provider": "openai-compatible",
        "model": model_name,
        "choice_count": choice_count,
        "finish_reason": finish_reason,
        "response_chars": len(choice_text),
    }


def _extract_choice_metadata(data: Any) -> tuple[str, object | None, int]:
    """Extract redaction-safe choice text length metadata from an LLM response."""
    if not isinstance(data, dict):
        return "", None, 0
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None, 0
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return "", None, len(choices)
    message = first_choice.get("message")
    text = ""
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        text = message["content"]
    return text, first_choice.get("finish_reason"), len(choices)


__all__ = ["call_openai_compatible"]
