"""Isolated UAT261 complete-v2 request-boundary evidence harness.

The harness is deliberately outside maintained tests. It creates a temporary
test-process database layout, patches only the route's provider-call seam, and
emits a redacted projection. It never delegates to a provider adapter.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import sys
import tempfile
import traceback
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx


EVIDENCE_PATH = Path(__file__).with_name("boundary-harness-evidence.json")
_USER_TURN = "uat261-boundary-probe"
_SYNTHETIC_REASONING = "synthetic-boundary-reasoning"
_SYNTHETIC_ANSWER = "synthetic-boundary-answer"
_FORBIDDEN_EVIDENCE_TEXT = (
    _USER_TURN,
    _SYNTHETIC_REASONING,
    _SYNTHETIC_ANSWER,
    "test-key",
    "api_key",
    "credentials_resolved",
    "reasoning",
)
_TEMP_ENV_KEYS = (
    "AUTH_MODE",
    "SINGLE_USER_API_KEY",
    "JWT_SECRET_KEY",
    "DATABASE_URL",
    "USER_DB_BASE_DIR",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE_URL",
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _require(condition: bool, stage: str) -> None:
    if not condition:
        raise RuntimeError(stage)


def _settings_fingerprint(kwargs: Mapping[str, Any]) -> str:
    """Hash only actual complete-v2 dispatch controls and no credentials/context."""
    settings = {
        "provider": kwargs.get("api_endpoint"),
        "model": kwargs.get("model"),
        "temperature": kwargs.get("temp"),
        "top_p": kwargs.get("top_p"),
        "repetition_penalty": kwargs.get("repetition_penalty"),
        "stop": kwargs.get("stop"),
        "max_tokens": kwargs.get("max_tokens"),
        "stream": kwargs.get("streaming"),
        "tools_hash": _sha256(_canonical(kwargs.get("tools"))),
        "tool_choice_hash": _sha256(_canonical(kwargs.get("tool_choice"))),
        "billing_prompt_cache_intent_hash": _sha256(
            _canonical(kwargs.get("billing_prompt_cache_intent"))
        ),
        "inference_prefix_cache_intent_hash": _sha256(
            _canonical(kwargs.get("inference_prefix_cache_intent"))
        ),
    }
    return f"sha256:{_sha256(_canonical(settings))}"


def _safe_usage(payload: Mapping[str, Any]) -> dict[str, int] | None:
    usage = payload.get("usage")
    allowed = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(usage, Mapping) or not all(isinstance(usage.get(key), int) for key in allowed):
        return None
    return {key: int(usage[key]) for key in allowed}


def _terminal_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    fingerprint = payload.get("system_fingerprint")
    return {
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
        "usage": _safe_usage(payload),
        "system_fingerprint_hash": (
            f"sha256:{_sha256(fingerprint)}" if isinstance(fingerprint, str) else None
        ),
    }


def _stream_frames(
    *, finish_reason: str, usage: dict[str, int] | None, system_fingerprint: str | None, include_final: bool
) -> list[str]:
    frames = [
        f"data: {_canonical({'id': 'synthetic-boundary', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'reasoning_content': _SYNTHETIC_REASONING}, 'finish_reason': None}]})}"
    ]
    if include_final:
        frames.append(
            f"data: {_canonical({'id': 'synthetic-boundary', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': _SYNTHETIC_ANSWER}, 'finish_reason': None}]})}"
        )
    terminal: dict[str, Any] = {
        "id": "synthetic-boundary",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        terminal["usage"] = usage
    if system_fingerprint is not None:
        terminal["system_fingerprint"] = system_fingerprint
    return [*frames, f"data: {_canonical(terminal)}", "data: [DONE]"]


def _capture_stream(*, capture: dict[str, Any], frames: list[str]) -> Iterator[str]:
    """Observe synthetic terminal fields while forwarding the exact stream frames."""
    answer_parts: list[str] = []
    for frame in frames:
        if frame.startswith("data: ") and frame.strip().lower() != "data: [done]":
            payload = json.loads(frame[6:])
            choices = payload.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else {}
            delta = choice.get("delta") if isinstance(choice, Mapping) else {}
            content = delta.get("content") if isinstance(delta, Mapping) else None
            if isinstance(content, str):
                answer_parts.append(content)
            if isinstance(choice, Mapping) and choice.get("finish_reason") is not None:
                capture["provider_response"] = _terminal_projection(payload)
                answer = "".join(answer_parts)
                capture["final_answer"] = {
                    "present": bool(answer),
                    "length": len(answer),
                    "sha256": _sha256(answer) if answer else None,
                }
        yield frame


async def _run(
    *, monkeypatch: Any | None = None, bind_provider_call: Any | None = None
) -> dict[str, Any]:
    # Imports occur after the isolated test environment is set in main().
    import tldw_Server_API.app.api.v1.endpoints.character_chat_sessions as chat_sessions
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.Chat.prompt_cost_envelope import build_prompt_cost_envelope
    from tldw_Server_API.app.main import app

    captures: list[dict[str, Any]] = []
    synthetic_terminal = (
        {"finish_reason": "stop", "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}, "system_fingerprint": "synthetic-system-a", "include_final": True},
        {"finish_reason": "length", "usage": None, "system_fingerprint": None, "include_final": False},
    )

    def fake_provider_call(*args: Any, **kwargs: Any) -> Iterator[str]:
        _require(not args, "provider-positional-arguments")
        outbound = bind_provider_call(kwargs) if bind_provider_call is not None else kwargs
        messages = outbound["messages_payload"]
        _require(isinstance(messages, list), "provider-messages-not-list")
        envelope = build_prompt_cost_envelope(messages)
        call_index = len(captures)
        terminal = synthetic_terminal[call_index]
        capture = {
            "capture_version": "uat261-boundary-v1",
            "call_label": "A" if call_index == 0 else "B",
            "message_fingerprint_version": envelope.fingerprint_version,
            "message_fingerprint": envelope.aggregate_fingerprint,
            "message_count": envelope.message_count,
            "message_roles": [str(message.get("role")) for message in messages if isinstance(message, Mapping)],
            "generation_settings_fingerprint": _settings_fingerprint(outbound),
            "provider_response": {"finish_reason": None, "usage": None, "system_fingerprint_hash": None},
            "final_answer": {"present": False, "length": 0, "sha256": None},
        }
        captures.append(capture)
        return _capture_stream(capture=capture, frames=_stream_frames(**terminal))

    settings = get_settings()
    headers = {"X-API-KEY": settings.SINGLE_USER_API_KEY}
    transport = httpx.ASGITransport(app=app)
    async def exercise() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            characters = await client.get("/api/v1/characters/", headers=headers)
            if characters.status_code != 200:
                raise RuntimeError("character-list")
            character_id = int(characters.json()[0]["id"])
            for _ in range(2):
                chat = await client.post("/api/v1/chats/", headers=headers, json={"character_id": character_id})
                if chat.status_code != 201:
                    raise RuntimeError("chat-create")
                chat_id = str(chat.json()["id"])
                async with client.stream(
                    "POST",
                    f"/api/v1/chats/{chat_id}/complete-v2",
                    headers=headers,
                    json={
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "append_user_message": _USER_TURN,
                        "save_to_db": False,
                        "stream": True,
                    },
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError("complete-v2")
                    async for _line in response.aiter_lines():
                        pass

    if monkeypatch is None:
        with patch.object(chat_sessions, "perform_chat_api_call", fake_provider_call):
            await exercise()
    else:
        monkeypatch.setattr(chat_sessions, "perform_chat_api_call", fake_provider_call)
        await exercise()

    _require(len(captures) == 2, "capture-count")
    _require(captures[0]["message_fingerprint"] == captures[1]["message_fingerprint"], "message-fingerprint-mismatch")
    _require(captures[0]["generation_settings_fingerprint"] == captures[1]["generation_settings_fingerprint"], "settings-fingerprint-mismatch")
    _require(captures[0]["provider_response"]["usage"] == {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}, "usage-capture")
    _require(captures[1]["provider_response"]["usage"] is None, "missing-usage")
    _require(captures[0]["final_answer"]["present"] is True, "final-answer-present")
    _require(captures[1]["final_answer"] == {"present": False, "length": 0, "sha256": None}, "missing-final-answer")

    evidence = {"harness_version": "uat261-boundary-v1", "calls": captures}
    serialized = _canonical(evidence)
    _require(all(forbidden not in serialized for forbidden in _FORBIDDEN_EVIDENCE_TEXT), "forbidden-evidence-leak")
    EVIDENCE_PATH.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return evidence


def _set_isolated_environment(root: Path) -> dict[str, str | None]:
    """Mirror the existing mock-e2e fixture with only disposable test paths."""
    old = {key: os.environ.get(key) for key in _TEMP_ENV_KEYS}
    os.environ.update(
        {
            "AUTH_MODE": "single_user",
            "SINGLE_USER_API_KEY": "test-key",
            "JWT_SECRET_KEY": secrets.token_urlsafe(32),
            "DATABASE_URL": f"sqlite:///{root / 'authnz.db'}",
            "USER_DB_BASE_DIR": str(root / "user_databases"),
            "OPENAI_API_KEY": "test-key",
            "OPENAI_API_BASE_URL": "http://mock.local",
        }
    )
    return old


def _restore_environment(old: Mapping[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


async def run_fixture_harness(
    monkeypatch: Any, tmp_path: Path, bind_provider_call: Any
) -> dict[str, Any]:
    """Run through Character_Chat pytest fixtures with disposable paths only."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'authnz.db'}")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://mock.local")
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    try:
        return await _run(monkeypatch=monkeypatch, bind_provider_call=bind_provider_call)
    finally:
        reset_settings()


if __name__ == "__main__":
    stage = "setup"
    root = Path(tempfile.mkdtemp(prefix="uat261_boundary_"))
    old_environment = _set_isolated_environment(root)
    try:
        stage = "complete-v2-boundary"
        result = asyncio.run(_run())
    except Exception as exc:  # Never print response bodies, prompts, credentials, or config.
        terminal_frame = traceback.extract_tb(exc.__traceback__)[-1]
        print(
            f"harness=fail stage={stage} type={type(exc).__name__} source={Path(terminal_frame.filename).name}:{terminal_frame.lineno}",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    finally:
        _restore_environment(old_environment)
        shutil.rmtree(root, ignore_errors=True)
    print(f"harness=pass calls={len(result['calls'])} evidence_sha256={_sha256(EVIDENCE_PATH.read_text(encoding='utf-8'))}")
