"""Strict-only raw body bounds and content-free validation preserve ASGI receive."""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.requests import ClientDisconnect
from starlette.responses import Response

from tldw_Server_API.app.api.v1.schemas.workspace_chat_startup_schemas import (
    STARTUP_BODY_BYTES_MAX,
    WorkspaceChatStartupRequest,
)

pytestmark = pytest.mark.unit

_VALID = {"scope_type": "workspace", "workspace_id": "ws", "workspace_assistant_selection": "none"}
_PRIVATE = "private-profile-credential-content"


def _transport():
    """Load the new contract at execution time so RED is not a collection failure."""
    name = "tldw_Server_API.app.api.v1.endpoints.workspace_chat_startup_transport"
    assert importlib.util.find_spec(name) is not None, "Strict startup transport is not implemented"
    return importlib.import_module(name)


async def _run(
    chunks: list[bytes], *, headers: list[tuple[bytes, bytes]] | None = None,
    after_body: bool = False, legacy: bool = False, abort: str | None = None,
    effects: list[str] | None = None, observe_dependency: bool = False, query: bytes = b"",
) -> tuple[Any, list[str], list[dict[str, Any]]]:
    """Exercise the real FastAPI ASGI route with scripted chunks and effect observation."""
    route_class = APIRoute if legacy else _transport().WorkspaceStartupRoute
    effects = effects if effects is not None else []
    receives: list[dict[str, Any]] = []
    events = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]

    async def receive():
        """Expose real exhaustion/disconnect instead of endless synthetic empty bodies."""
        if abort == "cancel":
            raise asyncio.CancelledError
        if abort == "disconnect":
            event = {"type": "http.disconnect"}
        else:
            event = events.pop(0) if events else {"type": "http.disconnect"}
        receives.append(event)
        return event

    async def database_dependency() -> None:
        """Observe dependency initialization that may schedule writes in production."""
        if observe_dependency:
            effects.append("dependency")

    async def endpoint(
        request: Request, payload: WorkspaceChatStartupRequest, database: None = Depends(database_dependency),
    ):
        """Stand in only for post-validation writes, never for the real model/parser."""
        effects.append("handler")
        if after_body:
            assert await request.receive() == {"type": "http.disconnect"}
        return {"workspace_id": payload.workspace_id}

    app = FastAPI()
    app.router.add_api_route("/strict", endpoint=endpoint, methods=["POST"], route_class_override=route_class)
    scope = {
        "type": "http", "method": "POST", "path": "/strict", "root_path": "",
        "scheme": "http", "http_version": "1.1", "query_string": query, "path_params": {},
        "headers": headers if headers is not None else [
            (b"content-type", b"application/json"), (b"idempotency-key", b"accepted"),
        ],
    }
    sent: list[dict[str, Any]] = []

    async def send(event):
        """Capture normal ASGI response messages without replacing framework stacks."""
        sent.append(event)

    await app(scope, receive, send)
    response = Response(
        content=b"".join(event.get("body", b"") for event in sent),
        status_code=next(event["status"] for event in sent if event["type"] == "http.response.start"),
    )
    return response, effects, receives


@pytest.mark.parametrize("size", [STARTUP_BODY_BYTES_MAX - 1, STARTUP_BODY_BYTES_MAX, STARTUP_BODY_BYTES_MAX + 1])
@pytest.mark.parametrize("declared", [None, b"1", b"65536"])
async def test_raw_ceiling_does_not_trust_content_length(size: int, declared: bytes | None) -> None:
    """The raw ceiling is exact even with missing or dishonest Content-Length."""
    raw = json.dumps(_VALID).encode()
    raw += b" " * (size - len(raw))
    headers = [(b"content-type", b"application/json"), (b"idempotency-key", b"accepted")]
    if declared is not None:
        headers.append((b"content-length", declared))
    response, effects, _ = await _run([raw], headers=headers)
    assert response.status_code == (413 if size > STARTUP_BODY_BYTES_MAX else 200)
    assert effects == ([] if size > STARTUP_BODY_BYTES_MAX else ["handler"])


async def test_streaming_ceiling_stops_at_first_excess_chunk() -> None:
    """Reject accumulated bytes before consuming subsequent chunks or invoking the model."""
    response, effects, receives = await _run([b" " * 32768, b" " * 32768, b"x", b"unread"])
    assert response.status_code == 413
    assert effects == []
    assert len(receives) == 3


async def test_exact_ceiling_can_arrive_in_multiple_chunks() -> None:
    """A valid bounded body is reconstructed once, regardless of ASGI chunk boundaries."""
    raw = json.dumps(_VALID).encode()
    raw += b" " * (STARTUP_BODY_BYTES_MAX - len(raw))
    response, effects, receives = await _run([raw[:5], raw[5:100], raw[100:]])
    assert response.status_code == 200
    assert effects == ["handler"]
    assert len(receives) == 3


@pytest.mark.parametrize("raw", [b"\xff", b'{"title":"\xed\xa0\x80"}', b'{"title":"\xc0\x80"}'])
async def test_invalid_raw_utf8_is_content_free_before_handler(raw: bytes) -> None:
    """Reject illegal UTF-8 without echoing private bytes or reaching business effects."""
    response, effects, _ = await _run([raw])
    assert response.status_code == 422
    assert json.loads(response.body)["detail"] == [{
        "loc": ["body"], "msg": "Invalid Workspace startup request", "type": "value_error",
    }]
    assert effects == []


@pytest.mark.parametrize("fields", [
    {"state": _PRIVATE}, {"title": "x" * 4097}, {"title": "\x00"}, {"title": "\ud800"},
    {_PRIVATE: _PRIVATE}, {"\ud800": _PRIVATE}, {"\x00": _PRIVATE},
    {"workspace_assistant_default_version": None},
])
async def test_decoded_and_unknown_field_errors_never_echo_input(fields: dict[str, Any]) -> None:
    """Escaped input still obeys field limits; unsafe error locations/messages stay private."""
    raw = json.dumps({**_VALID, **fields}).encode()
    response, effects, _ = await _run([raw])
    assert response.status_code == 422
    detail = json.loads(response.body)["detail"]
    assert all(item["msg"] == "Invalid Workspace startup request" and item["type"] == "value_error" for item in detail)
    assert _PRIVATE not in response.body.decode("utf-8")
    assert all(set(item) == {"loc", "msg", "type"} for item in detail)
    assert effects == []


@pytest.mark.parametrize("fields", [{"state": _PRIVATE}, {_PRIVATE: _PRIVATE}])
async def test_invalid_model_cannot_initialize_database_dependency(fields: dict[str, Any]) -> None:
    """FastAPI dependency initialization must not precede strict model rejection."""
    response, effects, _ = await _run([json.dumps({**_VALID, **fields}).encode()], observe_dependency=True)
    assert response.status_code == 422
    assert effects == []


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"])
async def test_json_encoding_detection_cannot_bypass_utf8(encoding: str) -> None:
    """UTF-16/32 ASCII bytes can decode as UTF-8 but must not become valid JSON."""
    response, effects, _ = await _run([json.dumps(_VALID).encode(encoding)], observe_dependency=True)
    assert response.status_code == 422
    assert effects == []


@pytest.mark.parametrize("media_type", [None, b"text/plain", b"application/x/y+json"])
async def test_non_json_content_type_rejects_before_database_dependency(media_type: bytes | None) -> None:
    """Missing/malformed/non-JSON media types cannot initialize storage."""
    headers = [(b"idempotency-key", b"accepted")]
    if media_type is not None:
        headers.append((b"content-type", media_type))
    response, effects, _ = await _run(
        [json.dumps(_VALID).encode()], headers=headers, observe_dependency=True,
    )
    assert response.status_code == 422
    assert effects == []


@pytest.mark.parametrize("keys", [[], [b""], [b"bad key"], [b"x" * 129], [b"same", b"same"]])
async def test_required_key_syntax_and_duplicates_reject_before_dependencies(keys: list[bytes]) -> None:
    """Even duplicate identical keys cannot initialize storage or reach business logic."""
    headers = [(b"content-type", b"application/json"), *((b"idempotency-key", key) for key in keys)]
    response, effects, _ = await _run([json.dumps(_VALID).encode()], headers=headers, observe_dependency=True)
    assert response.status_code == 422
    assert effects == []


@pytest.mark.parametrize("query", [b"seed_first_message=false", b"workspace_id=ws", b"scope_type=workspace", b"", b"private-secret=", b"&&"])
async def test_strict_query_options_are_not_silently_ignored(query: bytes) -> None:
    """Strict startup rejects every supplied option, leaving the empty query valid."""
    response, effects, _ = await _run([json.dumps(_VALID).encode()], query=query, observe_dependency=True)
    assert response.status_code == (422 if query else 200)
    assert effects == ([] if query else ["dependency", "handler"])
    assert "private-secret" not in response.body.decode()


def test_validation_detail_bounds_and_sanitizes_untrusted_error_records() -> None:
    """Even malicious error metadata yields at most 32 static UTF-8-safe details."""
    error = RequestValidationError([{
        "loc": ("body", _PRIVATE, "\ud800", "title", 999, "body", "title", "body"),
        "msg": _PRIVATE + "\ud800", "type": _PRIVATE, "input": _PRIVATE, "ctx": {"secret": _PRIVATE},
    }] * 40)
    detail = _transport().strict_startup_validation_detail(error)
    assert detail == [{"loc": ["body", "title", "body", "title"],
                       "msg": "Invalid Workspace startup request", "type": "value_error"}] * 32
    assert _PRIVATE not in json.dumps(detail, ensure_ascii=False).encode("utf-8").decode()


async def test_replayed_body_preserves_underlying_disconnect_receive() -> None:
    """After delivering buffered input once, delegate to the original receive callable."""
    response, effects, receives = await _run([json.dumps(_VALID).encode()], after_body=True)
    assert response.status_code == 200
    assert effects == ["handler"]
    assert receives[-1] == {"type": "http.disconnect"}
    assert len(receives) == 2


@pytest.mark.parametrize("abort", ["cancel", "disconnect"])
async def test_raw_read_propagates_cancellation_and_disconnect(abort: str) -> None:
    """A cancelled/disconnected request cannot reach parsing or business effects."""
    effects: list[str] = []
    with pytest.raises(asyncio.CancelledError if abort == "cancel" else ClientDisconnect):
        await _run([], abort=abort, effects=effects)
    assert effects == []


async def test_legacy_routes_do_not_acquire_the_strict_raw_ceiling() -> None:
    """The route override does not impose this new limit on legacy HTTP handlers."""
    raw = json.dumps(_VALID).encode() + b" " * STARTUP_BODY_BYTES_MAX
    response, effects, _ = await _run([raw], legacy=True)
    assert response.status_code == 200
    assert effects == ["handler"]
