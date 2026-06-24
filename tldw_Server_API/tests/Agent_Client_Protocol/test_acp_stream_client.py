import asyncio
import json
from typing import Any

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.stream_client import ACPStreamClient
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import (
    ACPMessage,
    ACPResponseError,
    ACPStdioClient,
)


@pytest.mark.asyncio
async def test_stream_client_call_roundtrip() -> None:
    sent: list[bytes] = []

    async def send_bytes(data: bytes) -> None:
        sent.append(data)

    client = ACPStreamClient(send_bytes=send_bytes)
    await client.start()

    task = asyncio.create_task(client.call("ping", {"a": 1}))
    await asyncio.sleep(0)

    assert sent, "client did not send request"
    payload = json.loads(sent[0].decode("utf-8").strip())
    assert payload["method"] == "ping"
    req_id = payload["id"]

    await client.feed_bytes(
        json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {"ok": True}}).encode("utf-8")
        + b"\n"
    )
    resp = await task
    assert resp.result == {"ok": True}


@pytest.mark.asyncio
async def test_stream_client_notification_handler() -> None:
    seen: list[ACPMessage] = []

    async def send_bytes(_: bytes) -> None:
        return

    async def on_note(msg: ACPMessage) -> None:
        seen.append(msg)

    client = ACPStreamClient(send_bytes=send_bytes)
    client.set_notification_handler(on_note)
    await client.start()

    await client.feed_bytes(
        json.dumps({"jsonrpc": "2.0", "method": "session/update", "params": {"x": 1}}).encode("utf-8")
        + b"\n"
    )

    assert len(seen) == 1
    assert seen[0].method == "session/update"


@pytest.mark.asyncio
async def test_stream_client_request_handler() -> None:
    sent: list[bytes] = []

    async def send_bytes(data: bytes) -> None:
        sent.append(data)

    async def on_request(msg: ACPMessage) -> ACPMessage:
        return ACPMessage(jsonrpc="2.0", id=msg.id, result={"outcome": {"outcome": "approved"}})

    client = ACPStreamClient(send_bytes=send_bytes)
    client.set_request_handler(on_request)
    await client.start()

    await client.feed_bytes(
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "session/request_permission", "params": {}}).encode("utf-8")
        + b"\n"
    )

    assert sent, "client did not send response"
    payload = json.loads(sent[0].decode("utf-8").strip())
    assert payload["id"] == 7
    assert payload["result"]["outcome"]["outcome"] == "approved"


@pytest.mark.asyncio
async def test_stream_client_call_times_out_and_cleans_pending() -> None:
    async def send_bytes(_: bytes) -> None:
        return None

    client = ACPStreamClient(send_bytes=send_bytes, rpc_timeout_sec=0.01)
    await client.start()

    with pytest.raises(ACPResponseError, match="timed out"):
        await client.call("ping", {})

    assert client._pending == {}


@pytest.mark.asyncio
async def test_stdio_client_call_times_out_and_cleans_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []
    client = ACPStdioClient("agent", [], rpc_timeout_sec=0.01)
    client._proc = object()

    async def send(payload: dict[str, Any]) -> None:
        sent.append(payload)

    monkeypatch.setattr(client, "_send", send)

    with pytest.raises(ACPResponseError, match="timed out"):
        await client.call("ping", {})

    assert sent and sent[0]["method"] == "ping"
    assert client._pending == {}
