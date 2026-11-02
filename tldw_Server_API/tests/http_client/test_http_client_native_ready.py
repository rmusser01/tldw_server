"""Native dispatch fences must follow preflight and cannot introduce retries."""

import httpx
import pytest

from tldw_Server_API.app.core import http_client as hc
from tldw_Server_API.app.core.exceptions import EgressPolicyError, NetworkError

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
URL = "https://ready.example/completion"


async def test_native_ready_follows_all_egress_and_pin_checks(monkeypatch):
    events = []

    async def validate(*args, **kwargs):
        events.append("egress")

    def check_pins(*args, **kwargs):
        events.append("pins")

    async def ready():
        events.append("ready")

    async def send(request):
        events.append("send")
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", validate)
    monkeypatch.setattr(hc, "_check_cert_pinning", check_pins)
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        client._tldw_cert_pinning = {"ready.example": {"a" * 64}}
        result = await hc.afetch_json(
            method="POST",
            url=URL,
            client=client,
            max_bytes=100,
            allow_redirects=False,
            retry=hc.RetryPolicy(attempts=1),
            on_native_ready=ready,
        )
    assert result == {"ok": True}
    assert events == ["egress", "egress", "pins", "ready", "send"]


@pytest.mark.parametrize("entry", ["json", "stream", "adapter", "native"])
@pytest.mark.parametrize("unsafe", ["retry", "implicit_retry", "implicit_client"])
async def test_native_ready_rejects_unsafe_direct_callers_before_admission(monkeypatch, entry, unsafe):
    events = []

    async def validate(*args, **kwargs):
        events.append("egress")

    async def ready():
        events.append("ready")

    async def send(request):
        events.append("send")
        return httpx.Response(200, json={})

    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", validate)
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        kwargs = {
            "method": "POST",
            "url": URL,
            "client": None if unsafe == "implicit_client" else client,
            "retry": None if unsafe == "implicit_retry" else hc.RetryPolicy(attempts=2 if unsafe == "retry" else 1),
            "on_native_ready": ready,
        }
        with pytest.raises(ValueError) as rejected:
            if entry == "json":
                await hc.afetch_json(max_bytes=100, allow_redirects=False, **kwargs)
            else:
                method = {
                    "stream": hc.astream_bytes,
                    "adapter": hc.HttpxAdapter().stream_bytes,
                    "native": hc._astream_bytes_httpx,
                }[entry]
                stream = method(**kwargs)
                try:
                    await stream.__anext__()
                finally:
                    await stream.aclose()
    expected = (
        "Native-ready callbacks require HTTPX"
        if unsafe == "implicit_client" and entry in {"json", "stream"}
        else "Native-ready callbacks require explicit single-attempt HTTPX"
    )
    assert str(rejected.value) == expected
    assert events == []


@pytest.mark.parametrize("unsafe", ["unbounded", "redirects", "implicit_redirects"])
async def test_native_ready_json_rejects_unbounded_or_redirecting_calls(unsafe):
    events = []

    async def ready():
        events.append("ready")

    async def send(request):
        events.append("send")
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        kwargs = {} if unsafe == "implicit_redirects" else {"allow_redirects": unsafe == "redirects"}
        with pytest.raises(ValueError) as rejected:
            await hc.afetch_json(
                method="POST",
                url=URL,
                client=client,
                max_bytes=None if unsafe == "unbounded" else 100,
                retry=hc.RetryPolicy(attempts=1),
                on_native_ready=ready,
                **kwargs,
            )
    assert str(rejected.value) == "Native-ready callbacks require bounded JSON without redirects"
    assert events == []


@pytest.mark.parametrize("denied_check", [1, 2])
async def test_native_ready_never_runs_on_egress_denial(monkeypatch, denied_check):
    events = []

    async def validate(*args, **kwargs):
        events.append("egress")
        if len(events) == denied_check:
            raise EgressPolicyError("denied")

    async def ready():
        events.append("ready")

    async def send(request):
        events.append("send")
        return httpx.Response(200, json={})

    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", validate)
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with pytest.raises(EgressPolicyError) as denied:
            await hc.afetch_json(
                method="POST",
                url=URL,
                client=client,
                max_bytes=100,
                allow_redirects=False,
                retry=hc.RetryPolicy(attempts=1),
                on_native_ready=ready,
            )
    assert str(denied.value) == "denied"
    assert events == ["egress"] * denied_check


@pytest.mark.parametrize("error_type", [NetworkError, httpx.ConnectError])
async def test_native_ready_callback_error_is_not_retried_or_sent(monkeypatch, error_type):
    events = []
    original_error = error_type("private")

    async def validate(*args, **kwargs):
        pass

    async def ready():
        events.append("ready")
        raise original_error

    async def send(request):
        events.append("send")
        return httpx.Response(200, json={})

    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", validate)
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with pytest.raises(error_type) as failed:
            await hc.afetch_json(
                method="POST",
                url=URL,
                client=client,
                max_bytes=100,
                allow_redirects=False,
                retry=hc.RetryPolicy(attempts=1),
                on_native_ready=ready,
            )
    assert failed.value is original_error
    assert events == ["ready"]


async def test_native_ready_single_attempt_cannot_change_during_egress(monkeypatch):
    events = []
    retry = hc.RetryPolicy(attempts=1, backoff_base_ms=0, backoff_cap_s=0)

    async def validate(*args, **kwargs):
        retry.attempts = 3

    async def ready():
        events.append("ready")

    async def send(request):
        events.append("send")
        return httpx.Response(500, json={})

    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", validate)
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        result = await hc.afetch_json(
            method="GET",
            url=URL,
            client=client,
            max_bytes=100,
            allow_redirects=False,
            retry=retry,
            on_native_ready=ready,
        )
    assert result == {}
    assert events == ["ready", "send"]


@pytest.mark.parametrize("entry", ["json", "stream", "adapter"])
async def test_absent_native_ready_adds_no_keyword_to_legacy_delegates(monkeypatch, entry):
    forwarded = []

    async def legacy_stream(**kwargs):
        forwarded.append(kwargs)
        yield b"{}"

    if entry == "json":
        monkeypatch.setattr(hc, "astream_bytes", legacy_stream)
        await hc.afetch_json(method="GET", url=URL, max_bytes=100, on_native_ready=None)
    else:
        monkeypatch.setattr(hc, "_astream_bytes_httpx", legacy_stream)
        method = hc.astream_bytes if entry == "stream" else hc.HttpxAdapter().stream_bytes
        async with httpx.AsyncClient() as client:
            stream = method(method="GET", url=URL, client=client, on_native_ready=None)
            try:
                assert [chunk async for chunk in stream] == [b"{}"]
            finally:
                await stream.aclose()
    assert len(forwarded) == 1
    assert "on_native_ready" not in forwarded[0]
