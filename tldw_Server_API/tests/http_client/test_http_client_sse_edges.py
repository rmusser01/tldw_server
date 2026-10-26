import pytest


pytestmark = pytest.mark.unit


def _has_httpx():
    try:
        import httpx  # noqa: F401
        return True
    except Exception:
        return False


requires_httpx = pytest.mark.skipif(not _has_httpx(), reason="httpx not installed")


@requires_httpx
@pytest.mark.asyncio
async def test_sse_multiline_and_comments():
    import httpx
    from tldw_Server_API.app.core.http_client import astream_sse, create_async_client

    # Includes comments (:) and multi-line data, and id/retry reordering
    content = (
        b": comment about stream\n\n"
        b"event: notice\n"
        b"id: 42\n"
        b"data: line1\n"
        b"data: line2\n\n"
        b"retry: 5000\n"
        b"data: onlydata\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, content=content, headers={"Content-Type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    client = create_async_client(transport=transport)
    try:
        events = []
        async for ev in astream_sse(url="http://93.184.216.34/stream", client=client):
            events.append(ev)
            if len(events) >= 2:
                break
        assert events[0].event == "notice"
        assert events[0].id == "42"
        assert events[0].data == "line1\nline2"
        assert events[1].event == "message"
        assert events[1].data == "onlydata"
    finally:
        import asyncio
        await client.aclose()


@requires_httpx
@pytest.mark.asyncio
async def test_sse_configured_endpoint_admits_its_private_origin_and_refuses_other_hops(monkeypatch):
    """configured_endpoint scopes astream_sse to one origin, as it does afetch."""
    import httpx

    from tldw_Server_API.app.core.exceptions import EgressPolicyError
    from tldw_Server_API.app.core.http_client import astream_sse, create_async_client
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.path == "/hop":
            return httpx.Response(302, request=request, headers={"Location": "http://10.0.0.9:3001/sse"})
        return httpx.Response(200, request=request, content=b"data: ok\n\n", headers={"Content-Type": "text/event-stream"})

    origin = "http://127.0.0.1:3001"
    scope = ConfiguredEndpointScope.from_url(origin)
    client = create_async_client(transport=httpx.MockTransport(handler))
    try:
        events = [ev async for ev in astream_sse(url=f"{origin}/sse", client=client, configured_endpoint=scope)]
        assert [ev.data for ev in events] == ["ok"]

        with pytest.raises(EgressPolicyError):
            async for _ev in astream_sse(url=f"{origin}/hop", client=client, configured_endpoint=scope):
                pass
        assert requested == [f"{origin}/sse", f"{origin}/hop"]

        # Without the scope the default policy refuses the loopback origin outright.
        with pytest.raises(EgressPolicyError):
            async for _ev in astream_sse(url=f"{origin}/sse", client=client):
                pass
        assert len(requested) == 2
    finally:
        await client.aclose()


@requires_httpx
@pytest.mark.asyncio
async def test_sse_same_origin_redirects_only_refuses_a_cross_origin_hop(monkeypatch):
    """allow_cross_origin_redirects=False follows same-origin hops and refuses others before requesting them."""
    import httpx

    from tldw_Server_API.app.core.exceptions import EgressPolicyError
    from tldw_Server_API.app.core.http_client import astream_sse, create_async_client

    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.path == "/same":
            return httpx.Response(302, request=request, headers={"Location": "/stream"})
        if request.url.path == "/cross":
            return httpx.Response(302, request=request, headers={"Location": "http://93.184.216.35/stream"})
        return httpx.Response(200, request=request, content=b"data: ok\n\n", headers={"Content-Type": "text/event-stream"})

    client = create_async_client(transport=httpx.MockTransport(handler))
    try:
        events = [
            ev
            async for ev in astream_sse(
                url="http://93.184.216.34/same", client=client, allow_cross_origin_redirects=False
            )
        ]
        assert [ev.data for ev in events] == ["ok"]
        assert requested == ["http://93.184.216.34/same", "http://93.184.216.34/stream"]

        with pytest.raises(EgressPolicyError) as refused:
            async for _ev in astream_sse(
                url="http://93.184.216.34/cross", client=client, allow_cross_origin_redirects=False
            ):
                pass
        assert refused.value.reason_code == "origin_mismatch"
        assert requested[2:] == ["http://93.184.216.34/cross"]
    finally:
        await client.aclose()
