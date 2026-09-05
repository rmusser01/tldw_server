"""Opened-descriptor response lifetime, bounded streaming and HTTP compatibility."""

from __future__ import annotations

import asyncio
import os
from threading import Event, get_ident

import anyio
import pytest
from starlette import responses as starlette_responses
from starlette.requests import ClientDisconnect
from starlette.responses import FileResponse

from tldw_Server_API.app.services.output_file_response import OpenedOutputResponse

pytestmark = [pytest.mark.unit, pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor protocol")]


def opened_response(fd, **kwargs):
    return OpenedOutputResponse(fd, **kwargs)


async def invoke(response, *, method="GET", headers=(), on_send=None, extensions=None):
    messages = []

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)
        if on_send:
            await on_send(message)

    await response(
        {
            "type": "http",
            "method": method,
            "headers": list(headers),
            "asgi": {"spec_version": "2.4"},
            "extensions": extensions or {},
        },
        receive,
        send,
    )
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in messages)
    return start["status"], dict(start["headers"]), body, messages


@pytest.mark.asyncio
async def test_legacy_path_response_reopens_recycled_path(tmp_path):
    """Negative control: current pathname response has the race being replaced."""
    path = tmp_path / "report.md"
    path.write_bytes(b"original")

    async def replace(message):
        if message["type"] == "http.response.start":
            path.unlink()
            path.write_bytes(b"intruder")

    status, _, body, _ = await invoke(FileResponse(path), on_send=replace)
    assert status == 200 and body == b"intruder"


@pytest.mark.asyncio
@pytest.mark.parametrize("range_header", [None, b"bytes=1-3", b"bytes=0-1,5-6"])
async def test_descriptor_bytes_survive_path_reuse_after_headers(tmp_path, range_header):
    path = tmp_path / "report.md"
    path.write_bytes(b"original")
    fd = os.open(path, os.O_RDONLY)
    try:
        response = opened_response(fd, filename=path.name, media_type="text/markdown")
    except BaseException:
        os.close(fd)
        raise

    async def replace(message):
        if message["type"] == "http.response.start":
            path.unlink()
            path.write_bytes(b"intruder with a different length")

    status, headers, body, messages = await invoke(
        response,
        headers=[(b"range", range_header)] if range_header else [],
        on_send=replace,
        extensions={"http.response.pathsend": {}},
    )
    assert status == (206 if range_header else 200)
    assert b"intruder" not in body
    if range_header is None:
        assert body == b"original"
    elif b"," not in range_header:
        assert body == b"rig" and headers[b"content-range"] == b"bytes 1-3/8"
    else:
        assert b"\r\nor\r\n" in body and b"\r\nna\r\n" in body
    assert int(headers[b"content-length"]) == len(body)
    assert all(message["type"] != "http.response.pathsend" for message in messages)
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, request_headers",
    [
        ("GET", []),
        ("HEAD", []),
        ("GET", [(b"range", b"bytes=2-5")]),
        ("GET", [(b"range", b"bytes=-3")]),
        ("GET", [(b"range", b"bytes=2-")]),
        ("GET", [(b"range", b"bytes=0-1,5-7")]),
        ("GET", [(b"range", b"bytes=0-4,2-8")]),
        ("HEAD", [(b"range", b"bytes=0-1,5-7")]),
        ("GET", [(b"range", b"bytes=99-")]),
        ("GET", [(b"range", b"nonsense")]),
        ("GET", [(b"range", b"bytes=2-5"), (b"if-range", b"stale")]),
    ],
)
async def test_descriptor_preserves_starlette_http_behavior(tmp_path, monkeypatch, method, request_headers):
    from tldw_Server_API.app.services import output_file_response

    path = tmp_path / "report.md"
    path.write_bytes(b"0123456789")
    monkeypatch.setattr(starlette_responses, "token_hex", lambda _length: "fixed-boundary")
    monkeypatch.setattr(output_file_response, "token_hex", lambda _length: "fixed-boundary")
    original = FileResponse(path, filename="report \u00e9.md", media_type="text/markdown")
    expected = await invoke(original, method=method, headers=request_headers)
    fd = os.open(path, os.O_RDONLY)
    actual = await invoke(
        opened_response(fd, filename="report \u00e9.md", media_type="text/markdown"),
        method=method,
        headers=request_headers,
    )
    assert actual[:3] == expected[:3]
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("validator", ["etag", "last-modified", "cache"])
async def test_descriptor_validators_derive_from_opened_inode(tmp_path, validator):
    path = tmp_path / "report.md"
    path.write_bytes(b"original")
    fd = os.open(path, os.O_RDONLY)
    response = opened_response(fd, filename=path.name, media_type="text/markdown")
    headers = dict(response.headers)
    path.unlink()
    path.write_bytes(b"other content")
    os.utime(path, (0, 0))
    if validator == "cache":
        requested = [
            (b"if-none-match", headers["etag"].encode()),
            (b"if-modified-since", headers["last-modified"].encode()),
        ]
    else:
        requested = [(b"range", b"bytes=1-3"), (b"if-range", headers[validator].encode())]
    status, actual_headers, body, _ = await invoke(response, headers=requested)
    assert status == (200 if validator == "cache" else 206)
    assert body == (b"original" if validator == "cache" else b"rig")
    assert actual_headers[b"etag"] == headers["etag"].encode()
    assert actual_headers[b"last-modified"] == headers["last-modified"].encode()
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [0, 1, 64 * 1024, 3 * 1024 * 1024 + 1])
async def test_descriptor_reads_are_bounded_offloaded_and_ignore_current_offset(tmp_path, monkeypatch, size):
    path = tmp_path / "report.md"
    path.write_bytes(b"x" * size)
    fd = os.open(path, os.O_RDONLY)
    os.lseek(fd, size, os.SEEK_SET)
    response = opened_response(fd, filename=path.name, media_type="text/markdown")
    real_read, main_thread, reads = os.pread, get_ident(), []

    def read(descriptor, amount, offset):
        assert get_ident() != main_thread
        reads.append((amount, offset))
        return real_read(descriptor, amount, offset)

    monkeypatch.setattr(os, "pread", read)
    _, headers, body, messages = await invoke(response)
    assert body == b"x" * size and int(headers[b"content-length"]) == size
    assert all(0 < amount <= 64 * 1024 for amount, _ in reads)
    assert all(len(message.get("body", b"")) <= 64 * 1024 for message in messages)
    assert bool(reads) == bool(size)
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["read", "short_read", "send", "send_disconnect"])
async def test_descriptor_errors_close_and_sanitize_file_failures(tmp_path, monkeypatch, failure):
    path = tmp_path / "report.md"
    path.write_bytes(b"original")
    fd = os.open(path, os.O_RDONLY)
    response = opened_response(fd, filename=path.name, media_type="text/markdown")

    def read(*args):
        if failure == "short_read":
            return b""
        raise OSError("private file details")

    async def send(message):
        if message["type"] == "http.response.body":
            if failure == "send":
                raise ValueError("send failed")
            if failure == "send_disconnect":
                raise OSError("peer disconnected")

    if not failure.startswith("send"):
        monkeypatch.setattr(os, "pread", read)
    expected = (
        "send failed"
        if failure == "send"
        else "output_source_unavailable" if failure == "short_read" else "output_storage_unavailable"
    )
    error = ClientDisconnect if failure == "send_disconnect" else (RuntimeError, ValueError)
    with pytest.raises(error, match="^$" if failure == "send_disconnect" else f"^{expected}$"):
        await asyncio.wait_for(invoke(response, on_send=send), 5)
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_kind", ["asyncio", "anyio", "disconnect", "asyncio_http24", "anyio_http24"])
async def test_descriptor_cancellation_drains_read_before_closing(tmp_path, monkeypatch, cancel_kind):
    path = tmp_path / "report.md"
    path.write_bytes(b"original")
    fd = os.open(path, os.O_RDONLY)
    response = opened_response(fd, filename=path.name, media_type="text/markdown")
    real_read, entered, release = os.pread, Event(), Event()
    disconnect = asyncio.Event()

    def read(*args):
        entered.set()
        assert release.wait(10)
        assert os.fstat(fd).st_size == 8, "response closed a descriptor with an active reader"
        return real_read(*args)

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(_message):
        pass

    async def serve():
        await response(
            {
                "type": "http",
                "method": "GET",
                "headers": [],
                "asgi": {"spec_version": "2.4" if cancel_kind.endswith("http24") else "2.3"},
            },
            receive,
            send,
        )

    monkeypatch.setattr(os, "pread", read)
    try:
        if cancel_kind.startswith("anyio"):
            async with anyio.create_task_group() as group:
                group.start_soon(serve)
                assert await asyncio.to_thread(entered.wait, 10)
                group.cancel_scope.cancel()
                with anyio.CancelScope(shield=True):
                    await anyio.sleep(0)
                    assert os.fstat(fd).st_size == 8
                    release.set()
        else:
            task = asyncio.create_task(serve())
            assert await asyncio.to_thread(entered.wait, 10)
            if cancel_kind.startswith("asyncio"):
                task.cancel()
            else:
                disconnect.set()
            await asyncio.sleep(0)
            assert not task.done()
            assert os.fstat(fd).st_size == 8
            release.set()
            if cancel_kind.startswith("asyncio"):
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
    finally:
        release.set()
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.asyncio
async def test_uninvoked_response_can_be_closed_without_reclosing_reused_fd(tmp_path):
    path = tmp_path / "report.md"
    path.write_bytes(b"original")
    fd = os.open(path, os.O_RDONLY)
    response = opened_response(fd, filename=path.name, media_type="text/markdown")
    response.close()
    replacement = os.open(path, os.O_RDONLY)
    try:
        response.close()
        with pytest.raises(RuntimeError, match="^output_response_closed$"):
            await invoke(response)
        assert os.fstat(replacement).st_size == 8
    finally:
        os.close(replacement)


def test_descriptor_construction_error_closes_and_sanitizes(tmp_path, monkeypatch):
    path = tmp_path / "report.md"
    path.write_bytes(b"original")
    fd = os.open(path, os.O_RDONLY)
    real_stat = os.fstat

    def fail_stat(_fd):
        raise OSError("private storage path details")

    monkeypatch.setattr(os, "fstat", fail_stat)
    with pytest.raises(RuntimeError, match="^output_storage_unavailable$"):
        opened_response(fd, filename=path.name, media_type="text/markdown")
    with pytest.raises(OSError):
        real_stat(fd)


def test_header_setup_failure_releases_transferred_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "report.md"
    path.write_bytes(b"original")
    fd = os.open(path, os.O_RDONLY)

    def fail_headers(_response, _info):
        assert os.fstat(fd).st_size == 8
        raise RuntimeError("header setup failed")

    monkeypatch.setattr(FileResponse, "set_stat_headers", fail_headers)
    with pytest.raises(RuntimeError, match="^header setup failed$"):
        opened_response(fd, filename=path.name, media_type="text/markdown")
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("kind", ["directory", "pipe"])
def test_descriptor_construction_rejects_nonregular_file_and_closes(tmp_path, kind):
    if kind == "directory":
        fd, write_fd = os.open(tmp_path, os.O_RDONLY), None
    else:
        fd, write_fd = os.pipe()
    try:
        with pytest.raises(RuntimeError, match="^output_source_unavailable$"):
            opened_response(fd, filename="report.md", media_type="text/markdown")
        with pytest.raises(OSError):
            os.fstat(fd)
    finally:
        if write_fd is not None:
            os.close(write_fd)


@pytest.mark.asyncio
async def test_successful_response_cannot_be_replayed(tmp_path):
    path = tmp_path / "report.md"
    path.write_bytes(b"original")
    fd = os.open(path, os.O_RDONLY)
    response = opened_response(fd, filename=path.name, media_type="text/markdown")
    assert (await invoke(response))[2] == b"original"
    with pytest.raises(RuntimeError, match="^output_response_closed$"):
        await invoke(response)
