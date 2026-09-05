"""Bounded responses over an already-authorized file descriptor; lookup wiring pending."""

from __future__ import annotations

import asyncio
import os
import stat
from functools import partial
from secrets import token_hex
from urllib.parse import quote

import anyio
from starlette.datastructures import Headers
from starlette.responses import (
    FileResponse,
    MalformedRangeHeader,
    PlainTextResponse,
    RangeNotSatisfiable,
    Response,
    StreamingResponse,
)
from starlette.types import Receive, Scope, Send

from tldw_Server_API.app.services.output_file_operations import _wait_worker


class OpenedOutputResponse(StreamingResponse):
    """Own one opened descriptor, never a path that could resolve to new bytes.

    The caller must authorize lookup/open under verified storage exclusion first.
    Ownership transfers on construction, including failed setup. Invoke once, or
    call ``close`` if abandoning the response before invocation. Construction
    includes fstat and belongs in the caller's offloaded protected-open interval.
    """

    chunk_size = FileResponse.chunk_size

    def __init__(self, fd: int, *, filename: str, media_type: str) -> None:
        self._fd: int | None = fd
        try:
            self.stat_result = os.fstat(fd)
            if not stat.S_ISREG(self.stat_result.st_mode):
                raise RuntimeError("output_source_unavailable")
            self._ranges = [(0, self.stat_result.st_size)]
            self._boundary: str | None = None
            self._range_header = None
            super().__init__(self._stream(), media_type=media_type)
            FileResponse.set_stat_headers(self, self.stat_result)
            self.headers["accept-ranges"] = "bytes"
            escaped = quote(filename)
            self.headers["content-disposition"] = (
                f"attachment; filename*=utf-8''{escaped}"
                if escaped != filename
                else f'attachment; filename="{filename}"'
            )
        except BaseException as exc:
            self.close()
            if isinstance(exc, OSError):
                raise RuntimeError("output_storage_unavailable") from None
            raise

    def close(self) -> None:
        """Release descriptor ownership once, including an uninvoked response."""
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            if self._fd is None:
                raise RuntimeError("output_response_closed")
            headers = Headers(scope=scope)
            requested = headers.get("range")
            if_range = headers.get("if-range")
            if requested is not None and (if_range is None or FileResponse._should_use_range(self, if_range)):
                try:
                    self._ranges = FileResponse._parse_range_header(requested, self.stat_result.st_size)
                except MalformedRangeHeader as exc:
                    return await PlainTextResponse(exc.content, status_code=400)(scope, receive, send)
                except RangeNotSatisfiable as exc:
                    return await PlainTextResponse(
                        status_code=416, headers={"Content-Range": f"bytes */{exc.max_size}"}
                    )(scope, receive, send)
                self.status_code = 206
                if len(self._ranges) == 1:
                    start, end = self._ranges[0]
                    self.headers["content-range"] = f"bytes {start}-{end - 1}/{self.stat_result.st_size}"
                    self.headers["content-length"] = str(end - start)
                else:
                    self._boundary = token_hex(13)
                    length, self._range_header = FileResponse.generate_multipart(
                        self, self._ranges, self._boundary, self.stat_result.st_size, self.headers["content-type"]
                    )
                    self.headers["content-type"] = f"multipart/byteranges; boundary={self._boundary}"
                    self.headers["content-length"] = str(length)
            if scope["method"].upper() == "HEAD":
                return await Response(status_code=self.status_code, headers=dict(self.headers))(scope, receive, send)
            # StreamingResponse handles disconnects; it never uses pathsend.
            await super().__call__(scope, receive, send)
        finally:
            self.close()

    async def _stream(self):
        for start, end in self._ranges:
            if self._range_header is not None:
                yield self._range_header(start, end)
            while start < end:
                try:
                    with anyio.CancelScope(shield=True):
                        chunk, cancelled = await _wait_worker(
                            partial(os.pread, self._fd, min(self.chunk_size, end - start), start)
                        )
                except OSError:
                    raise RuntimeError("output_storage_unavailable") from None
                if cancelled:
                    raise asyncio.CancelledError
                await anyio.lowlevel.checkpoint_if_cancelled()
                if not chunk:
                    raise RuntimeError("output_source_unavailable")
                start += len(chunk)
                yield chunk
            if self._boundary is not None:
                yield b"\r\n"
        if self._boundary is not None:
            yield f"--{self._boundary}--".encode("ascii")
