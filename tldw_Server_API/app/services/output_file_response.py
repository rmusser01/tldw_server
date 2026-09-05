"""Protected generic output lookup and bounded opened-descriptor responses."""

from __future__ import annotations

import asyncio
import json
import os
import stat
from functools import partial
from secrets import token_hex
from urllib.parse import quote

import anyio
from fastapi import HTTPException
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

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.services.output_file_operations import _require_identity, _stat_optional, _wait_worker
from tldw_Server_API.app.services.reading_artifact_cleanup_service import (
    ReadingStorageBusy,
    ReadingStorageUnavailable,
    _validated_storage_directory,
)


async def protected_output_response(
    db: CollectionsDatabase,
    *,
    output_id: int | None = None,
    title: str | None = None,
    format_: str | None = None,
    head_only: bool = False,
) -> Response | None:
    """Open activated downloads safely; None permits only genuinely inactive legacy dispatch."""
    response = None
    try:
        with anyio.CancelScope(shield=True):
            response, cancelled = await _wait_worker(
                partial(
                    _open_output_response,
                    db,
                    output_id=output_id,
                    title=title,
                    format_=format_,
                )
            )
        if cancelled:
            raise asyncio.CancelledError
        await anyio.lowlevel.checkpoint_if_cancelled()
        if head_only and response is not None:
            # Preserve the generic HEAD route's existing headers and range policy.
            headers = {key: response.headers[key] for key in ("content-type", "content-length")}
            response.close()
            return Response(headers=headers)
        return response
    except BaseException:
        if response is not None:
            response.close()
        raise


def _open_output_response(db, *, output_id, title, format_):
    response = None
    try:
        namespace = db.get_output_read_namespace()
        if namespace is None:
            return None
        root = DatabasePaths.resolve_user_base_directory(db.user_id) / DatabasePaths.OUTPUTS_SUBDIR
        with _validated_storage_directory(root, storage_namespace_id=namespace) as (_, directory):
            row, proof = db.get_output_file_read_state(namespace, output_id=output_id, title=title, format_=format_)
            name = db._output_operation_filename(row.storage_path)
            media_type = {
                "md": "text/markdown; charset=utf-8",
                "html": "text/html; charset=utf-8",
                "mp3": "audio/mpeg",
            }.get(row.format.lower(), "application/octet-stream")
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode):
                    raise RuntimeError("output_storage_unavailable")
                if proof is None:
                    if info.st_nlink != 1:
                        raise RuntimeError("output_storage_unavailable")
                else:
                    identity = json.loads(proof["publication_identity_json"] or "null")
                    stage_name = proof["stage_path"]
                    if (
                        not isinstance(identity, dict)
                        or identity.get("nlink") != 2
                        or any(type(value) is not int for value in identity.values())
                        or not isinstance(stage_name, str)
                        or not stage_name.startswith(".output-stage-")
                        or len(stage_name) != len(".output-stage-") + 32
                        or any(char not in "0123456789abcdef" for char in stage_name[len(".output-stage-") :])
                    ):
                        raise RuntimeError("output_storage_unavailable")
                    stage = _stat_optional(directory, stage_name)
                    _require_identity(info, {**identity, "nlink": 2 if stage else 1}, size=proof["written_bytes"])
                    if stage is not None:
                        _require_identity(stage, identity, size=proof["written_bytes"])
            except BaseException:
                os.close(fd)
                raise
            response = OpenedOutputResponse(fd, filename=name, media_type=media_type)
        return response
    except BaseException as exc:
        if response is not None:
            response.close()
        if isinstance(exc, KeyError):
            raise HTTPException(404, "output_not_found") from None
        if isinstance(exc, FileNotFoundError):
            raise HTTPException(404, "file_missing") from None
        if isinstance(exc, ReadingStorageBusy):
            raise HTTPException(409, "output_file_busy") from None
        if isinstance(exc, (OSError, ReadingStorageUnavailable, RuntimeError, ValueError, TypeError, DatabaseError)):
            raise HTTPException(503, "output_storage_unavailable") from None
        raise


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
