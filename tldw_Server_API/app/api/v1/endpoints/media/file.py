# Media File Serving Endpoint
# Serves original files (PDFs, documents) stored for media items
#
from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Response, status
from fastapi.responses import StreamingResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.Storage import get_storage_backend
from tldw_Server_API.app.core.Storage.storage_interface import StorageError

router = APIRouter(tags=["Media Files"])


def _build_content_disposition(disposition_type: str, filename: str) -> str:
    """
    Build RFC 5987 compliant Content-Disposition header.

    Handles both ASCII-safe fallback and UTF-8 encoded filename for
    proper handling of international characters and special characters.

    Args:
        disposition_type: Either "inline" or "attachment"
        filename: The original filename (may contain unicode/special chars)

    Returns:
        Properly formatted Content-Disposition header value
    """
    # ASCII-safe filename (fallback for older browsers)
    ascii_filename = filename.encode('ascii', 'ignore').decode('ascii')
    # Escape quotes in the ASCII filename
    ascii_filename = ascii_filename.replace('"', '\\"')
    # Remove any remaining problematic characters
    ascii_filename = ''.join(c for c in ascii_filename if c.isprintable() and c not in '\r\n')
    if not ascii_filename:
        ascii_filename = "file"

    # UTF-8 encoded filename for modern browsers (RFC 5987)
    utf8_filename = quote(filename, safe='')

    return f'{disposition_type}; filename="{ascii_filename}"; filename*=UTF-8\'\'{utf8_filename}'


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int] | None:
    """
    Parse RFC 7233 Range header.

    Supports single byte ranges only (no multipart ranges).

    Args:
        range_header: The Range header value (e.g., "bytes=0-1023")
        file_size: Total file size in bytes

    Returns:
        Tuple of (start, end) byte positions, or None if invalid
    """
    if not range_header or not range_header.startswith("bytes="):
        return None

    try:
        range_spec = range_header[6:]  # Remove "bytes="

        # Multiple ranges not supported
        if "," in range_spec:
            return None

        parts = range_spec.split("-", 1)
        if len(parts) != 2:
            return None

        start_str, end_str = parts

        if start_str == "":
            # Suffix range: bytes=-N (last N bytes)
            if not end_str:
                return None
            n = int(end_str)
            if n <= 0:
                return None
            start = max(0, file_size - n)
            end = file_size - 1
        else:
            start = int(start_str)
            if end_str:
                end = int(end_str)
            else:
                # Open-ended range: bytes=N-
                end = file_size - 1

        # Validate range
        if start < 0 or end < start or start >= file_size:
            return None

        # Clamp end to file size
        end = min(end, file_size - 1)

        return (start, end)

    except (ValueError, TypeError):
        return None


@router.get(
    "/{media_id:int}/file",
    status_code=status.HTTP_200_OK,
    summary="Get Original File for Media Item",
    responses={
        200: {
            "description": "File content streamed successfully",
            "content": {
                "application/pdf": {},
                "application/octet-stream": {},
            },
        },
        206: {
            "description": "Partial content (Range request)",
            "content": {
                "application/pdf": {},
                "application/octet-stream": {},
            },
        },
        304: {"description": "Not modified (ETag match)"},
        404: {"description": "Media item or file not found"},
        416: {"description": "Range not satisfiable"},
        500: {"description": "Storage error retrieving file"},
    },
)
async def get_media_file(
    media_id: int = Path(..., description="The ID of the media item"),
    file_type: str = Query(
        "original",
        description="Type of file to retrieve (e.g., 'original', 'thumbnail')",
    ),
    range_header: str | None = Header(None, alias="Range"),
    if_none_match: str | None = Header(None, alias="If-None-Match"),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """
    Retrieve the original file associated with a media item.

    This endpoint streams the original uploaded file (e.g., PDF, document)
    that was stored when the media item was ingested with `keep_original_file=True`.

    The file is streamed directly to the client for efficient handling of
    large files without loading them entirely into memory.
    """
    logger.debug(
        "Fetching file for media_id={}, file_type={}, user_id={}",
        media_id,
        file_type,
        getattr(current_user, "id", "?"),
    )

    # 1. Verify media item exists and is accessible
    media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media:
        logger.warning(
            "Media not found or not accessible for ID: {} (user: {})",
            media_id,
            getattr(current_user, "id", "?"),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or is inactive/trashed",
        )

    # 2. Get file record from database
    file_record = db.get_media_file(media_id, file_type)
    if not file_record:
        logger.debug(
            "No {} file record found for media_id={}",
            file_type,
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {file_type} file available for this media item",
        )

    storage_path = file_record.get("storage_path")
    if not storage_path:
        logger.error("File record has empty storage path")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File record is corrupted",
        )

    # 3. Verify file exists on storage
    storage = get_storage_backend()
    try:
        file_exists = await storage.exists(storage_path)
        if not file_exists:
            logger.error(
                "File missing from storage: {} (media_id={}, file_type={})",
                storage_path,
                media_id,
                file_type,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found on storage (may have been deleted)",
            )
    except StorageError as e:
        logger.error("Storage error checking file existence")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error accessing file storage",
        ) from e

    # 4. Stream the file
    try:
        # Determine content type and file metadata
        mime_type = file_record.get("mime_type") or "application/octet-stream"
        original_filename = file_record.get("original_filename") or f"file_{media_id}"
        file_size = file_record.get("file_size") or 0
        checksum = file_record.get("checksum")

        # Build ETag from checksum if available
        etag = f'"{checksum}"' if checksum else None

        # Handle ETag-based caching (If-None-Match)
        if etag and if_none_match:
            # Strip quotes and compare
            client_etag = if_none_match.strip().strip('"')
            if client_etag == checksum:
                logger.debug(f"ETag match for media_id={media_id}, returning 304")
                return Response(
                    status_code=status.HTTP_304_NOT_MODIFIED,
                    headers={"ETag": etag},
                )

        # Determine Content-Disposition: inline for PDFs (viewable), attachment for others
        disposition_type = "inline" if mime_type == "application/pdf" else "attachment"
        disposition = _build_content_disposition(disposition_type, original_filename)

        # Build base headers
        headers = {
            "Content-Disposition": disposition,
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
            "Accept-Ranges": "bytes",
        }
        if etag:
            headers["ETag"] = etag

        # Handle Range requests
        if range_header and file_size:
            range_result = _parse_range_header(range_header, file_size)

            if range_result is None:
                # Invalid range - return 416 Range Not Satisfiable
                logger.debug(f"Invalid range request for media_id={media_id}: {range_header}")
                return Response(
                    status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
                    headers={"Content-Range": f"bytes */{file_size}"},
                )

            start, end = range_result
            content_length = end - start + 1

            logger.info(
                "Streaming partial file: {} (bytes {}-{}/{}) for media_id={}",
                storage_path,
                start,
                end,
                file_size,
                media_id,
            )

            # Stream partial content
            async def partial_iterator():
                bytes_sent = 0
                target_bytes = content_length
                current_pos = 0

                async for chunk in storage.retrieve_stream(storage_path):
                    chunk_len = len(chunk)

                    # Skip chunks before start position
                    if current_pos + chunk_len <= start:
                        current_pos += chunk_len
                        continue

                    # Calculate slice within this chunk
                    chunk_start = max(0, start - current_pos)
                    chunk_end = min(chunk_len, end - current_pos + 1)

                    if chunk_start < chunk_end:
                        yield chunk[chunk_start:chunk_end]
                        bytes_sent += chunk_end - chunk_start

                    current_pos += chunk_len

                    # Stop if we've sent enough bytes
                    if bytes_sent >= target_bytes:
                        break

            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            headers["Content-Length"] = str(content_length)

            return StreamingResponse(
                partial_iterator(),
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                media_type=mime_type,
                headers=headers,
            )

        # Full file response
        logger.info(
            "Streaming file: {} ({} bytes, {}) for media_id={}",
            storage_path,
            file_record.get("file_size", "?"),
            mime_type,
            media_id,
        )

        async def file_iterator():
            async for chunk in storage.retrieve_stream(storage_path):
                yield chunk

        if file_size:
            headers["Content-Length"] = str(file_size)

        return StreamingResponse(
            file_iterator(),
            media_type=mime_type,
            headers=headers,
        )

    except FileNotFoundError:
        logger.error(
            "File not found during retrieval: {} (media_id={})",
            storage_path,
            media_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on storage",
        ) from None
    except StorageError as e:
        logger.error(
            "Storage error retrieving file {}: {}",
            storage_path,
            e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving file from storage",
        ) from e
    except Exception as e:
        logger.error(
            "Unexpected error streaming file {} for media_id={}: {}",
            storage_path,
            media_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error streaming file",
        ) from e


@router.head(
    "/{media_id:int}/file",
    status_code=status.HTTP_200_OK,
    summary="Check if Original File Exists",
)
async def head_media_file(
    media_id: int = Path(..., description="The ID of the media item"),
    file_type: str = Query(
        "original",
        description="Type of file to check (e.g., 'original', 'thumbnail')",
    ),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """
    Check if an original file exists for a media item without downloading it.

    Returns HTTP 200 with headers if file exists, 404 if not.
    Useful for checking file availability before attempting download.
    """
    # Verify media item exists
    media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    if not media:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found",
        )

    # Check file record
    file_record = db.get_media_file(media_id, file_type)
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )

    storage_path = file_record.get("storage_path")
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File record corrupted",
        )

    # Verify on storage
    storage = get_storage_backend()
    try:
        if not await storage.exists(storage_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found on storage",
            )

        file_size = await storage.get_size(storage_path)
    except StorageError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage error",
        ) from None

    mime_type = file_record.get("mime_type") or "application/octet-stream"
    original_filename = file_record.get("original_filename") or f"file_{media_id}"
    checksum = file_record.get("checksum")

    headers = {
        "Content-Type": mime_type,
        "Content-Length": str(file_size),
        "Content-Disposition": _build_content_disposition("inline", original_filename),
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, max-age=3600",
    }

    # Add ETag if checksum is available
    if checksum:
        headers["ETag"] = f'"{checksum}"'

    return Response(
        content=None,
        headers=headers,
    )


__all__ = ["router"]
