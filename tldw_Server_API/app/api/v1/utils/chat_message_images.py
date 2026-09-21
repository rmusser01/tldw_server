"""Bounded, complete saved-image reads shared by chat API endpoints."""

import io
from typing import Any, Optional

from fastapi import HTTPException
from PIL import Image

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.chacha.message_store import MAX_CHAT_ATTACHMENT_READ_BYTES
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError


def _detect_image_mime_type(data: bytes) -> Optional[str]:
    """
    Detect image MIME type from magic bytes.

    Args:
        data: Raw image bytes

    Returns:
        MIME type string if recognized image format, None otherwise
    """
    if not data or len(data) < 12:
        return None

    # Check PNG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"

    # Check JPEG (multiple signatures)
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"

    # Check GIF
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"

    # Check WebP (RIFF....WEBP)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    # Check BMP
    if data[:2] == b"BM":
        return "image/bmp"

    # Check ICO
    if data[:4] == b"\x00\x00\x01\x00":
        return "image/x-icon"

    return None


def _complete_message_images(message: dict[str, Any]) -> list[str]:
    """Expand every stored attachment, or fail the entire opt-in read."""
    import base64

    images = message.get("images") or []
    if not images and (message.get("image_data") is not None or message.get("image_mime_type") is not None):
        images = [message]
    result = []
    for image in images:
        data = image.get("image_data")
        if isinstance(data, memoryview):
            data = data.tobytes()
        mime = image.get("image_mime_type")
        detected_mime = _detect_image_mime_type(data) if isinstance(data, bytes) and data else None
        if not detected_mime or not isinstance(mime, str) or detected_mime != mime:
            raise HTTPException(
                status_code=409,
                detail="A saved chat attachment is incomplete or invalid. Reload after repairing the source message.",
            )
        if len(data) > int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024)):
            raise HTTPException(status_code=413, detail="A saved chat attachment exceeds the image read limit.")
        try:
            with Image.open(io.BytesIO(data)) as decoded:
                decoded.verify()
            with Image.open(io.BytesIO(data)) as decoded:
                decoded.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=409, detail="A saved chat attachment is incomplete or invalid.") from exc
        result.append(f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}")
    return result


def read_messages_with_images(
    db: CharactersRAGDB,
    chat_id: str,
    *,
    limit: int,
    offset: int = 0,
    include_deleted: bool = False,
    for_completions: bool = False,
    image_byte_limit: int = MAX_CHAT_ATTACHMENT_READ_BYTES,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Read one complete attachment page after the caller verifies ownership."""
    try:
        messages = db.get_messages_for_conversation(
            chat_id,
            limit=limit,
            offset=offset,
            include_deleted=include_deleted,
            strict_images=True,
            image_byte_limit=image_byte_limit,
        )
    except InputError as exc:
        raise HTTPException(status_code=413, detail="Chat attachments exceed the image read limit.") from exc
    except CharactersRAGDBError as exc:
        raise HTTPException(
            status_code=503,
            detail="Saved chat attachments could not be read completely. Retry loading the conversation.",
        ) from exc
    attachment_urls = {}
    decoded_total = 0
    for message in messages:
        images = message.get("images") or (
            [message] if message.get("image_data") is not None or message.get("image_mime_type") is not None else []
        )
        for image in images:
            data = image.get("image_data")
            if not isinstance(data, (bytes, memoryview)) or not data:
                raise HTTPException(status_code=409, detail="A saved chat attachment is incomplete or invalid.")
            decoded_total += len(data)
            if decoded_total > image_byte_limit:
                raise HTTPException(status_code=413, detail="Chat attachments exceed the image read limit.")
    encoded_total = 0
    for message in messages:
        urls = _complete_message_images(message)
        encoded_total += sum(len(url) for url in urls)
        if encoded_total > (image_byte_limit * 4 // 3) + 1024 * len(messages):
            raise HTTPException(status_code=413, detail="Chat attachments exceed the image read limit.")
        attachment_urls[message["id"]] = urls
        if urls:
            try:
                metadata = db.get_message_metadata(message["id"], strict=True)
            except CharactersRAGDBError as exc:
                raise HTTPException(
                    status_code=503, detail="Saved image options could not be read. Retry loading the conversation."
                ) from exc
            extra = metadata.get("extra") if isinstance(metadata, dict) else None
            message["image_details"] = extra.get("image_details") if isinstance(extra, dict) else None
            if for_completions:
                from tldw_Server_API.app.core.Chat.chat_service import _saved_image_text

                message["content"] = _saved_image_text(message, extra)

    return messages, attachment_urls


def format_message_content(
    text: str, image_urls: list[str], image_details: Any = None,
) -> str | list[dict[str, Any]]:
    """Keep plain text unchanged and append each stored image in its saved order."""
    if not image_urls:
        return text
    parts = [{"type": "text", "text": text}] if text else []
    from tldw_Server_API.app.core.Chat.chat_service import _saved_image_details

    details = _saved_image_details({"image_details": image_details}, len(image_urls))
    for url, detail in zip(image_urls, details):
        image_url = {"url": url}
        if image_details is not None:
            image_url["detail"] = detail
        parts.append({"type": "image_url", "image_url": image_url})
    return parts
