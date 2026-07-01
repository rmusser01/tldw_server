from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.email_service import get_email_service
from tldw_Server_API.app.core.Chat.document_generator import DocumentGeneratorService, DocumentType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

MAX_EMAIL_RECIPIENTS = 50
MAX_EMAIL_ATTACHMENTS = 5
MAX_EMAIL_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_EMAIL_ATTACHMENT_FILENAME_LENGTH = 255


@dataclass
class NotificationResult:
    channel: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


def _mask_email_address(address: str) -> str:
    local, separator, domain = address.partition("@")
    if not separator or not domain:
        return "[invalid-recipient]"
    prefix = local[:1] if local else "*"
    return f"{prefix}***@{domain}"


def _is_basic_email_address(value: str) -> bool:
    local, separator, domain = value.partition("@")
    if not separator or not local or not domain:
        return False
    if "@" in domain or "." not in domain:
        return False
    if any(char.isspace() for char in value):
        return False
    labels = domain.split(".")
    return all(labels) and all(label.strip() == label for label in labels)


def _normalize_email_recipients(recipients: list[str] | None) -> tuple[list[str], int]:
    normalized: list[str] = []
    seen: set[str] = set()
    invalid_count = 0
    for raw_recipient in recipients or []:
        if not isinstance(raw_recipient, str):
            invalid_count += 1
            continue
        recipient = raw_recipient.strip()
        if not recipient:
            continue
        if (
            len(recipient) > 254
            or "\r" in recipient
            or "\n" in recipient
            or not _is_basic_email_address(recipient)
        ):
            invalid_count += 1
            continue
        dedupe_key = recipient.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized.append(recipient)
    return normalized, invalid_count


def _redacted_exception_for_log(exc: BaseException) -> RuntimeError:
    return RuntimeError(f"{type(exc).__name__}: redacted").with_traceback(exc.__traceback__)


def _attachment_content_size_bytes(content: bytes | bytearray | memoryview | str) -> int:
    if isinstance(content, memoryview):
        return content.nbytes
    if isinstance(content, (bytes, bytearray)):
        return len(content)
    return len(content.encode("utf-8"))


def _validate_email_attachments(
    attachments: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not attachments:
        return None
    if len(attachments) > MAX_EMAIL_ATTACHMENTS:
        return {
            "reason": "too_many_attachments",
            "attachment_count": len(attachments),
            "max_attachments": MAX_EMAIL_ATTACHMENTS,
        }

    total_bytes = 0
    for attachment in attachments:
        if not isinstance(attachment, dict):
            return {
                "reason": "invalid_attachment",
                "attachment_count": len(attachments),
            }
        filename = str(attachment.get("filename") or "").strip()
        if (
            not filename
            or len(filename) > MAX_EMAIL_ATTACHMENT_FILENAME_LENGTH
            or "/" in filename
            or "\\" in filename
            or any(ord(char) < 32 or ord(char) == 127 for char in filename)
        ):
            return {
                "reason": "invalid_attachment_filename",
                "attachment_count": len(attachments),
            }
        if "content" not in attachment or not isinstance(
            attachment["content"],
            (bytes, bytearray, memoryview, str),
        ):
            return {
                "reason": "invalid_attachment_content",
                "attachment_count": len(attachments),
            }
        total_bytes += _attachment_content_size_bytes(attachment["content"])
        if total_bytes > MAX_EMAIL_ATTACHMENT_BYTES:
            return {
                "reason": "attachment_limit_exceeded",
                "attachment_count": len(attachments),
                "max_attachment_bytes": MAX_EMAIL_ATTACHMENT_BYTES,
            }
    return None


class NotificationsService:
    """
    Unified notifications helper to send watchlist outputs via email or persist them to Chatbook.
    """

    def __init__(self, *, user_id: int, user_email: str | None = None) -> None:
        self.user_id = int(user_id)
        self.user_email = user_email
        self._email_service: Any | None = None
        self._doc_service: DocumentGeneratorService | None = None

    def _ensure_email_service(self) -> Any:
        if self._email_service is None:
            self._email_service = get_email_service()
        return self._email_service

    def _ensure_doc_service(self) -> DocumentGeneratorService:
        if self._doc_service is None:
            db_path = DatabasePaths.get_chacha_db_path(self.user_id)
            db = CharactersRAGDB(db_path=str(db_path), client_id=str(self.user_id))
            self._doc_service = DocumentGeneratorService(db, user_id=str(self.user_id))
        return self._doc_service

    async def deliver_email(
        self,
        *,
        subject: str,
        html_body: str,
        text_body: str | None,
        recipients: list[str] | None,
        attachments: list[dict[str, Any]] | None = None,
        fallback_to_user_email: bool = True,
    ) -> NotificationResult:
        recips, invalid_count = _normalize_email_recipients(recipients)
        if not recips and fallback_to_user_email and self.user_email:
            recips, fallback_invalid_count = _normalize_email_recipients([self.user_email])
            invalid_count += fallback_invalid_count
        if invalid_count:
            return NotificationResult(
                channel="email",
                status="failed",
                details={
                    "reason": "invalid_recipients",
                    "invalid_recipient_count": invalid_count,
                },
            )
        if len(recips) > MAX_EMAIL_RECIPIENTS:
            return NotificationResult(
                channel="email",
                status="failed",
                details={
                    "reason": "too_many_recipients",
                    "recipient_count": len(recips),
                    "max_recipients": MAX_EMAIL_RECIPIENTS,
                },
            )
        if not recips:
            return NotificationResult(
                channel="email",
                status="skipped",
                details={"reason": "no_recipients"},
            )
        attachment_error = _validate_email_attachments(attachments)
        if attachment_error is not None:
            return NotificationResult(
                channel="email",
                status="failed",
                details=attachment_error,
            )

        deliveries: list[dict[str, Any]] = []
        email_service = self._ensure_email_service()
        for addr in recips:
            masked_addr = _mask_email_address(addr)
            try:
                ok = await email_service.send_email(
                    to_email=addr,
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body,
                    attachments=attachments,
                )
                deliveries.append({"recipient": masked_addr, "status": "sent" if ok else "failed"})
            except Exception as exc:
                logger.bind(
                    operation="notifications.deliver_email",
                    user_id=self.user_id,
                    recipient=masked_addr,
                    exception_type=type(exc).__name__,
                ).opt(exception=_redacted_exception_for_log(exc)).error(
                    "Email delivery failed"
                )
                deliveries.append(
                    {
                        "recipient": masked_addr,
                        "status": "error",
                        "error_type": type(exc).__name__,
                    }
                )

        if all(entry["status"] == "sent" for entry in deliveries):
            status = "sent"
        elif any(entry["status"] == "sent" for entry in deliveries):
            status = "partial"
        else:
            status = "failed"
        return NotificationResult(
            channel="email",
            status=status,
            details={"deliveries": deliveries, "recipient_count": len(recips)},
        )

    def deliver_chatbook(
        self,
        *,
        title: str,
        content: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        document_type: DocumentType = DocumentType.BRIEFING,
        provider: str = "watchlists",
        model: str = "watchlists",
        conversation_id: int | None = None,
    ) -> NotificationResult:
        try:
            svc = self._ensure_doc_service()
            extra_meta = dict(metadata or {})
            if description:
                extra_meta["description"] = description
            doc_id = svc.create_manual_document(
                title=title,
                content=content,
                document_type=document_type,
                metadata=extra_meta,
                provider=provider,
                model=model,
                conversation_id=conversation_id,
            )
            return NotificationResult(
                channel="chatbook",
                status="stored",
                details={"document_id": doc_id, "provider": provider, "model": model},
            )
        except Exception as exc:
            logger.error(f"Chatbook delivery failed: {exc}")
            return NotificationResult(
                channel="chatbook",
                status="failed",
                details={"error": str(exc)},
            )
