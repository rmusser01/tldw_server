from __future__ import annotations

import json
from typing import Any

CLAIMS_JOBS_DOMAIN = "claims"
CLAIMS_JOBS_DEFAULT_QUEUE = "default"

CLAIMS_REBUILD_MEDIA_JOB_TYPE = "claims_rebuild_media"
CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE = "claims_deliver_review_notification"
CLAIMS_DELIVER_ALERT_JOB_TYPE = "claims_deliver_alert"

CLAIMS_JOB_PAYLOAD_VERSION = 1
CLAIMS_ALERT_JOB_CHANNELS = {"slack", "webhook"}
SENSITIVE_PAYLOAD_KEYS = {
    "db_path",
    "path",
    "webhook_url",
    "slack_webhook_url",
    "email_recipients",
    "recipient",
    "recipients",
    "claim_text",
    "notification_body",
    "alert_payload",
    "api_key",
    "secret",
    "token",
}
CLAIMS_REBUILD_MEDIA_PAYLOAD_KEYS = {"version", "owner_user_id", "media_id"}
CLAIMS_REVIEW_NOTIFICATION_PAYLOAD_KEYS = {
    "version",
    "owner_user_id",
    "notification_ids",
}
CLAIMS_ALERT_DELIVERY_PAYLOAD_KEYS = {
    "version",
    "owner_user_id",
    "event_id",
    "alert_id",
    "channel",
}


class ClaimsJobError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        failure_code: str = "claims_job_failed",
        backoff_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.failure_code = str(failure_code)
        if backoff_seconds is not None:
            self.backoff_seconds = int(backoff_seconds)


def _normalize_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ClaimsJobError(
                "claims job payload is not valid JSON",
                retryable=False,
                failure_code="claims_invalid_payload",
            ) from exc
        if isinstance(parsed, dict):
            return dict(parsed)
    raise ClaimsJobError(
        "claims job payload must be an object",
        retryable=False,
        failure_code="claims_invalid_payload",
    )


def _reject_sensitive_keys(payload: dict[str, Any]) -> None:
    present = sorted(SENSITIVE_PAYLOAD_KEYS.intersection(payload))
    if present:
        raise ClaimsJobError(
            f"claims job payload contains disallowed keys: {', '.join(present)}",
            retryable=False,
            failure_code="claims_invalid_payload",
        )


def _reject_unknown_keys(payload: dict[str, Any], allowed_keys: set[str]) -> None:
    present = sorted(set(payload).difference(allowed_keys), key=str)
    if present:
        unknown_keys = ", ".join(str(key) for key in present)
        raise ClaimsJobError(
            f"claims job payload contains unknown keys: {unknown_keys}",
            retryable=False,
            failure_code="claims_invalid_payload",
        )


def _owner_user_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ClaimsJobError(
            "claims job payload missing real owner_user_id",
            retryable=False,
            failure_code="claims_missing_owner",
        )
    if isinstance(value, int):
        owner = str(value)
    else:
        owner = value
    if (
        not owner
        or owner != owner.strip()
        or not owner.isascii()
        or not owner.isdigit()
        or owner == "0"
        or (len(owner) > 1 and owner.startswith("0"))
    ):
        raise ClaimsJobError(
            "claims job payload owner_user_id must be a canonical positive integer",
            retryable=False,
            failure_code="claims_missing_owner",
        )
    return owner


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ClaimsJobError(
            f"claims job payload has invalid {field}",
            retryable=False,
            failure_code="claims_invalid_payload",
        )
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized or not all("0" <= char <= "9" for char in normalized):
            raise ClaimsJobError(
                f"claims job payload has invalid {field}",
                retryable=False,
                failure_code="claims_invalid_payload",
            )
        parsed = int(normalized, 10)
    else:
        raise ClaimsJobError(
            f"claims job payload has invalid {field}",
            retryable=False,
            failure_code="claims_invalid_payload",
        )
    if parsed <= 0:
        raise ClaimsJobError(
            f"claims job payload has invalid {field}",
            retryable=False,
            failure_code="claims_invalid_payload",
        )
    return parsed


def _version(payload: dict[str, Any]) -> int:
    version = _positive_int(payload.get("version"), "version")
    if version != CLAIMS_JOB_PAYLOAD_VERSION:
        raise ClaimsJobError(
            "unsupported claims job payload version",
            retryable=False,
            failure_code="claims_unsupported_payload_version",
        )
    return version


def validate_rebuild_media_payload(value: Any) -> dict[str, Any]:
    payload = _normalize_dict(value)
    _reject_sensitive_keys(payload)
    version = _version(payload)
    owner_user_id = _owner_user_id(payload.get("owner_user_id"))
    _reject_unknown_keys(payload, CLAIMS_REBUILD_MEDIA_PAYLOAD_KEYS)
    return {
        "version": version,
        "owner_user_id": owner_user_id,
        "media_id": _positive_int(payload.get("media_id"), "media_id"),
    }


def validate_review_notification_payload(value: Any) -> dict[str, Any]:
    payload = _normalize_dict(value)
    _reject_sensitive_keys(payload)
    version = _version(payload)
    owner_user_id = _owner_user_id(payload.get("owner_user_id"))
    _reject_unknown_keys(payload, CLAIMS_REVIEW_NOTIFICATION_PAYLOAD_KEYS)
    raw_ids = payload.get("notification_ids")
    if not isinstance(raw_ids, list):
        raise ClaimsJobError(
            "claims review notification payload requires notification_ids",
            retryable=False,
            failure_code="claims_invalid_payload",
        )
    ids = sorted({_positive_int(item, "notification_id") for item in raw_ids})
    if not ids:
        raise ClaimsJobError(
            "claims review notification payload requires notification_ids",
            retryable=False,
            failure_code="claims_invalid_payload",
        )
    return {
        "version": version,
        "owner_user_id": owner_user_id,
        "notification_ids": ids,
    }


def validate_alert_delivery_payload(value: Any) -> dict[str, Any]:
    payload = _normalize_dict(value)
    _reject_sensitive_keys(payload)
    version = _version(payload)
    owner_user_id = _owner_user_id(payload.get("owner_user_id"))
    _reject_unknown_keys(payload, CLAIMS_ALERT_DELIVERY_PAYLOAD_KEYS)
    channel = str(payload.get("channel") or "").strip().lower()
    if channel not in CLAIMS_ALERT_JOB_CHANNELS:
        raise ClaimsJobError(
            "unsupported claims alert channel",
            retryable=False,
            failure_code="claims_unsupported_channel",
        )
    return {
        "version": version,
        "owner_user_id": owner_user_id,
        "event_id": _positive_int(payload.get("event_id"), "event_id"),
        "alert_id": _positive_int(payload.get("alert_id"), "alert_id"),
        "channel": channel,
    }


def skipped_result(reason: str, /, **extra: Any) -> dict[str, Any]:
    return {**extra, "outcome": "skipped", "reason": str(reason)}


def ok_result(**extra: Any) -> dict[str, Any]:
    return {**extra, "outcome": "ok"}
