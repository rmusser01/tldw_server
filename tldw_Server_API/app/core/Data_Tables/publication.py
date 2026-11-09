"""Replay completed Data Tables results without replacing newer saved edits."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import sqlite3
from typing import Any
from uuid import UUID

from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError, DatabaseError, InputError
from tldw_Server_API.app.core.exceptions import DataTablesJobError

PROTOCOL_KEY = "data_table_publication_version"
RECEIPT_KEY = "_data_table_publication"
_PUBLIC_KEYS = ("table_id", "table_uuid", "row_count", "columns")


def check_result_size(result: dict[str, Any]) -> None:
    """Refuse a receipt which Jobs would truncate instead of publishing it."""
    encoded = json.dumps(result, allow_nan=False).encode("utf-8")
    if len(encoded) > int(os.getenv("JOBS_MAX_JSON_BYTES", "1048576") or "1048576"):
        raise DataTablesJobError("data_table_result_too_large", retryable=False)


def public_generation_result(job: dict[str, Any], publication_status: str | None = None) -> dict[str, Any] | None:
    """Project status without staging rows, source snapshots or completion authority."""
    result = job.get("result")
    if not isinstance(result, dict) and publication_status is None:
        return None
    public = {key: result[key] for key in _PUBLIC_KEYS if isinstance(result, dict) and key in result}
    if publication_status is not None:
        public["publication_status"] = publication_status
    return public


def read_generation_reservation(raw: str) -> dict[str, str]:
    """Validate the private durable lookup identity before accessing Jobs."""
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) not in (
        {"key", "owner", "queue"},
        {"key", "owner", "queue", "uuid"},
        {"key", "owner", "queue", "uuid", "attempt"},
    ):
        raise ValueError("invalid_generation_reservation")
    if any(not isinstance(item, str) or not item.strip() for item in value.values()):
        raise ValueError("invalid_generation_reservation")
    for key in ("key", "uuid"):
        if key in value and str(UUID(value[key])) != value[key]:
            raise ValueError("invalid_generation_reservation")
    if "attempt" in value and (
        len(value["attempt"]) != 64 or any(c not in "0123456789abcdef" for c in value["attempt"])
    ):
        raise ValueError("invalid_generation_attempt")
    return value


def reservation_for_job(job: dict[str, Any], payload: dict[str, Any], *, bind_attempt: bool = False) -> dict[str, str]:
    """Bind a modern idempotent admission, or one pre-reservation legacy Job."""
    if job.get("domain") != "data_tables" or job.get("job_type") != "data_table_generate":
        raise ValueError("invalid_generation_job")
    key = payload.get("generation_key")
    if key is None:
        if job.get("idempotency_key"):
            raise ValueError("unbound_generation_key")
        key = job.get("uuid")
    elif job.get("idempotency_key") != key and not (not job.get("idempotency_key") and key == job.get("uuid")):
        raise ValueError("unbound_generation_key")
    value = {
        "key": key,
        "owner": str(job.get("owner_user_id") or ""),
        "queue": job.get("queue"),
        "uuid": job.get("uuid"),
    }
    if bind_attempt:
        token = job.get("lease_id") if job.get("status") == "processing" else job.get("completion_token")
        if not token:
            raise ValueError("missing_generation_attempt")
        value["attempt"] = hashlib.sha256(str(token).encode()).hexdigest()
    return read_generation_reservation(json.dumps(value))


def recover_table_generation(jm: Any, db: Any, table: dict[str, Any]) -> dict[str, Any]:
    """Replay the current scoped admission after remount, without enqueuing work."""
    raw = table.get("generation_job_json")
    if raw is None:
        return table
    try:
        reservation = read_generation_reservation(raw)
        if reservation.get("uuid"):
            job = jm.get_job_or_archived_by_uuid(
                reservation["uuid"], domain="data_tables", owner_user_id=reservation["owner"]
            )
        else:
            job = jm.get_job_or_archived_by_idempotency_key(
                idempotency_key=reservation["key"],
                domain="data_tables",
                queue=reservation["queue"],
                job_type="data_table_generate",
                owner_user_id=reservation["owner"],
            )
        if not job:
            return table
        payload = job.get("payload") or {}
        bound = reservation_for_job(job, payload, bind_attempt="attempt" in reservation)
        if any(bound[key] != value for key, value in reservation.items()):
            return table
        if (
            payload.get("table_id") != table["id"]
            or payload.get("table_uuid") != table["uuid"]
            or str(payload.get("user_id") or job.get("owner_user_id") or "") != str(table["client_id"])
        ):
            return table
        state = publish_completed_generation(
            jm,
            job["id"],
            expected_uuid=job["uuid"],
            db=db if str(db.client_id) == str(table["client_id"]) else None,
        )
        if state in {"conflict", "unavailable"} and table.get("status") != "failed":
            db.update_data_table(
                table["id"],
                status="failed",
                last_error="Generation finished but could not replace the saved table.",
                owner_user_id=table["client_id"],
                expected_generation_job=raw,
                expected_version=int(table["version"]),
            )
        return db.get_data_table(table["id"], owner_user_id=table["client_id"]) or table
    except (ValueError, TypeError, KeyError, AttributeError, DatabaseError, sqlite3.Error, OSError, RuntimeError):
        return table


def _validated_receipt(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") or {}
    result = job.get("result") or {}
    receipt = result.get(RECEIPT_KEY)
    if (
        type(payload.get(PROTOCOL_KEY)) is not int
        or payload[PROTOCOL_KEY] != 1
        or not isinstance(receipt, dict)
        or type(receipt.get("version")) is not int
        or receipt["version"] != 1
    ):
        raise ValueError("missing_publication_receipt")
    for key, expected in (
        ("generation_key", (reservation_for_job(job, payload))["key"]),
        ("job_uuid", job.get("uuid")),
        ("job_owner", str(job.get("owner_user_id") or "")),
        ("table_id", payload.get("table_id")),
        ("table_uuid", payload.get("table_uuid")),
        ("owner", str(payload.get("user_id") or job.get("owner_user_id") or "")),
    ):
        if not expected or receipt.get(key) != expected:
            raise ValueError("publication_identity_mismatch")
    token = str(job.get("completion_token") or "")
    if not token or not hmac.compare_digest(
        str(receipt.get("lease_hash") or ""), hashlib.sha256(token.encode()).hexdigest()
    ):
        raise ValueError("publication_attempt_mismatch")
    if type(receipt.get("expected_version")) is not int or receipt["expected_version"] < 1:
        raise ValueError("invalid_publication_version")
    columns, rows = receipt.get("columns"), receipt.get("rows")
    if not isinstance(columns, list) or not columns or not isinstance(rows, list) or not rows:
        raise ValueError("missing_publication_content")
    if len(rows) > 2000 or len(columns) > 2000 or not isinstance(receipt.get("sources"), list):
        raise ValueError("invalid_publication_content")
    for records, key in ((columns, "column_id"), (rows, "row_id")):
        identifiers = [str(UUID(record[key])) for record in records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate_publication_identity")
    check_result_size(result)
    return receipt


def publish_completed_generation(jm: Any, job_id: int, *, expected_uuid: str | None = None, db: Any = None) -> str:
    """Apply the stored exact-completion receipt, or return its explicit recovery state."""
    try:
        job = (
            jm.get_job_or_archived_by_uuid(expected_uuid, domain="data_tables")
            if expected_uuid is not None
            else jm.get_job(int(job_id))
        )
        if not job or job.get("status") != "completed":
            return "not_completed"
        if job.get("domain") != "data_tables" or job.get("job_type") != "data_table_generate":
            return "unavailable"
        if expected_uuid is not None and job.get("uuid") != expected_uuid:
            return "unavailable"
        payload, result = job.get("payload") or {}, job.get("result") or {}
        # Old completed workers already persisted content; they have no new protocol marker.
        if PROTOCOL_KEY not in payload and RECEIPT_KEY not in result and not result.get("_truncated"):
            return "ready"
        receipt = _validated_receipt(job)
        if db is not None and str(db.client_id) != receipt["owner"]:
            return "unavailable"
        session = (
            contextlib.nullcontext(db)
            if db is not None
            else managed_media_database(
                receipt["owner"],
                db_path=get_user_media_db_path(receipt["owner"]),
                initialize=False,
                existing_only=True,
            )
        )
        with session as media:
            media.persist_data_table_generation(
                receipt["table_id"],
                columns=receipt["columns"],
                rows=receipt["rows"],
                sources=receipt["sources"],
                generation_model=receipt.get("model"),
                status="ready",
                row_count=len(receipt["rows"]),
                last_error=None,
                owner_user_id=receipt["owner"],
                expected_version=receipt["expected_version"],
                expected_uuid=receipt["table_uuid"],
                expected_generation_job=json.dumps(
                    reservation_for_job(job, payload, bind_attempt=True), sort_keys=True
                ),
            )
        return "ready"
    except ConflictError:
        return "conflict"
    except (InputError, ValueError, TypeError, KeyError, AttributeError, DataTablesJobError):
        return "unavailable"
    except (DatabaseError, sqlite3.Error, OSError, RuntimeError):
        return "pending"
