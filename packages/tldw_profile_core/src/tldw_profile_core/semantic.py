from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from .canonical import I_JSON_MAX_INTEGER, canonical_json_bytes, parse_portable_datetime

PROFILE_DIALECT_ID = "urn:tldw:profile-core:json-schema:dialect:1"
PROFILE_SCHEMA_ID = "urn:tldw:profile-core:schema:personal-context:1"
PROFILE_SEMANTIC_VOCABULARY_ID = (
    "urn:tldw:profile-core:json-schema:vocabulary:semantic:1"
)
PROFILE_SEMANTIC_KEYWORD = "x-tldw-profile-semantics"
PROFILE_SEMANTIC_RULES = {
    "canonicalization": "rfc8785-v1",
    "canonicalDateTime": "utc-milliseconds-v1",
    "canonicalPayloadMaxUtf8Bytes": 16 * 1024,
    "iJsonMaxSafeInteger": I_JSON_MAX_INTEGER,
    "pendingProposalExpiryDays": 90,
    "proposalIdentityAndVersionLinks": "exact-v1",
    "timestampInvariants": "exact-v1",
}


class ProfileSemanticError(ValueError):
    """Raised when structurally valid profile data violates semantic rules."""


def _timestamp(value: str) -> datetime:
    try:
        return parse_portable_datetime(value)
    except ValueError as error:
        raise ProfileSemanticError(str(error)) from error


def _canonical_payload_size(record: Mapping[str, Any]) -> int:
    payload = dict(record["payload"])
    payload.setdefault("schema_version", 1)
    payload.setdefault("kind", record["kind"])
    try:
        return len(canonical_json_bytes(payload))
    except ValueError as error:
        raise ProfileSemanticError("payload is not valid I-JSON") from error


def _ordered_timestamps(
    value: Mapping[str, Any], later: str
) -> tuple[datetime, datetime]:
    created_at = _timestamp(value["created_at"])
    later_at = _timestamp(value[later])
    if later_at < created_at:
        raise ProfileSemanticError(f"{later} precedes created_at")
    return created_at, later_at


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _ordered_timestamps(manifest, "updated_at")
    for field in ("revision", "purge_generation"):
        if not 0 <= manifest[field] <= I_JSON_MAX_INTEGER:
            raise ProfileSemanticError(f"{field} is outside the I-JSON exact range")


def _validate_scope(scope: Mapping[str, Any]) -> None:
    _ordered_timestamps(scope, "updated_at")


def _validate_record(record: Mapping[str, Any]) -> None:
    _, updated_at = _ordered_timestamps(record, "updated_at")
    if record.get("expires_at") is not None:
        expires_at = _timestamp(record["expires_at"])
        if expires_at <= updated_at:
            raise ProfileSemanticError("record expiry is not later than updated_at")
    if record.get("payload") is None:
        return
    if (
        _canonical_payload_size(record)
        > PROFILE_SEMANTIC_RULES["canonicalPayloadMaxUtf8Bytes"]
    ):
        raise ProfileSemanticError("payload exceeds 16 KiB canonical UTF-8 limit")


def _validate_proposal(proposal: Mapping[str, Any]) -> None:
    pending = proposal["state"] == "pending"
    operation = proposal["operation"]
    proposed_record = proposal.get("proposed_record")
    created_at, expires_at = _ordered_timestamps(proposal, "expires_at")
    if expires_at <= created_at:
        raise ProfileSemanticError("proposal expiry is not later than created_at")
    if pending:
        if expires_at != created_at + timedelta(
            days=PROFILE_SEMANTIC_RULES["pendingProposalExpiryDays"]
        ):
            raise ProfileSemanticError(
                "pending proposal expiry must be exactly 90 days"
            )
    if proposed_record is None:
        return
    _validate_record(proposed_record)
    if proposed_record["profile_id"] != proposal["profile_id"]:
        raise ProfileSemanticError("proposal and proposed record profile IDs differ")
    if proposed_record["scope_id"] != proposal["scope_id"]:
        raise ProfileSemanticError("proposal and proposed record scope IDs differ")
    if operation == "create":
        if proposed_record.get("parent_version_id") is not None:
            raise ProfileSemanticError("create proposal has a parent version")
    elif operation == "update":
        if proposed_record["record_id"] != proposal["target_record_id"]:
            raise ProfileSemanticError("proposal and proposed record IDs differ")
        if proposed_record.get("parent_version_id") != proposal["base_version_id"]:
            raise ProfileSemanticError("proposal base and parent versions differ")


def validate_profile_semantics(value: Mapping[str, Any]) -> None:
    """Validate semantic vocabulary rules after Draft 2020-12 validation.

    This package reference validator intentionally does not perform structural
    JSON Schema validation. Call a Draft 2020-12 structural validator first,
    then pass the same decoded object here.
    """

    if "proposal_id" in value:
        _validate_proposal(value)
    elif "record_id" in value:
        _validate_record(value)
    elif "revision" in value:
        _validate_manifest(value)
    elif "scope_id" in value:
        _validate_scope(value)
    else:
        raise ProfileSemanticError("unsupported profile object")
