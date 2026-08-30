import json
from pathlib import Path
from typing import Any, Union

from pydantic import TypeAdapter

from .models import ProfileManifest, ProfileProposal, ProfileRecord, ProfileScope
from .semantic import (
    PROFILE_DIALECT_ID,
    PROFILE_SCHEMA_ID,
    PROFILE_SEMANTIC_KEYWORD,
    PROFILE_SEMANTIC_RULES,
    PROFILE_SEMANTIC_VOCABULARY_ID,
)

CanonicalObject = Union[ProfileManifest, ProfileScope, ProfileRecord, ProfileProposal]
_PAYLOAD_DEFS = {
    "identity": "IdentityPayload",
    "preference": "PreferencePayload",
    "relationship": "RelationshipPayload",
    "correction": "CorrectionPayload",
    "constraint": "ConstraintPayload",
    "goal": "GoalPayload",
    "convention": "ConventionPayload",
    "working_context": "WorkingContextPayload",
    "legacy_unclassified": "LegacyUnclassifiedPayload",
}


def _when(properties: dict[str, Any], then: dict[str, Any]) -> dict[str, Any]:
    return {
        "if": {"properties": properties, "required": list(properties)},
        "then": then,
    }


def _non_null() -> dict[str, Any]:
    return {"not": {"type": "null"}}


def _active_record() -> dict[str, Any]:
    return {
        "allOf": [
            _non_null(),
            {
                "properties": {
                    "state": {"const": "active"},
                    "payload": _non_null(),
                },
                "required": ["state", "payload"],
            },
        ]
    }


def _record_conditionals() -> list[dict[str, Any]]:
    null = {"type": "null"}
    conditionals = [
        _when(
            {"state": {"const": "deleted"}},
            {
                "properties": {
                    "payload": null,
                    "semantic_key": null,
                    "expires_at": null,
                    "no_expiry": {"const": False},
                }
            },
        ),
        _when(
            {"state": {"not": {"const": "deleted"}}},
            {
                "properties": {"payload": _non_null()},
                "required": ["payload"],
            },
        ),
    ]
    conditionals.extend(
        _when(
            {"kind": {"const": kind}},
            {
                "properties": {
                    "payload": {
                        "anyOf": [
                            {"$ref": f"#/$defs/{payload_def}"},
                            null,
                        ]
                    }
                }
            },
        )
        for kind, payload_def in _PAYLOAD_DEFS.items()
    )
    conditionals.extend(
        [
            _when(
                {
                    "state": {"not": {"const": "deleted"}},
                    "kind": {"const": "working_context"},
                },
                {
                    "oneOf": [
                        {
                            "properties": {
                                "expires_at": _non_null(),
                                "no_expiry": {"const": False},
                            },
                            "required": ["expires_at"],
                        },
                        {
                            "properties": {
                                "expires_at": null,
                                "no_expiry": {"const": True},
                            },
                            "required": ["no_expiry"],
                        },
                    ]
                },
            ),
            _when(
                {"kind": {"not": {"const": "working_context"}}},
                {
                    "properties": {
                        "expires_at": null,
                        "no_expiry": {"const": False},
                    }
                },
            ),
        ]
    )
    return conditionals


def _proposal_conditionals() -> list[dict[str, Any]]:
    null = {"type": "null"}
    non_null = _non_null()
    content = {"proposed_record": _active_record()}
    target = {"target_record_id": non_null, "base_version_id": non_null}
    conditionals = [
        _when(
            {"state": {"const": "pending"}, "operation": {"const": "create"}},
            {
                "properties": {
                    "target_record_id": null,
                    "base_version_id": null,
                    **content,
                }
            },
        ),
        _when(
            {"state": {"const": "pending"}, "operation": {"const": "update"}},
            {"properties": {**target, **content}},
        ),
    ]
    conditionals.extend(
        _when(
            {"state": {"const": "pending"}, "operation": {"const": operation}},
            {"properties": {**target, "proposed_record": null}},
        )
        for operation in ("archive", "promote")
    )
    conditionals.extend(
        [
            _when(
                {"state": {"not": {"const": "pending"}}},
                {
                    "properties": {
                        "proposed_record": null,
                        "confidence": null,
                    }
                },
            ),
            _when(
                {
                    "state": {"not": {"const": "pending"}},
                    "operation": {"const": "create"},
                },
                {
                    "properties": {
                        "target_record_id": null,
                        "base_version_id": null,
                    }
                },
            ),
            _when(
                {
                    "state": {"not": {"const": "pending"}},
                    "operation": {"enum": ["update", "archive", "promote"]},
                },
                {"properties": target},
            ),
        ]
    )
    return conditionals


def export_json_schema(path: Path) -> None:
    schema = TypeAdapter(CanonicalObject).json_schema(ref_template="#/$defs/{model}")
    schema.update(
        {
            "$comment": (
                "Draft 2020-12 provides structural validation only; implementations must "
                "also process the required tldw semantic vocabulary. Canonical JSON uses "
                "RFC 8785: https://www.rfc-editor.org/rfc/rfc8785"
            ),
            "$id": PROFILE_SCHEMA_ID,
            "$schema": PROFILE_DIALECT_ID,
            "title": "tldw Personal Context Profile v1",
            "version": 1,
            PROFILE_SEMANTIC_KEYWORD: dict(PROFILE_SEMANTIC_RULES),
        }
    )
    schema["$defs"]["ProfileRecord"]["allOf"] = _record_conditionals()
    schema["$defs"]["ProfileProposal"]["allOf"] = _proposal_conditionals()
    path.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def export_profile_meta_schema(path: Path) -> None:
    draft = "https://json-schema.org/draft/2020-12"
    rules = {
        "type": "object",
        "additionalProperties": False,
        "description": (
            "Required semantic rules evaluated after structural Draft 2020-12 "
            "validation."
        ),
        "properties": {
            "canonicalization": {
                "const": "rfc8785-v1",
                "description": (
                    "Canonical JSON uses RFC 8785 JCS as published at "
                    "https://www.rfc-editor.org/rfc/rfc8785."
                ),
            },
            "canonicalDateTime": {
                "const": "utc-milliseconds-v1",
                "description": (
                    "Aware timestamps are limited to years 0001-9999, whole-minute "
                    "offsets, and millisecond precision, then rendered in UTC as "
                    "YYYY-MM-DDTHH:MM:SS.sssZ before JCS serialization."
                ),
            },
            "canonicalPayloadMaxUtf8Bytes": {
                "const": 16 * 1024,
                "description": (
                    "Maximum UTF-8 bytes of the canonical typed payload after "
                    "schema_version and kind defaults are applied."
                ),
            },
            "iJsonMaxSafeInteger": {
                "const": 2**53 - 1,
                "description": (
                    "Mutable integer counters stay within the exact interoperable "
                    "I-JSON range."
                ),
            },
            "pendingProposalExpiryDays": {
                "const": 90,
                "description": (
                    "Pending proposal expires_at must equal created_at plus exactly "
                    "90 days."
                ),
            },
            "proposalIdentityAndVersionLinks": {
                "const": "exact-v1",
                "description": (
                    "Nested profile and scope IDs equal the proposal; update record "
                    "and parent IDs equal target and base IDs; create parent is null."
                ),
            },
            "timestampInvariants": {
                "const": "exact-v1",
                "description": (
                    "Manifest, scope, and record updates do not precede creation; record "
                    "expiry follows update; proposal expiry follows creation; and all "
                    "nested proposed records satisfy the same record rules."
                ),
            },
        },
        "required": list(PROFILE_SEMANTIC_RULES),
    }
    meta_schema = {
        "$schema": f"{draft}/schema",
        "$id": PROFILE_DIALECT_ID,
        "$dynamicAnchor": "meta",
        "$vocabulary": {
            f"{draft}/vocab/core": True,
            f"{draft}/vocab/applicator": True,
            f"{draft}/vocab/unevaluated": True,
            f"{draft}/vocab/validation": True,
            f"{draft}/vocab/meta-data": True,
            f"{draft}/vocab/format-annotation": True,
            f"{draft}/vocab/content": True,
            PROFILE_SEMANTIC_VOCABULARY_ID: True,
        },
        "description": (
            "Draft 2020-12 provides structural validation. Conforming processors must "
            "also apply the required tldw semantic vocabulary."
        ),
        "title": "tldw Personal Context Profile v1 dialect",
        "allOf": [{"$ref": f"{draft}/schema"}],
        "properties": {PROFILE_SEMANTIC_KEYWORD: rules},
    }
    path.write_text(
        json.dumps(meta_schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
