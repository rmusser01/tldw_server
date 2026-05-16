"""Tests for Persona Visual external provider envelope normalization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Callable

import pytest

from tldw_Server_API.app.core.Persona.visual_jobs import (
    PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
)
from tldw_Server_API.app.core.Persona.visual_portability.provider_envelope import (
    CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE,
    build_provider_archive_import_preview_handoff,
    normalize_provider_result_envelope,
)


def _valid_portable_archive_envelope() -> dict[str, object]:
    return {
        "contract_version": 1,
        "result_type": "portable_archive",
        "review_required": True,
        "activation_allowed": False,
        "import_preview_required": True,
        "provider": {
            "id": "local-sprite-pose-maker",
            "display_name": "Local Sprite Pose Maker",
            "version": "1.0.0",
        },
        "pack": {
            "title": "Research Buddy Expressions",
            "renderer_type": "sprite_frames",
            "manifest_version": 1,
            "states_offered": ["idle", "thinking", "speaking"],
            "static_fallback_available": True,
            "asset_count": 12,
            "total_bytes": 1843200,
        },
        "diagnostics": {
            "status": "ready_for_import_preview",
            "blockers": [],
            "warnings": [{"code": "state_fallback", "message": "listening falls back to idle"}],
        },
        "provenance": {
            "source": "mcp_provider",
            "provider_pack_id": "expr-pack-2026-05-13",
            "author": "local-user",
            "license_label": "user-provided",
        },
        "payload": {
            "archive": {
                "mcp_resource_uri": (
                    "mcp://local-sprite-pose-maker/resources/"
                    "expr-pack-2026-05-13.tldw-persona-vpack"
                ),
                "sha256": "2f3a6c2c4b0b0c7f9f7ad3e2c0f9f95543e3013d6a45b69822ad0f01f54415be",
                "media_type": CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE,
            }
        },
    }


def _blocker_codes(normalized: dict[str, object]) -> list[str]:
    return [
        str(item.get("code"))
        for item in normalized["blockers"]  # type: ignore[index]
        if isinstance(item, dict)
    ]


class _BoundedIterationMapping(Mapping[str, str]):
    """Mapping that raises if sanitizer materializes past the allowed item cap."""

    def __getitem__(self, key: str) -> str:
        return f"value-{key}"

    def __iter__(self) -> Iterator[str]:
        for index in range(1_000):
            if index >= 64:
                raise AssertionError("sanitizer iterated beyond bounded metadata limit")
            yield f"field_{index}"

    def __len__(self) -> int:
        return 1_000


def test_normalize_provider_result_envelope_accepts_valid_portable_archive() -> None:
    """Normalize valid portable archive provider output as review-only metadata."""
    normalized = normalize_provider_result_envelope(_valid_portable_archive_envelope())

    assert normalized["commit_eligible"] is True
    assert normalized["review_required"] is True
    assert normalized["activation_allowed"] is False
    assert normalized["import_preview_required"] is True
    assert normalized["result_type"] == "portable_archive"
    assert normalized["blockers"] == []
    assert normalized["payload"]["archive"]["media_type"] == CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE


def test_normalize_provider_result_envelope_preserves_blocked_diagnostics() -> None:
    """Keep provider blockers visible while preventing blocked output from committing."""
    raw = _valid_portable_archive_envelope()
    raw["pack"] = {
        "title": "Expressive Live2D Buddy",
        "renderer_type": "live2d",
        "manifest_version": 2,
        "renderer_contract_version": 1,
        "static_fallback_available": False,
    }
    raw["diagnostics"] = {
        "status": "blocked_before_import_preview",
        "blockers": [
            {
                "code": "fallback_missing",
                "message": "Manifest V2 provider results require a raster static fallback.",
            }
        ],
        "warnings": [
            {
                "code": "runtime_not_claimed",
                "message": "Provider does not provide runtime support.",
            }
        ],
    }
    raw["payload"] = None

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert _blocker_codes(normalized) == ["fallback_missing"]
    assert normalized["warnings"] == [
        {"code": "runtime_not_claimed", "message": "Provider does not provide runtime support."}
    ]


def test_normalize_provider_result_envelope_requires_payload_when_blocked_without_blockers() -> None:
    """Blocked diagnostic status alone must not make a missing archive payload commit-eligible."""
    raw = _valid_portable_archive_envelope()
    raw["diagnostics"] = {
        "status": "blocked_before_import_preview",
        "blockers": [],
        "warnings": [],
    }
    raw["payload"] = None

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "archive_payload_missing" in _blocker_codes(normalized)


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("activation_allowed", True, "activation_not_allowed"),
        ("result_type", "runtime_plugin", "unsupported_result_type"),
        ("review_required", False, "review_required_missing"),
    ],
)
def test_normalize_provider_result_envelope_rejects_core_contract_violations(
    field: str,
    value: object,
    expected_code: str,
) -> None:
    """Fail closed when provider output violates review-first invariants."""
    raw = _valid_portable_archive_envelope()
    raw[field] = value

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert expected_code in _blocker_codes(normalized)


def test_normalize_provider_result_envelope_requires_import_preview_for_archives() -> None:
    """Portable archives must enter import preview before any durable commit."""
    raw = _valid_portable_archive_envelope()
    raw["import_preview_required"] = False

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "import_preview_required_missing" in _blocker_codes(normalized)


@pytest.mark.parametrize(
    "media_type",
    [
        "application/vnd.tldw.persona-visual-pack",
        "text/plain",
        "",
    ],
)
def test_normalize_provider_result_envelope_rejects_unsupported_archive_media_type(
    media_type: str,
) -> None:
    """Reject stale or unsupported archive MIME strings in provider output."""
    raw = _valid_portable_archive_envelope()
    raw["payload"]["archive"]["media_type"] = media_type  # type: ignore[index]

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "unsupported_archive_media_type" in _blocker_codes(normalized)


def test_normalize_provider_result_envelope_accepts_existing_zip_archive_media_type() -> None:
    """Allow existing application/zip exports while preserving zip validation later."""
    raw = _valid_portable_archive_envelope()
    raw["payload"]["archive"]["media_type"] = "application/zip"  # type: ignore[index]

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is True
    assert normalized["payload"]["archive"]["media_type"] == "application/zip"


def test_normalize_provider_result_envelope_rejects_string_diagnostics() -> None:
    """Diagnostics must use stable machine-readable code/message objects."""
    raw = _valid_portable_archive_envelope()
    raw["diagnostics"] = {
        "status": "ready_for_import_preview",
        "blockers": ["fallback_missing"],
        "warnings": [],
    }

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "malformed_diagnostics" in _blocker_codes(normalized)


@pytest.mark.parametrize(
    ("section", "key", "unsafe_value"),
    [
        ("provenance", "api_key", "sk-secret-provider-key"),
        ("provenance", "local_path", "/Users/alice/private/pack"),
        ("provider", "display_name", "builder on localhost"),
        ("payload", "note", "../outside-pack"),
    ],
)
def test_normalize_provider_result_envelope_rejects_unsafe_metadata_strings(
    section: str,
    key: str,
    unsafe_value: str,
) -> None:
    """Reject secrets, host-local identifiers, and path-like provider metadata."""
    raw = _valid_portable_archive_envelope()
    section_value = raw[section]
    assert isinstance(section_value, dict)
    section_value[key] = unsafe_value

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "unsafe_provider_metadata" in _blocker_codes(normalized)
    assert unsafe_value not in str(normalized)


def test_normalize_provider_result_envelope_checks_unsafe_text_before_truncation() -> None:
    """Reject sensitive values even when they appear after the bounded output prefix."""
    raw = _valid_portable_archive_envelope()
    unsafe_value = ("provider-note-" * 50) + "sk-hiddenprovidersecret"
    raw["provider"]["display_name"] = unsafe_value  # type: ignore[index]

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "unsafe_provider_metadata" in _blocker_codes(normalized)
    assert unsafe_value not in str(normalized)


def test_normalize_provider_result_envelope_rejects_oversized_metadata_strings() -> None:
    """Treat oversized provider strings as unsafe before regex scanning or output truncation."""
    raw = _valid_portable_archive_envelope()
    unsafe_value = "provider-note-" * 800
    raw["provider"]["display_name"] = unsafe_value  # type: ignore[index]

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "unsafe_provider_metadata" in _blocker_codes(normalized)
    assert unsafe_value not in str(normalized)


def test_normalize_provider_result_envelope_bounds_mapping_iteration() -> None:
    """Avoid materializing all items from untrusted mapping-like metadata."""
    raw = _valid_portable_archive_envelope()
    raw["provider"] = _BoundedIterationMapping()

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "unsafe_provider_metadata" in _blocker_codes(normalized)
    assert len(normalized["provider"]) == 64


def test_normalize_provider_result_envelope_bounds_integer_text_coercion() -> None:
    """Reject oversized integer text before coercing contract_version."""
    raw = _valid_portable_archive_envelope()
    raw["contract_version"] = "1" * 20_000

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "unsupported_contract_version" in _blocker_codes(normalized)


@pytest.mark.parametrize("contract_version", [1.0, 1.9])
def test_normalize_provider_result_envelope_rejects_non_integer_contract_version_numbers(
    contract_version: float,
) -> None:
    """Only true integers or exact integer strings are accepted as contract versions."""
    raw = _valid_portable_archive_envelope()
    raw["contract_version"] = contract_version

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "unsupported_contract_version" in _blocker_codes(normalized)


@pytest.mark.parametrize(
    "resource_uri",
    [
        "https://example.invalid/pack.tldw-persona-vpack",
        "ftp://example.invalid/pack.tldw-persona-vpack",
        "s3://bucket/pack.tldw-persona-vpack",
        "ws://example.invalid/pack.tldw-persona-vpack",
        "data:application/zip;base64,AAAA",
        "file:///Users/alice/private/pack.tldw-persona-vpack",
    ],
)
def test_normalize_provider_result_envelope_rejects_remote_or_embedded_resource_uris(
    resource_uri: str,
) -> None:
    """Reject provider payload handles that bypass authenticated MCP resource retrieval."""
    raw = _valid_portable_archive_envelope()
    raw["payload"]["archive"]["mcp_resource_uri"] = resource_uri  # type: ignore[index]

    normalized = normalize_provider_result_envelope(raw)

    assert normalized["commit_eligible"] is False
    assert "unsafe_provider_metadata" in _blocker_codes(normalized)
    assert resource_uri not in str(normalized)


def test_build_provider_archive_import_preview_handoff_accepts_valid_portable_archive() -> None:
    """Build a review-only MCP resource handoff for a portable archive result."""
    handoff = build_provider_archive_import_preview_handoff(
        _valid_portable_archive_envelope(),
        user_id="user-1",
        request_id="request-1",
        preview_id="preview-1",
        target_persona_id="persona-1",
    )

    assert handoff["ready"] is True
    assert handoff["operation"] == "import_preview"
    assert handoff["job_type"] == PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE
    assert handoff["request"] == {
        "user_id": "user-1",
        "preview_id": "preview-1",
        "request_id": "request-1",
        "target_persona_id": "persona-1",
    }
    assert handoff["archive"] == {
        "source_type": "mcp_resource",
        "mcp_resource_uri": (
            "mcp://local-sprite-pose-maker/resources/"
            "expr-pack-2026-05-13.tldw-persona-vpack"
        ),
        "sha256": (
            "2f3a6c2c4b0b0c7f9f7ad3e2c0f9f95543e3013d6a45b69822ad0f01f54415be"
        ),
        "media_type": CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE,
    }
    assert handoff["diagnostics"]["blockers"] == []
    assert "archive_path" not in str(handoff)
    assert "activation_allowed" not in str(handoff["archive"])


def _set_provider_result_type_generated_candidate(raw: dict[str, object]) -> None:
    raw["result_type"] = "generated_candidate"


def _allow_provider_activation(raw: dict[str, object]) -> None:
    raw["activation_allowed"] = True


def _remove_provider_archive_resource_uri(raw: dict[str, object]) -> None:
    payload = raw["payload"]
    assert isinstance(payload, dict)
    archive = payload["archive"]
    assert isinstance(archive, dict)
    archive.pop("mcp_resource_uri")


def _set_invalid_provider_archive_sha(raw: dict[str, object]) -> None:
    payload = raw["payload"]
    assert isinstance(payload, dict)
    archive = payload["archive"]
    assert isinstance(archive, dict)
    archive["sha256"] = "not-a-sha"


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (
            _set_provider_result_type_generated_candidate,
            "unsupported_archive_handoff_result_type",
        ),
        (_allow_provider_activation, "activation_not_allowed"),
        (_remove_provider_archive_resource_uri, "archive_resource_uri_missing"),
        (_set_invalid_provider_archive_sha, "archive_sha256_invalid"),
    ],
)
def test_build_provider_archive_import_preview_handoff_fails_closed_for_invalid_inputs(
    mutator: Callable[[dict[str, object]], object],
    expected_code: str,
) -> None:
    """Return deterministic blockers without enqueueing or materializing provider output."""
    raw = _valid_portable_archive_envelope()
    mutator(raw)

    handoff = build_provider_archive_import_preview_handoff(
        raw,
        user_id="user-1",
        request_id="request-1",
        preview_id="preview-1",
    )

    assert handoff["ready"] is False
    assert expected_code in _blocker_codes(handoff)
    assert handoff["archive"] is None


def test_provider_archive_handoff_skips_archive_parse_for_non_archives() -> None:
    """Avoid archive-payload diagnostics for provider result types that are not archives."""
    raw = _valid_portable_archive_envelope()
    raw["result_type"] = "generated_candidate"
    raw["payload"] = None

    handoff = build_provider_archive_import_preview_handoff(
        raw,
        user_id="user-1",
        request_id="request-1",
        preview_id="preview-1",
    )

    codes = _blocker_codes(handoff)
    assert "unsupported_archive_handoff_result_type" in codes
    assert "archive_payload_missing" not in codes
    assert handoff["archive"] is None
