"""Tests for Persona Visual provider archive retrieval materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.Persona.visual_jobs import (
    PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
)
from tldw_Server_API.app.core.Persona.visual_portability.fingerprints import sha256_bytes
from tldw_Server_API.app.core.Persona.visual_portability.provider_archive_retrieval import (
    materialize_provider_archive_import_preview_handoff,
)
from tldw_Server_API.app.core.Persona.visual_portability.provider_envelope import (
    CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE,
)


def _ready_handoff(
    *,
    archive_sha256: str,
    resource_uri: str = "mcp://provider/resources/pack",
    media_type: str = CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE,
) -> dict[str, object]:
    return {
        "ready": True,
        "operation": "import_preview",
        "job_type": PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
        "request": {
            "user_id": "user-1",
            "preview_id": "preview-1",
            "request_id": "request-1",
            "target_persona_id": "persona-1",
        },
        "archive": {
            "source_type": "mcp_resource",
            "mcp_resource_uri": resource_uri,
            "sha256": archive_sha256,
            "media_type": media_type,
        },
        "diagnostics": {"blockers": [], "warnings": []},
        "blockers": [],
        "warnings": [],
    }


def _codes(result: dict[str, object]) -> list[str]:
    diagnostics = result.get("diagnostics")
    assert isinstance(diagnostics, dict)
    blockers = diagnostics.get("blockers")
    assert isinstance(blockers, list)
    return [str(item.get("code")) for item in blockers if isinstance(item, dict)]


def test_materialize_provider_archive_import_preview_handoff_writes_job_payload(
    tmp_path: Path,
) -> None:
    """Materialize a ready MCP resource handoff into the existing import-preview payload shape."""
    archive_bytes = b"portable persona visual archive"
    handoff = _ready_handoff(archive_sha256=sha256_bytes(archive_bytes))

    result = materialize_provider_archive_import_preview_handoff(
        handoff,
        resource_reader=lambda uri: [archive_bytes[:9], archive_bytes[9:]],
        staging_root=tmp_path,
    )

    assert result["ready"] is True
    assert result["operation"] == "import_preview"
    assert result["job_type"] == PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE
    payload = result["job_payload"]
    assert payload == {
        "user_id": "user-1",
        "preview_id": "preview-1",
        "archive_path": payload["archive_path"],
        "request_id": "request-1",
        "target_persona_id": "persona-1",
    }
    archive_path = Path(str(payload["archive_path"]))
    assert archive_path.parent == tmp_path
    assert archive_path.suffix == ".tldw-persona-vpack"
    assert archive_path.read_bytes() == archive_bytes
    assert result["archive"] == {
        "source_type": "mcp_resource",
        "media_type": CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE,
        "sha256": sha256_bytes(archive_bytes),
        "size_bytes": len(archive_bytes),
    }
    assert "mcp://provider/resources/pack" not in str(result["diagnostics"])
    assert str(archive_path) not in str(result["diagnostics"])


def test_materialize_provider_archive_import_preview_handoff_accepts_case_insensitive_media_type(
    tmp_path: Path,
) -> None:
    """Treat compatible archive media types as case-insensitive metadata."""
    archive_bytes = b"portable persona visual archive"

    result = materialize_provider_archive_import_preview_handoff(
        _ready_handoff(
            archive_sha256=sha256_bytes(archive_bytes),
            media_type=CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE.upper(),
        ),
        resource_reader=lambda uri: archive_bytes,
        staging_root=tmp_path,
    )

    assert result["ready"] is True
    assert result["archive"]["media_type"] == CANONICAL_PERSONA_VISUAL_ARCHIVE_MEDIA_TYPE


def test_materialize_provider_archive_import_preview_handoff_accepts_bytearray_chunks(
    tmp_path: Path,
) -> None:
    """Allow bytearray chunks from resource readers because they are bytes-like."""
    archive_bytes = b"portable persona visual archive"

    result = materialize_provider_archive_import_preview_handoff(
        _ready_handoff(archive_sha256=sha256_bytes(archive_bytes)),
        resource_reader=lambda uri: [bytearray(archive_bytes[:8]), bytearray(archive_bytes[8:])],
        staging_root=tmp_path,
    )

    assert result["ready"] is True
    archive_path = Path(str(result["job_payload"]["archive_path"]))
    assert archive_path.read_bytes() == archive_bytes


def test_materialize_provider_archive_import_preview_handoff_fails_closed_for_blocked_handoff(
    tmp_path: Path,
) -> None:
    """Do not read or write archives when the provider handoff is not ready."""
    handoff = _ready_handoff(archive_sha256=sha256_bytes(b"archive"))
    handoff["ready"] = False
    handoff["blockers"] = [{"code": "archive_sha256_invalid", "message": "bad sha"}]

    def _reader(_uri: str) -> bytes:
        raise AssertionError("blocked handoff should not retrieve resources")

    result = materialize_provider_archive_import_preview_handoff(
        handoff,
        resource_reader=_reader,
        staging_root=tmp_path,
    )

    assert result["ready"] is False
    assert "handoff_not_ready" in _codes(result)
    assert list(tmp_path.iterdir()) == []


def test_materialize_provider_archive_import_preview_handoff_redacts_upstream_diagnostics(
    tmp_path: Path,
) -> None:
    """Keep upstream diagnostic codes without reflecting caller-provided messages."""
    handoff = _ready_handoff(archive_sha256=sha256_bytes(b"archive"))
    handoff["ready"] = False
    handoff["blockers"] = [
        {
            "code": "provider_failed",
            "message": "file:///Users/alice/secret-pack token=abc123",
        }
    ]
    handoff["warnings"] = [
        {
            "code": "provider_warning",
            "message": "mcp://provider/resources/private-pack",
        }
    ]

    result = materialize_provider_archive_import_preview_handoff(
        handoff,
        resource_reader=lambda uri: b"archive",
        staging_root=tmp_path,
    )

    assert result["ready"] is False
    assert "provider_failed" in _codes(result)
    serialized = str(result)
    assert "file:///Users/alice" not in serialized
    assert "token=abc123" not in serialized
    assert "mcp://provider/resources/private-pack" not in serialized
    assert "Provider archive handoff reported a diagnostic." in serialized


def test_materialize_provider_archive_import_preview_handoff_rejects_checksum_mismatch(
    tmp_path: Path,
) -> None:
    """Delete materialized bytes when the fetched resource checksum does not match."""
    result = materialize_provider_archive_import_preview_handoff(
        _ready_handoff(archive_sha256=sha256_bytes(b"expected")),
        resource_reader=lambda uri: b"actual",
        staging_root=tmp_path,
    )

    assert result["ready"] is False
    assert "archive_sha256_mismatch" in _codes(result)
    assert list(tmp_path.iterdir()) == []
    assert "actual" not in str(result)


def test_materialize_provider_archive_import_preview_handoff_rejects_oversized_resource(
    tmp_path: Path,
) -> None:
    """Stop writing and clean up when the fetched resource exceeds the configured byte cap."""
    archive_bytes = b"0123456789"

    result = materialize_provider_archive_import_preview_handoff(
        _ready_handoff(archive_sha256=sha256_bytes(archive_bytes)),
        resource_reader=lambda uri: [b"01234", b"56789"],
        staging_root=tmp_path,
        max_archive_size_bytes=8,
    )

    assert result["ready"] is False
    assert "archive_too_large" in _codes(result)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad_value",
    [
        "not bytes",
        {"bytes": b"archive"},
        [b"ok", "not bytes"],
    ],
)
def test_materialize_provider_archive_import_preview_handoff_rejects_non_byte_resources(
    tmp_path: Path,
    bad_value: object,
) -> None:
    """Require readers to return bytes or byte chunks so text/object data cannot leak."""
    result = materialize_provider_archive_import_preview_handoff(
        _ready_handoff(archive_sha256=sha256_bytes(b"archive")),
        resource_reader=lambda uri: bad_value,
        staging_root=tmp_path,
    )

    assert result["ready"] is False
    assert "archive_resource_invalid" in _codes(result)
    assert list(tmp_path.iterdir()) == []


def test_materialize_provider_archive_import_preview_handoff_rejects_unsafe_resource_uri(
    tmp_path: Path,
) -> None:
    """Only MCP resource handles are accepted for provider archive retrieval."""
    result = materialize_provider_archive_import_preview_handoff(
        _ready_handoff(
            archive_sha256=sha256_bytes(b"archive"),
            resource_uri="file:///Users/alice/archive.tldw-persona-vpack",
        ),
        resource_reader=lambda uri: b"archive",
        staging_root=tmp_path,
    )

    assert result["ready"] is False
    assert "archive_resource_uri_missing" in _codes(result)
    assert "file:///Users/alice" not in str(result)
    assert list(tmp_path.iterdir()) == []
