from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chatbooks import openwebui_hydration as hydration


pytestmark = pytest.mark.unit


def _patch_allowed_roots(monkeypatch: pytest.MonkeyPatch, allowed_root: Path) -> None:
    monkeypatch.setattr(
        hydration,
        "get_ingestion_source_allowed_roots",
        lambda *, reload=False: (allowed_root.resolve(strict=False),),
    )


def _write_openwebui_data_root(base: Path, *, include_webui_db: bool = True, include_uploads: bool = True) -> Path:
    data_root = base / "openwebui"
    data_root.mkdir(parents=True)
    if include_webui_db:
        (data_root / "webui.db").write_bytes(b"SQLite format 3\x00")
    if include_uploads:
        (data_root / "uploads").mkdir()
    return data_root


def test_data_root_outside_allowed_roots_is_rejected(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside" / "openwebui"
    allowed_root.mkdir()
    outside_root.mkdir(parents=True)
    _patch_allowed_roots(monkeypatch, allowed_root)

    with pytest.raises(ValueError, match="allowed roots"):
        hydration.validate_openwebui_data_root(outside_root)


def test_missing_webui_db_is_rejected(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    data_root = _write_openwebui_data_root(allowed_root, include_webui_db=False)
    _patch_allowed_roots(monkeypatch, allowed_root)

    with pytest.raises(ValueError, match="webui.db"):
        hydration.validate_openwebui_data_root(data_root)


def test_missing_uploads_is_required_only_when_file_bytes_are_needed(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    data_root = _write_openwebui_data_root(allowed_root, include_uploads=False)
    _patch_allowed_roots(monkeypatch, allowed_root)

    validated = hydration.validate_openwebui_data_root(data_root)

    assert validated.root_path == data_root.resolve(strict=False)
    assert validated.uploads_path == data_root.resolve(strict=False) / "uploads"
    with pytest.raises(ValueError, match="uploads"):
        hydration.validate_openwebui_data_root(data_root, require_uploads=True)


def test_file_path_traversal_is_rejected(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    data_root = _write_openwebui_data_root(allowed_root)
    _patch_allowed_roots(monkeypatch, allowed_root)
    validated = hydration.validate_openwebui_data_root(data_root, require_uploads=True)

    resolved = hydration.resolve_openwebui_file_path(
        {"id": "file-a", "filename": "notes.txt", "path": "../secret.txt"},
        validated,
    )

    assert resolved.status == "path_rejected"
    assert resolved.path is None
    assert resolved.warning_codes == ("path_rejected",)


def test_symlink_escape_is_rejected_by_canonical_target_checks(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    data_root = _write_openwebui_data_root(allowed_root)
    outside_root.mkdir()
    target = outside_root / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    (data_root / "uploads" / "escape.txt").symlink_to(target)
    _patch_allowed_roots(monkeypatch, allowed_root)
    validated = hydration.validate_openwebui_data_root(data_root, require_uploads=True)

    resolved = hydration.resolve_openwebui_file_path(
        {"id": "file-a", "filename": "escape.txt", "path": "uploads/escape.txt"},
        validated,
    )

    assert resolved.status == "path_rejected"
    assert resolved.path is None


def test_uploads_id_filename_fallback_resolves_when_safe(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    data_root = _write_openwebui_data_root(allowed_root)
    fallback_path = data_root / "uploads" / "file-a_notes.pdf"
    fallback_path.write_bytes(b"%PDF-1.4\n")
    _patch_allowed_roots(monkeypatch, allowed_root)
    validated = hydration.validate_openwebui_data_root(data_root, require_uploads=True)

    resolved = hydration.resolve_openwebui_file_path(
        {"id": "file-a", "filename": "notes.pdf", "path": None},
        validated,
    )

    assert resolved.status == "resolved"
    assert resolved.path == fallback_path.resolve(strict=False)
    assert resolved.source == "uploads_id_filename"
    assert resolved.file_kind == "document"
    assert resolved.mime_type == "application/pdf"


def test_resolved_image_file_path_classifies_basic_file_kind(tmp_path, monkeypatch):
    allowed_root = tmp_path / "allowed"
    data_root = _write_openwebui_data_root(allowed_root)
    image_path = data_root / "uploads" / "file-image_image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    _patch_allowed_roots(monkeypatch, allowed_root)
    validated = hydration.validate_openwebui_data_root(data_root, require_uploads=True)

    resolved = hydration.resolve_openwebui_file_path(
        {"id": "file-image", "filename": "image.png", "path": None},
        validated,
    )

    assert resolved.status == "resolved"
    assert resolved.file_kind == "image"
    assert resolved.mime_type == "image/png"
