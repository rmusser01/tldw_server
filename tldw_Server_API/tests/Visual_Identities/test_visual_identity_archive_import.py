from __future__ import annotations

import json
import stat
import zipfile
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, Generator

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.Visual_Identities.archive_import import (
    import_visual_identity_expression_zip,
)


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="visual-identity-import-test-client")
    yield database
    database.close_connection()


@pytest.fixture
def repo(chacha_db: CharactersRAGDB) -> VisualIdentityRepository:
    return VisualIdentityRepository.initialized(chacha_db)


def test_archive_import_rejects_path_traversal_and_marks_draft_failed(
    repo: VisualIdentityRepository,
    tmp_path: Path,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Traversal", source_kind="zip")
    archive_path = _zip_with_entries(tmp_path / "traversal.zip", {"../happy.png": _png_bytes()})

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    assert result["status"] == "failed"
    assert repo.list_draft_assets(draft["id"], owner_user_id=1) == []
    summary = json.loads(repo.get_draft(draft["id"], owner_user_id=1)["validation_summary_json"])
    assert summary["accepted"] == []
    assert _error_codes(summary) == {"unsafe_archive_path"}


def test_archive_import_maps_default_aliases_and_custom_slots(
    repo: VisualIdentityRepository,
    tmp_path: Path,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Aliases", source_kind="zip")
    archive_path = _zip_with_entries(
        tmp_path / "aliases.zip",
        {
            "sprites/default.png": _png_bytes("green"),
            "poses/smirk.png": _png_bytes("blue"),
        },
    )

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    assert result["status"] == "ready_for_review"
    slot_map = json.loads(result["slot_map_json"])
    assert set(slot_map) == {"neutral", "custom:smirk"}
    assert slot_map["neutral"]["source_filename"] == "sprites/default.png"
    assert slot_map["custom:smirk"]["display_label"] == "Smirk"


def test_duplicate_expression_keys_import_first_by_normalized_path_and_report_duplicates(
    repo: VisualIdentityRepository,
    tmp_path: Path,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Duplicates", source_kind="zip")
    archive_path = _zip_with_entries(
        tmp_path / "duplicates.zip",
        {
            "z/happy.png": _png_bytes("red"),
            "a/joy.png": _png_bytes("green"),
            "m/cheerful.png": _png_bytes("blue"),
        },
    )

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    slot_map = json.loads(result["slot_map_json"])
    summary = json.loads(result["validation_summary_json"])
    assert slot_map["happy"]["source_filename"] == "a/joy.png"
    assert [duplicate["source_filename"] for duplicate in summary["duplicates"]] == [
        "m/cheerful.png",
        "z/happy.png",
    ]
    assert repo.list_draft_assets(draft["id"], owner_user_id=1)[0]["source_filename"] == "a/joy.png"


@pytest.mark.parametrize(
    ("entries", "expected_code"),
    [
        ({"nested.zip": b"PK\x03\x04nested"}, "nested_archive"),
        ({"link.png": b""}, "symlink_entry"),
        ({"a/happy.png": b"first", r"a\happy.png": b"second"}, "duplicate_archive_path"),
    ],
)
def test_archive_import_rejects_unsafe_zip_entries(
    repo: VisualIdentityRepository,
    tmp_path: Path,
    entries: Mapping[str, bytes],
    expected_code: str,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Unsafe", source_kind="zip")
    archive_path = tmp_path / "unsafe.zip"
    if expected_code == "symlink_entry":
        _write_zip_with_symlink(archive_path, "link.png")
    else:
        _zip_with_entries(archive_path, entries)

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    assert result["status"] == "failed"
    assert repo.list_draft_assets(draft["id"], owner_user_id=1) == []
    assert expected_code in _error_codes(json.loads(result["validation_summary_json"]))


def test_archive_import_rejects_encrypted_entry(
    repo: VisualIdentityRepository,
    tmp_path: Path,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Encrypted", source_kind="zip")
    archive_path = _zip_with_entries(tmp_path / "encrypted.zip", {"happy.png": _png_bytes()})
    _mark_first_zip_entry_encrypted(archive_path)

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    assert result["status"] == "failed"
    assert "encrypted_entry" in _error_codes(json.loads(result["validation_summary_json"]))


def test_archive_import_rejects_decompression_ratio(
    repo: VisualIdentityRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Visual_Identities.archive_import.MAX_EXPRESSION_ZIP_DECOMPRESSION_RATIO",
        2,
    )
    draft = repo.create_draft(owner_user_id=1, title="Ratio", source_kind="zip")
    archive_path = tmp_path / "ratio.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("happy.png", b"0" * 5000)

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    assert result["status"] == "failed"
    assert "decompression_ratio_exceeded" in _error_codes(json.loads(result["validation_summary_json"]))


def test_archive_import_rejects_total_uncompressed_and_entry_limits(
    repo: VisualIdentityRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Visual_Identities.archive_import.MAX_EXPRESSION_ZIP_TOTAL_UNCOMPRESSED_BYTES",
        3,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Visual_Identities.archive_import.MAX_EXPRESSION_ASSET_BYTES",
        4,
    )
    draft = repo.create_draft(owner_user_id=1, title="Limits", source_kind="zip")
    archive_path = _zip_with_entries(tmp_path / "limits.zip", {"happy.png": b"12345"})

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    assert result["status"] == "failed"
    assert {"entry_uncompressed_size_exceeded", "total_uncompressed_size_exceeded"}.issubset(
        _error_codes(json.loads(result["validation_summary_json"]))
    )


def test_archive_import_marks_ready_for_review_and_creates_asset_rows_for_valid_images(
    repo: VisualIdentityRepository,
    tmp_path: Path,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Valid", source_kind="zip")
    archive_path = _zip_with_entries(
        tmp_path / "valid.zip",
        {
            "neutral.png": _png_bytes("green"),
            "notes.txt": b"not an image",
        },
    )

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    assert result["status"] == "ready_for_review"
    assets = repo.list_draft_assets(draft["id"], owner_user_id=1)
    assert len(assets) == 1
    assert assets[0]["expression_key"] == "neutral"
    assert assets[0]["content_type"] == "image/png"
    summary = json.loads(result["validation_summary_json"])
    assert summary["accepted"][0]["source_filename"] == "neutral.png"
    assert "unsupported_archive_entry" in _error_codes(summary)


def _png_bytes(color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _zip_with_entries(path: Path, entries: Mapping[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


def _write_zip_with_symlink(path: Path, name: str) -> Path:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, "target.png")
    return path


def _mark_first_zip_entry_encrypted(path: Path) -> None:
    content = bytearray(path.read_bytes())
    local_header = content.find(b"PK\x03\x04")
    central_header = content.find(b"PK\x01\x02")
    assert local_header >= 0
    assert central_header >= 0
    content[local_header + 6 : local_header + 8] = (
        int.from_bytes(content[local_header + 6 : local_header + 8], "little") | 0x1
    ).to_bytes(2, "little")
    content[central_header + 8 : central_header + 10] = (
        int.from_bytes(content[central_header + 8 : central_header + 10], "little") | 0x1
    ).to_bytes(2, "little")
    path.write_bytes(content)


def _error_codes(summary: Mapping[str, Any]) -> set[str]:
    return {str(error["code"]) for error in summary.get("errors", [])}
