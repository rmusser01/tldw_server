from __future__ import annotations

import json
import stat
import zipfile
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.Visual_Identities import archive_import
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


def test_archive_import_temp_names_are_unique_for_same_member_basename(
    repo: VisualIdentityRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Temp Names", source_kind="zip")
    archive_path = _zip_with_entries(
        tmp_path / "same-basename.zip",
        {
            "poses/left/sprite.png": _png_bytes("green"),
            "poses/right/sprite.png": _png_bytes("blue"),
        },
    )
    temp_names: list[str] = []
    stored_count = 0

    def fake_copy_zip_entry_to_temp_file(
        archive: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        temp_path: Path,
    ) -> None:
        temp_names.append(temp_path.name)
        temp_path.write_bytes(_png_bytes())

    def fake_validate_and_store_visual_identity_asset(**kwargs: Any) -> SimpleNamespace:
        nonlocal stored_count
        stored_count += 1
        return SimpleNamespace(
            relpath=f"stored/{stored_count}.png",
            content_type="image/png",
            bytes=12,
            sha256=f"sha256-{stored_count}",
            width=16,
            height=16,
            is_animated=False,
            frame_count=1,
            duration_ms=None,
            preview_relpath=None,
        )

    monkeypatch.setattr(
        archive_import,
        "_copy_zip_entry_to_temp_file",
        fake_copy_zip_entry_to_temp_file,
    )
    monkeypatch.setattr(
        archive_import,
        "validate_and_store_visual_identity_asset",
        fake_validate_and_store_visual_identity_asset,
    )

    with zipfile.ZipFile(archive_path) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        slot_map = archive_import._store_candidates(
            archive,
            [
                archive_import._ImportCandidate(
                    info=infos["poses/left/sprite.png"],
                    normalized_path="poses/left/sprite.png",
                    expression_key="left",
                ),
                archive_import._ImportCandidate(
                    info=infos["poses/right/sprite.png"],
                    normalized_path="poses/right/sprite.png",
                    expression_key="right",
                ),
            ],
            repo=repo,
            owner_user_id=1,
            draft_id=draft["id"],
            storage_root=tmp_path / "store",
            summary=archive_import._empty_summary(source_filename="collisions.zip"),
        )

    assert set(slot_map) == {"left", "right"}
    assert len(temp_names) == 2
    assert len(set(temp_names)) == 2


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
        ({r"sprites\happy.png": b"data"}, "unsafe_archive_path"),
        ({"a/happy.png": b"first", "a/./happy.png": b"second"}, "duplicate_archive_path"),
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


@pytest.mark.parametrize("member_name", ["C:happy.png", "sprites/C:happy.png"])
def test_archive_import_rejects_windows_drive_letter_segments(
    repo: VisualIdentityRepository,
    tmp_path: Path,
    member_name: str,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Drive Letter", source_kind="zip")
    archive_path = _zip_with_entries(tmp_path / "drive-letter.zip", {member_name: _png_bytes()})

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    assert result["status"] == "failed"
    assert repo.list_draft_assets(draft["id"], owner_user_id=1) == []
    assert "unsafe_archive_path" in _error_codes(json.loads(result["validation_summary_json"]))


@pytest.mark.parametrize(
    ("kind", "expected_code"),
    [
        ("encrypted", "encrypted_entry"),
        ("symlink", "symlink_entry"),
    ],
)
def test_archive_import_rejects_unsafe_directory_entries(
    repo: VisualIdentityRepository,
    tmp_path: Path,
    kind: str,
    expected_code: str,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Unsafe Directory", source_kind="zip")
    archive_path = _write_zip_directory(tmp_path / f"{kind}-directory.zip", "sprites/", kind=kind)

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    assert result["status"] == "failed"
    assert expected_code in _error_codes(json.loads(result["validation_summary_json"]))


def test_archive_import_allows_plain_directory_entries_without_errors(
    repo: VisualIdentityRepository,
    tmp_path: Path,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Plain Directory", source_kind="zip")
    archive_path = tmp_path / "plain-directory.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        directory = zipfile.ZipInfo("sprites/")
        directory.create_system = 3
        directory.external_attr = (stat.S_IFDIR | 0o755) << 16
        archive.writestr(directory, b"")
        archive.writestr("sprites/happy.png", _png_bytes())

    result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=archive_path,
        storage_root=tmp_path / "store",
    )

    summary = json.loads(result["validation_summary_json"])
    assert result["status"] == "ready_for_review"
    assert summary["directories"] == ["sprites"]
    assert summary["errors"] == []


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


def test_archive_import_failed_reimport_clears_previous_visible_assets(
    repo: VisualIdentityRepository,
    tmp_path: Path,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Reimport", source_kind="zip")
    first_archive = _zip_with_entries(tmp_path / "valid-reimport.zip", {"happy.png": _png_bytes()})

    first_result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=first_archive,
        storage_root=tmp_path / "store",
    )

    assert first_result["status"] == "ready_for_review"
    first_assets = repo.list_draft_assets(draft["id"], owner_user_id=1)
    assert len(first_assets) == 1

    failed_archive = _zip_with_entries(tmp_path / "failed-reimport.zip", {"../sad.png": _png_bytes("blue")})
    failed_result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=failed_archive,
        storage_root=tmp_path / "store",
    )

    assert failed_result["status"] == "failed"
    assert json.loads(failed_result["slot_map_json"]) == {}
    assert repo.list_draft_assets(draft["id"], owner_user_id=1) == []
    assert repo.get_asset(first_assets[0]["id"], owner_user_id=1, include_deleted=True)["deleted"] == 1


@pytest.mark.parametrize(
    ("failure_kind", "expected_code"),
    [
        ("missing", "zip_archive_not_found"),
        ("oversized", "archive_size_exceeded"),
    ],
)
def test_archive_import_pre_open_failure_clears_previous_visible_assets(
    repo: VisualIdentityRepository,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_code: str,
) -> None:
    draft = repo.create_draft(owner_user_id=1, title="Pre-open Reimport", source_kind="zip")
    first_archive = _zip_with_entries(tmp_path / "valid-before-pre-open-failure.zip", {"happy.png": _png_bytes()})
    first_result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=first_archive,
        storage_root=tmp_path / "store",
    )
    assert first_result["status"] == "ready_for_review"
    first_assets = repo.list_draft_assets(draft["id"], owner_user_id=1)
    assert len(first_assets) == 1

    if failure_kind == "missing":
        failed_archive = tmp_path / "does-not-exist.zip"
    else:
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Visual_Identities.archive_import.MAX_EXPRESSION_ZIP_BYTES",
            1,
        )
        failed_archive = _zip_with_entries(tmp_path / "oversized.zip", {"sad.png": _png_bytes("blue")})

    failed_result = import_visual_identity_expression_zip(
        repo,
        owner_user_id=1,
        draft_id=draft["id"],
        archive_path=failed_archive,
        storage_root=tmp_path / "store",
    )

    assert failed_result["status"] == "failed"
    assert json.loads(failed_result["slot_map_json"]) == {}
    assert expected_code in _error_codes(json.loads(failed_result["validation_summary_json"]))
    assert repo.list_draft_assets(draft["id"], owner_user_id=1) == []
    assert repo.get_asset(first_assets[0]["id"], owner_user_id=1, include_deleted=True)["deleted"] == 1


def _png_bytes(color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGBA", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _zip_with_entries(path: Path, entries: Mapping[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            info = zipfile.ZipInfo("entry")
            info.filename = name
            archive.writestr(info, content)
    return path


def _write_zip_with_symlink(path: Path, name: str) -> Path:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, "target.png")
    return path


def _write_zip_directory(path: Path, name: str, *, kind: str) -> Path:
    info = zipfile.ZipInfo(name if name.endswith("/") else f"{name}/")
    if kind == "symlink":
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
    else:
        info.create_system = 3
        info.external_attr = (stat.S_IFDIR | 0o755) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, b"")
    if kind == "encrypted":
        _mark_first_zip_entry_encrypted(path)
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
