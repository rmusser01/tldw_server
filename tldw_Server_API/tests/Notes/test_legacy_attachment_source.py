from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

OWNER_ID = "owner-1"
_SECURE_SOURCE_FLAGS = ("O_NOFOLLOW", "O_NONBLOCK", "O_DIRECTORY")
# Positive descriptor tests require the same no-follow primitives as the reader,
# plus native support for its descriptor-relative opens and directory scans.
_requires_secure_descriptors = pytest.mark.skipif(
    not (
        all(getattr(os, name, 0) for name in _SECURE_SOURCE_FLAGS)
        and os.open in os.supports_dir_fd
        and os.scandir in os.supports_fd
    ),
    reason="requires no-follow directory descriptors, dir_fd opens, and fd scans",
)


def _source_type() -> type[Any]:
    try:
        module = importlib.import_module(
            "tldw_Server_API.app.core.Notes.legacy_attachment_source"
        )
    except ModuleNotFoundError:
        pytest.fail("legacy attachment source seam is missing")
    return module.LegacyAttachmentSource


def _source_error_code(exc: BaseException) -> str | None:
    return getattr(exc, "error_code", None)


def _note_dir(user_root: Path, note_id: str) -> Path:
    from tldw_Server_API.app.api.v1.endpoints.notes import (
        _safe_note_attachment_dirname,
    )

    return user_root / "notes_attachments" / _safe_note_attachment_dirname(note_id)


@pytest.fixture()
def note_db(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "notes.db", client_id=OWNER_ID)
    try:
        yield db
    finally:
        db.close_connection()


@pytest.mark.unit
def test_note_pages_are_id_ordered_bounded_and_include_soft_deleted(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    note_ids = [f"note-{index:03d}" for index in range(202, 0, -1)]
    for note_id in note_ids:
        note_db.note_store.add_note(note_id, "body", note_id=note_id)
    assert note_db.note_store.soft_delete_note("note-002", expected_version=1)
    source = _source_type()(note_db, owner_user_id=OWNER_ID, user_root=tmp_path / "owner")

    first = source.list_note_ids(limit=200)
    second = source.list_note_ids(after_note_id=first[-1], limit=200)

    assert first == tuple(sorted(note_ids)[:200])
    assert second == tuple(sorted(note_ids)[200:])
    assert "note-002" in first
    with pytest.raises(ValueError, match="1..200"):
        source.list_note_ids(limit=201)


@pytest.mark.unit
@_requires_secure_descriptors
def test_candidates_use_authoritative_note_directory_and_sorted_stable_keys(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    note_id = "../../note-owned"
    note_db.note_store.add_note("owned", "body", note_id=note_id)
    user_root = tmp_path / "owner"
    note_dir = _note_dir(user_root, note_id)
    note_dir.mkdir(parents=True)
    (note_dir / "zeta.txt").write_bytes(b"zeta")
    (note_dir / "alpha.txt").write_bytes(b"alpha")
    (note_dir / "alpha.txt.meta.json").write_text(
        json.dumps({"content_type": "text/plain"}),
        encoding="utf-8",
    )
    source = _source_type()(note_db, owner_user_id=OWNER_ID, user_root=user_root)

    before = {item.name: item.read_bytes() for item in note_dir.iterdir()}
    candidates = source.list_candidates(note_id)
    (note_dir / "alpha.txt").touch()
    repeated = source.list_candidates(note_id)

    assert source.note_directory(note_id) == note_dir
    assert source.note_directory(note_id).is_relative_to(user_root / "notes_attachments")
    assert [item.file_name for item in candidates] == ["alpha.txt", "zeta.txt"]
    assert [item.source_key for item in repeated] == [item.source_key for item in candidates]
    assert candidates[0].sha256 == "sha256:" + hashlib.sha256(b"alpha").hexdigest()
    assert candidates[0].metadata == {"content_type": "text/plain"}
    assert {item.name: item.read_bytes() for item in note_dir.iterdir()} == before


@pytest.mark.unit
@_requires_secure_descriptors
def test_candidate_enumeration_is_capped_per_note_and_cursor_is_bounded(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    note_id = "note-many"
    note_db.note_store.add_note("many", "body", note_id=note_id)
    user_root = tmp_path / "owner"
    note_dir = _note_dir(user_root, note_id)
    note_dir.mkdir(parents=True)
    for index in range(1001):
        (note_dir / f"item-{index:04d}.txt").write_text("x", encoding="utf-8")
    source = _source_type()(note_db, owner_user_id=OWNER_ID, user_root=user_root)

    first = source.list_candidates(note_id)
    second = source.list_candidates(note_id, after_source_key=first[-1].source_key)

    assert len(first) == 1000
    assert len(second) == 1
    with pytest.raises(Exception) as raised:
        source.list_candidates(note_id, after_source_key="x" * 4097)
    assert _source_error_code(raised.value) == "notes_attachment_source_cursor_too_large"
    with pytest.raises(Exception) as malformed:
        source.list_candidates(note_id, after_source_key="../foreign-source")
    assert _source_error_code(malformed.value) == "notes_attachment_source_cursor_invalid"


@pytest.mark.unit
def test_legacy_route_helper_rejects_attachment_root_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint

    user_root = tmp_path / "owner"
    outside = tmp_path / "outside"
    user_root.mkdir()
    outside.mkdir()
    try:
        (user_root / "notes_attachments").symlink_to(
            outside,
            target_is_directory=True,
        )
    except (NotImplementedError, OSError):
        pytest.skip("filesystem symlinks are unavailable")
    monkeypatch.setattr(
        notes_endpoint.DatabasePaths,
        "get_user_base_directory",
        lambda _user_id: user_root,
    )

    with pytest.raises(HTTPException) as raised:
        notes_endpoint._get_note_attachments_base_dir(OWNER_ID)

    assert raised.value.status_code == 500


@pytest.mark.unit
@_requires_secure_descriptors
def test_sidecar_is_read_with_a_64_kib_limit(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    note_id = "note-sidecar"
    note_db.note_store.add_note("sidecar", "body", note_id=note_id)
    user_root = tmp_path / "owner"
    note_dir = _note_dir(user_root, note_id)
    note_dir.mkdir(parents=True)
    (note_dir / "payload.txt").write_text("body", encoding="utf-8")
    (note_dir / "payload.txt.meta.json").write_bytes(b"{" + b"x" * 65536)
    source = _source_type()(note_db, owner_user_id=OWNER_ID, user_root=user_root)

    with pytest.raises(Exception) as raised:
        source.list_candidates(note_id)

    assert _source_error_code(raised.value) == "notes_attachment_sidecar_too_large"


@pytest.mark.unit
@_requires_secure_descriptors
def test_symlinked_note_directory_and_candidate_fail_closed(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    note_id = "note-link"
    note_db.note_store.add_note("link", "body", note_id=note_id)
    user_root = tmp_path / "owner"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    note_dir = _note_dir(user_root, note_id)
    note_dir.parent.mkdir(parents=True)
    try:
        note_dir.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("filesystem symlinks are unavailable")
    source = _source_type()(note_db, owner_user_id=OWNER_ID, user_root=user_root)

    with pytest.raises(Exception) as directory_error:
        source.list_candidates(note_id)
    assert _source_error_code(directory_error.value) == "notes_attachment_source_unsafe"

    note_dir.unlink()
    note_dir.mkdir()
    (note_dir / "secret.txt").symlink_to(outside / "secret.txt")
    with pytest.raises(Exception) as candidate_error:
        source.list_candidates(note_id)
    assert _source_error_code(candidate_error.value) == "notes_attachment_source_unsafe"


@pytest.mark.unit
def test_source_rejects_note_ids_not_owned_by_the_database(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    source = _source_type()(note_db, owner_user_id=OWNER_ID, user_root=tmp_path / "owner")

    with pytest.raises(Exception) as raised:
        source.list_candidates("unknown-note")

    assert _source_error_code(raised.value) == "notes_attachment_source_note_not_found"


def _trap_source_io(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_flags: tuple[str, ...] = (),
) -> tuple[Any, list[str]]:
    """Replace only the reader's OS reference; keep shared stdlib calls native."""
    module = importlib.import_module(
        "tldw_Server_API.app.core.Notes.legacy_attachment_source"
    )
    values = dict(vars(os))
    for name in missing_flags:
        values.pop(name, None)
    calls: list[str] = []

    def trap(*_args: Any, **_kwargs: Any) -> None:
        calls.append("descriptor I/O")
        pytest.fail("unsupported source must not open, read, or scan")

    for name in ("open", "read", "scandir"):
        values[name] = trap
    monkeypatch.setattr(module, "os", SimpleNamespace(**values))
    return module, calls


@pytest.mark.unit
@pytest.mark.parametrize("directory", [False, True], ids=["regular", "directory"])
def test_native_secure_flags_fail_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    module, calls = _trap_source_io(monkeypatch)
    required = ("O_NOFOLLOW", "O_NONBLOCK")
    if directory:
        required += ("O_DIRECTORY",)
    if all(hasattr(os, name) for name in required):
        flags = module.LegacyAttachmentSource._secure_open_flags(directory=directory)
        for name in required:
            assert flags & getattr(os, name) == getattr(os, name)
    else:
        with pytest.raises(module.LegacyAttachmentSourceError) as raised:
            if directory:
                module.LegacyAttachmentSource._open_directory("unopened-directory")
            else:
                module.LegacyAttachmentSource._open_regular("unopened-file", dir_fd=0)
        assert raised.value.error_code == "notes_attachment_source_platform_unsupported"
    assert calls == []
    assert os.open is not module.os.open


@pytest.mark.unit
@pytest.mark.parametrize("missing_flag", _SECURE_SOURCE_FLAGS)
@pytest.mark.parametrize("operation", ["enumerate", "verify"])
def test_missing_secure_flag_rejects_source_before_descriptor_io(
    tmp_path: Path,
    note_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    missing_flag: str,
    operation: str,
) -> None:
    note_id = "note-capabilities"
    note_db.note_store.add_note("owned", "body", note_id=note_id)
    user_root = tmp_path / "owner"
    note_dir = _note_dir(user_root, note_id)
    note_dir.mkdir(parents=True)
    payload = note_dir / "payload.txt"
    payload.write_bytes(b"unchanged")
    source = _source_type()(note_db, owner_user_id=OWNER_ID, user_root=user_root)
    module, calls = _trap_source_io(monkeypatch, missing_flags=(missing_flag,))
    snapshot = payload.stat()
    source_key = "notes_attachments/note-capabilities/payload.txt"
    candidate = module.LegacyAttachmentCandidate(
        note_id=note_id,
        file_name="payload.txt",
        source_key=source_key,
        relative_path=source_key,
        size_bytes=snapshot.st_size,
        modified_ns=snapshot.st_mtime_ns,
        sha256="sha256:" + hashlib.sha256(b"unchanged").hexdigest(),
        metadata={},
    )

    with pytest.raises(module.LegacyAttachmentSourceError) as raised:
        if operation == "enumerate":
            source.list_candidates(note_id)
        else:
            source.verify_candidate(candidate)

    assert raised.value.error_code == "notes_attachment_source_platform_unsupported"
    assert calls == []
    assert payload.read_bytes() == b"unchanged"
    assert os.open is not module.os.open


@pytest.mark.unit
@pytest.mark.parametrize("missing_flag", ["O_NOFOLLOW", "O_NONBLOCK"])
def test_missing_regular_file_flag_rejects_before_descriptor_io(
    monkeypatch: pytest.MonkeyPatch,
    missing_flag: str,
) -> None:
    module, calls = _trap_source_io(monkeypatch, missing_flags=(missing_flag,))

    with pytest.raises(module.LegacyAttachmentSourceError) as raised:
        module.LegacyAttachmentSource._open_regular("unopened-file", dir_fd=0)

    assert raised.value.error_code == "notes_attachment_source_platform_unsupported"
    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize("limit", [0, 1001])
def test_candidate_limit_guard_runs_without_secure_descriptors(
    tmp_path: Path,
    note_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
    module, calls = _trap_source_io(monkeypatch, missing_flags=_SECURE_SOURCE_FLAGS)
    source = module.LegacyAttachmentSource(
        note_db, owner_user_id=OWNER_ID, user_root=tmp_path / "owner"
    )

    with pytest.raises(ValueError, match="1..1000"):
        source.list_candidates("unknown-note", limit=limit)

    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cursor", "error_code"),
    [
        ("x" * 4097, "notes_attachment_source_cursor_too_large"),
        ("../foreign-source", "notes_attachment_source_cursor_invalid"),
    ],
)
def test_candidate_cursor_guard_runs_without_secure_descriptors(
    tmp_path: Path,
    note_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    cursor: str,
    error_code: str,
) -> None:
    module, calls = _trap_source_io(monkeypatch, missing_flags=_SECURE_SOURCE_FLAGS)
    source = module.LegacyAttachmentSource(
        note_db, owner_user_id=OWNER_ID, user_root=tmp_path / "owner"
    )

    with pytest.raises(module.LegacyAttachmentSourceError) as raised:
        source.list_candidates("unknown-note", after_source_key=cursor)

    assert raised.value.error_code == error_code
    assert calls == []
