#!/usr/bin/env python3
"""Create and verify a deterministic, media-bearing full-account Chatbook fixture."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ChatbookVersion, ConflictResolution
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.config import clear_config_cache
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    create_media_database,
    get_media_by_id,
    get_media_by_title,
    get_media_transcripts,
    get_unvectorized_chunk_count,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_transcripts import upsert_transcript
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo

SOURCE_EMAIL = "chatbooks-backup-source@example.com"
DESTINATION_EMAIL = "chatbooks-backup-destination@example.com"
SOURCE_SETTINGS = {
    "preferences.ui.locale": "en-US",
    "preferences.ui.theme": "paper",
}
CHARACTER_NAME = "Chatbooks UAT Archivist"
MEDIA_TITLE = "Chatbooks full-account stored media"
MEDIA_BYTES = b"tldw full-account UAT stored media bytes\x00\x01\xff"
MEDIA_VECTOR = b"tldw-media-vector-v1\x00\x10\x20"
COLLECTION_NAME = "chatbooks_full_account_uat"
COLLECTION_IDS = ["uat-chunk-001", "uat-chunk-002"]
COLLECTION_EMBEDDINGS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


class FixtureVerificationError(AssertionError):
    """Raised when destination stores do not match the prepared account state."""


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest for deterministic fixture comparisons."""
    return hashlib.sha256(payload).hexdigest()


def _fixture_password_hash(username: str) -> str:
    """Return a deterministic, non-authenticating password hash for fixture users."""
    return sha256_bytes(f"chatbooks-uat-disabled-login:{username}".encode("utf-8"))


def _phase_root(root: Path, phase: str) -> Path:
    resolved_root = root.expanduser().resolve()
    phase_root = (resolved_root / phase).resolve()
    if not phase_root.is_relative_to(resolved_root):
        raise ValueError("Fixture phase escaped its root")
    return phase_root


async def _activate_phase(root: Path, phase: str) -> Path:
    phase_root = _phase_root(root, phase)
    phase_root.mkdir(parents=True, exist_ok=True)
    auth_db = phase_root / "users.db"
    os.environ.update(
        {
            "AUTH_MODE": "multi_user",
            "PROFILE": "multi-user-sqlite",
            "DATABASE_URL": f"sqlite:///{auth_db}",
            "USER_DB_BASE_DIR": str(phase_root / "user_databases"),
            "TESTING": "1",
            "TEST_MODE": "true",
            "JWT_SECRET_KEY": sha256_bytes(b"chatbooks-full-account-uat-jwt"),
        }
    )
    await reset_db_pool()
    reset_settings()
    clear_config_cache()
    await ensure_authnz_schema_ready_once()
    return phase_root


async def _create_user(*, username: str, email: str) -> int:
    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    created = await users_db.create_user(
        username=username,
        email=email,
        password_hash=_fixture_password_hash(username),
        role="user",
        is_active=True,
        is_verified=True,
    )
    if isinstance(created, dict):
        return int(created["id"])
    return int(created)


def _new_chatbook_service(user_id: int) -> ChatbookService:
    chacha_path = DatabasePaths.get_chacha_db_path(user_id)
    db = CharactersRAGDB(db_path=str(chacha_path), client_id=str(user_id))
    return ChatbookService(user_id=user_id, db=db, user_id_int=user_id)


async def _seed_source_account(source_root: Path) -> tuple[int, ChatbookService, dict[str, Any]]:
    source_user_id = await _create_user(username="chatbooks-backup-source", email=SOURCE_EMAIL)
    pool = await get_db_pool()
    overrides = UserProfileOverridesRepo(pool)
    await overrides.ensure_tables()
    for key, value in SOURCE_SETTINGS.items():
        await overrides.upsert_override(
            user_id=source_user_id,
            key=key,
            value=value,
            updated_by=source_user_id,
        )

    service = _new_chatbook_service(source_user_id)
    character_id = service.db.add_character_card(
        {
            "name": CHARACTER_NAME,
            "description": "Deterministic character included in the full-account UAT archive.",
            "first_message": "Archive state is ready for verification.",
            "tags": ["chatbooks", "uat"],
        }
    )
    if character_id is None:
        raise RuntimeError("Failed to seed the source character")

    user_root = DatabasePaths.resolve_user_base_directory(source_user_id)
    media_db = create_media_database(
        str(source_user_id),
        db_path=str(DatabasePaths.get_media_db_path(source_user_id)),
    )
    media_content = "First deterministic media chunk. Second deterministic media chunk."
    chunks = [
        {
            "text": "First deterministic media chunk.",
            "start_char": 0,
            "end_char": 32,
            "chunk_type": "semantic",
        },
        {
            "text": "Second deterministic media chunk.",
            "start_char": 33,
            "end_char": len(media_content),
            "chunk_type": "semantic",
        },
    ]
    media_id, media_uuid, _ = media_db.add_media_with_keywords(
        url="https://example.com/chatbooks/full-account-uat",
        title=MEDIA_TITLE,
        media_type="document",
        content=media_content,
        keywords=["chatbooks", "uat"],
        transcription_model="fixture-transcriber",
        author="tldw UAT",
        chunks=chunks,
        owner_user_id=source_user_id,
    )
    if media_id is None:
        raise RuntimeError("Failed to seed source media")
    upsert_transcript(
        media_db,
        int(media_id),
        transcription="Deterministic transcript restored from the full-account archive.",
        whisper_model="fixture-transcriber",
    )

    artifact_path = user_root / "stored_media" / "full-account-uat.bin"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(MEDIA_BYTES)
    media_db.insert_media_file(
        media_id=int(media_id),
        file_type="original",
        storage_path="stored_media/full-account-uat.bin",
        original_filename="full-account-uat.bin",
        file_size=len(MEDIA_BYTES),
        mime_type="application/octet-stream",
        checksum=f"sha256:{sha256_bytes(MEDIA_BYTES)}",
    )
    with media_db.transaction() as conn:
        media_row = media_db._fetchone_with_connection(
            conn,
            "SELECT uuid, version FROM Media WHERE id = ? AND deleted = 0",
            (int(media_id),),
        )
        if not media_row:
            raise RuntimeError("Seeded source media row was not found")
        current_version = int(media_row.get("version") or 1)
        next_version = current_version + 1
        now = media_db._get_current_utc_timestamp_str()
        cursor = media_db._execute_with_connection(
            conn,
            """
            UPDATE Media
               SET vector_embedding = ?, last_modified = ?, version = ?, client_id = ?
             WHERE id = ? AND version = ? AND deleted = 0
            """,
            (
                MEDIA_VECTOR,
                now,
                next_version,
                media_db.client_id,
                int(media_id),
                current_version,
            ),
        )
        if getattr(cursor, "rowcount", 0) != 1:
            raise RuntimeError("Failed to attach the source media vector")
        media_db._log_sync_event(
            conn,
            "Media",
            str(media_row["uuid"]),
            "update",
            next_version,
            {
                "id": int(media_id),
                "uuid": str(media_row["uuid"]),
                "version": next_version,
                "last_modified": now,
                "client_id": media_db.client_id,
                "vector_embedding_seeded": True,
            },
        )

    chroma = service._get_chroma_manager()
    if chroma is None:
        raise RuntimeError("ChromaDB is required for the full-account UAT fixture")
    chroma.store_in_chroma(
        collection_name=COLLECTION_NAME,
        texts=[chunk["text"] for chunk in chunks],
        embeddings=COLLECTION_EMBEDDINGS,
        ids=COLLECTION_IDS,
        metadatas=[
            {"media_id": str(media_id), "media_uuid": str(media_uuid), "chunk_index": index}
            for index in range(len(chunks))
        ],
    )
    return source_user_id, service, {
        "character_id": int(character_id),
        "media_id": int(media_id),
        "media_uuid": str(media_uuid),
    }


async def prepare(root: str | Path) -> dict[str, Any]:
    """Seed a source account and write its full-account archive plus expected state."""
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    source_root = _phase_root(root_path, "source")
    if source_root.exists():
        shutil.rmtree(source_root)
    source_root = await _activate_phase(root_path, "source")
    source_user_id, service, seeded = await _seed_source_account(source_root)

    success, message, generated_path = await service.create_chatbook(
        name="Full account",
        description="Deterministic media-bearing full-account UAT archive",
        content_selections=None,
        include_media=True,
        include_embeddings=True,
        include_generated_content=True,
        format_version=ChatbookVersion.V1_1,
        async_mode=False,
    )
    if not success or not generated_path:
        raise RuntimeError(f"Full-account fixture export failed: {message}")
    archive_path = source_root / "full-account.chatbook"
    shutil.copyfile(generated_path, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        media_payload_path = f"content/media/media_{seeded['media_id']}.json"
        media_payload = json.loads(archive.read(media_payload_path))
        bundled_artifacts = [
            item
            for item in media_payload.get("stored_artifacts", [])
            if item.get("bundled") and item.get("archive_path")
        ]
        if len(bundled_artifacts) != 1:
            raise RuntimeError("Expected exactly one bundled source media artifact")
        archive_media_path = str(bundled_artifacts[0]["archive_path"])
        archived_media_bytes = archive.read(archive_media_path)
    if archived_media_bytes != MEDIA_BYTES:
        raise RuntimeError("Exporter did not bundle the exact stored source media bytes")

    expected = {
        "schema_version": "1.0",
        "source_user_id": source_user_id,
        "destination_user_id": 2,
        "profile": {"identity.email": SOURCE_EMAIL},
        "settings": dict(sorted(SOURCE_SETTINGS.items())),
        "character": {"name": CHARACTER_NAME},
        "media": {
            "title": MEDIA_TITLE,
            "archive_path": archive_media_path,
            "artifact_sha256": sha256_bytes(MEDIA_BYTES),
            "vector_sha256": sha256_bytes(MEDIA_VECTOR),
            "transcript_count": 1,
            "chunk_count": 2,
        },
        "embeddings": {
            "collection_name": COLLECTION_NAME,
            "collection_ids": sorted(COLLECTION_IDS),
        },
    }
    expected_path = root_path / "expected.json"
    expected_path.write_text(json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "source_user_id": source_user_id,
        "archive_path": str(archive_path),
        "expected_path": str(expected_path),
    }


async def reset_destination(root: str | Path) -> dict[str, Any]:
    """Initialize a distinct, empty destination account without copying source stores."""
    root_path = Path(root).expanduser().resolve()
    expected = _load_expected(root_path)
    destination_root = _phase_root(root_path, "destination")
    if destination_root.exists():
        shutil.rmtree(destination_root)
    await _activate_phase(root_path, "destination")
    await _create_user(username="chatbooks-destination-placeholder", email="placeholder@example.com")
    destination_user_id = await _create_user(
        username="chatbooks-backup-destination",
        email=DESTINATION_EMAIL,
    )
    if destination_user_id != int(expected["destination_user_id"]):
        raise RuntimeError("Destination fixture user id is not deterministic")

    service = _new_chatbook_service(destination_user_id)
    create_media_database(
        str(destination_user_id),
        db_path=str(DatabasePaths.get_media_db_path(destination_user_id)),
    )
    counts = _content_counts(service, destination_user_id)
    if any(counts.values()):
        raise RuntimeError("Destination reset did not create empty account stores")
    return {"destination_user_id": destination_user_id, "counts": counts}


async def import_archive(root: str | Path, archive_path: str | Path | None = None) -> dict[str, Any]:
    """Import the prepared archive into the initialized clean destination account."""
    root_path = Path(root).expanduser().resolve()
    expected = _load_expected(root_path)
    await _activate_phase(root_path, "destination")
    destination_user_id = int(expected["destination_user_id"])
    service = _new_chatbook_service(destination_user_id)
    source_archive = Path(archive_path or (root_path / "source" / "full-account.chatbook")).resolve()
    if not source_archive.is_file():
        raise FileNotFoundError(f"Prepared archive not found: {source_archive}")
    imported_archive = Path(service.import_dir) / source_archive.name
    shutil.copyfile(source_archive, imported_archive)
    if sha256_bytes(imported_archive.read_bytes()) != sha256_bytes(source_archive.read_bytes()):
        raise RuntimeError("Destination upload copy does not match the prepared archive")

    success, message, result = await service.import_chatbook(
        file_path=str(imported_archive),
        conflict_resolution=ConflictResolution.SKIP,
        source_format="chatbook",
        import_media=True,
        import_embeddings=True,
        async_mode=False,
    )
    result_data = result if isinstance(result, dict) else {}
    return {
        "success": success,
        "message": message,
        "imported_items": result_data.get("imported_items", {}),
        "warnings": result_data.get("warnings", []),
    }


async def verify(root: str | Path) -> dict[str, Any]:
    """Read destination stores directly and fail unless every expected value is present."""
    root_path = Path(root).expanduser().resolve()
    expected = _load_expected(root_path)
    await _activate_phase(root_path, "destination")
    destination_user_id = int(expected["destination_user_id"])
    pool = await get_db_pool()
    user = await AuthnzUsersRepo(db_pool=pool).get_user_by_id(destination_user_id)
    if user is None:
        raise FixtureVerificationError("destination account record is missing")
    profile = {"identity.email": str(user.get("email") or "")}
    _require_equal(profile, expected["profile"], "destination profile")

    settings_rows = await UserProfileOverridesRepo(pool).list_overrides_for_user(destination_user_id)
    settings = {
        str(row["key"]): row.get("value")
        for row in settings_rows
        if str(row.get("key") or "") in expected["settings"]
    }
    _require_equal(settings, expected["settings"], "destination settings")

    service = _new_chatbook_service(destination_user_id)
    character = service.db.get_character_card_by_name(expected["character"]["name"])
    if not character:
        raise FixtureVerificationError("destination character is missing")

    media_db = create_media_database(
        str(destination_user_id),
        db_path=str(DatabasePaths.get_media_db_path(destination_user_id)),
    )
    media = get_media_by_title(media_db, expected["media"]["title"])
    if not media:
        raise FixtureVerificationError("destination media record is missing")
    media_id = int(media["id"])
    media_row = get_media_by_id(media_db, media_id)
    if not media_row:
        raise FixtureVerificationError("destination media record could not be reloaded")
    transcripts = get_media_transcripts(media_db, media_id)
    transcript_count = len(transcripts or [])
    chunk_count = int(get_unvectorized_chunk_count(media_db, media_id) or 0)
    _require_equal(transcript_count, expected["media"]["transcript_count"], "destination transcript count")
    _require_equal(chunk_count, expected["media"]["chunk_count"], "destination chunk count")

    stored_files = media_db.get_media_files(media_id)
    bundled = [row for row in stored_files if str(row.get("file_type")) == "original"]
    if len(bundled) != 1:
        raise FixtureVerificationError("destination stored media artifact is missing")
    user_root = DatabasePaths.resolve_user_base_directory(destination_user_id).resolve()
    artifact_path = (user_root / str(bundled[0]["storage_path"])).resolve()
    if not artifact_path.is_relative_to(user_root) or not artifact_path.is_file():
        raise FixtureVerificationError("destination stored media artifact path is invalid")
    artifact_sha256 = sha256_bytes(artifact_path.read_bytes())
    if artifact_sha256 != expected["media"]["artifact_sha256"]:
        raise FixtureVerificationError("destination stored media artifact SHA-256 does not match")

    vector_blob = media_row.get("vector_embedding")
    if isinstance(vector_blob, memoryview):
        vector_blob = vector_blob.tobytes()
    if isinstance(vector_blob, bytearray):
        vector_blob = bytes(vector_blob)
    if not isinstance(vector_blob, bytes):
        raise FixtureVerificationError("destination media vector blob is missing")
    vector_sha256 = sha256_bytes(vector_blob)
    _require_equal(vector_sha256, expected["media"]["vector_sha256"], "destination media vector SHA-256")

    chroma = service._get_chroma_manager()
    if chroma is None:
        raise FixtureVerificationError("destination embedding store is unavailable")
    try:
        collection = chroma.get_collection(expected["embeddings"]["collection_name"])
        collection_result = collection.get(ids=expected["embeddings"]["collection_ids"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise FixtureVerificationError("destination embedding collection is missing") from exc
    collection_ids = sorted(str(value) for value in collection_result.get("ids", []))
    _require_equal(
        collection_ids,
        expected["embeddings"]["collection_ids"],
        "destination embedding identifiers",
    )

    return {
        "source_user_id": int(expected["source_user_id"]),
        "destination_user_id": destination_user_id,
        "profile": profile,
        "settings": settings,
        "character": {"name": str(character.get("name") or "")},
        "media": {
            "title": str(media_row.get("title") or ""),
            "transcript_count": transcript_count,
            "chunk_count": chunk_count,
            "artifact_path": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "vector_sha256": vector_sha256,
        },
        "embeddings": {
            "collection_name": expected["embeddings"]["collection_name"],
            "collection_ids": collection_ids,
        },
        "expected": expected,
    }


def _load_expected(root: Path) -> dict[str, Any]:
    expected_path = root / "expected.json"
    if not expected_path.is_file():
        raise FileNotFoundError(f"Fixture expected state not found: {expected_path}")
    payload = json.loads(expected_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("Fixture expected state is invalid")
    return payload


def _content_counts(service: ChatbookService, user_id: int) -> dict[str, int]:
    media_db = create_media_database(
        str(user_id),
        db_path=str(DatabasePaths.get_media_db_path(user_id)),
    )
    chroma = service._get_chroma_manager()
    if chroma is None:
        raise RuntimeError("ChromaDB is required to verify the destination is empty")
    embedding_count = sum(int(collection.count()) for collection in chroma.list_collections())
    return {
        "characters": 1 if service.db.get_character_card_by_name(CHARACTER_NAME) else 0,
        "media_records": int(media_db.count_chatbook_scope_category("media_records") or 0),
        "media_stored_artifacts": int(
            media_db.count_chatbook_scope_category("media_stored_artifacts") or 0
        ),
        "embeddings": embedding_count,
    }


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise FixtureVerificationError(f"{label} does not match expected destination state")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "reset-destination", "import", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--root", required=True, type=Path)
    return parser


async def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "prepare":
        return await prepare(args.root)
    if args.command == "reset-destination":
        return await reset_destination(args.root)
    if args.command == "import":
        return await import_archive(args.root)
    if args.command == "verify":
        return await verify(args.root)
    raise ValueError(f"Unsupported command: {args.command}")


def main() -> int:
    args = _build_parser().parse_args()
    result = asyncio.run(_run_command(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
