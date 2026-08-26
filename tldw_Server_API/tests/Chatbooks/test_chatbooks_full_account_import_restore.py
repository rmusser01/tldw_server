import asyncio
import base64
import hashlib
import json
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chat.document_generator import DocumentGeneratorService
from tldw_Server_API.app.core.Chatbooks import chatbook_service as chatbook_service_module
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ConflictResolution, ImportJob, ImportStatus
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.UserProfiles.response_mappers import (
    LegacyProfileCommandResult,
)
from tldw_Server_API.app.services import core_jobs_worker

pytestmark = pytest.mark.unit


class FakePromptsDB:
    def __init__(self) -> None:
        self.prompts: list[dict] = []

    def get_prompt_by_name(self, name: str):
        return next((prompt for prompt in self.prompts if prompt["name"] == name), None)

    def add_prompt(self, **kwargs):
        self.prompts.append(kwargs)
        return len(self.prompts), f"prompt-{len(self.prompts)}", "added"


class FakeEvaluationsDB:
    def __init__(self) -> None:
        self.evaluations: dict[str, dict] = {}
        self.runs: list[dict] = []

    def get_evaluation(self, eval_id: str, *, created_by=None):
        return self.evaluations.get(eval_id)

    def create_evaluation(self, **kwargs):
        eval_id = kwargs.get("eval_id") or f"eval-{len(self.evaluations) + 1}"
        self.evaluations[eval_id] = kwargs
        return eval_id

    def create_run(self, eval_id, target_model=None, config=None, webhook_url=None, *, run_id=None):
        run_id = run_id or f"run-{len(self.runs) + 1}"
        self.runs.append(
            {
                "id": run_id,
                "eval_id": eval_id,
                "target_model": target_model,
                "config": config,
                "webhook_url": webhook_url,
            }
        )
        return run_id

    @contextmanager
    def get_connection(self):
        yield SimpleNamespace(execute=lambda *_args, **_kwargs: None, commit=lambda: None)


class FakeMediaDB:
    def __init__(self) -> None:
        self.media: dict[int, dict] = {}
        self.files: list[dict] = []
        self.vector_updates: list[tuple[int, bytes]] = []

    def get_media_by_title(self, title: str):
        return next((row for row in self.media.values() if row["title"] == title), None)

    def add_media_with_keywords(self, **kwargs):
        media_id = len(self.media) + 1
        self.media[media_id] = kwargs
        return media_id, f"media-uuid-{media_id}", "added"

    def insert_media_file(self, **kwargs):
        self.files.append(kwargs)
        return f"file-{len(self.files)}"

    @contextmanager
    def transaction(self):
        yield object()

    def _execute_with_connection(self, _conn, _sql, params):
        vector_blob, media_id = params
        self.vector_updates.append((int(media_id), bytes(vector_blob)))


class FakeChromaManager:
    def __init__(self, *, existing_ids: set[str] | None = None) -> None:
        self.stored: list[dict] = []
        self.existing_ids = existing_ids or set()

    def get_collection(self, collection_name):
        if not self.existing_ids:
            raise KeyError(collection_name)

        class _Collection:
            def __init__(self, ids: set[str]) -> None:
                self.ids = ids

            def get(self, ids):
                return {"ids": [item_id for item_id in ids if item_id in self.ids]}

        return _Collection(self.existing_ids)

    def store_in_chroma(self, collection_name, texts, embeddings, ids, metadatas):
        self.stored.append(
            {
                "collection_name": collection_name,
                "texts": texts,
                "embeddings": embeddings,
                "ids": ids,
                "metadatas": metadatas,
            }
        )


def _hash_entry(path: str, payload: bytes, *, media_type: str = "application/json") -> dict:
    return {
        "path": path,
        "media_type": media_type,
        "size_bytes": len(payload),
        "integrity": {
            "status": "verified",
            "algorithm": "sha256",
            "value": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        },
        "role": "payload",
        "content_item_ids": [],
    }


def _write_full_account_restore_fixture(
    path: Path,
    *,
    omit_media_artifact_inventory: bool = False,
) -> Path:
    files: dict[str, bytes] = {
        "content/notes/note_n1.md": b"---\ntitle: Restored note\n---\n\nNote body",
        "content/prompts/prompt_p1.json": json.dumps(
            {
                "id": 1,
                "name": "Prompt restore",
                "author": "tester",
                "details": "details",
                "system_prompt": "system",
                "user_prompt": "user",
                "prompt_format": "legacy",
                "keywords": ["restore"],
            }
        ).encode("utf-8"),
        "content/evaluations/evaluation_e1.json": json.dumps(
            {
                "id": "e1",
                "name": "Eval restore",
                "eval_type": "rag",
                "eval_spec": {"judge": "exact"},
                "metadata": {"scope": "test"},
                "runs": [
                    {
                        "id": "run-1",
                        "status": "completed",
                        "target_model": "test-model",
                        "config": {"temperature": 0},
                        "results": {"score": 1},
                    }
                ],
            }
        ).encode("utf-8"),
        "content/media/files/media_m1/file_1.txt": b"account-owned stored media bytes",
        "content/media/media_m1.json": json.dumps(
            {
                "id": "m1",
                "uuid": "media-source-uuid",
                "url": "https://example.test/source.txt",
                "title": "Media restore",
                "type": "document",
                "content": "media body",
                "author": "tester",
                "transcription_model": "whisper-test",
                "transcripts": [
                    {
                        "id": 1,
                        "transcription": "transcript text",
                        "whisper_model": "whisper-test",
                    }
                ],
                "chunks": [
                    {
                        "chunk_text": "media body",
                        "start_char": 0,
                        "end_char": 10,
                        "chunk_type": "semantic",
                    }
                ],
                "stored_artifacts": [
                    {
                        "id": 1,
                        "file_type": "original",
                        "original_filename": "source.txt",
                        "file_size": 32,
                        "mime_type": "text/plain",
                        "checksum": "sha256:test",
                        "bundled": True,
                        "pointer_only": False,
                        "archive_path": "content/media/files/media_m1/file_1.txt",
                    },
                    {
                        "id": 2,
                        "file_type": "source_pointer",
                        "original_filename": "external.txt",
                        "file_size": 64,
                        "mime_type": "text/plain",
                        "bundled": False,
                        "pointer_only": True,
                    },
                ],
            }
        ).encode("utf-8"),
        "content/embeddings/collection_restore.json": json.dumps(
            {
                "embedding_set_id": "restore_collection",
                "chunks": [
                    {
                        "id": "chunk-1",
                        "document": "embedded text",
                        "metadata": {"source": "fixture"},
                        "embedding": [0.1, 0.2, 0.3],
                    }
                ],
            }
        ).encode("utf-8"),
        "content/embeddings/embedding_media_m1.json": json.dumps(
            {
                "id": "media:m1",
                "source": {"media_id": "m1", "media_uuid": "media-source-uuid"},
                "encoding": "base64",
                "vector": base64.b64encode(b"vector-bytes").decode("ascii"),
            }
        ).encode("utf-8"),
        "content/generated_documents/document_d1.json": json.dumps(
            {
                "id": "d1",
                "document_type": "summary",
                "title": "Generated restore",
                "content": "generated content",
                "provider": "test-provider",
                "model": "test-model",
                "generation_time_ms": 5,
                "token_count": 2,
                "metadata": {"watchlist_id": "watchlist-1", "source": "fixture"},
            }
        ).encode("utf-8"),
    }
    manifest = {
        "version": "1.1.0",
        "name": "Full import restore",
        "description": "full account restore fixture",
        "author": None,
        "created_at": "2026-07-09T12:00:00+00:00",
        "updated_at": "2026-07-09T12:00:00+00:00",
        "export_id": "full-import-restore",
        "content_items": [
            {"id": "n1", "type": "note", "title": "Restored note", "file_path": "content/notes/note_n1.md"},
            {"id": "p1", "type": "prompt", "title": "Prompt restore", "file_path": "content/prompts/prompt_p1.json"},
            {"id": "e1", "type": "evaluation", "title": "Eval restore", "file_path": "content/evaluations/evaluation_e1.json"},
            {"id": "m1", "type": "media", "title": "Media restore", "file_path": "content/media/media_m1.json"},
            {"id": "collection:restore_collection", "type": "embedding", "title": "Collection embedding", "file_path": "content/embeddings/collection_restore.json"},
            {"id": "media:m1", "type": "embedding", "title": "Media embedding", "file_path": "content/embeddings/embedding_media_m1.json"},
            {"id": "d1", "type": "generated_document", "title": "Generated restore", "file_path": "content/generated_documents/document_d1.json"},
        ],
        "relationships": [],
        "configuration": {
            "include_media": True,
            "include_embeddings": True,
            "include_generated_content": True,
            "media_quality": "original",
            "max_file_size_mb": 100,
        },
        "statistics": {
            "total_conversations": 0,
            "total_notes": 1,
            "total_characters": 0,
            "total_media_items": 1,
            "total_prompts": 1,
            "total_evaluations": 1,
            "total_embeddings": 2,
            "total_world_books": 0,
            "total_dictionaries": 0,
            "total_documents": 1,
            "total_size_bytes": sum(len(payload) for payload in files.values()),
        },
        "metadata": {"tags": [], "categories": [], "language": "en", "license": None},
        "user_info": {"user_id": "redacted"},
        "features_used": ["file_inventory", "account_inventory"],
        "producer": {"name": "tldw_server"},
        "source_instance": {},
        "compatibility": {"min_reader_version": "1.0.0", "recommended_reader_version": "1.1.0"},
        "file_inventory": [
            _hash_entry(file_path, payload)
            for file_path, payload in files.items()
            if not (omit_media_artifact_inventory and file_path == "content/media/files/media_m1/file_1.txt")
        ],
        "account_inventory": [
            {"category": "notes", "restore_status": "restorable"},
            {"category": "prompts", "restore_status": "restorable"},
            {"category": "evaluations", "restore_status": "restorable"},
            {"category": "media_records", "restore_status": "restorable"},
            {"category": "media_stored_artifacts", "restore_status": "restorable"},
            {
                "category": "media_pointers",
                "restore_status": "pointer_only",
                "warning": "External media URLs and local paths restore as references only.",
            },
            {"category": "embeddings", "restore_status": "restorable"},
            {"category": "generated_documents", "restore_status": "restorable"},
            {
                "category": "sensitive_user_values",
                "restore_status": "non_restorable",
                "warning": "SECRET_TOKEN_SHOULD_NOT_SURFACE",
            },
        ],
        "account_inventory_summary": {
            "counts": {
                "notes": 1,
                "prompts": 1,
                "evaluations": 1,
                "media_records": 1,
                "media_stored_artifacts": 2,
                "media_pointers": 1,
                "embeddings": 2,
                "generated_documents": 1,
                "sensitive_user_values": 1,
            },
            "pointer_only_count": 1,
            "sensitive_category_count": 1,
            "warning_count": 2,
            "archive_size_bytes": 0,
            "post_write_verification": True,
        },
    }
    with zipfile.ZipFile(path, "w") as zf:
        for file_path, payload in files.items():
            zf.writestr(file_path, payload)
        zf.writestr("manifest.json", json.dumps(manifest))
    return path


def _write_account_state_contract_fixture(
    path: Path,
    *,
    include_profile_payload: bool,
    include_profile_inventory: bool,
) -> Path:
    profile_path = "json/account_profile.json"
    profile_payload = json.dumps(
        {
            "schema_version": "1.0",
            "category": "account_profile",
            "profile": {"identity.email": "source-account@example.com"},
            "policy": {},
        }
    ).encode("utf-8")
    manifest = {
        "version": "1.1.0",
        "name": "Account state contract",
        "description": "fail-closed account state fixture",
        "created_at": "2026-07-09T12:00:00+00:00",
        "updated_at": "2026-07-09T12:00:00+00:00",
        "content_items": [],
        "relationships": [],
        "configuration": {},
        "statistics": {"total_size_bytes": len(profile_payload)},
        "metadata": {"tags": [], "categories": [], "language": "en", "license": None},
        "user_info": {"user_id": "redacted"},
        "features_used": ["file_inventory"],
        "producer": {"name": "tldw_server"},
        "source_instance": {},
        "compatibility": {
            "min_reader_version": "1.0.0",
            "recommended_reader_version": "1.1.0",
        },
        "file_inventory": (
            [_hash_entry(profile_path, profile_payload)]
            if include_profile_inventory
            else []
        ),
        "account_inventory": [
            {
                "category": "account_profile",
                "restore_status": "restorable",
                "export_representation": profile_path,
            }
        ],
        "account_inventory_summary": {
            "counts": {"account_profiles": 1, "account_settings": 0}
        },
    }
    with zipfile.ZipFile(path, "w") as zf:
        if include_profile_payload:
            zf.writestr(profile_path, profile_payload)
        zf.writestr("manifest.json", json.dumps(manifest))
    return path


@pytest.mark.asyncio
async def test_service_default_archive_import_restores_restorable_full_account_payloads(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "restore.db"), client_id="restore-test")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)
    archive_path = _write_full_account_restore_fixture(Path(service.import_dir) / "full_restore.chatbook")
    prompts_db = FakePromptsDB()
    evals_db = FakeEvaluationsDB()
    media_db = FakeMediaDB()
    chroma = FakeChromaManager()
    transcript_calls: list[dict] = []

    monkeypatch.setattr(service, "_get_prompts_db", lambda: prompts_db)
    monkeypatch.setattr(service, "_get_evaluations_db", lambda: evals_db)
    monkeypatch.setattr(service, "_get_media_db", lambda: media_db)
    monkeypatch.setattr(service, "_get_chroma_manager", lambda: chroma)
    monkeypatch.setattr(
        chatbook_service_module,
        "upsert_transcript",
        lambda *args, **kwargs: transcript_calls.append({"args": args, "kwargs": kwargs}) or {},
    )

    success, message, result = await service.import_chatbook(
        file_path=str(archive_path),
        conflict_resolution=ConflictResolution.SKIP,
        source_format="chatbook",
        async_mode=False,
    )

    assert success is True, message
    assert result["imported_items"]["note"] == 1
    assert result["imported_items"]["prompt"] == 1
    assert result["imported_items"]["evaluation"] == 1
    assert result["imported_items"]["media"] == 1
    assert result["imported_items"]["embedding"] == 2
    assert result["imported_items"]["generated_document"] == 1
    assert prompts_db.prompts[0]["name"] == "Prompt restore"
    assert evals_db.evaluations["e1"]["name"] == "Eval restore"
    assert evals_db.runs[0]["id"] == "run-1"
    assert media_db.media[1]["title"] == "Media restore"
    assert media_db.files[0]["storage_path"].startswith("imported_media/media_1/")
    assert media_db.vector_updates == [(1, b"vector-bytes")]
    assert transcript_calls
    assert chroma.stored[0]["collection_name"] == "restore_collection"
    [restored_document] = DocumentGeneratorService(db, user_id="1").get_generated_documents(limit=10)
    assert restored_document["metadata"]["watchlist_id"] == "watchlist-1"
    assert result["skipped_non_restorable"]["media_pointers"] == 1
    assert result["skipped_non_restorable"]["sensitive_user_values"] == 1
    warnings = "\n".join(result["warnings"])
    assert "pointer" in warnings.lower()
    assert "SECRET" not in warnings


@pytest.mark.asyncio
async def test_live_worker_restores_bundled_media_and_records_inventory_result(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "live_worker_restore.db"), client_id="live-worker-restore")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)
    archive_path = _write_full_account_restore_fixture(
        Path(service.import_dir) / "live_worker_full_restore.chatbook"
    )
    import_job = ImportJob(
        job_id="live-worker-import",
        user_id="1",
        status=ImportStatus.PENDING,
        chatbook_path=str(archive_path),
    )
    service._save_import_job(import_job)
    prompts_db = FakePromptsDB()
    evals_db = FakeEvaluationsDB()
    media_db = FakeMediaDB()
    chroma = FakeChromaManager()
    stop_event = asyncio.Event()
    public_job_status = "processing"
    failures = []

    monkeypatch.setattr(service, "_get_prompts_db", lambda: prompts_db)
    monkeypatch.setattr(service, "_get_evaluations_db", lambda: evals_db)
    monkeypatch.setattr(service, "_get_media_db", lambda: media_db)
    monkeypatch.setattr(service, "_get_chroma_manager", lambda: chroma)
    monkeypatch.setattr(chatbook_service_module, "upsert_transcript", lambda *args, **kwargs: {})

    class FakeJobManager:
        def acquire_next_job(self, **_kwargs):
            return {
                "id": 43,
                "owner_user_id": "1",
                "payload": {
                    "action": "import",
                    "chatbooks_job_id": import_job.job_id,
                    "file_token": str(archive_path),
                    "source_format": "chatbook",
                    "content_selections": None,
                    "import_media": True,
                    "import_embeddings": True,
                },
                "lease_id": "lease-2",
            }

        def get_job(self, _job_id):
            return {"status": public_job_status}

        def complete_job(self, *_args, **_kwargs):
            nonlocal public_job_status
            public_job_status = "completed"
            stop_event.set()

        def fail_job(self, *_args, **kwargs):
            nonlocal public_job_status
            public_job_status = "failed"
            failures.append(kwargs)
            stop_event.set()

        def finalize_cancelled(self, *_args, **_kwargs):
            stop_event.set()

        def renew_job_lease(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(core_jobs_worker, "JobManager", FakeJobManager)
    monkeypatch.setattr(core_jobs_worker, "_build_chacha_db_for_user", lambda _owner: db)
    monkeypatch.setattr(core_jobs_worker, "ChatbookService", lambda *_args, **_kwargs: service)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.01")

    await asyncio.wait_for(core_jobs_worker.run_chatbooks_core_jobs_worker(stop_event), timeout=2)

    restored_job = service._get_import_job(import_job.job_id)
    assert failures == []
    assert public_job_status == "completed"
    assert restored_job is not None
    assert restored_job.status is ImportStatus.COMPLETED
    assert restored_job.metadata["imported_items"]["media"] == 1
    assert restored_job.metadata["inventory_summary"]["counts"]["media_stored_artifacts"] == 2
    assert media_db.files
    user_root = DatabasePaths.resolve_user_base_directory(1).resolve()
    restored_path = (user_root / media_db.files[0]["storage_path"]).resolve()
    assert restored_path.is_relative_to(user_root)
    assert restored_path.read_bytes() == b"account-owned stored media bytes"


def test_v1_1_import_rejects_bundled_media_artifact_missing_inventory(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "artifact_inventory.db"), client_id="artifact-inventory")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)
    archive_path = _write_full_account_restore_fixture(
        Path(service.import_dir) / "missing_artifact_inventory.chatbook",
        omit_media_artifact_inventory=True,
    )

    success, message, result = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=True,
        import_embeddings=True,
    )

    assert success is False
    assert "File inventory validation failed for content/media/files/media_m1/file_1.txt" in message
    assert result["imported_items"] == {}


@pytest.mark.parametrize(
    ("include_profile_payload", "include_profile_inventory", "expected_error"),
    [
        (False, False, "missing its serialized restore payload"),
        (True, False, "missing verified file inventory entry"),
    ],
)
def test_sync_import_rejects_unverified_or_missing_account_profile_payload(
    tmp_path,
    monkeypatch,
    include_profile_payload,
    include_profile_inventory,
    expected_error,
):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "account_contract.db"), client_id="account-contract")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)
    archive_path = _write_account_state_contract_fixture(
        Path(service.import_dir) / "account_contract.chatbook",
        include_profile_payload=include_profile_payload,
        include_profile_inventory=include_profile_inventory,
    )

    success, message, result = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=True,
        import_embeddings=True,
    )

    assert success is False
    assert expected_error in message
    assert result["imported_items"] == {}


@pytest.mark.asyncio
async def test_finalize_account_restore_removes_private_payload_before_return(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "account_finalize.db"), client_id="account-finalize")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)
    source_email = "source-account@example.com"
    private_key = chatbook_service_module._ACCOUNT_RESTORE_PAYLOAD_KEY
    private_payload = {
        "account_profile": {
            "schema_version": "1.0",
            "category": "account_profile",
            "profile": {"identity.email": source_email},
        }
    }
    observed_payload = None

    async def _restore(payload):
        nonlocal observed_payload
        observed_payload = payload
        return {"account_profile": 1, "account_settings": 0}

    monkeypatch.setattr(service, "_restore_account_state_payloads", _restore)
    success, message, result = await service.finalize_account_restore(
        True,
        "Import completed",
        {"imported_items": {}, private_key: private_payload},
    )

    assert success is True
    assert observed_payload == private_payload
    assert result["imported_items"]["account_profile"] == 1
    assert private_key not in result
    assert source_email not in json.dumps(result)
    assert source_email not in message


@pytest.mark.asyncio
async def test_account_restore_delegates_ordered_profile_updates_to_command_service(
    monkeypatch,
):
    from tldw_Server_API.app.core.AuthNZ import database as database_module
    from tldw_Server_API.app.core.AuthNZ.repos import users_repo as users_repo_module
    from tldw_Server_API.app.core.UserProfiles import command_service as command_service_module

    captured: dict[str, object] = {}

    class _Pool:
        @asynccontextmanager
        async def transaction(self):
            connection = object()
            captured["connection"] = connection
            yield connection

    pool = _Pool()

    async def _get_pool():
        return pool

    class _Repo:
        def __init__(self, *, db_pool):
            assert db_pool is pool

        async def get_user_by_id(self, user_id):
            return {"id": user_id}

    class _CommandService:
        def __init__(self, *, db_pool):
            assert db_pool is pool

        async def apply(self, command, *, db_conn, scope):
            captured["command"] = command
            captured["db_conn"] = db_conn
            captured["scope"] = scope
            return LegacyProfileCommandResult(
                applied=(
                    "identity.email",
                    "preferences.audio.voice",
                    "preferences.ui.theme",
                )
            )

    monkeypatch.setattr(database_module, "get_db_pool", _get_pool)
    monkeypatch.setattr(users_repo_module, "AuthnzUsersRepo", _Repo)
    monkeypatch.setattr(command_service_module, "ProfileCommandService", _CommandService)

    service = SimpleNamespace(user_id_int=7)
    result = await ChatbookService._restore_account_state_payloads(
        service,
        {
            "account_profile": {
                "profile": {"identity.email": "restored@example.com"},
            },
            "account_settings": {
                "overrides": {
                    "preferences.ui.theme": "paper",
                    "preferences.audio.voice": "alloy",
                },
            },
        },
    )

    command = captured["command"]
    assert command.updates == (
        ("identity.email", "restored@example.com"),
        ("preferences.audio.voice", "alloy"),
        ("preferences.ui.theme", "paper"),
    )
    assert captured["db_conn"] is captured["connection"]
    assert captured["scope"] is None
    assert result == {"account_profile": 1, "account_settings": 1}


@pytest.mark.asyncio
async def test_account_restore_skips_unchanged_synthetic_single_user_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idempotent restore must not revalidate the bootstrap-only .local email."""
    from tldw_Server_API.app.core.AuthNZ import database as database_module
    from tldw_Server_API.app.core.AuthNZ.repos import users_repo as users_repo_module
    from tldw_Server_API.app.core.UserProfiles import command_service as command_service_module

    class _Pool:
        """Provide the transaction context used by account restoration."""

        @asynccontextmanager
        async def transaction(self) -> AsyncIterator[object]:
            """Yield a stand-in database connection."""
            yield object()

    pool = _Pool()

    async def _get_pool() -> _Pool:
        """Return the stubbed AuthNZ database pool."""
        return pool

    class _Repo:
        """Return the existing synthetic single-user identity."""

        def __init__(self, *, db_pool: object) -> None:
            """Validate that restoration uses the expected pool."""
            assert db_pool is pool

        async def get_user_by_id(self, user_id: int) -> dict[str, object]:
            """Return the unchanged bootstrap account."""
            return {"id": user_id, "email": "single_user@example.local"}

    class _CommandService:
        """Fail if unchanged values reach profile validation."""

        def __init__(self, *, db_pool: object) -> None:
            """Reject construction because no profile command is expected."""
            raise AssertionError("unchanged profile values must not be sent for validation")

    monkeypatch.setattr(database_module, "get_db_pool", _get_pool)
    monkeypatch.setattr(users_repo_module, "AuthnzUsersRepo", _Repo)
    monkeypatch.setattr(command_service_module, "ProfileCommandService", _CommandService)

    service = SimpleNamespace(user_id_int=1)
    result = await ChatbookService._restore_account_state_payloads(
        service,
        {
            "account_profile": {
                "profile": {"identity.email": "single_user@example.local"},
            }
        },
    )

    assert result == {"account_profile": 1, "account_settings": 0}


@pytest.mark.asyncio
async def test_embedding_import_skip_does_not_overwrite_existing_chroma_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "embedding_conflict.db"), client_id="embedding-conflict")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)
    archive_path = _write_full_account_restore_fixture(Path(service.import_dir) / "embedding_conflict.chatbook")
    media_db = FakeMediaDB()
    chroma = FakeChromaManager(existing_ids={"chunk-1"})

    monkeypatch.setattr(service, "_get_prompts_db", lambda: FakePromptsDB())
    monkeypatch.setattr(service, "_get_evaluations_db", lambda: FakeEvaluationsDB())
    monkeypatch.setattr(service, "_get_media_db", lambda: media_db)
    monkeypatch.setattr(service, "_get_chroma_manager", lambda: chroma)
    monkeypatch.setattr(chatbook_service_module, "upsert_transcript", lambda *args, **kwargs: {})

    success, message, result = await service.import_chatbook(
        file_path=str(archive_path),
        conflict_resolution=ConflictResolution.SKIP,
        source_format="chatbook",
        async_mode=False,
    )

    assert success is True, message
    assert chroma.stored == []
    assert media_db.vector_updates == [(1, b"vector-bytes")]
    assert result["imported_items"]["embedding"] == 1
    assert any("Skipped 1 existing embedding id" in warning for warning in result["warnings"])


def test_sync_import_fails_on_unknown_restorable_inventory_category(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "unknown.db"), client_id="unknown-inventory")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)
    archive_path = Path(service.import_dir) / "unknown_inventory.chatbook"
    manifest = {
        "version": "1.1.0",
        "name": "Unknown inventory",
        "description": "should fail closed",
        "created_at": "2026-07-09T12:00:00+00:00",
        "updated_at": "2026-07-09T12:00:00+00:00",
        "content_items": [],
        "relationships": [],
        "configuration": {},
        "statistics": {"total_size_bytes": 0},
        "metadata": {"tags": [], "categories": [], "language": "en", "license": None},
        "user_info": {"user_id": "redacted"},
        "features_used": ["account_inventory"],
        "producer": {"name": "tldw_server"},
        "source_instance": {},
        "compatibility": {"min_reader_version": "1.0.0", "recommended_reader_version": "1.1.0"},
        "file_inventory": [],
        "account_inventory": [{"category": "future_records", "restore_status": "restorable"}],
        "account_inventory_summary": {"counts": {"future_records": 1}},
    }
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))

    success, message, result = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections=None,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=True,
        import_embeddings=True,
    )

    assert success is False
    assert "inventory restore coverage missing" in message.lower()
    assert "future_records" in result["errors"][0]


@pytest.mark.asyncio
async def test_service_allows_explicit_chatbook_media_and_embedding_restore_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "explicit.db"), client_id="explicit-flags")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)
    archive_path = Path(service.import_dir) / "empty.chatbook"
    manifest = {
        "version": "1.0.0",
        "name": "Empty",
        "description": "explicit flags should not be rejected",
        "created_at": "2026-07-09T12:00:00+00:00",
        "updated_at": "2026-07-09T12:00:00+00:00",
        "content_items": [],
        "relationships": [],
        "configuration": {},
        "statistics": {},
        "metadata": {},
        "user_info": {"user_id": "redacted"},
    }
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))

    success, message, result = await service.import_chatbook(
        file_path=str(archive_path),
        conflict_resolution=ConflictResolution.SKIP,
        source_format="chatbook",
        import_media=True,
        import_embeddings=True,
        async_mode=False,
    )

    assert success is True, message
    assert result == {"imported_items": {}, "warnings": []}


@pytest.mark.asyncio
async def test_service_rejects_openwebui_archive_restore_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db = CharactersRAGDB(db_path=str(tmp_path / "openwebui.db"), client_id="openwebui-flags")
    service = ChatbookService(user_id="1", db=db, user_id_int=1)

    success, message, result = await service.import_chatbook(
        file_path=str(tmp_path / "openwebui.json"),
        source_format="openwebui_json",
        import_media=True,
        import_embeddings=False,
        async_mode=False,
    )

    assert success is False
    assert result is None
    assert "OpenWebUI imports do not use archive media or embedding restore options" in message
