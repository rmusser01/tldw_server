from __future__ import annotations

import asyncio
from contextlib import suppress
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackCreateJobRequest, StudyPackSourceSelection
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK


pytestmark = pytest.mark.asyncio


def _load_modules():
    try:
        types_mod = import_module("tldw_Server_API.app.core.StudyPacks.types")
        generation_mod = import_module("tldw_Server_API.app.core.StudyPacks.generation_service")
        worker_mod = import_module("tldw_Server_API.app.services.study_pack_jobs_worker")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Study pack jobs worker modules are missing: {exc}")
    except ImportError as exc:
        pytest.fail(f"Study pack jobs worker imports are not yet usable: {exc}")
    return types_mod, generation_mod, worker_mod


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "study-pack-worker.db"), client_id="study-pack-worker-tests")
    chacha.upsert_workspace("ws-1", "Workspace 1")
    try:
        yield chacha
    finally:
        chacha.close_connection()


@pytest.fixture
def jobs_db_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "study-pack-worker-jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(path))
    return path


def _bundle(types_mod: Any):
    return types_mod.StudySourceBundle(
        items=[
            types_mod.StudySourceBundleItem(
                source_type="note",
                source_id="note-1",
                label="Slow Start Notes",
                evidence_text="Slow start doubles the congestion window each RTT.",
                locator={"note_id": "note-1"},
            )
        ]
    )


def _request() -> StudyPackCreateJobRequest:
    return StudyPackCreateJobRequest(
        title="TCP Fundamentals",
        workspace_id="ws-1",
        source_items=[StudyPackSourceSelection(source_type="note", source_id="note-1")],
    )


def _generation_result(types_mod: Any):
    return types_mod.StudyPackGenerationResult(
        cards=[
            types_mod.StudyPackCardDraft(
                front="What is TCP slow start?",
                back="It doubles the congestion window each RTT.",
                citations=[
                    types_mod.StudyCitationDraft(
                        source_type="note",
                        source_id="note-1",
                        citation_text="Slow start doubles the congestion window each RTT.",
                        locator={"note_id": "note-1"},
                    )
                ],
            )
        ]
    )


def _fake_job(
    *,
    job_id: int = 1,
    regenerate_from_pack_id: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": "TCP Fundamentals",
        "workspace_id": "ws-1",
        "source_items": [{"source_type": "note", "source_id": "note-1"}],
    }
    if regenerate_from_pack_id is not None:
        payload["regenerate_from_pack_id"] = regenerate_from_pack_id
    return {
        "id": job_id,
        "owner_user_id": "1",
        "payload": payload,
    }


async def test_handle_study_pack_job_rolls_back_late_persistence_failures(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    types_mod, generation_mod, worker_mod = _load_modules()
    async def fake_get_databases_for_user(user_id: str):
        assert user_id == "1"  # nosec B101
        return db, SimpleNamespace()

    monkeypatch.setattr(worker_mod, "_get_databases_for_user", fake_get_databases_for_user)
    monkeypatch.setattr(generation_mod.StudySourceResolver, "resolve", lambda self, selections: _bundle(types_mod))

    async def fake_generate_validated_cards(self, resolved_bundle: Any, request: Any):
        return _generation_result(types_mod)

    monkeypatch.setattr(
        generation_mod.StudyPackGenerationService,
        "generate_validated_cards",
        fake_generate_validated_cards,
    )

    original_add_study_pack_cards = db.add_study_pack_cards

    def fail_membership_insert(study_pack_id: int, flashcard_uuids: list[str]) -> int:
        original_add_study_pack_cards(study_pack_id, flashcard_uuids)
        raise CharactersRAGDBError("membership write failed")

    monkeypatch.setattr(db, "add_study_pack_cards", fail_membership_insert)

    with pytest.raises(CharactersRAGDBError, match="membership write failed"):
        await worker_mod.handle_study_pack_job(_fake_job())

    deck_names = [deck["name"] for deck in db.list_decks(include_workspace_items=True)]
    study_packs = db.execute_query("SELECT COUNT(*) AS count FROM study_packs WHERE deleted = 0").fetchone()

    assert "TCP Fundamentals" not in deck_names  # nosec B101
    assert int(study_packs["count"]) == 0  # nosec B101


async def test_handle_study_pack_job_regeneration_supersedes_only_after_successful_commit(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    types_mod, generation_mod, worker_mod = _load_modules()
    original_deck_id = db.add_deck("TCP Fundamentals Original", workspace_id="ws-1")
    original_pack_id = db.create_study_pack(
        title="TCP Fundamentals Original",
        workspace_id="ws-1",
        deck_id=original_deck_id,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "note-seed"}]},
        generation_options_json={"deck_mode": "new"},
    )

    async def fake_get_databases_for_user(user_id: str):
        assert user_id == "1"  # nosec B101
        return db, SimpleNamespace()

    monkeypatch.setattr(worker_mod, "_get_databases_for_user", fake_get_databases_for_user)
    monkeypatch.setattr(generation_mod.StudySourceResolver, "resolve", lambda self, selections: _bundle(types_mod))

    async def fake_generate_validated_cards(self, resolved_bundle: Any, request: Any):
        return _generation_result(types_mod)

    monkeypatch.setattr(
        generation_mod.StudyPackGenerationService,
        "generate_validated_cards",
        fake_generate_validated_cards,
    )

    original_persist = generation_mod.FlashcardProvenanceStore.persist_flashcard_citations
    failure_state = {"should_fail": True}

    def fail_once(self, flashcard_uuid: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
        if failure_state["should_fail"]:
            raise CharactersRAGDBError("citation persistence failed")
        return original_persist(self, flashcard_uuid, citations)

    monkeypatch.setattr(generation_mod.FlashcardProvenanceStore, "persist_flashcard_citations", fail_once)

    with pytest.raises(CharactersRAGDBError, match="citation persistence failed"):
        await worker_mod.handle_study_pack_job(
            _fake_job(job_id=2, regenerate_from_pack_id=original_pack_id)
        )

    failed_original_row = db.execute_query(
        "SELECT status, superseded_by_pack_id FROM study_packs WHERE id = ?",
        (original_pack_id,),
    ).fetchone()

    assert failed_original_row["status"] == "active"  # nosec B101
    assert failed_original_row["superseded_by_pack_id"] is None  # nosec B101

    failure_state["should_fail"] = False
    result = await worker_mod.handle_study_pack_job(
        _fake_job(job_id=3, regenerate_from_pack_id=original_pack_id)
    )

    original_row = db.execute_query(
        "SELECT status, superseded_by_pack_id FROM study_packs WHERE id = ?",
        (original_pack_id,),
    ).fetchone()

    assert int(result["pack_id"]) > 0  # nosec B101
    assert original_row["status"] == "superseded"  # nosec B101
    assert int(original_row["superseded_by_pack_id"]) == int(result["pack_id"])  # nosec B101


async def test_worker_sdk_cancellation_marks_study_pack_job_cancelled_without_running_handler(
    jobs_db_path: Path,
):
    _types_mod, _generation_mod, worker_mod = _load_modules()
    jm = JobManager(db_path=jobs_db_path)
    job = jm.create_job(
        domain="study_packs",
        queue="default",
        job_type="study_pack_generate",
        payload=_request().model_dump(),
        owner_user_id="1",
        priority=5,
        max_retries=2,
    )

    cfg = WorkerConfig(
        domain="study_packs",
        queue="default",
        worker_id="study-pack-worker-test",
        lease_seconds=5,
        renew_threshold_seconds=1,
        renew_jitter_seconds=0,
    )
    sdk = WorkerSDK(jm, cfg)

    async def handler(job_row: dict[str, Any]):
        pytest.fail("Study-pack handler should not run when cancel_check returns true")

    cancel_check_started = asyncio.Event()

    async def cancel_check(job_row: dict[str, Any]) -> bool:
        cancel_check_started.set()
        jm.cancel_job(int(job_row["id"]), reason="requested")
        return await worker_mod._should_cancel(job_row, job_manager=jm)

    task = asyncio.create_task(sdk.run(handler=handler, cancel_check=cancel_check))
    await asyncio.wait_for(cancel_check_started.wait(), timeout=1)
    sdk.stop()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    stored = jm.get_job(int(job["id"]))

    assert stored is not None  # nosec B101
    assert stored["status"] == "cancelled"  # nosec B101


async def test_get_databases_for_user_closes_note_db_when_media_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    _types_mod, _generation_mod, worker_mod = _load_modules()

    class FakeNoteDb:
        def __init__(self) -> None:
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    note_db = FakeNoteDb()

    async def fake_get_chacha_db_for_user_id(user_id: int, *, client_id: str):
        assert user_id == 1  # nosec B101
        assert client_id == "study-pack-worker-1"  # nosec B101
        return note_db

    def fake_get_media_db_for_owner(user_id: int):
        assert user_id == 1  # nosec B101
        raise RuntimeError("media lookup failed")

    monkeypatch.setattr(worker_mod, "get_chacha_db_for_user_id", fake_get_chacha_db_for_user_id)
    monkeypatch.setattr(worker_mod, "get_media_db_for_owner", fake_get_media_db_for_owner)

    with pytest.raises(RuntimeError, match="media lookup failed"):
        await worker_mod._get_databases_for_user("1")

    assert note_db.closed is True  # nosec B101


async def test_handle_study_pack_job_resolves_default_provider_and_model(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    types_mod, generation_mod, worker_mod = _load_modules()
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_MODEL_OPENAI", "gpt-default-study-pack")
    async def fake_get_databases_for_user(user_id: str):
        assert user_id == "1"  # nosec B101
        return db, SimpleNamespace()

    monkeypatch.setattr(worker_mod, "_get_databases_for_user", fake_get_databases_for_user)

    captured: dict[str, Any] = {}

    async def fake_create_from_request(
        self,
        request: Any,
        *,
        regenerate_from_pack_id: int | None = None,
        expected_regenerate_version: int | None = None,
    ):
        captured["provider"] = self.provider
        captured["model"] = self.model
        captured["expected_regenerate_version"] = expected_regenerate_version
        return types_mod.StudyPackCreationResult(
            pack_id=11,
            deck_id=22,
            deck_name="TCP Fundamentals",
            card_uuids=["card-1"],
            cards=_generation_result(types_mod).cards,
            regenerated_from_pack_id=regenerate_from_pack_id,
        )

    monkeypatch.setattr(generation_mod.StudyPackGenerationService, "create_from_request", fake_create_from_request)

    result = await worker_mod.handle_study_pack_job(_fake_job())

    assert captured["provider"] == "openai"  # nosec B101
    assert captured["model"] == "gpt-default-study-pack"  # nosec B101
    assert captured["expected_regenerate_version"] is None  # nosec B101
    assert int(result["pack_id"]) == 11  # nosec B101
