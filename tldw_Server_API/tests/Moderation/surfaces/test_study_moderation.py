"""Study tools name their surface and a refusal propagates (TASK-13950.27 PR H)."""
# ruff: noqa: F811 - fixtures are imported by name, then requested as arguments


from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.tests.Flashcards.test_flashcards_endpoint_integration import (  # noqa: F401 - fixtures
    AUTH_HEADERS,
    client_with_flashcards_db,
    flashcards_db,
)
from tldw_Server_API.tests.Moderation.conftest import reply

pytestmark = pytest.mark.integration

PROVIDER = "anthropic"  # a real adapter name outside chat's TEST_MODE mock set; the gateway registry is stubbed


@pytest.fixture
def spy(stub_llm_registry):
    """A test pre-hook recording ``(call.surface, call.user_id)`` for every gateway call."""
    seen: list[tuple[str | None, int | None]] = []
    remove = gateway.add_policy_hooks(pre_call=lambda call: seen.append((call.surface, call.user_id)))
    yield seen
    remove()


# ---- study assistant ------------------------------------------------------------------------------------------


def _respond(client, **extra):
    created = client.post(
        "/api/v1/flashcards",
        json={"front": "What is the glomerulus?", "back": "It filters blood."},
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200, created.text
    return client.post(
        f"/api/v1/flashcards/{created.json()['uuid']}/assistant/respond",
        json={"action": "freeform", "provider": PROVIDER, "model": "claude-test", **extra},
        headers=AUTH_HEADERS,
    )


def test_study_assistant_input_block_400(client_with_flashcards_db, stub_llm_registry, spy, moderation_hooks):
    response = _respond(client_with_flashcards_db, message="forbidden words")

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Input violates moderation policy"
    assert stub_llm_registry.requests == []
    assert [surface for surface, _ in spy] == ["study.assistant"]


def test_study_assistant_output_redact(client_with_flashcards_db, stub_llm_registry, moderation_hooks):
    stub_llm_registry.reply = reply("the secret reply")

    response = _respond(client_with_flashcards_db, message="hello")

    assert response.status_code == 200, response.text
    assert response.json()["assistant_message"]["content"] == "the [X] reply"


def test_study_assistant_output_block_400(client_with_flashcards_db, stub_llm_registry, moderation_hooks):
    """A blocked answer (finish_reason content_filter) is a refusal, not the stored assistant text."""
    stub_llm_registry.reply = reply("a forbidden reply")

    response = _respond(client_with_flashcards_db, message="hello")

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Output violates moderation policy"


# ---- study packs ----------------------------------------------------------------------------------------------


def _service(db, monkeypatch):
    from tldw_Server_API.app.core.StudyPacks import generation_service

    return generation_service.StudyPackGenerationService(
        note_db=db, media_db=SimpleNamespace(), provider=PROVIDER, model="claude-test"
    )


def _bundle(text: str):
    from tldw_Server_API.app.core.StudyPacks.types import StudySourceBundle, StudySourceBundleItem

    return StudySourceBundle(
        items=[
            StudySourceBundleItem(
                source_type="note", source_id="note-1", label="Notes", evidence_text=text, locator={"note_id": "note-1"}
            )
        ]
    )


def _job_request():
    from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackCreateJobRequest, StudyPackSourceSelection

    return StudyPackCreateJobRequest(
        title="Pack", workspace_id="ws-1", source_items=[StudyPackSourceSelection(source_type="note", source_id="note-1")]
    )


@pytest.fixture
def pack_db(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(str(tmp_path / "pack.db"), client_id="study-moderation")
    db.workspace_store.upsert_workspace("ws-1", "Workspace 1")
    try:
        yield db
    finally:
        db.close_connection()


@pytest.fixture
def pack_job(pack_db, monkeypatch):
    """Run ``handle_study_pack_job`` for owner 7 over a bundle of ``evidence`` text; returns the runner."""
    from tldw_Server_API.app.core.StudyPacks import generation_service
    from tldw_Server_API.app.services import study_pack_jobs_worker as worker

    async def databases(_user_id: str):
        return pack_db, SimpleNamespace()

    monkeypatch.setattr(worker, "_get_databases_for_user", databases)

    async def run(evidence: str):
        monkeypatch.setattr(generation_service.StudySourceResolver, "resolve", lambda self, selections: _bundle(evidence))
        job = {"id": 1, "owner_user_id": "7", "payload": _job_request().model_dump(mode="json")}
        return await worker.handle_study_pack_job(job)

    return run


@pytest.mark.asyncio
async def test_study_pack_generation_block_fails_job(pack_job, stub_llm_registry, spy, moderation_hooks, monkeypatch):
    from tldw_Server_API.app.core.StudyPacks import generation_service

    monkeypatch.setattr(generation_service, "_resolve_generation_provider_and_model", lambda p, m: (PROVIDER, "claude-test", {}))
    with pytest.raises(LLMPolicyRefusal) as exc:
        await pack_job("forbidden words in the notes")

    assert exc.value.detail == "Input violates moderation policy"
    assert exc.value.retryable is False  # WorkerSDK reads this: the job fails, it is not re-queued
    assert stub_llm_registry.requests == []
    assert [surface for surface, _ in spy] == ["study.packs"]


@pytest.mark.asyncio
async def test_study_pack_worker_call_carries_owner_user_id(pack_job, stub_llm_registry, spy, monkeypatch):
    """The worker's ``scoped_context(user_id=owner)`` reaches ``call.user_id``, so a per-user override applies."""
    from tldw_Server_API.app.core.StudyPacks import generation_service

    monkeypatch.setattr(generation_service, "_resolve_generation_provider_and_model", lambda p, m: (PROVIDER, "claude-test", {}))
    with pytest.raises(generation_service.StudyPackMalformedResponseError):  # the stub's reply is not a card list; only the call's identity matters here
        await pack_job("slow start doubles the window")

    assert spy and spy[0] == ("study.packs", 7)


@pytest.mark.asyncio
async def test_study_pack_output_block_is_not_repaired(pack_db, stub_llm_registry, moderation_hooks, monkeypatch):
    """A blocked answer is refused at once: no repair pass, no second provider call."""
    from tldw_Server_API.app.core.StudyPacks import generation_service

    monkeypatch.setattr(generation_service, "_resolve_generation_provider_and_model", lambda p, m: (PROVIDER, "claude-test", {}))
    stub_llm_registry.reply = reply("a forbidden reply")
    service = _service(pack_db, monkeypatch)

    with pytest.raises(LLMPolicyRefusal) as exc:
        await service.generate_validated_cards(_bundle("slow start"), _job_request())

    assert exc.value.detail == "Output violates moderation policy"
    assert len(stub_llm_registry.requests) == 1


# ---- quizzes --------------------------------------------------------------------------------------------------


@pytest.fixture
def quiz_llm(monkeypatch):
    from tldw_Server_API.app.services import quiz_generator

    monkeypatch.setattr(quiz_generator, "load_and_log_configs", lambda: {})
    monkeypatch.setattr(quiz_generator, "resolve_provider_api_key", lambda *_a, **_k: ("k", {}))
    return quiz_generator._call_quiz_generation_llm


@pytest.mark.asyncio
async def test_quiz_generation_surface(quiz_llm, stub_llm_registry, spy):
    result = await quiz_llm(prompt="make a quiz", api_provider=PROVIDER, model="claude-test")

    assert [surface for surface, _ in spy] == ["study.quizzes"]
    assert result["choices"][0]["message"]["content"] == "provider reply"


@pytest.mark.asyncio
async def test_quiz_generation_input_block(quiz_llm, stub_llm_registry, moderation_hooks):
    with pytest.raises(LLMPolicyRefusal) as exc:
        await quiz_llm(prompt="forbidden words", api_provider=PROVIDER, model="claude-test")

    assert exc.value.detail == "Input violates moderation policy"
    assert stub_llm_registry.requests == []


@pytest.mark.asyncio
async def test_quiz_generation_content_filter_is_refused_and_not_retried(quiz_llm, stub_llm_registry, moderation_hooks):
    """The reply is replaced by moderation: it is never parsed as quiz JSON, and the provider is called once."""
    stub_llm_registry.reply = reply("a forbidden reply")

    with pytest.raises(LLMPolicyRefusal) as exc:
        await quiz_llm(prompt="make a quiz", api_provider=PROVIDER, model="claude-test")

    assert exc.value.detail == "Output violates moderation policy"
    assert len(stub_llm_registry.requests) == 1


@pytest.mark.asyncio
async def test_osce_generation_does_not_turn_a_refusal_into_a_provider_error(quiz_llm, stub_llm_registry, moderation_hooks, monkeypatch):
    """OSCE wraps every provider failure as OsceProviderError (a 502); a moderation refusal keeps its 400."""
    from tldw_Server_API.app.services import quiz_generator
    from tldw_Server_API.app.services.osce_generator import generate_osce_stations_from_sources

    async def refused(**_kwargs):
        return await quiz_llm(prompt="forbidden words", api_provider=PROVIDER, model="claude-test")

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", refused)  # a patched call opts out of test mode
    evidence = [{"source_type": "note", "source_id": "n1", "chunk_id": "c1", "text": "Warfarin needs INR checks."}]

    with pytest.raises(LLMPolicyRefusal):
        await generate_osce_stations_from_sources(
            evidence=evidence,
            normalized_sources=[{"source_type": "note", "source_id": "n1"}],
            num_stations=1,
            difficulty="mixed",
            focus_topics=[],
            model="claude-test",
            api_provider=PROVIDER,
            verification_provider=None,
            verification_model=None,
        )


@pytest.mark.asyncio
async def test_mcp_quizzes_module_surface_and_refusal(stub_llm_registry, moderation_hooks, spy):
    from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.quizzes_module import QuizzesModule

    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))
    assert await module._call_llm("make a quiz", provider=PROVIDER, model="claude-test") == "provider reply"
    assert [surface for surface, _ in spy] == ["study.quizzes"]

    with pytest.raises(LLMPolicyRefusal):
        await module._call_llm("forbidden words", provider=PROVIDER, model="claude-test")
    stub_llm_registry.reply = reply("a forbidden reply")
    with pytest.raises(LLMPolicyRefusal):
        await module._call_llm("make a quiz", provider=PROVIDER, model="claude-test")


def test_quiz_generator_no_longer_calls_an_adapter_directly():
    """R6: the direct adapter call moved to the gateway, so conftest's model-validation stub is what replaced it."""
    from tldw_Server_API.app.services import quiz_generator

    assert not hasattr(quiz_generator, "_get_adapter")
    assert quiz_generator.perform_chat_api_call is gateway.perform_chat_api_call
