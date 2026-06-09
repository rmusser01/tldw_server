import json
import os
import time
import types
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_wf(tmp_path, monkeypatch, auth_headers):
     # Force test mode for adapters that check it
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_SQLITE_POOL_SIZE", "4")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    # Provide a temporary USER_DB_BASE_DIR for embedding/chroma
    base = tmp_path / "user_databases"
    base.mkdir(parents=True, exist_ok=True)
    from tldw_Server_API.app.core import config as _cfg
    _cfg.settings["USER_DB_BASE_DIR"] = str(base)
    # Chroma stub client
    monkeypatch.setenv("CHROMADB_FORCE_STUB", "1")

    db = WorkflowsDatabase(str(tmp_path / "wf.db"))

    async def override_user():
        return User(
            id=1,
            username="tester",
            email="t@e.com",
            is_active=True,
            is_admin=True,
            tenant_id="default",
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ],
        )

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=1,
            username="tester",
            email="t@e.com",
            roles=["admin"],
            permissions=[WORKFLOWS_RUNS_READ],
        )

    def override_db():

        return db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[wf_mod._get_db] = override_db

    try:
        with TestClient(app, headers=auth_headers) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        db.close()


def _run_data_from_overridden_db(client: TestClient, run_id: str) -> dict | None:
    override_db = client.app.dependency_overrides.get(wf_mod._get_db)
    if not callable(override_db):
        return None
    run = override_db().get_run(run_id)
    if run is None:
        return None
    outputs = json.loads(run.outputs_json or "null") if run.outputs_json else None
    return {
        "id": run.run_id,
        "run_id": run.run_id,
        "status": run.status,
        "outputs": outputs,
        "error": run.error,
    }


def _wait_terminal(client: TestClient, run_id: str, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/workflows/runs/{run_id}")
        if r.status_code == 404:
            data = _run_data_from_overridden_db(client, run_id)
            if data and data["status"] in ("succeeded", "failed", "cancelled"):
                return data
            time.sleep(0.05)
            continue
        r.raise_for_status()
        data = r.json()
        if data["status"] in ("succeeded", "failed", "cancelled"):
            return data
        time.sleep(0.05)
    raise AssertionError("run did not complete")


def test_rss_fetch_step_test_mode(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "rss",
        "version": 1,
        "steps": [
            {"id": "a", "type": "rss_fetch", "config": {"urls": ["https://example.com/feed.xml"], "limit": 3}}
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert isinstance(out.get("results"), list)
    assert out.get("count") == 1


def test_embed_step_with_stub(monkeypatch, client_with_wf: TestClient):
    client = client_with_wf
    # Monkeypatch embeddings to avoid heavy deps
    import tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create as EC
    async def _fake_batch(texts, user_app_config, model_id_override=None):
        return [[0.1, 0.2, 0.3] for _ in texts]
    monkeypatch.setattr(EC, "create_embeddings_batch_async", _fake_batch)

    definition = {
        "name": "embedder",
        "version": 1,
        "steps": [
            {"id": "p", "type": "prompt", "config": {"template": "hello world"}},
            {"id": "e", "type": "embed", "config": {"texts": "{{ last.text }}", "collection": "user_1_workflows"}}
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("upserted") == 1


def test_translate_step_simulated(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "translate",
        "version": 1,
        "steps": [
            {"id": "p", "type": "prompt", "config": {"template": "Bonjour"}},
            {"id": "t", "type": "translate", "config": {"target_lang": "en"}}
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("target_lang") == "en"
    # In TEST_MODE, returns original text
    assert out.get("text")


def test_notify_step_test_mode(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "notify",
        "version": 1,
        "steps": [
            {"id": "n", "type": "notify", "config": {"url": "https://hooks.slack.com/services/test", "message": "{{ inputs.m }}"}}
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"m": "Hello"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    out = (data.get("outputs") or {})
    assert out.get("test_mode") is True


def test_llm_step_test_mode(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "llm",
        "version": 1,
        "steps": [
            {"id": "l1", "type": "llm", "config": {"provider": "openai", "prompt": "Hello {{ inputs.name }}"}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"name": "Rui"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "Rui" in (out.get("text") or "")


def test_kanban_step_crud(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "kanban-crud",
        "version": 1,
        "steps": [
            {"id": "b", "type": "kanban", "config": {"action": "board.create", "name": "Board {{ inputs.name }}", "client_id": "wf-board-1"}},
            {"id": "l", "type": "kanban", "config": {"action": "list.create", "board_id": "{{ last.board.id }}", "name": "To Do", "client_id": "wf-list-1"}},
            {"id": "c", "type": "kanban", "config": {"action": "card.create", "list_id": "{{ last.list.id }}", "title": "Card {{ inputs.name }}", "client_id": "wf-card-1"}},
            {"id": "g", "type": "kanban", "config": {"action": "card.get", "card_id": "{{ last.card.id }}", "include_details": True}},
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"name": "Kanban"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    card = out.get("card") or {}
    assert card.get("title") == "Card Kanban"
    assert isinstance(card.get("checklists"), list)


def test_diff_change_detector(client_with_wf: TestClient):
    client = client_with_wf
    definition = {
        "name": "diff",
        "version": 1,
        "steps": [
            {"id": "p", "type": "prompt", "config": {"template": "hello"}},
            {"id": "d", "type": "diff_change_detector", "config": {"current": "hello world", "method": "ratio", "threshold": 0.99}}
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("changed") is True


def test_stt_transcribe_with_mock(monkeypatch, tmp_path, client_with_wf: TestClient):
    client = client_with_wf
    # Create a dummy wav file path (we won't actually read it since we mock)
    fake_wav = tmp_path / "fake.wav"
    fake_wav.write_bytes(b"RIFF\x00\x00\x00WAVEfmt ")

    # Patch speech_to_text to avoid heavy deps
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as ATL
    def _fake_stt(
        path,
        whisper_model="large-v3",
        selected_source_lang=None,
        vad_filter=False,
        diarize=False,
        *,
        word_timestamps=False,
        return_language=False,
        hotwords=None,
        **kwargs,
    ):
        # Workflow adapter should pass None when language is omitted,
        # allowing the STT backend to auto-detect.
        assert selected_source_lang is None
        segments = [{"Text": "hello world", "start_seconds": 0.0, "end_seconds": 1.0}]
        return (segments, 'en') if return_language else segments
    monkeypatch.setattr(ATL, "speech_to_text", _fake_stt)

    definition = {
        "name": "stt",
        "version": 1,
        "steps": [
            {"id": "s", "type": "stt_transcribe", "config": {"file_uri": f"file://{fake_wav}", "model": "large-v3", "word_timestamps": False}}
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "hello world" in out.get("text", "")


# ---------------------------------------------------------------------------
# Stage 1 Adapters: Notes, Prompts, Chunking
# ---------------------------------------------------------------------------

def test_notes_step_create_test_mode(client_with_wf: TestClient):
    """Test notes adapter create action in test mode."""
    client = client_with_wf
    definition = {
        "name": "notes-create",
        "version": 1,
        "steps": [
            {
                "id": "n1",
                "type": "notes",
                "config": {
                    "action": "create",
                    "title": "Note from {{ inputs.user }}",
                    "content": "This is the content from workflow"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"user": "Alice"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("success") is True
    note = out.get("note") or {}
    assert "Alice" in note.get("title", "")


def test_notes_step_list_test_mode(client_with_wf: TestClient):
    """Test notes adapter list action in test mode."""
    client = client_with_wf
    definition = {
        "name": "notes-list",
        "version": 1,
        "steps": [
            {
                "id": "n1",
                "type": "notes",
                "config": {"action": "list", "limit": 10}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("notes"), list)


def test_notes_step_search_test_mode(client_with_wf: TestClient):
    """Test notes adapter search action in test mode."""
    client = client_with_wf
    definition = {
        "name": "notes-search",
        "version": 1,
        "steps": [
            {
                "id": "n1",
                "type": "notes",
                "config": {"action": "search", "query": "{{ inputs.q }}"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"q": "test query"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True


def test_prompts_step_create_test_mode(client_with_wf: TestClient):
    """Test prompts adapter create action in test mode."""
    client = client_with_wf
    definition = {
        "name": "prompts-create",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "prompts",
                "config": {
                    "action": "create",
                    "name": "My Prompt {{ inputs.version }}",
                    "prompt": "You are a helpful assistant.",
                    "keywords": ["test", "workflow"]
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"version": "v1"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("success") is True


def test_prompts_step_list_test_mode(client_with_wf: TestClient):
    """Test prompts adapter list action in test mode."""
    client = client_with_wf
    definition = {
        "name": "prompts-list",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "prompts",
                "config": {"action": "list", "limit": 5}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("prompts"), list)


def test_prompts_step_search_test_mode(client_with_wf: TestClient):
    """Test prompts adapter search action in test mode."""
    client = client_with_wf
    definition = {
        "name": "prompts-search",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "prompts",
                "config": {"action": "search", "query": "{{ inputs.q }}"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"q": "assistant"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True


def test_chunking_step_test_mode(client_with_wf: TestClient):
    """Test chunking adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "chunk-text",
        "version": 1,
        "steps": [
            {"id": "p", "type": "prompt", "config": {"template": "This is a long text that should be chunked into smaller pieces for processing. It has multiple sentences."}},
            {
                "id": "c",
                "type": "chunking",
                "config": {
                    "method": "sentences",
                    "max_size": 50,
                    "overlap": 0
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("chunks"), list)
    assert out.get("count") > 0
    assert out.get("method") == "sentences"


def test_chunking_step_with_explicit_text(client_with_wf: TestClient):
    """Test chunking adapter with explicit text input."""
    client = client_with_wf
    definition = {
        "name": "chunk-explicit",
        "version": 1,
        "steps": [
            {
                "id": "c",
                "type": "chunking",
                "config": {
                    "text": "{{ inputs.content }}",
                    "method": "words",
                    "max_size": 100
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"content": "One two three four five six seven eight nine ten eleven twelve"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("count") >= 1


def test_chunking_step_empty_text(client_with_wf: TestClient):
    """Test chunking adapter with empty text returns empty chunks."""
    client = client_with_wf
    definition = {
        "name": "chunk-empty",
        "version": 1,
        "steps": [
            {
                "id": "c",
                "type": "chunking",
                "config": {"text": "", "method": "sentences"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("chunks") == []
    assert out.get("count") == 0


def test_chunking_step_invalid_method(client_with_wf: TestClient):
    """Test chunking adapter with invalid method returns error."""
    client = client_with_wf
    definition = {
        "name": "chunk-invalid",
        "version": 1,
        "steps": [
            {
                "id": "c",
                "type": "chunking",
                "config": {"text": "Some text", "method": "invalid_method"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"  # Step completes but with error output
    out = data.get("outputs") or {}
    assert "invalid_method" in (out.get("error") or "")


def test_notes_and_chunking_chained(client_with_wf: TestClient):
    """Test notes create followed by chunking the content."""
    client = client_with_wf
    definition = {
        "name": "notes-chunking-chain",
        "version": 1,
        "steps": [
            {
                "id": "n1",
                "type": "notes",
                "config": {
                    "action": "create",
                    "title": "My Note",
                    "content": "This is a long note with multiple sentences. It should be chunked."
                }
            },
            {
                "id": "c1",
                "type": "chunking",
                "config": {
                    "text": "{{ last.note.content }}",
                    "method": "sentences",
                    "max_size": 100
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("chunks"), list)


# ---------------------------------------------------------------------------
# Stage 2 Adapters: Web Search, Collections, Chatbooks
# ---------------------------------------------------------------------------

def test_web_search_step_test_mode(client_with_wf: TestClient):
    """Test web_search adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "web-search",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "web_search",
                "config": {
                    "query": "{{ inputs.q }}",
                    "engine": "google",
                    "num_results": 5
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"q": "python tutorials"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("results"), list)
    assert out.get("count") >= 1
    assert "python tutorials" in out.get("query", "")


def test_web_search_step_missing_query(client_with_wf: TestClient):
    """Test web_search adapter returns error when query is missing."""
    client = client_with_wf
    definition = {
        "name": "web-search-no-query",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "web_search",
                "config": {"engine": "bing"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_query" in (out.get("error") or "")


def test_web_search_step_invalid_engine(client_with_wf: TestClient):
    """Test web_search adapter returns error for invalid engine."""
    client = client_with_wf
    definition = {
        "name": "web-search-bad-engine",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "web_search",
                "config": {"query": "test", "engine": "invalid_engine"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "invalid_engine" in (out.get("error") or "")


def test_collections_step_save_test_mode(client_with_wf: TestClient):
    """Test collections adapter save action in test mode."""
    client = client_with_wf
    definition = {
        "name": "collections-save",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "collections",
                "config": {
                    "action": "save",
                    "url": "https://example.com/article/{{ inputs.id }}",
                    "tags": ["workflow", "test"]
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"id": "123"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("created") is True
    assert "example.com" in (out.get("item") or {}).get("url", "")


def test_collections_step_list_test_mode(client_with_wf: TestClient):
    """Test collections adapter list action in test mode."""
    client = client_with_wf
    definition = {
        "name": "collections-list",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "collections",
                "config": {"action": "list", "limit": 10}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("items"), list)


def test_collections_step_search_test_mode(client_with_wf: TestClient):
    """Test collections adapter search action in test mode."""
    client = client_with_wf
    definition = {
        "name": "collections-search",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "collections",
                "config": {"action": "search", "query": "{{ inputs.q }}"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"q": "python"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True


def test_chatbooks_step_export_test_mode(client_with_wf: TestClient):
    """Test chatbooks adapter export action in test mode."""
    client = client_with_wf
    definition = {
        "name": "chatbooks-export",
        "version": 1,
        "steps": [
            {
                "id": "cb1",
                "type": "chatbooks",
                "config": {
                    "action": "export",
                    "name": "Workflow Export {{ inputs.suffix }}",
                    "content_types": ["conversations", "notes"]
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"suffix": "2024"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("job_id") is not None


def test_chatbooks_step_list_jobs_test_mode(client_with_wf: TestClient):
    """Test chatbooks adapter list_jobs action in test mode."""
    client = client_with_wf
    definition = {
        "name": "chatbooks-list-jobs",
        "version": 1,
        "steps": [
            {
                "id": "cb1",
                "type": "chatbooks",
                "config": {"action": "list_jobs"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("jobs"), list)


def test_chatbooks_step_preview_test_mode(client_with_wf: TestClient):
    """Test chatbooks adapter preview action in test mode."""
    client = client_with_wf
    definition = {
        "name": "chatbooks-preview",
        "version": 1,
        "steps": [
            {
                "id": "cb1",
                "type": "chatbooks",
                "config": {
                    "action": "preview",
                    "content_types": ["conversations", "notes", "prompts"]
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "preview" in out


def test_web_search_and_chunking_chain(client_with_wf: TestClient):
    """Test web search followed by chunking the results."""
    client = client_with_wf
    definition = {
        "name": "search-chunk-chain",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "web_search",
                "config": {"query": "machine learning", "num_results": 3}
            },
            {
                "id": "c1",
                "type": "chunking",
                "config": {
                    "text": "{{ last.text }}",
                    "method": "sentences",
                    "max_size": 100
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("chunks"), list)


# ---------------------------------------------------------------------------
# Stage 3 Adapters: Evaluations, Claims Extract, Character Chat
# ---------------------------------------------------------------------------

def test_evaluations_step_geval_test_mode(client_with_wf: TestClient):
    """Test evaluations adapter geval action in test mode."""
    client = client_with_wf
    definition = {
        "name": "eval-geval",
        "version": 1,
        "steps": [
            {
                "id": "e1",
                "type": "evaluations",
                "config": {
                    "action": "geval",
                    "response": "{{ inputs.summary }}",
                    "context": "{{ inputs.source }}",
                    "threshold": 0.6
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"summary": "This is a test summary.", "source": "This is the original text."}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "score" in out
    assert out.get("passed") is True


def test_evaluations_step_rag_test_mode(client_with_wf: TestClient):
    """Test evaluations adapter rag action in test mode."""
    client = client_with_wf
    definition = {
        "name": "eval-rag",
        "version": 1,
        "steps": [
            {
                "id": "e1",
                "type": "evaluations",
                "config": {
                    "action": "rag",
                    "question": "{{ inputs.question }}",
                    "response": "{{ inputs.answer }}",
                    "retrieved_contexts": ["Context 1", "Context 2"]
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"question": "What is RAG?", "answer": "RAG is Retrieval-Augmented Generation."}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "score" in out
    assert "metrics" in out


def test_evaluations_step_response_quality_test_mode(client_with_wf: TestClient):
    """Test evaluations adapter response_quality action in test mode."""
    client = client_with_wf
    definition = {
        "name": "eval-quality",
        "version": 1,
        "steps": [
            {
                "id": "e1",
                "type": "evaluations",
                "config": {
                    "action": "response_quality",
                    "prompt": "Write a haiku about coding",
                    "response": "Code flows like water\nBugs emerge from hidden depths\nDebug brings new light"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "score" in out


def test_evaluations_step_list_runs_test_mode(client_with_wf: TestClient):
    """Test evaluations adapter list_runs action in test mode."""
    client = client_with_wf
    definition = {
        "name": "eval-list-runs",
        "version": 1,
        "steps": [
            {
                "id": "e1",
                "type": "evaluations",
                "config": {"action": "list_runs", "limit": 10}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("runs"), list)


def test_claims_extract_step_extract_test_mode(client_with_wf: TestClient):
    """Test claims_extract adapter extract action in test mode."""
    client = client_with_wf
    definition = {
        "name": "claims-extract",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "claims_extract",
                "config": {
                    "action": "extract",
                    "text": "{{ inputs.text }}"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"text": "The Earth is round. Water is essential for life."}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("claims"), list)
    assert out.get("count") >= 1


def test_claims_extract_step_search_test_mode(client_with_wf: TestClient):
    """Test claims_extract adapter search action in test mode."""
    client = client_with_wf
    definition = {
        "name": "claims-search",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "claims_extract",
                "config": {
                    "action": "search",
                    "query": "{{ inputs.q }}"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"q": "climate"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True


def test_claims_extract_step_list_test_mode(client_with_wf: TestClient):
    """Test claims_extract adapter list action in test mode."""
    client = client_with_wf
    definition = {
        "name": "claims-list",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "claims_extract",
                "config": {"action": "list", "limit": 10}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("claims"), list)


def test_character_chat_step_start_test_mode(client_with_wf: TestClient):
    """Test character_chat adapter start action in test mode."""
    client = client_with_wf
    definition = {
        "name": "char-chat-start",
        "version": 1,
        "steps": [
            {
                "id": "cc1",
                "type": "character_chat",
                "config": {
                    "action": "start",
                    "character_id": 1
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "conversation_id" in out
    assert "character_name" in out


def test_character_chat_step_message_test_mode(client_with_wf: TestClient):
    """Test character_chat adapter message action in test mode."""
    client = client_with_wf
    definition = {
        "name": "char-chat-message",
        "version": 1,
        "steps": [
            {
                "id": "cc1",
                "type": "character_chat",
                "config": {
                    "action": "message",
                    "conversation_id": "test-conv-123",
                    "message": "{{ inputs.msg }}"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"msg": "Hello, how are you?"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "response" in out
    assert "turn_count" in out


def test_character_chat_step_load_test_mode(client_with_wf: TestClient):
    """Test character_chat adapter load action in test mode."""
    client = client_with_wf
    definition = {
        "name": "char-chat-load",
        "version": 1,
        "steps": [
            {
                "id": "cc1",
                "type": "character_chat",
                "config": {
                    "action": "load",
                    "conversation_id": "test-conv-456"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "conversation_id" in out
    assert "history" in out


def test_evaluations_missing_action(client_with_wf: TestClient):
    """Test evaluations adapter returns error when action is missing."""
    client = client_with_wf
    definition = {
        "name": "eval-no-action",
        "version": 1,
        "steps": [
            {
                "id": "e1",
                "type": "evaluations",
                "config": {"response": "test"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_action" in (out.get("error") or "")


def test_claims_extract_missing_action(client_with_wf: TestClient):
    """Test claims_extract adapter returns error when action is missing."""
    client = client_with_wf
    definition = {
        "name": "claims-no-action",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "claims_extract",
                "config": {"text": "test"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_action" in (out.get("error") or "")


# ---------------------------------------------------------------------------
# Stage 4 Adapters: Moderation, Sandbox Exec, Image Generation
# ---------------------------------------------------------------------------

def test_moderation_step_check_allowed_test_mode(client_with_wf: TestClient):
    """Test moderation adapter check action returns allowed for safe text."""
    client = client_with_wf
    definition = {
        "name": "mod-check-allowed",
        "version": 1,
        "steps": [
            {
                "id": "m1",
                "type": "moderation",
                "config": {
                    "action": "check",
                    "text": "{{ inputs.text }}",
                    "action_type": "generic"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"text": "This is a completely safe and normal message."}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("allowed") is True
    assert out.get("reason") == "passed"


def test_moderation_step_check_blocked_test_mode(client_with_wf: TestClient):
    """Test moderation adapter check action detects blocked content."""
    client = client_with_wf
    definition = {
        "name": "mod-check-blocked",
        "version": 1,
        "steps": [
            {
                "id": "m1",
                "type": "moderation",
                "config": {
                    "action": "check",
                    "text": "This message contains blocked content"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("allowed") is False
    assert "blocked" in (out.get("reason") or "").lower()


def test_moderation_step_redact_test_mode(client_with_wf: TestClient):
    """Test moderation adapter redact action removes sensitive content."""
    client = client_with_wf
    definition = {
        "name": "mod-redact",
        "version": 1,
        "steps": [
            {
                "id": "m1",
                "type": "moderation",
                "config": {
                    "action": "redact",
                    "text": "My password is secret123"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "[REDACTED]" in out.get("redacted_text", "")
    assert "password" not in out.get("redacted_text", "").lower()
    assert out.get("text") is not None  # Alias for chaining


def test_moderation_step_redact_custom_patterns_test_mode(client_with_wf: TestClient):
    """Test moderation adapter redact action with custom patterns."""
    client = client_with_wf
    definition = {
        "name": "mod-redact-custom",
        "version": 1,
        "steps": [
            {
                "id": "m1",
                "type": "moderation",
                "config": {
                    "action": "redact",
                    "text": "Contact john@example.com or call 555-1234",
                    "patterns": [
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
                        r"\b\d{3}-\d{4}\b"  # phone pattern
                    ]
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    redacted = out.get("redacted_text", "")
    # Custom patterns should have redacted email and phone
    assert "john@example.com" not in redacted
    assert "555-1234" not in redacted
    assert "[REDACTED]" in redacted


def test_moderation_step_missing_text(client_with_wf: TestClient):
    """Test moderation adapter returns error when text is missing."""
    client = client_with_wf
    definition = {
        "name": "mod-no-text",
        "version": 1,
        "steps": [
            {
                "id": "m1",
                "type": "moderation",
                "config": {"action": "check"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_text" in (out.get("error") or "")


def test_sandbox_exec_step_python_test_mode(client_with_wf: TestClient):
    """Test sandbox_exec adapter executes Python code in test mode."""
    client = client_with_wf
    definition = {
        "name": "sandbox-python",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "sandbox_exec",
                "config": {
                    "code": "print('Hello from sandbox!')",
                    "language": "python",
                    "timeout_seconds": 10
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("exit_code") == 0
    assert out.get("timed_out") is False
    assert "TEST_MODE" in out.get("stdout", "")
    assert out.get("language") == "python"


def test_sandbox_exec_step_bash_test_mode(client_with_wf: TestClient):
    """Test sandbox_exec adapter executes bash code in test mode."""
    client = client_with_wf
    definition = {
        "name": "sandbox-bash",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "sandbox_exec",
                "config": {
                    "code": "echo 'Hello from bash'",
                    "language": "bash",
                    "timeout_seconds": 5
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("exit_code") == 0
    assert out.get("language") == "bash"


def test_sandbox_exec_step_with_templating(client_with_wf: TestClient):
    """Test sandbox_exec adapter with templated code."""
    client = client_with_wf
    definition = {
        "name": "sandbox-template",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "sandbox_exec",
                "config": {
                    "code": "print('Processing {{ inputs.data }}')",
                    "language": "python"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"data": "test_value"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True


def test_sandbox_exec_step_missing_code(client_with_wf: TestClient):
    """Test sandbox_exec adapter returns error when code is missing."""
    client = client_with_wf
    definition = {
        "name": "sandbox-no-code",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "sandbox_exec",
                "config": {"language": "python"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_code" in (out.get("error") or "")


def test_sandbox_exec_step_unsupported_language(client_with_wf: TestClient):
    """Test sandbox_exec adapter returns error for unsupported language."""
    client = client_with_wf
    definition = {
        "name": "sandbox-bad-lang",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "sandbox_exec",
                "config": {"code": "puts 'hi'", "language": "ruby"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "unsupported_language" in (out.get("error") or "")


def test_sandbox_exec_step_javascript_test_mode(client_with_wf: TestClient):
    """Test sandbox_exec adapter executes JavaScript code in test mode."""
    client = client_with_wf
    definition = {
        "name": "sandbox-js",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "sandbox_exec",
                "config": {
                    "code": "console.log('Hello from Node.js')",
                    "language": "javascript",
                    "timeout_seconds": 10
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("exit_code") == 0
    assert out.get("language") == "javascript"


def test_sandbox_exec_step_node_alias_test_mode(client_with_wf: TestClient):
    """Test sandbox_exec adapter accepts 'node' as alias for javascript."""
    client = client_with_wf
    definition = {
        "name": "sandbox-node",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "sandbox_exec",
                "config": {
                    "code": "console.log('test')",
                    "language": "node"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("language") == "javascript"


def test_sandbox_exec_step_js_alias_test_mode(client_with_wf: TestClient):
    """Test sandbox_exec adapter accepts 'js' as alias for javascript."""
    client = client_with_wf
    definition = {
        "name": "sandbox-js-alias",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "sandbox_exec",
                "config": {
                    "code": "console.log('test')",
                    "language": "js"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("language") == "javascript"


def test_image_gen_step_basic_test_mode(client_with_wf: TestClient):
    """Test image_gen adapter generates an image in test mode."""
    client = client_with_wf
    definition = {
        "name": "image-gen-basic",
        "version": 1,
        "steps": [
            {
                "id": "i1",
                "type": "image_gen",
                "config": {
                    "prompt": "{{ inputs.prompt }}",
                    "width": 512,
                    "height": 512,
                    "steps": 20
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"prompt": "A beautiful sunset over mountains"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("count") == 1
    assert isinstance(out.get("images"), list)
    assert len(out["images"]) == 1
    assert "uri" in out["images"][0]
    assert out["images"][0]["width"] == 512
    assert out["images"][0]["height"] == 512


def test_image_gen_step_with_params_test_mode(client_with_wf: TestClient):
    """Test image_gen adapter with custom parameters."""
    client = client_with_wf
    definition = {
        "name": "image-gen-params",
        "version": 1,
        "steps": [
            {
                "id": "i1",
                "type": "image_gen",
                "config": {
                    "prompt": "A cat wearing a hat",
                    "negative_prompt": "blurry, low quality",
                    "backend": "stable_diffusion_cpp",
                    "width": 768,
                    "height": 512,
                    "steps": 30,
                    "cfg_scale": 8.5,
                    "seed": 42,
                    "format": "png"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("backend") == "stable_diffusion_cpp"
    assert out["images"][0]["format"] == "png"


def test_image_gen_step_missing_prompt(client_with_wf: TestClient):
    """Test image_gen adapter returns error when prompt is missing."""
    client = client_with_wf
    definition = {
        "name": "image-gen-no-prompt",
        "version": 1,
        "steps": [
            {
                "id": "i1",
                "type": "image_gen",
                "config": {"width": 512, "height": 512}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_prompt" in (out.get("error") or "")


def test_image_gen_from_llm_output_chain(client_with_wf: TestClient):
    """Test image_gen adapter using prompt from previous LLM step."""
    client = client_with_wf
    definition = {
        "name": "llm-to-image",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "prompt",
                "config": {"template": "A majestic eagle soaring through clouds"}
            },
            {
                "id": "i1",
                "type": "image_gen",
                "config": {
                    "prompt": "{{ last.text }}",
                    "width": 512,
                    "height": 512
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "eagle" in (out.get("prompt") or "").lower()


def test_moderation_and_image_gen_chain(client_with_wf: TestClient):
    """Test moderation check followed by conditional image generation."""
    client = client_with_wf
    definition = {
        "name": "mod-then-image",
        "version": 1,
        "steps": [
            {
                "id": "m1",
                "type": "moderation",
                "config": {
                    "action": "check",
                    "text": "A beautiful landscape with trees"
                }
            },
            {
                "id": "i1",
                "type": "image_gen",
                "config": {
                    "prompt": "A beautiful landscape with trees",
                    "width": 512,
                    "height": 512
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("images"), list)


# ---------------------------------------------------------------------------
# Stage 5 Adapters: Summarize, Query Expand, Rerank, Citations
# ---------------------------------------------------------------------------

def test_summarize_step_basic_test_mode(client_with_wf: TestClient):
    """Test summarize adapter with basic text in test mode."""
    client = client_with_wf
    definition = {
        "name": "summarize-basic",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "summarize",
                "config": {
                    "text": "{{ inputs.text }}",
                    "api_name": "openai"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"text": "This is a long document that needs to be summarized. It contains many important points about machine learning and artificial intelligence."}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "summary" in out
    assert "text" in out  # Alias for chaining
    assert out.get("api_name") == "openai"


def test_summarize_step_with_custom_prompt_test_mode(client_with_wf: TestClient):
    """Test summarize adapter with custom prompt in test mode."""
    client = client_with_wf
    definition = {
        "name": "summarize-custom",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "summarize",
                "config": {
                    "text": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
                    "custom_prompt": "Summarize in one sentence.",
                    "temperature": 0.5
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("input_length") > 0
    assert out.get("output_length") > 0


def test_summarize_step_from_last_text(client_with_wf: TestClient):
    """Test summarize adapter getting text from previous step."""
    client = client_with_wf
    definition = {
        "name": "summarize-chain",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "prompt",
                "config": {"template": "This is the text to summarize from a prompt step."}
            },
            {
                "id": "s1",
                "type": "summarize",
                "config": {"api_name": "anthropic"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "summary" in out


def test_summarize_step_missing_text(client_with_wf: TestClient):
    """Test summarize adapter returns error when text is missing."""
    client = client_with_wf
    definition = {
        "name": "summarize-no-text",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "summarize",
                "config": {"api_name": "openai"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_text" in (out.get("error") or "")


def test_query_expand_step_synonym_test_mode(client_with_wf: TestClient):
    """Test query_expand adapter with synonym strategy in test mode."""
    client = client_with_wf
    definition = {
        "name": "query-expand-syn",
        "version": 1,
        "steps": [
            {
                "id": "q1",
                "type": "query_expand",
                "config": {
                    "query": "{{ inputs.query }}",
                    "strategies": ["synonym"],
                    "max_expansions": 3
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"query": "machine learning algorithms"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("original") == "machine learning algorithms"
    assert isinstance(out.get("variations"), list)
    assert isinstance(out.get("keywords"), list)
    assert "combined" in out


def test_query_expand_step_multi_strategy_test_mode(client_with_wf: TestClient):
    """Test query_expand adapter with multiple strategies in test mode."""
    client = client_with_wf
    definition = {
        "name": "query-expand-multi",
        "version": 1,
        "steps": [
            {
                "id": "q1",
                "type": "query_expand",
                "config": {
                    "query": "RAG pipeline implementation",
                    "strategies": ["acronym", "synonym", "entity"],
                    "max_expansions": 5
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "acronym" in out.get("strategies_used", [])
    assert "synonym" in out.get("strategies_used", [])


def test_query_expand_step_missing_query(client_with_wf: TestClient):
    """Test query_expand adapter returns error when query is missing."""
    client = client_with_wf
    definition = {
        "name": "query-expand-no-query",
        "version": 1,
        "steps": [
            {
                "id": "q1",
                "type": "query_expand",
                "config": {"strategies": ["synonym"]}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_query" in (out.get("error") or "")


def test_rerank_step_flashrank_test_mode(client_with_wf: TestClient):
    """Test rerank adapter with flashrank strategy in test mode."""
    client = client_with_wf
    definition = {
        "name": "rerank-flash",
        "version": 1,
        "steps": [
            {
                "id": "r1",
                "type": "rerank",
                "config": {
                    "query": "{{ inputs.query }}",
                    "documents": [
                        {"content": "First document about machine learning", "id": "doc1"},
                        {"content": "Second document about deep learning", "id": "doc2"},
                        {"content": "Third document about neural networks", "id": "doc3"}
                    ],
                    "strategy": "flashrank",
                    "top_k": 2
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"query": "neural networks"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("documents"), list)
    assert out.get("count") == 2
    assert out.get("strategy") == "flashrank"
    # Check documents have scores
    for doc in out.get("documents", []):
        assert "score" in doc
        assert "content" in doc


def test_rerank_step_top_k_test_mode(client_with_wf: TestClient):
    """Test rerank adapter respects top_k parameter."""
    client = client_with_wf
    definition = {
        "name": "rerank-topk",
        "version": 1,
        "steps": [
            {
                "id": "r1",
                "type": "rerank",
                "config": {
                    "query": "artificial intelligence",
                    "documents": [
                        {"content": "Doc 1", "id": "d1"},
                        {"content": "Doc 2", "id": "d2"},
                        {"content": "Doc 3", "id": "d3"},
                        {"content": "Doc 4", "id": "d4"},
                        {"content": "Doc 5", "id": "d5"}
                    ],
                    "strategy": "diversity",
                    "top_k": 3
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("count") == 3


def test_rerank_step_missing_query(client_with_wf: TestClient):
    """Test rerank adapter returns error when query is missing."""
    client = client_with_wf
    definition = {
        "name": "rerank-no-query",
        "version": 1,
        "steps": [
            {
                "id": "r1",
                "type": "rerank",
                "config": {
                    "documents": [{"content": "test"}],
                    "strategy": "flashrank"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_query" in (out.get("error") or "")


def test_rerank_step_missing_documents(client_with_wf: TestClient):
    """Test rerank adapter returns error when documents are missing."""
    client = client_with_wf
    definition = {
        "name": "rerank-no-docs",
        "version": 1,
        "steps": [
            {
                "id": "r1",
                "type": "rerank",
                "config": {
                    "query": "test query",
                    "strategy": "flashrank"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_documents" in (out.get("error") or "")


def test_citations_step_apa_test_mode(client_with_wf: TestClient):
    """Test citations adapter with APA style in test mode."""
    client = client_with_wf
    definition = {
        "name": "citations-apa",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "citations",
                "config": {
                    "documents": [
                        {"content": "Document content here", "id": "doc1", "metadata": {"author": "Smith, J.", "title": "Machine Learning Basics", "date": "2023"}},
                        {"content": "Another document", "id": "doc2", "metadata": {"author": "Jones, A.", "title": "Deep Learning Guide", "date": "2024"}}
                    ],
                    "style": "apa",
                    "include_inline": True
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("style") == "apa"
    assert isinstance(out.get("citations"), list)
    assert len(out["citations"]) == 2
    assert isinstance(out.get("inline_markers"), dict)
    assert isinstance(out.get("citation_map"), dict)


def test_citations_step_mla_test_mode(client_with_wf: TestClient):
    """Test citations adapter with MLA style in test mode."""
    client = client_with_wf
    definition = {
        "name": "citations-mla",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "citations",
                "config": {
                    "documents": [
                        {"content": "Doc content", "author": "Brown, T.", "title": "AI Research", "date": "2023"}
                    ],
                    "style": "mla"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("style") == "mla"
    assert out.get("count") == 1


def test_citations_step_missing_documents(client_with_wf: TestClient):
    """Test citations adapter returns error when documents are missing."""
    client = client_with_wf
    definition = {
        "name": "citations-no-docs",
        "version": 1,
        "steps": [
            {
                "id": "c1",
                "type": "citations",
                "config": {"style": "apa"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_documents" in (out.get("error") or "")


def test_query_expand_and_rerank_chain(client_with_wf: TestClient):
    """Test query expansion followed by reranking."""
    client = client_with_wf
    definition = {
        "name": "expand-rerank-chain",
        "version": 1,
        "steps": [
            {
                "id": "q1",
                "type": "query_expand",
                "config": {
                    "query": "machine learning",
                    "strategies": ["synonym"]
                }
            },
            {
                "id": "r1",
                "type": "rerank",
                "config": {
                    "query": "{{ last.combined }}",
                    "documents": [
                        {"content": "Intro to ML", "id": "d1"},
                        {"content": "Deep learning basics", "id": "d2"}
                    ],
                    "top_k": 2
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("documents"), list)


def test_rerank_and_citations_chain(client_with_wf: TestClient):
    """Test reranking followed by citation generation."""
    client = client_with_wf
    definition = {
        "name": "rerank-cite-chain",
        "version": 1,
        "steps": [
            {
                "id": "r1",
                "type": "rerank",
                "config": {
                    "query": "neural networks",
                    "documents": [
                        {"content": "NN doc 1", "id": "d1", "metadata": {"author": "A", "title": "T1", "date": "2023"}},
                        {"content": "NN doc 2", "id": "d2", "metadata": {"author": "B", "title": "T2", "date": "2024"}}
                    ],
                    "top_k": 2
                }
            },
            {
                "id": "c1",
                "type": "citations",
                "config": {
                    "style": "chicago"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("style") == "chicago"
    assert isinstance(out.get("citations"), list)


def test_summarize_and_citations_chain(client_with_wf: TestClient):
    """Test summarization followed by citation generation."""
    client = client_with_wf
    definition = {
        "name": "sum-cite-chain",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "summarize",
                "config": {
                    "text": "This is a long document about artificial intelligence and its applications in healthcare, finance, and education."
                }
            },
            {
                "id": "c1",
                "type": "citations",
                "config": {
                    "documents": [
                        {"content": "{{ last.summary }}", "id": "summary", "author": "AI Summary", "title": "Generated Summary", "date": "2024"}
                    ],
                    "style": "ieee"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("style") == "ieee"


# ---------------------------------------------------------------------------
# Stage 6 Adapters: OCR, PDF Extract, Voice Intent
# ---------------------------------------------------------------------------

def test_ocr_step_basic_test_mode(tmp_path, client_with_wf: TestClient):
    """Test ocr adapter with basic image in test mode."""
    client = client_with_wf
    # Create a dummy image file
    img_path = tmp_path / "test_image.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header

    definition = {
        "name": "ocr-basic",
        "version": 1,
        "steps": [
            {
                "id": "o1",
                "type": "ocr",
                "config": {
                    "image_uri": f"file://{img_path}",
                    "backend": "tesseract",
                    "language": "eng"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "text" in out
    assert out.get("format") == "text"
    assert isinstance(out.get("blocks"), list)
    assert isinstance(out.get("warnings"), list)


def test_ocr_step_structured_output_test_mode(tmp_path, client_with_wf: TestClient):
    """Test ocr adapter with structured output format in test mode."""
    client = client_with_wf
    # Create a dummy image file
    img_path = tmp_path / "test_document.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    definition = {
        "name": "ocr-structured",
        "version": 1,
        "steps": [
            {
                "id": "o1",
                "type": "ocr",
                "config": {
                    "image_uri": f"file://{img_path}",
                    "output_format": "markdown",
                    "backend": "auto"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("format") == "markdown"


def test_ocr_step_missing_uri(client_with_wf: TestClient):
    """Test ocr adapter returns error when image_uri is missing."""
    client = client_with_wf
    definition = {
        "name": "ocr-no-uri",
        "version": 1,
        "steps": [
            {
                "id": "o1",
                "type": "ocr",
                "config": {"backend": "tesseract"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_image_uri" in (out.get("error") or "")


def test_pdf_extract_step_basic_test_mode(tmp_path, client_with_wf: TestClient):
    """Test pdf_extract adapter with basic PDF in test mode."""
    client = client_with_wf
    # Create a dummy PDF file (just a header)
    pdf_path = tmp_path / "test_document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")

    definition = {
        "name": "pdf-basic",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "pdf_extract",
                "config": {
                    "pdf_uri": f"file://{pdf_path}",
                    "parser": "pymupdf4llm"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("status") == "Success"
    assert "content" in out
    assert "text" in out  # Alias for chaining
    assert isinstance(out.get("metadata"), dict)
    assert out.get("page_count") > 0


def test_pdf_extract_step_with_chunking_test_mode(tmp_path, client_with_wf: TestClient):
    """Test pdf_extract adapter with chunking enabled in test mode."""
    client = client_with_wf
    # Create a dummy PDF file
    pdf_path = tmp_path / "test_chunked.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test document")

    definition = {
        "name": "pdf-chunked",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "pdf_extract",
                "config": {
                    "pdf_uri": f"file://{pdf_path}",
                    "parser": "pymupdf",
                    "perform_chunking": True,
                    "chunk_method": "sentences",
                    "max_chunk_size": 200,
                    "chunk_overlap": 50
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("chunks"), list)
    assert len(out.get("chunks", [])) > 0


def test_pdf_extract_step_with_metadata_override_test_mode(tmp_path, client_with_wf: TestClient):
    """Test pdf_extract adapter with title/author overrides in test mode."""
    client = client_with_wf
    # Create a dummy PDF file
    pdf_path = tmp_path / "test_meta.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%metadata test")

    definition = {
        "name": "pdf-metadata",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "pdf_extract",
                "config": {
                    "pdf_uri": f"file://{pdf_path}",
                    "title": "Custom Title {{ inputs.version }}",
                    "author": "Test Author"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {"version": "v2"}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    metadata = out.get("metadata") or {}
    assert "Custom Title v2" in metadata.get("title", "")


def test_pdf_extract_step_missing_uri(client_with_wf: TestClient):
    """Test pdf_extract adapter returns error when pdf_uri is missing."""
    client = client_with_wf
    definition = {
        "name": "pdf-no-uri",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "pdf_extract",
                "config": {"parser": "pymupdf"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_pdf_uri" in (out.get("error") or "")


def test_voice_intent_step_search_pattern_test_mode(client_with_wf: TestClient):
    """Test voice_intent adapter with search pattern in test mode."""
    client = client_with_wf
    definition = {
        "name": "voice-search",
        "version": 1,
        "steps": [
            {
                "id": "v1",
                "type": "voice_intent",
                "config": {
                    "text": "{{ inputs.transcript }}"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"transcript": "search for machine learning tutorials"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("intent") == "search"
    assert out.get("action_type") == "mcp_tool"
    assert out.get("match_method") == "pattern"
    entities = out.get("entities") or {}
    assert "query" in entities


def test_voice_intent_step_note_pattern_test_mode(client_with_wf: TestClient):
    """Test voice_intent adapter with note pattern in test mode."""
    client = client_with_wf
    definition = {
        "name": "voice-note",
        "version": 1,
        "steps": [
            {
                "id": "v1",
                "type": "voice_intent",
                "config": {
                    "text": "take a note that meeting is at 3pm"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("intent") == "create_note"
    assert out.get("match_method") == "pattern"


def test_voice_intent_step_confirmation_yes_test_mode(client_with_wf: TestClient):
    """Test voice_intent adapter with confirmation (yes) in test mode."""
    client = client_with_wf
    definition = {
        "name": "voice-confirm-yes",
        "version": 1,
        "steps": [
            {
                "id": "v1",
                "type": "voice_intent",
                "config": {
                    "text": "yes",
                    "awaiting_confirmation": True
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("intent") == "confirmation"
    assert out.get("match_method") == "confirmation"
    assert out.get("action_config", {}).get("confirmed") is True


def test_voice_intent_step_confirmation_no_test_mode(client_with_wf: TestClient):
    """Test voice_intent adapter with confirmation (no) in test mode."""
    client = client_with_wf
    definition = {
        "name": "voice-confirm-no",
        "version": 1,
        "steps": [
            {
                "id": "v1",
                "type": "voice_intent",
                "config": {
                    "text": "cancel",
                    "awaiting_confirmation": True
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("action_config", {}).get("confirmed") is False


def test_voice_intent_step_default_chat_test_mode(client_with_wf: TestClient):
    """Test voice_intent adapter falls back to chat for unknown patterns."""
    client = client_with_wf
    definition = {
        "name": "voice-chat",
        "version": 1,
        "steps": [
            {
                "id": "v1",
                "type": "voice_intent",
                "config": {
                    "text": "what is the weather like today"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("intent") == "chat"
    assert out.get("action_type") == "llm_chat"
    assert out.get("match_method") == "default"


def test_voice_intent_step_missing_text(client_with_wf: TestClient):
    """Test voice_intent adapter returns error when text is missing."""
    client = client_with_wf
    definition = {
        "name": "voice-no-text",
        "version": 1,
        "steps": [
            {
                "id": "v1",
                "type": "voice_intent",
                "config": {"llm_enabled": False}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_text" in (out.get("error") or "")


def test_pdf_extract_and_summarize_chain(tmp_path, client_with_wf: TestClient):
    """Test pdf_extract followed by summarization."""
    client = client_with_wf
    # Create a dummy PDF file
    pdf_path = tmp_path / "test_chain.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%chain test")

    definition = {
        "name": "pdf-summarize-chain",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "pdf_extract",
                "config": {
                    "pdf_uri": f"file://{pdf_path}",
                    "perform_chunking": False
                }
            },
            {
                "id": "s1",
                "type": "summarize",
                "config": {
                    "text": "{{ last.content }}"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "summary" in out


def test_voice_intent_and_search_chain(client_with_wf: TestClient):
    """Test voice_intent followed by web_search using extracted query."""
    client = client_with_wf
    definition = {
        "name": "voice-search-chain",
        "version": 1,
        "steps": [
            {
                "id": "v1",
                "type": "voice_intent",
                "config": {
                    "text": "search for python programming"
                }
            },
            {
                "id": "s1",
                "type": "web_search",
                "config": {
                    "query": "{{ last.entities.query }}",
                    "num_results": 3
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("results"), list)


# ---------------------------------------------------------------------------
# Phase 2 Adapters: Research/Academic, Document Utilities, Timing, Scheduling
# ---------------------------------------------------------------------------

def test_arxiv_search_step_test_mode(client_with_wf: TestClient):
    """Test arxiv_search adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "arxiv-search",
        "version": 1,
        "steps": [
            {
                "id": "a1",
                "type": "arxiv_search",
                "config": {
                    "query": "{{ inputs.query }}",
                    "max_results": 5
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"query": "machine learning transformers"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("papers"), list)
    assert out.get("total_results") >= 1
    assert "machine learning" in out.get("query", "").lower()


def test_arxiv_search_step_missing_query(client_with_wf: TestClient):
    """Test arxiv_search adapter returns error when query is missing."""
    client = client_with_wf
    definition = {
        "name": "arxiv-no-query",
        "version": 1,
        "steps": [
            {
                "id": "a1",
                "type": "arxiv_search",
                "config": {"max_results": 5}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_query" in (out.get("error") or "")


def test_arxiv_download_step_test_mode(client_with_wf: TestClient):
    """Test arxiv_download adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "arxiv-download",
        "version": 1,
        "steps": [
            {
                "id": "a1",
                "type": "arxiv_download",
                "config": {
                    "arxiv_id": "2301.00001"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("downloaded") is True
    assert "pdf_path" in out


def test_arxiv_download_step_from_search(client_with_wf: TestClient):
    """Test arxiv_download using arxiv_id from previous search step."""
    client = client_with_wf
    definition = {
        "name": "arxiv-search-download-chain",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "arxiv_search",
                "config": {"query": "deep learning", "max_results": 1}
            },
            {
                "id": "d1",
                "type": "arxiv_download",
                "config": {"arxiv_id": "{{ last.papers[0].arxiv_id }}"}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("downloaded") is True


def test_pubmed_search_step_test_mode(client_with_wf: TestClient):
    """Test pubmed_search adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "pubmed-search",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "pubmed_search",
                "config": {
                    "query": "{{ inputs.query }}",
                    "max_results": 5
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"query": "cancer treatment"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("papers"), list)
    assert out.get("total_results") >= 1


def test_pubmed_search_step_missing_query(client_with_wf: TestClient):
    """Test pubmed_search adapter returns error when query is missing."""
    client = client_with_wf
    definition = {
        "name": "pubmed-no-query",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "pubmed_search",
                "config": {"max_results": 5}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_query" in (out.get("error") or "")


def test_semantic_scholar_search_step_test_mode(client_with_wf: TestClient):
    """Test semantic_scholar_search adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "semantic-search",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "semantic_scholar_search",
                "config": {
                    "query": "{{ inputs.query }}",
                    "max_results": 5
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"query": "neural networks"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("papers"), list)
    assert out.get("total_results") >= 1


def test_semantic_scholar_search_step_missing_query(client_with_wf: TestClient):
    """Test semantic_scholar_search adapter returns error when query is missing."""
    client = client_with_wf
    definition = {
        "name": "semantic-no-query",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "semantic_scholar_search",
                "config": {}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_query" in (out.get("error") or "")


def test_google_scholar_search_step_test_mode(client_with_wf: TestClient):
    """Test google_scholar_search adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "gscholar-search",
        "version": 1,
        "steps": [
            {
                "id": "g1",
                "type": "google_scholar_search",
                "config": {
                    "query": "{{ inputs.query }}",
                    "max_results": 3
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"query": "reinforcement learning"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("papers"), list)


def test_google_scholar_search_step_missing_query(client_with_wf: TestClient):
    """Test google_scholar_search adapter returns error when query is missing."""
    client = client_with_wf
    definition = {
        "name": "gscholar-no-query",
        "version": 1,
        "steps": [
            {
                "id": "g1",
                "type": "google_scholar_search",
                "config": {}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_query" in (out.get("error") or "")


def test_patent_search_step_test_mode(client_with_wf: TestClient):
    """Test patent_search adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "patent-search",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "patent_search",
                "config": {
                    "query": "{{ inputs.query }}",
                    "max_results": 5
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"query": "electric vehicle battery"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert isinstance(out.get("patents"), list)
    assert out.get("total_results") >= 1


def test_patent_search_step_missing_query(client_with_wf: TestClient):
    """Test patent_search adapter returns error when query is missing."""
    client = client_with_wf
    definition = {
        "name": "patent-no-query",
        "version": 1,
        "steps": [
            {
                "id": "p1",
                "type": "patent_search",
                "config": {}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_query" in (out.get("error") or "")


def test_doi_resolve_step_test_mode(client_with_wf: TestClient):
    """Test doi_resolve adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "doi-resolve",
        "version": 1,
        "steps": [
            {
                "id": "d1",
                "type": "doi_resolve",
                "config": {
                    "doi": "10.1000/example123"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("resolved") is True
    assert isinstance(out.get("metadata"), dict)
    assert out["metadata"].get("doi") == "10.1000/example123"


def test_doi_resolve_step_cleans_url_prefix(client_with_wf: TestClient):
    """Test doi_resolve adapter strips https://doi.org/ prefix."""
    client = client_with_wf
    definition = {
        "name": "doi-resolve-url",
        "version": 1,
        "steps": [
            {
                "id": "d1",
                "type": "doi_resolve",
                "config": {
                    "doi": "https://doi.org/10.1000/xyz"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("resolved") is True
    # Should have cleaned the doi
    assert out["metadata"].get("doi") == "10.1000/xyz"


def test_doi_resolve_step_missing_doi(client_with_wf: TestClient):
    """Test doi_resolve adapter returns error when doi is missing."""
    client = client_with_wf
    definition = {
        "name": "doi-no-doi",
        "version": 1,
        "steps": [
            {
                "id": "d1",
                "type": "doi_resolve",
                "config": {}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_doi" in (out.get("error") or "")


def test_bibtex_generate_step_basic(client_with_wf: TestClient):
    """Test bibtex_generate adapter creates valid BibTeX."""
    client = client_with_wf
    definition = {
        "name": "bibtex-basic",
        "version": 1,
        "steps": [
            {
                "id": "b1",
                "type": "bibtex_generate",
                "config": {
                    "metadata": {
                        "title": "Machine Learning Basics",
                        "authors": ["Smith, John", "Jones, Jane"],
                        "year": "2023",
                        "journal": "AI Journal"
                    },
                    "entry_type": "article"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "bibtex" in out
    assert "@article{" in out["bibtex"]
    assert "Machine Learning Basics" in out["bibtex"]
    assert "cite_key" in out
    # Check that text key exists for chaining
    assert out.get("text") == out.get("bibtex")


def test_bibtex_generate_step_auto_cite_key(client_with_wf: TestClient):
    """Test bibtex_generate adapter generates cite key from author and year."""
    client = client_with_wf
    definition = {
        "name": "bibtex-autokey",
        "version": 1,
        "steps": [
            {
                "id": "b1",
                "type": "bibtex_generate",
                "config": {
                    "metadata": {
                        "title": "Test Paper",
                        "authors": ["Einstein, Albert"],
                        "year": "1905"
                    }
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    # The adapter uses the last word of the first author name (split()[-1])
    # "Einstein, Albert" -> last word is "Albert"
    assert out.get("cite_key") == "albert1905"


def test_bibtex_generate_step_from_doi_resolve(client_with_wf: TestClient):
    """Test bibtex_generate using metadata from doi_resolve."""
    client = client_with_wf
    definition = {
        "name": "doi-to-bibtex",
        "version": 1,
        "steps": [
            {
                "id": "d1",
                "type": "doi_resolve",
                "config": {"doi": "10.1234/test"}
            },
            {
                "id": "b1",
                "type": "bibtex_generate",
                "config": {}  # Gets metadata from previous step
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "bibtex" in out


def test_email_send_step_test_mode(client_with_wf: TestClient):
    """Test email_send adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "email-send",
        "version": 1,
        "steps": [
            {
                "id": "e1",
                "type": "email_send",
                "config": {
                    "to": "test@example.com",
                    "subject": "Test Email {{ inputs.subject }}",
                    "body": "This is a test email body."
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(
        f"/api/v1/workflows/{wid}/run",
        json={"inputs": {"subject": "from Workflow"}}
    ).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert out.get("sent") is True
    assert "test@example.com" in out.get("recipients", [])


def test_email_send_step_missing_recipient(client_with_wf: TestClient):
    """Test email_send adapter returns error when recipient is missing."""
    client = client_with_wf
    definition = {
        "name": "email-no-recipient",
        "version": 1,
        "steps": [
            {
                "id": "e1",
                "type": "email_send",
                "config": {
                    "subject": "Test",
                    "body": "Test body"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_recipient" in (out.get("error") or "")


def test_email_send_step_invalid_email(client_with_wf: TestClient):
    """Test email_send adapter returns error for invalid email address."""
    client = client_with_wf
    definition = {
        "name": "email-invalid",
        "version": 1,
        "steps": [
            {
                "id": "e1",
                "type": "email_send",
                "config": {
                    "to": "invalid-email-address",
                    "subject": "Test",
                    "body": "Test body"
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "invalid_email" in (out.get("error") or "")


def test_screenshot_capture_step_test_mode(client_with_wf: TestClient):
    """Test screenshot_capture adapter in test mode."""
    client = client_with_wf
    definition = {
        "name": "screenshot",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "screenshot_capture",
                "config": {
                    "url": "https://example.com",
                    "width": 1280,
                    "height": 720
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    assert "screenshot_path" in out


def test_screenshot_capture_step_missing_url(client_with_wf: TestClient):
    """Test screenshot_capture adapter returns error when url is missing."""
    client = client_with_wf
    definition = {
        "name": "screenshot-no-url",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "screenshot_capture",
                "config": {"width": 1280}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "missing_url" in (out.get("error") or "")


def test_schedule_workflow_step_basic(client_with_wf: TestClient):
    """Test schedule_workflow adapter basic scheduling."""
    client = client_with_wf
    definition = {
        "name": "schedule-basic",
        "version": 1,
        "steps": [
            {
                "id": "s1",
                "type": "schedule_workflow",
                "config": {
                    "workflow_id": "some-workflow-id",
                    "delay_seconds": 60,
                    "inputs": {"key": "value"}
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("scheduled") is True
    assert "schedule_id" in out
    assert "run_at" in out


def test_research_search_to_bibtex_chain(client_with_wf: TestClient):
    """Test arxiv search followed by bibtex generation chain."""
    client = client_with_wf
    definition = {
        "name": "research-bibtex-chain",
        "version": 1,
        "steps": [
            {
                "id": "a1",
                "type": "arxiv_search",
                "config": {"query": "transformers", "max_results": 1}
            },
            {
                "id": "b1",
                "type": "bibtex_generate",
                "config": {
                    "metadata": {
                        "title": "{{ last.papers[0].title }}",
                        "authors": "{{ last.papers[0].authors }}",
                        "year": "2023"
                    }
                }
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert "bibtex" in out


def test_multi_source_research_chain(client_with_wf: TestClient):
    """Test multiple research sources in a workflow."""
    client = client_with_wf
    definition = {
        "name": "multi-research",
        "version": 1,
        "steps": [
            {
                "id": "a1",
                "type": "arxiv_search",
                "config": {"query": "AI", "max_results": 1}
            },
            {
                "id": "p1",
                "type": "pubmed_search",
                "config": {"query": "AI in medicine", "max_results": 1}
            },
            {
                "id": "s1",
                "type": "semantic_scholar_search",
                "config": {"query": "artificial intelligence", "max_results": 1}
            }
        ],
    }
    wid = client.post("/api/v1/workflows", json=definition).json()["id"]
    run_id = client.post(f"/api/v1/workflows/{wid}/run", json={"inputs": {}}).json()["run_id"]
    data = _wait_terminal(client, run_id)
    assert data["status"] == "succeeded"
    out = data.get("outputs") or {}
    assert out.get("simulated") is True
    # Output should have results from the last step
    assert isinstance(out.get("papers"), list)
