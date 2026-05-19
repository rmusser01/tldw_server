import asyncio
import json
import threading
import time
from types import SimpleNamespace
from typing import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


pytestmark = pytest.mark.critical


def _set_user_db_base(monkeypatch: pytest.MonkeyPatch, base_dir) -> str | None:
    from tldw_Server_API.app.core.config import settings

    previous = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    return previous


def _restore_user_db_base(previous: str | None) -> None:
    from tldw_Server_API.app.core.config import settings

    if previous is not None:
        settings.USER_DB_BASE_DIR = previous
        return
    try:
        del settings.USER_DB_BASE_DIR
    except AttributeError:
        pass


def _seed_chat_thread(*, owner_user_id: str, chat_db_path, chat_id: str | None = None) -> str:
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    chat_db = CharactersRAGDB(str(chat_db_path), client_id=owner_user_id)
    character_id = chat_db.add_character_card(
        {
            "name": DEFAULT_CHARACTER_NAME,
            "description": "Research thread helper",
            "personality": "Helpful",
            "scenario": "Research follow-up testing",
            "system_prompt": "You are a research assistant.",
            "first_message": "Hello",
            "creator_notes": "Created by deep research tests",
        }
    )
    chat_id = chat_db.add_conversation(
        {
            "id": chat_id,
            "character_id": character_id,
            "title": "Deep Research Chat",
            "client_id": owner_user_id,
        }
    )
    if chat_id is None:
        pytest.fail("Failed to seed deep research chat handoff conversation")
    return chat_id


def _parse_sse_block(block: str) -> dict[str, object] | None:
    event_name: str | None = None
    event_id: int | None = None
    data_lines: list[str] = []
    for line in block.splitlines():
        if line.startswith("id:"):
            raw_event_id = line.partition(":")[2].strip()
            if raw_event_id.isdigit():
                event_id = int(raw_event_id)
        elif line.startswith("event: "):
            event_name = line.removeprefix("event: ").strip()
        elif line.startswith("data: "):
            data_lines.append(line.removeprefix("data: "))
    if event_name is None:
        return None
    payload = "\n".join(data_lines)
    return {
        "event": event_name,
        "id": event_id,
        "data": json.loads(payload) if payload else None,
    }


def _collect_sse_events(
    app: FastAPI,
    url: str,
    sink: list[dict[str, object]],
    snapshot_seen: threading.Event,
    *,
    stop_when: Callable[[dict[str, object], list[dict[str, object]]], bool] | None = None,
) -> None:
    with TestClient(app) as client:
        with client.stream("GET", url) as response:
            sink.append({"event": "__meta__", "data": {"status_code": response.status_code}})
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk.replace("\r\n", "\n")
                if not snapshot_seen.is_set() and "event: snapshot" in buffer:
                    snapshot_seen.set()
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    if not block.strip():
                        continue
                    parsed = _parse_sse_block(block)
                    if parsed is None:
                        continue
                    sink.append(parsed)
                    if parsed["event"] == "snapshot":
                        snapshot_seen.set()
                    if stop_when is not None and stop_when(parsed, sink):
                        return


def _build_stub_synthesizer(first_source: dict[str, object], first_note: dict[str, object]):
    from tldw_Server_API.app.core.Research.synthesizer import ResearchSynthesizer

    class StubSynthesisProvider:
        async def summarize(self, **kwargs):
            assert kwargs["config"]["provider"] == "openai"
            assert kwargs["config"]["model"] == "gpt-4.1-mini"
            return {
                "outline_sections": [
                    {
                        "title": "Background",
                        "focus_area": first_source["focus_area"],
                        "source_ids": [first_source["source_id"]],
                        "note_ids": [first_note["note_id"]],
                    }
                ],
                "claims": [
                    {
                        "text": "Supported claim",
                        "focus_area": first_source["focus_area"],
                        "source_ids": [first_source["source_id"]],
                        "citations": [{"source_id": first_source["source_id"]}],
                        "confidence": 0.81,
                    }
                ],
                "report_sections": [
                    {
                        "title": "Background",
                        "markdown": "Evidence-backed section text.",
                    }
                ],
                "unresolved_questions": [],
                "summary": {"mode": "llm_backed"},
            }

    return ResearchSynthesizer(synthesis_provider=StubSynthesisProvider())


def test_deep_research_run_can_be_approved_and_exported(tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import research_runs
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.File_Artifacts.adapter_registry import FileAdapterRegistry
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.service import ResearchService
    from tldw_Server_API.app.core.Research.synthesizer import ResearchSynthesizer

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 11, "uuid": "job-11", "status": "queued", **kwargs}

    research_db_path = tmp_path / "research.db"
    outputs_dir = tmp_path / "outputs"
    service = ResearchService(
        research_db_path=research_db_path,
        outputs_dir=outputs_dir,
        job_manager=DummyJobs(),
    )

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[research_runs.get_research_service] = lambda: service

    with TestClient(app) as client:
        create_resp = client.post(
            "/api/v1/research/runs",
            json={
                "query": "Test deep research run",
                "provider_overrides": {
                    "local": {"top_k": 4, "sources": ["media_db"]},
                    "web": {"engine": "kagi", "result_count": 3},
                    "academic": {"providers": ["arxiv", "pubmed"], "max_results": 2},
                    "synthesis": {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.2},
                },
            },
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["id"]

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 11,
                "payload": {
                    "session_id": session_id,
                    "phase": "drafting_plan",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    db = ResearchSessionsDB(research_db_path)
    session = db.get_session(session_id)
    assert session is not None
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        provider_config_resp = client.get(f"/api/v1/research/runs/{session_id}/artifacts/provider_config.json")
        approve_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={"patch_payload": {"focus_areas": ["background", "counterevidence"]}},
        )
        assert provider_config_resp.status_code == 200
        assert provider_config_resp.json()["content"]["web"]["engine"] == "kagi"
        assert provider_config_resp.json()["content"]["academic"]["providers"] == ["arxiv", "pubmed"]
        assert approve_resp.status_code == 200
        assert approve_resp.json()["phase"] == "collecting"

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 12,
                "payload": {
                    "session_id": session_id,
                    "phase": "collecting",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    session = db.get_session(session_id)
    assert session is not None
    assert session.phase == "awaiting_source_review"
    assert session.latest_checkpoint_id is not None
    assert (outputs_dir / "research" / session_id / "source_registry.json").exists()
    assert (outputs_dir / "research" / session_id / "evidence_notes.jsonl").exists()
    assert (outputs_dir / "research" / session_id / "collection_summary.json").exists()
    source_registry = ResearchArtifactStore(base_dir=outputs_dir, db=db).read_json(
        session_id=session_id,
        artifact_name="source_registry.json",
    )
    assert source_registry is not None
    assert {"local_corpus", "arxiv", "pubmed", "kagi"} <= {
        item["provider"] for item in source_registry["sources"]
    }

    with TestClient(app) as client:
        approve_source_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={},
        )
        assert approve_source_resp.status_code == 200
        assert approve_source_resp.json()["phase"] == "synthesizing"

    store = ResearchArtifactStore(base_dir=outputs_dir, db=db)
    source_registry = store.read_json(session_id=session_id, artifact_name="source_registry.json")
    evidence_notes = store.read_jsonl(session_id=session_id, artifact_name="evidence_notes.jsonl")
    assert source_registry is not None
    assert evidence_notes is not None
    first_source = source_registry["sources"][0]
    first_note = evidence_notes[0]

    class StubSynthesisProvider:
        async def summarize(self, **kwargs):
            assert kwargs["config"]["provider"] == "openai"
            assert kwargs["config"]["model"] == "gpt-4.1-mini"
            return {
                "outline_sections": [
                    {
                        "title": "Background",
                        "focus_area": first_source["focus_area"],
                        "source_ids": [first_source["source_id"]],
                        "note_ids": [first_note["note_id"]],
                    }
                ],
                "claims": [
                    {
                        "text": "Supported claim",
                        "focus_area": first_source["focus_area"],
                        "source_ids": [first_source["source_id"]],
                        "citations": [{"source_id": first_source["source_id"]}],
                        "confidence": 0.81,
                    }
                ],
                "report_sections": [
                    {
                        "title": "Background",
                        "markdown": "Evidence-backed section text.",
                    }
                ],
                "unresolved_questions": [],
                "summary": {"mode": "llm_backed"},
            }

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 13,
                "payload": {
                    "session_id": session_id,
                    "phase": "synthesizing",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
            synthesizer=ResearchSynthesizer(synthesis_provider=StubSynthesisProvider()),
        )
    )

    session = db.get_session(session_id)
    assert session is not None
    assert session.phase == "awaiting_outline_review"
    assert session.latest_checkpoint_id is not None
    assert (outputs_dir / "research" / session_id / "outline_v1.json").exists()
    assert (outputs_dir / "research" / session_id / "claims.json").exists()
    assert (outputs_dir / "research" / session_id / "report_v1.md").exists()
    assert (outputs_dir / "research" / session_id / "synthesis_summary.json").exists()
    synthesis_summary = store.read_json(session_id=session_id, artifact_name="synthesis_summary.json")
    assert synthesis_summary is not None
    assert synthesis_summary["mode"] == "llm_backed"

    with TestClient(app) as client:
        approve_outline_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={},
        )
        assert approve_outline_resp.status_code == 200
        assert approve_outline_resp.json()["phase"] == "packaging"

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 14,
                "payload": {
                    "session_id": session_id,
                    "phase": "packaging",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    session = db.get_session(session_id)
    assert session is not None
    assert session.phase == "completed"
    assert session.status == "completed"
    assert session.completed_at is not None

    package = store.read_json(session_id=session_id, artifact_name="bundle.json")
    assert package is not None
    assert package["verification_summary"]["supported_claim_count"] == 1
    assert package["verification_summary"]["unsupported_claim_count"] == 0
    assert package["unsupported_claims"] == []
    assert package["contradictions"] == []
    matching_source_trust = next(
        item for item in package["source_trust"] if item["source_id"] == first_source["source_id"]
    )
    assert matching_source_trust["snapshot_policy"] in {"full_artifact", "metadata_excerpt"}

    with TestClient(app) as client:
        run_resp = client.get(f"/api/v1/research/runs/{session_id}")
        bundle_resp = client.get(f"/api/v1/research/runs/{session_id}/bundle")
        artifact_resp = client.get(f"/api/v1/research/runs/{session_id}/artifacts/report_v1.md")

        assert run_resp.status_code == 200
        assert run_resp.json()["phase"] == "completed"
        assert run_resp.json()["completed_at"] is not None
        assert bundle_resp.status_code == 200
        assert bundle_resp.json()["question"] == "Test deep research run"
        assert bundle_resp.json()["verification_summary"]["supported_claim_count"] == 1
        assert bundle_resp.json()["unsupported_claims"] == []
        assert bundle_resp.json()["contradictions"] == []
        assert artifact_resp.status_code == 200
        assert artifact_resp.json()["artifact_name"] == "report_v1.md"
        assert artifact_resp.json()["content_type"] == "text/markdown"
        assert artifact_resp.json()["content"].startswith("# Research Report")

    adapter = FileAdapterRegistry().get_adapter("research_package")
    assert adapter is not None
    export = adapter.export(package, format="md")
    assert export.status == "ready"
    assert export.content.startswith(b"# Research Report")
    assert (outputs_dir / "research" / session_id / "bundle.json").exists()


def test_deep_research_run_creation_persists_chat_handoff(tmp_path, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import research_runs
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 21, "uuid": "job-21", "status": "queued", **kwargs}

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_dbs")
    research_db_path = tmp_path / "research.db"
    outputs_dir = tmp_path / "outputs"
    service = ResearchService(
        research_db_path=research_db_path,
        outputs_dir=outputs_dir,
        job_manager=DummyJobs(),
    )

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[research_runs.get_research_service] = lambda: service

    try:
        _seed_chat_thread(
            owner_user_id="1",
            chat_db_path=DatabasePaths.get_chacha_db_path("1"),
            chat_id="chat_123",
        )

        with TestClient(app) as client:
            create_resp = client.post(
                "/api/v1/research/runs",
                json={
                    "query": "Trace the policy timeline",
                    "chat_handoff": {
                        "chat_id": "chat_123",
                        "launch_message_id": "msg_456",
                    },
                },
            )
    finally:
        _restore_user_db_base(previous_user_db_base)

    assert create_resp.status_code == 200
    session_id = create_resp.json()["id"]
    db = ResearchSessionsDB(research_db_path)
    handoff = db.get_chat_handoff(session_id)

    assert handoff is not None
    assert handoff.session_id == session_id
    assert handoff.owner_user_id == "1"
    assert handoff.chat_id == "chat_123"
    assert handoff.launch_message_id == "msg_456"
    assert handoff.handoff_status == "pending"


def test_deep_research_run_supports_recollection_loop_and_outline_resynthesis(tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import research_runs
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.models import (
        ResearchCollectionResult,
        ResearchEvidenceNote,
        ResearchSourceRecord,
    )
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def __init__(self):
            self.created_jobs: list[dict[str, object]] = []

        def create_job(self, **kwargs):
            job = {"id": len(self.created_jobs) + 21, "uuid": f"job-{len(self.created_jobs) + 21}", "status": "queued", **kwargs}
            self.created_jobs.append(job)
            return job

    class InitialBroker:
        async def collect_focus_area(self, **kwargs):
            focus_area = kwargs["focus_area"]
            if focus_area == "background":
                return ResearchCollectionResult(
                    sources=[
                        ResearchSourceRecord(
                            source_id="src_keep",
                            focus_area="background",
                            source_type="local_document",
                            provider="local_corpus",
                            title="Pinned source",
                            url=None,
                            snippet="Pinned source snippet",
                            published_at=None,
                            retrieved_at="2026-03-07T00:00:00+00:00",
                            fingerprint="fp_keep",
                            trust_tier="internal",
                            metadata={},
                        )
                    ],
                    evidence_notes=[
                        ResearchEvidenceNote(
                            note_id="note_keep",
                            source_id="src_keep",
                            focus_area="background",
                            kind="summary",
                            text="Pinned evidence remains useful.",
                            citation_locator=None,
                            confidence=0.8,
                            metadata={},
                        )
                    ],
                    collection_metrics={"lane_counts": {"local": 1, "academic": 0, "web": 0}, "deduped_sources": 0},
                    remaining_gaps=[],
                )
            return ResearchCollectionResult(
                sources=[
                    ResearchSourceRecord(
                        source_id="src_drop",
                        focus_area="counterevidence",
                        source_type="web_result",
                        provider="kagi",
                        title="Dropped source",
                        url="https://example.com/drop",
                        snippet="Dropped source snippet",
                        published_at=None,
                        retrieved_at="2026-03-07T00:00:00+00:00",
                        fingerprint="fp_drop",
                        trust_tier="medium",
                        metadata={},
                    )
                ],
                evidence_notes=[
                    ResearchEvidenceNote(
                        note_id="note_drop",
                        source_id="src_drop",
                        focus_area="counterevidence",
                        kind="summary",
                        text="This source should be dropped before synthesis.",
                        citation_locator=None,
                        confidence=0.4,
                        metadata={},
                    )
                ],
                collection_metrics={"lane_counts": {"local": 0, "academic": 0, "web": 1}, "deduped_sources": 0},
                remaining_gaps=[],
            )

    class RecollectBroker:
        async def collect_focus_area(self, **kwargs):
            focus_area = kwargs["focus_area"]
            if focus_area != "background":
                return ResearchCollectionResult(
                    sources=[],
                    evidence_notes=[],
                    collection_metrics={"lane_counts": {"local": 0, "academic": 0, "web": 0}, "deduped_sources": 0},
                    remaining_gaps=["missing evidence for focus area: counterevidence"],
                )
            return ResearchCollectionResult(
                sources=[
                    ResearchSourceRecord(
                        source_id="src_new_background",
                        focus_area=focus_area,
                        source_type="academic_paper",
                        provider="arxiv",
                        title="New corroborating source",
                        url="https://arxiv.org/abs/1234.5678",
                        snippet="New source snippet",
                        published_at=None,
                        retrieved_at="2026-03-07T00:00:00+00:00",
                        fingerprint="fp_new_background",
                        trust_tier="high",
                        metadata={},
                    )
                ],
                evidence_notes=[
                    ResearchEvidenceNote(
                        note_id="note_new_background",
                        source_id="src_new_background",
                        focus_area=focus_area,
                        kind="summary",
                        text=f"New recollected evidence for {focus_area}.",
                        citation_locator=None,
                        confidence=0.9,
                        metadata={},
                    )
                ],
                collection_metrics={"lane_counts": {"local": 0, "academic": 1, "web": 0}, "deduped_sources": 0},
                remaining_gaps=[],
            )

    jobs = DummyJobs()
    research_db_path = tmp_path / "research.db"
    outputs_dir = tmp_path / "outputs"
    service = ResearchService(
        research_db_path=research_db_path,
        outputs_dir=outputs_dir,
        job_manager=jobs,
    )

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[research_runs.get_research_service] = lambda: service

    with TestClient(app) as client:
        create_resp = client.post("/api/v1/research/runs", json={"query": "Edited checkpoint run"})
        assert create_resp.status_code == 200
        session_id = create_resp.json()["id"]

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 21,
                "payload": {
                    "session_id": session_id,
                    "phase": "drafting_plan",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    db = ResearchSessionsDB(research_db_path)
    store = ResearchArtifactStore(base_dir=outputs_dir, db=db)
    session = db.get_session(session_id)
    assert session is not None
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        approve_plan_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={"patch_payload": {"focus_areas": ["background", "counterevidence"]}},
        )
        assert approve_plan_resp.status_code == 200
        assert approve_plan_resp.json()["phase"] == "collecting"

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 22,
                "payload": {
                    "session_id": session_id,
                    "phase": "collecting",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
            broker=InitialBroker(),
        )
    )

    session = db.get_session(session_id)
    assert session is not None
    assert session.phase == "awaiting_source_review"
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        recollect_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={
                "patch_payload": {
                    "pinned_source_ids": ["src_keep"],
                    "dropped_source_ids": ["src_drop"],
                    "prioritized_source_ids": ["src_keep"],
                    "recollect": {
                        "enabled": True,
                        "need_primary_sources": True,
                        "need_contradictions": True,
                        "guidance": "Find better contradictory primary sources.",
                    },
                }
            },
        )
        assert recollect_resp.status_code == 200
        assert recollect_resp.json()["phase"] == "collecting"

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 23,
                "payload": {
                    "session_id": session_id,
                    "phase": "collecting",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
            broker=RecollectBroker(),
        )
    )

    session = db.get_session(session_id)
    source_registry = store.read_json(session_id=session_id, artifact_name="source_registry.json")
    assert session is not None
    assert session.phase == "awaiting_source_review"
    assert session.latest_checkpoint_id is not None
    assert source_registry is not None
    assert [item["source_id"] for item in source_registry["sources"]] == ["src_keep", "src_new_background"]

    with TestClient(app) as client:
        approve_sources_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={},
        )
        assert approve_sources_resp.status_code == 200
        assert approve_sources_resp.json()["phase"] == "synthesizing"

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 24,
                "payload": {
                    "session_id": session_id,
                    "phase": "synthesizing",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    session = db.get_session(session_id)
    assert session is not None
    assert session.phase == "awaiting_outline_review"
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        approve_outline_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={
                "patch_payload": {
                    "sections": [
                        {"title": "Counterevidence First", "focus_area": "counterevidence"},
                        {"title": "Background Context", "focus_area": "background"},
                    ]
                }
            },
        )
        assert approve_outline_resp.status_code == 200
        assert approve_outline_resp.json()["phase"] == "synthesizing"

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 25,
                "payload": {
                    "session_id": session_id,
                    "phase": "synthesizing",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                    "approved_outline_locked": True,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    session = db.get_session(session_id)
    outline = store.read_json(session_id=session_id, artifact_name="outline_v1.json")
    report_markdown = store.read_text(session_id=session_id, artifact_name="report_v1.md")
    synthesis_summary = store.read_json(session_id=session_id, artifact_name="synthesis_summary.json")

    assert session is not None
    assert session.phase == "packaging"
    assert outline is not None
    assert [section["title"] for section in outline["sections"]] == [
        "Counterevidence First",
        "Background Context",
    ]
    assert report_markdown is not None
    assert "## Counterevidence First" in report_markdown
    assert synthesis_summary is not None
    assert "missing evidence for focus area: counterevidence" in synthesis_summary["unresolved_questions"]


def test_deep_research_run_controls_support_pause_resume_cancel_and_progress_polling(tmp_path):
    from tldw_Server_API.app.api.v1.endpoints import research_runs
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def __init__(self):
            self.next_id = 50
            self.cancelled: list[tuple[int, str | None]] = []

        def create_job(self, **kwargs):
            self.next_id += 1
            return {"id": self.next_id, "uuid": f"job-{self.next_id}", "status": "queued", **kwargs}

        def cancel_job(self, job_id: int, *, reason: str | None = None):
            self.cancelled.append((job_id, reason))
            return True

    jobs = DummyJobs()
    research_db_path = tmp_path / "research.db"
    outputs_dir = tmp_path / "outputs"
    service = ResearchService(
        research_db_path=research_db_path,
        outputs_dir=outputs_dir,
        job_manager=jobs,
    )

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[research_runs.get_research_service] = lambda: service

    with TestClient(app) as client:
        create_resp = client.post(
            "/api/v1/research/runs",
            json={"query": "Pause and cancel deep research", "autonomy_mode": "checkpointed"},
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["id"]

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 51,
                "payload": {
                    "session_id": session_id,
                    "phase": "drafting_plan",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    db = ResearchSessionsDB(research_db_path)
    session = db.get_session(session_id)
    assert session is not None
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        poll_resp = client.get(f"/api/v1/research/runs/{session_id}")
        pause_resp = client.post(f"/api/v1/research/runs/{session_id}/pause")
        paused_approve_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={},
        )
        resume_resp = client.post(f"/api/v1/research/runs/{session_id}/resume")
        approve_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={},
        )
        cancel_resp = client.post(f"/api/v1/research/runs/{session_id}/cancel")
        cancelled_poll_resp = client.get(f"/api/v1/research/runs/{session_id}")
        cancelled_resume_resp = client.post(f"/api/v1/research/runs/{session_id}/resume")

        assert poll_resp.status_code == 200
        assert poll_resp.json()["progress_percent"] == 10.0
        assert poll_resp.json()["progress_message"] == "planning research"
        assert pause_resp.status_code == 200
        assert pause_resp.json()["control_state"] == "paused"
        assert paused_approve_resp.status_code == 400
        assert resume_resp.status_code == 200
        assert resume_resp.json()["status"] == "waiting_human"
        assert resume_resp.json()["control_state"] == "running"
        assert approve_resp.status_code == 200
        assert approve_resp.json()["phase"] == "collecting"
        assert approve_resp.json()["active_job_id"] == "52"
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["control_state"] == "cancel_requested"
        assert cancel_resp.json()["active_job_id"] == "52"
        assert cancelled_poll_resp.status_code == 200
        assert cancelled_poll_resp.json()["control_state"] == "cancel_requested"
        assert cancelled_resume_resp.status_code == 400

    assert jobs.cancelled == [(52, "research_cancel_requested")]


def test_deep_research_live_progress_stream_reports_checkpoint_and_terminal_events(tmp_path, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import research_runs
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 11, "uuid": "job-11", "status": "queued", **kwargs}

    monkeypatch.setenv("RESEARCH_RUNS_SSE_POLL_INTERVAL", "0.05")
    monkeypatch.setenv("RESEARCH_RUNS_SSE_TEST_MAX_SECONDS", "5.0")

    research_db_path = tmp_path / "research.db"
    outputs_dir = tmp_path / "outputs"
    service = ResearchService(
        research_db_path=research_db_path,
        outputs_dir=outputs_dir,
        job_manager=DummyJobs(),
    )

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[research_runs.get_research_service] = lambda: service

    with TestClient(app) as client:
        create_resp = client.post(
            "/api/v1/research/runs",
            json={
                "query": "Stream deep research run",
                "provider_overrides": {
                    "local": {"top_k": 4, "sources": ["media_db"]},
                    "web": {"engine": "kagi", "result_count": 3},
                    "academic": {"providers": ["arxiv", "pubmed"], "max_results": 2},
                    "synthesis": {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.2},
                },
            },
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["id"]

    events: list[dict[str, object]] = []
    snapshot_seen = threading.Event()
    stream_thread = threading.Thread(
        target=_collect_sse_events,
        args=(app, f"/api/v1/research/runs/{session_id}/events/stream", events, snapshot_seen),
        daemon=True,
    )
    stream_thread.start()
    time.sleep(0.2)

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 11,
                "payload": {
                    "session_id": session_id,
                    "phase": "drafting_plan",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )
    time.sleep(0.15)

    db = ResearchSessionsDB(research_db_path)
    session = db.get_session(session_id)
    assert session is not None
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        approve_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={"patch_payload": {"focus_areas": ["background", "counterevidence"]}},
        )
        assert approve_resp.status_code == 200

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 12,
                "payload": {
                    "session_id": session_id,
                    "phase": "collecting",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )
    time.sleep(0.15)

    session = db.get_session(session_id)
    assert session is not None
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        approve_source_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={},
        )
        assert approve_source_resp.status_code == 200

    store = ResearchArtifactStore(base_dir=outputs_dir, db=db)
    source_registry = store.read_json(session_id=session_id, artifact_name="source_registry.json")
    evidence_notes = store.read_jsonl(session_id=session_id, artifact_name="evidence_notes.jsonl")
    assert source_registry is not None
    assert evidence_notes is not None

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 13,
                "payload": {
                    "session_id": session_id,
                    "phase": "synthesizing",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
            synthesizer=_build_stub_synthesizer(source_registry["sources"][0], evidence_notes[0]),
        )
    )
    time.sleep(0.15)

    session = db.get_session(session_id)
    assert session is not None
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        approve_outline_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={},
        )
        assert approve_outline_resp.status_code == 200

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 14,
                "payload": {
                    "session_id": session_id,
                    "phase": "packaging",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    stream_thread.join(timeout=5.0)
    assert not stream_thread.is_alive()

    meta = next(item for item in events if item["event"] == "__meta__")
    assert meta["data"]["status_code"] == 200

    event_names = [item["event"] for item in events if item["event"] != "__meta__"]
    assert event_names[0] == "snapshot"
    assert "status" in event_names
    assert "checkpoint" in event_names
    assert event_names[-1] == "terminal"

    checkpoint_events = [item for item in events if item["event"] == "checkpoint"]
    assert any(item["data"]["checkpoint_type"] == "plan_review" for item in checkpoint_events)

    terminal_event = next(item for item in reversed(events) if item["event"] == "terminal")
    assert terminal_event["data"]["status"] == "completed"


def test_deep_research_live_progress_stream_reconnect_replays_only_missed_events(tmp_path, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import research_runs
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 21, "uuid": "job-21", "status": "queued", **kwargs}

    monkeypatch.setenv("RESEARCH_RUNS_SSE_POLL_INTERVAL", "0.05")
    monkeypatch.setenv("RESEARCH_RUNS_SSE_TEST_MAX_SECONDS", "5.0")

    research_db_path = tmp_path / "research.db"
    outputs_dir = tmp_path / "outputs"
    service = ResearchService(
        research_db_path=research_db_path,
        outputs_dir=outputs_dir,
        job_manager=DummyJobs(),
    )

    app = FastAPI()
    app.include_router(research_runs.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[research_runs.get_research_service] = lambda: service

    with TestClient(app) as client:
        create_resp = client.post(
            "/api/v1/research/runs",
            json={
                "query": "Replay deep research events",
                "provider_overrides": {
                    "local": {"top_k": 4, "sources": ["media_db"]},
                    "web": {"engine": "kagi", "result_count": 3},
                    "academic": {"providers": ["arxiv", "pubmed"], "max_results": 2},
                    "synthesis": {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.2},
                },
            },
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["id"]

    initial_events: list[dict[str, object]] = []
    snapshot_seen = threading.Event()
    first_stream = threading.Thread(
        target=_collect_sse_events,
        args=(app, f"/api/v1/research/runs/{session_id}/events/stream", initial_events, snapshot_seen),
        kwargs={
            "stop_when": lambda event, _sink: (
                event["event"] == "checkpoint"
                and isinstance(event["data"], dict)
                and event["data"].get("checkpoint_type") == "plan_review"
            )
        },
        daemon=True,
    )
    first_stream.start()
    time.sleep(0.2)

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 21,
                "payload": {
                    "session_id": session_id,
                    "phase": "drafting_plan",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    first_stream.join(timeout=5.0)
    assert not first_stream.is_alive()

    first_replayable_ids = [
        event["id"] for event in initial_events if isinstance(event.get("id"), int)
    ]
    assert first_replayable_ids
    last_seen = max(first_replayable_ids)

    db = ResearchSessionsDB(research_db_path)
    session = db.get_session(session_id)
    assert session is not None
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        approve_plan_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={"patch_payload": {"focus_areas": ["background", "counterevidence"]}},
        )
        assert approve_plan_resp.status_code == 200

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 22,
                "payload": {
                    "session_id": session_id,
                    "phase": "collecting",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    session = db.get_session(session_id)
    assert session is not None
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        approve_source_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={},
        )
        assert approve_source_resp.status_code == 200

    store = ResearchArtifactStore(base_dir=outputs_dir, db=db)
    source_registry = store.read_json(session_id=session_id, artifact_name="source_registry.json")
    evidence_notes = store.read_jsonl(session_id=session_id, artifact_name="evidence_notes.jsonl")
    assert source_registry is not None
    assert evidence_notes is not None

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 23,
                "payload": {
                    "session_id": session_id,
                    "phase": "synthesizing",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
            synthesizer=_build_stub_synthesizer(source_registry["sources"][0], evidence_notes[0]),
        )
    )

    session = db.get_session(session_id)
    assert session is not None
    assert session.latest_checkpoint_id is not None

    with TestClient(app) as client:
        approve_outline_resp = client.post(
            f"/api/v1/research/runs/{session_id}/checkpoints/{session.latest_checkpoint_id}/patch-and-approve",
            json={},
        )
        assert approve_outline_resp.status_code == 200

    asyncio.run(
        handle_research_phase_job(
            {
                "id": 24,
                "payload": {
                    "session_id": session_id,
                    "phase": "packaging",
                    "checkpoint_id": session.latest_checkpoint_id,
                    "policy_version": 1,
                },
            },
            research_db_path=research_db_path,
            outputs_dir=outputs_dir,
        )
    )

    reconnect_events: list[dict[str, object]] = []
    reconnect_snapshot_seen = threading.Event()
    replay_stream = threading.Thread(
        target=_collect_sse_events,
        args=(
            app,
            f"/api/v1/research/runs/{session_id}/events/stream?after_id={last_seen}",
            reconnect_events,
            reconnect_snapshot_seen,
        ),
        daemon=True,
    )
    replay_stream.start()
    replay_stream.join(timeout=12.0)
    assert not replay_stream.is_alive()

    meta = next(item for item in reconnect_events if item["event"] == "__meta__")
    assert meta["data"]["status_code"] == 200

    replay_payload_events = [
        event for event in reconnect_events if event["event"] not in {"__meta__", "snapshot"}
    ]
    assert replay_payload_events
    assert all(isinstance(event["id"], int) and event["id"] > last_seen for event in replay_payload_events)
    assert all(event["data"]["event_id"] > last_seen for event in replay_payload_events)
    assert all(event["data"]["replayed"] is True for event in replay_payload_events)
    assert reconnect_events[1]["event"] == "snapshot"
    assert replay_payload_events[-1]["event"] == "terminal"
