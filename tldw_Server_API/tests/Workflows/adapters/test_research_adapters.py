"""Comprehensive tests for research and bibliography adapters.

This module tests all 11 research adapters:
1. run_arxiv_search_adapter - Search arXiv
2. run_arxiv_download_adapter - Download arXiv papers
3. run_pubmed_search_adapter - Search PubMed
4. run_semantic_scholar_search_adapter - Search Semantic Scholar
5. run_google_scholar_search_adapter - Search Google Scholar
6. run_patent_search_adapter - Search patents
7. run_doi_resolve_adapter - Resolve DOI to metadata
8. run_reference_parse_adapter - Parse reference strings
9. run_bibtex_generate_adapter - Generate BibTeX
10. run_literature_review_adapter - Generate literature review
11. run_deep_research_adapter - Launch a deep research session
12. run_deep_research_wait_adapter - Wait for a deep research session
13. run_deep_research_load_bundle_adapter - Load bundle refs from a completed deep research session
14. run_deep_research_select_bundle_fields_adapter - Select canonical bundle fields from a completed deep research session
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _patch_module_import(monkeypatch, module_name: str, fake_module: object) -> None:
    """Patch a locally imported optional module without replacing global package state."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name:
            return fake_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _fake_httpx_module_with_error(message: str) -> object:
    class _ExplodingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            raise RuntimeError(message)

    return types.SimpleNamespace(AsyncClient=lambda *args, **kwargs: _ExplodingClient())


# =============================================================================
# Deep Research Launch Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_deep_research_adapter_launches_session_and_records_artifact(monkeypatch, tmp_path):
    """Test deep research workflow adapter launches a session and persists a JSON launch artifact."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_adapter,
    )

    captured: dict[str, object] = {}

    class _FakeSession:
        id = "research-session-1"
        status = "queued"
        phase = "drafting_plan"
        control_state = "running"

    class _FakeResearchService:
        def __init__(self, *args, **kwargs):
            pass

        def create_session(self, **kwargs):
            captured["create_session_kwargs"] = kwargs
            return _FakeSession()

    added_artifacts: list[dict[str, object]] = []

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.launch._build_research_service",
        lambda: _FakeResearchService(),
    )

    context = {
        "user_id": "42",
        "tenant_id": "default",
        "step_run_id": "wf-step-1",
        "inputs": {"topic": "federated learning"},
        "add_artifact": lambda **kwargs: added_artifacts.append(kwargs),
    }
    config = {
        "query": "{{ inputs.topic }}",
        "source_policy": "balanced",
        "autonomy_mode": "checkpointed",
        "save_artifact": True,
    }

    result = await run_deep_research_adapter(config, context)

    assert result == {
        "run_id": "research-session-1",
        "status": "queued",
        "phase": "drafting_plan",
        "control_state": "running",
        "console_url": "/research?run=research-session-1",
        "bundle_url": "/api/v1/research/runs/research-session-1/bundle",
        "query": "federated learning",
        "source_policy": "balanced",
        "autonomy_mode": "checkpointed",
    }
    assert captured["create_session_kwargs"] == {
        "owner_user_id": "42",
        "query": "federated learning",
        "source_policy": "balanced",
        "autonomy_mode": "checkpointed",
        "limits_json": None,
        "provider_overrides": None,
    }
    assert len(added_artifacts) == 1
    assert added_artifacts[0]["type"] == "deep_research_launch"
    assert added_artifacts[0]["mime_type"] == "application/json"
    artifact_uri = str(added_artifacts[0]["uri"])
    assert artifact_uri.startswith("file://")
    artifact_path = Path(artifact_uri.removeprefix("file://"))
    assert artifact_path.name == "deep_research_launch.json"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == result


@pytest.mark.asyncio
async def test_deep_research_adapter_rejects_empty_rendered_query(monkeypatch):
    """Test deep research workflow adapter rejects an empty templated query."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_adapter,
    )

    with pytest.raises(ValueError, match="query"):
        await run_deep_research_adapter(
            {"query": "{{ inputs.topic }}"},
            {"user_id": "7", "inputs": {"topic": ""}},
        )


@pytest.mark.asyncio
async def test_deep_research_wait_adapter_returns_bundle_and_records_artifact(monkeypatch):
    """Test deep research wait adapter returns terminal metadata plus bundle and writes an artifact."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_wait_adapter,
    )

    captured: dict[str, object] = {}

    class _FakeSession:
        id = "research-session-1"
        status = "completed"
        phase = "packaging"
        control_state = "running"
        completed_at = "2026-03-07T12:00:00+00:00"

    class _FakeResearchService:
        def get_session(self, **kwargs):
            captured["get_session_kwargs"] = kwargs
            return _FakeSession()

        def get_bundle(self, **kwargs):
            captured["get_bundle_kwargs"] = kwargs
            return {"concise_answer": "done"}

    added_artifacts: list[dict[str, object]] = []

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _FakeResearchService(),
    )

    result = await run_deep_research_wait_adapter(
        {
            "run_id": "{{ inputs.run_id }}",
            "include_bundle": True,
            "save_artifact": True,
        },
        {
            "user_id": "42",
            "step_run_id": "wf-step-2",
            "inputs": {"run_id": "research-session-1"},
            "add_artifact": lambda **kwargs: added_artifacts.append(kwargs),
        },
    )

    assert result == {
        "run_id": "research-session-1",
        "status": "completed",
        "phase": "packaging",
        "control_state": "running",
        "completed_at": "2026-03-07T12:00:00+00:00",
        "bundle_url": "/api/v1/research/runs/research-session-1/bundle",
        "bundle": {"concise_answer": "done"},
    }
    assert captured["get_session_kwargs"] == {
        "owner_user_id": "42",
        "session_id": "research-session-1",
    }
    assert captured["get_bundle_kwargs"] == {
        "owner_user_id": "42",
        "session_id": "research-session-1",
    }
    assert len(added_artifacts) == 1
    assert added_artifacts[0]["type"] == "deep_research_wait"
    assert added_artifacts[0]["mime_type"] == "application/json"
    artifact_uri = str(added_artifacts[0]["uri"])
    artifact_path = Path(artifact_uri.removeprefix("file://"))
    assert artifact_path.name == "deep_research_wait.json"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == result


@pytest.mark.asyncio
async def test_deep_research_wait_adapter_accepts_launch_output_object(monkeypatch):
    """Test deep research wait adapter can resolve the run from a launch-step output object."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_wait_adapter,
    )

    captured: dict[str, object] = {}

    class _FakeSession:
        id = "research-session-2"
        status = "completed"
        phase = "packaging"
        control_state = "running"
        completed_at = "2026-03-07T13:00:00+00:00"

    class _FakeResearchService:
        def get_session(self, **kwargs):
            captured["get_session_kwargs"] = kwargs
            return _FakeSession()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _FakeResearchService(),
    )

    result = await run_deep_research_wait_adapter(
        {
            "run": {
                "run_id": "research-session-2",
                "console_url": "/research?run=research-session-2",
            },
            "include_bundle": False,
            "save_artifact": False,
        },
        {"user_id": "84"},
    )

    assert result == {
        "run_id": "research-session-2",
        "status": "completed",
        "phase": "packaging",
        "control_state": "running",
        "completed_at": "2026-03-07T13:00:00+00:00",
        "bundle_url": "/api/v1/research/runs/research-session-2/bundle",
        "bundle": None,
    }
    assert captured["get_session_kwargs"] == {
        "owner_user_id": "84",
        "session_id": "research-session-2",
    }


@pytest.mark.asyncio
async def test_deep_research_wait_adapter_returns_cancelled_status_when_workflow_is_cancelled(monkeypatch):
    """Test deep research wait adapter exits promptly when the enclosing workflow is cancelled."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_wait_adapter,
    )

    result = await run_deep_research_wait_adapter(
        {"run_id": "research-session-3"},
        {"user_id": "84", "is_cancelled": lambda: True},
    )

    assert result == {"__status__": "cancelled"}


@pytest.mark.asyncio
async def test_deep_research_wait_adapter_times_out_for_nonterminal_run(monkeypatch):
    """Test deep research wait adapter times out when the run never reaches terminal state."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_wait_adapter,
    )

    class _FakeSession:
        id = "research-session-4"
        status = "queued"
        phase = "collecting"
        control_state = "running"
        completed_at = None

    class _FakeResearchService:
        def get_session(self, **kwargs):
            return _FakeSession()

    times = iter([0.0, 2.5])

    async def _fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _FakeResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._now",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._sleep",
        _fake_sleep,
    )

    with pytest.raises(TimeoutError):
        await run_deep_research_wait_adapter(
            {
                "run_id": "research-session-4",
                "poll_interval_seconds": 0.1,
                "timeout_seconds": 2,
            },
            {"user_id": "84"},
        )


@pytest.mark.asyncio
async def test_deep_research_wait_adapter_returns_waiting_human_for_research_checkpoint(monkeypatch):
    """Test deep research wait adapter yields a workflow wait payload for research checkpoints."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_wait_adapter,
    )

    class _CheckpointSession:
        id = "research-session-6"
        status = "waiting_human"
        phase = "awaiting_source_review"
        control_state = "running"
        completed_at = None
        latest_checkpoint_id = "checkpoint-1"

    class _CheckpointSnapshot:
        checkpoint = {
            "checkpoint_id": "checkpoint-1",
            "checkpoint_type": "sources_review",
        }

    class _FakeResearchService:
        def get_session(self, **kwargs):
            assert kwargs["owner_user_id"] == "42"
            assert kwargs["session_id"] == "research-session-6"
            return _CheckpointSession()

        def get_stream_snapshot(self, **kwargs):
            assert kwargs["owner_user_id"] == "42"
            assert kwargs["session_id"] == "research-session-6"
            return _CheckpointSnapshot()

    times = iter([0.0, 0.0, 2.5])

    async def _fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _FakeResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._now",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._sleep",
        _fake_sleep,
    )

    result = await run_deep_research_wait_adapter(
        {
            "run_id": "research-session-6",
            "poll_interval_seconds": 0.1,
            "timeout_seconds": 2,
            "include_bundle": False,
            "save_artifact": False,
        },
        {"user_id": "42"},
    )

    assert result == {
        "__status__": "waiting_human",
        "reason": "research_checkpoint",
        "run_id": "research-session-6",
        "research_phase": "awaiting_source_review",
        "research_control_state": "running",
        "research_checkpoint_id": "checkpoint-1",
        "research_checkpoint_type": "sources_review",
        "research_console_url": "/research?run=research-session-6",
        "active_poll_seconds": pytest.approx(0.0, rel=0.1),
    }


@pytest.mark.asyncio
async def test_deep_research_wait_adapter_reuses_active_poll_seconds_after_resume(monkeypatch):
    """Test deep research wait adapter restores active polling time from prior workflow wait output."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_wait_adapter,
    )

    class _CheckpointSession:
        id = "research-session-7"
        status = "waiting_human"
        phase = "awaiting_outline_review"
        control_state = "running"
        completed_at = None
        latest_checkpoint_id = "checkpoint-2"

    class _CheckpointSnapshot:
        checkpoint = {
            "checkpoint_id": "checkpoint-2",
            "checkpoint_type": "outline_review",
        }

    class _FakeResearchService:
        def get_session(self, **kwargs):
            return _CheckpointSession()

        def get_stream_snapshot(self, **kwargs):
            return _CheckpointSnapshot()

    times = iter([0.0, 0.0, 2.5])

    async def _fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._build_research_service",
        lambda: _FakeResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._now",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.wait._sleep",
        _fake_sleep,
    )

    result = await run_deep_research_wait_adapter(
        {
            "run_id": "research-session-7",
            "poll_interval_seconds": 0.1,
            "timeout_seconds": 2,
            "include_bundle": False,
            "save_artifact": False,
        },
        {
            "user_id": "42",
            "prev": {"active_poll_seconds": 1.5},
        },
    )

    assert result["__status__"] == "waiting_human"
    assert result["research_checkpoint_type"] == "outline_review"
    assert result["active_poll_seconds"] == pytest.approx(1.5, rel=0.1)


@pytest.mark.asyncio
async def test_deep_research_load_bundle_adapter_returns_bundle_refs_and_records_artifact(monkeypatch):
    """Test deep research load-bundle adapter returns compact refs and persists a JSON artifact."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_load_bundle_adapter,
    )

    captured: dict[str, object] = {}

    class _FakeSession:
        id = "research-session-5"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-07T14:00:00+00:00"

    class _FakeSnapshot:
        artifacts = [
            {
                "artifact_name": "bundle.json",
                "artifact_version": 1,
                "content_type": "application/json",
                "phase": "packaging",
                "job_id": "job-7",
            }
        ]

    class _FakeResearchService:
        def get_session(self, **kwargs):
            captured["get_session_kwargs"] = kwargs
            return _FakeSession()

        def get_bundle(self, **kwargs):
            captured["get_bundle_kwargs"] = kwargs
            return {
                "question": "What changed in the evidence base?",
                "outline": {
                    "sections": [
                        {"title": "Overview"},
                        {"title": "Findings"},
                    ]
                },
                "claims": [
                    {"text": "Claim A", "citations": [{"source_id": "src_1"}]},
                    {"text": "Claim B", "citations": [{"source_id": "src_2"}]},
                ],
                "source_inventory": [
                    {"source_id": "src_1", "title": "Source 1"},
                    {"source_id": "src_2", "title": "Source 2"},
                    {"source_id": "src_3", "title": "Source 3"},
                ],
                "unresolved_questions": ["Need stronger contradictory evidence"],
            }

        def get_stream_snapshot(self, **kwargs):
            captured["get_stream_snapshot_kwargs"] = kwargs
            return _FakeSnapshot()

    added_artifacts: list[dict[str, object]] = []

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.load_bundle._build_research_service",
        lambda: _FakeResearchService(),
    )

    result = await run_deep_research_load_bundle_adapter(
        {
            "run_id": "{{ inputs.run_id }}",
            "save_artifact": True,
        },
        {
            "user_id": "42",
            "step_run_id": "wf-step-3",
            "inputs": {"run_id": "research-session-5"},
            "add_artifact": lambda **kwargs: added_artifacts.append(kwargs),
        },
    )

    assert result == {
        "run_id": "research-session-5",
        "status": "completed",
        "phase": "completed",
        "control_state": "running",
        "completed_at": "2026-03-07T14:00:00+00:00",
        "bundle_url": "/api/v1/research/runs/research-session-5/bundle",
        "bundle_summary": {
            "question": "What changed in the evidence base?",
            "outline_titles": ["Overview", "Findings"],
            "claim_count": 2,
            "source_count": 3,
            "unresolved_question_count": 1,
        },
        "artifacts": [
            {
                "artifact_name": "bundle.json",
                "artifact_version": 1,
                "content_type": "application/json",
                "phase": "packaging",
                "job_id": "job-7",
            }
        ],
    }
    assert captured["get_session_kwargs"] == {
        "owner_user_id": "42",
        "session_id": "research-session-5",
    }
    assert captured["get_bundle_kwargs"] == {
        "owner_user_id": "42",
        "session_id": "research-session-5",
    }
    assert captured["get_stream_snapshot_kwargs"] == {
        "owner_user_id": "42",
        "session_id": "research-session-5",
    }
    assert len(added_artifacts) == 1
    assert added_artifacts[0]["type"] == "deep_research_bundle_ref"
    assert added_artifacts[0]["mime_type"] == "application/json"
    artifact_uri = str(added_artifacts[0]["uri"])
    artifact_path = Path(artifact_uri.removeprefix("file://"))
    assert artifact_path.name == "deep_research_bundle_ref.json"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == result


@pytest.mark.asyncio
async def test_deep_research_load_bundle_adapter_accepts_wait_output_object(monkeypatch):
    """Test deep research load-bundle adapter can resolve the run from a prior step output object."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_load_bundle_adapter,
    )

    captured: dict[str, object] = {}

    class _FakeSession:
        id = "research-session-6"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-07T15:00:00+00:00"

    class _FakeSnapshot:
        artifacts = []

    class _FakeResearchService:
        def get_session(self, **kwargs):
            captured["get_session_kwargs"] = kwargs
            return _FakeSession()

        def get_bundle(self, **kwargs):
            return {
                "question": "What changed?",
                "outline": {"sections": []},
                "claims": [],
                "source_inventory": [],
                "unresolved_questions": [],
            }

        def get_stream_snapshot(self, **kwargs):
            return _FakeSnapshot()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.load_bundle._build_research_service",
        lambda: _FakeResearchService(),
    )

    result = await run_deep_research_load_bundle_adapter(
        {
            "run": {
                "run_id": "research-session-6",
                "bundle_url": "/api/v1/research/runs/research-session-6/bundle",
            },
            "save_artifact": False,
        },
        {"user_id": "84"},
    )

    assert result["run_id"] == "research-session-6"
    assert result["bundle_summary"]["question"] == "What changed?"
    assert captured["get_session_kwargs"] == {
        "owner_user_id": "84",
        "session_id": "research-session-6",
    }


@pytest.mark.asyncio
async def test_deep_research_load_bundle_adapter_rejects_non_completed_runs(monkeypatch):
    """Test deep research load-bundle adapter rejects runs that have not completed."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_load_bundle_adapter,
    )

    class _FakeSession:
        id = "research-session-7"
        status = "synthesizing"
        phase = "synthesizing"
        control_state = "running"
        completed_at = None

    class _FakeResearchService:
        def get_session(self, **kwargs):
            return _FakeSession()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.load_bundle._build_research_service",
        lambda: _FakeResearchService(),
    )

    with pytest.raises(RuntimeError, match="completed runs only"):
        await run_deep_research_load_bundle_adapter(
            {"run_id": "research-session-7"},
            {"user_id": "84"},
        )


@pytest.mark.asyncio
async def test_deep_research_select_bundle_fields_adapter_returns_selected_fields_and_records_artifact(
    monkeypatch,
):
    """Test deep research bundle-field selector returns only requested fields and persists a JSON artifact."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_select_bundle_fields_adapter,
    )

    captured: dict[str, object] = {}

    class _FakeSession:
        id = "research-session-50"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-08T08:00:00+00:00"

    class _FakeResearchService:
        def get_session(self, **kwargs):
            captured["get_session_kwargs"] = kwargs
            return _FakeSession()

        def get_bundle(self, **kwargs):
            captured["get_bundle_kwargs"] = kwargs
            return {
                "question": "What changed?",
                "claims": [{"text": "Claim A"}],
                "verification_summary": {"supported_claim_count": 1},
            }

    added_artifacts: list[dict[str, object]] = []

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.select_bundle_fields._build_research_service",
        lambda: _FakeResearchService(),
    )

    result = await run_deep_research_select_bundle_fields_adapter(
        {
            "run_id": "{{ inputs.run_id }}",
            "fields": ["question", "claims", "unsupported_claims"],
            "save_artifact": True,
        },
        {
            "user_id": "42",
            "step_run_id": "wf-step-4",
            "inputs": {"run_id": "research-session-50"},
            "add_artifact": lambda **kwargs: added_artifacts.append(kwargs),
        },
    )

    assert result == {
        "run_id": "research-session-50",
        "status": "completed",
        "selected_fields": {
            "question": "What changed?",
            "claims": [{"text": "Claim A"}],
            "unsupported_claims": None,
        },
    }
    assert captured["get_session_kwargs"] == {
        "owner_user_id": "42",
        "session_id": "research-session-50",
    }
    assert captured["get_bundle_kwargs"] == {
        "owner_user_id": "42",
        "session_id": "research-session-50",
    }
    assert len(added_artifacts) == 1
    assert added_artifacts[0]["type"] == "deep_research_selected_fields"
    assert added_artifacts[0]["mime_type"] == "application/json"
    artifact_uri = str(added_artifacts[0]["uri"])
    artifact_path = Path(artifact_uri.removeprefix("file://"))
    assert artifact_path.name == "deep_research_selected_fields.json"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == result


@pytest.mark.asyncio
async def test_deep_research_select_bundle_fields_adapter_accepts_wait_output_object(monkeypatch):
    """Test deep research bundle-field selector resolves the run from a prior-step output object."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_select_bundle_fields_adapter,
    )

    class _FakeSession:
        id = "research-session-51"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-08T08:15:00+00:00"

    class _FakeResearchService:
        def get_session(self, **kwargs):
            return _FakeSession()

        def get_bundle(self, **kwargs):
            return {
                "question": "What changed next?",
                "verification_summary": {"supported_claim_count": 2},
            }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.select_bundle_fields._build_research_service",
        lambda: _FakeResearchService(),
    )

    result = await run_deep_research_select_bundle_fields_adapter(
        {
            "run": {
                "run_id": "research-session-51",
                "bundle_url": "/api/v1/research/runs/research-session-51/bundle",
            },
            "fields": ["question", "verification_summary"],
            "save_artifact": False,
        },
        {"user_id": "84"},
    )

    assert result == {
        "run_id": "research-session-51",
        "status": "completed",
        "selected_fields": {
            "question": "What changed next?",
            "verification_summary": {"supported_claim_count": 2},
        },
    }


@pytest.mark.asyncio
async def test_deep_research_select_bundle_fields_adapter_dedupes_fields_in_requested_order(monkeypatch):
    """Test duplicate requested fields are deduped while preserving first-seen order."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_select_bundle_fields_adapter,
    )

    class _FakeSession:
        id = "research-session-52"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-08T08:20:00+00:00"

    class _FakeResearchService:
        def get_session(self, **kwargs):
            return _FakeSession()

        def get_bundle(self, **kwargs):
            return {
                "verification_summary": {"supported_claim_count": 1},
                "question": "Ordered fields",
            }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.select_bundle_fields._build_research_service",
        lambda: _FakeResearchService(),
    )

    result = await run_deep_research_select_bundle_fields_adapter(
        {
            "run_id": "research-session-52",
            "fields": [
                "verification_summary",
                "question",
                "verification_summary",
                "question",
            ],
            "save_artifact": False,
        },
        {"user_id": "42"},
    )

    assert list(result["selected_fields"].keys()) == [
        "verification_summary",
        "question",
    ]


@pytest.mark.asyncio
async def test_deep_research_select_bundle_fields_adapter_rejects_invalid_field_name(monkeypatch):
    """Test deep research bundle-field selector rejects field names outside the fixed allowlist."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_select_bundle_fields_adapter,
    )

    with pytest.raises(Exception, match="fields"):
        await run_deep_research_select_bundle_fields_adapter(
            {
                "run_id": "research-session-53",
                "fields": ["question", "made_up_field"],
            },
            {"user_id": "42"},
        )


@pytest.mark.asyncio
async def test_deep_research_select_bundle_fields_adapter_rejects_unknown_config_keys(monkeypatch):
    """Test deep research bundle-field selector rejects unexpected config keys at runtime."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_select_bundle_fields_adapter,
    )

    with pytest.raises(Exception, match="extra|unknown|forbidden"):
        await run_deep_research_select_bundle_fields_adapter(
            {
                "run_id": "research-session-54",
                "fields": ["question"],
                "unexpected": True,
            },
            {"user_id": "42"},
        )


@pytest.mark.asyncio
async def test_deep_research_select_bundle_fields_adapter_rejects_non_completed_runs(monkeypatch):
    """Test deep research bundle-field selector rejects runs that have not completed."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_select_bundle_fields_adapter,
    )

    class _FakeSession:
        id = "research-session-55"
        status = "synthesizing"
        phase = "synthesizing"
        control_state = "running"
        completed_at = None

    class _FakeResearchService:
        def get_session(self, **kwargs):
            return _FakeSession()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.select_bundle_fields._build_research_service",
        lambda: _FakeResearchService(),
    )

    with pytest.raises(RuntimeError, match="completed runs only"):
        await run_deep_research_select_bundle_fields_adapter(
            {"run_id": "research-session-55", "fields": ["question"]},
            {"user_id": "42"},
        )


@pytest.mark.asyncio
async def test_deep_research_select_bundle_fields_adapter_rejects_oversized_inline_payload(
    monkeypatch,
):
    """Test deep research bundle-field selector enforces the inline selected-fields size limit."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_deep_research_select_bundle_fields_adapter,
    )

    class _FakeSession:
        id = "research-session-56"
        status = "completed"
        phase = "completed"
        control_state = "running"
        completed_at = "2026-03-08T08:25:00+00:00"

    class _FakeResearchService:
        def get_session(self, **kwargs):
            return _FakeSession()

        def get_bundle(self, **kwargs):
            return {
                "report_markdown": "x" * 2048,
            }

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.select_bundle_fields._build_research_service",
        lambda: _FakeResearchService(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.research.select_bundle_fields.MAX_SELECTED_FIELDS_BYTES",
        64,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="inline size limit"):
        await run_deep_research_select_bundle_fields_adapter(
            {
                "run_id": "research-session-56",
                "fields": ["report_markdown"],
                "save_artifact": False,
            },
            {"user_id": "42"},
        )


# =============================================================================
# arXiv Search Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_arxiv_search_adapter_test_mode(monkeypatch):
    """Test arXiv search returns simulated results in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_search_adapter,
    )

    config = {"query": "machine learning", "max_results": 5}
    context = {}

    result = await run_arxiv_search_adapter(config, context)

    assert "papers" in result
    assert "total_results" in result
    assert result.get("simulated") is True
    assert result["query"] == "machine learning"
    assert len(result["papers"]) == 1
    paper = result["papers"][0]
    assert "arxiv_id" in paper
    assert "title" in paper
    assert "authors" in paper
    assert "machine learning" in paper["title"]


@pytest.mark.asyncio
async def test_arxiv_search_adapter_test_mode_y(monkeypatch):
    """Test arXiv search treats TEST_MODE=y as enabled."""
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_search_adapter,
    )

    result = await run_arxiv_search_adapter({"query": "graph theory"}, {})

    assert result.get("simulated") is True
    assert result["query"] == "graph theory"


@pytest.mark.asyncio
async def test_arxiv_search_adapter_empty_query(monkeypatch):
    """Test arXiv search handles empty query gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_search_adapter,
    )

    config = {"query": "", "max_results": 5}
    context = {}

    result = await run_arxiv_search_adapter(config, context)

    assert result["papers"] == []
    assert result.get("error") == "missing_query"


@pytest.mark.asyncio
async def test_arxiv_search_adapter_with_template_context(monkeypatch):
    """Test arXiv search with template substitution from context."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_search_adapter,
    )

    config = {"query": "{{ inputs.topic }}", "max_results": 10}
    context = {"inputs": {"topic": "deep learning"}}

    result = await run_arxiv_search_adapter(config, context)

    assert result.get("simulated") is True
    assert "deep learning" in result["query"]


@pytest.mark.asyncio
async def test_arxiv_search_adapter_cancelled(monkeypatch):
    """Test arXiv search respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_search_adapter,
    )

    config = {"query": "test", "max_results": 5}
    context = {"is_cancelled": lambda: True}

    result = await run_arxiv_search_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_arxiv_search_adapter_sort_options(monkeypatch):
    """Test arXiv search with different sort options."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_search_adapter,
    )

    config = {
        "query": "neural networks",
        "max_results": 5,
        "sort_by": "submittedDate",
        "sort_order": "ascending",
    }
    context = {}

    result = await run_arxiv_search_adapter(config, context)

    assert result.get("simulated") is True
    assert "papers" in result


@pytest.mark.asyncio
async def test_arxiv_search_adapter_sanitizes_backend_errors(monkeypatch):
    """Test arXiv search hides provider exception details."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_search_adapter,
        search as search_module,
    )

    class _ExplodingSearch:
        def __init__(self, *args, **kwargs):
            pass

        def results(self):
            raise RuntimeError("arXiv backend exploded at /private/arxiv-cache")

    fake_arxiv = types.SimpleNamespace(
        Search=_ExplodingSearch,
        SortCriterion=types.SimpleNamespace(
            Relevance="relevance",
            LastUpdatedDate="last_updated",
            SubmittedDate="submitted",
        ),
        SortOrder=types.SimpleNamespace(Ascending="ascending", Descending="descending"),
    )
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv)

    messages: list[str] = []
    sink_id = search_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_arxiv_search_adapter({"query": "private topic"}, {})
    finally:
        search_module.logger.remove(sink_id)

    assert result == {"papers": [], "error": "arXiv search failed"}
    joined = "\n".join(messages)
    assert "arXiv search error" in joined
    assert "arXiv backend exploded" not in joined
    assert "/private/arxiv-cache" not in joined


# =============================================================================
# arXiv Download Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_arxiv_download_adapter_test_mode(monkeypatch):
    """Test arXiv download returns simulated path in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_download_adapter,
    )

    config = {"arxiv_id": "2301.00001"}
    context = {}

    result = await run_arxiv_download_adapter(config, context)

    assert result.get("simulated") is True
    assert result.get("downloaded") is True
    assert "pdf_path" in result
    assert "2301.00001" in result["pdf_path"]
    assert result["arxiv_id"] == "2301.00001"


@pytest.mark.asyncio
async def test_arxiv_download_adapter_with_pdf_url(monkeypatch):
    """Test arXiv download with direct PDF URL."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_download_adapter,
    )

    config = {"pdf_url": "https://arxiv.org/pdf/2301.00001.pdf"}
    context = {}

    result = await run_arxiv_download_adapter(config, context)

    assert result.get("simulated") is True
    assert result.get("downloaded") is True


@pytest.mark.asyncio
async def test_arxiv_download_adapter_missing_id(monkeypatch):
    """Test arXiv download handles missing ID gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_download_adapter,
    )

    config = {}
    context = {}

    result = await run_arxiv_download_adapter(config, context)

    assert result.get("downloaded") is False
    assert result.get("error") == "missing_arxiv_id_or_pdf_url"


@pytest.mark.asyncio
async def test_arxiv_download_adapter_from_context(monkeypatch):
    """Test arXiv download extracts ID from previous step context."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_download_adapter,
    )

    config = {}
    context = {"prev": {"arxiv_id": "2301.00002", "pdf_url": ""}}

    result = await run_arxiv_download_adapter(config, context)

    assert result.get("simulated") is True
    assert result.get("downloaded") is True
    assert result["arxiv_id"] == "2301.00002"


@pytest.mark.asyncio
async def test_arxiv_download_adapter_cancelled(monkeypatch):
    """Test arXiv download respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_download_adapter,
    )

    config = {"arxiv_id": "2301.00001"}
    context = {"is_cancelled": lambda: True}

    result = await run_arxiv_download_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_arxiv_download_adapter_sanitizes_backend_errors(monkeypatch):
    """Test arXiv download hides provider exception details."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_download_adapter,
        search as search_module,
    )

    class _ExplodingSearch:
        def __init__(self, *args, **kwargs):
            pass

        def results(self):
            raise RuntimeError("arXiv download token at /private/arxiv-download-cache")

    fake_arxiv = types.SimpleNamespace(Search=_ExplodingSearch)
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv)

    messages: list[str] = []
    sink_id = search_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_arxiv_download_adapter({"arxiv_id": "2301.00001"}, {"step_run_id": "arxiv-test"})
    finally:
        search_module.logger.remove(sink_id)

    assert result == {"error": "arXiv download failed", "downloaded": False}
    joined = "\n".join(messages)
    assert "arXiv download error" in joined
    assert "arXiv download token" not in joined
    assert "/private/arxiv-download-cache" not in joined


# =============================================================================
# PubMed Search Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_pubmed_search_adapter_test_mode(monkeypatch):
    """Test PubMed search returns simulated results in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_pubmed_search_adapter,
    )

    config = {"query": "cancer treatment", "max_results": 10}
    context = {}

    result = await run_pubmed_search_adapter(config, context)

    assert "papers" in result
    assert "total_results" in result
    assert result.get("simulated") is True
    assert result["query"] == "cancer treatment"
    assert len(result["papers"]) == 1
    paper = result["papers"][0]
    assert "pmid" in paper
    assert "title" in paper
    assert "authors" in paper
    assert "cancer treatment" in paper["title"]


@pytest.mark.asyncio
async def test_pubmed_search_adapter_empty_query(monkeypatch):
    """Test PubMed search handles empty query gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_pubmed_search_adapter,
    )

    config = {"query": ""}
    context = {}

    result = await run_pubmed_search_adapter(config, context)

    assert result["papers"] == []
    assert result.get("error") == "missing_query"


@pytest.mark.asyncio
async def test_pubmed_search_adapter_with_template(monkeypatch):
    """Test PubMed search with template substitution."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_pubmed_search_adapter,
    )

    config = {"query": "{{ inputs.disease }}"}
    context = {"inputs": {"disease": "diabetes"}}

    result = await run_pubmed_search_adapter(config, context)

    assert result.get("simulated") is True
    assert "diabetes" in result["query"]


@pytest.mark.asyncio
async def test_pubmed_search_adapter_cancelled(monkeypatch):
    """Test PubMed search respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_pubmed_search_adapter,
    )

    config = {"query": "test"}
    context = {"is_cancelled": lambda: True}

    result = await run_pubmed_search_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_pubmed_search_adapter_sanitizes_backend_errors(monkeypatch):
    """Test PubMed search hides HTTP backend exception details."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_pubmed_search_adapter,
        search as search_module,
    )

    _patch_module_import(
        monkeypatch,
        "httpx",
        _fake_httpx_module_with_error("PubMed token at /private/pubmed-cache"),
    )

    messages: list[str] = []
    sink_id = search_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_pubmed_search_adapter({"query": "private topic"}, {})
    finally:
        search_module.logger.remove(sink_id)

    assert result == {"papers": [], "error": "PubMed search failed"}
    joined = "\n".join(messages)
    assert "PubMed search error" in joined
    assert "PubMed token" not in joined
    assert "/private/pubmed-cache" not in joined


# =============================================================================
# Semantic Scholar Search Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_semantic_scholar_search_adapter_test_mode(monkeypatch):
    """Test Semantic Scholar search returns simulated results in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_semantic_scholar_search_adapter,
    )

    config = {"query": "transformer architecture", "max_results": 5}
    context = {}

    result = await run_semantic_scholar_search_adapter(config, context)

    assert "papers" in result
    assert "total_results" in result
    assert result.get("simulated") is True
    assert result["query"] == "transformer architecture"
    assert len(result["papers"]) == 1
    paper = result["papers"][0]
    assert "paper_id" in paper
    assert "title" in paper
    assert "authors" in paper
    assert "citation_count" in paper
    assert "transformer architecture" in paper["title"]


@pytest.mark.asyncio
async def test_semantic_scholar_search_adapter_empty_query(monkeypatch):
    """Test Semantic Scholar search handles empty query gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_semantic_scholar_search_adapter,
    )

    config = {"query": ""}
    context = {}

    result = await run_semantic_scholar_search_adapter(config, context)

    assert result["papers"] == []
    assert result.get("error") == "missing_query"


@pytest.mark.asyncio
async def test_semantic_scholar_search_adapter_with_fields(monkeypatch):
    """Test Semantic Scholar search with custom fields."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_semantic_scholar_search_adapter,
    )

    config = {
        "query": "attention mechanism",
        "max_results": 10,
        "fields": ["title", "abstract", "year"],
    }
    context = {}

    result = await run_semantic_scholar_search_adapter(config, context)

    assert result.get("simulated") is True
    assert "papers" in result


@pytest.mark.asyncio
async def test_semantic_scholar_search_adapter_cancelled(monkeypatch):
    """Test Semantic Scholar search respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_semantic_scholar_search_adapter,
    )

    config = {"query": "test"}
    context = {"is_cancelled": lambda: True}

    result = await run_semantic_scholar_search_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_semantic_scholar_search_adapter_sanitizes_backend_errors(monkeypatch):
    """Test Semantic Scholar search hides HTTP backend exception details."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_semantic_scholar_search_adapter,
        search as search_module,
    )

    _patch_module_import(
        monkeypatch,
        "httpx",
        _fake_httpx_module_with_error("Semantic Scholar token at /private/s2-cache"),
    )

    messages: list[str] = []
    sink_id = search_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_semantic_scholar_search_adapter({"query": "private topic"}, {})
    finally:
        search_module.logger.remove(sink_id)

    assert result == {"papers": [], "error": "Semantic Scholar search failed"}
    joined = "\n".join(messages)
    assert "Semantic Scholar search error" in joined
    assert "Semantic Scholar token" not in joined
    assert "/private/s2-cache" not in joined


# =============================================================================
# Google Scholar Search Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_google_scholar_search_adapter_test_mode(monkeypatch):
    """Test Google Scholar search returns simulated results in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_google_scholar_search_adapter,
    )

    config = {"query": "natural language processing", "max_results": 10}
    context = {}

    result = await run_google_scholar_search_adapter(config, context)

    assert "papers" in result
    assert "total_results" in result
    assert result.get("simulated") is True
    assert result["query"] == "natural language processing"
    assert len(result["papers"]) == 1
    paper = result["papers"][0]
    assert "title" in paper
    assert "authors" in paper
    assert "citation_count" in paper
    assert "natural language processing" in paper["title"]


@pytest.mark.asyncio
async def test_google_scholar_search_adapter_empty_query(monkeypatch):
    """Test Google Scholar search handles empty query gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_google_scholar_search_adapter,
    )

    config = {"query": ""}
    context = {}

    result = await run_google_scholar_search_adapter(config, context)

    assert result["papers"] == []
    assert result.get("error") == "missing_query"


@pytest.mark.asyncio
async def test_google_scholar_search_adapter_with_template(monkeypatch):
    """Test Google Scholar search with template substitution."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_google_scholar_search_adapter,
    )

    config = {"query": "{{ inputs.research_topic }}"}
    context = {"inputs": {"research_topic": "reinforcement learning"}}

    result = await run_google_scholar_search_adapter(config, context)

    assert result.get("simulated") is True
    assert "reinforcement learning" in result["query"]


@pytest.mark.asyncio
async def test_google_scholar_search_adapter_cancelled(monkeypatch):
    """Test Google Scholar search respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_google_scholar_search_adapter,
    )

    config = {"query": "test"}
    context = {"is_cancelled": lambda: True}

    result = await run_google_scholar_search_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_google_scholar_search_adapter_sanitizes_backend_errors(monkeypatch):
    """Test Google Scholar search hides provider exception details."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_google_scholar_search_adapter,
        search as search_module,
    )

    fake_scholarly = types.SimpleNamespace(
        search_pubs=lambda query: (_ for _ in ()).throw(
            RuntimeError("Google Scholar token at /private/scholar-cache")
        )
    )
    monkeypatch.setitem(sys.modules, "scholarly", types.SimpleNamespace(scholarly=fake_scholarly))

    messages: list[str] = []
    sink_id = search_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_google_scholar_search_adapter({"query": "private topic"}, {})
    finally:
        search_module.logger.remove(sink_id)

    assert result == {"papers": [], "error": "Google Scholar search failed"}
    joined = "\n".join(messages)
    assert "Google Scholar search error" in joined
    assert "Google Scholar token" not in joined
    assert "/private/scholar-cache" not in joined


# =============================================================================
# Patent Search Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_patent_search_adapter_test_mode(monkeypatch):
    """Test patent search returns simulated results in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_patent_search_adapter,
    )

    config = {"query": "solar panel efficiency", "max_results": 10}
    context = {}

    result = await run_patent_search_adapter(config, context)

    assert "patents" in result
    assert "total_results" in result
    assert result.get("simulated") is True
    assert result["query"] == "solar panel efficiency"
    assert len(result["patents"]) == 1
    patent = result["patents"][0]
    assert "patent_id" in patent
    assert "title" in patent
    assert "assignee" in patent
    assert "inventors" in patent
    assert "solar panel efficiency" in patent["title"]


@pytest.mark.asyncio
async def test_patent_search_adapter_empty_query(monkeypatch):
    """Test patent search handles empty query gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_patent_search_adapter,
    )

    config = {"query": ""}
    context = {}

    result = await run_patent_search_adapter(config, context)

    assert result["patents"] == []
    assert result.get("error") == "missing_query"


@pytest.mark.asyncio
async def test_patent_search_adapter_with_template(monkeypatch):
    """Test patent search with template substitution."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_patent_search_adapter,
    )

    config = {"query": "{{ inputs.invention }}"}
    context = {"inputs": {"invention": "battery technology"}}

    result = await run_patent_search_adapter(config, context)

    assert result.get("simulated") is True
    assert "battery technology" in result["query"]


@pytest.mark.asyncio
async def test_patent_search_adapter_cancelled(monkeypatch):
    """Test patent search respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_patent_search_adapter,
    )

    config = {"query": "test"}
    context = {"is_cancelled": lambda: True}

    result = await run_patent_search_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_patent_search_adapter_sanitizes_backend_errors(monkeypatch):
    """Test patent search hides HTTP backend exception details."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_patent_search_adapter,
        search as search_module,
    )

    _patch_module_import(
        monkeypatch,
        "httpx",
        _fake_httpx_module_with_error("Patent token at /private/patent-cache"),
    )

    messages: list[str] = []
    sink_id = search_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_patent_search_adapter({"query": "private topic"}, {})
    finally:
        search_module.logger.remove(sink_id)

    assert result == {"patents": [], "error": "Patent search failed"}
    joined = "\n".join(messages)
    assert "Patent search error" in joined
    assert "Patent token" not in joined
    assert "/private/patent-cache" not in joined


# =============================================================================
# DOI Resolve Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_doi_resolve_adapter_test_mode(monkeypatch):
    """Test DOI resolve returns simulated metadata in test mode."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_doi_resolve_adapter,
    )

    config = {"doi": "10.1234/example"}
    context = {}

    result = await run_doi_resolve_adapter(config, context)

    assert "metadata" in result
    assert result.get("resolved") is True
    assert result.get("simulated") is True
    metadata = result["metadata"]
    assert "doi" in metadata
    assert "title" in metadata
    assert "authors" in metadata
    assert metadata["doi"] == "10.1234/example"


@pytest.mark.asyncio
async def test_doi_resolve_adapter_with_prefix(monkeypatch):
    """Test DOI resolve handles various DOI formats."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_doi_resolve_adapter,
    )

    # Test with https://doi.org/ prefix
    config = {"doi": "https://doi.org/10.1234/example"}
    context = {}

    result = await run_doi_resolve_adapter(config, context)

    assert result.get("resolved") is True
    assert result["metadata"]["doi"] == "10.1234/example"

    # Test with doi: prefix
    config = {"doi": "doi:10.1234/example2"}
    result = await run_doi_resolve_adapter(config, context)

    assert result.get("resolved") is True
    assert result["metadata"]["doi"] == "10.1234/example2"


@pytest.mark.asyncio
async def test_doi_resolve_adapter_missing_doi(monkeypatch):
    """Test DOI resolve handles missing DOI gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_doi_resolve_adapter,
    )

    config = {}
    context = {}

    result = await run_doi_resolve_adapter(config, context)

    assert result.get("resolved") is False
    assert result.get("error") == "missing_doi"
    assert result["metadata"] == {}


@pytest.mark.asyncio
async def test_doi_resolve_adapter_from_context(monkeypatch):
    """Test DOI resolve extracts DOI from previous step context."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_doi_resolve_adapter,
    )

    config = {}
    context = {"prev": {"doi": "10.5678/fromcontext"}}

    result = await run_doi_resolve_adapter(config, context)

    assert result.get("resolved") is True
    assert result["metadata"]["doi"] == "10.5678/fromcontext"


@pytest.mark.asyncio
async def test_doi_resolve_adapter_cancelled(monkeypatch):
    """Test DOI resolve respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_doi_resolve_adapter,
    )

    config = {"doi": "10.1234/example"}
    context = {"is_cancelled": lambda: True}

    result = await run_doi_resolve_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_doi_resolve_adapter_sanitizes_backend_errors(monkeypatch):
    """Test DOI resolve hides backend exception details."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        bibliography as bibliography_module,
        run_doi_resolve_adapter,
    )

    class ExplodingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            raise RuntimeError("DOI backend exploded at /private/doi-cache")

    class FakeHttpx:
        AsyncClient = staticmethod(lambda *args, **kwargs: ExplodingClient())

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "httpx":
            return FakeHttpx
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    messages: list[str] = []
    sink_id = bibliography_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_doi_resolve_adapter({"doi": "10.1234/example"}, {})
    finally:
        bibliography_module.logger.remove(sink_id)

    assert result == {
        "metadata": {},
        "error": "DOI resolution failed",
        "resolved": False,
    }
    joined = "\n".join(messages)
    assert "DOI resolve error" in joined
    assert "DOI backend exploded" not in joined
    assert "/private/doi-cache" not in joined


# =============================================================================
# Reference Parse Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_reference_parse_adapter_basic(monkeypatch):
    """Test reference parsing with mocked LLM call."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_reference_parse_adapter,
    )

    # Mock the chat service to return structured JSON
    async def mock_chat(*args, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"authors": ["Smith, J.", "Doe, A."], "title": "Test Paper", "journal": "Test Journal", "year": "2023", "volume": "1", "pages": "1-10"}'
                    }
                }
            ]
        }

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat)

    config = {
        "citation": "Smith, J. & Doe, A. (2023). Test Paper. Test Journal, 1, 1-10.",
        "provider": "openai",
        "model": "gpt-4",
    }
    context = {}

    result = await run_reference_parse_adapter(config, context)

    assert "parsed" in result
    parsed = result["parsed"]
    assert parsed.get("title") == "Test Paper"
    assert "Smith" in str(parsed.get("authors"))


@pytest.mark.asyncio
async def test_reference_parse_adapter_empty_citation(monkeypatch):
    """Test reference parse handles empty citation gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_reference_parse_adapter,
    )

    config = {"citation": ""}
    context = {}

    result = await run_reference_parse_adapter(config, context)

    assert result["parsed"] == {}
    assert result.get("error") == "missing_citation"


@pytest.mark.asyncio
async def test_reference_parse_adapter_with_template(monkeypatch):
    """Test reference parse with template substitution."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_reference_parse_adapter,
    )

    async def mock_chat(*args, **kwargs):
        return {
            "choices": [
                {"message": {"content": '{"title": "Parsed Title", "year": "2024"}'}}
            ]
        }

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat)

    config = {"citation": "{{ inputs.ref }}"}
    context = {"inputs": {"ref": "Test citation string"}}

    result = await run_reference_parse_adapter(config, context)

    assert "parsed" in result


@pytest.mark.asyncio
async def test_reference_parse_adapter_cancelled(monkeypatch):
    """Test reference parse respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_reference_parse_adapter,
    )

    config = {"citation": "test citation"}
    context = {"is_cancelled": lambda: True}

    result = await run_reference_parse_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_reference_parse_adapter_invalid_json_response(monkeypatch):
    """Test reference parse handles invalid JSON from LLM gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_reference_parse_adapter,
    )

    async def mock_chat(*args, **kwargs):
        return {"choices": [{"message": {"content": "This is not valid JSON"}}]}

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat)

    config = {"citation": "Some citation text"}
    context = {}

    result = await run_reference_parse_adapter(config, context)

    assert result["parsed"] == {}
    assert "raw_text" in result


@pytest.mark.asyncio
async def test_reference_parse_adapter_sanitizes_backend_errors(monkeypatch):
    """Test reference parse hides LLM backend exception details."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        bibliography as bibliography_module,
        run_reference_parse_adapter,
    )

    async def explode_chat(*args, **kwargs):
        raise RuntimeError("reference backend exploded at /private/reference-cache")

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc

    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", explode_chat)

    messages: list[str] = []
    sink_id = bibliography_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_reference_parse_adapter({"citation": "Smith 2024. Example."}, {})
    finally:
        bibliography_module.logger.remove(sink_id)

    assert result == {"parsed": {}, "error": "Reference parsing failed"}
    joined = "\n".join(messages)
    assert "Reference parse error" in joined
    assert "reference backend exploded" not in joined
    assert "/private/reference-cache" not in joined


# =============================================================================
# BibTeX Generate Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_bibtex_generate_adapter_basic(monkeypatch):
    """Test BibTeX generation from metadata."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_bibtex_generate_adapter,
    )

    config = {
        "metadata": {
            "title": "Test Paper Title",
            "authors": ["John Smith", "Jane Doe"],
            "journal": "Test Journal",
            "year": 2023,
            "volume": "10",
            "pages": "1-15",
            "doi": "10.1234/test",
        },
        "entry_type": "article",
    }
    context = {}

    result = await run_bibtex_generate_adapter(config, context)

    assert "bibtex" in result
    assert "@article" in result["bibtex"]
    assert "Test Paper Title" in result["bibtex"]
    assert "John Smith and Jane Doe" in result["bibtex"]
    assert "cite_key" in result


@pytest.mark.asyncio
async def test_bibtex_generate_adapter_custom_cite_key(monkeypatch):
    """Test BibTeX generation with custom citation key."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_bibtex_generate_adapter,
    )

    config = {
        "metadata": {
            "title": "Custom Key Paper",
            "authors": ["Author Name"],
            "year": 2024,
        },
        "entry_type": "inproceedings",
        "cite_key": "customkey2024",
    }
    context = {}

    result = await run_bibtex_generate_adapter(config, context)

    assert "bibtex" in result
    assert "@inproceedings{customkey2024," in result["bibtex"]
    assert result["cite_key"] == "customkey2024"


@pytest.mark.asyncio
async def test_bibtex_generate_adapter_auto_cite_key(monkeypatch):
    """Test BibTeX generation with auto-generated citation key."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_bibtex_generate_adapter,
    )

    config = {
        "metadata": {
            "title": "Auto Key Paper",
            "authors": ["Smith, John"],
            "year": 2023,
        },
    }
    context = {}

    result = await run_bibtex_generate_adapter(config, context)

    assert "bibtex" in result
    # Auto-generated key should be based on last name and year
    assert "smith2023" in result["cite_key"].lower() or "john2023" in result["cite_key"].lower()


@pytest.mark.asyncio
async def test_bibtex_generate_adapter_missing_metadata(monkeypatch):
    """Test BibTeX generation handles missing metadata gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_bibtex_generate_adapter,
    )

    config = {}
    context = {}

    result = await run_bibtex_generate_adapter(config, context)

    assert result["bibtex"] == ""
    assert result.get("error") == "missing_metadata"


@pytest.mark.asyncio
async def test_bibtex_generate_adapter_from_context(monkeypatch):
    """Test BibTeX generation uses metadata from previous step."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_bibtex_generate_adapter,
    )

    config = {"entry_type": "book"}
    context = {
        "prev": {
            "metadata": {
                "title": "Context Paper",
                "authors": ["Context Author"],
                "year": 2022,
                "publisher": "Test Publisher",
            }
        }
    }

    result = await run_bibtex_generate_adapter(config, context)

    assert "bibtex" in result
    assert "@book" in result["bibtex"]
    assert "Context Paper" in result["bibtex"]


@pytest.mark.asyncio
async def test_bibtex_generate_adapter_cancelled(monkeypatch):
    """Test BibTeX generation respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_bibtex_generate_adapter,
    )

    config = {"metadata": {"title": "test"}}
    context = {"is_cancelled": lambda: True}

    result = await run_bibtex_generate_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_bibtex_generate_adapter_all_fields(monkeypatch):
    """Test BibTeX generation with all available fields."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_bibtex_generate_adapter,
    )

    config = {
        "metadata": {
            "title": "Complete Paper",
            "authors": ["First Author", "Second Author", "Third Author"],
            "journal": "Prestigious Journal",
            "year": 2024,
            "volume": "42",
            "number": "3",
            "pages": "100-150",
            "doi": "10.1234/complete",
            "url": "https://example.com/paper",
            "publisher": "Academic Press",
            "booktitle": "Conference Proceedings",
            "abstract": "This is the abstract of the paper.",
        },
        "entry_type": "article",
    }
    context = {}

    result = await run_bibtex_generate_adapter(config, context)

    bibtex = result["bibtex"]
    assert "title = {Complete Paper}" in bibtex
    assert "journal = {Prestigious Journal}" in bibtex
    assert "volume = {42}" in bibtex
    assert "doi = {10.1234/complete}" in bibtex


# =============================================================================
# Literature Review Adapter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_literature_review_adapter_basic(monkeypatch):
    """Test literature review generation with mocked LLM."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_literature_review_adapter,
    )

    async def mock_chat(*args, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": "This literature review covers recent advances in machine learning..."
                    }
                }
            ]
        }

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat)

    config = {
        "papers": [
            {
                "title": "Paper 1",
                "authors": ["Author A"],
                "year": 2023,
                "summary": "Summary of paper 1",
            },
            {
                "title": "Paper 2",
                "authors": ["Author B"],
                "year": 2024,
                "abstract": "Abstract of paper 2",
            },
        ],
        "topic": "machine learning",
        "style": "brief",
    }
    context = {}

    result = await run_literature_review_adapter(config, context)

    assert "review" in result
    assert result["paper_count"] == 2
    assert result["style"] == "brief"
    assert "machine learning" in result["review"]


@pytest.mark.asyncio
async def test_literature_review_adapter_missing_papers(monkeypatch):
    """Test literature review handles missing papers gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_literature_review_adapter,
    )

    config = {"topic": "test topic"}
    context = {}

    result = await run_literature_review_adapter(config, context)

    assert result["review"] == ""
    assert result.get("error") == "missing_papers"


@pytest.mark.asyncio
async def test_literature_review_adapter_from_context(monkeypatch):
    """Test literature review uses papers from previous step."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_literature_review_adapter,
    )

    async def mock_chat(*args, **kwargs):
        return {
            "choices": [
                {"message": {"content": "Review from context papers..."}}
            ]
        }

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat)

    config = {"topic": "neural networks", "style": "detailed"}
    context = {
        "prev": {
            "papers": [
                {"title": "Context Paper", "authors": ["Author"], "year": 2023}
            ]
        }
    }

    result = await run_literature_review_adapter(config, context)

    assert "review" in result
    assert result["paper_count"] == 1


@pytest.mark.asyncio
async def test_literature_review_adapter_styles(monkeypatch):
    """Test literature review with different style options."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_literature_review_adapter,
    )

    async def mock_chat(*args, **kwargs):
        return {"choices": [{"message": {"content": "Style-specific review..."}}]}

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat)

    papers = [{"title": "Test", "authors": ["A"], "year": 2024}]

    for style in ["brief", "detailed", "comparative"]:
        config = {"papers": papers, "style": style}
        context = {}

        result = await run_literature_review_adapter(config, context)

        assert result["style"] == style
        assert "review" in result


@pytest.mark.asyncio
async def test_literature_review_adapter_cancelled(monkeypatch):
    """Test literature review respects cancellation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_literature_review_adapter,
    )

    config = {"papers": [{"title": "Test", "authors": [], "year": 2024}]}
    context = {"is_cancelled": lambda: True}

    result = await run_literature_review_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_literature_review_adapter_with_template(monkeypatch):
    """Test literature review with template substitution for topic."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_literature_review_adapter,
    )

    async def mock_chat(*args, **kwargs):
        return {"choices": [{"message": {"content": "Review on AI..."}}]}

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat)

    config = {
        "papers": [{"title": "AI Paper", "authors": ["Researcher"], "year": 2024}],
        "topic": "{{ inputs.research_area }}",
    }
    context = {"inputs": {"research_area": "artificial intelligence"}}

    result = await run_literature_review_adapter(config, context)

    assert "review" in result


@pytest.mark.asyncio
async def test_literature_review_adapter_sanitizes_backend_errors(monkeypatch):
    """Test literature review hides LLM backend exception details."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        bibliography as bibliography_module,
        run_literature_review_adapter,
    )

    async def explode_chat(*args, **kwargs):
        raise RuntimeError("literature backend exploded at /private/literature-cache")

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc

    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", explode_chat)

    messages: list[str] = []
    sink_id = bibliography_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        result = await run_literature_review_adapter(
            {"papers": [{"title": "Paper", "authors": ["Author"], "year": 2024}]},
            {},
        )
    finally:
        bibliography_module.logger.remove(sink_id)

    assert result == {
        "review": "",
        "error": "Literature review generation failed",
    }
    joined = "\n".join(messages)
    assert "Literature review error" in joined
    assert "literature backend exploded" not in joined
    assert "/private/literature-cache" not in joined


# =============================================================================
# Integration Tests - Testing Adapter Chaining
# =============================================================================


@pytest.mark.asyncio
async def test_search_to_bibtex_chain(monkeypatch):
    """Test chaining arXiv search to BibTeX generation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_arxiv_search_adapter,
        run_bibtex_generate_adapter,
    )

    # Step 1: Search
    search_config = {"query": "quantum computing", "max_results": 1}
    context = {}

    search_result = await run_arxiv_search_adapter(search_config, context)

    assert search_result.get("simulated") is True
    assert len(search_result["papers"]) == 1

    paper = search_result["papers"][0]

    # Step 2: Generate BibTeX from search result
    bibtex_config = {
        "metadata": {
            "title": paper["title"],
            "authors": paper["authors"],
            "year": 2023,
            "doi": paper.get("doi"),
        },
        "entry_type": "article",
    }

    bibtex_result = await run_bibtex_generate_adapter(bibtex_config, context)

    assert "bibtex" in bibtex_result
    assert paper["title"] in bibtex_result["bibtex"]


@pytest.mark.asyncio
async def test_doi_resolve_to_bibtex_chain(monkeypatch):
    """Test chaining DOI resolve to BibTeX generation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_doi_resolve_adapter,
        run_bibtex_generate_adapter,
    )

    # Step 1: Resolve DOI
    resolve_config = {"doi": "10.1234/testdoi"}
    context = {}

    resolve_result = await run_doi_resolve_adapter(resolve_config, context)

    assert resolve_result.get("resolved") is True

    # Step 2: Generate BibTeX using metadata from context
    bibtex_config = {"entry_type": "article"}
    context_with_prev = {"prev": resolve_result}

    bibtex_result = await run_bibtex_generate_adapter(bibtex_config, context_with_prev)

    assert "bibtex" in bibtex_result
    assert "10.1234/testdoi" in bibtex_result["bibtex"]


@pytest.mark.asyncio
async def test_search_to_literature_review_chain(monkeypatch):
    """Test chaining search to literature review generation."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_semantic_scholar_search_adapter,
        run_literature_review_adapter,
    )

    async def mock_chat(*args, **kwargs):
        return {
            "choices": [
                {"message": {"content": "Comprehensive review of the papers..."}}
            ]
        }

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat)

    # Step 1: Search
    search_config = {"query": "deep learning optimization", "max_results": 5}
    context = {}

    search_result = await run_semantic_scholar_search_adapter(search_config, context)

    assert search_result.get("simulated") is True

    # Step 2: Generate literature review
    review_config = {"topic": "deep learning optimization", "style": "detailed"}
    context_with_prev = {"prev": search_result}

    review_result = await run_literature_review_adapter(review_config, context_with_prev)

    assert "review" in review_result
    assert review_result["paper_count"] == 1  # From simulated search


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_literature_review_adapter_llm_error(monkeypatch):
    """Test literature review handles LLM errors gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_literature_review_adapter,
    )

    async def mock_chat_error(*args, **kwargs):
        raise RuntimeError("LLM service unavailable")

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat_error)

    config = {
        "papers": [{"title": "Test", "authors": ["Author"], "year": 2024}],
        "topic": "test",
    }
    context = {}

    result = await run_literature_review_adapter(config, context)

    assert result["review"] == ""
    assert result.get("error") == "Literature review generation failed"


@pytest.mark.asyncio
async def test_reference_parse_adapter_llm_error(monkeypatch):
    """Test reference parse handles LLM errors gracefully."""
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.Workflows.adapters.research import (
        run_reference_parse_adapter,
    )

    async def mock_chat_error(*args, **kwargs):
        raise RuntimeError("API timeout")

    import tldw_Server_API.app.core.Chat.chat_service as chat_svc
    monkeypatch.setattr(chat_svc, "perform_chat_api_call_async", mock_chat_error)

    config = {"citation": "Smith, J. (2023). Test."}
    context = {}

    result = await run_reference_parse_adapter(config, context)

    assert result["parsed"] == {}
    assert result.get("error") == "Reference parsing failed"


# =============================================================================
# Registry Tests - Verify adapters are properly registered
# =============================================================================


def test_research_adapters_registered():
    """Verify all research adapters are registered in the registry."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    expected_adapters = [
        "arxiv_search",
        "arxiv_download",
        "pubmed_search",
        "semantic_scholar_search",
        "google_scholar_search",
        "patent_search",
        "doi_resolve",
        "reference_parse",
        "bibtex_generate",
        "literature_review",
    ]

    all_adapters = registry.list_adapters()

    for adapter_name in expected_adapters:
        assert adapter_name in all_adapters, f"Missing adapter: {adapter_name}"


def test_research_adapters_have_config_models():
    """Verify all research adapters have config models."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    research_adapters = [
        "arxiv_search",
        "arxiv_download",
        "pubmed_search",
        "semantic_scholar_search",
        "google_scholar_search",
        "patent_search",
        "doi_resolve",
        "reference_parse",
        "bibtex_generate",
        "literature_review",
    ]

    for adapter_name in research_adapters:
        spec = registry.get_spec(adapter_name)
        assert spec is not None, f"Missing spec for {adapter_name}"
        assert spec.config_model is not None, f"Missing config_model for {adapter_name}"


def test_research_adapters_in_research_category():
    """Verify research adapters are in the research category."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    expected_adapters = {
        "arxiv_search",
        "arxiv_download",
        "pubmed_search",
        "semantic_scholar_search",
        "google_scholar_search",
        "patent_search",
        "doi_resolve",
        "reference_parse",
        "bibtex_generate",
        "literature_review",
    }

    # Verify each expected adapter is in the research category
    for adapter_name in expected_adapters:
        spec = registry.get_spec(adapter_name)
        assert spec is not None, f"Missing adapter: {adapter_name}"
        assert spec.category == "research", f"{adapter_name} is in category '{spec.category}', expected 'research'"

    # Also verify using get_by_category
    research_adapters = registry.get_by_category("research")
    for adapter_name in expected_adapters:
        assert adapter_name in research_adapters, f"{adapter_name} not found in research category list"
