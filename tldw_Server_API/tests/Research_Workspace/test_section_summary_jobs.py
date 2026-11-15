"""Complete section coverage and durable checkpoint regressions."""

from __future__ import annotations

import copy
import importlib
import json

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.exceptions import ResearchWorkspaceOutputJobError
from tldw_Server_API.app.core.Research_Workspace.research_workspace_outputs import ResearchWorkspaceOutputSubmitRequest

pytestmark = pytest.mark.unit


def request(version="v1", kind="summary"):
    return ResearchWorkspaceOutputSubmitRequest.model_validate(
        {
            "artifact_type": kind,
            "source_ids": ["source"],
            "settings": {"sections": [{"source_id": "source", "node_id": "dsi:1", "navigation_version": version}]},
        }
    )


def test_summary_accepts_reviewed_sections_and_reasoning_budget():
    assert request().settings.max_tokens == 4096
    assert request().settings.sections[0].node_id == "dsi:1"


def test_comparison_rejects_single_section():
    with pytest.raises(ValidationError):
        request(kind="compare_sources")


class Workspace:
    def __init__(self):
        self.row = {"id": "summary-1", "version": 1, "version_metadata": {}}

    def list_workspace_sources(self, workspace_id):
        return [{"id": "source", "media_id": 1, "title": "Collected essays"}]

    def get_workspace_artifact(self, workspace_id, artifact_id):
        return copy.deepcopy(self.row)

    def update_workspace_artifact(self, workspace_id, artifact_id, updates, *, expected_version):
        assert expected_version == self.row["version"]
        self.row.update(copy.deepcopy(updates))
        self.row["version"] += 1
        return copy.deepcopy(self.row)


class Jobs:
    cancelled = False

    def get_job(self, job_id):
        return {"status": "processing", "cancel_requested_at": "now" if self.cancelled else None}

    def update_job_progress(self, *args, **kwargs):
        pass


def args(settings):
    return {
        "payload": {
            "workspace_id": "ws",
            "artifact_id": "summary-1",
            "artifact_type": "summary",
            "source_ids": ["source"],
        },
        "workspace_db": Workspace(),
        "media_db": object(),
        "settings": settings,
        "job_id": 1,
        "job_manager": Jobs(),
        "provider_credentials": object(),
    }


@pytest.mark.asyncio
async def test_failed_batch_resume_preserves_full_ranges_and_coverage(monkeypatch):
    mod = importlib.import_module("tldw_Server_API.app.core.Research_Workspace.section_summaries")
    text = "Author A supports retaining the bank.\n" + "x" * 24_000 + "\nAuthor B opposes it."
    snapshot = {
        "content": text,
        "navigation_version": "v1",
        "page_map": [],
        "nodes": [
            {
                "id": "dsi:1",
                "title": "Two authors",
                "level": 1,
                "target_type": "char_range",
                "target_start": 0,
                "target_end": len(text),
            }
        ],
    }

    async def load(*args, **kwargs):
        return snapshot

    monkeypatch.setattr(mod, "load_navigation_sections", load)
    seen = []
    fail = True

    async def chat(**kwargs):
        nonlocal fail
        assert kwargs["temp"] == 0.7 and kwargs["topp"] == 0.8
        assert kwargs["_provider_call_credentials"] is options["provider_credentials"]
        prompt = kwargs["messages_payload"][0]["content"]
        seen.append(prompt)
        if "Author B" in prompt and fail:
            fail = False
            raise RuntimeError("Temporary provider failure")
        content = (
            json.dumps(
                {
                    "summary": "A proposal is described.",
                    "evidence": [
                        {
                            "claim": "The contribution proposes a bank policy",
                            "quote": "Author A supports retaining the bank."
                            if "SOURCE:" in prompt and "Author A" in prompt
                            else "Author B opposes it.",
                        }
                    ],
                }
            )
            if "SOURCE:" in prompt
            else "The authors disagree [E1.1.1]."
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"temperature": 0.7, "top_p": 0.8}))
    await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    assert row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]["status"] == "failed"
    assert "Partial coverage" in row["content"]
    completed_calls = sum("Author A" in prompt for prompt in seen)
    await mod.process_section_summary(**options)
    section = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]
    assert section["status"] == "completed"
    assert [(b["start_char"], b["end_char"]) for b in section["batches"]] == [(0, 24000), (24000, len(text))]
    assert sum("Author A" in prompt for prompt in seen) == completed_calls
    assert "Coverage: 1 of 1 selected sections" in row["content"]
    assert "PDF page unavailable" in row["content"]


@pytest.mark.asyncio
async def test_stale_navigation_rejected_before_generation(monkeypatch):
    mod = importlib.import_module("tldw_Server_API.app.core.Research_Workspace.section_summaries")

    async def load(*args, **kwargs):
        return {"content": "source", "navigation_version": "new", "nodes": [], "page_map": []}

    monkeypatch.setattr(mod, "load_navigation_sections", load)
    with pytest.raises(ResearchWorkspaceOutputJobError, match="navigation_changed"):
        await mod.process_section_summary(**args(request("old").settings))


def test_markdown_toc_reconstructs_wrapped_numbered_chapters_and_opposing_subsections(monkeypatch):
    from tldw_Server_API.app.core.Media import navigation

    content = """#### **Contents**
**SECTION 1: GOVERNMENT** ........ 19
1. FIRST CHAPTER ...... 23
First Author
2. SECOND CHAPTER
AND RELATED OFFICES ..... 43
Second Author
3. DEBATE CHAPTER ...... 69
THE CASE FOR RETENTION ...... 69
Author A
THE CASE FOR CLOSURE ...... 75
Author B
**SECTION 2: ECONOMY** ..... 87
4. FINAL CHAPTER ...... 91
Final Author
ACKNOWLEDGMENTS
Body starts here.
"""
    monkeypatch.setattr(navigation, "_get_media_text", lambda **kwargs: content)
    nodes = navigation._extract_generated_toc_nodes(1, object(), {"type": "pdf"})
    assert [n["title"] for n in nodes if n["level"] == 2] == [
        "1. FIRST CHAPTER",
        "2. SECOND CHAPTER AND RELATED OFFICES",
        "3. DEBATE CHAPTER",
        "4. FINAL CHAPTER",
    ]
    opposing = [n for n in nodes if n["title"].startswith("THE CASE")]
    assert len(opposing) == 2 and opposing[0]["parent_id"] == opposing[1]["parent_id"]


def test_section_selection_rejects_unselected_source():
    with pytest.raises(ValidationError, match="selected sources"):
        ResearchWorkspaceOutputSubmitRequest.model_validate(
            {
                "artifact_type": "summary",
                "source_ids": ["source"],
                "settings": {"sections": [{"source_id": "other", "node_id": "dsi:1", "navigation_version": "v1"}]},
            }
        )


def test_cancellation_rejects_other_owner_before_mutation():
    from tldw_Server_API.app.core.Research_Workspace.output_jobs import cancel_research_workspace_output_job

    class OtherOwnerJobs:
        def get_job(self, _job_id):
            return {
                "owner_user_id": "someone-else",
                "domain": "research_workspace",
                "job_type": "research_workspace_output",
                "payload": {"workspace_id": "ws"},
            }

        def cancel_job(self, *args, **kwargs):
            pytest.fail("Unauthorized cancellation reached the mutation")

    with pytest.raises(ResearchWorkspaceOutputJobError):
        cancel_research_workspace_output_job(
            workspace_id="ws", job_id=7, job_manager=OtherOwnerJobs(), user_id=1, workspace_db=Workspace()
        )


@pytest.mark.asyncio
async def test_cancellation_retains_checkpoint_without_stopping_worker(monkeypatch):
    from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerTerminalOutcome
    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod
    from tldw_Server_API.app.core.Research_Workspace.output_jobs import process_research_workspace_output_payload

    async def cancel(**kwargs):
        raise mod.SectionSummaryCancelled()

    monkeypatch.setattr(mod, "process_section_summary", cancel)
    from unittest.mock import AsyncMock, MagicMock

    from tldw_Server_API.app.core.Research_Workspace import output_jobs

    runtime = MagicMock(resolve=AsyncMock(return_value=object()), close=AsyncMock())
    monkeypatch.setattr(output_jobs, "ProviderCredentialRuntime", lambda **kwargs: runtime)
    monkeypatch.setattr(output_jobs, "list_active_team_memberships_for_user", AsyncMock(return_value=[]))
    monkeypatch.setattr(output_jobs, "list_org_memberships_for_user", AsyncMock(return_value=[]))
    result = await process_research_workspace_output_payload(
        job={"id": 7},
        payload={"workspace_id": "ws", "artifact_id": "a", "artifact_type": "summary"},
        workspace_db=Workspace(),
        media_db=object(),
        user_id=1,
        job_manager=Jobs(),
    )
    assert isinstance(result, WorkerTerminalOutcome) and result.status == "cancelled"


@pytest.mark.asyncio
async def test_unresolved_section_remains_visible_while_other_section_completes(monkeypatch):
    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    async def load(*_args, **_kwargs):
        return {
            "content": "A supported proposal.",
            "navigation_version": "v1",
            "page_map": [],
            "nodes": [
                {
                    "id": "dsi:1",
                    "title": "Exact section",
                    "level": 1,
                    "target_type": "char_range",
                    "target_start": 0,
                    "target_end": 21,
                },
                {
                    "id": "missing",
                    "title": "Unresolved section",
                    "level": 2,
                    "target_type": "page",
                    "target_start": 20,
                    "source": "generated_toc",
                },
            ],
        }

    async def chat(**kwargs):
        content = (
            json.dumps(
                {
                    "summary": "A proposal.",
                    "evidence": [{"claim": "A proposal is present", "quote": "A supported proposal."}],
                }
            )
            if "SOURCE:" in kwargs["messages_payload"][0]["content"]
            else "A proposal [E1.1.1]."
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "load_navigation_sections", load)
    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    settings = request().settings
    settings.sections.append(type(settings.sections[0])(source_id="source", node_id="missing", navigation_version="v1"))
    options = args(settings)
    await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    assert "Coverage: 1 of 2 selected sections" in row["content"]
    assert "Unresolved section" in row["content"] and "Partial coverage" in row["content"]


def test_report_uses_complete_source_output_contract():
    assert request(kind="report").artifact_type == "report"


def test_submission_persists_job_identity_before_processing_and_resume_keeps_artifact():
    from tldw_Server_API.app.core.Research_Workspace.output_jobs import submit_research_workspace_output_job

    class WritableWorkspace(Workspace):
        def add_workspace_artifact(self, workspace_id, payload):
            self.row = {"version": 1, "workspace_id": workspace_id, **copy.deepcopy(payload)}

    class Queue:
        def __init__(self):
            self.jobs = {}

        def create_job(self, **kwargs):
            job = {"id": len(self.jobs) + 71, "status": "queued", **kwargs}
            self.jobs[job["id"]] = job
            return job

        def get_job(self, job_id):
            return self.jobs[job_id]

    workspace, queue = WritableWorkspace(), Queue()
    first = submit_research_workspace_output_job(
        workspace_id="ws", request=request(), workspace_db=workspace, job_manager=queue, user_id=1
    )
    assert workspace.row["producer_metadata"]["job_id"] == 71
    assert workspace.row["status"] == "generating"
    resume = request()
    resume.resume_artifact_id = first.artifact_id
    with pytest.raises(ResearchWorkspaceOutputJobError, match="already_running"):
        submit_research_workspace_output_job(
            workspace_id="ws", request=resume, workspace_db=workspace, job_manager=queue, user_id=1
        )
    queue.jobs[71]["status"] = "failed"
    retried = submit_research_workspace_output_job(
        workspace_id="ws", request=resume, workspace_db=workspace, job_manager=queue, user_id=1
    )
    assert retried.artifact_id == first.artifact_id and workspace.row["producer_metadata"]["job_id"] == 72


def test_markdown_export_links_verified_evidence_to_authenticated_source_page():
    from tldw_Server_API.app.core.Research_Workspace.section_summaries import render_section_analysis

    markdown = render_section_analysis(
        {
            "title": "Selected chapters",
            "sections": [
                {
                    "media_id": 12,
                    "title": "Debate",
                    "status": "completed",
                    "start_char": 0,
                    "end_char": 100,
                    "batches": [
                        {
                            "evidence": [
                                {
                                    "id": "E1.1.1",
                                    "claim": "A proposal",
                                    "quote": "Exact supporting passage",
                                    "author": "Author A",
                                    "page_spans": [{"page_number": 352, "printed_page_label": "319"}],
                                }
                            ]
                        }
                    ],
                }
            ],
        }
    )
    assert "[PDF page 352](/document-workspace?open=12&page=352) (printed 319)" in markdown
    assert "Exact supporting passage" in markdown


def test_comparison_with_explicit_sections_requires_two_sections_even_with_two_sources():
    with pytest.raises(ValidationError, match="two"):
        ResearchWorkspaceOutputSubmitRequest.model_validate(
            {
                "artifact_type": "compare_sources",
                "source_ids": ["source", "another"],
                "settings": {"sections": [{"source_id": "source", "node_id": "n1", "navigation_version": "v1"}]},
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_section_job_resolves_owner_credentials_and_closes_runtime(monkeypatch, fails):
    from tldw_Server_API.app.core.Research_Workspace import output_jobs as jobs
    from tldw_Server_API.app.core.Research_Workspace import section_summaries as sections

    calls = {}
    handle = object()

    async def teams(owner):
        calls["owner"] = owner
        return [{"team_id": 13}]

    async def orgs(owner):
        return [{"org_id": 17, "status": "active"}, {"org_id": 19, "status": "inactive"}]

    class Runtime:
        def __init__(self, **kwargs):
            calls["scope"] = kwargs

        async def resolve(self, provider, *, model):
            calls["selection"] = (provider, model)
            return handle

        async def close(self):
            calls["closed"] = True

    async def process(**kwargs):
        assert kwargs.get("provider_credentials") is handle
        if fails:
            raise RuntimeError("processor failure")
        return {"partial": False}

    monkeypatch.setattr(jobs, "list_active_team_memberships_for_user", teams, raising=False)
    monkeypatch.setattr(jobs, "list_org_memberships_for_user", orgs, raising=False)
    monkeypatch.setattr(jobs, "ProviderCredentialRuntime", Runtime, raising=False)
    monkeypatch.setattr(sections, "process_section_summary", process)
    kwargs = {
        "job": {"id": 7},
        "payload": {
            "workspace_id": "ws",
            "artifact_id": "a",
            "artifact_type": "summary",
            "user_id": 42,
            "settings": {"provider": "deepseek", "model": "reasoner"},
        },
        "workspace_db": Workspace(),
        "media_db": object(),
        "user_id": 42,
        "job_manager": Jobs(),
    }
    if fails:
        with pytest.raises(RuntimeError, match="processor failure"):
            await jobs.process_research_workspace_output_payload(**kwargs)
    else:
        await jobs.process_research_workspace_output_payload(**kwargs)
    assert calls["scope"]["user_id"] == 42
    assert calls["scope"]["team_ids"] == [13] and calls["scope"]["org_ids"] == [17]
    assert calls["selection"] == ("deepseek", "reasoner") and calls["closed"]


def test_queued_cancel_persists_terminal_artifact_and_retains_completed_evidence():
    from tldw_Server_API.app.core.Research_Workspace.output_jobs import cancel_research_workspace_output_job

    class QueuedJobs:
        status = "queued"

        def get_job(self, _job_id):
            return {
                "id": 7,
                "uuid": "uuid-7",
                "owner_user_id": "1",
                "status": self.status,
                "domain": "research_workspace",
                "job_type": "research_workspace_output",
                "payload": {"workspace_id": "ws", "artifact_id": "summary-1", "artifact_type": "summary"},
            }

        def cancel_job(self, *args, **kwargs):
            self.status = "cancelled"
            return True

    db = Workspace()
    db.row.update(
        status="generating", content="Completed quotation", producer_metadata={"job_id": 7, "source_ids": ["source"]}
    )
    cancel_research_workspace_output_job(
        workspace_id="ws", job_id=7, job_manager=QueuedJobs(), user_id=1, workspace_db=db
    )
    assert db.row["status"] == "failed"
    assert db.row["content"] == "Completed quotation"
    assert db.row["producer_metadata"]["job_status"] == "cancelled"
    assert db.row["producer_metadata"]["job_id"] == 7
