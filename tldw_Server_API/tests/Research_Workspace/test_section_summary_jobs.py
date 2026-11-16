"""Complete section coverage and durable checkpoint regressions."""

from __future__ import annotations

import copy
import importlib
import json
import re

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
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
                            "source_span_ids": ["P1"],
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
    jobs = MagicMock(finalize_cancelled=MagicMock(return_value=True))
    result = await process_research_workspace_output_payload(
        job={"id": 7, "uuid": "original", "worker_id": "worker", "lease_id": "lease"},
        payload={"workspace_id": "ws", "artifact_id": "a", "artifact_type": "summary"},
        workspace_db=Workspace(),
        media_db=object(),
        user_id=1,
        job_manager=jobs,
    )
    assert result == {"artifact_id": "a", "cancelled": True}
    jobs.finalize_cancelled.assert_called_once_with(7, expected_uuid="original", worker_id="worker", lease_id="lease",
        reason="Section output cancelled; completed sections retained")


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
                    "evidence": [{"claim": "A proposal is present", "source_span_ids": ["P1"]}],
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "code"),
    [
        ({"choices": [{"finish_reason": "length", "message": {"content": "{unfinished"}}]}, "output_truncated"),
        ({"choices": [{"message": {"content": ""}}]}, "empty_response"),
        ({"choices": [{"message": {"content": "not JSON"}}]}, "invalid_json"),
        (
            {
                "choices": [
                    {"message": {"content": '{"summary":"S","evidence":[{"claim":"C","source_span_ids":["P999999"]}]}'}}
                ]
            },
            "unknown_source_span",
        ),
    ],
)
async def test_batch_failure_reports_actionable_bounded_reason(monkeypatch, response, code):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(
            return_value={"content": "Exact source proposal", "navigation_version": "v1", "nodes": [], "page_map": []}
        ),
    )
    monkeypatch.setattr(mod, "perform_chat_api_call_async", AsyncMock(return_value=response))
    options = args(request().settings.model_copy(update={"sections": []}))
    result = await mod.process_section_summary(**options)
    section = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][
        0
    ]
    assert section["batches"][0]["error_code"] == code
    assert section["batches"][0]["error"] in options["workspace_db"].row["content"]
    assert result["partial"] and options["workspace_db"].row["status"] == "failed"


@pytest.mark.asyncio
async def test_repeated_unknown_passage_stops_unread_work_and_retry_keeps_completed_batch(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    source = "Unique first portion. " + "supported proposal. " * 7000
    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(return_value={"content": source, "navigation_version": "v1", "nodes": [], "page_map": []}),
    )
    calls = []
    fail = True

    async def chat(**kwargs):
        prompt = kwargs["messages_payload"][0]["content"]
        calls.append(prompt)
        if "SOURCE:" not in prompt:
            return {"choices": [{"message": {"content": "Supported summary [E1.1.1]."}}]}
        passage_id = "P999999" if fail and len(calls) > 1 else "P1"
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "A proposal",
                                "evidence": [{"claim": "A supported claim", "source_span_ids": [passage_id]}],
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": []}))
    result = await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    analysis = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    batches = analysis["sections"][0]["batches"]
    assert len(calls) == 3
    assert [b["status"] for b in batches[:3]] == ["completed", "failed", "failed"]
    assert all(b["status"] == "unread" for b in batches[3:])
    assert analysis["stopped_error_code"] == "unknown_source_span"
    assert "Stopped" in row["content"] and result["partial"]
    first_prompt = calls[0]
    fail = False
    await mod.process_section_summary(**options)
    assert calls.count(first_prompt) == 1
    assert row["status"] == "complete"
    assert "error" not in row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]
    assert "Retry needed" not in row["content"]
    assert "stopped_error_code" not in row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["section", "final"])
async def test_truncated_synthesis_never_marks_output_complete(monkeypatch, stage):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(
            return_value={"content": "Exact source proposal", "navigation_version": "v1", "nodes": [], "page_map": []}
        ),
    )
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"summary": "Summary", "evidence": [{"claim": "Claim", "source_span_ids": ["P1"]}]}
                        )
                    }
                }
            ]
        }
    ]
    if stage == "final":
        responses.append({"choices": [{"message": {"content": "Section summary [E1.1.1]."}}]})
    responses.append({"choices": [{"finish_reason": "length", "message": {"content": "Incomplete synthesis"}}]})
    monkeypatch.setattr(mod, "perform_chat_api_call_async", AsyncMock(side_effect=responses))
    options = args(request().settings.model_copy(update={"sections": []}))
    result = await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    assert row["status"] == "failed" and result["partial"]
    assert "Increase Max tokens" in row["content"]
    assert "Incomplete synthesis" not in row["content"]


def test_deepseek_uses_reasoning_budget_only_when_output_budget_is_unset():
    from tldw_Server_API.app.core.Research_Workspace.research_workspace_outputs import ResearchWorkspaceOutputSettings

    assert ResearchWorkspaceOutputSettings(provider="deepseek").max_tokens == 8192
    assert ResearchWorkspaceOutputSettings(provider="deepseek", max_tokens=4096).max_tokens == 4096
    assert ResearchWorkspaceOutputSettings(provider="other").max_tokens == 4096


@pytest.mark.asyncio
async def test_deepseek_json_mode_retains_exact_original_quote_and_offsets(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    source = "Before. Director of Com\u00ad munications. After."
    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(return_value={"content": source, "navigation_version": "v1", "nodes": [], "page_map": []}),
    )
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        if "SOURCE:" in kwargs["messages_payload"][0]["content"]:
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "A role",
                                    "evidence": [{"claim": "A named role", "source_span_ids": ["P1"]}],
                                }
                            )
                        },
                    }
                ]
            }
        return {"choices": [{"message": {"content": "A role [E1.1.1]."}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": [], "provider": "deepseek"}))
    await mod.process_section_summary(**options)
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
    row = options["workspace_db"].row
    evidence = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]["batches"][0][
        "evidence"
    ][0]
    assert evidence["quote"] == source
    assert source[evidence["start_char"] : evidence["end_char"]] == evidence["quote"]
    assert row["status"] == "complete"


@pytest.mark.parametrize(
    "text",
    [
        "A sentence.\n\nAnother paragraph.",
        "x" * 3500,
        "emoji 😀 and Unicode é. " * 150,
        "First. " + "long words " * 300,
    ],
)
def test_source_passages_partition_every_original_character_without_gaps(text):
    from tldw_Server_API.app.core.Research_Workspace.section_summaries import build_section_extraction_prompt

    prompt, passages = build_section_extraction_prompt("Section", "Summarize", text)
    assert "".join(p["text"] for p in passages) == text
    assert all(p["text"] == text[p["start_char"] : p["end_char"]] and 0 < len(p["text"]) <= 1000 for p in passages)
    assert [p["id"] for p in passages] == [f"P{i + 1}" for i in range(len(passages))]
    assert passages[0]["start_char"] == 0 and passages[-1]["end_char"] == len(text)
    assert all(left["end_char"] == right["start_char"] for left, right in zip(passages, passages[1:]))
    assert "source_span_ids" in prompt


@pytest.mark.asyncio
async def test_selected_passages_copy_exact_source_instead_of_model_retyped_quote(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    source = "Author A proposes\nretention. Soft\u00ad hyphen and ‘quotes’."
    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(return_value={"content": source, "navigation_version": "v1", "nodes": [], "page_map": []}),
    )
    response = {
        "summary": "A proposal",
        "evidence": [{"claim": "Retention is proposed", "source_span_ids": ["P1"], "quote": "Untrusted retyped text"}],
    }
    chat = AsyncMock(
        side_effect=[
            {"choices": [{"message": {"content": json.dumps(response)}}]},
            {"choices": [{"message": {"content": "Summary [E1.1.1]."}}]},
            {"choices": [{"message": {"content": "Synthesis [E1.1.1]."}}]},
        ]
    )
    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(
        request().settings.model_copy(update={"sections": [], "provider": "deepseek", "model": "deepseek-flash"})
    )
    await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    evidence = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]["batches"][0][
        "evidence"
    ][0]
    assert evidence["quote"] == source
    assert evidence["start_char"] == 0 and evidence["end_char"] == len(source)
    assert all(call.kwargs["thinking"] == {"type": "disabled"} for call in chat.call_args_list)
    assert row["status"] == "complete"


@hypothesis_settings(max_examples=30, deadline=None)
@given(st.text(min_size=1, max_size=5000))
def test_passage_partition_preserves_arbitrary_unicode_source(text):
    from tldw_Server_API.app.core.Research_Workspace.section_summaries import build_section_extraction_prompt

    _, passages = build_section_extraction_prompt("Section", "Summarize", text)
    assert "".join(p["text"] for p in passages) == text
    assert all(p["text"] == text[p["start_char"] : p["end_char"]] and len(p["text"]) <= 1000 for p in passages)


@pytest.mark.asyncio
async def test_gateway_validation_failure_checkpoints_and_stops_unread_work(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.exceptions import ChatBadRequestError
    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(
            return_value={
                "content": "Source proposal. " * 6000,
                "navigation_version": "v1",
                "nodes": [],
                "page_map": [],
            }
        ),
    )
    chat = AsyncMock(side_effect=ChatBadRequestError("Unsupported fields for provider: private detail"))
    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": []}))
    result = await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    analysis = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert chat.await_count == 2
    assert analysis["stopped_error_code"] == "provider_failed"
    assert analysis["sections"][0]["batches"][2]["status"] == "unread"
    assert row["status"] == "failed" and result["partial"]
    assert "private detail" not in row["content"]


@pytest.mark.asyncio
async def test_rejected_evidence_logs_only_shape_metadata(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(return_value={"content": "x" * 5000, "navigation_version": "v1", "nodes": [], "page_map": []}),
    )
    logs = []
    monkeypatch.setattr(mod.logger, "warning", lambda message, *values: logs.append((message, values)))
    response = {"summary": "Private summary", "evidence": [{"claim": "Private claim contents", "source_span_ids": []}]}
    monkeypatch.setattr(
        mod,
        "perform_chat_api_call_async",
        AsyncMock(return_value={"choices": [{"message": {"content": json.dumps(response)}}]}),
    )
    result = await mod.process_section_summary(**args(request().settings.model_copy(update={"sections": []})))
    assert result["partial"]
    assert any("Evidence shape rejected" in message and values == ("str", False, "list", 0) for message, values in logs)
    assert "Private claim" not in repr(logs) and "Private summary" not in repr(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_multi_passage_claim_keeps_every_exact_known_source_span(monkeypatch, unknown):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    source = "x" * 5000
    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(return_value={"content": source, "navigation_version": "v1", "nodes": [], "page_map": []}),
    )
    response = {
        "summary": "Combined proposal",
        "evidence": [
            {
                "claim": "A supported multi-passage claim",
                "source_span_ids": ["P1", "P2", "P3", "P999" if unknown else "P4"],
            }
        ],
    }
    chat = AsyncMock(
        side_effect=[
            {"choices": [{"message": {"content": json.dumps(response)}}]},
            {"choices": [{"message": {"content": "Section [E1.1.1] [E1.1.2] [E1.1.3] [E1.1.4]"}}]},
            {"choices": [{"message": {"content": "Synthesis [E1.1.4]"}}]},
        ]
    )
    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": []}))
    await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    batch = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]["batches"][0]
    if unknown:
        assert batch["error_code"] == "unknown_source_span"
        assert row["status"] == "failed"
        assert chat.await_count == 1
    else:
        assert row["status"] == "complete"
        assert [entry["source_span_id"] for entry in batch["evidence"]] == ["P1", "P2", "P3", "P4"]
        assert all(entry["quote"] == source[entry["start_char"] : entry["end_char"]] for entry in batch["evidence"])


@pytest.mark.asyncio
@pytest.mark.parametrize("output_budget", [1024, 8192])
async def test_hierarchical_reduction_caps_only_intermediate_calls_and_preserves_completed_batches(
    monkeypatch, output_budget
):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(return_value={"content": "Source " * 6000, "navigation_version": "v1", "nodes": [], "page_map": []}),
    )
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        if "SOURCE:" in kwargs["messages_payload"][0]["content"]:
            content = json.dumps(
                {"summary": "Main point. " * 1500, "evidence": [{"claim": "Source claim", "source_span_ids": ["P1"]}]}
            )
        else:
            content = "Opposing positions remain distinct [" + re.search(r"\bE\d+\.\d+\.\d+\b", kwargs["messages_payload"][0]["content"]).group(0) + "]."
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": [], "max_tokens": output_budget, "summary_instruction": "Final-only detailed report request"}))
    await mod.process_section_summary(**options)
    assert [call["max_tokens"] for call in calls] == [
        output_budget,
        output_budget,
        min(output_budget, 2048),
        min(output_budget, 2048),
        output_budget,
        output_budget,
    ]
    reduction_prompt = calls[2]["messages_payload"][0]["content"]
    assert "Final-only detailed report request" not in reduction_prompt
    assert "AUTHORITATIVE COVERAGE" not in reduction_prompt
    assert "600–1200" not in reduction_prompt
    assert "Compress only the supplied portions" in reduction_prompt
    assert "Final-only detailed report request" in calls[-1]["messages_payload"][0]["content"]
    assert "reference lists may be shared across a chapter" in calls[0]["messages_payload"][0]["content"]
    assert all("Cited names and bibliographic links do not establish" in call["system_message"] for call in calls)
    assert "at most 250 words" in calls[2]["system_message"]
    assert "concrete policy proposals and opposing policy stances" in calls[2]["system_message"]
    assert options["workspace_db"].row["status"] == "complete"
    calls.clear()
    await mod.process_section_summary(**options)
    assert len(calls) == 1 and "SOURCE:" not in calls[0]["messages_payload"][0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_reduction", [False, True])
async def test_nonreducing_synthesis_stops_without_accepting_or_discarding_saved_evidence(monkeypatch, slow_reduction):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(return_value={"content": "Source " * 6000, "navigation_version": "v1", "nodes": [], "page_map": []}),
    )
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        content = (
            json.dumps(
                {"summary": "Main point. " * 1500, "evidence": [{"claim": "Source claim", "source_span_ids": ["P1"]}]}
            )
            if "SOURCE:" in kwargs["messages_payload"][0]["content"]
            else "Verbose expansion. " * 2000
        )
        if slow_reduction and "SOURCE:" not in kwargs["messages_payload"][0]["content"]:
            content = "x" * (17000 - 10 * (len(calls) - 2))
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": [], "max_tokens": 8192}))
    await mod.process_section_summary(**options)
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert len(calls) == (18 if slow_reduction else 4)
    assert all(call["max_tokens"] == 2048 for call in calls[2:])
    assert options["workspace_db"].row["status"] == "failed"
    assert analysis["sections"][0]["error_code"] == "synthesis_not_reduced"
    assert all(batch["status"] == "completed" and batch["evidence"] for batch in analysis["sections"][0]["batches"])
    assert "Verbose expansion" not in options["workspace_db"].row["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_combined_synthesis_progress_reports_completed_and_failed_sections(monkeypatch, partial):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(
            return_value={
                "content": "Alpha. Bravo.",
                "navigation_version": "v1",
                "page_map": [],
                "nodes": [
                    {
                        "id": "dsi:1",
                        "title": "A",
                        "level": 1,
                        "target_type": "char_range",
                        "target_start": 0,
                        "target_end": 7,
                    },
                    {
                        "id": "dsi:2",
                        "title": "B",
                        "level": 1,
                        "target_type": "char_range",
                        "target_start": 7,
                        "target_end": 13,
                    },
                ],
            }
        ),
    )
    progress = SimpleNamespace(message="")
    final_progress = []
    final_prompts = []

    async def chat(**kwargs):
        prompt = kwargs["messages_payload"][0]["content"]
        if "SOURCE:" in prompt:
            content = (
                "invalid JSON"
                if partial and "Bravo" in prompt
                else json.dumps({"summary": "Alpha", "evidence": [{"claim": "Claim", "source_span_ids": ["P1"]}]})
            )
        else:
            content = "Alpha [" + re.search(r"\bE\d+\.\d+\.\d+\b", prompt).group(0) + "]."
            if "Synthesize the completed" in prompt:
                final_progress.append(progress.message)
                final_prompts.append(prompt)
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    settings = request().settings
    settings.sections.append(settings.sections[0].model_copy(update={"node_id": "dsi:2"}))
    options = args(settings)
    await mod.process_section_summary(**options, progress=progress)
    assert len(final_progress) == 1
    assert f"Combining {1 if partial else 2} of 2 reviewed sections" in final_progress[0]
    assert ("1 section needs retry" in final_progress[0]) is partial
    prompt = final_prompts[0]
    assert "AUTHORITATIVE COVERAGE: " in prompt
    coverage = json.loads(prompt.split("AUTHORITATIVE COVERAGE: ")[1].split("\n")[0])
    assert coverage == {
        "selected_sections": 2,
        "completed_sections": 1 if partial else 2,
        "failed_sections": 1 if partial else 0,
        "unread_sections": 0,
        "completed_portions": 1 if partial else 2,
        "failed_portions": 1 if partial else 0,
        "unread_portions": 0,
    }
    assert "600–1200 words" in prompt
    assert "Unless a different length is requested" in prompt
    assert "what each side supports or opposes" in prompt
    assert "Do not infer unread or truncated selected material from intermediate summaries" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("citation,accepted", [("[E1.1.1, E1.1.1]", True), ("[E1.1.1, E99.1.1]", False)])
async def test_grouped_synthesis_citations_reject_unknown_evidence_ids(monkeypatch, citation, accepted):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(
        mod,
        "load_navigation_sections",
        AsyncMock(return_value={"content": "Exact source.", "navigation_version": "v1", "nodes": [], "page_map": []}),
    )

    async def chat(**kwargs):
        content = (
            json.dumps({"summary": "Summary", "evidence": [{"claim": "Claim", "source_span_ids": ["P1"]}]})
            if "SOURCE:" in kwargs["messages_payload"][0]["content"]
            else "Summary " + citation
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": []}))
    await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    assert row["status"] == ("complete" if accepted else "failed")
    if not accepted:
        section = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]
        assert section["error_code"] == "unknown_evidence_reference"
        assert citation not in row["content"]


@pytest.mark.asyncio
async def test_comparison_uses_exact_child_boundaries_and_keeps_shared_reference_text(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    content = "# Chapter\n**Abolish**\nAuthor A closing position.\n**Retain**\nAuthor B supports retention.\n**ENDNOTES**\nShared citations.\nNext chapter"
    next_start = content.index("Next chapter")
    nodes = [
        {"id": "chapter", "title": "1. Two views", "level": 1, "target_start": 1},
        {"id": "a", "title": "Abolish", "level": 2, "parent_id": "chapter", "target_start": 1},
        {"id": "b", "title": "Retain", "level": 2, "parent_id": "chapter", "target_start": 1},
        {"id": "next", "title": "2. Next", "level": 1, "target_start": 2},
    ]
    for node in nodes:
        node.update(target_type="page", source="generated_toc")
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": content, "navigation_version": "v1", "nodes": nodes,
        "page_map": [{"page_number": 1, "printed_page_label": "1", "start_char": 0, "end_char": next_start},
                     {"page_number": 2, "printed_page_label": "2", "start_char": next_start, "end_char": len(content)}],
    }))
    prompts = []

    async def chat(**kwargs):
        prompt = kwargs["messages_payload"][0]["content"]
        if "SOURCE:" in prompt:
            prompts.append(prompt.split("SOURCE:", 1)[1])
            text = json.dumps({"summary": "An attributed position", "evidence": [{"claim": "Position", "source_span_ids": ["P1"]}]})
        else:
            text = "Two positions remain distinct."
        return {"choices": [{"message": {"content": text}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    selections = [{"source_id": "source", "node_id": node, "navigation_version": "v1"} for node in ("a", "b")]
    settings = request().settings.model_validate({"sections": selections})
    options = args(settings)
    options["payload"]["artifact_type"] = "compare_sources"
    await mod.process_section_summary(**options)
    assert len(prompts) == 2
    assert "Author A closing position" in prompts[0] and "Author B" not in prompts[0]
    assert "Author B supports retention" in prompts[1] and "Author A" not in prompts[1]
    assert "Shared citations" in prompts[1]
    assert options["workspace_db"].row["status"] == "complete"


@pytest.mark.asyncio
async def test_changed_section_checkpoint_returns_typed_terminal_failure_and_preserves_evidence(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from tldw_Server_API.app.core.Research_Workspace import output_jobs, section_summaries

    workspace = Workspace()
    checkpoint = {"sections": [{"content": "Original completed evidence", "batches": [{"status": "completed"}]}]}
    producer = {"origin": "research_workspace_output_job", "job_id": 6, "source_ids": ["source"], "settings": {"sections": []}}
    workspace.row.update(status="generating", content="Original completed evidence", content_type="text/markdown",
                         producer_metadata=copy.deepcopy(producer), version_metadata={"workspaceClient": {"data": {"sectionAnalysis": checkpoint}}})
    runtime = MagicMock(resolve=AsyncMock(return_value=object()), close=AsyncMock())
    monkeypatch.setattr(output_jobs, "ProviderCredentialRuntime", lambda **kwargs: runtime)
    monkeypatch.setattr(output_jobs, "list_active_team_memberships_for_user", AsyncMock(return_value=[]))
    monkeypatch.setattr(output_jobs, "list_org_memberships_for_user", AsyncMock(return_value=[]))
    monkeypatch.setattr(section_summaries, "process_section_summary", AsyncMock(side_effect=ResearchWorkspaceOutputJobError("section_checkpoint_changed", status_code=409)))
    with pytest.raises(ResearchWorkspaceOutputJobError, match="section_checkpoint_changed") as error:
        await output_jobs.process_research_workspace_output_payload(
            job={"id": 7}, payload={"workspace_id": "ws", "artifact_id": "summary-1", "artifact_type": "summary"},
            workspace_db=workspace, media_db=object(), user_id=1, job_manager=Jobs(),
        )
    assert error.value.failure_code == "section_checkpoint_changed" and error.value.retryable is False
    assert workspace.row["status"] == "failed" and workspace.row["content_type"] == "text/markdown"
    assert workspace.row["content"] == "Original completed evidence"
    assert workspace.row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"] == checkpoint
    assert workspace.row["producer_metadata"] == {**producer, "job_id": 7, "artifact_type": "summary", "error": "section_checkpoint_changed"}
    runtime.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True], ids=["changed-checkpoint", "cancelled"])
async def test_research_section_job_terminalizes_through_real_worker_sdk(monkeypatch, tmp_path, cancelled):
    """Research jobs must use generic lease-fenced finalization without provider calls."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
    from tldw_Server_API.app.core.Research_Workspace import output_jobs, section_summaries

    workspace = Workspace()
    checkpoint = {"fingerprint": "old-boundaries", "sections": [{"status": "completed", "batches": [{"status": "completed"}]}]}
    workspace.row.update(status="generating", content="Original completed evidence", content_type="text/markdown")
    if not cancelled:
        workspace.row["version_metadata"] = {"workspaceClient": {"data": {"sectionAnalysis": copy.deepcopy(checkpoint)}}}
    runtime = MagicMock(resolve=AsyncMock(return_value=object()), close=AsyncMock())
    monkeypatch.setattr(output_jobs, "ProviderCredentialRuntime", lambda **kwargs: runtime)
    monkeypatch.setattr(output_jobs, "list_active_team_memberships_for_user", AsyncMock(return_value=[]))
    monkeypatch.setattr(output_jobs, "list_org_memberships_for_user", AsyncMock(return_value=[]))
    monkeypatch.setattr(section_summaries, "load_navigation_sections", AsyncMock(return_value={
        "content": "Exact source evidence.", "navigation_version": "v1", "page_map": [], "nodes": [],
    }))
    provider = AsyncMock()
    monkeypatch.setattr(section_summaries, "perform_chat_api_call_async", provider)
    manager = JobManager(tmp_path / "research-output-sdk.db")
    payload = {"workspace_id": "ws", "artifact_id": "summary-1", "artifact_type": "summary",
               "source_ids": ["source"], "settings": {"provider": "deepseek", "model": "deepseek-flash"}}
    created = manager.create_job(domain="research_workspace", queue="default", job_type="research_workspace_output",
                                 payload=payload, owner_user_id="1", max_retries=3)
    sdk = WorkerSDK(manager, WorkerConfig(domain="research_workspace", queue="default",
                    worker_id="research-test", retry_on_exception=True, bind_completion_token=True))

    async def handler(job):
        sdk.stop()
        if cancelled:
            manager.cancel_job(job["id"], expected_uuid=job["uuid"], reason="User cancellation")
        return await output_jobs.process_research_workspace_output_payload(job=job, payload=payload,
            workspace_db=workspace, media_db=object(), user_id=1, job_manager=manager)

    await asyncio.wait_for(sdk.run(handler=handler), timeout=5)
    stored = manager.get_job(created["id"])
    assert stored["status"] == ("cancelled" if cancelled else "failed")
    if not cancelled:
        assert stored["error_code"] == "section_checkpoint_changed"
        assert workspace.row["content"] == "Original completed evidence"
        assert workspace.row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"] == checkpoint
    else:
        assert workspace.row["producer_metadata"]["job_status"] == "cancelled"
    assert workspace.row["status"] == "failed"
    provider.assert_not_awaited()
    runtime.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_synthesis_cannot_borrow_existing_citations_from_another_section(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    text = "Alice supports abolition. Bob supports retention."
    split = text.index("Bob")
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": text, "navigation_version": "v1", "page_map": [], "nodes": [
            {"id": "a", "title": "Abolition", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": split},
            {"id": "b", "title": "Retention", "level": 1, "target_type": "char_range", "target_start": split, "target_end": len(text)},
        ],
    }))
    prompts = []
    async def chat(**kwargs):
        prompt = kwargs["messages_payload"][0]["content"]
        prompts.append(prompt)
        if "SOURCE:" in prompt:
            author = "Alice" if "Alice" in prompt else "Bob"
            answer = json.dumps({"summary": "A position.", "evidence": [{"claim": "A policy proposal", "author": author, "source_span_ids": ["P1"]}]})
        else:
            answer = "A policy proposal [E1.1.1]."
        return {"choices": [{"message": {"content": answer}}]}
    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    settings = request().settings.model_copy(update={"sections": [
        request().settings.sections[0].model_copy(update={"node_id": name}) for name in ("a", "b")
    ]})
    options = args(settings)
    await mod.process_section_summary(**options)
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert analysis["sections"][1]["status"] == "failed"
    assert analysis["sections"][1]["error_code"] == "unknown_evidence_reference"
    first_synthesis = next(p for p in prompts if p.startswith("Summarize all portions of Abolition"))
    assert "Alice" in first_synthesis and "Abolition" in first_synthesis


def test_generic_faithfulness_instructions_cover_the_observed_failure_mechanisms():
    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod
    prompt, _ = mod.build_section_extraction_prompt("Two contributions", "Summarize", "Source text.")
    reduction = mod.build_section_reduction_prompt(["A source-backed position [E1.1.1]."])
    for text in (prompt, reduction):
        assert "quoted speaker" in text
        assert "both positions" in text
        assert "agency names" in text
        assert "numeric sequence" in text


@pytest.mark.asyncio
async def test_synthesis_does_not_trust_citation_shaped_text_in_a_model_summary(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": "Exact source.", "navigation_version": "v1", "nodes": [], "page_map": [],
    }))
    async def chat(**kwargs):
        content = json.dumps({"summary": "A model-invented reference [E99.1.1]",
            "evidence": [{"claim": "Source claim", "source_span_ids": ["P1"]}]}) if "SOURCE:" in kwargs["messages_payload"][0]["content"] else "Claim [E99.1.1]."
        return {"choices": [{"message": {"content": content}}]}
    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": []}))
    await mod.process_section_summary(**options)
    section = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]
    assert section["status"] == "failed" and section["error_code"] == "unknown_evidence_reference"
