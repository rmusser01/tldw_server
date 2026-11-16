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
    completed_calls = sum("Author A" in prompt.split("SOURCE:\n", 1)[1] for prompt in seen if "SOURCE:\n" in prompt)
    await mod.process_section_summary(**options)
    section = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]
    assert section["status"] == "completed"
    assert [(b["start_char"], b["end_char"]) for b in section["batches"]] == [(0, 24000), (24000, len(text))]
    assert sum("Author A" in prompt.split("SOURCE:\n", 1)[1] for prompt in seen if "SOURCE:\n" in prompt) == completed_calls
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
@pytest.mark.parametrize("provider, json_mode", [("deepseek", True), ("openai", True), ("anthropic", False)])
async def test_provider_json_mode_retains_exact_original_quote_and_offsets(monkeypatch, provider, json_mode):
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
    options = args(request().settings.model_copy(update={"sections": [], "provider": provider}))
    await mod.process_section_summary(**options)
    if json_mode:
        assert calls[0]["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in calls[0]
    assert all("response_format" not in call for call in calls[1:])
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


@pytest.mark.asyncio
async def test_synthesis_vocabulary_contains_only_literal_registered_ids_for_that_call(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    source = "x" * 3000
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": source, "navigation_version": "v1", "nodes": [], "page_map": [],
    }))
    prompts = []

    async def chat(**kwargs):
        prompt = kwargs["messages_payload"][0]["content"]
        prompts.append(prompt)
        if "SOURCE:" in prompt:
            content = json.dumps({"summary": "Source summary includes an invented reference E99.1.1.", "evidence": [
                {"claim": "A combined claim", "source_span_ids": ["P1", "P2", "P3"]}]})
        elif len(prompts) == 2:
            content = "A supported range [E1.1.1–E1.1.3]."
        else:
            content = "An inferred range interior [E1.1.2]."
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": []}))
    await mod.process_section_summary(**options)
    vocabularies = [json.loads(p.split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0]) for p in prompts[1:]]
    assert vocabularies == [["E1.1.1", "E1.1.2", "E1.1.3"], ["E1.1.1", "E1.1.3"]]
    assert all("individual IDs" in p and "range interiors" in p for p in prompts[1:])
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert analysis["sections"][0]["status"] == "completed"
    assert analysis["synthesis_error_code"] == "unknown_evidence_reference"
    assert options["workspace_db"].row["status"] == "failed"
    assert len(prompts) == 3


@pytest.mark.asyncio
async def test_failed_synthesis_diagnostics_are_bounded_and_keep_first_response_across_resume(monkeypatch):
    import hashlib
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": "Exact source.", "navigation_version": "v1", "nodes": [], "page_map": [],
    }))
    rejected, calls, logs = [], [], []
    monkeypatch.setattr(mod.logger, "warning", lambda message, *values: logs.append((message, values)))

    async def chat(**kwargs):
        prompt = kwargs["messages_payload"][0]["content"]
        calls.append(prompt)
        if "SOURCE:" in prompt:
            content = json.dumps({"summary": "Source summary", "evidence": [
                {"claim": "A source claim", "source_span_ids": ["P1"]}]})
        elif "Synthesize the completed" not in prompt:
            content = "A supported claim [E1.1.1]."
        else:
            content = f"PRIVATE REJECTED RESPONSE {len(rejected)} " + "x" * 2500
            content += " ".join(f"[E99.1.{i}]" for i in range(20)) + " [E" + "9" * 100 + ".1.1]"
            rejected.append(content)
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": []}))
    for _ in range(7):
        await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    analysis = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    diagnostics = analysis["generation_failures"]
    assert len(diagnostics) == 5 and analysis["generation_failures_omitted"] == 2
    first = diagnostics[0]
    assert first["response_sha256"] == hashlib.sha256(rejected[0].encode()).hexdigest()
    assert first["response_preview"] == rejected[0][:2048]
    assert first["response_preview_truncated"] is True
    assert first["response_chars"] == len(rejected[0])
    assert first["unknown_id_count"] == 21 and len(first["unknown_ids"]) == 8
    assert all(len(value) <= 64 for value in first["unknown_ids"])
    assert first["unknown_ids_truncated"] is True
    assert first["stage"] == "overview_synthesis" and first["reduction_round"] == first["reduction_group"] == 0
    assert first["allowed_id_count"] == 1 and first["error_code"] == "unknown_evidence_reference"
    assert "PRIVATE REJECTED RESPONSE" not in row["content"] and "PRIVATE REJECTED RESPONSE" not in repr(logs)
    assert row["status"] == "failed" and analysis["sections"][0]["status"] == "completed"
    assert len(calls) == 9
    assert all("prompt" not in key and "credential" not in key for key in first)


@pytest.mark.asyncio
async def test_rejected_reduction_diagnostic_identifies_round_group_and_preserves_portions(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": "Source " * 6000, "navigation_version": "v1", "nodes": [], "page_map": [],
    }))

    async def chat(**kwargs):
        content = (json.dumps({"summary": "Main point. " * 1500, "evidence": [
            {"claim": "Source claim", "source_span_ids": ["P1"]}]})
            if "SOURCE:" in kwargs["messages_payload"][0]["content"] else "Rejected [E99.1.1].")
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings.model_copy(update={"sections": []}))
    await mod.process_section_summary(**options)
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    first = analysis["generation_failures"][0]
    assert (first["stage"], first["section_index"], first["reduction_round"], first["reduction_group"]) == ("section_synthesis", 1, 1, 1)
    assert first["unknown_ids"] == ["E99.1.1"]
    assert all(batch["status"] == "completed" for batch in analysis["sections"][0]["batches"])
    assert options["workspace_db"].row["status"] == "failed"


@pytest.mark.asyncio
async def test_verified_heading_continues_across_portions_and_refreshes_on_legacy_resume(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    source = "# Policy\n\n**Alice Example**\n\n" + "A retained proposal. " * 1500
    snapshot = {"content": source, "navigation_version": "v1", "page_map": [], "nodes": [
        {"id": "dsi:1", "title": "Policy", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": len(source)}]}
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value=snapshot))
    prompts = []

    async def chat(**kwargs):
        prompt = kwargs["messages_payload"][0]["content"]
        prompts.append(prompt)
        content = (json.dumps({"summary": "A retained proposal", "evidence": [
            {"claim": "Retain the policy", "source_span_ids": ["P1"], "author": "Alice Example"}]})
            if "SOURCE:" in prompt else "The proposal is retained [E1.1.1].")
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings)
    await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    analysis = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    section = analysis["sections"][0]
    assert [batch["evidence"][0]["author"] for batch in section["batches"]] == ["Alice Example", "Alice Example"]
    assert "Alice Example" in prompts[1] and "Alice Example" not in prompts[1].split("SOURCE:\n", 1)[1]
    late_evidence = section["batches"][1]["evidence"][0]
    assert f"[P1] source characters [{late_evidence['start_char']}, {late_evidence['end_char']})\n" in prompts[1]
    assert section["attribution_contexts"]
    chapter_prompt = next(p for p in prompts if p.startswith("Summarize all portions"))
    assert "Processing portion 1 of 2" in chapter_prompt and "Processing portion 2 of 2" in chapter_prompt
    assert f"source characters 24000–{len(source)}" in chapter_prompt
    assert "processing boundary never establishes a new contributor" in chapter_prompt
    assert all(e["quote"] == source[e["start_char"]:e["end_char"]] for b in section["batches"] for e in b["evidence"])
    fingerprint = analysis["fingerprint"]
    section["attribution_contexts"] = [{"heading_quote": "Spoofed context"}]
    before = len(prompts)
    await mod.process_section_summary(**options)
    refreshed = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert refreshed["fingerprint"] == fingerprint
    assert "Spoofed context" not in repr(refreshed["sections"][0]["attribution_contexts"])
    assert len(prompts) == before + 1 and "SOURCE:" not in prompts[-1]


@pytest.mark.asyncio
async def test_evidence_author_is_checked_against_its_own_passage_not_mixed_author_batch(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    first = "# Policy\n\n**Retain**\n\n**Alice Example**\n\n" + "Retain the current policy.\n\n" * 60
    second = "**Abolish**\n\n**Bob Example**\n\n" + "Abolish the current policy.\n\n" * 60
    source = first + second
    nodes = [
        {"id": "dsi:1", "title": "Policy", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": len(source)},
        {"id": "keep", "parent_id": "dsi:1", "title": "Retain", "level": 2, "target_type": "char_range", "target_start": source.index("**Retain**"), "target_end": len(first)},
        {"id": "change", "parent_id": "dsi:1", "title": "Abolish", "level": 2, "target_type": "char_range", "target_start": len(first), "target_end": len(source)},
    ]
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": source, "navigation_version": "v1", "page_map": [], "nodes": nodes,
    }))
    _, passages = mod.build_section_extraction_prompt("Policy", "Summarize", source)
    selected = next(p for p in passages if p["start_char"] > len(first) + 40)

    async def chat(**kwargs):
        content = (json.dumps({"summary": "Two proposals", "evidence": [
            {"claim": "Wrong contributor", "source_span_ids": [selected["id"]], "author": "Alice Example"},
            {"claim": "Actual contributor", "source_span_ids": [selected["id"]], "author": "Bob Example"}]})
            if "SOURCE:" in kwargs["messages_payload"][0]["content"] else "A proposal [E1.1.2].")
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings)
    await mod.process_section_summary(**options)
    evidence = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]["sections"][0]["batches"][0]["evidence"]
    assert [item["author"] for item in evidence] == ["", "Bob Example"]
    assert all(item["quote"] == selected["text"] and item["start_char"] == selected["start_char"] for item in evidence)


@pytest.mark.parametrize("source_start", [0, 24000])
def test_extraction_passage_absolute_labels_preserve_source_and_crossing_boundaries(source_start):
    from tldw_Server_API.app.core.Research_Workspace.section_summaries import build_section_extraction_prompt

    source = "a" * 700 + "**Second contributor**" + "b" * 700
    legacy_prompt, legacy_passages = build_section_extraction_prompt("Policy", "Summarize", source)
    prompt, passages = build_section_extraction_prompt("Policy", "Summarize", source, source_start=source_start)
    assert passages == legacy_passages
    assert legacy_prompt.split("SOURCE:\n", 1)[1] == "\n\n".join(f"[{p['id']}]\n{p['text']}" for p in passages)
    assert prompt.split("SOURCE:\n", 1)[1] == "\n\n".join(
        f"[{p['id']}] source characters [{source_start + p['start_char']}, {source_start + p['end_char']})\n{p['text']}"
        for p in passages
    )
    assert any(p["start_char"] < 700 < p["end_char"] for p in passages)
    assert all(p["text"] == source[p["start_char"]:p["end_char"]] for p in passages)


@pytest.mark.parametrize("source_start", [-1, True, 1.5])
def test_extraction_rejects_invalid_absolute_source_start(source_start):
    from tldw_Server_API.app.core.Research_Workspace.section_summaries import build_section_extraction_prompt

    with pytest.raises(ValueError, match="nonnegative integer"):
        build_section_extraction_prompt("Policy", "Summarize", "Source", source_start=source_start)


def test_contextual_extraction_lists_only_current_passage_ids_as_claim_evidence():
    from tldw_Server_API.app.core.Research_Workspace.section_summaries import build_section_extraction_prompt

    context = 'SOURCE HEADING CONTEXT: source_id="11111111-2222-4333-8444-555555555555", node_id="chapter", heading="Author"'
    source = "A source claim mentions P999 as text. " * 40
    default, plain_passages = build_section_extraction_prompt("Policy", "Summarize", source)
    prompt, passages = build_section_extraction_prompt("Policy", "Summarize", source, attribution_context=context, source_start=24000)
    assert passages == plain_passages
    allowed = json.loads(prompt.split("ALLOWED SOURCE PASSAGE IDS: ")[1].split("\n")[0])
    assert allowed == [p["id"] for p in passages]
    assert "P999" not in allowed and "11111111-2222-4333-8444-555555555555" not in allowed
    assert "context, source, or node IDs" in prompt and "attribution only" in prompt
    assert "ALLOWED SOURCE PASSAGE IDS: " not in default
    assert default == build_section_extraction_prompt("Policy", "Summarize", source, attribution_context="")[0]


@pytest.mark.asyncio
async def test_extraction_failure_diagnostics_report_parsed_uuid_ids_and_actual_passage_vocabulary(monkeypatch):
    import hashlib
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    source = "x" * 1500
    _, passages = mod.build_section_extraction_prompt("Policy", "Summarize", source)
    uuid = "11111111-2222-4333-8444-555555555555"
    unknown_ids = [uuid, "P999"] + [f"context-id-{i}" for i in range(10)] + ["z" * 100]
    body = {"summary": "PRIVATE EXTRACTION contains unselected E99.1.1 " + "x" * 2500,
            "evidence": [{"claim": "Valid source claim", "source_span_ids": ["P1"]}]
            + [{"claim": "Unsupported context claim", "source_span_ids": [value]} for value in unknown_ids]}
    raw = json.dumps(body)
    provider = AsyncMock(return_value={"choices": [{"finish_reason": "stop", "message": {"content": raw}}]})
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": source, "navigation_version": "v1", "nodes": [], "page_map": [],
    }))
    monkeypatch.setattr(mod, "perform_chat_api_call_async", provider)
    logs = []
    monkeypatch.setattr(mod.logger, "warning", lambda message, *values: logs.append((message, values)))
    options = args(request().settings.model_copy(update={"sections": []}))
    for _ in range(7):
        await mod.process_section_summary(**options)
    row = options["workspace_db"].row
    analysis = row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    first = analysis["generation_failures"][0]
    assert first["allowed_id_count"] == len(passages)
    assert first["unknown_id_count"] == len(unknown_ids)
    assert first["unknown_ids"] == [value[:64] for value in sorted(unknown_ids)[:8]]
    assert uuid in first["unknown_ids"] and "E99.1.1" not in first["unknown_ids"]
    assert first["unknown_ids_truncated"] is True
    assert first["response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert first["response_preview"] == raw[:2048] and first["response_preview_truncated"] is True
    assert first["error_code"] == "unknown_source_span" and first["stage"] == "extraction"
    assert (first["section_index"], first["portion_index"]) == (1, 1)
    assert len(analysis["generation_failures"]) == 5 and analysis["generation_failures_omitted"] == 2
    assert row["status"] == "failed" and analysis["sections"][0]["batches"][0]["status"] == "failed"
    assert "PRIVATE EXTRACTION" not in row["content"] and "PRIVATE EXTRACTION" not in repr(logs)
    assert provider.await_count == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_reduction", [False, True])
async def test_synthesis_does_not_turn_unassigned_endnotes_into_another_contributor(monkeypatch, needs_reduction):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    source = "# Policy\n\n**Avery Example**\n\n**ENDNOTES**\n\n" + "Source. " * 6000
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": source, "navigation_version": "v1", "page_map": [], "nodes": [
            {"id": "dsi:1", "title": "Policy", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": len(source)},
        ],
    }))
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages_payload"][0]["content"]
        if "SOURCE:" in prompt:
            content = json.dumps({"summary": "Main point. " * (1500 if needs_reduction else 1), "evidence": [
                {"claim": "Source commentary", "author": "Avery Example", "source_span_ids": ["P1"]},
            ]})
        else:
            allowed = json.loads(prompt.split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0])
            content = f"The chapter endnotes discuss source commentary [{allowed[0]}]."
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings)
    await mod.process_section_summary(**options)
    synthesis = [call for call in calls if "SOURCE:" not in call["messages_payload"][0]["content"]]
    assert any("intermediate compression step" in call["system_message"] for call in synthesis) is needs_reduction
    for call in synthesis:
        system = call["system_message"]
        assert "Missing or empty author labels and reference-only ranges" in system
        assert "not evidence of another contributor" in system
        assert "not grounds to deny the chapter byline" in system
        assert "chapter endnotes" in system and "without forcing an author assignment" in system
        assert "quoted or cited voices distinct" in system
    extraction = [call for call in calls if "SOURCE:" in call["messages_payload"][0]["content"]]
    assert extraction and all("Missing or empty author labels" not in call["system_message"] for call in extraction)
    assert "Missing or empty author labels" not in mod.build_section_extraction_prompt("Policy", "Summarize", source)[0]
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert options["workspace_db"].row["status"] == "complete"
    assert all(evidence["author"] == "" for section in analysis["sections"] for batch in section["batches"] for evidence in batch["evidence"])


@pytest.mark.asyncio
async def test_collective_authorship_caveat_reaches_reduction_and_final_synthesis(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    caveat = "I wrote only a small portion. No views should be attributed to any single contributor."
    source = "# Policy\n\n**Avery Example**\n\n" + "Source. " * 6000 + "\n\nAUTHOR'S NOTE: " + caveat
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": source, "navigation_version": "v1", "page_map": [], "nodes": [
            {"id": "dsi:1", "title": "Policy", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": len(source)},
        ],
    }))
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages_payload"][0]["content"]
        if "SOURCE:" in prompt:
            claim = caveat if caveat in prompt.split("SOURCE:", 1)[1] else "The chapter proposes a policy."
            content = json.dumps({"summary": "Main point. " * 1500, "evidence": [{"claim": claim, "source_span_ids": ["P1"]}]})
        else:
            allowed = json.loads(prompt.split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0])
            content = (caveat if caveat in prompt else "Chapter policy.") + f" [{allowed[0]}]"
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings)
    await mod.process_section_summary(**options)
    synthesis = [c for c in calls if "SOURCE:" not in c["messages_payload"][0]["content"]]
    reductions = [c for c in synthesis if "intermediate compression step" in c["system_message"]]
    finals = [c for c in synthesis if c not in reductions]
    assert reductions and any(caveat in c["messages_payload"][0]["content"] for c in reductions)
    assert len(finals) == 2 and all(caveat in c["messages_payload"][0]["content"] for c in finals)
    for call in synthesis:
        system = call["system_message"]
        assert "explicit collective-authorship or disavowal caveats" in system
        assert "byline does not authorize assigning all proposals to that individual" in system
        assert "chapter-level wording" in system and "Keep the caveat visible" in system
        assert "Identify the supplied byline as source attribution" in system
        assert "Organize a single contribution by policy topic" in system
        assert "separate named proponent sections only for source-supported distinct contributions or explicit opposing positions" in system
    assert all("collective-authorship" not in c["system_message"] for c in calls if c not in synthesis)
    assert "collective-authorship" not in mod.build_section_extraction_prompt("Policy", "Summarize", source)[0]
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert options["workspace_db"].row["status"] == "complete"
    assert all(e["author"] == "" for s in analysis["sections"] for b in s["batches"] for e in b["evidence"])


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["single", "siblings", "whole-document"])
async def test_verified_attribution_context_survives_name_dropping_reductions(monkeypatch, scope):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    first = "# Policy\n\n**Avery Example**\n\nSource mentions E1.2.1 as text. " + "First policy. " * 2000
    second = "# Alternative\n\n**Blair Sample**\n\n" + "Second policy. " * 1700
    source = first + (second if scope == "siblings" else "Further policy. " * 1700) + "\n\n**ENDNOTES**\n\nReference only."
    nodes = [{"id": "dsi:1", "title": "Policy", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": len(source)}]
    if scope == "siblings":
        nodes.append({"id": "child", "parent_id": "dsi:1", "title": "Alternative", "level": 2,
                      "target_type": "char_range", "target_start": len(first), "target_end": len(source)})
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": source, "navigation_version": "v1", "page_map": [], "nodes": nodes,
    }))
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages_payload"][0]["content"]
        if "SOURCE:" in prompt:
            content = json.dumps({"summary": "A policy point. " * 1000, "evidence": [
                {"claim": "A source-backed policy", "source_span_ids": ["P1"]},
            ]})
        else:
            allowed = json.loads(prompt.split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0])
            content = "Compressed policy without names " + " ".join(f"[{value}]" for value in allowed)
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    settings = request().settings.model_copy(update={"provider": "openai"})
    if scope == "whole-document":
        settings = settings.model_copy(update={"sections": []})
    options = args(settings)
    await mod.process_section_summary(**options)
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert options["workspace_db"].row["status"] == "complete"
    reductions = [c for c in calls if "intermediate compression step" in c["system_message"]]
    final = next(c for c in calls if c["messages_payload"][0]["content"].startswith("Summarize all portions"))
    assert reductions
    first_allowed = json.loads(reductions[0]["messages_payload"][0]["content"].split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0])
    assert first_allowed == ["E1.1.1"]  # Context cannot authorize another portion's registered evidence.
    contexts = analysis["sections"][0]["attribution_contexts"]
    for call in [*reductions, final]:
        assert "response_format" not in call
        prompt = call["messages_payload"][0]["content"]
        if scope == "whole-document":
            assert not contexts and "SOURCE HEADING CONTEXT" not in prompt
        else:
            rendered = mod.render_attribution_context(contexts)
            assert rendered in prompt
            assert prompt.count("SOURCE HEADING CONTEXT") == 1
            assert "Avery Example" in prompt and "not citable passages" in prompt
            if scope == "siblings":
                assert "Blair Sample" in prompt
                assert [(c["start_char"], c["end_char"]) for c in contexts if c["type"] == "heading"] == [
                    (0, len(first)), (len(first), source.index("**ENDNOTES**")),
                ]
            assert contexts[-1]["type"] == "reference" and contexts[-1]["start_char"] == source.index("**ENDNOTES**")
        allowed = json.loads(prompt.split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0])
        assert all(re.fullmatch(r"E1\.\d+\.1", value) for value in allowed)
    overview = next(c for c in calls if c["messages_payload"][0]["content"].startswith("Synthesize the completed"))
    assert "response_format" not in overview
    assert "SOURCE HEADING CONTEXT" not in overview["messages_payload"][0]["content"]


@pytest.mark.asyncio
async def test_late_literal_author_note_reaches_early_extraction_and_name_dropping_synthesis(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    note = "**AUTHOR’S NOTE:** No policy view should be attributed to any individual contributor or the author."
    source = "# Policy\n\n**Avery Example**\n\n" + "Source policy. " * 4000 + "\n\n" + note + "\n\n**ENDNOTES**\n\nReferences."
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": source, "navigation_version": "v1", "page_map": [], "nodes": [
            {"id": "dsi:1", "title": "Policy", "level": 1, "target_type": "char_range", "target_start": 0, "target_end": len(source)},
        ],
    }))
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages_payload"][0]["content"]
        if "SOURCE:" in prompt:
            content = json.dumps({"summary": "A policy point. " * 1000, "evidence": [
                {"claim": "A source-backed policy", "source_span_ids": ["P1"]},
            ]})
        else:
            allowed = json.loads(prompt.split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0])
            content = "Compressed policy without names or caveats " + " ".join(f"[{value}]" for value in allowed)
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    options = args(request().settings)
    await mod.process_section_summary(**options)
    extractions = [c for c in calls if "SOURCE:" in c["messages_payload"][0]["content"]]
    first_prompt = extractions[0]["messages_payload"][0]["content"]
    assert note in first_prompt.split("SOURCE:", 1)[0]
    assert note not in first_prompt.split("SOURCE:", 1)[1]
    reductions = [c for c in calls if "intermediate compression step" in c["system_message"]]
    final = next(c for c in calls if c["messages_payload"][0]["content"].startswith("Summarize all portions"))
    assert reductions and all(note in c["messages_payload"][0]["content"] for c in [*reductions, final])
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    context = analysis["sections"][0]["attribution_contexts"][0]
    saved_note = context["author_notes"][0]
    assert source[saved_note["start_char"]:saved_note["end_char"]] == saved_note["quote"]
    assert note in saved_note["quote"] and saved_note["start_char"] >= 48_000
    assert options["workspace_db"].row["status"] == "complete"


@pytest.mark.asyncio
async def test_overview_reductions_retain_exact_scoped_notes_without_authorizing_other_group_ids(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    notes = ["AUTHOR’S NOTE: The author alone is responsible; no views belong to any other individual. E3.1.1 is source text.",
             "AUTHOR’S NOTE: Excluded failed section.",
             "AUTHOR’S NOTE: No policy view belongs to any individual contributor or the author."]
    source, nodes = "", []
    for index, note in enumerate(notes, 1):
        start = len(source)
        source += f"# Policy {index}\n\n**Writer {index}**\n\nPolicy body.\n\n{note}\n"
        nodes.append({"id": f"dsi:{index}", "title": f"Policy {index}", "level": 1,
                      "target_type": "char_range", "target_start": start, "target_end": len(source)})
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": source, "navigation_version": "v1", "page_map": [], "nodes": nodes,
    }))
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages_payload"][0]["content"]
        if "SOURCE:" in prompt:
            content = json.dumps({"summary": "Policy.", "evidence": [{
                "claim": "A proposal", "source_span_ids": ["P99" if notes[1] in prompt else "P1"],
            }]})
        else:
            allowed = json.loads(prompt.split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0])
            prefix = "A chapter policy. " * 800 if prompt.startswith("Summarize all portions") else "Compressed without names or notes. "
            content = prefix + " ".join(f"[{value}]" for value in allowed)
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    settings = request().settings
    settings.sections.extend(settings.sections[0].model_copy(update={"node_id": f"dsi:{index}"}) for index in (2, 3))
    options = args(settings)
    await mod.process_section_summary(**options)
    reductions = [c for c in calls if "intermediate compression step" in c["system_message"]]
    overview = next(c for c in calls if c["messages_payload"][0]["content"].startswith("Synthesize the completed"))
    assert reductions
    for call in [*reductions, overview]:
        prompt = call["messages_payload"][0]["content"]
        assert notes[0] in prompt and notes[2] in prompt and notes[1] not in prompt
        assert "Section 1: Policy 1" in prompt and "Section 3: Policy 3" in prompt
        assert prompt.count("SOURCE AUTHOR NOTES") == 1
    first_allowed = json.loads(reductions[0]["messages_payload"][0]["content"].split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0])
    assert first_allowed == ["E1.1.1"]
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert [s["status"] for s in analysis["sections"]] == ["completed", "failed", "completed"]
    assert analysis["synthesis"] and options["workspace_db"].row["status"] == "failed"


@pytest.mark.asyncio
async def test_oversized_overview_author_notes_fail_actionably_and_keep_completed_chapters(monkeypatch):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.Research_Workspace import section_summaries as mod

    source, nodes = "", []
    for index in range(1, 23):
        start = len(source)
        source += f"# Policy {index}\n\n**Writer {index}**\n\nPolicy body.\n\nAUTHOR’S NOTE: " + "x" * 850 + "\n"
        nodes.append({"id": f"dsi:{index}", "title": f"Policy {index}", "level": 1,
                      "target_type": "char_range", "target_start": start, "target_end": len(source)})
    monkeypatch.setattr(mod, "load_navigation_sections", AsyncMock(return_value={
        "content": source, "navigation_version": "v1", "page_map": [], "nodes": nodes,
    }))
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages_payload"][0]["content"]
        if "SOURCE:" in prompt:
            content = json.dumps({"summary": "Policy.", "evidence": [{"claim": "A proposal", "source_span_ids": ["P1"]}]})
        else:
            allowed = json.loads(prompt.split("ALLOWED EVIDENCE IDS: ")[1].split("\n")[0])
            content = "A chapter policy. " + " ".join(f"[{value}]" for value in allowed)
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(mod, "perform_chat_api_call_async", chat)
    settings = request().settings
    settings.sections.extend(settings.sections[0].model_copy(update={"node_id": f"dsi:{index}"}) for index in range(2, 23))
    options = args(settings)
    result = await mod.process_section_summary(**options)
    analysis = options["workspace_db"].row["version_metadata"]["workspaceClient"]["data"]["sectionAnalysis"]
    assert result["partial"] and options["workspace_db"].row["status"] == "failed"
    assert all(s["status"] == "completed" and s["content"] and s["batches"][0]["evidence"] for s in analysis["sections"])
    assert analysis["synthesis_error_code"] == "author_note_context_too_large"
    assert "fewer sections" in analysis["synthesis_error"] and "saved chapters remain" in analysis["synthesis_error"]
    assert not any(c["messages_payload"][0]["content"].startswith("Synthesize the completed") for c in calls)
    assert not analysis.get("generation_failures")
