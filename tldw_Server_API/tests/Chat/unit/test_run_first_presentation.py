from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat.run_first_presentation import present_chat_tools
from tldw_Server_API.tests.run_first_constants import PHASE2C_RUN_FIRST_COHORT


RUN_TOOL = {
    "type": "function",
    "function": {
        "name": "run",
        "description": "Execute shell commands.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
}

NOTES_TOOL = {
    "type": "function",
    "function": {
        "name": "notes.search",
        "description": "Search notes for relevant passages.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}

GEMINI_NATIVE_TOOLS = {
    "function_declarations": [
        {
            "name": "notes.search",
            "description": "Search notes for relevant passages.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
        {
            "name": "run",
            "description": "Execute shell commands.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    ]
}


@pytest.mark.unit
def test_present_chat_tools_orders_run_first_when_eligible() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="gated",
        provider_key="openai:gpt-4o-mini",
        streaming=False,
    )

    assert [tool["function"]["name"] for tool in presented.llm_tools] == ["run", "notes.search"]
    assert presented.effective_tool_names == ["run", "notes.search"]
    assert presented.eligible is True
    assert presented.prompt_fragment is not None
    assert "run(command)" in presented.prompt_fragment
    assert presented.tool_choice in (None, "auto")


@pytest.mark.unit
def test_present_chat_tools_orders_run_first_when_default_on_and_in_cohort() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="default_on",
        provider_key="openai:gpt-4o",
        provider_allowlist=PHASE2C_RUN_FIRST_COHORT,
        streaming=False,
    )

    assert presented.eligible is True
    assert [tool["function"]["name"] for tool in presented.llm_tools] == ["run", "notes.search"]
    assert presented.prompt_fragment is not None


@pytest.mark.unit
def test_present_chat_tools_marks_google_gemini_flash_default_on_when_in_cohort() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="default_on",
        provider_key="google:gemini-2.5-flash",
        provider_allowlist=PHASE2C_RUN_FIRST_COHORT,
        streaming=False,
    )

    assert presented.eligible is True
    assert presented.ineligible_reason is None
    assert [tool["function"]["name"] for tool in presented.llm_tools] == ["run", "notes.search"]
    assert presented.prompt_fragment is not None


@pytest.mark.unit
def test_present_chat_tools_demotes_typed_tools_when_run_is_eligible() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="gated",
        provider_key="openai:gpt-4o-mini",
        streaming=False,
    )

    run_description = presented.llm_tools[0]["function"]["description"]
    typed_description = presented.llm_tools[1]["function"]["description"]

    assert "Preferred first tool" in run_description
    assert typed_description.startswith("Fallback tool:")


@pytest.mark.unit
def test_present_chat_tools_is_ineligible_when_run_absent_after_filtering() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["notes.*"],
        rollout_mode="gated",
        provider_key="openai:gpt-4o-mini",
        streaming=False,
    )

    assert [tool["function"]["name"] for tool in presented.llm_tools] == ["notes.search"]
    assert presented.effective_tool_names == ["notes.search"]
    assert presented.eligible is False
    assert presented.ineligible_reason == "run_missing_after_filtering"
    assert presented.prompt_fragment is None
    assert presented.tool_choice is None


@pytest.mark.unit
def test_present_chat_tools_is_ineligible_when_provider_not_in_rollout_allowlist() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="gated",
        provider_key="openai:gpt-4o-mini",
        provider_allowlist=["anthropic:claude-3-7-sonnet"],
        streaming=False,
    )

    assert presented.eligible is False
    assert presented.ineligible_reason == "provider_not_in_rollout_allowlist"
    assert presented.prompt_fragment is None
    assert [tool["function"]["name"] for tool in presented.llm_tools] == ["run", "notes.search"]


@pytest.mark.unit
def test_present_chat_tools_is_eligible_when_provider_matches_rollout_allowlist() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="gated",
        provider_key="openai:gpt-4o-mini",
        provider_allowlist=["openai:gpt-4o-mini"],
        streaming=False,
    )

    assert presented.eligible is True
    assert presented.ineligible_reason is None
    assert presented.prompt_fragment is not None
    assert [tool["function"]["name"] for tool in presented.llm_tools] == ["run", "notes.search"]


@pytest.mark.unit
def test_present_chat_tools_is_out_of_cohort_when_default_on_provider_mismatches() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="default_on",
        provider_key="openai:gpt-4o-mini",
        provider_allowlist=["anthropic:claude-3-7-sonnet"],
        streaming=False,
    )

    assert presented.eligible is False
    assert presented.ineligible_reason == "provider_not_in_rollout_allowlist"
    assert presented.prompt_fragment is None
    assert [tool["function"]["name"] for tool in presented.llm_tools] == ["run", "notes.search"]


@pytest.mark.unit
def test_present_chat_tools_fails_closed_for_default_on_when_allowlist_is_empty() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="default_on",
        provider_key="openai:gpt-4o-mini",
        provider_allowlist=[],
        streaming=False,
    )

    assert presented.eligible is False
    assert presented.ineligible_reason == "provider_not_in_rollout_allowlist"
    assert presented.prompt_fragment is None


@pytest.mark.unit
def test_present_chat_tools_reorders_gemini_native_declarations_and_tracks_all_names() -> None:
    presented = present_chat_tools(
        tools=[GEMINI_NATIVE_TOOLS],
        allow_catalog=["run", "notes.*"],
        rollout_mode="gated",
        provider_key="openai:gpt-4o-mini",
        provider_allowlist=["openai:gpt-4o-mini"],
        streaming=False,
    )

    assert presented.eligible is True
    assert presented.effective_tool_names == ["run", "notes.search"]
    assert [decl["name"] for decl in presented.llm_tools[0]["function_declarations"]] == [
        "run",
        "notes.search",
    ]
    assert presented.llm_tools[0]["function_declarations"][0]["description"].startswith(
        "Execute shell commands."
    )
    assert presented.llm_tools[0]["function_declarations"][1]["description"].startswith(
        "Fallback tool:"
    )


@pytest.mark.unit
def test_present_chat_tools_does_not_force_tool_choice_to_run() -> None:
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="gated",
        provider_key="openai:gpt-4o-mini",
        streaming=False,
    )

    assert presented.tool_choice in (None, "auto")
