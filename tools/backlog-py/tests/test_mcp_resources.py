from pathlib import Path

import pytest

from backlog_py.mcp.resources import read_resource
from backlog_py.mcp.server import is_mcp_sdk_available, main
from backlog_py.mcp.tools import task_create, task_edit, task_search, task_view
from backlog_py.storage.project import discover_project


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "repos" / "basic"


def _project():
    return discover_project(Path.cwd(), explicit_cwd=FIXTURE_REPO)


def test_workflow_overview_resource_returns_task_workflow_guidance():
    content = read_resource("backlog://workflow/overview")

    assert "Backlog.md" in content
    assert "task" in content.casefold()
    assert "read-only" in content.casefold()


def test_task_workflow_resource_alias_matches_overview():
    overview = read_resource("backlog://workflow/overview")
    alias = read_resource("backlog://docs/task-workflow")

    assert alias == overview


def test_unknown_resource_uri_raises_clear_error():
    with pytest.raises(KeyError, match="Unsupported Backlog MCP resource"):
        read_resource("backlog://unknown")


def test_task_search_returns_fixture_backed_readonly_dicts():
    results = task_search(_project(), "parser preservation")

    assert results == [
        {
            "id": "TASK-1",
            "title": "Example task",
            "status": "In Progress",
            "description": (
                "Implement a fixture that exercises parser preservation behavior.\n"
                "This paragraph must remain untouched by a no-op render."
            ),
            "path": "backlog/tasks/task-1 - Example-task.md",
        }
    ]


def test_task_search_honors_limit():
    assert task_search(_project(), "", limit=0) == []


def test_task_view_returns_fixture_backed_readonly_dict():
    result = task_view(_project(), "task-1")

    assert result["id"] == "TASK-1"
    assert result["title"] == "Example task"
    assert result["status"] == "In Progress"
    assert "Implement a fixture" in result["description"]
    assert result["path"] == "backlog/tasks/task-1 - Example-task.md"
    assert "Trailing unowned body content" in result["raw_source"]


def test_unsupported_mutation_tools_raise_clear_not_implemented_errors():
    with pytest.raises(NotImplementedError, match="Task mutation MCP tools are not implemented until Task 7"):
        task_create(_project(), title="New task")

    with pytest.raises(NotImplementedError, match="Task mutation MCP tools are not implemented until Task 7"):
        task_edit(_project(), task_id="TASK-1", title="Edited task")


def test_server_stub_reports_missing_sdk_without_importing_mcp():
    if is_mcp_sdk_available():
        expected_message = "MCP SDK adapter is not implemented"
    else:
        expected_message = "MCP SDK is not installed"

    with pytest.raises(RuntimeError, match=expected_message):
        main()
