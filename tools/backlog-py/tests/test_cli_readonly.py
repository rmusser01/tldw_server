from pathlib import Path

from click.testing import CliRunner

from backlog_py.cli.main import main


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "repos" / "basic"


def _invoke(*args: str):
    return CliRunner().invoke(main, ["--cwd", str(FIXTURE_REPO), *args])


def test_top_level_help_includes_readonly_commands():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--cwd" in result.output
    assert "task" in result.output
    assert "search" in result.output
    assert "board" in result.output
    assert "config" in result.output


def test_task_list_plain_outputs_task_id():
    result = _invoke("task", "list", "--plain")

    assert result.exit_code == 0
    assert "TASK-1" in result.output
    assert "Example task" in result.output


def test_task_view_plain_outputs_task_body():
    result = _invoke("task", "TASK-1", "--plain")

    assert result.exit_code == 0
    assert "TASK-1" in result.output
    assert "Implement a fixture" in result.output


def test_search_plain_outputs_matching_task():
    result = _invoke("search", "parser preservation", "--plain")

    assert result.exit_code == 0
    assert "TASK-1" in result.output
    assert "Example task" in result.output


def test_board_outputs_status_grouping():
    result = _invoke("board")

    assert result.exit_code == 0
    assert "To Do" in result.output
    assert "In Progress" in result.output
    assert "TASK-1" in result.output
    assert "Done" in result.output


def test_config_list_outputs_safe_defaults():
    result = _invoke("config", "list")

    assert result.exit_code == 0
    assert "projectName: basic-fixture" in result.output
    assert "autoCommit: false" in result.output
    assert "remoteOperations: false" in result.output
