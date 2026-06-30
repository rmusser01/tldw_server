from pathlib import Path

from backlog_py.core.repository import ReadOnlyRepository


FIXTURE_REPO = Path(__file__).parent / "fixtures" / "repos" / "basic"


def _snapshot_files(root: Path) -> dict[Path, str]:
    return {
        path.relative_to(root): path.read_text(encoding="utf-8")
        for path in sorted((root / "backlog").rglob("*"))
        if path.is_file()
    }


def test_repository_lists_tasks_from_fixture_repo():
    repository = ReadOnlyRepository.from_path(FIXTURE_REPO)

    tasks = repository.list_tasks()

    assert [task.id for task in tasks] == ["TASK-1"]
    assert tasks[0].title == "Example task"
    assert tasks[0].status == "In Progress"


def test_repository_views_task_by_id():
    repository = ReadOnlyRepository.from_path(FIXTURE_REPO)

    task = repository.get_task("TASK-1")

    assert task.id == "TASK-1"
    assert "Implement a fixture" in task.description
    assert task.path.name == "task-1 - Example-task.md"


def test_repository_searches_title_and_body():
    repository = ReadOnlyRepository.from_path(FIXTURE_REPO)

    title_matches = repository.search_tasks("example")
    body_matches = repository.search_tasks("parser preservation")

    assert [task.id for task in title_matches] == ["TASK-1"]
    assert [task.id for task in body_matches] == ["TASK-1"]


def test_repository_groups_board_by_status():
    repository = ReadOnlyRepository.from_path(FIXTURE_REPO)

    board = repository.board()

    assert list(board) == ["To Do", "In Progress", "Done"]
    assert board["To Do"] == []
    assert [task.id for task in board["In Progress"]] == ["TASK-1"]
    assert board["Done"] == []


def test_readonly_operations_do_not_change_backlog_files():
    before = _snapshot_files(FIXTURE_REPO)
    repository = ReadOnlyRepository.from_path(FIXTURE_REPO)

    repository.list_tasks()
    repository.get_task("TASK-1")
    repository.search_tasks("fixture")
    repository.board()

    assert _snapshot_files(FIXTURE_REPO) == before


def test_repository_from_path_ignores_backlog_cwd_environment(tmp_path, monkeypatch):
    env_backlog = tmp_path / "backlog"
    env_tasks = env_backlog / "tasks"
    env_tasks.mkdir(parents=True)
    (env_backlog / "config.yml").write_text("projectName: env-demo\n", encoding="utf-8")
    (env_tasks / "task-99 - Env.md").write_text(
        "---\nid: TASK-99\ntitle: Env task\nstatus: To Do\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BACKLOG_CWD", str(tmp_path))

    tasks = ReadOnlyRepository.from_path(FIXTURE_REPO).list_tasks()

    assert [task.id for task in tasks] == ["TASK-1"]


def test_repository_sorts_dotted_task_ids(tmp_path):
    backlog_dir = tmp_path / "backlog"
    task_dir = backlog_dir / "tasks"
    task_dir.mkdir(parents=True)
    (backlog_dir / "config.yml").write_text(
        "projectName: dotted\n",
        encoding="utf-8",
    )
    (task_dir / "task-2.1 - Child.md").write_text(
        "---\nid: TASK-2.1\ntitle: Child\nstatus: To Do\n---\n",
        encoding="utf-8",
    )
    (task_dir / "task-10 - Later.md").write_text(
        "---\nid: TASK-10\ntitle: Later\nstatus: To Do\n---\n",
        encoding="utf-8",
    )

    tasks = ReadOnlyRepository.from_path(tmp_path).list_tasks()

    assert [task.id for task in tasks] == ["TASK-2.1", "TASK-10"]
