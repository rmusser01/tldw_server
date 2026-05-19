from pathlib import Path

import pytest

from backlog_py.storage.config import load_config
from backlog_py.storage.project import discover_project


def test_discovers_folder_local_config(tmp_path):
    (tmp_path / "backlog").mkdir()
    (tmp_path / "backlog" / "config.yml").write_text(
        "project_name: demo\nremote_operations: false\n",
        encoding="utf-8",
    )

    project = discover_project(tmp_path)

    assert project.root == tmp_path
    assert project.backlog_dir == tmp_path / "backlog"
    assert project.config.remote_operations is False


def test_backlog_cwd_overrides_process_cwd(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    (project_root / "backlog").mkdir(parents=True)
    (project_root / "backlog" / "config.yml").write_text(
        "project_name: env-demo\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BACKLOG_CWD", str(project_root))

    project = discover_project(tmp_path)

    assert project.root == project_root
    assert project.config.project_name == "env-demo"


def test_discovers_root_config_file(tmp_path):
    (tmp_path / "backlog.config.yml").write_text(
        "project_name: root-demo\n",
        encoding="utf-8",
    )

    project = discover_project(tmp_path)

    assert project.root == tmp_path
    assert project.backlog_dir == tmp_path / "backlog"
    assert project.config_path == tmp_path / "backlog.config.yml"
    assert project.config.project_name == "root-demo"


def test_discovers_dot_backlog_config(tmp_path):
    (tmp_path / ".backlog").mkdir()
    (tmp_path / ".backlog" / "config.yml").write_text(
        "project_name: dot-demo\n",
        encoding="utf-8",
    )

    project = discover_project(tmp_path)

    assert project.root == tmp_path
    assert project.backlog_dir == tmp_path / ".backlog"
    assert project.config_path == tmp_path / ".backlog" / "config.yml"
    assert project.config.project_name == "dot-demo"


def test_load_config_accepts_snake_case_keys(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: snake-demo",
                "default_status: In Progress",
                "remote_operations: false",
                "auto_commit: true",
                "bypass_git_hooks: true",
                "check_active_branches: false",
                "active_branch_days: 14",
                "definition_of_done:",
                "  - Tests pass",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project_name == "snake-demo"
    assert config.default_status == "In Progress"
    assert config.remote_operations is False
    assert config.auto_commit is True
    assert config.bypass_git_hooks is True
    assert config.check_active_branches is False
    assert config.active_branch_days == 14
    assert config.definition_of_done == ["Tests pass"]


def test_load_config_accepts_camel_case_keys(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "projectName: camel-demo",
                "defaultStatus: Done",
                "remoteOperations: false",
                "autoCommit: true",
                "bypassGitHooks: true",
                "checkActiveBranches: false",
                "activeBranchDays: 7",
                "definitionOfDone:",
                "  - Review complete",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.project_name == "camel-demo"
    assert config.default_status == "Done"
    assert config.remote_operations is False
    assert config.auto_commit is True
    assert config.bypass_git_hooks is True
    assert config.check_active_branches is False
    assert config.active_branch_days == 7
    assert config.definition_of_done == ["Review complete"]


def test_load_config_supports_no_git_style_flags(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: no-git-demo",
                "remote_operations: false",
                "auto_commit: false",
                "check_active_branches: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.remote_operations is False
    assert config.auto_commit is False
    assert config.check_active_branches is False


def test_load_config_rejects_string_boolean_values(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project_name: malformed-bool-demo",
                'remote_operations: "false"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="remote_operations"):
        load_config(config_path)


def test_explicit_cwd_takes_precedence_over_backlog_cwd(tmp_path, monkeypatch):
    env_root = tmp_path / "env"
    explicit_root = tmp_path / "explicit"
    (env_root / "backlog").mkdir(parents=True)
    (explicit_root / "backlog").mkdir(parents=True)
    (env_root / "backlog" / "config.yml").write_text(
        "project_name: env-demo\n",
        encoding="utf-8",
    )
    (explicit_root / "backlog" / "config.yml").write_text(
        "project_name: explicit-demo\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BACKLOG_CWD", str(env_root))

    project = discover_project(tmp_path, explicit_cwd=explicit_root)

    assert project.root == explicit_root
    assert project.config.project_name == "explicit-demo"
