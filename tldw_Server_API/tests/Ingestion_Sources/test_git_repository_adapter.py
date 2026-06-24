from __future__ import annotations

import base64
from pathlib import Path
import subprocess

import pytest


@pytest.mark.unit
def test_git_repository_adapter_accepts_local_repo_within_allowed_roots(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Sources.git_repository import (
        validate_git_repository_source,
    )

    allowed_root = tmp_path / "allowed"
    repo_dir = allowed_root / "notes-repo"
    (repo_dir / ".git").mkdir(parents=True)

    monkeypatch.setenv("INGESTION_SOURCE_ALLOWED_ROOTS", str(allowed_root))

    config = validate_git_repository_source(
        {
            "mode": "local_repo",
            "path": str(repo_dir),
            "ref": " main ",
            "root_subpath": "/docs/notes/",
            "include_globs": [" docs/**/*.md ", "", "README.md"],
            "exclude_globs": "archive/**\n*.tmp",
            "respect_gitignore": False,
        }
    )

    assert config["mode"] == "local_repo"
    assert config["path"] == str(Path(repo_dir).resolve(strict=False))
    assert config["ref"] == "main"
    assert config["root_subpath"] == "docs/notes"
    assert config["include_globs"] == ["docs/**/*.md", "README.md"]
    assert config["exclude_globs"] == ["archive/**", "*.tmp"]
    assert config["respect_gitignore"] is False


@pytest.mark.unit
def test_git_repository_adapter_normalizes_remote_github_repo_config():
    from tldw_Server_API.app.core.Ingestion_Sources.git_repository import (
        validate_git_repository_source,
    )

    config = validate_git_repository_source(
        {
            "mode": "remote_github_repo",
            "repo_url": "https://github.com/example/project.git/",
            "ref": " feature/notes ",
            "root_subpath": "notes/imports/",
            "account_id": "12",
        }
    )

    assert config == {
        "mode": "remote_github_repo",
        "repo_url": "https://github.com/example/project",
        "repo_owner": "example",
        "repo_name": "project",
        "ref": "feature/notes",
        "root_subpath": "notes/imports",
        "account_id": 12,
    }


@pytest.mark.unit
def test_git_repository_adapter_rejects_non_github_remote_url():
    from tldw_Server_API.app.core.Ingestion_Sources.git_repository import (
        validate_git_repository_source,
    )

    with pytest.raises(ValueError, match="GitHub"):
        validate_git_repository_source(
            {
                "mode": "remote_github_repo",
                "repo_url": "https://gitlab.com/example/project",
            }
        )


@pytest.mark.unit
def test_git_repository_snapshot_respects_gitignore_and_root_subpath(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Sources.git_repository import (
        build_git_repository_snapshot_with_failures,
    )

    allowed_root = tmp_path / "allowed"
    repo_dir = allowed_root / "notes-repo"
    notes_dir = repo_dir / "docs" / "notes"
    notes_dir.mkdir(parents=True)
    subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
    (repo_dir / ".gitignore").write_text("docs/notes/ignored.md\n", encoding="utf-8")
    (notes_dir / "alpha.md").write_text("# Alpha\n\nBody\n", encoding="utf-8")
    (notes_dir / "ignored.md").write_text("# Ignored\n\nSkip\n", encoding="utf-8")
    (notes_dir / "todo.txt").write_text("Checklist\n", encoding="utf-8")
    (notes_dir / "image.png").write_bytes(b"not-a-note")

    monkeypatch.setenv("INGESTION_SOURCE_ALLOWED_ROOTS", str(allowed_root))

    items, failures = build_git_repository_snapshot_with_failures(
        {
            "mode": "local_repo",
            "path": str(repo_dir),
            "root_subpath": "docs/notes",
            "respect_gitignore": True,
        },
        sink_type="notes",
    )

    assert failures == {}
    assert set(items) == {"alpha.md", "todo.txt"}
    assert items["alpha.md"]["raw_metadata"]["repo_relative_path"] == "docs/notes/alpha.md"
    assert items["todo.txt"]["text"] == "Checklist\n"


@pytest.mark.unit
def test_git_repository_snapshot_loads_remote_github_tree(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Sources.git_repository as git_repository

    seen_calls: list[tuple[str, str | None]] = []

    def _fake_github_api_get_json(
        url: str,
        *,
        access_token: str | None,
        accept: str = "application/vnd.github+json",
        max_response_bytes: int | None = None,
    ):
        seen_calls.append((url, access_token))
        if url == "https://api.github.com/repos/example/project":
            return {"default_branch": "main"}
        if "/git/trees/main?recursive=1" in url:
            return {
                "tree": [
                    {
                        "path": "notes/alpha.md",
                        "type": "blob",
                        "url": "https://api.github.com/blob-alpha",
                        "size": 14,
                        "sha": "sha-alpha",
                    },
                    {
                        "path": "notes/image.png",
                        "type": "blob",
                        "url": "https://api.github.com/blob-image",
                        "size": 4,
                        "sha": "sha-image",
                    },
                ]
            }
        if url == "https://api.github.com/blob-alpha":
            return {
                "content": base64.b64encode(b"# Alpha\n\nBody\n").decode("ascii"),
                "encoding": "base64",
            }
        raise AssertionError(f"Unexpected GitHub API request: {url}")

    monkeypatch.setattr(git_repository, "_github_api_get_json", _fake_github_api_get_json)

    items, failures = git_repository.build_git_repository_snapshot_with_failures(
        {
            "mode": "remote_github_repo",
            "repo_url": "https://github.com/example/project",
            "root_subpath": "notes",
        },
        sink_type="notes",
        access_token="token-123",
    )

    assert failures == {}
    assert set(items) == {"alpha.md"}
    assert items["alpha.md"]["raw_metadata"]["repo_ref"] == "main"
    assert items["alpha.md"]["raw_metadata"]["repo_blob_sha"] == "sha-alpha"
    assert seen_calls[0] == ("https://api.github.com/repos/example/project", "token-123")


@pytest.mark.unit
def test_git_repository_snapshot_rejects_truncated_remote_tree(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Sources.git_repository as git_repository

    def _fake_github_api_get_json(
        url: str,
        *,
        access_token: str | None,
        accept: str = "application/vnd.github+json",
        max_response_bytes: int | None = None,
    ):
        if url == "https://api.github.com/repos/example/project":
            return {"default_branch": "main"}
        if "/git/trees/main?recursive=1" in url:
            return {
                "truncated": True,
                "tree": [],
            }
        raise AssertionError(f"Unexpected GitHub API request: {url}")

    monkeypatch.setattr(git_repository, "_github_api_get_json", _fake_github_api_get_json)

    with pytest.raises(ValueError, match="truncated"):
        git_repository.build_git_repository_snapshot_with_failures(
            {
                "mode": "remote_github_repo",
                "repo_url": "https://github.com/example/project",
            },
            sink_type="notes",
            access_token="token-123",
        )


@pytest.mark.unit
def test_git_repository_snapshot_marks_oversized_remote_blob_as_failure(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Sources.git_repository as git_repository

    seen_calls: list[str] = []

    def _fake_github_api_get_json(
        url: str,
        *,
        access_token: str | None,
        accept: str = "application/vnd.github+json",
        max_response_bytes: int | None = None,
    ):
        seen_calls.append(url)
        if url == "https://api.github.com/repos/example/project":
            return {"default_branch": "main"}
        if "/git/trees/main?recursive=1" in url:
            return {
                "tree": [
                    {
                        "path": "notes/huge.md",
                        "type": "blob",
                        "url": "https://api.github.com/blob-huge",
                        "size": 10 * 1024 * 1024,
                        "sha": "sha-huge",
                    }
                ]
            }
        if url == "https://api.github.com/blob-huge":
            return {
                "content": base64.b64encode(b"# huge\n").decode("ascii"),
                "encoding": "base64",
            }
        raise AssertionError(f"Unexpected GitHub API request: {url}")

    monkeypatch.setattr(git_repository, "_github_api_get_json", _fake_github_api_get_json)

    items, failures = git_repository.build_git_repository_snapshot_with_failures(
        {
            "mode": "remote_github_repo",
            "repo_url": "https://github.com/example/project",
            "root_subpath": "notes",
        },
        sink_type="notes",
        access_token="token-123",
    )

    assert items == {}
    assert failures == {
        "huge.md": {
            "relative_path": "huge.md",
            "source_format": "md",
            "size": 10 * 1024 * 1024,
            "modified_at": failures["huge.md"]["modified_at"],
            "error": failures["huge.md"]["error"],
        }
    }
    assert "too large" in failures["huge.md"]["error"].lower()
    assert "https://api.github.com/blob-huge" not in seen_calls


@pytest.mark.unit
def test_git_repository_snapshot_marks_oversized_local_file_as_failure(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Sources.git_repository import (
        build_git_repository_snapshot_with_failures,
    )

    allowed_root = tmp_path / "allowed"
    repo_dir = allowed_root / "notes-repo"
    repo_dir.mkdir(parents=True)
    subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
    (repo_dir / "too-large.md").write_text("12345", encoding="utf-8")

    monkeypatch.setenv("INGESTION_SOURCE_ALLOWED_ROOTS", str(allowed_root))
    monkeypatch.setenv("INGESTION_SOURCES_LOCAL_FILE_MAX_BYTES", "4")

    items, failures = build_git_repository_snapshot_with_failures(
        {
            "mode": "local_repo",
            "path": str(repo_dir),
            "respect_gitignore": False,
        },
        sink_type="notes",
    )

    assert items == {}
    assert set(failures) == {"too-large.md"}
    assert "exceeds local file size limit" in failures["too-large.md"]["error"]


@pytest.mark.unit
def test_git_ls_files_timeout_is_reported_as_value_error(tmp_path, monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Sources.git_repository as git_repository

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    def _timeout_run(*args, **kwargs):
        del args, kwargs
        raise subprocess.TimeoutExpired(cmd="git ls-files", timeout=1)

    monkeypatch.setenv("INGESTION_SOURCES_GIT_LS_FILES_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(git_repository.subprocess, "run", _timeout_run)
    git_ls_files = getattr(git_repository, "_git_ls_files")

    with pytest.raises(ValueError, match="Timed out enumerating git repository files"):
        git_ls_files(
            repo_root=repo_dir,
            root_subpath=None,
            respect_gitignore=True,
        )
