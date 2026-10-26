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


# ---------------------------------------------------------------------------
# Remote GitHub sync through core.http_client (TASK-13950.11)
# ---------------------------------------------------------------------------

_REPO_API_URL = "https://api.github.com/repos/example/project"
_TREE_URL = "https://api.github.com/repos/example/project/git/trees/main?recursive=1"
_BLOB_URL = "https://api.github.com/repos/example/project/git/blobs/sha-alpha"
_ALPHA_TREE = {
    "tree": [{"path": "notes/alpha.md", "type": "blob", "url": _BLOB_URL, "size": 14, "sha": "sha-alpha"}]
}
_ALPHA_BLOB = {"content": base64.b64encode(b"# Alpha\n\nBody\n").decode("ascii"), "encoding": "base64"}


def _route_github_api_through_mock_transport(monkeypatch, handler):
    """Send every GitHub API GET through the real ``stream_response`` over a MockTransport.

    Egress checks run for real; sockets and DNS are blocked, so a urlopen or raw-client
    fetch fails loudly instead of reaching the network.
    """
    import socket

    import httpx

    import tldw_Server_API.app.core.Ingestion_Sources.git_repository as git_repository
    from tldw_Server_API.app.core import http_client

    def _no_real_network(*_args, **_kwargs):
        raise AssertionError("git_repository GitHub sync bypassed core.http_client")

    monkeypatch.setattr(socket, "create_connection", _no_real_network)
    monkeypatch.setattr(socket.socket, "connect", _no_real_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_real_network)
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    client = http_client.create_client(transport=httpx.MockTransport(handler))

    def _stream_via_mock(**kwargs):
        return http_client.stream_response(client=client, **kwargs)

    monkeypatch.setattr(git_repository, "stream_response", _stream_via_mock, raising=False)
    return client


def _sync_remote(monkeypatch, handler):
    import tldw_Server_API.app.core.Ingestion_Sources.git_repository as git_repository

    client = _route_github_api_through_mock_transport(monkeypatch, handler)
    try:
        return git_repository.build_git_repository_snapshot_with_failures(
            {"mode": "remote_github_repo", "repo_url": "https://github.com/example/project", "root_subpath": "notes"},
            sink_type="notes",
            access_token="token-123",
        )
    finally:
        client.close()


def _serve(routes, seen):
    """MockTransport handler: record each request, answer from ``routes`` (url -> response)."""

    def handler(request):
        seen.append(request)
        return routes[str(request.url)]

    return handler


@pytest.mark.unit
def test_remote_github_sync_fetches_through_http_client(monkeypatch):
    """The repo, tree and blob GETs carry the same headers and timeout as before."""
    import httpx

    seen: list[httpx.Request] = []
    routes = {
        _REPO_API_URL: httpx.Response(200, json={"default_branch": "main"}),
        _TREE_URL: httpx.Response(200, json=_ALPHA_TREE),
        _BLOB_URL: httpx.Response(200, json=_ALPHA_BLOB),
    }

    items, failures = _sync_remote(monkeypatch, _serve(routes, seen))

    assert failures == {}
    assert items["alpha.md"]["text"] == "# Alpha\n\nBody\n"
    assert items["alpha.md"]["raw_metadata"]["repo_blob_sha"] == "sha-alpha"
    assert [str(r.url) for r in seen] == [_REPO_API_URL, _TREE_URL, _BLOB_URL]
    for request in seen:
        assert request.headers["Authorization"] == "Bearer token-123"
        assert request.headers["Accept"] == "application/vnd.github+json"
        assert request.headers["User-Agent"] == "tldw-server-ingestion-sources"
        assert request.extensions["timeout"] == {"connect": 30.0, "read": 30.0, "write": 30.0, "pool": 30.0}


@pytest.mark.unit
def test_remote_github_sync_follows_a_redirect_within_api_github_com(monkeypatch):
    """A renamed repository's 301 to api.github.com/repositories/<id> is still followed."""
    import httpx

    moved = "https://api.github.com/repositories/42"
    seen: list[httpx.Request] = []
    routes = {
        _REPO_API_URL: httpx.Response(301, headers={"Location": moved}),
        moved: httpx.Response(200, json={"default_branch": "main"}),
        _TREE_URL: httpx.Response(200, json={"tree": []}),
    }

    items, failures = _sync_remote(monkeypatch, _serve(routes, seen))

    assert (items, failures) == ({}, {})
    assert [str(r.url) for r in seen] == [_REPO_API_URL, moved, _TREE_URL]
    assert seen[1].headers["Authorization"] == "Bearer token-123"


@pytest.mark.unit
@pytest.mark.parametrize(
    "location",
    ["http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:8080/", "https://example.com/collect"],
)
def test_remote_github_sync_refuses_a_redirect_off_api_github_com(monkeypatch, location):
    """A redirect to metadata, loopback or any other host is refused before it is requested."""
    import httpx

    seen: list[httpx.Request] = []
    routes = {_REPO_API_URL: httpx.Response(302, headers={"Location": location})}

    with pytest.raises(ValueError, match="GitHub API HTTPS URLs"):
        _sync_remote(monkeypatch, _serve(routes, seen))

    assert [str(r.url) for r in seen] == [_REPO_API_URL]


@pytest.mark.unit
def test_remote_github_sync_refuses_an_egress_denied_api_host(monkeypatch):
    """The operator egress denylist now applies to the GitHub API host."""
    import httpx

    seen: list[httpx.Request] = []
    monkeypatch.setenv("EGRESS_DENYLIST", "api.github.com")

    with pytest.raises(ValueError, match="denylist"):
        _sync_remote(monkeypatch, _serve({}, seen))

    assert seen == []


@pytest.mark.unit
def test_remote_github_sync_refuses_api_host_resolving_to_metadata(monkeypatch):
    """If api.github.com resolves to link-local metadata, the token is never sent."""
    import httpx

    from tldw_Server_API.app.core import http_client
    from tldw_Server_API.app.core.Security import egress

    # Outside pytest the egress check resolves hostnames; make it do so here.
    monkeypatch.setattr(http_client, "is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(http_client, "env_flag_enabled", lambda _name: False)
    monkeypatch.setattr(http_client, "is_test_mode", lambda: False)
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda _host, **_kwargs: ["169.254.169.254"])
    seen: list[httpx.Request] = []

    with pytest.raises(ValueError, match="private or reserved"):
        _sync_remote(monkeypatch, _serve({}, seen))

    assert seen == []


@pytest.mark.unit
def test_remote_github_blob_read_stops_at_the_size_cap(monkeypatch):
    """An oversized blob body is refused while streaming, not after reading it all."""
    import httpx

    chunks = [b"x" * 16384] * 20
    yielded: list[int] = []

    class _CountingStream(httpx.SyncByteStream):
        def __iter__(self):
            for chunk in chunks:
                yielded.append(len(chunk))
                yield chunk

    seen: list[httpx.Request] = []
    routes = {
        _REPO_API_URL: httpx.Response(200, json={"default_branch": "main"}),
        _TREE_URL: httpx.Response(200, json=_ALPHA_TREE),
        _BLOB_URL: httpx.Response(200, stream=_CountingStream()),
    }

    items, failures = _sync_remote(monkeypatch, _serve(routes, seen))

    assert items == {}
    assert "exceeded size limit" in failures["alpha.md"]["error"]
    assert len(yielded) < len(chunks)


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["http_404", "connect_error"])
def test_remote_github_blob_fetch_failure_is_recorded_not_raised(monkeypatch, failure):
    """A failed blob GET becomes a failed item; the worker's error contract (ValueError) holds."""
    import httpx

    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        url = str(request.url)
        if url == _REPO_API_URL:
            return httpx.Response(200, json={"default_branch": "main"})
        if url == _TREE_URL:
            return httpx.Response(200, json=_ALPHA_TREE)
        if failure == "connect_error":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(404, json={"message": "Not Found"})

    items, failures = _sync_remote(monkeypatch, handler)

    assert items == {}
    assert set(failures) == {"alpha.md"}
    expected = "HTTP 404" if failure == "http_404" else "connection refused"
    assert expected in failures["alpha.md"]["error"]


@pytest.mark.unit
@pytest.mark.parametrize(("failure", "status_code"), [("http_429", 429), ("connect_error", None)])
def test_github_api_failures_carry_their_http_status(monkeypatch, failure, status_code):
    """GitHubAPIError is a ValueError (the worker's contract) that keeps the status for callers."""
    import httpx

    import tldw_Server_API.app.core.Ingestion_Sources.git_repository as git_repository

    def handler(request):
        if failure == "connect_error":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(429, json={"message": "rate limited"})

    client = _route_github_api_through_mock_transport(monkeypatch, handler)
    try:
        with pytest.raises(git_repository.GitHubAPIError) as raised:
            git_repository._github_api_get_json(_REPO_API_URL, access_token=None)
    finally:
        client.close()

    assert raised.value.status_code == status_code
    assert isinstance(raised.value, ValueError)
