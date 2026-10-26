"""Live GitHub API integration coverage for the git repository adapter.

Opt-in via ``RUN_EXTERNAL_API_TESTS=1``. The default target is a small public
repository with a stable ``README.md`` path so the adapter exercises the real
metadata -> tree -> blob fetch path without requiring private credentials.
"""

from __future__ import annotations

import os

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.external_api]


def _require_external() -> None:
    if os.getenv("RUN_EXTERNAL_API_TESTS", "0") != "1":
        pytest.skip("External API tests disabled. Set RUN_EXTERNAL_API_TESTS=1 to enable.")


def _env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    text = value.strip()
    return text or default


def test_git_repository_snapshot_live_github_api_round_trip():
    """Fetch a real GitHub tree and blob for a public notes-compatible file."""
    _require_external()

    from tldw_Server_API.app.core.Ingestion_Sources.git_repository import (
        build_git_repository_snapshot_with_failures,
        validate_git_repository_source,
    )

    repo_url = _env_or_default(
        "TLDW_GITHUB_EXTERNAL_TEST_REPO_URL",
        "https://github.com/octocat/Spoon-Knife",
    )
    expected_path = _env_or_default(
        "TLDW_GITHUB_EXTERNAL_TEST_EXPECTED_PATH",
        "README.md",
    )
    requested_ref = _env_or_default("TLDW_GITHUB_EXTERNAL_TEST_REF")
    access_token = (
        _env_or_default("TLDW_GITHUB_EXTERNAL_TEST_TOKEN")
        or _env_or_default("GITHUB_TOKEN")
    )

    if not repo_url or not expected_path:
        pytest.fail("External GitHub adapter test requires a repo URL and expected path.")

    config = {
        "mode": "remote_github_repo",
        "repo_url": repo_url,
        "include_globs": [expected_path],
    }
    if requested_ref:
        config["ref"] = requested_ref

    normalized = validate_git_repository_source(config)

    try:
        items, failures = build_git_repository_snapshot_with_failures(
            normalized,
            sink_type="notes",
            access_token=access_token,
        )
    except ValueError as exc:
        # The adapter raises ValueError for every GitHub API failure (TASK-13950.11).
        message = str(exc)
        if any(f"with HTTP {code}" in message for code in (403, 429, 500, 502, 503, 504)):
            pytest.skip(f"GitHub API unavailable or rate limited: {message}")
        if message.startswith("GitHub API request failed: "):
            pytest.skip(f"GitHub API unreachable in this environment: {message}")
        raise

    assert failures == {}  # nosec B101
    assert expected_path in items  # nosec B101

    item = items[expected_path]
    assert item["relative_path"] == expected_path  # nosec B101
    assert item["text"].strip()  # nosec B101
    assert item["source_format"] == expected_path.rsplit(".", 1)[-1].lower()  # nosec B101

    metadata = item["raw_metadata"]
    assert metadata["repo_mode"] == "remote_github_repo"  # nosec B101
    assert metadata["repo_url"] == normalized["repo_url"]  # nosec B101
    assert metadata["repo_relative_path"] == expected_path  # nosec B101
    assert metadata["repo_ref"]  # nosec B101
    assert metadata["repo_blob_sha"]  # nosec B101
