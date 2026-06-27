from __future__ import annotations

import json
from pathlib import Path

import pytest

from Helper_Scripts.cats_fuzz.summary import CatsRunSummary, mask_command, write_summary


@pytest.mark.unit
def test_mask_command_hides_api_key() -> None:
    masked = mask_command(["cats", "-H", "X-API-KEY=secret-value"])

    assert "secret-value" not in " ".join(masked)
    assert "X-API-KEY=$X-API-KEY" in masked


@pytest.mark.unit
def test_mask_command_hides_authorization_header() -> None:
    masked = mask_command(["cats", "-H", "Authorization=Bearer secret-value"])

    assert "secret-value" not in " ".join(masked)
    assert "Authorization=$Authorization" in masked


@pytest.mark.unit
def test_write_summary_persists_expected_shape(tmp_path: Path) -> None:
    summary = CatsRunSummary(
        block="public-read",
        cats_version="13.8.0",
        openapi_sha256="abc",
        command=["cats", "--blackbox"],
        masked_command=["cats", "--blackbox"],
        exit_code=0,
        failure_class="ok",
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        report_dir="report",
    )

    output = write_summary(summary, tmp_path / "summary.json")
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data["block"] == "public-read"
    assert data["failure_class"] == "ok"
    assert data["command"] == ["cats", "--blackbox"]


@pytest.mark.unit
def test_write_summary_never_persists_raw_api_key(tmp_path: Path) -> None:
    summary = CatsRunSummary(
        block="public-read",
        cats_version="13.8.0",
        openapi_sha256="abc",
        command=["cats", "-H", "X-API-KEY=secret-value"],
        masked_command=["cats", "-H", "X-API-KEY=secret-value"],
        exit_code=1,
        failure_class="api",
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        report_dir="report",
    )

    output = write_summary(summary, tmp_path / "summary.json")
    raw_json = output.read_text(encoding="utf-8")
    data = json.loads(raw_json)

    assert "secret-value" not in raw_json
    assert data["command"] == ["cats", "-H", "X-API-KEY=$X-API-KEY"]
    assert data["masked_command"] == ["cats", "-H", "X-API-KEY=$X-API-KEY"]
