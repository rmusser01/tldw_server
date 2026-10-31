"""Regression tests for evidence provenance in the Email M2 metrics CLI."""

from __future__ import annotations

import importlib.util
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "Helper_Scripts/checks/email_m2_gate_validation.py"
)
_BEFORE = '''email_sync_runs_total{provider="gmail",status="success"} 4
email_sync_lag_seconds_bucket{provider="gmail",le="60"} 4
email_sync_lag_seconds_bucket{provider="gmail",le="+Inf"} 4
'''
_AFTER = '''email_sync_runs_total{provider="gmail",status="success"} 10
email_sync_lag_seconds_bucket{provider="gmail",le="60"} 10
email_sync_lag_seconds_bucket{provider="gmail",le="+Inf"} 10
'''


def _run_cli(tmp_path: Path, metrics: str, *extra: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    """Exercise argument parsing, filesystem input, report writing and exit status."""
    metrics_path = tmp_path / "metrics.prom"
    metrics_path.write_text(metrics, encoding="utf-8")
    report_path = tmp_path / "report.json"
    # Fixed Python executable and local checker; no shell or network input.
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(_SCRIPT_PATH),
            "--metrics-file", str(metrics_path),
            "--output-json", str(report_path),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result, json.loads(report_path.read_text(encoding="utf-8"))


@pytest.mark.unit
@pytest.mark.parametrize(("delta", "success_runs"), [(False, 4), (True, 6)])
def test_offline_success_labels_fixture_without_claiming_staging_validation(
    tmp_path: Path, delta: bool, success_runs: int,
) -> None:
    """A successful fixture must not be mistaken for measured staging evidence."""
    after_path = tmp_path / "after.prom"
    after_path.write_text(_AFTER, encoding="utf-8")
    extra = ["--metrics-file-after", str(after_path)] if delta else []

    result, report = _run_cli(tmp_path, _BEFORE, *extra)

    assert result.returncode == 0, result.stderr
    assert "staging lag SLO validated" not in result.stdout
    assert "offline_fixture" in result.stdout
    assert "staging evidence remains unverified" in result.stdout
    assert report["evidence_source"] == "offline_fixture"
    assert "staging evidence remains unverified" in report["evidence_note"]
    assert report["mode"] == ("delta" if delta else "snapshot")
    assert report["runs"]["success"] == success_runs
    assert report["lag_seconds"] == pytest.approx({"p50": 30.0, "p95": 57.0, "threshold_p50_max": 300.0})
    assert report["passed"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("metrics", "extra", "lag_pass", "runs_pass"),
    [
        (_BEFORE, ["--max-median-lag-seconds", "29"], False, True),
        (_BEFORE, ["--min-success-runs", "5"], True, False),
        ("", [], False, False),
    ],
)
def test_offline_failure_preserves_checks_and_exit_status(
    tmp_path: Path, metrics: str, extra: list[str], lag_pass: bool, runs_pass: bool,
) -> None:
    """Evidence labels must not turn failed threshold checks into a passing gate."""
    result, report = _run_cli(tmp_path, metrics, *extra)

    assert result.returncode == 1
    assert "[FAIL]" in result.stdout
    assert "[PASS]" not in result.stdout
    assert report["evidence_source"] == "offline_fixture"
    assert report["checks"]["lag_slo_pass"] is lag_pass
    assert report["checks"]["success_runs_pass"] is runs_pass
    assert report["passed"] is False


@pytest.mark.unit
@pytest.mark.parametrize(("window", "mode", "success_runs"), [(0, "snapshot", 4), (1, "delta", 6)])
def test_live_input_reports_endpoint_source_and_sampling_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    window: int, mode: str, success_runs: int,
) -> None:
    """Mock only external transport and waiting; exercise the real report path."""
    spec = importlib.util.spec_from_file_location("email_m2_gate_validation", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    snapshots = iter([_BEFORE, _AFTER])
    monkeypatch.setattr(script, "_fetch_metrics_text", lambda **kwargs: next(snapshots))
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)
    report_path = tmp_path / "live.json"

    result = script.main(["--window-seconds", str(window), "--output-json", str(report_path)])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert result == 0
    assert "live_endpoint" in output
    assert "offline fixture" not in output.lower()
    assert report["evidence_source"] == "live_endpoint"
    assert report["mode"] == mode
    assert report["runs"]["success"] == success_runs
    assert report["passed"] is True
