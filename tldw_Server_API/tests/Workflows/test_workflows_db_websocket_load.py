"""Real-network workflow latency harness contract; no live server or user data."""

import contextlib
import json
import os
import signal

# fixed local CLI exercised with the active interpreter.
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Helper_Scripts/benchmarks/workflows_websocket_load.py"


def test_workflow_websocket_load_benchmark_validates_real_delivery(tmp_path):
    assert SCRIPT.is_file(), "The real-network workflow latency benchmark is missing"
    output = tmp_path / "latency.json"
    env_file = tmp_path / "external.env"
    env_file.write_text("TESTING=true\nTLDW_TEST_MODE=true\n")
    # Own a process group so a timed-out harness cannot leave its server running.
    process = subprocess.Popen(  # nosec B603
        [
            sys.executable,
            str(SCRIPT),
            "--clients",
            "1",
            "3",
            "--backlog",
            "0",
            "3",
            "--rounds",
            "2",
            "--trials",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
        env={**os.environ, "TESTING": "true", "TLDW_TEST_MODE": "true", "TLDW_ENV_FILE": str(env_file)},
    )
    try:
        _, stderr = process.communicate(timeout=120)
        assert process.returncode == 0, stderr[-4000:]
    finally:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
        process.wait(timeout=5)
    report = json.loads(output.read_text())
    assert report["valid"] is True
    assert report["server"]["errors"] == {}
    assert report["server"]["test_mode"] is False
    assert report["server"]["explicit_pytest_runtime"] is False
    assert report["transport"] == "loopback TCP / production ASGI app"
    assert len(report["trials"]) == 4
    for trial in report["trials"]:
        assert trial["received"] == trial["clients"] * 2
        assert trial["errors"] == {}
        assert trial["mismatches"] == 0
        assert trial["handshake_ms"]["count"] == trial["clients"]
        assert trial["snapshot_ms"]["p95"] >= trial["handshake_ms"]["p50"]
        assert trial["event_ms"]["count"] == trial["clients"] * 2
        assert trial["loop_lag_ms"]["count"] > 0
        assert trial["history_ms"]["count"] == (trial["clients"] if trial["backlog"] else 0)
    assert report["authorization"] == {"missing": 403, "foreign_owner": 403, "missing_run": 403}


@pytest.mark.parametrize("option", ["--clients", "--rounds", "--trials"])
def test_workflow_websocket_load_benchmark_rejects_empty_workloads(option):
    assert SCRIPT.is_file(), "The real-network workflow latency benchmark is missing"
    # fixed local CLI and literal invalid flags.
    result = subprocess.run(  # nosec B603
        [sys.executable, str(SCRIPT), option, "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "positive" in result.stderr


def test_workflow_websocket_benchmark_does_not_start_after_parent_exit(tmp_path):
    # Direct internal server mode has no children; the timeout kills this process itself.
    try:
        result = subprocess.run(  # nosec B603
            [sys.executable, str(SCRIPT), "--serve", str(tmp_path)],
            cwd=ROOT,
            env={**os.environ, "WORKFLOWS_WS_LOAD_PARENT_PID": "0"},
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Benchmark server stayed alive after its parent exited")
    assert result.returncode == 0
    assert not (tmp_path / "port.json").exists()
