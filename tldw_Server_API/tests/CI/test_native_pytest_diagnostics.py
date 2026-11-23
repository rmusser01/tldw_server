"""Native pytest diagnostics must observe failures, never change their outcome."""

from __future__ import annotations

import os
import subprocess  # nosec B404 -- Isolated pytest child processes.
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "Helper_Scripts/ci/native_pytest.py"
NATIVE_JOBS = (
    "full-suite-macos-312-shards",
    "full-suite-windows-312-shards",
    "full-suite-os-313-release-shards",
    "full-suite-windows-313-release-shards",
)


@pytest.mark.parametrize("succeeds", [True, False])
def test_native_runner_preserves_pytest_result_and_arguments(tmp_path: Path, succeeds: bool) -> None:
    test_path = tmp_path / "test_sample.py"
    test_path.write_text(f"def test_sample():\n    assert {succeeds!r}\n", encoding="utf-8")
    report = tmp_path / "result with spaces.xml"
    result = subprocess.run(  # nosec B603 -- Fixed interpreter and repository script.
        [sys.executable, str(RUNNER), "--noconftest", "-q", str(test_path), f"--junitxml={report}"],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == (0 if succeeds else 1), result.stdout + result.stderr
    assert report.exists()
    assert f"[native-pytest] pytest-return exit={result.returncode}" in result.stdout


@pytest.mark.parametrize("hang_phase", ["test", "interpreter"])
def test_native_runner_observes_captured_tests_and_interpreter_exit(tmp_path: Path, hang_phase: str) -> None:
    """Real test/exit hangs remain hangs, with diagnostics outside pytest capture."""
    (tmp_path / "test_sample.py").write_text(
        "import threading\ndef test_sample():\n    canary = 'private-diagnostic-canary'\n    threading.Event().wait(60)\n"
        if hang_phase == "test" else "def test_sample():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "hang_plugin.py").write_text(
        "import threading\n"
        "def hold():\n"
        "    canary = 'private-diagnostic-canary'\n"
        "    threading.Event().wait(60)\n"
        "def pytest_sessionfinish(session, exitstatus):\n"
        + ("    threading.Thread(target=hold).start()\n" if hang_phase == "interpreter" else "    pass\n"),
        encoding="utf-8",
    )
    program = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "from Helper_Scripts.ci.native_pytest import main; "
        "raise SystemExit(main(sys.argv[1:], traceback_interval=0.1))"
    )
    output, errors = tmp_path / "stdout.log", tmp_path / "stderr.log"
    with output.open("w", encoding="utf-8") as stdout, errors.open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen(  # nosec B603 -- Fixed interpreter, local synthetic test/plugin.
            [sys.executable, "-c", program, "--noconftest", "-q", "-p", "hang_plugin", "test_sample.py"],
            cwd=tmp_path,
            env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": str(tmp_path)},
            stdout=stdout, stderr=stderr,
        )
        try:
            deadline = time.monotonic() + 15
            expected_frame = "in test_sample" if hang_phase == "test" else "_shutdown"
            while time.monotonic() < deadline:
                if expected_frame in errors.read_text(encoding="utf-8") or child.poll() is not None:
                    break
                time.sleep(0.02)
            assert child.poll() is None, output.read_text() + errors.read_text()
            assert ("[native-pytest] pytest-return exit=0" in output.read_text()) == (hang_phase == "interpreter")
            diagnostic = errors.read_text(encoding="utf-8")
            assert "[native-pytest] thread-dump" in diagnostic and expected_frame in diagnostic and "wait" in diagnostic
            if hang_phase == "interpreter":
                assert "in hold" in diagnostic
            assert "private-diagnostic-canary" not in diagnostic
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


def test_native_stack_dump_excludes_live_locals(tmp_path: Path) -> None:
    """The inspected caller frame really contains the value that must remain private."""
    from Helper_Scripts.ci.native_pytest import _dump_thread_stacks

    diagnostic_canary = "private-diagnostic-canary"
    output = tmp_path / "live-local.log"
    with output.open("wb") as stream:
        _dump_thread_stacks(stream.fileno())
    diagnostic = output.read_text(encoding="utf-8")
    assert "in test_native_stack_dump_excludes_live_locals" in diagnostic
    assert diagnostic_canary not in diagnostic


def test_native_workflow_preserves_limits_paths_and_timeout_backends() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    for name in NATIVE_JOBS:
        job = workflow["jobs"][name]
        assert job["timeout-minutes"] == (60 if "312" in name else 35)
        steps = [step for step in job["steps"] if "--junit-xml=" in step.get("run", "")]
        assert len(steps) == 2
        for step in steps:
            assert "python Helper_Scripts/ci/native_pytest.py -q" in step["run"]
            assert '-m "not jobs and not e2e"' in step["run"]
            assert not step.get("continue-on-error", False)
        ordinary = steps[0]["run"]
        assert "$TEST_PATHS" in ordinary and "--durations=50" in ordinary
        if name == "full-suite-macos-312-shards":
            assert "-p pytest_timeout -o timeout_method=signal" in ordinary
        elif name == "full-suite-windows-312-shards":
            assert '"${EXTRA_PYTEST_ARGS[@]}"' in ordinary
        else:
            assert "-p pytest_timeout -o timeout_method=${{ runner.os == 'Windows' && 'thread' || 'signal' }}" in ordinary
        legacy = steps[1]["run"]
        assert "TLDW_DISABLE_LEGACY_MEDIA=1 python" in legacy
        assert "tldw_Server_API/tests/MediaIngestion_NEW" in legacy and "--durations=25" in legacy
