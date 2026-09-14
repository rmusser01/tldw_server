"""Portable orchestration and real child-process cancellation regressions."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess  # nosec B404
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def drill() -> ModuleType:
    """Load the entrypoint under pytest's importlib collection mode."""
    path = Path(__file__).resolve().parents[1] / "scripts/vz-failure-drill.py"
    spec = importlib.util.spec_from_file_location("failure_workflow", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_running(pid: int) -> bool:
    """Treat an exited orphan awaiting system reaping as no longer executing."""
    # Fixed local process query, without a shell.
    result = subprocess.run(  # nosec B603, B607
        ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, check=False
    )
    if result.stderr:
        raise OSError(result.stderr)
    return result.returncode == 0 and bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


@pytest.mark.integration
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cancellation contract")
@pytest.mark.parametrize("mode", ["sigterm", "timeout"])
def test_logged_stops_process_tree_before_unwinding(drill: ModuleType, tmp_path: Path, mode: str) -> None:
    """Parent-only termination or timeout must not leave a descendant writing evidence."""
    ids = tmp_path / "children.json"
    ready = tmp_path / "ready"
    ids_ready = tmp_path / "ids-ready"
    grandchild = "import signal,time,sys; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(sys.argv[1]).touch(); time.sleep(60)"
    child = (
        "import subprocess,signal,sys,os,json,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"p=subprocess.Popen([sys.executable,'-c',{grandchild!r},{str(ready)!r}]); "
        f"Path({str(ids)!r}).write_text(json.dumps([os.getpid(),p.pid])); "
        f"Path({str(ids_ready)!r}).touch(); time.sleep(60)"
    )
    wrapper = """
import runpy,signal,sys
from pathlib import Path
namespace=runpy.run_path(sys.argv[1])
def stop(signum, frame):
    raise KeyboardInterrupt('parent-only signal')
signal.signal(signal.SIGTERM,stop)
try:
    namespace['logged']([sys.executable,'-c',sys.argv[2]],Path(sys.argv[3]),timeout=5 if sys.argv[4]=='timeout' else 30)
except BaseException as exc:
    Path(sys.argv[5]).write_text(type(exc).__name__)
"""
    finished = tmp_path / "finished"
    # Controlled test wrapper, without a shell.
    with subprocess.Popen(  # nosec B603
        [sys.executable, "-c", wrapper, drill.__file__, child, str(tmp_path / "child.log"), mode, str(finished)],
        start_new_session=True,
    ) as parent:
        children = []
        try:
            deadline = time.monotonic() + 4
            while not (ready.exists() and ids_ready.exists()) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert ready.exists() and ids_ready.exists(), "Child fixture did not start"
            children = json.loads(ids.read_text())
            if mode == "sigterm":
                parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=15)
            assert finished.exists(), "Expected logged() cancellation to unwind"
            assert finished.read_text() == ("KeyboardInterrupt" if mode == "sigterm" else "TimeoutExpired")
            assert not any(process_running(pid) for pid in children), "A child outlived workflow cleanup"
        finally:
            # Keep even the RED run from leaking the deliberately uncooperative tree.
            if not children and ids.exists():
                children = json.loads(ids.read_text())
            for pid in children:
                if process_running(pid):
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)


@pytest.mark.integration
@pytest.mark.parametrize("exit_code", [0, 7])
def test_logged_preserves_output_and_exit_code(drill: ModuleType, tmp_path: Path, exit_code: int) -> None:
    """Success and failure both retain combined output and respect cwd/environment."""
    command = [
        sys.executable,
        "-c",
        "import os,sys; print(os.getcwd()); print(os.environ['DRILL_VALUE'],file=sys.stderr); "
        f"sys.exit({exit_code})",
    ]
    log = tmp_path / "command.log"
    assert drill.logged(command, log, cwd=tmp_path, env={**os.environ, "DRILL_VALUE": "stderr-proof"}) == exit_code
    assert set(log.read_text().splitlines()) == {str(tmp_path), "stderr-proof"}


@pytest.mark.integration
@pytest.mark.skipif(os.name != "posix", reason="POSIX process ownership contract")
def test_logged_handles_signal_before_spawn_returns(
    drill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation inside spawn must wait until the child has an assigned owner."""
    popen = subprocess.Popen
    children: list[subprocess.Popen] = []

    def interrupted_spawn(*args: object, **kwargs: object) -> subprocess.Popen:
        """Deliver SIGINT after OS creation but before returning the process handle."""
        child = popen(*args, **kwargs)
        children.append(child)
        signal.raise_signal(signal.SIGINT)
        return child

    original_handler = signal.getsignal(signal.SIGINT)
    monkeypatch.setattr(drill.subprocess, "Popen", interrupted_spawn)
    try:
        with pytest.raises(KeyboardInterrupt):
            drill.logged([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "spawn.log")
        assert signal.getsignal(signal.SIGINT) is original_handler
        assert len(children) == 1
        assert children[0].returncode is not None, "Cancellation failed to reap the newly spawned child"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


@pytest.mark.unit
def test_main_hashes_loaded_dependencies_even_on_failure(
    drill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locally changed materializer or helperctl must change recorded provenance."""
    # main mutates these process settings; restore them after this test.
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", "1")
    source = tmp_path / "source"
    source.mkdir()
    (source / "rootfs.img").write_bytes(b"disk")
    helper = tmp_path / "helper"
    helper.write_bytes(b"helper")
    materializer = SimpleNamespace(
        validate_bundle=lambda path: None,
        read_optional_json=lambda path: {},
        bundle_artifact_names=lambda path: ["rootfs.img"],
    )
    monkeypatch.setattr(drill, "load_module", lambda name, path: materializer)
    monkeypatch.setattr(drill.sys, "platform", "darwin")
    monkeypatch.setattr(drill.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(drill, "logged", lambda *args, **kwargs: 1)
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client",
        SimpleNamespace(MacOSVirtualizationHelperClient=object),
    )
    evidence = tmp_path / "evidence"
    assert (
        drill.main(
            [
                "--allow-fault-injection",
                "--source-bundle",
                str(source),
                "--helper",
                str(helper),
                "--evidence-dir",
                str(evidence),
            ]
        )
        == 1
    )
    receipt = json.loads((evidence / "receipt.json").read_text())
    for relative in (
        "tools/vz-linux-image/scripts/prepare-smoke-bundle.py",
        "tools/macos-vz-helper/scripts/vz-helperctl.py",
    ):
        assert receipt["input_sha256"].get(relative) == drill.digest(drill.REPO / relative)
    assert receipt["source_before"] == receipt["source_after"]
    assert (evidence / "error.log").is_file()
