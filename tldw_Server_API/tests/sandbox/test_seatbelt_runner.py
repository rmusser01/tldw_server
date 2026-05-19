from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from unittest.mock import MagicMock

import pytest

import tldw_Server_API.app.core.Sandbox.runners.seatbelt_runner as seatbelt_module
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType, TrustLevel
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicy, SandboxPolicyConfig
from tldw_Server_API.app.core.Sandbox.runners.seatbelt_runner import SeatbeltRunner
from tldw_Server_API.app.core.Sandbox.streams import get_hub


def test_seatbelt_preflight_defaults_to_trusted_only(monkeypatch) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED", raising=False)

    result = SeatbeltRunner().preflight(network_policy="deny_all")

    assert "trusted" in result.supported_trust_levels
    assert "standard" not in result.supported_trust_levels
    assert "untrusted" not in result.supported_trust_levels


def test_seatbelt_preflight_allows_standard_when_explicitly_enabled(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED", "1")

    result = SeatbeltRunner().preflight(network_policy="deny_all")

    assert "trusted" in result.supported_trust_levels
    assert "standard" in result.supported_trust_levels
    assert "untrusted" not in result.supported_trust_levels


def test_seatbelt_preflight_reports_missing_launcher(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_AVAILABLE", "1")
    monkeypatch.setattr(seatbelt_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        seatbelt_module,
        "vz_host_facts",
        lambda: {"os": "darwin", "arch": "arm64", "apple_silicon": True},
    )
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: False)

    result = SeatbeltRunner().preflight(network_policy="deny_all")

    assert result.available is False
    assert "sandbox_exec_missing" in result.reasons


def test_seatbelt_preflight_rejects_allowlist_even_when_launcher_exists(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_AVAILABLE", "1")
    monkeypatch.setattr(seatbelt_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        seatbelt_module,
        "vz_host_facts",
        lambda: {"os": "darwin", "arch": "arm64", "apple_silicon": True},
    )
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: True)

    result = SeatbeltRunner().preflight(network_policy="allowlist")

    assert result.available is False
    assert "strict_allowlist_not_supported" in result.reasons


def test_seatbelt_fake_run_completes(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC", "1")

    runner = SeatbeltRunner()
    status = runner.start_run(
        run_id="run-seatbelt-1",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image=None,
            command=["echo", "ok"],
            network_policy="deny_all",
            trust_level=TrustLevel.trusted,
        ),
    )

    assert status.phase == RunPhase.completed
    assert status.exit_code == 0


def test_seatbelt_start_run_executes_real_subprocess_and_collects_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC", raising=False)
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: True)

    run_root = tmp_path / "seatbelt-run"
    run_root.mkdir()
    workspace_root = run_root / "workspace"
    control_root = run_root / "control"
    home_root = run_root / "home"
    temp_root = run_root / "tmp"

    monkeypatch.setattr(seatbelt_module.tempfile, "mkdtemp", lambda prefix: str(run_root))

    class _FakePopen:
        def __init__(self, argv, cwd=None, env=None, stdout=None, stderr=None, stdin=None, start_new_session=None):
            assert stdout is not None
            assert stderr is not None
            del stdin, start_new_session
            assert argv[0] == "/usr/bin/sandbox-exec"
            assert argv[1] == "-f"
            profile_path = Path(argv[2])
            assert profile_path.parent == control_root
            assert cwd == str(workspace_root)
            assert env is not None
            assert env["HOME"] == str(home_root)
            assert env["TMPDIR"] == str(temp_root)
            assert (workspace_root / "inputs" / "seed.txt").read_bytes() == b"seed"
            (workspace_root / "generated.txt").write_bytes(b"artifact-data")
            external_artifact = run_root / "outside.txt"
            external_artifact.write_bytes(b"outside-secret")
            (workspace_root / "leak.txt").symlink_to(external_artifact)
            stdout.write(b"stdout-line\n")
            stderr.write(b"stderr-line\n")
            stdout.flush()
            stderr.flush()
            self.pid = 4242
            self.returncode = 0

        def wait(self, timeout=None):
            assert timeout == 7
            return 0

    monkeypatch.setattr(seatbelt_module.subprocess, "Popen", _FakePopen)

    runner = SeatbeltRunner()
    run_id = "run-seatbelt-real-1"
    hub = get_hub()
    hub._buffers.pop(run_id, None)  # type: ignore[attr-defined]

    status = runner.start_run(
        run_id=run_id,
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image="host-local",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            trust_level=TrustLevel.trusted,
            timeout_sec=7,
            files_inline=[("inputs/seed.txt", b"seed")],
            capture_patterns=["*.txt"],
        ),
        session_workspace=None,
    )

    assert status.phase == RunPhase.completed
    assert status.exit_code == 0
    assert status.artifacts == {"generated.txt": b"artifact-data", "inputs/seed.txt": b"seed"}
    frames = list(hub._buffers.get(run_id, []))  # type: ignore[attr-defined]
    assert any(frame.get("type") == "stdout" and "stdout-line" in frame.get("data", "") for frame in frames)
    assert any(frame.get("type") == "stderr" and "stderr-line" in frame.get("data", "") for frame in frames)
    assert not run_root.exists()


def test_seatbelt_start_run_applies_artifact_and_log_caps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Seatbelt runs should share artifact and log-cap resource counters."""
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC", raising=False)
    monkeypatch.setattr(
        SandboxPolicyConfig,
        "from_settings",
        classmethod(lambda cls: cls(max_artifact_file_bytes=5, max_artifact_total_bytes=8)),
    )
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: True)
    monkeypatch.setattr(SeatbeltRunner, "_max_log_bytes", staticmethod(lambda: 5))

    run_root = tmp_path / "seatbelt-cap-run"
    run_root.mkdir()
    workspace_root = run_root / "workspace"

    monkeypatch.setattr(seatbelt_module.tempfile, "mkdtemp", lambda prefix: str(run_root))

    class _CapPopen:
        def __init__(
            self,
            argv: object,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            stdout: BinaryIO | None = None,
            stderr: BinaryIO | None = None,
            stdin: object | None = None,
            start_new_session: bool | None = None,
        ) -> None:
            assert stdout is not None
            assert stderr is not None
            del argv, env, stdin, start_new_session
            assert cwd == str(workspace_root)
            (workspace_root / "small.txt").write_bytes(b"1234")
            (workspace_root / "too-large.txt").write_bytes(b"123456")
            (workspace_root / "would-exceed-total.txt").write_bytes(b"56789")
            stdout.write(b"abcdef")
            stdout.flush()
            self.pid = 5151
            self.returncode = 0

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(seatbelt_module.subprocess, "Popen", _CapPopen)

    run_id = "run-seatbelt-cap-contract"
    hub = get_hub()
    hub.cleanup_run(run_id)

    status = SeatbeltRunner().start_run(
        run_id=run_id,
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image="host-local",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            trust_level=TrustLevel.trusted,
            timeout_sec=7,
            capture_patterns=["*.txt"],
        ),
        session_workspace=None,
    )

    assert status.phase == RunPhase.completed
    assert status.artifacts == {"small.txt": b"1234"}
    assert status.resource_usage["artifact_limit_file_bytes"] == 5
    assert status.resource_usage["artifact_limit_total_bytes"] == 8
    assert status.resource_usage["artifact_files_collected"] == 1
    assert status.resource_usage["artifact_files_skipped"] == 2
    assert status.resource_usage["artifact_skip_file_limit"] == 1
    assert status.resource_usage["artifact_skip_total_limit"] == 1
    assert status.resource_usage["artifact_bytes_collected"] == 4
    assert status.resource_usage["artifact_bytes"] == 4
    assert status.resource_usage["log_limit_bytes"] == 5
    assert status.resource_usage["log_truncated"] == 1


def test_seatbelt_cancelled_run_drops_artifact_counters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Canceled seatbelt runs should not report artifact counters for discarded artifacts."""
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC", raising=False)
    monkeypatch.setattr(
        SandboxPolicyConfig,
        "from_settings",
        classmethod(lambda cls: cls(max_artifact_file_bytes=100, max_artifact_total_bytes=100)),
    )
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: True)

    run_root = tmp_path / "seatbelt-cancel-run"
    run_root.mkdir()
    workspace_root = run_root / "workspace"

    monkeypatch.setattr(seatbelt_module.tempfile, "mkdtemp", lambda prefix: str(run_root))
    monkeypatch.setattr(SeatbeltRunner, "_consume_cancelled", classmethod(lambda cls, run_id: True))

    class _CancelPopen:
        pid = 5252
        returncode = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            stdout = kwargs.get("stdout")
            assert stdout is not None
            (workspace_root / "artifact.txt").write_bytes(b"artifact")
            stdout.write(b"ok")
            stdout.flush()

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(seatbelt_module.subprocess, "Popen", _CancelPopen)

    status = SeatbeltRunner().start_run(
        run_id="run-seatbelt-cancel-counters",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image="host-local",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            trust_level=TrustLevel.trusted,
            timeout_sec=7,
            capture_patterns=["*.txt"],
        ),
        session_workspace=None,
    )

    assert status.phase == RunPhase.killed
    assert status.artifacts is None
    assert "artifact_files_collected" not in status.resource_usage


def test_seatbelt_start_run_times_out_and_cleans_up(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC", raising=False)
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: True)

    run_root = tmp_path / "seatbelt-timeout-run"
    run_root.mkdir()
    workspace_root = run_root / "workspace"

    monkeypatch.setattr(seatbelt_module.tempfile, "mkdtemp", lambda prefix: str(run_root))
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(seatbelt_module.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))

    class _TimeoutPopen:
        _timeout_seen = False

        def __init__(self, argv, cwd=None, env=None, stdout=None, stderr=None, stdin=None, start_new_session=None):
            assert stdout is not None
            assert stderr is not None
            del argv, env, stdin, start_new_session
            assert cwd == str(workspace_root)
            self._stdout = stdout
            self._stderr = stderr
            self.pid = 2121
            self.returncode = None

        def wait(self, timeout=None):
            if not self._timeout_seen:
                self._timeout_seen = True
                self._stdout.write(b"partial-stdout\n")
                self._stderr.write(b"partial-stderr\n")
                self._stdout.flush()
                self._stderr.flush()
                raise seatbelt_module.subprocess.TimeoutExpired(
                    cmd=["/usr/bin/sandbox-exec"],
                    timeout=timeout,
                )
            self.returncode = 124
            return 124

    monkeypatch.setattr(seatbelt_module.subprocess, "Popen", _TimeoutPopen)

    runner = SeatbeltRunner()
    run_id = "run-seatbelt-timeout-1"
    hub = get_hub()
    hub._buffers.pop(run_id, None)  # type: ignore[attr-defined]

    status = runner.start_run(
        run_id=run_id,
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image="host-local",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            trust_level=TrustLevel.trusted,
            timeout_sec=3,
        ),
        session_workspace=None,
    )

    assert status.phase == RunPhase.timed_out
    assert status.message == "execution_timeout"
    frames = list(hub._buffers.get(run_id, []))  # type: ignore[attr-defined]
    assert any(frame.get("type") == "stdout" and "partial-stdout" in frame.get("data", "") for frame in frames)
    assert any(frame.get("type") == "stderr" and "partial-stderr" in frame.get("data", "") for frame in frames)
    assert any(frame.get("type") == "event" and frame.get("event") == "end" and frame.get("data", {}).get("reason") == "execution_timeout" for frame in frames)
    assert killpg_calls == [(2121, seatbelt_module.signal.SIGTERM)]
    assert not run_root.exists()


def test_seatbelt_runner_docstring_covers_constraints() -> None:
    doc = SeatbeltRunner.__doc__ or ""

    assert "trusted" in doc
    assert "best-effort" in doc
    assert "sandbox-exec" in doc


def test_seatbelt_fake_run_logs_publish_failures(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC", "1")

    class _BrokenHub:
        def publish_event(self, run_id: str, event: str, data: dict | None = None) -> None:
            del run_id, event, data
            raise OSError("hub down")

    mock_logger = MagicMock()
    monkeypatch.setattr(seatbelt_module, "get_hub", lambda: _BrokenHub())
    monkeypatch.setattr(seatbelt_module, "logger", mock_logger, raising=False)

    status = SeatbeltRunner().start_run(
        run_id="run-seatbelt-log-1",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image=None,
            command=["echo", "ok"],
            network_policy="deny_all",
            trust_level=TrustLevel.trusted,
        ),
    )

    assert status.phase == RunPhase.completed
    mock_logger.warning.assert_called()


def test_seatbelt_start_run_raises_runtime_unavailable_when_launcher_missing(monkeypatch) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC", raising=False)
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: False)

    with pytest.raises(SandboxPolicy.RuntimeUnavailable) as exc_info:
        SeatbeltRunner().start_run(
            run_id="run-seatbelt-missing-launcher",
            spec=RunSpec(
                session_id=None,
                runtime=RuntimeType.seatbelt,
                base_image="host-local",
                command=["/bin/echo", "ok"],
                network_policy="deny_all",
                trust_level=TrustLevel.trusted,
            ),
        )

    assert exc_info.value.runtime == RuntimeType.seatbelt
    assert exc_info.value.reasons == ["sandbox_exec_missing"]


def test_seatbelt_start_run_rejects_symlinked_session_workspace(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC", raising=False)
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: True)

    run_root = tmp_path / "seatbelt-symlink-source"
    run_root.mkdir()
    monkeypatch.setattr(seatbelt_module.tempfile, "mkdtemp", lambda prefix: str(run_root))

    session_workspace = tmp_path / "session-workspace"
    session_workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (session_workspace / "leak.txt").symlink_to(outside)

    status = SeatbeltRunner().start_run(
        run_id="run-seatbelt-symlink-session",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image="host-local",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            trust_level=TrustLevel.trusted,
        ),
        session_workspace=str(session_workspace),
    )

    assert status.phase == RunPhase.failed
    assert "Refusing symlink workspace entry" in (status.message or "")
    assert not run_root.exists()


def test_seatbelt_start_run_rejects_inline_file_escape_via_symlink(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC", raising=False)
    monkeypatch.setattr(seatbelt_module, "_sandbox_exec_exists", lambda: True)

    run_root = tmp_path / "seatbelt-inline-escape"
    workspace_root = run_root / "workspace"
    workspace_root.mkdir(parents=True)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (workspace_root / "escape").symlink_to(outside_dir, target_is_directory=True)
    monkeypatch.setattr(seatbelt_module.tempfile, "mkdtemp", lambda prefix: str(run_root))

    status = SeatbeltRunner().start_run(
        run_id="run-seatbelt-inline-escape",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image="host-local",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            trust_level=TrustLevel.trusted,
            files_inline=[("escape/payload.txt", b"boom")],
        ),
    )

    assert status.phase == RunPhase.failed
    assert "escapes workspace" in (status.message or "")
    assert not (outside_dir / "payload.txt").exists()
