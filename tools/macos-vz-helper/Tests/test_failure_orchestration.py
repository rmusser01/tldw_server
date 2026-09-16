"""Portable behavioral tests for fault preparation and eight-case orchestration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


@pytest.fixture
def drill() -> ModuleType:
    """Import the operator script without invoking its host-only entrypoint."""
    path = Path(__file__).resolve().parents[1] / "scripts/vz-failure-drill.py"
    spec = importlib.util.spec_from_file_location("failure_orchestration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def materializer(drill: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the real image-store adapter while restoring import state after use."""
    name = "orchestration_materializer"
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.syspath_prepend(str(drill.REPO))
    return drill.load_module(name, drill.REPO / "tools/vz-linux-image/scripts/prepare-smoke-bundle.py")


@pytest.fixture
def source_bundle(tmp_path: Path) -> Path:
    """Create tiny boot artifacts with manifest-selected, nondefault names."""
    source = tmp_path / "source"
    source.mkdir()
    for name, content in {"vmlinuz": b"kernel", "rootfs.img": b"healthy", "initramfs": b"initrd"}.items():
        (source / name).write_bytes(content)
    (source / "manifest.json").write_text(
        json.dumps({"kernel": "vmlinuz", "rootfs": "rootfs.img", "initrd": "initramfs"}), encoding="utf-8"
    )
    return source


@pytest.mark.unit
def test_load_module_executes_and_registers_dataclasses(
    drill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Register before execution so postponed dataclass annotations resolve."""
    path = tmp_path / "dependency.py"
    path.write_text(
        '"""Local dependency fixture."""\n'
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Record:\n"
        '    """Prove that the dynamically imported dependency executed."""\n'
        "    value: int = 42\n",
        encoding="utf-8",
    )
    name = "orchestration_dependency"
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    loaded = drill.load_module(name, path)
    assert sys.modules[name] is loaded
    assert loaded.Record().value == 42
    assert Path(loaded.__file__) == path


@pytest.mark.unit
@pytest.mark.parametrize("content", [b"", b"\x00\xff\n" * 100_000])
def test_digest_hashes_binary_artifacts(drill: ModuleType, tmp_path: Path, content: bytes) -> None:
    """Hash all binary bytes, including empty artifacts, without text decoding."""
    artifact = tmp_path / "artifact"
    artifact.write_bytes(content)
    assert drill.digest(artifact) == hashlib.sha256(content).hexdigest()


@pytest.mark.unit
def test_write_json_persists_readable_receipt(drill: ModuleType, tmp_path: Path) -> None:
    """Receipts retain nested data, UTF-8 text, indentation, and a final newline."""
    path = tmp_path / "receipt.json"
    value = {"profile": "readiness", "details": [None, True, {"label": "caf\u00e9"}]}
    drill.write_json(path, value)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == value
    assert text.startswith('{\n  "profile":')
    assert text.endswith("\n")


@pytest.mark.unit
@pytest.mark.parametrize("profile", ["mismatch", "readiness", "protocol", "workspace"])
@pytest.mark.parametrize("build_fails", [False, True])
def test_build_agent_uses_overlay_and_retains_build_evidence(
    drill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str, build_fails: bool
) -> None:
    """The compiler sees only an overlay; canonical inputs survive success and failure."""
    production = (drill.REPO / "tools/tldw-agent/internal/guest/vsock_client.go").read_bytes()
    repo = tmp_path / "repo"
    original = repo / "tools/tldw-agent/internal/guest/vsock_client.go"
    original.parent.mkdir(parents=True)
    original.write_bytes(production)
    monkeypatch.setattr(drill, "REPO", repo)
    packet = tmp_path / "packet"
    packet.mkdir()
    binary = packet / "tldw-agent-guest"

    def compile_guest(command: list[str], log: Path, *, cwd: Path, env: dict[str, str]) -> int:
        """Inspect the compiler boundary and produce a tiny artifact or failure log."""
        assert cwd == repo / "tools/tldw-agent"
        assert command == [
            "go",
            "build",
            "-overlay=" + str(packet / "overlay.json"),
            "-o",
            str(binary),
            "./cmd/tldw-agent-guest",
        ]
        assert {key: env[key] for key in ("GOOS", "GOARCH", "CGO_ENABLED", "GOPROXY", "GOSUMDB", "GOTOOLCHAIN")} == {
            "GOOS": "linux",
            "GOARCH": "arm64",
            "CGO_ENABLED": "0",
            "GOPROXY": "off",
            "GOSUMDB": "off",
            "GOTOOLCHAIN": "local",
        }
        assert json.loads((packet / "overlay.json").read_text()) == {
            "Replace": {str(original): str(packet / "vsock_client.go")}
        }
        assert original.read_bytes() == production
        log.write_text("compiler failed" if build_fails else "compiled", encoding="utf-8")
        if not build_fails:
            binary.write_bytes(b"linux-arm64-test-agent")
        return int(build_fails)

    compiler = Mock(side_effect=compile_guest)
    monkeypatch.setattr(drill, "logged", compiler)
    if build_fails:
        with pytest.raises(RuntimeError, match=f"{profile} guest build failed"):
            drill.build_agent(profile, packet)
        assert not (packet / "build.json").exists()
        assert (packet / "build.log").read_text() == "compiler failed"
    else:
        assert drill.build_agent(profile, packet) == binary
        receipt = json.loads((packet / "build.json").read_text())
        assert receipt == {
            "profile": profile,
            "command": compiler.call_args.args[0],
            "production_source_sha256": hashlib.sha256(production).hexdigest(),
            "overlay_sha256": hashlib.sha256((packet / "vsock_client.go").read_bytes()).hexdigest(),
            "agent_sha256": hashlib.sha256(b"linux-arm64-test-agent").hexdigest(),
        }
    compiler.assert_called_once()
    assert original.read_bytes() == production
    overlay = (packet / "vsock_client.go").read_bytes()
    if profile == "mismatch":
        assert b'return []string{"exec", "output_cap_v1"}' not in overlay
        assert (
            overlay.replace(b'return []string{"output_cap_v1"}', b'return []string{"exec", "output_cap_v1"}')
            == production
        )
    elif profile == "protocol":
        assert b"ProtocolVersion: testProtocolVersion," in overlay
        injection = (drill.FIXTURES / "protocol-mismatch.go.txt").read_bytes()
        assert overlay.count(injection) == 1
        assert (
            overlay.replace(injection, b"").replace(
                b"ProtocolVersion: testProtocolVersion,", b"ProtocolVersion: ProtocolVersion,"
            )
            == production
        )
    elif profile == "workspace":
        injection = (drill.FIXTURES / "workspace-mismatch.go.txt").read_bytes()
        assert overlay.count(injection) == 1
        assert (
            overlay.replace(injection, b"").replace(
                b"WorkspaceRoot:   testWorkspaceRoot,", b"WorkspaceRoot:   c.cfg.WorkspaceRoot,"
            )
            == production
        )
    else:
        injection = (drill.FIXTURES / "withhold-ready.go.txt").read_bytes()
        assert overlay.count(injection) == 1
        assert overlay.replace(injection, b"") == production


@pytest.mark.unit
@pytest.mark.parametrize("with_provenance", [False, True])
def test_fingerprint_includes_selected_artifacts_and_optional_metadata(
    drill: ModuleType, materializer: ModuleType, source_bundle: Path, with_provenance: bool
) -> None:
    """Fingerprint boot inputs and available provenance, not unrelated bundle files."""
    names = ["vmlinuz", "rootfs.img", "initramfs", "manifest.json"]
    if with_provenance:
        (source_bundle / "build-info.json").write_text('{"builder":"fixture"}', encoding="utf-8")
        names.append("build-info.json")
    (source_bundle / "notes.txt").write_text("not a boot input", encoding="utf-8")
    assert drill.fingerprint(materializer, source_bundle) == {
        name: hashlib.sha256((source_bundle / name).read_bytes()).hexdigest() for name in names
    }


@pytest.mark.unit
def test_clone_materializes_independent_image_store_runs_and_tracks_disks(
    drill: ModuleType, materializer: ModuleType, source_bundle: Path, tmp_path: Path
) -> None:
    """Real image-store clones retain metadata and register only their own disk paths."""
    (source_bundle / "build-info.json").write_text('{"builder":"fixture"}', encoding="utf-8")
    before = {path.name: path.read_bytes() for path in source_bundle.iterdir()}
    disks = [tmp_path / "previous.img"]
    first = drill.clone(materializer, source_bundle, tmp_path, "first", "healthy", disks)
    second = drill.clone(materializer, source_bundle, tmp_path, "second", "healthy", disks)
    assert first == tmp_path / "image-store/runs/first/bundle"
    assert second == tmp_path / "image-store/runs/second/bundle"
    assert disks == [tmp_path / "previous.img", first / "rootfs.img", second / "rootfs.img"]
    for bundle in (first, second):
        assert {path.name: path.read_bytes() for path in bundle.iterdir()} == before
        manifest = json.loads((bundle.parent / "manifest.json").read_text())
        assert manifest["template_id"] == "vz_linux:healthy"
    (first / "rootfs.img").write_bytes(b"fault")
    assert (second / "rootfs.img").read_bytes() == b"healthy"
    assert {path.name: path.read_bytes() for path in source_bundle.iterdir()} == before


@pytest.mark.unit
@pytest.mark.parametrize("outcome", ["success", "exit", "hash", "exec", "create", "terminate"])
def test_install_agent_verifies_offline_copy_and_terminates_preparer(
    drill: ModuleType, tmp_path: Path, outcome: str
) -> None:
    """Installation logs survive failures and every allocated preparer is terminated."""
    boot, target, packet = (tmp_path / name for name in ("boot", "offline", "mismatch-prepare"))
    for directory in (boot, target, packet):
        directory.mkdir()
    binary = tmp_path / "agent"
    binary.write_bytes(b"fault-agent")
    helper = Mock()
    helper.create_vm.return_value = SimpleNamespace(vm_id="preparer-vm")
    helper.terminate_vm.return_value = outcome != "terminate"
    if outcome == "create":
        helper.create_vm.side_effect = OSError("create unavailable")

    def exec_guest(*, vm_id: str, request: dict[str, Any]) -> SimpleNamespace:
        """Emulate only guest execution, including its installed-binary readback."""
        assert vm_id == "preparer-vm"
        assert request == {
            "argv": ["/bin/sh", "-eu", "-c", (drill.FIXTURES / "install-agent.sh").read_text()],
            "cwd": "/workspace",
            "timeout_sec": 90,
        }
        assert (target / "fault-agent").read_bytes() == b"fault-agent"
        if outcome == "exec":
            raise OSError("exec unavailable")
        (target / "installed-agent").write_bytes(b"wrong-agent" if outcome == "hash" else b"fault-agent")
        return SimpleNamespace(
            stdout=b"installation output\n", stderr=b"diagnostic\n", exit_code=int(outcome == "exit")
        )

    helper.exec_guest.side_effect = exec_guest
    errors = {
        "exit": (RuntimeError, "offline installation failed"),
        "hash": (RuntimeError, "offline installation failed"),
        "exec": (OSError, "exec unavailable"),
        "create": (OSError, "create unavailable"),
        "terminate": (RuntimeError, "preparer termination not confirmed"),
    }
    if outcome == "success":
        drill.install_agent(helper, boot, target, binary, packet)
    else:
        error_type, message = errors[outcome]
        with pytest.raises(error_type, match=message):
            drill.install_agent(helper, boot, target, binary, packet)
    helper.create_vm.assert_called_once_with(
        {
            "runtime": "vz_linux",
            "template": str(boot),
            "workspace_path": str(target),
            "workspace_mount": "virtiofs",
            "network_policy": "deny_all",
            "timeout_sec": 90,
            "owner": "vz-failure-drill",
            "run_id": "mismatch-prepare-offline-preparer",
        }
    )
    if outcome == "create":
        helper.exec_guest.assert_not_called()
        helper.terminate_vm.assert_not_called()
        assert not (packet / "preparer.json").exists()
    else:
        helper.exec_guest.assert_called_once()
        helper.terminate_vm.assert_called_once_with("preparer-vm")
        assert json.loads((packet / "preparer.json").read_text()) == {
            "vm_id": "preparer-vm",
            "boot": str(boot),
            "offline_target": str(target),
        }
        if outcome != "exec":
            assert (packet / "install.stdout.log").read_bytes() == b"installation output\n"
            assert (packet / "install.stderr.log").read_bytes() == b"diagnostic\n"


@pytest.mark.unit
@pytest.mark.parametrize("termination_raises", [False, True])
def test_install_agent_preserves_primary_error_when_cleanup_also_fails(
    drill: ModuleType, tmp_path: Path, termination_raises: bool
) -> None:
    """Keep the exec error primary and retain termination diagnostics in its traceback."""
    binary = tmp_path / "agent"
    binary.write_bytes(b"fault-agent")
    helper = Mock()
    helper.create_vm.return_value = SimpleNamespace(vm_id="preparer-vm")
    helper.exec_guest.side_effect = OSError("original execution failure")
    helper.terminate_vm.return_value = False
    if termination_raises:
        helper.terminate_vm.side_effect = OSError("termination RPC failed")
    with pytest.raises(OSError, match="original execution failure") as caught:
        drill.install_agent(helper, tmp_path / "boot", tmp_path, binary, tmp_path)
    helper.terminate_vm.assert_called_once_with("preparer-vm")
    assert any("preparer cleanup failed" in note for note in caught.value.__notes__)


@pytest.mark.unit
@pytest.mark.parametrize(
    "profile,negative,blocked_by",
    [
        ("mismatch", False, None),
        ("readiness", False, None),
        ("mismatch", True, None),
        ("readiness", True, None),
        ("protocol", False, None),
        ("protocol", True, None),
        ("protocol", True, "report"),
        ("protocol", True, "proof"),
        ("workspace", False, None),
        ("workspace", True, None),
        ("workspace", True, "report"),
        ("workspace", True, "proof"),
        ("readiness", True, "report"),
        ("readiness", True, "proof"),
        ("readiness", False, "vms"),
        ("readiness", False, "disks"),
    ],
)
def test_run_case_parses_reports_and_combines_acceptance_gates(
    drill: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    negative: bool,
    blocked_by: str | None,
) -> None:
    """A real parsed report must agree with dispatch proof, VM inventory, and disk checks."""
    filename, test_name, failure, image_env, opt_in = {
        "mismatch": (
            "test_vz_linux_guest_mismatch_host_gated.py",
            "test_vz_linux_rejects_real_guest_missing_exec_then_runs_healthy_session",
            "Mismatched guest was not rejected",
            "TLDW_SANDBOX_VZ_LINUX_MISMATCH_BASE_IMAGE",
            "TLDW_SANDBOX_VZ_LINUX_GUEST_MISMATCH_DRILL",
        ),
        "readiness": (
            "test_vz_linux_readiness_host_gated.py",
            "test_vz_linux_readiness_timeout_then_healthy_session_reuse",
            "Readiness withholding did not fail",
            "TLDW_SANDBOX_VZ_LINUX_READINESS_BASE_IMAGE",
            "TLDW_SANDBOX_VZ_LINUX_READINESS_DRILL",
        ),
        "protocol": (
            "test_vz_linux_protocol_host_gated.py",
            "test_vz_linux_protocol_mismatch_then_healthy_session_reuse",
            "Protocol mismatch was not rejected",
            "TLDW_SANDBOX_VZ_LINUX_PROTOCOL_BASE_IMAGE",
            "TLDW_SANDBOX_VZ_LINUX_PROTOCOL_DRILL",
        ),
        "workspace": (
            "test_vz_linux_workspace_host_gated.py",
            "test_vz_linux_workspace_mismatch_then_healthy_session_reuse",
            "Workspace mismatch was not rejected",
            "TLDW_SANDBOX_VZ_LINUX_WORKSPACE_BASE_IMAGE",
            "TLDW_SANDBOX_VZ_LINUX_WORKSPACE_DRILL",
        ),
    }[profile]
    dirty_keys = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC")
    for key in dirty_keys:
        monkeypatch.setenv(key, "inherited-value")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("ORCHESTRATION_SENTINEL", "retained")
    healthy, fault, socket_path, binary = (tmp_path / name for name in ("healthy", "fault", "helper.sock", "helper"))

    def logged(command: list[str], log: Path, *, env: dict[str, str]) -> int:
        """Replace pytest execution with its report, checking the isolated environment."""
        assert command == [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"tldw_Server_API/tests/sandbox/{filename}::{test_name}",
            "--timeout=90",
            "--timeout-method=signal",
            "--basetemp=" + str(tmp_path / "pytest"),
            "--junitxml=" + str(tmp_path / "host.xml"),
        ] + (["-p", f"vz_{profile}_negative_control"] if negative else [])
        assert not any(key in env for key in dirty_keys)
        expected = {
            "TEST_MODE": "0",
            "TLDW_SANDBOX_VZ_LINUX_E2E": "1",
            opt_in: "1",
            "TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE": str(healthy),
            image_env: str(fault),
            "TLDW_SANDBOX_MACOS_HELPER_SOCKET": str(socket_path),
            "TLDW_SANDBOX_MACOS_HELPER_BINARY": str(binary),
            "PYTHONPATH": os.pathsep.join([str(drill.REPO), str(drill.FIXTURES)]),
            "ORCHESTRATION_SENTINEL": "retained",
        }
        assert {key: env[key] for key in expected} == expected
        message = "unrelated boot failure" if blocked_by == "report" else f"Failed: {failure}: completed"
        outcome = f'<failure message="{message}"/>' if negative else ""
        (tmp_path / "host.xml").write_text(
            f'<testsuites><testsuite><testcase name="{test_name}">{outcome}</testcase></testsuite></testsuites>',
            encoding="utf-8",
        )
        log.write_text("pytest output", encoding="utf-8")
        return int(negative)

    command_runner = Mock(side_effect=logged)
    proof = Mock(return_value={"ok": blocked_by != "proof", "vm_id": "fault-vm"})
    disk_probe = Mock(return_value={"ok": blocked_by != "disks"})
    helper = Mock()
    helper.list_vms.return_value = SimpleNamespace(
        vms=[SimpleNamespace(vm_id="remaining")] if blocked_by == "vms" else []
    )
    monkeypatch.setattr(drill, "logged", command_runner)
    monkeypatch.setattr(drill, "negative_execution", proof)
    monkeypatch.setattr(drill, "disk_handles", disk_probe)
    result = drill.run_case(helper, socket_path, binary, profile, negative, healthy, fault, tmp_path)
    assert result["ok"] is (blocked_by is None)
    assert {key: result[key] for key in ("tests", "exit_code", "failure", "error", "skipped")} == {
        "tests": 1,
        "exit_code": int(negative),
        "failure": int(negative),
        "error": 0,
        "skipped": 0,
    }
    assert json.loads((tmp_path / "result.json").read_text()) == result
    assert result["remaining_vm_ids"] == (["remaining"] if blocked_by == "vms" else [])
    command_runner.assert_called_once()
    helper.list_vms.assert_called_once()
    disk_probe.assert_called_once_with([healthy / "rootfs.img", fault / "rootfs.img"])
    if negative:
        proof.assert_called_once_with(tmp_path, profile)
        assert result["execution_proof"]["ok"] is (blocked_by != "proof")
    else:
        proof.assert_not_called()
        assert "execution_proof" not in result
    assert all(os.environ[key] == "inherited-value" for key in dirty_keys)
    assert os.environ["TEST_MODE"] == "1"


@dataclass
class HelperStatus:
    """Supply the dataclass result expected by the real managed-helper lifecycle."""

    ok: bool = True


@pytest.mark.unit
@pytest.mark.parametrize("failure", [None, "rejected", "raised"])
def test_exercise_isolates_eight_cases_and_unwinds_failed_runs(
    drill: ModuleType,
    materializer: ModuleType,
    source_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    """Independent real clones prevent case contamination and failures exit managed cleanup."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    source_before = {path.name: path.read_bytes() for path in source_bundle.iterdir()}
    receipt: dict[str, Any] = {"errors": [], "source_before": drill.fingerprint(materializer, source_bundle)}
    helper = Mock()
    helper.list_vms.return_value = SimpleNamespace(vms=[])
    helper.terminate_vm.return_value = True
    ctl = Mock()
    ctl.collect_check_results.return_value = [("preflight", HelperStatus())]
    ctl.start_helper.return_value = HelperStatus()
    ctl.stop_helper.return_value = HelperStatus()
    ctl.socket_accepts_connection.return_value = False
    factory = Mock(return_value=helper)
    seen: list[tuple[str, bool, Path, Path]] = []

    def build_agent(profile: str, packet: Path) -> Path:
        """Provide build artifacts without a compiler; retain real preparation flow."""
        binary = packet / "agent"
        binary.write_bytes(profile.encode())
        (packet / "build.json").write_text(json.dumps({"profile": profile}), encoding="utf-8")
        return binary

    def install_agent(client: Any, boot: Path, target: Path, binary: Path, packet: Path) -> None:
        """Change only the offline fault-source disk, never its separate boot clone."""
        assert client is helper
        assert boot != target
        assert (boot / "rootfs.img").read_bytes() == b"healthy"
        (target / "rootfs.img").write_bytes(binary.read_bytes())

    def run_case(
        client: Any,
        socket_path: Path,
        binary: Path,
        profile: str,
        negative: bool,
        healthy: Path,
        fault: Path,
        packet: Path,
    ) -> dict[str, Any]:
        """Mutate each run disk so later cases expose any accidental clone reuse."""
        assert client is helper
        assert socket_path == factory.call_args.args[0]
        assert binary == tmp_path / "helper"
        assert (healthy / "rootfs.img").read_bytes() == b"healthy"
        assert (fault / "rootfs.img").read_bytes() == profile.encode()
        assert packet.name == profile + ("-negative" if negative else "-positive")
        seen.append((profile, negative, healthy, fault))
        (healthy / "rootfs.img").write_bytes(b"used-healthy")
        (fault / "rootfs.img").write_bytes(b"used-fault")
        if failure and len(seen) == 2:
            helper.list_vms.side_effect = [
                SimpleNamespace(vms=[SimpleNamespace(vm_id="leftover")]),
                SimpleNamespace(vms=[]),
            ]
            if failure == "raised":
                raise OSError("case interrupted")
            return {"ok": False}
        return {"ok": True}

    builder, installer = Mock(side_effect=build_agent), Mock(side_effect=install_agent)
    disk_probe = Mock(return_value={"ok": True})
    monkeypatch.setattr(drill, "build_agent", builder)
    monkeypatch.setattr(drill, "install_agent", installer)
    monkeypatch.setattr(drill, "run_case", run_case)
    monkeypatch.setattr(drill, "disk_handles", disk_probe)
    if failure:
        error = OSError if failure == "raised" else RuntimeError
        with pytest.raises(error, match="case interrupted|failed acceptance"):
            drill.exercise(materializer, ctl, source_bundle, tmp_path / "helper", evidence, receipt, factory)
    else:
        drill.exercise(materializer, ctl, source_bundle, tmp_path / "helper", evidence, receipt, factory)
    expected = [
        (profile, negative)
        for profile in ("mismatch", "readiness", "protocol", "workspace")
        for negative in (False, True)
    ]
    assert [(profile, negative) for profile, negative, _, _ in seen] == (expected[:2] if failure else expected)
    run_paths = [path for _, _, healthy, fault in seen for path in (healthy, fault)]
    assert len(set(run_paths)) == 2 * len(seen)
    assert all(path.is_relative_to(evidence / "image-store/runs") for path in run_paths)
    assert builder.call_count == 4
    assert [call.args[0] for call in builder.call_args_list] == ["mismatch", "readiness", "protocol", "workspace"]
    assert installer.call_count == (1 if failure else 4)
    expected_cases = {
        "mismatch-positive": {"ok": True},
        "mismatch-negative": {"ok": failure != "rejected"},
        "readiness-positive": {"ok": True},
        "readiness-negative": {"ok": True},
        "protocol-positive": {"ok": True},
        "protocol-negative": {"ok": True},
        "workspace-positive": {"ok": True},
        "workspace-negative": {"ok": True},
    }
    if failure:
        expected_cases = {name: result for name, result in expected_cases.items() if name.startswith("mismatch")}
        if failure == "raised":
            del expected_cases["mismatch-negative"]
    assert receipt["cases"] == expected_cases
    assert json.loads((evidence / "receipt.json").read_text())["cases"] == expected_cases
    for profile in receipt["fault_sources"]:
        assert (evidence / f"image-store/runs/{profile}-source/bundle/rootfs.img").read_bytes() == profile.encode()
    assert {path.name: path.read_bytes() for path in source_bundle.iterdir()} == source_before
    ctl.stop_helper.assert_called_once()
    assert receipt["lifecycle"]["runtime_removed"] is True
    assert not Path(receipt["lifecycle"]["runtime_dir"]).exists()
    assert receipt["errors"] == []
    tracked_disks = disk_probe.call_args.args[0]
    assert len(tracked_disks) == (6 if failure else 24)
    assert {path / "rootfs.img" for path in run_paths}.issubset(set(tracked_disks))
    if failure:
        helper.terminate_vm.assert_called_once_with("leftover")
