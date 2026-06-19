from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


IMAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = IMAGE_DIR.parents[1]
SMOKE_SCRIPT = IMAGE_DIR / "scripts" / "run-host-e2e-smoke.sh"
EVIDENCE_FILES = {
    "host-smoke-evidence.json",
    "source-bundle-hashes-before.txt",
    "source-bundle-hashes-after.txt",
    "run-bundle-hashes.txt",
    "runtime-paths.txt",
    "cleanup-status.txt",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}  .+$", re.MULTILINE)


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a test fixture file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _planned_evidence_files(stdout: str) -> set[str]:
    """Extract evidence file paths printed by a dry-run plan."""
    return {
        line.removeprefix("evidence file: ")
        for line in stdout.splitlines()
        if line.startswith("evidence file: ")
    }


def _assert_owner_only(path: Path) -> None:
    """Assert that a path is not group/world accessible."""
    assert path.stat().st_mode & 0o077 == 0


def _read_hash_file(path: Path) -> str:
    """Read an evidence hash artifact and verify it contains SHA-256 rows."""
    text = path.read_text(encoding="utf-8")
    assert SHA256_RE.search(text)
    return text


def _run_smoke_script(*args: str, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_overrides or {})
    return subprocess.run(
        [str(SMOKE_SCRIPT), *args],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_host_e2e_smoke_script_help_mentions_required_bundle() -> None:
    result = _run_smoke_script("--help")

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "--bundle PATH" in result.stdout
    assert "Canonical source vz_linux bundle" in result.stdout
    assert "--evidence-dir PATH" in result.stdout  # nosec B101
    assert "--include-failure-drills" in result.stdout


def test_host_e2e_smoke_script_requires_bundle() -> None:
    result = _run_smoke_script("--dry-run")

    assert result.returncode != 0
    assert "--bundle is required" in result.stderr


def test_host_e2e_smoke_script_dry_run_prints_helper_and_pytest_commands(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"
    entitlements = tmp_path / "helper.entitlements"
    entitlements.write_text("<plist/>", encoding="utf-8")
    socket_path = tmp_path / "helper.sock"
    serial_log_dir = tmp_path / "serial"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--entitlements",
        str(entitlements),
        "--socket",
        str(socket_path),
        "--serial-log-dir",
        str(serial_log_dir),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "dry-run" in result.stdout
    assert "swift build" in result.stdout
    assert "codesign --force --sign - --entitlements" in result.stdout
    assert f"TLDW_SANDBOX_MACOS_HELPER_SOCKET={socket_path}" in result.stdout
    assert "prepare-smoke-bundle.py" in result.stdout
    assert f"--source-bundle {bundle}" in result.stdout
    run_bundle_match = re.search(
        r"TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=([^ ]+/image-store/runs/[^ ]+/bundle)",
        result.stdout,
    )
    assert run_bundle_match is not None
    run_bundle = run_bundle_match.group(1)
    assert f"TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH={run_bundle}" in result.stdout
    assert f"TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE={bundle}" not in result.stdout
    assert f"TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR={serial_log_dir}" in result.stdout  # nosec B101
    assert "test_macos_virtualization_helper_daemon_host_gated.py" in result.stdout
    assert "test_vz_linux_real_host_e2e.py" in result.stdout
    assert "-m vz_linux_host_smoke" in result.stdout
    assert "vz_linux_host_failure_drill" not in result.stdout
    assert not serial_log_dir.exists()


def test_host_e2e_smoke_script_dry_run_uses_disposable_run_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "prepare-smoke-bundle.py" in result.stdout
    assert f"--source-bundle {bundle}" in result.stdout
    run_bundle_match = re.search(
        r"TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE=([^ ]+/image-store/runs/[^ ]+/bundle)",
        result.stdout,
    )
    assert run_bundle_match is not None
    run_bundle = run_bundle_match.group(1)
    assert run_bundle != str(bundle)
    assert f"TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH={run_bundle}" in result.stdout
    assert f"TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE={bundle}" not in result.stdout


def test_host_e2e_smoke_script_dry_run_prints_default_evidence_bundle(tmp_path: Path) -> None:
    """Dry-run prints every evidence file under the default evidence directory."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "evidence directory:" in result.stdout
    evidence_match = re.search(r"evidence directory: ([^\n]+/evidence)", result.stdout)
    assert evidence_match is not None
    evidence_dir_text = evidence_match.group(1)
    evidence_dir = Path(evidence_dir_text)
    assert evidence_dir.name == "evidence"
    assert not evidence_dir.exists()
    assert f"export TLDW_SANDBOX_VZ_EVIDENCE_DIR={evidence_dir_text}" in result.stdout  # nosec B101
    expected_paths = {f"{evidence_dir_text}/{evidence_file}" for evidence_file in EVIDENCE_FILES}
    assert expected_paths <= _planned_evidence_files(result.stdout)


def test_host_e2e_smoke_script_dry_run_accepts_evidence_dir_override(tmp_path: Path) -> None:
    """Dry-run accepts a custom evidence directory without creating it."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"
    evidence_dir = tmp_path / "custom-evidence"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--evidence-dir",
        str(evidence_dir),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert f"evidence directory: {evidence_dir}" in result.stdout
    assert f"export TLDW_SANDBOX_VZ_EVIDENCE_DIR={evidence_dir}" in result.stdout  # nosec B101
    assert not evidence_dir.exists()
    expected_paths = {str(evidence_dir / evidence_file) for evidence_file in EVIDENCE_FILES}
    assert expected_paths <= _planned_evidence_files(result.stdout)


def test_host_e2e_smoke_script_dry_run_uses_materializer_normalized_path(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"
    relative_store = "relative-image-store"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--image-store-root",
        relative_store,
        "--smoke-run-id",
        "  smoke-run  ",
        "--python",
        sys.executable,
    )

    expected_run_bundle = REPO_ROOT / relative_store / "runs" / "smoke-run" / "bundle"
    assert result.returncode == 0, result.stderr
    assert f"TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH={expected_run_bundle}" in result.stdout
    assert f"TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE={expected_run_bundle}" in result.stdout
    assert "  smoke-run  " not in result.stdout.split(f"TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH={expected_run_bundle}", 1)[-1]
    assert not (REPO_ROOT / relative_store).exists()


def test_host_e2e_smoke_script_dry_run_rejects_invalid_run_id(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--smoke-run-id",
        "../escape",
        "--python",
        sys.executable,
    )

    assert result.returncode != 0
    assert "run_id_invalid" in result.stderr


def test_host_e2e_smoke_script_real_run_passes_disposable_bundle_to_pytest(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source_rootfs = bundle / "rootfs.img"
    (bundle / "kernel").write_bytes(b"kernel")
    source_rootfs.write_bytes(b"rootfs")
    (bundle / "manifest.json").write_text(
        '{"bundle_version":"1","kernel":"kernel","rootfs":"rootfs.img"}',
        encoding="utf-8",
    )
    source_before = source_rootfs.read_bytes()
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    marker = tmp_path / "base-images.txt"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        f"real_python = {str(sys.executable)!r}\n"
        f"marker = {str(marker)!r}\n"
        "if len(sys.argv) > 1 and sys.argv[1].endswith('prepare-smoke-bundle.py'):\n"
        "    os.execv(real_python, [real_python, *sys.argv[1:]])\n"
        "if sys.argv[1:3] != ['-m', 'pytest']:\n"
        "    sys.exit(2)\n"
        "with open(marker, 'a', encoding='utf-8') as handle:\n"
        "    handle.write(os.environ.get('TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH', '') + '\\n')\n"
        "    handle.write(os.environ.get('TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE', '') + '\\n')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={
            "TMPDIR": str(tmp_dir),
            "TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    recorded_paths = [line for line in marker.read_text(encoding="utf-8").splitlines() if line]
    assert recorded_paths
    assert str(bundle) not in recorded_paths
    assert all(path.endswith("/bundle") for path in recorded_paths)
    assert all((Path(path) / "rootfs.img").is_file() for path in recorded_paths)
    assert source_rootfs.read_bytes() == source_before


def test_host_e2e_smoke_script_real_run_writes_evidence_bundle(tmp_path: Path) -> None:
    """Real fake-helper runs write private, hashed, redacted evidence artifacts."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    kernel = bundle / "kernel"
    rootfs = bundle / "rootfs.img"
    kernel.write_bytes(b"kernel")
    rootfs.write_bytes(b"rootfs")
    (bundle / "manifest.json").write_text(
        '{"bundle_version":"1","kernel":"kernel","rootfs":"rootfs.img"}',
        encoding="utf-8",
    )
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    evidence_dir = tmp_path / "evidence"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        f"real_python = {str(sys.executable)!r}\n"
        "if len(sys.argv) > 1 and sys.argv[1].endswith('prepare-smoke-bundle.py'):\n"
        "    os.execv(real_python, [real_python, *sys.argv[1:]])\n"
        "if sys.argv[1:3] != ['-m', 'pytest']:\n"
        "    sys.exit(2)\n"
        "serial_dir = os.environ.get('TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR')\n"
        "if serial_dir:\n"
        "    os.makedirs(serial_dir, exist_ok=True)\n"
        "    with open(os.path.join(serial_dir, 'guest.log'), 'w', encoding='utf-8') as handle:\n"
        "        handle.write('raw serial log should stay out of evidence')\n"
        "    unreadable = os.path.join(serial_dir, 'unreadable.log')\n"
        "    with open(unreadable, 'w', encoding='utf-8') as handle:\n"
        "        handle.write('unreadable raw serial log should stay out of evidence')\n"
        "    os.chmod(unreadable, 0)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--evidence-dir",
        str(evidence_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={
            "TMPDIR": str(tmp_dir),
            "TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1",
        },
    )

    try:
        assert result.returncode == 0, result.stderr
        assert {path.name for path in evidence_dir.iterdir()} >= EVIDENCE_FILES
        _assert_owner_only(evidence_dir)
        for evidence_file in EVIDENCE_FILES:
            _assert_owner_only(evidence_dir / evidence_file)
        source_before_hashes = _read_hash_file(evidence_dir / "source-bundle-hashes-before.txt")
        source_after_hashes = _read_hash_file(evidence_dir / "source-bundle-hashes-after.txt")
        run_hashes = _read_hash_file(evidence_dir / "run-bundle-hashes.txt")
        assert f"{_sha256_file(kernel)}  kernel" in source_before_hashes
        assert f"{_sha256_file(rootfs)}  rootfs.img" in source_after_hashes
        assert f"{_sha256_file(rootfs)}  rootfs.img" in run_hashes
        assert f"export TLDW_SANDBOX_VZ_EVIDENCE_DIR={evidence_dir}" in result.stdout  # nosec B101
        evidence = json.loads((evidence_dir / "host-smoke-evidence.json").read_text(encoding="utf-8"))
    finally:
        for unreadable_log in tmp_dir.glob("**/unreadable.log"):
            unreadable_log.chmod(0o600)
    assert evidence["schema_version"] == 1
    assert evidence["source_bundle_path"] == str(bundle)
    assert evidence["evidence_dir"] == str(evidence_dir)
    assert evidence["serial_log_dir"].endswith("/serial")
    assert evidence["helper_pid_file"].endswith("/helper.pid")
    assert evidence["final_exit_code"] == 0
    assert evidence["phases"]["real_host_smoke"]["status"] == "ok"
    assert evidence["cleanup"]["socket_present_after_cleanup"] is False
    assert Path(evidence["run_bundle_path"]).name == "bundle"
    assert "raw_log_contents" not in evidence
    guest_log = next(item for item in evidence["log_artifacts"] if Path(item["path"]).name == "guest.log")
    assert guest_log["size_bytes"] > 0
    assert guest_log["sha256"] == _sha256_file(Path(guest_log["path"]))
    assert "raw serial log should stay out of evidence" not in json.dumps(evidence)
    assert all(Path(item["path"]).name != "unreadable.log" for item in evidence["log_artifacts"])


def test_host_e2e_smoke_script_refuses_non_private_evidence_directory(tmp_path: Path) -> None:
    """Real runs reject pre-existing evidence directories that are not private."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    socket_dir = tmp_path / "private-runtime"
    socket_dir.mkdir(mode=0o700)
    socket_dir.chmod(0o700)
    serial_log_dir = tmp_path / "serial"
    evidence_dir = tmp_path / "public-evidence"
    evidence_dir.mkdir(mode=0o755)
    evidence_dir.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--socket",
        str(socket_dir / "helper.sock"),
        "--serial-log-dir",
        str(serial_log_dir),
        "--evidence-dir",
        str(evidence_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={"TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1"},
    )

    assert result.returncode != 0
    assert "evidence directory must be owner-only" in result.stderr


def test_host_e2e_smoke_script_prepare_failure_does_not_hash_repo_as_run_bundle(tmp_path: Path) -> None:
    """Early failures write a missing run-bundle hash artifact instead of hashing the cwd."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    evidence_dir = tmp_path / "evidence"

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--evidence-dir",
        str(evidence_dir),
        "--skip-build",
        "--skip-sign",
    )

    assert result.returncode != 0
    run_hashes = (evidence_dir / "run-bundle-hashes.txt").read_text(encoding="utf-8")
    assert "# missing: <empty bundle path>" in run_hashes
    assert "run-host-e2e-smoke.sh" not in run_hashes


def test_host_e2e_smoke_script_evidence_falls_back_to_valid_python_bin(tmp_path: Path) -> None:
    """Evidence generation skips invalid PATH Python shims and uses a valid --python."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    (bundle / "manifest.json").write_text(
        '{"bundle_version":"1","kernel":"kernel","rootfs":"rootfs.img"}',
        encoding="utf-8",
    )
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    evidence_dir = tmp_path / "evidence"
    fake_path = tmp_path / "fake-path"
    fake_path.mkdir()
    failing_python3 = fake_path / "python3"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    failing_python3.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        f"real_python = {str(sys.executable)!r}\n"
        "if sys.argv[1:2] == ['-c']:\n"
        "    os.execv(real_python, [real_python, *sys.argv[1:]])\n"
        "if len(sys.argv) > 1 and sys.argv[1].endswith('prepare-smoke-bundle.py'):\n"
        "    os.execv(real_python, [real_python, *sys.argv[1:]])\n"
        "if sys.argv[1:3] != ['-m', 'pytest']:\n"
        "    sys.exit(2)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/bin/sh\n"
        "trap 'exit 0' TERM\n"
        "while true; do sleep 1; done\n",
        encoding="utf-8",
    )
    failing_python3.chmod(0o755)
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--evidence-dir",
        str(evidence_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={
            "PATH": f"{fake_path}:{os.environ['PATH']}",
            "TMPDIR": str(tmp_dir),
            "TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert (evidence_dir / "host-smoke-evidence.json").is_file()


def test_host_e2e_smoke_script_late_failure_preserves_exit_and_writes_evidence(tmp_path: Path) -> None:
    """Late pytest failures preserve the pytest exit code and still write evidence."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    (bundle / "manifest.json").write_text(
        '{"bundle_version":"1","kernel":"kernel","rootfs":"rootfs.img"}',
        encoding="utf-8",
    )
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    evidence_dir = tmp_path / "evidence"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        f"real_python = {str(sys.executable)!r}\n"
        "if len(sys.argv) > 1 and sys.argv[1].endswith('prepare-smoke-bundle.py'):\n"
        "    os.execv(real_python, [real_python, *sys.argv[1:]])\n"
        "if sys.argv[1:3] != ['-m', 'pytest']:\n"
        "    sys.exit(2)\n"
        "if any(arg.endswith('test_vz_linux_real_host_e2e.py') for arg in sys.argv):\n"
        "    sys.exit(7)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--evidence-dir",
        str(evidence_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={
            "TMPDIR": str(tmp_dir),
            "TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1",
        },
    )

    assert result.returncode == 7
    assert (evidence_dir / "cleanup-status.txt").is_file()
    evidence = json.loads((evidence_dir / "host-smoke-evidence.json").read_text(encoding="utf-8"))
    assert evidence["final_exit_code"] == 7
    assert evidence["phases"]["real_host_smoke"]["status"] == "failed"
    assert evidence["phases"]["real_host_smoke"]["exit_code"] == 7
    assert evidence["cleanup"]["socket_present_after_cleanup"] is False


def test_host_e2e_smoke_script_dry_run_includes_failure_drills_when_requested(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--include-failure-drills",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "-m vz_linux_host_smoke" in result.stdout
    assert "-m vz_linux_host_failure_drill" in result.stdout


def test_host_e2e_smoke_script_default_dry_run_omits_restart_lease(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED" not in result.stdout
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE" not in result.stdout


def test_host_e2e_smoke_script_failure_drill_dry_run_includes_restart_lease(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--include-failure-drills",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED=1" in result.stdout
    assert "TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE=" in result.stdout
    assert "/helper.pid" in result.stdout
    assert f"TLDW_SANDBOX_MACOS_HELPER_BINARY={helper}" in result.stdout


def test_host_e2e_smoke_script_dry_run_does_not_kill_existing_pid_file_process(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    socket_dir = tmp_path / "runtime"
    socket_dir.mkdir(mode=0o700)
    pid_file = socket_dir / "helper.pid"
    helper = tmp_path / "macos-vz-helper"
    helper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    helper.chmod(0o755)
    proc = subprocess.Popen(  # nosec B603
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
        pid_file.chmod(0o600)

        result = _run_smoke_script(
            "--dry-run",
            "--bundle",
            str(bundle),
            "--helper",
            str(helper),
            "--socket",
            str(socket_dir / "helper.sock"),
            "--python",
            sys.executable,
        )

        assert result.returncode == 0, result.stderr
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=0.2)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_host_e2e_smoke_script_validation_failure_does_not_kill_existing_pid_file_process(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    socket_dir = tmp_path / "runtime"
    socket_dir.mkdir(mode=0o700)
    pid_file = socket_dir / "helper.pid"
    helper = tmp_path / "macos-vz-helper"
    helper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    helper.chmod(0o755)
    proc = subprocess.Popen(  # nosec B603
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
        pid_file.chmod(0o600)

        result = _run_smoke_script(
            "--bundle",
            str(bundle),
            "--helper",
            str(helper),
            "--socket",
            str(socket_dir / "helper.sock"),
            "--python",
            sys.executable,
            "--skip-build",
            "--skip-sign",
        )

        assert result.returncode != 0
        assert "bundle missing kernel" in result.stderr
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(timeout=0.2)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_host_e2e_smoke_script_default_socket_uses_private_runtime_dir(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    helper = tmp_path / "macos-vz-helper"

    result = _run_smoke_script(
        "--dry-run",
        "--bundle",
        str(bundle),
        "--helper",
        str(helper),
        "--python",
        sys.executable,
    )

    assert result.returncode == 0, result.stderr
    assert "TLDW_SANDBOX_MACOS_HELPER_SOCKET=" in result.stdout
    assert "/tldw-vz-helper-e2e-" in result.stdout
    assert "/helper.sock" in result.stdout
    assert "TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR=" in result.stdout
    assert "/serial" in result.stdout


def test_host_e2e_smoke_script_default_runtime_dir_is_private_for_real_run(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        f"real_python = {str(sys.executable)!r}\n"
        "if len(sys.argv) > 1 and sys.argv[1].endswith('prepare-smoke-bundle.py'):\n"
        "    os.execv(real_python, [real_python, *sys.argv[1:]])\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={
            "TMPDIR": str(tmp_dir),
            "TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    socket_match = re.search(
        r"TLDW_SANDBOX_MACOS_HELPER_SOCKET=([^ ]+/helper\.sock)",
        result.stdout,
    )
    assert socket_match is not None
    runtime_dir = Path(socket_match.group(1)).parent
    assert runtime_dir.exists()
    assert runtime_dir.stat().st_mode & 0o077 == 0
    serial_dir = runtime_dir / "serial"
    assert serial_dir.exists()
    assert serial_dir.stat().st_mode & 0o077 == 0


def test_host_e2e_smoke_script_removes_stale_socket_before_helper_start(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    serial_log_dir = tmp_path / "serial"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        f"real_python = {str(sys.executable)!r}\n"
        "if len(sys.argv) > 1 and sys.argv[1].endswith('prepare-smoke-bundle.py'):\n"
        "    os.execv(real_python, [real_python, *sys.argv[1:]])\n"
        "if len(sys.argv) > 2 and sys.argv[1] == '-c':\n"
        "    raise SystemExit(1)\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    # AF_UNIX paths must stay short on macOS; this mirrors the existing smoke-test socket fixture.
    with tempfile.TemporaryDirectory(  # nosec B108
        prefix="vz-smoke-",
        dir="/tmp",  # nosec B108
    ) as socket_dir:
        socket_path = Path(socket_dir) / "helper.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale_socket:
            try:
                stale_socket.bind(str(socket_path))
            except PermissionError:
                pytest.skip("AF_UNIX socket binding is not permitted in this sandbox")

            result = _run_smoke_script(
                "--bundle",
                str(bundle),
                "--helper",
                str(fake_helper),
                "--socket",
                str(socket_path),
                "--serial-log-dir",
                str(serial_log_dir),
                "--python",
                str(fake_python),
                "--skip-build",
                "--skip-sign",
                env_overrides={"TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1"},
            )

        assert result.returncode == 0, result.stderr
    assert not socket_path.exists()


def test_host_e2e_smoke_script_refuses_live_socket_before_helper_start(tmp_path: Path) -> None:
    """Refuse a socket with an active listener instead of replacing it."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    serial_log_dir = tmp_path / "serial"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if len(sys.argv) > 2 and sys.argv[1] == '-c':\n"
        "    code = sys.argv[2]\n"
        "    sys.argv = [sys.argv[0], *sys.argv[3:]]\n"
        "    exec(code)\n"
        "    raise SystemExit(0)\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    # AF_UNIX paths must stay short on macOS; this mirrors the existing smoke-test socket fixture.
    with tempfile.TemporaryDirectory(  # nosec B108
        prefix="vz-smoke-",
        dir="/tmp",  # nosec B108
    ) as socket_dir:
        socket_path = Path(socket_dir) / "helper.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as live_socket:
            try:
                live_socket.bind(str(socket_path))
                live_socket.listen(1)
            except PermissionError:
                pytest.skip("AF_UNIX socket binding is not permitted in this sandbox")

            result = _run_smoke_script(
                "--bundle",
                str(bundle),
                "--helper",
                str(fake_helper),
                "--socket",
                str(socket_path),
                "--serial-log-dir",
                str(serial_log_dir),
                "--python",
                str(fake_python),
                "--skip-build",
                "--skip-sign",
            )

            assert result.returncode != 0  # nosec B101
            assert "helper socket path is already in use" in result.stderr  # nosec B101
            assert socket_path.exists()  # nosec B101


def test_host_e2e_smoke_script_refuses_unsafe_socket_probe_result(tmp_path: Path) -> None:
    """Refuse to delete an existing socket when the probe cannot classify it as stale."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    serial_log_dir = tmp_path / "serial"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if len(sys.argv) > 2 and sys.argv[1] == '-c':\n"
        "    raise SystemExit(2)\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    # AF_UNIX paths must stay short on macOS; this mirrors the existing smoke-test socket fixture.
    with tempfile.TemporaryDirectory(  # nosec B108
        prefix="vz-smoke-",
        dir="/tmp",  # nosec B108
    ) as socket_dir:
        socket_path = Path(socket_dir) / "helper.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as existing_socket:
            try:
                existing_socket.bind(str(socket_path))
            except PermissionError:
                pytest.skip("AF_UNIX socket binding is not permitted in this sandbox")

            result = _run_smoke_script(
                "--bundle",
                str(bundle),
                "--helper",
                str(fake_helper),
                "--socket",
                str(socket_path),
                "--serial-log-dir",
                str(serial_log_dir),
                "--python",
                str(fake_python),
                "--skip-build",
                "--skip-sign",
            )

            assert result.returncode != 0  # nosec B101
            assert "refusing to remove" in result.stderr  # nosec B101
            assert socket_path.exists()  # nosec B101


def test_host_e2e_smoke_script_refuses_non_private_socket_parent(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    serial_log_dir = tmp_path / "serial"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)
    socket_dir = tmp_path / "public-runtime"
    socket_dir.mkdir(mode=0o755)
    socket_dir.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--socket",
        str(socket_dir / "helper.sock"),
        "--serial-log-dir",
        str(serial_log_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={"TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1"},
    )

    assert result.returncode != 0
    assert "helper socket directory must be owner-only" in result.stderr


def test_host_e2e_smoke_script_refuses_non_private_serial_log_directory(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    socket_dir = tmp_path / "private-runtime"
    socket_dir.mkdir(mode=0o700)
    socket_dir.chmod(0o700)
    serial_log_dir = tmp_path / "public-serial"
    serial_log_dir.mkdir(mode=0o755)
    serial_log_dir.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--socket",
        str(socket_dir / "helper.sock"),
        "--serial-log-dir",
        str(serial_log_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={"TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1"},
    )

    assert result.returncode != 0  # nosec B101
    assert "serial log directory must be owner-only" in result.stderr  # nosec B101


def test_host_e2e_smoke_script_refuses_regular_file_socket_path(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    serial_log_dir = tmp_path / "serial"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.exit(0 if sys.argv[1:3] == ['-m', 'pytest'] else 2)\n",
        encoding="utf-8",
    )
    fake_helper.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)
    socket_path = tmp_path / "not-a-socket"
    socket_path.write_text("do not delete", encoding="utf-8")

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--socket",
        str(socket_path),
        "--serial-log-dir",
        str(serial_log_dir),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        env_overrides={"TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1"},
    )

    assert result.returncode != 0
    assert "helper socket path already exists and is not a UNIX socket" in result.stderr
    assert socket_path.read_text(encoding="utf-8") == "do not delete"


def test_host_e2e_smoke_script_cleanup_uses_replacement_helper_pid(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    marker = tmp_path / "replacement_pid.txt"
    fake_python = tmp_path / "fake-python"
    fake_helper = tmp_path / "fake-helper"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        f"real_python = {str(sys.executable)!r}\n"
        "if len(sys.argv) > 1 and sys.argv[1].endswith('prepare-smoke-bundle.py'):\n"
        "    os.execv(real_python, [real_python, *sys.argv[1:]])\n"
        "if sys.argv[1:3] != ['-m', 'pytest']:\n"
        "    sys.exit(2)\n"
        "if 'vz_linux_host_failure_drill' in sys.argv:\n"
        "    pid_file = os.environ['TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE']\n"
        "    marker = os.environ['TLDW_TEST_REPLACEMENT_PID_MARKER']\n"
        "    proc = subprocess.Popen(\n"
        "        [os.environ['TLDW_SANDBOX_MACOS_HELPER_BINARY']],\n"
        "        stdin=subprocess.DEVNULL,\n"
        "        stdout=subprocess.DEVNULL,\n"
        "        stderr=subprocess.DEVNULL,\n"
        "    )\n"
        "    with open(pid_file, 'w', encoding='utf-8') as handle:\n"
        "        handle.write(str(proc.pid) + '\\n')\n"
        "    os.chmod(pid_file, 0o600)\n"
        "    with open(marker, 'w', encoding='utf-8') as handle:\n"
        "        handle.write(str(proc.pid) + '\\n')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_helper.chmod(0o755)

    result = _run_smoke_script(
        "--bundle",
        str(bundle),
        "--helper",
        str(fake_helper),
        "--python",
        str(fake_python),
        "--skip-build",
        "--skip-sign",
        "--include-failure-drills",
        env_overrides={
            "TMPDIR": str(tmp_dir),
            "TLDW_HOST_E2E_SMOKE_SKIP_SOCKET_WAIT": "1",
            "TLDW_TEST_REPLACEMENT_PID_MARKER": str(marker),
        },
    )

    assert result.returncode == 0, result.stderr
    replacement_pid = int(marker.read_text(encoding="utf-8").strip())
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(replacement_pid, 0)
    finally:
        try:
            os.kill(replacement_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
