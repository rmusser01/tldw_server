"""Portable proof and initrd-overlay contracts for the manual boot-stall drill."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def boot_stall():
    """Load the test-only boot fixture without starting a VM."""
    path = Path(__file__).resolve().parent / "failure_drill/boot_stall.py"
    assert path.is_file(), "The boot-stall fixture is not implemented"
    spec = importlib.util.spec_from_file_location("boot_stall_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_serial_proof_requires_fresh_nonce_vm_and_stage(boot_stall, tmp_path):
    """A matching line proves only this VM reached the test initramfs wrapper."""
    proof = {"nonce": "abc123", "vm_id": "AB12-34", "mode": "stall", "stage": "initramfs"}
    log = tmp_path / "serial.log"
    log.write_text("kernel boot\nTLDW_BOOT_PROOF " + json.dumps(proof) + "\n")
    assert boot_stall.read_proof(log, "abc123", "stall", "AB12-34") == proof


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value", [("nonce", "stale"), ("vm_id", "other"), ("mode", "continue"), ("stage", "agent")]
)
def test_serial_proof_rejects_unrelated_boot(boot_stall, tmp_path, field, value):
    """Old or wrong-stage output must not turn transport failure into acceptance."""
    proof = {"nonce": "abc123", "vm_id": "AB12-34", "mode": "stall", "stage": "initramfs"}
    proof[field] = value
    log = tmp_path / "serial.log"
    log.write_text("TLDW_BOOT_PROOF " + json.dumps(proof) + "\n")
    with pytest.raises(ValueError, match="boot proof"):
        boot_stall.read_proof(log, "abc123", "stall", "AB12-34")


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    ["", "TLDW_BOOT_PROOF {oops}\n", "TLDW_BOOT_PROOF []\n", "x" * (1024 * 1024 + 1)],
    ids=["missing", "malformed", "wrong-type", "oversized"],
)
def test_serial_proof_rejects_missing_malformed_or_oversized_log(boot_stall, tmp_path, raw):
    """Read only bounded regular logs, not arbitrary/unrelated console content."""
    log = tmp_path / "serial.log"
    log.write_text(raw)
    with pytest.raises(ValueError, match="boot proof"):
        boot_stall.read_proof(log, "abc123", "stall", "vm-1")


@pytest.mark.unit
def test_serial_proof_rejects_symlink(boot_stall, tmp_path):
    """Do not read an evidence symlink outside the selected VM log."""
    target = tmp_path / "target"
    target.write_text("old")
    log = tmp_path / "serial.log"
    log.symlink_to(target)
    with pytest.raises(ValueError, match="boot proof"):
        boot_stall.read_proof(log, "abc123", "stall", "vm-1")


@pytest.mark.unit
@pytest.mark.timeout(2, method="signal")
def test_serial_proof_rejects_fifo_without_blocking(boot_stall, tmp_path):
    """A substituted FIFO must not hang diagnostics or the test harness."""
    import os

    log = tmp_path / "serial.log"
    os.mkfifo(log)
    with pytest.raises(ValueError, match="boot proof"):
        boot_stall.read_proof(log, "abc123", "stall", "vm-1")


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["stall", "continue"])
def test_overlay_contains_only_fixed_initramfs_entries(boot_stall, tmp_path, mode):
    """Native cpio output preserves original init and binds the one mode change."""
    import subprocess  # nosec B404 - checked-in local archive fixture only

    original = b"#!/bin/sh\necho original-init\n"
    archive = boot_stall.overlay(original, "abc123", mode)
    result = subprocess.run(
        ["cpio", "-it"], input=archive, capture_output=True, check=True, timeout=10
    )  # nosec B603 B607
    assert set(result.stdout.decode().splitlines()) == {
        "init",
        "init.tldw-original",
        "tldw-boot-nonce",
        "tldw-boot-mode",
    }
    subprocess.run(
        ["cpio", "-id"], input=archive, cwd=tmp_path, capture_output=True, check=True, timeout=10
    )  # nosec B603 B607
    assert (tmp_path / "init.tldw-original").read_bytes() == original
    assert (tmp_path / "tldw-boot-mode").read_text() == mode + "\n"
    assert (tmp_path / "tldw-boot-nonce").read_text() == "abc123\n"
    assert (tmp_path / "init").stat().st_mode & 0o777 == 0o755


@pytest.mark.unit
@pytest.mark.parametrize("nonce,mode", [("bad\nnonce", "stall"), ("abc123", "invalid")])
def test_overlay_refuses_untrusted_marker_values(boot_stall, nonce, mode):
    """Only a hex nonce and the two intentional control modes are accepted."""
    with pytest.raises(ValueError):
        boot_stall.overlay(b"#!/bin/sh\n", nonce, mode)


@pytest.mark.integration
def test_challenge_changes_only_nonce_and_mode(boot_stall, tmp_path):
    """The control cannot replace the wrapper or original init."""
    import subprocess  # nosec B404 - fixed local cpio fixture

    archive = boot_stall.challenge("abc123", "continue")
    subprocess.run(
        ["cpio", "-id"], input=archive, cwd=tmp_path, capture_output=True, check=True, timeout=10
    )  # nosec B603 B607
    assert {path.name: path.read_text() for path in tmp_path.iterdir()} == {
        "tldw-boot-nonce": "abc123\n",
        "tldw-boot-mode": "continue\n",
    }


@pytest.mark.unit
@pytest.mark.parametrize("name", ["../initrd", "/initrd", "initrd\nother", ".", "..", None])
def test_initrd_selection_rejects_unsafe_or_absent_manifest_entry(boot_stall, tmp_path, name):
    """Preparation must not append to a path outside its disposable bundle."""
    (tmp_path / "manifest.json").write_text(json.dumps({"initrd": name}))
    with pytest.raises(ValueError, match="initrd"):
        boot_stall.initrd_path(tmp_path)


@pytest.mark.unit
def test_initrd_selection_uses_manifest_and_refuses_symlinks(boot_stall, tmp_path):
    """Custom boot artifact names are respected without following aliases."""
    (tmp_path / "manifest.json").write_text(json.dumps({"initrd": "initramfs"}))
    path = tmp_path / "initramfs"
    path.write_bytes(b"original")
    assert boot_stall.initrd_path(tmp_path) == path
    path.unlink()
    path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="initrd"):
        boot_stall.initrd_path(tmp_path)


@pytest.mark.unit
def test_append_archive_preserves_original_bytes_and_aligns_newc_header(boot_stall, tmp_path):
    """Linux uncompressed newc headers must start at a four-byte boundary."""
    initrd = tmp_path / "initrd"
    initrd.write_bytes(b"original-gzip")
    boot_stall.append_archive(initrd, b"070701fixture")
    assert initrd.read_bytes() == b"original-gzip" + b"\0" * 3 + b"070701fixture"


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["stall", "continue"])
def test_wrapper_initializes_console_before_proof_and_only_control_runs_original(tmp_path, mode):
    """Execute the real shell flow with mount/module/device calls isolated, not privileged."""
    import os
    import signal
    import subprocess  # nosec B404 - local checked-in fixture and private stub paths
    import time

    (tmp_path / "proc").mkdir()
    device = tmp_path / "sys/class/tty/hvc0"
    device.mkdir(parents=True)
    (tmp_path / "proc/cmdline").write_text("systemd.setenv=TLDW_AGENT_GUEST_VM_ID=vm-1\n")
    (tmp_path / "tldw-boot-nonce").write_text("abc123\n")
    (tmp_path / "tldw-boot-mode").write_text(mode + "\n")
    original = tmp_path / "init.tldw-original"
    original.write_text("#!/bin/sh\nprintf 'original-started\\n'\n")
    original.chmod(0o755)
    script = (Path(__file__).parent / "failure_drill/boot-stall-init.sh").read_text()
    for path in ("/proc", "/sys", "/tldw-boot-nonce", "/tldw-boot-mode", "/tldw-boot-console", "/init.tldw-original"):
        script = script.replace(path, str(tmp_path) + path)
    stubs = """
mount() { printf 'mount %s\n' "$*" >> "$CALLS"; }
umount() { printf 'umount %s\n' "$*" >> "$CALLS"; }
modprobe() {
    printf 'modprobe %s\n' "$*" >> "$CALLS"
    if [ "$1" = virtio_console ]; then printf '229:0\n' > "$DEVICE"; fi
}
mknod() { printf 'mknod\n' >> "$CALLS"; : > "$1"; }
sleep() { /bin/sleep 0.05; }
"""
    fixture = tmp_path / "fixture.sh"
    fixture.write_text(stubs + script)
    console = tmp_path / "tldw-boot-console"
    calls = tmp_path / "calls"
    env = {**os.environ, "CALLS": str(calls), "DEVICE": str(device / "dev")}
    process = subprocess.Popen(
        ["/bin/sh", str(fixture)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
    )  # nosec B603
    try:
        if mode == "continue":
            process.communicate(timeout=5)
            assert process.returncode == 0
        else:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and process.poll() is None:
                if console.exists() and "TLDW_BOOT_PROOF" in console.read_text():
                    break
                time.sleep(0.02)
            assert process.poll() is None
        output = console.read_text()
        assert 'TLDW_BOOT_PROOF {"nonce":"abc123","vm_id":"vm-1","mode":"' + mode in output
        assert ("original-started" in output) is (mode == "continue")
        assert "modprobe virtio_pci\nmodprobe virtio_console\nmknod\n" in calls.read_text()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=5)
