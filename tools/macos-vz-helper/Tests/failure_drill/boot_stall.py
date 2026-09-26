"""Test-only initramfs overlay and bounded serial proof reader."""

import json
import os
import re
import stat
import subprocess  # nosec B404 - native cpio, fixed private archive entries only
import tempfile
from pathlib import Path


def initrd_path(bundle: Path) -> Path:
    """Require the manifest-selected, regular initrd inside a disposable bundle."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    name = manifest.get("initrd") if isinstance(manifest, dict) else None
    if not isinstance(name, str) or re.fullmatch(r"[a-zA-Z0-9._-]+", name) is None or name in (".", ".."):
        raise ValueError("boot drill requires a simple initrd filename in manifest.json")
    path = bundle / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("boot drill requires a regular initrd")
    return path


def append_archive(path: Path, archive: bytes) -> None:
    """Append aligned newc bytes to an existing regular disposable initrd."""
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise ValueError("invalid disposable initrd")
        with os.fdopen(fd, "ab", closefd=False) as handle:
            handle.write(b"\0" * (-info.st_size % 4) + archive)
    finally:
        os.close(fd)


def read_proof(path: Path, nonce: str, mode: str, vm_id: str) -> dict[str, str]:
    """Require one bounded, regular, fresh VM-correlated serial marker."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        try:
            if path.is_symlink() or not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("invalid boot proof file")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                raw = handle.read(1024 * 1024 + 1)
        finally:
            os.close(fd)
        if len(raw) > 1024 * 1024:
            raise ValueError("oversized boot proof log")
        prefix = b"TLDW_BOOT_PROOF "
        markers = [line[len(prefix) :] for line in raw.splitlines() if line.startswith(prefix)]
        if len(markers) != 1:
            raise ValueError("expected one boot proof")
        proof = json.loads(markers[0])
        expected = {"nonce": nonce, "mode": mode, "vm_id": vm_id, "stage": "initramfs"}
        if not nonce or not vm_id or proof != expected:
            raise ValueError("wrong VM, nonce, mode or stage in boot proof")
        return proof
    except (OSError, ValueError) as exc:
        raise ValueError("invalid boot proof") from exc


def _archive(files: dict[str, bytes]) -> bytes:
    """Build a native newc archive of fixed entries in a private temporary tree."""
    with tempfile.TemporaryDirectory(prefix="vz-boot-cpio.") as directory:
        root = Path(directory)
        for name, contents in files.items():
            path = root / name
            path.write_bytes(contents)
            path.chmod(0o755 if name.startswith("init") else 0o644)
        return subprocess.run(  # nosec B603 B607 - fixed native tool and fixture filenames
            ["cpio", "-o", "-H", "newc"],
            input=("\n".join(files) + "\n").encode(),
            cwd=root,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout


def _settings(nonce: str, mode: str) -> dict[str, bytes]:
    """Reject shell/JSON injection in test marker data before archive creation."""
    if re.fullmatch(r"[a-f0-9]{6,64}", nonce) is None or mode not in ("stall", "continue"):
        raise ValueError("invalid boot challenge")
    return {"tldw-boot-nonce": (nonce + "\n").encode(), "tldw-boot-mode": (mode + "\n").encode()}


def challenge(nonce: str, mode: str) -> bytes:
    """Override only nonce and stall/continue mode for one disposable boot."""
    return _archive(_settings(nonce, mode))


def overlay(original_init: bytes, nonce: str, mode: str) -> bytes:
    """Preserve Debian init while prepending the test-only early-userspace wrapper."""
    settings = _settings(nonce, mode)
    if not original_init.startswith(b"#!/bin/sh") or len(original_init) > 128 * 1024:
        raise ValueError("unsupported original initramfs init")
    return _archive(
        {
            "init": Path(__file__).with_name("boot-stall-init.sh").read_bytes(),
            "init.tldw-original": original_init,
            **settings,
        }
    )
