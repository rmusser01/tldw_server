from __future__ import annotations

import subprocess
from pathlib import Path


IMAGE_DIR = Path(__file__).resolve().parents[1]
PROFILES_DIR = IMAGE_DIR / "profiles"
DEFAULTS_SCRIPT = IMAGE_DIR / "scripts" / "builder-defaults.sh"


def _read_profile(name: str) -> list[str]:
    profile_path = PROFILES_DIR / name
    return [
        line.strip()
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _run_defaults(*args: str) -> subprocess.CompletedProcess[str]:
    command = [
        "bash",
        "-lc",
        (
            f"source '{DEFAULTS_SCRIPT}' && "
            + " ".join(subprocess.list2cmdline([arg]) for arg in args)
        ),
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True)


def test_minimal_profile_contains_required_boot_packages() -> None:
    packages = _read_profile("minimal.packages")

    assert "systemd" in packages
    assert "initramfs-tools" in packages


def test_debug_profile_extends_minimal_without_duplicates() -> None:
    minimal = _read_profile("minimal.packages")

    result = _run_defaults("compose_package_profiles", "minimal", "debug")
    assert result.returncode == 0, result.stderr

    debug = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert set(minimal).issubset(set(debug))
    assert len(debug) == len(set(debug))


def test_builder_defaults_expose_pinned_kernel_and_architecture() -> None:
    result = _run_defaults("print_builder_defaults")
    assert result.returncode == 0, result.stderr

    defaults = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if "=" in line
    }
    assert defaults["TLDW_VZ_LINUX_BUILDER_SUITE"] == "bookworm"
    assert defaults["TLDW_VZ_LINUX_BUILDER_ARCH"] == "arm64"
    assert defaults["TLDW_VZ_LINUX_BUILDER_KERNEL_PACKAGE"]
