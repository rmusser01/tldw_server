"""Run the complete upgrade script with isolated external-command fixtures."""

import os
import shutil
import subprocess  # nosec B404 - isolated whole-script subprocess contract tests
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "Helper_Scripts" / "upgrade.sh"


def _bash_executable() -> str:
    """Prefer Git Bash over Windows' WSL launcher, as in sidecar tests."""
    if sys.platform == "win32":
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            root = os.environ.get(variable)
            if root:
                for relative in ("Git/bin/bash.exe", "Git/usr/bin/bash.exe"):
                    candidate = Path(root) / relative
                    if candidate.is_file():
                        return str(candidate)
    bash = shutil.which("bash")
    if bash and "WindowsApps" not in bash:
        return bash
    pytest.fail("Bash (Git Bash on Windows) is required for upgrade contract tests")


@pytest.fixture
def upgrade_script(tmp_path):
    repository = tmp_path / "installation"
    helpers = repository / "Helper_Scripts"
    helpers.mkdir(parents=True)
    script = helpers / "upgrade.sh"
    # Native Windows checkouts may be CRLF; execute the complete script as LF.
    script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    databases = repository / "Databases"
    databases.mkdir()
    (databases / "retained.txt").write_text("prior release record")
    commands = tmp_path / "commands.txt"
    tools = tmp_path / "tools"
    tools.mkdir()
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    stub = tools / "python"
    stub.write_text(
        r"""#!/bin/sh
if [ "$#:$*" = "3:-m Helper_Scripts.upgrade_helpers --json" ]; then
    status="${UPGRADE_PRECHECK_EXIT:-0}"
    proceed=false
    if [ "$status" = 0 ]; then proceed=true; fi
    printf '{"can_proceed":%s,"summary":"fixture preflight","checks":[]}\n' "$proceed"
    exit "$status"
fi
if [ "${1:-}" = "-c" ]; then
    printf '%s\n' 'fixture summary'
    exit 0
fi
printf '%s\n' "python $*" >> "$UPGRADE_TEST_COMMANDS"
case "$#:$*" in
    '2:-m Helper_Scripts.upgrade_helpers') exit "${UPGRADE_POSTCHECK_EXIT:-0}" ;;
    '5:-m pip install -e .') exit 0 ;;
    *) exit 99 ;;
esac
""",
        encoding="utf-8",
        newline="\n",
    )
    stub.chmod(0o700)
    git = tools / "git"
    git.write_text(
        r"""#!/bin/sh
printf '%s\n' "git $*" >> "$UPGRADE_TEST_COMMANDS"
case "$#:$*" in
    '3:rev-parse --abbrev-ref HEAD') printf '%s\n' 'main' ;;
    '2:rev-parse HEAD') printf '%s\n' 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' ;;
    '3:pull origin main') exit 0 ;;
    *) exit 99 ;;
esac
""",
        encoding="utf-8",
        newline="\n",
    )
    git.chmod(0o700)

    def run(*arguments, precheck=0, postcheck=0):
        environment = os.environ.copy()
        environment.update(
            PYTHON="../tools/python",
            TMPDIR="../temporary",
            UPGRADE_TEST_COMMANDS="../commands.txt",
            UPGRADE_PRECHECK_EXIT=str(precheck),
            UPGRADE_POSTCHECK_EXIT=str(postcheck),
        )
        result = subprocess.run(  # nosec B603 - real script with isolated Git/pip fixtures
            [
                _bash_executable(), "--noprofile", "--norc", "-c",
                'export PATH="$PWD/../tools:$PATH"; '
                '[[ "$(command -v git)" = "$PWD/../tools/git" ]] || exit 98; '
                'exec "$BASH" Helper_Scripts/upgrade.sh "$@"',
                "upgrade-contract", *arguments,
            ],
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        operations = commands.read_text().splitlines() if commands.exists() else []
        return result, operations, repository, temporary

    return run


def test_postcheck_failure_cannot_report_upgrade_success(upgrade_script):
    result, operations, _, _ = upgrade_script("--confirm-offline", postcheck=1)
    assert "python -m Helper_Scripts.upgrade_helpers" in operations  # nosec B101 - pytest actual postcheck witness
    assert result.returncode != 0  # nosec B101 - pytest failure propagation
    assert "Upgrade complete" not in result.stdout  # nosec B101 - pytest truthful status


def test_successful_code_update_keeps_migration_validation_pending(upgrade_script):
    result, operations, repository, temporary = upgrade_script("--confirm-offline")
    assert result.returncode == 0  # nosec B101 - pytest valid update positive control
    assert "git pull origin main" in operations  # nosec B101 - pytest actual update witness
    assert "migrations are pending" in result.stdout.lower()  # nosec B101 - pytest truthful startup boundary
    backups = list((repository / "Backups").glob("upgrade_*/Databases/retained.txt"))
    assert len(backups) == 1 and backups[0].read_text() == "prior release record"  # nosec B101 - pytest backup preservation
    assert not list(temporary.iterdir())  # nosec B101 - pytest own temporary cleanup


def test_mutating_upgrade_requires_stopped_writer_confirmation(upgrade_script):
    result, operations, repository, _ = upgrade_script()
    assert result.returncode != 0  # nosec B101 - pytest safety admission
    assert not operations  # nosec B101 - pytest no Git/pip before confirmation
    assert not (repository / "Backups").exists()  # nosec B101 - pytest no unsafe backup


def test_dry_run_needs_no_offline_confirmation_or_mutations(upgrade_script):
    result, operations, repository, temporary = upgrade_script("--dry-run")
    assert result.returncode == 0  # nosec B101 - pytest read-only positive control
    assert not operations  # nosec B101 - pytest no Git/pip in preflight-only mode
    assert not (repository / "Backups").exists()  # nosec B101 - pytest no backup in dry run
    assert not list(temporary.iterdir())  # nosec B101 - pytest own temporary cleanup


def test_failed_preflight_prevents_update_and_backup(upgrade_script):
    result, operations, repository, temporary = upgrade_script("--confirm-offline", precheck=1)
    assert result.returncode != 0  # nosec B101 - pytest preflight refusal
    assert not operations  # nosec B101 - pytest no Git/pip after refusal
    assert not (repository / "Backups").exists()  # nosec B101 - pytest no backup after refusal
    assert not list(temporary.iterdir())  # nosec B101 - pytest own temporary cleanup


def test_harness_does_not_depend_on_native_python_shebang(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", r"C:\Program Files\Python\python.exe")
    run = upgrade_script.__wrapped__(tmp_path / "path with spaces")
    result, operations, _, _ = run("--confirm-offline")
    assert result.returncode == 0, result.stderr  # nosec B101 - pytest portable launcher witness
    assert "python -m pip install -e ." in operations  # nosec B101 - pytest isolated pip witness


def test_harness_runs_complete_crlf_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout.sh"
    checkout.write_bytes(SCRIPT.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    monkeypatch.setitem(globals(), "SCRIPT", checkout)
    run = upgrade_script.__wrapped__(tmp_path / "path with spaces")
    result, operations, _, _ = run("--confirm-offline")
    assert result.returncode == 0, result.stderr  # nosec B101 - pytest native checkout line endings
    assert "git pull origin main" in operations  # nosec B101 - pytest complete script witness


@pytest.mark.parametrize(
    ("variable", "relative"),
    [("ProgramFiles", "Git/bin/bash.exe"), ("ProgramFiles(x86)", "Git/usr/bin/bash.exe")],
)
def test_bash_discovery_prefers_git_bash_over_wsl(tmp_path, monkeypatch, variable, relative):
    root = tmp_path / "Program Files"
    bash = root / relative
    bash.parent.mkdir(parents=True)
    bash.touch()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setenv(variable, str(root))
    monkeypatch.setattr(shutil, "which", lambda _: "C:/WindowsApps/bash.exe")
    assert _bash_executable() == str(bash)  # nosec B101 - pytest native Git Bash discovery


def test_missing_bash_fails_instead_of_skipping(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("ProgramFiles", raising=False)
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(pytest.fail.Exception, match="Bash .* is required"):
        _bash_executable()
