"""Run the complete upgrade script with isolated external-command fixtures."""

import os
import shutil
import subprocess  # nosec B404 - isolated whole-script subprocess contract tests
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "Helper_Scripts" / "upgrade.sh"


@pytest.fixture
def upgrade_script(tmp_path):
    repository = tmp_path / "installation"
    helpers = repository / "Helper_Scripts"
    helpers.mkdir(parents=True)
    script = helpers / "upgrade.sh"
    shutil.copyfile(SCRIPT, script)
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
        f"#!{sys.executable}\n"
        + r"""import json
import os
import sys
from pathlib import Path
arguments = sys.argv[1:]
if arguments == ["-m", "Helper_Scripts.upgrade_helpers", "--json"]:
    status = int(os.environ.get("UPGRADE_PRECHECK_EXIT", "0"))
    print(json.dumps({"can_proceed": status == 0, "summary": "fixture preflight", "checks": []}))
    raise SystemExit(status)
if arguments and arguments[0] == "-c":
    print("fixture summary")
    raise SystemExit(0)
with Path(os.environ["UPGRADE_TEST_COMMANDS"]).open("a") as output:
    output.write("python " + " ".join(arguments) + "\n")
if arguments == ["-m", "Helper_Scripts.upgrade_helpers"]:
    raise SystemExit(int(os.environ.get("UPGRADE_POSTCHECK_EXIT", "0")))
if arguments == ["-m", "pip", "install", "-e", "."]:
    raise SystemExit(0)
raise SystemExit(99)
"""
    )
    stub.chmod(0o700)
    git = tools / "git"
    git.write_text(
        f"#!{sys.executable}\n"
        + r"""import os
import sys
from pathlib import Path
arguments = sys.argv[1:]
with Path(os.environ["UPGRADE_TEST_COMMANDS"]).open("a") as output:
    output.write("git " + " ".join(arguments) + "\n")
if arguments == ["rev-parse", "--abbrev-ref", "HEAD"]:
    print("main")
elif arguments == ["rev-parse", "HEAD"]:
    print("a" * 40)
elif arguments != ["pull", "origin", "main"]:
    raise SystemExit(99)
"""
    )
    git.chmod(0o700)

    def run(*arguments, precheck=0, postcheck=0):
        environment = os.environ.copy()
        environment.update(
            PYTHON=str(stub),
            PATH=str(tools) + os.pathsep + environment.get("PATH", ""),
            TMPDIR=str(temporary),
            UPGRADE_TEST_COMMANDS=str(commands),
            UPGRADE_PRECHECK_EXIT=str(precheck),
            UPGRADE_POSTCHECK_EXIT=str(postcheck),
        )
        result = subprocess.run(  # nosec B603 - real script with isolated Git/pip fixtures
            ["/bin/bash", str(script), *arguments],
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
