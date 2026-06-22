# MCP Unified Internal RC Artifact Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the standalone MCP package project under `apps/mcp-unified/` and add a private release-candidate pipeline that builds, installs, UATs, and reports on the built package artifacts.

**Architecture:** Keep the root `tldw-server` package and standalone `mcp-unified` package as separate build subjects. The standalone project becomes `apps/mcp-unified/` with a `src/mcp_unified/` import package, while a root-level RC harness owns build, install, UAT, and evidence workflows. Package-boundary tests enforce that the wheel is tested from installed artifacts and that root `mcp_unified/` does not remain a second source tree.

**Tech Stack:** Python 3.10+, setuptools/build/twine, pytest, GitHub Actions, Make, existing `mcp_unified` gateway/smoke CLIs.

---

## File Structure

- Move: `mcp_unified/` -> `apps/mcp-unified/src/mcp_unified/`
- Move: `mcp_unified/pyproject.toml` -> `apps/mcp-unified/pyproject.toml`
- Move: `mcp_unified/pytest-artifact-gate.ini` -> `apps/mcp-unified/pytest-artifact-gate.ini`
- Move: `mcp_unified/README.md` -> `apps/mcp-unified/README.md`
- Move: `mcp_unified/USER_GUIDE.md` -> `apps/mcp-unified/USER_GUIDE.md`
- Create package-resource copies:
  - `apps/mcp-unified/src/mcp_unified/README.md`
  - `apps/mcp-unified/src/mcp_unified/USER_GUIDE.md`
- Create: `Helper_Scripts/mcp_unified_rc.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
- Modify: `.github/tests/test_mcp_unified_artifact_gate.py`
- Modify: `Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py`
- Modify: `Makefile`
- Modify: `.github/workflows/pypi-package.yml`
- Create: `.github/workflows/mcp-unified-rc.yml`

## Task 1: Move The Standalone Package Project Under `apps/`

**Files:**
- Move: `mcp_unified/` -> `apps/mcp-unified/src/mcp_unified/`
- Modify: `apps/mcp-unified/pyproject.toml`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [ ] **Step 1: Write the failing app-location boundary test**

Replace the path constants at the top of `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py` with:

```python
REPO_ROOT = Path(__file__).resolve().parents[5]
STANDALONE_PROJECT_ROOT = REPO_ROOT / "apps" / "mcp-unified"
PACKAGE_ROOT = STANDALONE_PROJECT_ROOT / "src" / "mcp_unified"
STANDALONE_PYPROJECT = STANDALONE_PROJECT_ROOT / "pyproject.toml"
PY_TYPED_MARKER = PACKAGE_ROOT / "py.typed"
PACKAGE_README = STANDALONE_PROJECT_ROOT / "README.md"
PACKAGE_USER_GUIDE = STANDALONE_PROJECT_ROOT / "USER_GUIDE.md"
PACKAGE_RESOURCE_README = PACKAGE_ROOT / "README.md"
PACKAGE_RESOURCE_USER_GUIDE = PACKAGE_ROOT / "USER_GUIDE.md"
```

Add this test below the constants:

```python
def test_mcp_unified_package_project_lives_under_apps() -> None:
    """The standalone project must live under apps/mcp-unified."""

    assert STANDALONE_PROJECT_ROOT.is_dir()  # nosec B101
    assert STANDALONE_PYPROJECT.is_file()  # nosec B101
    assert PACKAGE_ROOT.is_dir()  # nosec B101
    assert not (REPO_ROOT / "mcp_unified").exists()  # nosec B101
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_package_project_lives_under_apps -v
```

Expected: fails because `apps/mcp-unified/` does not exist yet.

- [ ] **Step 3: Move the project files**

Run these mechanical moves from the repository root:

```bash
mkdir -p apps/mcp-unified/src
git mv mcp_unified apps/mcp-unified/src/mcp_unified
git mv apps/mcp-unified/src/mcp_unified/pyproject.toml apps/mcp-unified/pyproject.toml
git mv apps/mcp-unified/src/mcp_unified/pytest-artifact-gate.ini apps/mcp-unified/pytest-artifact-gate.ini
git mv apps/mcp-unified/src/mcp_unified/README.md apps/mcp-unified/README.md
git mv apps/mcp-unified/src/mcp_unified/USER_GUIDE.md apps/mcp-unified/USER_GUIDE.md
cp apps/mcp-unified/README.md apps/mcp-unified/src/mcp_unified/README.md
cp apps/mcp-unified/USER_GUIDE.md apps/mcp-unified/src/mcp_unified/USER_GUIDE.md
git add apps/mcp-unified/src/mcp_unified/README.md apps/mcp-unified/src/mcp_unified/USER_GUIDE.md
```

- [ ] **Step 4: Update the package descriptor for `src/` layout**

In `apps/mcp-unified/pyproject.toml`, replace the current `[tool.setuptools.package-dir]` table with:

```toml
[tool.setuptools.package-dir]
"" = "src"
```

Keep the existing package list, script names, dependencies, extras, and package data:

```toml
[tool.setuptools.package-data]
mcp_unified = ["py.typed", "README.md", "USER_GUIDE.md"]
```

- [ ] **Step 5: Update docs-resource assertions**

In `test_mcp_unified_package_docs_are_local_to_package_boundary`, assert both project docs and package-resource copies:

```python
assert PACKAGE_README.is_file()  # nosec B101
assert PACKAGE_USER_GUIDE.is_file()  # nosec B101
assert PACKAGE_RESOURCE_README.is_file()  # nosec B101
assert PACKAGE_RESOURCE_USER_GUIDE.is_file()  # nosec B101
assert PACKAGE_RESOURCE_README.read_text(encoding="utf-8") == PACKAGE_README.read_text(encoding="utf-8")  # nosec B101
assert PACKAGE_RESOURCE_USER_GUIDE.read_text(encoding="utf-8") == PACKAGE_USER_GUIDE.read_text(encoding="utf-8")  # nosec B101
```

- [ ] **Step 6: Run focused relocation tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_package_project_lives_under_apps \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_package_declares_pep561_typed_marker \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_package_docs_are_local_to_package_boundary \
  -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the package relocation**

Run:

```bash
git add apps/mcp-unified tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
git commit -m "refactor(mcp): move standalone package under apps"
```

## Task 2: Update Artifact Boundary Tests For The App Package Root

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`
- Modify: `.github/tests/test_mcp_unified_artifact_gate.py`

- [ ] **Step 1: Update the artifact build helper to copy the project root**

Change `_build_standalone_distributions` so it copies `STANDALONE_PROJECT_ROOT` instead of `PACKAGE_ROOT`:

```python
def _build_standalone_distributions(tmp_path: Path) -> tuple[Path, Path]:
    """Build standalone MCP Unified wheel and sdist into a temporary directory."""

    _assert_artifact_gate_build_tools_available()

    package_source = tmp_path / "mcp_unified_project"
    shutil.copytree(
        STANDALONE_PROJECT_ROOT,
        package_source,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
            str(package_source),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        },
    )
    _assert_subprocess_succeeded(result, "python -m build")
```

- [ ] **Step 2: Update pyproject assertions for `src/` layout**

In `test_mcp_unified_standalone_pyproject_matches_release_metadata`, replace the package-dir assertions with:

```python
assert setuptools_config["packages"] == [
    "mcp_unified",
    "mcp_unified.federation",
    "mcp_unified.filesystem_locks",
    "mcp_unified.gateway",
    "mcp_unified.interfaces",
    "mcp_unified.profiles",
    "mcp_unified.smoke",
    "mcp_unified.storage",
    "mcp_unified.tool_hooks",
    "mcp_unified.tool_use_reporting",
]  # nosec B101
assert pyproject["tool"]["setuptools"]["package-dir"] == {"": "src"}  # nosec B101
assert pyproject["tool"]["setuptools"]["package-data"] == {  # nosec B101
    "mcp_unified": ["py.typed", "README.md", "USER_GUIDE.md"],
}
```

- [ ] **Step 3: Tighten artifact member assertions**

Update `test_mcp_unified_standalone_sdist_contains_only_package_boundary` to assert allowed project-root members and forbidden host-root members:

```python
assert any(member.endswith("/pyproject.toml") for member in members)  # nosec B101
assert any(member.endswith("/README.md") for member in members)  # nosec B101
assert any(member.endswith("/USER_GUIDE.md") for member in members)  # nosec B101
assert any("/src/mcp_unified/__init__.py" in member for member in members)  # nosec B101
assert not any(member.startswith("tldw_Server_API/") for member in members)  # nosec B101
assert not any(member.startswith("apps/tldw-frontend/") for member in members)  # nosec B101
assert not any(member.startswith("mcp_unified/") for member in members)  # nosec B101
```

- [ ] **Step 4: Update wheel docs assertions**

In `test_mcp_unified_standalone_artifacts_include_package_docs`, assert the wheel contains package-resource docs under `mcp_unified/`:

```python
wheel, sdist = standalone_distributions
wheel_members = _read_wheel_members(wheel)
sdist_members = _read_sdist_members(sdist)

assert "mcp_unified/README.md" in wheel_members  # nosec B101
assert "mcp_unified/USER_GUIDE.md" in wheel_members  # nosec B101
assert any(member.endswith("/README.md") for member in sdist_members)  # nosec B101
assert any(member.endswith("/USER_GUIDE.md") for member in sdist_members)  # nosec B101
```

- [ ] **Step 5: Update the artifact-gate shim comments**

In `.github/tests/test_mcp_unified_artifact_gate.py`, keep the path to host boundary tests but update the docstring to say it validates `apps/mcp-unified` artifacts through the host boundary-test module.

- [ ] **Step 6: Run artifact boundary tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_standalone_pyproject_matches_release_metadata \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_standalone_distribution_metadata_matches_extras \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_standalone_sdist_contains_only_package_boundary \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_standalone_artifacts_include_typed_marker \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_standalone_artifacts_include_package_docs \
  -v
```

Expected: all selected tests pass or skip only when local offline build tools are missing.

- [ ] **Step 7: Commit artifact-boundary updates**

Run:

```bash
git add tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py .github/tests/test_mcp_unified_artifact_gate.py
git commit -m "test(mcp): validate app package artifact boundary"
```

## Task 3: Add The Internal RC Harness

**Files:**
- Create: `Helper_Scripts/mcp_unified_rc.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py`

- [ ] **Step 1: Write harness unit tests first**

Create `tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Helper_Scripts import mcp_unified_rc


def test_default_paths_point_to_apps_package() -> None:
    paths = mcp_unified_rc.RcPaths.from_repo_root(Path("/repo"))

    assert paths.package_project == Path("/repo/apps/mcp-unified")
    assert paths.package_src == Path("/repo/apps/mcp-unified/src/mcp_unified")
    assert paths.evidence_dir == Path("/repo/.artifacts/mcp-unified-rc")


def test_redact_text_removes_secret_like_values() -> None:
    raw = "token=abc123\nAPI_KEY=secret-value\nnormal=value"

    redacted = mcp_unified_rc.redact_text(raw)

    assert "secret-value" not in redacted
    assert "abc123" not in redacted
    assert "normal=value" in redacted


def test_result_recorder_writes_json_and_markdown(tmp_path: Path) -> None:
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="not-published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )
    recorder.record(
        phase="artifact_metadata",
        name="wheel_metadata",
        status="passed",
        duration_ms=12,
    )

    json_path, markdown_path = recorder.write()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["package"]["source_path"] == "apps/mcp-unified"
    assert payload["summary"] == {"passed": 1, "failed": 0, "skipped": 0}
    assert "wheel_metadata" in markdown_path.read_text(encoding="utf-8")


def test_result_recorder_marks_required_failure(tmp_path: Path) -> None:
    recorder = mcp_unified_rc.RcEvidenceRecorder(
        evidence_dir=tmp_path,
        package_name="mcp-unified",
        package_version="0.1.0",
        package_status="internal-experimental",
        publishing_status="not-published",
        commit="abc1234",
        source_path="apps/mcp-unified",
        layout="src",
    )
    recorder.record(
        phase="fresh_install",
        name="normal_install",
        status="failed",
        duration_ms=15,
        reason="pip failed",
    )

    json_path, _markdown_path = recorder.write()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["summary"] == {"passed": 0, "failed": 1, "skipped": 0}
```

- [ ] **Step 2: Run harness tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py -v
```

Expected: import failure because `Helper_Scripts/mcp_unified_rc.py` does not exist.

- [ ] **Step 3: Create the harness module skeleton**

Create `Helper_Scripts/mcp_unified_rc.py` with these public names and keep command execution behind small helpers:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess  # nosec B404
import sys
import time
import venv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

SECRET_LINE_PATTERN = re.compile(r"(?i)(api[_-]?key|token|secret|password)=([^\\s]+)")
RESULT_STATUSES = {"passed", "failed", "skipped"}


@dataclass(frozen=True)
class RcPaths:
    repo_root: Path
    package_project: Path
    package_src: Path
    evidence_dir: Path
    dist_dir: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "RcPaths":
        return cls(
            repo_root=repo_root,
            package_project=repo_root / "apps" / "mcp-unified",
            package_src=repo_root / "apps" / "mcp-unified" / "src" / "mcp_unified",
            evidence_dir=repo_root / ".artifacts" / "mcp-unified-rc",
            dist_dir=repo_root / ".artifacts" / "mcp-unified-rc" / "dist",
        )


@dataclass
class RcCommandResult:
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int


@dataclass
class RcEvidenceRecorder:
    evidence_dir: Path
    package_name: str
    package_version: str
    package_status: str
    publishing_status: str
    commit: str
    source_path: str
    layout: str
    results: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        phase: str,
        name: str,
        status: str,
        duration_ms: int,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if status not in RESULT_STATUSES:
            raise ValueError(f"invalid RC result status: {status}")
        entry: dict[str, Any] = {
            "phase": phase,
            "name": name,
            "status": status,
            "duration_ms": duration_ms,
        }
        if reason:
            entry["reason"] = reason
        if details:
            entry["details"] = details
        self.results.append(entry)

    def write(self) -> tuple[Path, Path]:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "passed": sum(1 for result in self.results if result["status"] == "passed"),
            "failed": sum(1 for result in self.results if result["status"] == "failed"),
            "skipped": sum(1 for result in self.results if result["status"] == "skipped"),
        }
        payload = {
            "schema_version": "1",
            "ok": summary["failed"] == 0,
            "package": {
                "name": self.package_name,
                "version": self.package_version,
                "commit": self.commit,
                "source_path": self.source_path,
                "layout": self.layout,
                "status": self.package_status,
                "publishing_status": self.publishing_status,
            },
            "environment": {
                "os": platform.system(),
                "python": platform.python_version(),
                "runner": os.environ.get("GITHUB_ACTIONS", "local"),
            },
            "results": self.results,
            "summary": summary,
            "known_limitations": [],
        }
        json_path = self.evidence_dir / "mcp-unified-rc-evidence.json"
        markdown_path = self.evidence_dir / "mcp-unified-rc-summary.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        markdown_lines = ["# MCP Unified RC Summary", "", f"OK: {payload['ok']}", ""]
        markdown_lines.extend(
            f"- {result['status']}: {result['phase']} / {result['name']}"
            for result in self.results
        )
        markdown_path.write_text("\\n".join(markdown_lines) + "\\n", encoding="utf-8")
        return json_path, markdown_path


def redact_text(value: str) -> str:
    return SECRET_LINE_PATTERN.sub(lambda match: f"{match.group(1)}=[redacted]", value)
```

- [ ] **Step 4: Add command execution helpers**

In `Helper_Scripts/mcp_unified_rc.py`, add:

```python
def run_command(command: Sequence[str], *, cwd: Path, timeout: int = 180) -> RcCommandResult:
    started = time.perf_counter()
    completed = subprocess.run(  # nosec B603
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )
    return RcCommandResult(
        command=list(command),
        cwd=str(cwd),
        returncode=completed.returncode,
        stdout=redact_text(completed.stdout[-6000:]),
        stderr=redact_text(completed.stderr[-6000:]),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

- [ ] **Step 5: Add `build`, `artifact-gate`, `install-smoke`, `cli-uat`, `smoke-uat`, and `all` subcommands**

Use `argparse` with subcommands named exactly:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcp-unified-rc")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "artifact-gate", "install-smoke", "extras-matrix", "cli-uat", "smoke-uat", "evidence", "all"):
        subparsers.add_parser(name)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = RcPaths.from_repo_root(args.repo_root.resolve())
    if args.command == "build":
        return run_build(paths)
    if args.command == "artifact-gate":
        return run_artifact_gate(paths)
    if args.command == "install-smoke":
        return run_install_smoke(paths)
    if args.command == "extras-matrix":
        return run_extras_matrix(paths)
    if args.command == "cli-uat":
        return run_cli_uat(paths)
    if args.command == "smoke-uat":
        return run_smoke_uat(paths)
    if args.command == "evidence":
        return run_evidence(paths)
    return run_all(paths)


if __name__ == "__main__":
    raise SystemExit(main())
```

Implement each `run_*` function by calling existing CLIs and tests first. Keep advanced behavior small in this slice:

- `run_build`: `python -m build --wheel --sdist --outdir .artifacts/mcp-unified-rc/dist apps/mcp-unified`
- `run_artifact_gate`: `python -m pytest -c apps/mcp-unified/pytest-artifact-gate.ini .github/tests/test_mcp_unified_artifact_gate.py -q`
- `run_install_smoke`: create a temporary venv, install the newest wheel from `dist_dir`, then run `python -c "import mcp_unified"` and both CLI help commands.
- `run_extras_matrix`: loop over `core`, `gateway`, `sqlite`, and `dev`, creating one temporary venv per extra.
- `run_cli_uat`: call `Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py` with the built wheel once Task 5 adds wheel mode.
- `run_smoke_uat`: run `mcp-unified-smoke inprocess --json-report .artifacts/mcp-unified-rc/smoke-inprocess.json` from the install-smoke venv.
- `run_all`: run the other required phases and return nonzero if any required phase records `failed`.

- [ ] **Step 6: Run harness tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the RC harness**

Run:

```bash
git add Helper_Scripts/mcp_unified_rc.py tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py
git commit -m "feat(mcp): add internal RC harness"
```

## Task 4: Add Make Targets And CI Workflow

**Files:**
- Modify: `Makefile`
- Modify: `.github/workflows/pypi-package.yml`
- Create: `.github/workflows/mcp-unified-rc.yml`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [ ] **Step 1: Add workflow/Makefile assertions**

Add this test to `test_runtime_package_boundary.py`:

```python
def test_mcp_unified_rc_workflow_uses_private_permissions() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "mcp-unified-rc.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert workflow["permissions"] == {"contents": "read"}  # nosec B101
    serialized = json.dumps(workflow)
    assert "id-token" not in serialized  # nosec B101
    assert "apps/mcp-unified" in serialized  # nosec B101
    assert "make mcp-unified-rc" in serialized  # nosec B101
```

Add this test:

```python
def test_mcp_unified_make_targets_do_not_call_root_pypi_check() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "mcp-unified-build:" in makefile  # nosec B101
    assert "mcp-unified-check:" in makefile  # nosec B101
    assert "mcp-unified-uat:" in makefile  # nosec B101
    assert "mcp-unified-rc:" in makefile  # nosec B101
    mcp_section = makefile.split("# MCP Unified standalone RC", 1)[1]
    assert "pypi-check" not in mcp_section  # nosec B101
```

- [ ] **Step 2: Run the workflow/Makefile tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_rc_workflow_uses_private_permissions \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_make_targets_do_not_call_root_pypi_check \
  -v
```

Expected: fails because the workflow and targets do not exist yet.

- [ ] **Step 3: Add Make targets**

Add this section near the existing PyPI packaging helpers in `Makefile`:

```make
# -----------------------------------------------------------------------------
# MCP Unified standalone RC
# -----------------------------------------------------------------------------
.PHONY: mcp-unified-build mcp-unified-check mcp-unified-uat mcp-unified-rc

MCP_UNIFIED_RC ?= $(PYTHON) Helper_Scripts/mcp_unified_rc.py

mcp-unified-build:
	$(MCP_UNIFIED_RC) build

mcp-unified-check:
	$(MCP_UNIFIED_RC) build
	$(MCP_UNIFIED_RC) artifact-gate
	$(MCP_UNIFIED_RC) install-smoke

mcp-unified-uat:
	$(MCP_UNIFIED_RC) cli-uat
	$(MCP_UNIFIED_RC) smoke-uat
	$(MCP_UNIFIED_RC) extras-matrix

mcp-unified-rc:
	$(MCP_UNIFIED_RC) all
```

Also add the four target names to the `.PHONY` list near the top and to the `help` output under Testing.

- [ ] **Step 4: Create the private RC workflow**

Create `.github/workflows/mcp-unified-rc.yml`:

```yaml
name: MCP Unified Internal RC

on:
  pull_request:
    branches:
      - main
      - dev
    paths:
      - apps/mcp-unified/**
      - Helper_Scripts/mcp_unified_rc.py
      - Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py
      - .github/tests/test_mcp_unified_artifact_gate.py
      - tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
      - tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py
      - .github/workflows/mcp-unified-rc.yml
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: mcp-unified-rc-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  internal-rc:
    name: Build, install, and UAT standalone MCP package
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - name: Install packaging tools
        run: |
          python -m pip install --upgrade pip
          python -m pip install build twine setuptools wheel pytest pytest-asyncio bandit

      - name: Run internal RC
        run: make mcp-unified-rc

      - name: Upload MCP Unified RC artifacts
        uses: actions/upload-artifact@v7
        with:
          name: mcp-unified-rc
          path: .artifacts/mcp-unified-rc/**
          if-no-files-found: error
```

- [ ] **Step 5: Retire MCP-specific work from the root PyPI package workflow**

In `.github/workflows/pypi-package.yml`, remove `mcp_unified/**` from triggers, remove `pip install -e "mcp_unified[dev]"`, remove the artifact-gate pytest step, and keep `make pypi-check` as root `tldw-server` validation. Rename the artifact upload name from `pypi-dist` only if the workflow name stays ambiguous; use `tldw-server-pypi-dist` when renaming.

- [ ] **Step 6: Run workflow/Makefile tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_rc_workflow_uses_private_permissions \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py::test_mcp_unified_make_targets_do_not_call_root_pypi_check \
  -v
```

Expected: both tests pass.

- [ ] **Step 7: Commit Make and CI changes**

Run:

```bash
git add Makefile .github/workflows/pypi-package.yml .github/workflows/mcp-unified-rc.yml tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
git commit -m "ci(mcp): add internal RC workflow"
```

## Task 5: Update UAT Harness To Install Built Wheels

**Files:**
- Modify: `Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py`

- [ ] **Step 1: Add a unit test for install target selection**

In `test_mcp_unified_rc_harness.py`, add a file-path loader because `Testing-related` contains a hyphen:

```python
def _load_user_guide_harness() -> object:
    import importlib.util

    path = Path("Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py").resolve()
    spec = importlib.util.spec_from_file_location("_mcp_standalone_user_guide_uat", path)
    assert spec is not None and spec.loader is not None  # nosec B101
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_user_guide_uat_install_spec_uses_apps_project_by_default() -> None:
    harness = _load_user_guide_harness()

    assert harness.default_package_project(Path("/repo")) == Path("/repo/apps/mcp-unified")
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=None,
        editable=False,
    ) == ["/repo/apps/mcp-unified[gateway]"]
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=None,
        editable=True,
    ) == ["-e", "/repo/apps/mcp-unified[gateway]"]
    assert harness.package_install_spec(
        repo_root=Path("/repo"),
        wheel_path=Path("/tmp/mcp_unified-0.1.0-py3-none-any.whl"),
        editable=False,
    ) == ["/tmp/mcp_unified-0.1.0-py3-none-any.whl"]
```

- [ ] **Step 2: Add explicit install-spec helpers to the UAT harness**

In `Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py`, add:

```python
def default_package_project(repo_root: Path) -> Path:
    """Return the standalone MCP package project path."""

    return repo_root / "apps" / "mcp-unified"


def package_install_spec(
    *,
    repo_root: Path,
    wheel_path: Path | None,
    editable: bool,
) -> list[str]:
    """Return pip install arguments for the standalone package under test."""

    if wheel_path is not None:
        return [str(wheel_path)]
    project = default_package_project(repo_root)
    if editable:
        return ["-e", f"{project}[gateway]"]
    return [f"{project}[gateway]"]
```

- [ ] **Step 3: Replace the old editable install path**

In `build_uat_plan`, replace:

```python
"-e",
f"{repo_root / 'mcp_unified'}[gateway]",
```

with a `package_install_args` parameter passed into `build_uat_plan`:

```python
package_install_args: list[str],
```

and use:

```python
command=[
    str(venv_python),
    "-m",
    "pip",
    "install",
    *package_install_args,
],
```

- [ ] **Step 4: Add CLI arguments for wheel mode**

Add arguments to the UAT harness parser:

```python
parser.add_argument("--wheel", type=Path, help="Built wheel to install for installed-artifact UAT.")
parser.add_argument("--editable", action="store_true", help="Install the app package project in editable mode for local guide iteration.")
```

When building the plan, pass:

```python
package_install_args=package_install_spec(
    repo_root=repo_root,
    wheel_path=args.wheel,
    editable=args.editable,
)
```

- [ ] **Step 5: Run UAT harness parser smoke**

Run:

```bash
source .venv/bin/activate
python Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py --help
```

Expected: exits 0 and includes `--wheel`.

- [ ] **Step 6: Commit UAT install-mode changes**

Run:

```bash
git add Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py
git commit -m "test(mcp): run UAT against built wheels"
```

## Task 6: Run The Internal RC Gate And Security Checks

**Files:**
- Modify only files that fail validation in changed scope.

- [ ] **Step 1: Run package boundary tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -v
```

Expected: all tests pass or package build tests skip only when offline build tools are missing.

- [ ] **Step 2: Run RC harness tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run the artifact gate**

Run:

```bash
source .venv/bin/activate
python -m pytest -c apps/mcp-unified/pytest-artifact-gate.ini .github/tests/test_mcp_unified_artifact_gate.py -q
```

Expected: all artifact-gate tests pass.

- [ ] **Step 4: Run the local RC target**

Run:

```bash
source .venv/bin/activate
make mcp-unified-rc
```

Expected: exits 0 and writes:

```text
.artifacts/mcp-unified-rc/mcp-unified-rc-evidence.json
.artifacts/mcp-unified-rc/mcp-unified-rc-summary.md
```

- [ ] **Step 5: Run Bandit on touched Python scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r Helper_Scripts/mcp_unified_rc.py Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py -f json -o /tmp/bandit_mcp_unified_rc.json
```

Expected: exits 0 or reports only accepted test-harness subprocess assertions already marked with `# nosec`.

- [ ] **Step 6: Commit final validation fixes**

If validation required any fixes, run:

Stage the files changed by the validation fix. For this plan, the expected validation-fix scope is one or more of:

```bash
git add apps/mcp-unified Helper_Scripts/mcp_unified_rc.py Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py .github/tests/test_mcp_unified_artifact_gate.py .github/workflows/mcp-unified-rc.yml Makefile
git commit -m "fix(mcp): complete internal RC validation"
```

If no fixes were needed, do not create an empty commit.

## Task 7: Final Review And Handoff

**Files:**
- Modify: `backlog/tasks/task-2398 - Plan-MCP-Unified-internal-RC-artifact-pipeline-implementation.md` during execution only when recording results.

- [ ] **Step 1: Inspect final diff**

Run:

```bash
git status --short
git log --oneline --max-count=8
```

Expected: working tree clean and commits are task-sized.

- [ ] **Step 2: Record verification evidence in Backlog**

Update `TASK-2398` or the implementation task used during execution with the exact command names, exit codes, and output summaries. Include the path to `.artifacts/mcp-unified-rc/mcp-unified-rc-evidence.json` when `make mcp-unified-rc` succeeds.

- [ ] **Step 3: Prepare PR summary**

Use this structure:

```markdown
## Change summary

- Moved the standalone MCP package project to `apps/mcp-unified/` using a `src/mcp_unified/` layout so packaging tests no longer rely on root source imports.
- Added an internal RC harness plus Make and GitHub Actions entry points to build, install, UAT, and report on private MCP package artifacts.
- Updated package-boundary, artifact-gate, and UAT coverage so the built wheel is the artifact under test.

## Why

The standalone MCP package needs a private release-candidate path before TestPyPI/PyPI. The new layout and RC gate reduce root-package ambiguity, catch missing wheel files, and preserve evidence for UAT and release decisions.

## Verification

- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -v`
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py -v`
- `python -m pytest -c apps/mcp-unified/pytest-artifact-gate.ini .github/tests/test_mcp_unified_artifact_gate.py -q`
- `make mcp-unified-rc`
- `python -m bandit -r Helper_Scripts/mcp_unified_rc.py Helper_Scripts/Testing-related/mcp_standalone_user_guide_uat.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_mcp_unified_rc_harness.py -f json -o /tmp/bandit_mcp_unified_rc.json`
```
