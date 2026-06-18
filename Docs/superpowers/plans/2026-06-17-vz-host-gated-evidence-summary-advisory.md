# VZ Host-Gated Evidence Summary Advisory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an advisory GitHub step-summary report for VZ Linux host-gated smoke evidence without changing VM execution, artifact upload semantics, or host-gated pass/fail behavior.

**Architecture:** Add a small standard-library Python summarizer under `tools/vz-linux-image/scripts` that reads only known direct child evidence files, renders sanitized Markdown, and exits `0` for all advisory failure modes. Wire the host-gated workflow to run the summarizer with `if: always()` after the smoke wrapper and before artifact uploads, with shell guards for missing checkout/script/interpreter cases. Keep validation portable through focused pytest coverage and workflow contract tests.

**Tech Stack:** Python 3 standard library, pytest, GitHub Actions YAML, Bash workflow shell, Markdown operator docs, Backlog.md task tracking.

---

## File Structure

- Create `tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py`
  - Owns evidence probing, JSON parsing, Markdown rendering, output fallback, and advisory CLI exit behavior.
  - Must not import project modules or require the repo package to be installed.
- Create `tools/vz-linux-image/tests/test_summarize_host_e2e_evidence.py`
  - Owns portable unit/CLI tests for complete, missing, malformed, unsafe, and fallback evidence cases.
- Modify `.github/workflows/vz-linux-host-gated.yml`
  - Adds the always-run summary step between smoke execution and artifact uploads.
- Modify `tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py`
  - Adds workflow contract tests for summary step ordering, guarded invocation, advisory behavior, and evidence path.
- Modify `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
  - Documents the GitHub step summary as the first inline diagnostic surface and preserves artifact inspection order.
- Modify `backlog/tasks/task-2381 - Add-advisory-VZ-host-smoke-evidence-summary.md`
  - Tracks plan path, touched files, verification, known skips, and final status.

Do not modify `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` in this slice unless an implementation issue proves the evidence contract is wrong. The summarizer consumes existing evidence; it does not generate evidence.

---

### Task 1: Add Failing Summarizer Tests

**Files:**
- Create: `tools/vz-linux-image/tests/test_summarize_host_e2e_evidence.py`
- Read: `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh`
- Read: `Docs/superpowers/specs/2026-06-17-vz-host-gated-evidence-summary-advisory-design.md`

- [ ] **Step 1: Create test scaffolding**

Create the test file with helpers that run the future script through `subprocess.run`:

```python
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

IMAGE_DIR = Path(__file__).resolve().parents[1]
SUMMARY_SCRIPT = IMAGE_DIR / "scripts" / "summarize-host-e2e-evidence.py"
EXPECTED_EVIDENCE_FILES = {
    "host-smoke-evidence.json",
    "source-bundle-hashes-before.txt",
    "source-bundle-hashes-after.txt",
    "run-bundle-hashes.txt",
    "runtime-paths.txt",
    "cleanup-status.txt",
}


def _run_summary(
    evidence_dir: Path,
    *,
    summary_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if summary_path is not None:
        env["GITHUB_STEP_SUMMARY"] = str(summary_path)
    else:
        env.pop("GITHUB_STEP_SUMMARY", None)
    return subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT), "--evidence-dir", str(evidence_dir)],
        cwd=IMAGE_DIR,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
```

- [ ] **Step 2: Add complete evidence test**

Add a test that creates all expected files and writes representative JSON:

```python
def test_summary_reports_complete_evidence(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    for evidence_file in EXPECTED_EVIDENCE_FILES - {"host-smoke-evidence.json"}:
        (evidence_dir / evidence_file).write_text(f"{evidence_file}\n", encoding="utf-8")
    (evidence_dir / "host-smoke-evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": "2026-06-17T00:00:00Z",
                "source_bundle_path": "/private/source/bundle",
                "run_bundle_path": "/private/run/bundle",
                "image_store_root": "/private/image-store",
                "smoke_run_id": "ci-123",
                "socket_path": "/private/runtime/helper.sock",
                "serial_log_dir": "/private/runtime/serial",
                "evidence_dir": str(evidence_dir),
                "helper_path": "/private/helper",
                "helper_pid_file": "/private/helper.pid",
                "skip_build": False,
                "skip_sign": False,
                "include_failure_drills": False,
                "final_exit_code": 7,
                "phases": {
                    "real_host_smoke": {
                        "status": "failed",
                        "exit_code": 7,
                        "timestamp": "2026-06-17T00:00:01Z",
                    },
                    "cleanup": {
                        "status": "ok",
                        "exit_code": 0,
                        "timestamp": "2026-06-17T00:00:02Z",
                    },
                },
                "cleanup": {
                    "status": 0,
                    "helper_pid": "123",
                    "helper_running_after_cleanup": False,
                    "socket_present_after_cleanup": False,
                },
                "evidence_files": {
                    evidence_file: str(evidence_dir / evidence_file)
                    for evidence_file in EXPECTED_EVIDENCE_FILES
                },
                "log_artifacts": [
                    {
                        "path": "/private/runtime/serial/vm.log",
                        "size_bytes": 128,
                        "sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run_summary(evidence_dir)

    assert result.returncode == 0, result.stderr
    assert "VZ Linux Host Smoke Evidence Summary" in result.stdout
    assert "Advisory only" in result.stdout
    assert "final_exit_code" in result.stdout
    assert "7" in result.stdout
    assert "real_host_smoke" in result.stdout
    assert "cleanup" in result.stdout
    assert "vz-linux-host-gated-evidence" in result.stdout
    for evidence_file in EXPECTED_EVIDENCE_FILES:
        assert evidence_file in result.stdout
```

- [ ] **Step 3: Add missing and partial evidence tests**

Add tests for:

- nonexistent evidence directory
- evidence path that exists but is a regular file
- evidence directory with only `cleanup-status.txt`

Expected assertions:

- return code is `0`
- output contains `warning`
- output contains expected file names
- no traceback is printed

- [ ] **Step 4: Add malformed, oversized, and symlink tests**

Add tests for:

- malformed `host-smoke-evidence.json`
- JSON larger than `1 MiB`
- `host-smoke-evidence.json` symlink to another file
- expected evidence file that is a directory

Expected assertions:

- return code is `0`
- output warns about the specific unsafe/unavailable file
- symlink target content is not included in output
- no traceback is printed

- [ ] **Step 5: Add summary output fallback tests**

Add tests for:

- valid `GITHUB_STEP_SUMMARY` path appends Markdown to the file and keeps stdout minimal or empty
- invalid/unwritable `GITHUB_STEP_SUMMARY` falls back to stdout or stderr and exits `0`

Use a directory path as `GITHUB_STEP_SUMMARY` for the unwritable target case, because opening a directory for append should fail portably.

- [ ] **Step 6: Run tests to verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/vz-linux-image/tests/test_summarize_host_e2e_evidence.py -q
```

Expected: FAIL because `tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py` does not exist yet.

---

### Task 2: Implement Advisory Evidence Summarizer

**Files:**
- Create: `tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py`
- Test: `tools/vz-linux-image/tests/test_summarize_host_e2e_evidence.py`

- [ ] **Step 1: Add constants and dataclasses**

Create the script with standard-library-only imports:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_EVIDENCE_FILES = (
    "host-smoke-evidence.json",
    "source-bundle-hashes-before.txt",
    "source-bundle-hashes-after.txt",
    "run-bundle-hashes.txt",
    "runtime-paths.txt",
    "cleanup-status.txt",
)
JSON_MAX_BYTES = 1024 * 1024
DISPLAY_MAX_CHARS = 240


@dataclass(frozen=True)
class EvidenceFileStatus:
    name: str
    present: bool
    readable: bool
    reason: str
    size_bytes: int | None = None
```

- [ ] **Step 2: Implement safe rendering helpers**

Add helpers:

```python
def _display(value: object, *, max_chars: int = DISPLAY_MAX_CHARS) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.replace("\r", "\n").splitlines())
    text = html.escape(text, quote=False)
    text = text.replace("|", "\\|")
    if len(text) > max_chars:
        return text[: max_chars - 1] + "..."
    return text
```

Also add a helper that renders Markdown tables from rows and applies `_display` to every cell.

- [ ] **Step 3: Implement direct-child evidence probing**

Implement:

```python
def _probe_evidence_dir(evidence_dir: Path) -> tuple[bool, list[str]]:
    try:
        metadata = evidence_dir.lstat()
    except FileNotFoundError:
        return False, [f"warning: evidence directory is missing: {evidence_dir}"]
    except OSError as exc:
        return False, [f"warning: evidence directory cannot be inspected: {type(exc).__name__}: {exc}"]
    if stat.S_ISLNK(metadata.st_mode):
        return False, [f"warning: evidence directory is a symlink and was not read: {evidence_dir}"]
    if not stat.S_ISDIR(metadata.st_mode):
        return False, [f"warning: evidence path is not a directory and was not read: {evidence_dir}"]
    return True, []

def _probe_expected_file(evidence_dir: Path, name: str) -> EvidenceFileStatus:
    path = evidence_dir / name
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return EvidenceFileStatus(name=name, present=False, readable=False, reason="missing")
    except OSError as exc:
        return EvidenceFileStatus(name=name, present=False, readable=False, reason=type(exc).__name__)
    if stat.S_ISLNK(metadata.st_mode):
        return EvidenceFileStatus(name=name, present=True, readable=False, reason="symlink skipped")
    if not stat.S_ISREG(metadata.st_mode):
        return EvidenceFileStatus(name=name, present=True, readable=False, reason="non-regular file skipped")
    return EvidenceFileStatus(
        name=name,
        present=True,
        readable=True,
        reason="ok",
        size_bytes=metadata.st_size,
    )
```

If `_probe_evidence_dir` returns `False`, render the warning plus a missing
checklist for all expected files and do not call `_probe_expected_file`. Use
`lstat`; do not use recursive globbing; do not resolve or read through symlinks.

- [ ] **Step 4: Implement bounded JSON loading**

Implement:

```python
def _load_evidence_json(
    evidence_dir: Path,
    file_statuses: dict[str, EvidenceFileStatus],
) -> tuple[dict[str, Any] | None, list[str]]:
    json_status = file_statuses["host-smoke-evidence.json"]
    if not json_status.readable:
        return None, [f"structured metadata unavailable: {json_status.reason}"]
    if json_status.size_bytes is not None and json_status.size_bytes > JSON_MAX_BYTES:
        return None, [f"structured metadata skipped: exceeds {JSON_MAX_BYTES} bytes"]
    try:
        raw_text = (evidence_dir / "host-smoke-evidence.json").read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [f"structured metadata parse failed: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return None, ["structured metadata parse failed: top-level JSON is not an object"]
    return payload, []
```

Do not include raw JSON content in warnings.

- [ ] **Step 5: Implement Markdown report generation**

Build `render_summary(evidence_dir: Path) -> str` with these sections:

- `# VZ Linux Host Smoke Evidence Summary`
- advisory notice
- evidence directory path
- warning block when any warnings exist
- final exit code and smoke run id when JSON exists
- expected files checklist table with `present/readable/reason/size`
- phase outcomes table when `phases` is a dict
- cleanup table when `cleanup` is a dict
- runtime/artifact pointers table for known JSON keys:
  - `source_bundle_path`
  - `run_bundle_path`
  - `image_store_root`
  - `socket_path`
  - `serial_log_dir`
  - `helper_pid_file`
  - `evidence_dir`
- log artifact table from `log_artifacts` when it is a list of dicts
- footer that names `vz-linux-host-gated-evidence` as the primary artifact and `vz-linux-host-gated-helper-logs` as the raw-log fallback

Every dynamic value must pass through `_display`.

- [ ] **Step 6: Implement output and advisory CLI boundary**

Implement:

```python
def _write_summary(markdown: str, summary_path: str | None) -> None:
    if not summary_path:
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")
        return
    try:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(markdown)
            if not markdown.endswith("\n"):
                handle.write("\n")
    except OSError as exc:
        print(
            f"warning: unable to append to GITHUB_STEP_SUMMARY: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args(argv)
    try:
        markdown = render_summary(Path(args.evidence_dir))
        _write_summary(markdown, os.environ.get("GITHUB_STEP_SUMMARY"))
    except Exception as exc:  # advisory CLI boundary; do not mask smoke failures
        print(f"warning: evidence summary unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0
```

Keep the broad `Exception` catch only at the CLI boundary. Do not use broad catches inside core helpers when narrower exceptions are available.

- [ ] **Step 7: Run focused GREEN tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/vz-linux-image/tests/test_summarize_host_e2e_evidence.py -q
python tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py --evidence-dir /tmp/does-not-exist >/tmp/vz-summary.md
```

Expected: tests pass; the CLI exits `0` and writes an advisory missing-evidence summary.

- [ ] **Step 8: Commit summarizer and tests**

```bash
git add tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py \
  tools/vz-linux-image/tests/test_summarize_host_e2e_evidence.py
git commit -m "tools: summarize VZ host smoke evidence"
```

---

### Task 3: Wire Host-Gated Workflow Summary Step

**Files:**
- Modify: `tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py`
- Modify: `.github/workflows/vz-linux-host-gated.yml`

- [ ] **Step 1: Add failing workflow contract tests**

Add helper if needed:

```python
def _workflow_step_names(steps: list[dict[str, Any]]) -> list[str]:
    return [str(step.get("name", "")) for step in steps]
```

Add a test:

```python
def test_vz_linux_host_gated_workflow_summarizes_evidence_before_uploads() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["vz-linux-host-gated-smoke"]["steps"]
    names = _workflow_step_names(steps)

    assert "Run managed host smoke" in names
    assert "Summarize smoke evidence" in names
    assert "Upload smoke evidence" in names
    assert names.index("Run managed host smoke") < names.index("Summarize smoke evidence")
    assert names.index("Summarize smoke evidence") < names.index("Upload smoke evidence")

    summary_step = steps[names.index("Summarize smoke evidence")]
    assert summary_step["if"] == "always()"
    assert summary_step["shell"] == "bash"
    run_block = summary_step["run"]
    assert "summarize-host-e2e-evidence.py" in run_block
    assert '${RUNNER_TEMP}/tldw-vz-helper-ci/evidence' in run_block
    assert "GITHUB_STEP_SUMMARY" in run_block
```

Add a second test:

```python
def test_vz_linux_host_gated_workflow_summary_step_is_guarded() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["vz-linux-host-gated-smoke"]["steps"]
    run_block = _workflow_step_run(steps, "Summarize smoke evidence")

    assert "command -v python" in run_block
    assert "command -v python3" in run_block
    assert "[[ -f" in run_block
    assert "exit 0" in run_block
    assert "pip install" not in run_block
    assert "${{ runner.temp }}/tldw-vz-helper-ci/**" not in run_block
```

- [ ] **Step 2: Run workflow tests to verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
```

Expected: FAIL because the workflow has no summary step yet.

- [ ] **Step 3: Add workflow step**

Insert after `Run managed host smoke` and before `Upload smoke evidence`:

```yaml
      - name: Summarize smoke evidence
        if: always()
        shell: bash
        run: |
          set +e
          evidence_dir="${RUNNER_TEMP}/tldw-vz-helper-ci/evidence"
          summary_script="tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py"
          append_summary_warning() {
            local message="$1"
            if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
              {
                echo "# VZ Linux Host Smoke Evidence Summary"
                echo
                echo "> Advisory only: ${message}"
              } >> "${GITHUB_STEP_SUMMARY}" 2>/dev/null || true
            fi
            echo "warning: ${message}" >&2
          }

          python_bin=""
          if command -v python >/dev/null 2>&1; then
            python_bin="$(command -v python)"
          elif command -v python3 >/dev/null 2>&1; then
            python_bin="$(command -v python3)"
          fi

          if [[ ! -f "${summary_script}" ]]; then
            append_summary_warning "evidence summary script is unavailable; inspect uploaded artifacts when present"
            exit 0
          fi
          if [[ -z "${python_bin}" ]]; then
            append_summary_warning "python interpreter is unavailable; inspect uploaded artifacts when present"
            exit 0
          fi

          "${python_bin}" "${summary_script}" --evidence-dir "${evidence_dir}" || true
          exit 0
```

Do not install dependencies. Keep `set +e`, explicit `exit 0`, and `|| true` so this advisory step cannot become the job's primary failure.

- [ ] **Step 4: Run workflow GREEN tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit workflow wiring**

```bash
git add .github/workflows/vz-linux-host-gated.yml \
  tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py
git commit -m "ci: summarize VZ host smoke evidence"
```

---

### Task 4: Update Docs, Backlog, And Final Verification

**Files:**
- Modify: `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- Modify: `backlog/tasks/task-2381 - Add-advisory-VZ-host-smoke-evidence-summary.md`
- Read: `Docs/superpowers/specs/2026-06-17-vz-host-gated-evidence-summary-advisory-design.md`
- Read: `Docs/superpowers/plans/2026-06-17-vz-host-gated-evidence-summary-advisory.md`

- [ ] **Step 1: Add policy docs test if not already covered**

If the existing workflow policy tests do not cover the summary, add an assertion to `test_vz_linux_host_gated_policy_prioritizes_evidence_artifact` or a new focused test requiring:

- `GitHub step summary`
- `advisory`
- `vz-linux-host-gated-evidence`
- `vz-linux-host-gated-helper-logs`
- evidence summary does not replace artifacts

- [ ] **Step 2: Update host-gated policy**

In `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`, document the operator order:

1. Inspect the GitHub Actions step summary for an advisory run overview.
2. Inspect `vz-linux-host-gated-evidence` for structured evidence files.
3. Inspect `vz-linux-host-gated-helper-logs` for raw serial/helper logs only when needed.

State that missing/malformed summary output in this first slice is advisory and should not change the smoke step result.

- [ ] **Step 3: Run focused verification**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tools/vz-linux-image/tests/test_summarize_host_e2e_evidence.py \
  tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py -q
bash -n tools/vz-linux-image/scripts/run-host-e2e-smoke.sh
python tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py --evidence-dir /tmp/does-not-exist >/tmp/tldw-vz-evidence-summary-smoke.md
git diff --check
```

Expected:

- summarizer tests pass
- workflow/doc contract tests pass
- smoke wrapper shell syntax remains valid
- missing evidence CLI smoke exits `0`
- diff check is clean

- [ ] **Step 4: Run Bandit on touched production Python**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py -f json -o /tmp/bandit_vz_evidence_summary.json
```

Expected: no new security findings in the touched script. If Bandit is not installed, install/use the repo dev environment only if that is already the project pattern; otherwise record the exact skip reason in Backlog.

- [ ] **Step 5: Update Backlog task**

Update `TASK-2381`:

- mark acceptance criteria complete when satisfied
- record touched files
- record exact verification commands and results
- record Bandit result or skip reason
- record real VZ smoke as not run because this is host-independent workflow/reporting only
- add final summary

- [ ] **Step 6: Commit docs/backlog/final verification**

```bash
git add Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md \
  tldw_Server_API/tests/Infrastructure/test_vz_linux_host_gated_workflow.py \
  Docs/superpowers/plans/2026-06-17-vz-host-gated-evidence-summary-advisory.md \
  "backlog/tasks/task-2381 - Add-advisory-VZ-host-smoke-evidence-summary.md"
git commit -m "docs: document VZ evidence summary workflow"
```

- [ ] **Step 7: Final branch review**

Run:

```bash
git status --short
git log --oneline dev..HEAD
```

Expected: clean worktree with a small stack of commits for spec, plan, tests/implementation, workflow, and docs/backlog.

---

## PR Checklist

- Keep the PR scoped to advisory evidence summary only.
- Do not run real VZ VM smoke as part of normal local verification unless the user explicitly asks for prepared-host validation.
- Do not change host-gated triggers, self-hosted runner labels, artifact upload paths, helper lifecycle, image-store clone behavior, or evidence generation schema.
- Include a human-owned `Change summary` placeholder in the PR body because AI-generated PRs are merge-blocked without a human-written rationale.
