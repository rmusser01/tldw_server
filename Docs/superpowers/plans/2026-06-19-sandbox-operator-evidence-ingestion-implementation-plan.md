# Sandbox Operator Evidence Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `GET /api/v1/sandbox/admin/operator-status` with bounded, advisory host-gated VZ smoke evidence when `TLDW_SANDBOX_VZ_EVIDENCE_DIR` is configured.

**Architecture:** Add one focused parser module that safely reads only the configured evidence directory and normalizes untrusted `host-smoke-evidence.json` into a bounded summary. Extend `operator_status.py` to project that summary into the existing `sections.evidence` placeholder, and update `SandboxService.operator_status()` to collect evidence after runtime/macOS diagnostics with operational failure isolation.

**Tech Stack:** Python 3, FastAPI service layer, Pydantic response models with extensible section fields, pytest, descriptor-relative filesystem operations via `os.open`/`os.stat`, Bandit.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-06-19-sandbox-operator-evidence-ingestion-design.md`
- Parent operator-status design: `Docs/superpowers/specs/2026-06-18-sandbox-operator-status-consolidation-design.md`
- Backlog task: `TASK-2392`

## File Structure

- Create `tldw_Server_API/app/core/Sandbox/operator_evidence.py`
  - Owns env lookup, descriptor-safe evidence directory traversal, fixed child-file probing, bounded JSON read, metadata normalization, timestamp/staleness classification, and stable parser reason codes.
  - Exposes `collect_operator_evidence(environ: Mapping[str, str] | None = None, now: datetime | None = None) -> dict[str, object]`.
  - Does not import the CLI summarizer and does not render Markdown.

- Modify `tldw_Server_API/app/core/Sandbox/operator_status.py`
  - Add optional `evidence_summary: Mapping[str, Any] | None = None` to `build_operator_status()`.
  - Add `_project_evidence_section(...)` and small helper functions for evidence status/action/overall impact.
  - Keep projection-only logic here; do not add filesystem reads.

- Modify `tldw_Server_API/app/core/Sandbox/service.py`
  - Import `collect_operator_evidence`.
  - In `SandboxService.operator_status()`, collect evidence after runtime/macOS diagnostics.
  - Catch expected operational evidence exceptions as section-local evidence failure; propagate programming errors.

- Add `tldw_Server_API/tests/sandbox/test_operator_evidence.py`
  - Parser-focused portable unit tests. No real VZ execution.

- Modify `tldw_Server_API/tests/sandbox/test_operator_status.py`
  - Projection and service integration tests.

- Optionally modify `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`
  - Only if endpoint-level schema coverage needs an evidence assertion; the endpoint can remain unchanged because it already calls `SandboxService.operator_status()`.

- Modify `backlog/tasks/task-2392 - Implement-sandbox-operator-status-evidence-ingestion-Slice-2.md`
  - Keep notes and final verification current.

## Implementation Constraints

- Do not run VMs, helper lifecycle commands, repair, cleanup, or smoke commands.
- Do not accept evidence paths from requests.
- Do not recursively scan evidence/artifact directories.
- Do not parse Markdown summary files.
- Do not import `tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py`.
- Do not expose raw logs, arbitrary JSON keys, environment variables, helper command lines, guest output, or `helper_path`.
- Treat evidence JSON as external input even though it is produced by a local helper workflow.
- Use stable reason codes in API payloads; logs may include exception context, API payloads should not include raw exception strings.
- Do not commit an intermediate implementation that reads configured evidence
  with ordinary symlink-following `Path.read_text()` or `Path.open()`. The
  first parser commit must already use descriptor-safe direct-child reads for
  configured evidence.

## Shared Parser Contract

Implement these constants in `operator_evidence.py`:

```python
ENV_VZ_EVIDENCE_DIR = "TLDW_SANDBOX_VZ_EVIDENCE_DIR"
SOURCE_HOST_SMOKE_EVIDENCE = "host_smoke_evidence"
JSON_MAX_BYTES = 1024 * 1024
DISPLAY_MAX_CHARS = 240
MAX_PHASES = 16
STALE_AFTER_SECONDS = 7 * 24 * 60 * 60

EXPECTED_EVIDENCE_FILES = (
    "host-smoke-evidence.json",
    "source-bundle-hashes-before.txt",
    "source-bundle-hashes-after.txt",
    "run-bundle-hashes.txt",
    "runtime-paths.txt",
    "cleanup-status.txt",
)

RUNTIME_POINTER_KEYS = (
    "source_bundle_path",
    "run_bundle_path",
    "image_store_root",
    "socket_path",
    "serial_log_dir",
    "helper_pid_file",
    "evidence_dir",
)

CLEANUP_KEYS = (
    "status",
    "helper_pid",
    "helper_running_after_cleanup",
    "socket_present_after_cleanup",
)
```

Recommended normalized summary shape:

```python
{
    "configured": True,
    "source": "host_smoke_evidence",
    "evidence_dir": "/bounded/configured/path",
    "available": True,
    "valid": True,
    "schema_version": 1,
    "created_at": "2026-06-19T12:00:00+00:00",
    "age_seconds": 60,
    "stale": False,
    "smoke_run_id": "run-123",
    "final_exit_code": 0,
    "phases": {"boot": {"status": "ok", "exit_code": 0, "timestamp": "..."}},
    "cleanup": {"status": 0, "helper_pid": "123"},
    "runtime_pointers": {"socket_path": "/bounded/path"},
    "expected_files": {
        "host-smoke-evidence.json": {
            "present": True,
            "readable": True,
            "reason": "ok",
            "size_bytes": 1234,
        },
    },
    "skip_flags": {
        "skip_build": False,
        "skip_sign": False,
        "include_failure_drills": False,
    },
    "reasons": [],
}
```

For unconfigured env, return:

```python
{
    "configured": False,
    "source": "host_smoke_evidence",
    "available": False,
    "valid": False,
    "reasons": ["evidence_not_configured"],
}
```

For configured but invalid/unavailable evidence, return `configured=True`, `available=False` or `valid=False`, and stable reasons such as:

- `evidence_path_contains_nul`
- `evidence_directory_missing`
- `evidence_directory_not_directory`
- `evidence_directory_symlink`
- `evidence_directory_unreadable`
- `evidence_safe_open_unavailable`
- `evidence_json_missing`
- `evidence_json_symlink`
- `evidence_json_non_regular`
- `evidence_json_oversized`
- `evidence_json_malformed_utf8`
- `evidence_json_malformed`
- `evidence_json_top_level_not_object`
- `evidence_schema_version_missing`
- `evidence_schema_version_unsupported`
- `evidence_created_at_malformed`
- `evidence_created_at_in_future`
- `evidence_final_exit_code_invalid`
- `evidence_skip_flag_invalid`

## Task 1: Parser Skeleton And Happy Path

**Files:**
- Create: `tldw_Server_API/app/core/Sandbox/operator_evidence.py`
- Create: `tldw_Server_API/tests/sandbox/test_operator_evidence.py`

- [ ] **Step 1: Write failing tests for unconfigured and valid evidence**

Add `test_operator_evidence.py` with helpers:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.operator_evidence import (
    ENV_VZ_EVIDENCE_DIR,
    collect_operator_evidence,
    _dir_fd_operations_available,
)


NOW = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)


def _valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": "2026-06-19T11:59:00+00:00",
        "source_bundle_path": "/tmp/source",
        "run_bundle_path": "/tmp/run",
        "image_store_root": "/tmp/store",
        "smoke_run_id": "smoke-123",
        "socket_path": "/tmp/helper.sock",
        "serial_log_dir": "/tmp/serial",
        "evidence_dir": "/tmp/evidence",
        "helper_path": "/tmp/private-helper-path-must-not-leak",
        "helper_pid_file": "/tmp/helper.pid",
        "skip_build": False,
        "skip_sign": False,
        "include_failure_drills": False,
        "final_exit_code": 0,
        "phases": {
            "build": {"status": "ok", "exit_code": 0, "timestamp": "2026-06-19T11:59:10Z"},
        },
        "cleanup": {
            "status": 0,
            "helper_pid": "123",
            "helper_running_after_cleanup": False,
            "socket_present_after_cleanup": False,
        },
        "evidence_files": {},
        "log_artifacts": [{"path": "/tmp/serial.log", "size_bytes": 10, "sha256": "abc"}],
    }
    payload.update(overrides)
    return payload


def _write_evidence(root: Path, payload: dict[str, object]) -> None:
    root.mkdir(mode=0o700)
    (root / "host-smoke-evidence.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    for name in (
        "source-bundle-hashes-before.txt",
        "source-bundle-hashes-after.txt",
        "run-bundle-hashes.txt",
        "runtime-paths.txt",
        "cleanup-status.txt",
    ):
        (root / name).write_text("ok\n", encoding="utf-8")


def test_collect_operator_evidence_unconfigured() -> None:
    summary = collect_operator_evidence(environ={}, now=NOW)

    assert summary["configured"] is False
    assert summary["source"] == "host_smoke_evidence"
    assert summary["reasons"] == ["evidence_not_configured"]


def test_collect_operator_evidence_valid_bundle(tmp_path: Path) -> None:
    if not _dir_fd_operations_available():
        pytest.skip("descriptor-safe directory operations are unavailable")
    _write_evidence(tmp_path, _valid_payload())

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(tmp_path)},
        now=NOW,
    )

    assert summary["configured"] is True
    assert summary["available"] is True
    assert summary["valid"] is True
    assert summary["schema_version"] == 1
    assert summary["age_seconds"] == 60
    assert summary["final_exit_code"] == 0
    assert summary["skip_flags"]["include_failure_drills"] is False
    assert "helper_path" not in summary.get("runtime_pointers", {})
    assert summary["expected_files"]["host-smoke-evidence.json"]["readable"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_evidence.py -q
```

Expected: import failure for missing `operator_evidence.py`.

- [ ] **Step 3: Add minimal descriptor-safe parser skeleton**

Implement:

- constants from the shared parser contract
- `_bounded_str(value: object) -> str`
- `_safe_bool(value: object) -> bool | None`
- `_safe_int(value: object) -> int | None`
- `_dir_fd_operations_available()`
- descriptor-safe `_open_evidence_dir(...)`, `_probe_expected_file(...)`, and
  `_read_json_bytes(...)` helpers sufficient for the valid direct-child bundle
  case
- `collect_operator_evidence(...)`

Keep the first implementation minimal enough to pass the happy-path tests, but
do not use ordinary symlink-following `Path.read_text()` or `Path.open()` for
configured evidence. Task 2 adds exhaustive negative coverage, but Task 1 must
already fail closed if descriptor-safe directory operations are unavailable.

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_evidence.py -q
```

Expected: Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add tldw_Server_API/app/core/Sandbox/operator_evidence.py \
  tldw_Server_API/tests/sandbox/test_operator_evidence.py
git commit -m "feat: add sandbox operator evidence parser skeleton"
```

## Task 2: Descriptor-Safe Filesystem Validation

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/operator_evidence.py`
- Modify: `tldw_Server_API/tests/sandbox/test_operator_evidence.py`

- [ ] **Step 1: Add failing filesystem safety tests**

Add tests for:

```python
def test_collect_operator_evidence_rejects_nul_path() -> None:
    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: "/tmp/bad\0path"},
        now=NOW,
    )
    assert summary["configured"] is True
    assert summary["available"] is False
    assert "evidence_path_contains_nul" in summary["reasons"]


def test_collect_operator_evidence_reports_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(missing)},
        now=NOW,
    )
    assert summary["available"] is False
    assert "evidence_directory_missing" in summary["reasons"]


def test_collect_operator_evidence_rejects_directory_symlink(tmp_path: Path) -> None:
    if not _dir_fd_operations_available():
        pytest.skip("descriptor-safe directory operations are unavailable")
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(link)},
        now=NOW,
    )

    assert summary["available"] is False
    assert "evidence_directory_symlink" in summary["reasons"]


def test_collect_operator_evidence_rejects_json_symlink(tmp_path: Path) -> None:
    if not _dir_fd_operations_available():
        pytest.skip("descriptor-safe directory operations are unavailable")
    _write_evidence(tmp_path, _valid_payload())
    (tmp_path / "target.json").write_text("{}", encoding="utf-8")
    (tmp_path / "host-smoke-evidence.json").unlink()
    try:
        (tmp_path / "host-smoke-evidence.json").symlink_to(tmp_path / "target.json")
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(tmp_path)},
        now=NOW,
    )

    assert summary["valid"] is False
    assert "evidence_json_symlink" in summary["reasons"]
```

Also monkeypatch the safe-open capability helper:

```python
def test_collect_operator_evidence_fails_closed_without_safe_open(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_evidence(tmp_path, _valid_payload())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.operator_evidence._dir_fd_operations_available",
        lambda: False,
    )

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(tmp_path)},
        now=NOW,
    )

    assert summary["available"] is False
    assert "evidence_safe_open_unavailable" in summary["reasons"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_evidence.py -q
```

Expected: safety tests fail until descriptor-safe traversal is implemented.

- [ ] **Step 3: Complete descriptor-safe traversal**

Complete the safe traversal helpers introduced in Task 1 by porting the
remaining safe traversal ideas from
`tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py`, while keeping
implementation local and server-focused:

- `_dir_fd_operations_available()`
- `_open_dir_flags()`
- `_open_json_flags()`
- `_normalize_macos_temp_alias(path: Path) -> Path`
- `_evidence_dir_components(path: Path) -> tuple[str, list[str], str | None]`
- `_open_evidence_dir(path: Path) -> EvidenceDirHandle | summary failure`
- `_probe_expected_file(handle, name) -> dict[str, object]`
- `_read_json_bytes(handle) -> bytes | None plus reason`

Important details:

- Use `os.open(..., dir_fd=fd)` and `os.stat(..., follow_symlinks=False)` for direct child inspection.
- Reject `..` components.
- Fail closed if safe descriptor operations are not available.
- Cap `host-smoke-evidence.json` at `JSON_MAX_BYTES` using metadata and read length.
- Do not recurse.

- [ ] **Step 4: Run parser tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_evidence.py -q
```

Expected: parser tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add tldw_Server_API/app/core/Sandbox/operator_evidence.py \
  tldw_Server_API/tests/sandbox/test_operator_evidence.py
git commit -m "feat: validate sandbox evidence paths safely"
```

## Task 3: JSON Normalization, Bounds, And Staleness

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/operator_evidence.py`
- Modify: `tldw_Server_API/tests/sandbox/test_operator_evidence.py`

- [ ] **Step 1: Add failing normalization tests**

Add tests for:

- oversized JSON returns `evidence_json_oversized`
- malformed UTF-8 and malformed JSON do not leak raw input
- top-level JSON array returns `evidence_json_top_level_not_object`
- missing schema version returns `evidence_schema_version_missing`
- unsupported schema version returns `evidence_schema_version_unsupported`
- boolean/container `final_exit_code` returns `evidence_final_exit_code_invalid`
- non-boolean skip flags return `evidence_skip_flag_invalid` and are not treated as truthy
- arbitrary nested values are not exposed
- arbitrary path fields and `helper_path` are not exposed
- phases are capped to `MAX_PHASES`
- phase fields are limited to `status`, `exit_code`, `timestamp`
- stale evidence sets `stale=True`
- naive/future/malformed timestamps produce stable reasons

Example focused tests:

```python
def test_collect_operator_evidence_rejects_bool_exit_code(tmp_path: Path) -> None:
    _write_evidence(tmp_path, _valid_payload(final_exit_code=True))

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(tmp_path)},
        now=NOW,
    )

    assert summary["valid"] is False
    assert "evidence_final_exit_code_invalid" in summary["reasons"]


def test_collect_operator_evidence_bounds_and_allowlists_metadata(tmp_path: Path) -> None:
    phases = {
        f"phase-{index}": {
            "status": "ok",
            "exit_code": 0,
            "timestamp": "2026-06-19T11:59:10Z",
            "raw_output": "must-not-leak",
        }
        for index in range(25)
    }
    _write_evidence(
        tmp_path,
        _valid_payload(
            helper_path="/secret/helper",
            unexpected_path="/secret/other",
            phases=phases,
            nested={"raw": "must-not-leak"},
        ),
    )

    summary = collect_operator_evidence(
        environ={ENV_VZ_EVIDENCE_DIR: str(tmp_path)},
        now=NOW,
    )

    assert "helper_path" not in summary["runtime_pointers"]
    assert "unexpected_path" not in summary["runtime_pointers"]
    assert len(summary["phases"]) == 16
    assert "raw_output" not in next(iter(summary["phases"].values()))
    assert "nested" not in summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_evidence.py -q
```

Expected: normalization tests fail until coercion/bounds are complete.

- [ ] **Step 3: Implement normalization helpers**

Implement:

- `_coerce_schema_version(payload) -> int | None`
- `_parse_created_at(value, now) -> tuple[str | None, int | None, bool, list[str]]`
- `_normalize_exit_code(value) -> int | None`
- `_normalize_skip_flags(payload) -> tuple[dict[str, bool | None], list[str]]`
- `_normalize_phases(value) -> dict[str, dict[str, object]]`
- `_normalize_cleanup(value) -> dict[str, object]`
- `_normalize_runtime_pointers(payload) -> dict[str, str]`
- `_normalize_expected_files(file_statuses) -> dict[str, dict[str, object]]`

Use scalar coercion only. Reject `bool` as an integer. Truncate dynamic strings to `DISPLAY_MAX_CHARS`. Clamp `age_seconds` to non-negative only after adding a future-date reason; do not return negative ages.

- [ ] **Step 4: Run parser tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_evidence.py -q
```

Expected: parser tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add tldw_Server_API/app/core/Sandbox/operator_evidence.py \
  tldw_Server_API/tests/sandbox/test_operator_evidence.py
git commit -m "feat: normalize sandbox evidence metadata"
```

## Task 4: Operator-Status Projection

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/operator_status.py`
- Modify: `tldw_Server_API/tests/sandbox/test_operator_status.py`

- [ ] **Step 1: Add failing projection tests**

Add helper summaries in `test_operator_status.py`:

```python
def _evidence_ready(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "configured": True,
        "source": "host_smoke_evidence",
        "available": True,
        "valid": True,
        "evidence_dir": "/tmp/evidence",
        "schema_version": 1,
        "created_at": "2026-06-19T11:59:00+00:00",
        "age_seconds": 60,
        "stale": False,
        "smoke_run_id": "smoke-123",
        "final_exit_code": 0,
        "phases": {},
        "cleanup": {},
        "runtime_pointers": {},
        "expected_files": {},
        "skip_flags": {
            "skip_build": False,
            "skip_sign": False,
            "include_failure_drills": False,
        },
        "reasons": [],
    }
    payload.update(overrides)
    return payload
```

Add tests:

- unconfigured evidence does not degrade otherwise ready status
- invalid configured evidence degrades overall status and adds `inspect_host_gated_evidence`
- ready evidence remains ready
- stale success evidence degrades and adds `review_expected_skips`
- build/sign skips degrade and add `review_expected_skips`
- `include_failure_drills=False` stays informational
- non-zero final exit makes evidence `action_required`, adds `run_host_gated_smoke`, and makes overall `action_required` unless runtime failure is more severe
- malformed evidence booleans/lists are safely treated as unknown/degraded, not as truthy
- response validates with `SandboxAdminOperatorStatusResponse`

Example:

```python
def test_operator_status_projects_successful_evidence_as_ready() -> None:
    payload = build_operator_status(
        runtime_diagnostics=_runtime_diagnostics(),
        macos_diagnostics=_macos_diagnostics_unconfigured(),
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
        evidence_summary=_evidence_ready(),
    )

    model = SandboxAdminOperatorStatusResponse.model_validate(payload)

    assert payload["overall_status"] == "ready"
    assert payload["sections"]["evidence"]["status"] == "ready"
    assert payload["sections"]["evidence"]["final_exit_code"] == 0
    assert model.sections["evidence"].status == "ready"
```

- [ ] **Step 2: Run projection tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py -q
```

Expected: failures because `build_operator_status()` does not accept `evidence_summary` yet.

- [ ] **Step 3: Implement projection**

In `operator_status.py`:

- Extend `build_operator_status(..., evidence_summary: Mapping[str, Any] | None = None)`.
- Add `_project_evidence_section(evidence_summary) -> tuple[OperatorSection, list[dict[str, object]], bool, bool]` where booleans represent degraded/action-required impact.
- Preserve existing overall precedence:
  - runtime diagnostics failure keeps overall `unknown`
  - zero ready runtimes keeps overall `unavailable`
  - startup/helper/image-store/repair action-required remains dominant
  - evidence action-required can set overall `action_required`
  - evidence degraded can set overall `degraded`
- Add actions:
  - `inspect_host_gated_evidence`
  - `run_host_gated_smoke`
  - `review_expected_skips`

Recommended projection rules:

```python
if not configured:
    section = _section("not_configured", source="host_smoke_evidence", configured=False)
elif available is False:
    section = _section("unavailable", severity="warning", ...)
elif valid is False:
    section = _section("unknown", severity="warning", ...)
elif final_exit_code != 0:
    section = _section("action_required", severity="error", ...)
elif stale or skip_build or skip_sign:
    section = _section("degraded", severity="warning", ...)
else:
    section = _section("ready", severity="info", ...)
```

- [ ] **Step 4: Run projection tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py -q
```

Expected: operator-status tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add tldw_Server_API/app/core/Sandbox/operator_status.py \
  tldw_Server_API/tests/sandbox/test_operator_status.py
git commit -m "feat: project sandbox evidence into operator status"
```

## Task 5: Service Integration And Failure Isolation

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/service.py`
- Modify: `tldw_Server_API/tests/sandbox/test_operator_status.py`
- Optionally modify: `tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py`

- [ ] **Step 1: Add failing service tests**

In `test_operator_status.py`, add:

```python
def test_service_operator_status_collects_evidence(monkeypatch) -> None:
    svc = SandboxService()
    monkeypatch.setattr(svc, "runtime_diagnostics_summary", lambda: _runtime_diagnostics())
    monkeypatch.setattr(svc, "macos_diagnostics", lambda: _macos_diagnostics_unconfigured())

    evidence = _evidence_ready()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.service.collect_operator_evidence",
        lambda: evidence,
    )

    payload = svc.operator_status(
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["sections"]["evidence"]["status"] == "ready"


def test_service_operator_status_isolates_evidence_operational_failure(monkeypatch) -> None:
    svc = SandboxService()
    monkeypatch.setattr(svc, "runtime_diagnostics_summary", lambda: _runtime_diagnostics())
    monkeypatch.setattr(svc, "macos_diagnostics", lambda: _macos_diagnostics_unconfigured())

    def fail_evidence() -> dict[str, object]:
        raise OSError("boom")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.service.collect_operator_evidence",
        fail_evidence,
    )

    payload = svc.operator_status(
        startup_warning_summary={"present": False, "blocking": False, "codes": []},
    )

    assert payload["sections"]["evidence"]["status"] == "unknown"
    assert payload["sections"]["evidence"]["reasons"] == ["evidence_collection_failed"]
    assert payload["sections"]["runtime_readiness"]["status"] == "ready"


def test_service_operator_status_propagates_evidence_programming_error(monkeypatch) -> None:
    svc = SandboxService()
    monkeypatch.setattr(svc, "runtime_diagnostics_summary", lambda: _runtime_diagnostics())
    monkeypatch.setattr(svc, "macos_diagnostics", lambda: _macos_diagnostics_unconfigured())

    def fail_evidence() -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.service.collect_operator_evidence",
        fail_evidence,
    )

    with pytest.raises(RuntimeError, match="boom"):
        svc.operator_status(
            startup_warning_summary={"present": False, "blocking": False, "codes": []},
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py -q
```

Expected: service evidence tests fail until integration is added.

- [ ] **Step 3: Integrate parser in service**

In `service.py`:

- Add `from .operator_evidence import collect_operator_evidence`.
- Add an evidence collection block after macOS diagnostics:

```python
try:
    evidence_summary = collect_operator_evidence()
except _SANDBOX_OPERATOR_STATUS_OPERATIONAL_EXCEPTIONS as exc:
    logger.opt(exception=exc).warning(
        "Sandbox operator status evidence unavailable"
    )
    evidence_summary = {
        "configured": True,
        "source": "host_smoke_evidence",
        "available": False,
        "valid": False,
        "reasons": ["evidence_collection_failed"],
    }
```

- Pass `evidence_summary=evidence_summary` into `build_operator_status(...)`.

Do not catch `RuntimeError`, `AssertionError`, `TypeError`, or other programming defects beyond the existing operational tuple.

- [ ] **Step 4: Optionally add endpoint-level assertion**

Only add endpoint coverage if schema validation fails or the existing endpoint test does not exercise the changed service shape. If needed, adjust `fake_operator_status()` in `test_admin_macos_diagnostics.py` to return an `evidence` section with extra fields and assert response serialization keeps them.

- [ ] **Step 5: Run focused service/API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_operator_status.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py::test_admin_operator_status_returns_structured_payload \
  tldw_Server_API/tests/sandbox/test_admin_rbac.py::test_admin_endpoints_require_admin_role \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add tldw_Server_API/app/core/Sandbox/service.py \
  tldw_Server_API/tests/sandbox/test_operator_status.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py
git commit -m "feat: collect sandbox evidence for operator status"
```

## Task 6: Final Verification, Task Update, And PR Prep

**Files:**
- Modify: `backlog/tasks/task-2392 - Implement-sandbox-operator-status-evidence-ingestion-Slice-2.md`

- [ ] **Step 1: Run full focused test set**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/sandbox/test_operator_evidence.py \
  tldw_Server_API/tests/sandbox/test_operator_status.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py::test_admin_operator_status_returns_structured_payload \
  tldw_Server_API/tests/sandbox/test_admin_rbac.py::test_admin_endpoints_require_admin_role \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output, exit code 0.

- [ ] **Step 3: Run Bandit on touched server files**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Sandbox/operator_evidence.py \
  tldw_Server_API/app/core/Sandbox/operator_status.py \
  tldw_Server_API/app/core/Sandbox/service.py \
  -f json -o /tmp/bandit_sandbox_operator_evidence.json
```

Expected: no new actionable findings in touched code. If Bandit is unavailable, document the environment skip in `TASK-2392` and final response.

- [ ] **Step 4: Update Backlog task**

Update `TASK-2392` with:

- implementation notes listing parser/projection/service/test changes
- final verification commands and outcomes
- checked acceptance criteria and DOD only after verification

- [ ] **Step 5: Final commit**

```bash
git add backlog/tasks/task-2392\ -\ Implement-sandbox-operator-status-evidence-ingestion-Slice-2.md
git commit -m "docs: update sandbox evidence ingestion task"
```

- [ ] **Step 6: Check branch status**

Run:

```bash
git status --short --branch
git log --oneline --decorate -5
```

Expected: clean worktree on `codex/sandbox-evidence-status`, with implementation commits after the spec commits.

## Review Checklist Before Implementation

- Parser and projection stay separate.
- Evidence collection is read-only and env-configured only.
- Safe traversal fails closed when descriptor-safe operations are unavailable.
- `helper_path`, arbitrary JSON keys, raw logs, and guest/helper output never enter the API payload.
- Evidence action-required/degraded impacts overall status without overriding more severe runtime failures.
- Service catches only operational evidence failures; programming errors still propagate.
- Tests are portable and do not require macOS VZ or a helper.
