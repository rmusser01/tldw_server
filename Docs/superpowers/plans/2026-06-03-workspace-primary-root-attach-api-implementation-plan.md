# Workspace Primary Root Attach API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first write API for attaching or replacing one Workspace-owned primary project root, while preserving the canonical Workspace Core model and redacted read contracts.

**Architecture:** Keep `/api/v1/workspaces` as the canonical API family. Add a focused Workspace Core root-binding service that validates host-local and sandbox-volume requests, then delegates persistence to `CharactersRAGDB` with DB-transactional optimistic locking. Endpoint code stays thin and only maps service/DB outcomes into public HTTP responses and existing `WorkspaceRootsResponse` models.

**Tech Stack:** FastAPI, Pydantic v2, SQLite/PostgreSQL-compatible `CharactersRAGDB`, existing Workspace Core context/capability projection helpers, pytest, Bandit.

---

## Scope

This plan implements `TASK-2235` from the approved spec:

- Spec: `Docs/superpowers/specs/2026-06-03-workspace-primary-root-attach-api-design.md`
- Backlog: `backlog/tasks/task-2235 - Implement-Workspace-primary-root-attach-write-API-slice.md`

In scope:

- `PUT /api/v1/workspaces/{workspace_id}/roots/primary`
- One primary root only.
- `host_local` and `sandbox_volume` backends.
- Workspace-specific project-root allowlist config.
- Redacted response behavior through existing `/roots`, `/context`, and `/capabilities`.
- Fail-closed sandbox-volume default resolver behavior.

Out of scope:

- Secondary roots.
- Detach/archive/delete roots.
- File inventory workers.
- Git operations.
- MCP trusted-root mutation.
- ACP/harness launch mutation.
- Full Sandbox volume lifecycle.
- UI changes.

## File Structure

- Modify: `tldw_Server_API/app/core/config.py`
  - Add Workspace project-root allowlist config helpers, separate from ingestion roots.
- Modify: `tldw_Server_API/Config_Files/config.txt`
  - Document `[WORKSPACES].project_root_allowed_base_paths`.
- Modify: `tldw_Server_API/Config_Files/README.md`
  - Add operator-facing note for project-root allowlist env/config.
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Extend `upsert_workspace_primary_root` for `expected_workspace_version`, replacement across backend with the same root id, and same-root operational repair when requested by validated data.
- Create: `tldw_Server_API/app/core/Workspaces/root_binding_service.py`
  - Own request normalization, host-local validation, sandbox-volume resolver protocol, replacement/idempotency policy, and DB orchestration.
- Modify: `tldw_Server_API/app/core/Workspaces/context.py`
  - Fail closed when a sandbox-volume primary root is attached but mount state is not ready.
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
  - Add `WorkspacePrimaryRootAttachRequest`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
  - Add thin `PUT /{workspace_id}/roots/primary` endpoint and reuse existing redacted root response helpers.
- Modify: `tldw_Server_API/tests/Config/test_startup_validation.py`
  - Add config helper coverage.
- Modify: `tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py`
  - Add DB optimistic-lock and operational-repair coverage.
- Create: `tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py`
  - Add pure service tests.
- Modify: `tldw_Server_API/tests/Workspaces/test_workspace_core_context.py`
  - Add fail-closed sandbox mount coverage.
- Modify: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
  - Add endpoint integration coverage.

## Task 1: Workspace Project-Root Config Helper

**Files:**
- Modify: `tldw_Server_API/app/core/config.py:365-421`
- Modify: `tldw_Server_API/Config_Files/config.txt:118-126`
- Modify: `tldw_Server_API/Config_Files/README.md`
- Test: `tldw_Server_API/tests/Config/test_startup_validation.py`

- [ ] **Step 1: Write failing config tests**

Add tests next to `test_get_ingestion_source_allowed_roots_resolves_relative_paths_from_project_root`.

```python
def test_get_workspace_project_root_allowed_roots_prefers_workspace_specific_config(monkeypatch):
    workspace_root = config.ACTUAL_PROJECT_ROOT / "workspace-projects"
    acp_root = config.ACTUAL_PROJECT_ROOT / "legacy-acp"

    def _fake_get_config_value(section: str, key: str, default=None, *, reload: bool = False):
        values = {
            ("WORKSPACES", "project_root_allowed_base_paths"): "workspace-projects",
            ("ACP-WORKSPACE", "allowed_base_paths"): "legacy-acp",
        }
        return values.get((section, key), default)

    monkeypatch.setattr(config, "get_config_value", _fake_get_config_value, raising=True)
    monkeypatch.delenv("WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)
    monkeypatch.delenv("TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)
    monkeypatch.delenv("ACP_WORKSPACE_ALLOWED_BASE_PATHS", raising=False)

    assert config.get_workspace_project_root_allowed_roots() == (workspace_root,)
    assert acp_root not in config.get_workspace_project_root_allowed_roots()
```

Also add:

```python
def test_get_workspace_project_root_allowed_roots_uses_acp_fallback_only_when_workspace_empty(monkeypatch):
    def _fake_get_config_value(section: str, key: str, default=None, *, reload: bool = False):
        values = {
            ("WORKSPACES", "project_root_allowed_base_paths"): "",
            ("ACP-WORKSPACE", "allowed_base_paths"): "legacy-acp",
        }
        return values.get((section, key), default)

    monkeypatch.setattr(config, "get_config_value", _fake_get_config_value, raising=True)
    monkeypatch.delenv("WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)
    monkeypatch.delenv("TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)
    monkeypatch.delenv("ACP_WORKSPACE_ALLOWED_BASE_PATHS", raising=False)

    assert config.get_workspace_project_root_allowed_roots() == (
        config.ACTUAL_PROJECT_ROOT / "legacy-acp",
    )
```

And:

```python
def test_get_workspace_project_root_allowed_roots_dedupes_config_and_env(monkeypatch):
    def _fake_get_config_value(section: str, key: str, default=None, *, reload: bool = False):
        if (section, key) == ("WORKSPACES", "project_root_allowed_base_paths"):
            return "projects,projects"
        return default

    monkeypatch.setattr(config, "get_config_value", _fake_get_config_value, raising=True)
    monkeypatch.setenv("WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", "projects")
    monkeypatch.delenv("TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS", raising=False)

    assert config.get_workspace_project_root_allowed_roots() == (
        config.ACTUAL_PROJECT_ROOT / "projects",
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Config/test_startup_validation.py -q
```

Expected: FAIL because `get_workspace_project_root_allowed_roots` is not defined.

- [ ] **Step 3: Add config helper**

Implement:

```python
_WORKSPACE_PROJECT_ROOT_ALLOWED_ROOT_SECTION = "WORKSPACES"
_WORKSPACE_PROJECT_ROOT_ALLOWED_ROOT_KEY = "project_root_allowed_base_paths"
_WORKSPACE_PROJECT_ROOT_ALLOWED_ROOT_ENV_KEYS = (
    "WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS",
    "TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS",
)
_WORKSPACE_PROJECT_ROOT_ACP_FALLBACK_SECTION = "ACP-WORKSPACE"
_WORKSPACE_PROJECT_ROOT_ACP_FALLBACK_KEY = "allowed_base_paths"
_WORKSPACE_PROJECT_ROOT_ACP_FALLBACK_ENV_KEYS = ("ACP_WORKSPACE_ALLOWED_BASE_PATHS",)


def _dedupe_normalized_allowed_roots(raw_values: list[str]) -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        normalized = _normalize_allowed_root_entry(raw_value)
        if normalized is None:
            continue
        marker = str(normalized)
        if marker in seen:
            continue
        seen.add(marker)
        roots.append(normalized)
    return tuple(roots)


def get_workspace_project_root_allowed_roots(*, reload: bool = False) -> tuple[Path, ...]:
    workspace_values = _split_allowed_root_values(
        get_config_value(
            _WORKSPACE_PROJECT_ROOT_ALLOWED_ROOT_SECTION,
            _WORKSPACE_PROJECT_ROOT_ALLOWED_ROOT_KEY,
            default="",
            reload=reload,
        )
    )
    for env_name in _WORKSPACE_PROJECT_ROOT_ALLOWED_ROOT_ENV_KEYS:
        workspace_values.extend(_split_allowed_root_values(os.getenv(env_name)))
    workspace_roots = _dedupe_normalized_allowed_roots(workspace_values)
    if workspace_roots:
        return workspace_roots

    fallback_values = _split_allowed_root_values(
        get_config_value(
            _WORKSPACE_PROJECT_ROOT_ACP_FALLBACK_SECTION,
            _WORKSPACE_PROJECT_ROOT_ACP_FALLBACK_KEY,
            default="",
            reload=reload,
        )
    )
    for env_name in _WORKSPACE_PROJECT_ROOT_ACP_FALLBACK_ENV_KEYS:
        fallback_values.extend(_split_allowed_root_values(os.getenv(env_name)))
    return _dedupe_normalized_allowed_roots(fallback_values)
```

Refactor `get_ingestion_source_allowed_roots` to reuse `_dedupe_normalized_allowed_roots` without changing its behavior.

- [ ] **Step 4: Add config docs**

In `tldw_Server_API/Config_Files/config.txt`, add a new section before `[ACP-WORKSPACE]`:

```ini
[WORKSPACES]
# Optional. Project Workspace host-local roots must be under one of these paths.
# This is intentionally separate from ingestion_source_allowed_roots.
project_root_allowed_base_paths =
```

In `tldw_Server_API/Config_Files/README.md`, add a short note:

```markdown
### Workspace Project Roots

Project Workspace host-local roots are gated by `[WORKSPACES].project_root_allowed_base_paths`,
`WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS`, or `TLDW_WORKSPACE_PROJECT_ROOT_ALLOWED_BASE_PATHS`.
If unset, the Workspace API falls back to ACP workspace roots for compatibility. Ingestion source
roots are not used for project-root writes.
```

- [ ] **Step 5: Run config tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Config/test_startup_validation.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/config.py tldw_Server_API/Config_Files/config.txt tldw_Server_API/Config_Files/README.md tldw_Server_API/tests/Config/test_startup_validation.py
git commit -m "feat: add workspace project root allowlist config"
```

## Task 2: DB Primary Root Write Semantics

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:15531-15688`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py`

- [ ] **Step 1: Write failing DB tests**

Add:

```python
def test_upsert_primary_root_enforces_expected_workspace_version(db):
    db.upsert_workspace("ws-1", "Workspace")

    with pytest.raises(ConflictError):
        db.upsert_workspace_primary_root(
            "ws-1",
            {
                "root_id": "primary",
                "backend": "host_local",
                "absolute_root": "/Users/example/project",
                "expected_workspace_version": 999,
            },
        )
```

Add:

```python
def test_upsert_primary_root_replaces_same_root_id_when_replace_existing_true(db):
    db.upsert_workspace("ws-1", "Workspace")
    db.upsert_workspace_primary_root(
        "ws-1",
        {"root_id": "primary", "backend": "host_local", "absolute_root": "/old"},
    )

    root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "primary",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
            "replace_existing": True,
        },
    )

    assert root["backend"] == "sandbox_volume"
    assert root["sandbox_volume_id"] == "volume-1"
    assert root["absolute_root"] is None
```

Update the existing `test_retrying_same_primary_root_upsert_preserves_operational_state` so the retry omits operational fields:

```python
retried = db.upsert_workspace_primary_root(
    "ws-1",
    {"root_id": "root-1", "backend": "host_local"},
)
```

Add:

```python
def test_retrying_same_primary_root_upsert_repairs_operational_state_when_provided(db):
    db.upsert_workspace("ws-1", "Workspace")
    root = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "primary",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
            "root_state": "attached",
            "sandbox_mount_state": "unavailable",
        },
    )

    repaired = db.upsert_workspace_primary_root(
        "ws-1",
        {
            "root_id": "primary",
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-1",
            "root_state": "attached",
            "sandbox_mount_state": "ready",
        },
    )

    assert repaired["sandbox_mount_state"] == "ready"
    assert repaired["version"] == root["version"] + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py -q
```

Expected: FAIL on version enforcement, same-root backend replacement, or operational repair.

- [ ] **Step 3: Extend DB method**

Update `upsert_workspace_primary_root`:

- Parse `expected_workspace_version = data.get("expected_workspace_version")`.
- Validate it is int-like when present; raise `InputError` if malformed.
- Parse `replace_existing = bool(data.get("replace_existing", False))`.
- Inside the same transaction that loads the workspace row, compare `expected_workspace_version` with `workspace["version"]`; raise `ConflictError` on mismatch.
- Treat same `root_id` with different backend as replacement only when `replace_existing` is true; otherwise keep the existing backend mismatch `ConflictError`.
- Include operational columns in the same-root update set only when the key is present in `data`:
  - `root_state`
  - `git_state`
  - `file_inventory_state`
  - `indexing_state`
  - `sandbox_mount_state`
  - `mcp_trust_state`
- Keep omitted operational fields unchanged on same-root retries.
- Keep existing `sqlite3.IntegrityError` and `BackendDatabaseError` mapping.

The same-root branch should conceptually look like:

```python
binding_columns = (
    "absolute_root",
    "sandbox_volume_id",
    "display_name",
    "metadata_json",
)
operational_columns = (
    "root_state",
    "git_state",
    "file_inventory_state",
    "indexing_state",
    "sandbox_mount_state",
    "mcp_trust_state",
)
for column in (*binding_columns, *operational_columns):
    if column not in data:
        continue
    new_value = metadata_json if column == "metadata_json" else values[column]
    if existing_primary.get(column) != new_value:
        root_updates[column] = new_value
```

- [ ] **Step 4: Run DB tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py
git commit -m "feat: harden workspace primary root db upsert"
```

## Task 3: Root Binding Service

**Files:**
- Create: `tldw_Server_API/app/core/Workspaces/root_binding_service.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py`

- [ ] **Step 1: Write failing service tests**

Create `test_workspace_root_binding_service.py` with a `CharactersRAGDB` fixture and tests for:

```python
def test_host_local_attach_validates_allowlist_and_redacts_by_persisting_absolute_path(db, tmp_path):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-1", "Workspace")

    root = attach_primary_workspace_root(
        db=db,
        workspace_id="ws-1",
        user_id="1",
        request=WorkspaceRootAttachRequest(
            backend="host_local",
            absolute_root=str(project),
        ),
        allowed_roots=(allowed,),
    )

    assert root["root_id"] == "primary"
    assert root["backend"] == "host_local"
    assert root["absolute_root"] == str(project.resolve())
    assert root["root_state"] == "attached"
```

Add targeted tests:

- omitted `root_id` is idempotent on retry and resolves to the current primary root id for same-binding retry.
- invalid `root_id` raises `WorkspaceRootInputError`.
- `display_name` longer than 120 characters raises `WorkspaceRootInputError`.
- no host-local allowed roots raises `WorkspaceRootConfigurationError` with `code == "workspace_project_roots_not_configured"`.
- host-local outside allowlist raises `WorkspaceRootValidationError` with `code == "workspace_project_root_outside_allowed_roots"`.
- host-local symlink root raises `WorkspaceRootInputError`.
- different primary root without `replace_existing` raises `WorkspaceRootConflictError`.
- different primary root with `replace_existing` replaces.
- default sandbox resolver persists fail-closed `sandbox_mount_state == "not_configured"`.
- fake sandbox resolver with `state="ready"` repairs a prior `sandbox_mount_state == "unavailable"` retry.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py -q
```

Expected: FAIL because service module does not exist.

- [ ] **Step 3: Implement service types and exceptions**

Create `root_binding_service.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Sequence

from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path

_ROOT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SANDBOX_VOLUME_ID_RE = _ROOT_ID_RE
_DEFAULT_ROOT_ID = "primary"


class WorkspaceRootServiceError(Exception):
    status_code = 400
    code = "workspace_root_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class WorkspaceRootInputError(WorkspaceRootServiceError):
    status_code = 400
    code = "workspace_root_invalid_request"


class WorkspaceRootValidationError(WorkspaceRootServiceError):
    status_code = 403
    code = "workspace_project_root_not_allowed"


class WorkspaceRootConflictError(WorkspaceRootServiceError):
    status_code = 409
    code = "workspace_primary_root_exists"


class WorkspaceRootConfigurationError(WorkspaceRootServiceError):
    status_code = 503
    code = "workspace_project_roots_not_configured"


@dataclass(frozen=True)
class WorkspaceRootAttachRequest:
    backend: Literal["host_local", "sandbox_volume"]
    root_id: str | None = None
    absolute_root: str | None = None
    sandbox_volume_id: str | None = None
    display_name: str | None = None
    replace_existing: bool = False
    expected_workspace_version: int | None = None
    strict_sandbox_validation: bool = False


@dataclass(frozen=True)
class SandboxVolumeBinding:
    sandbox_volume_id: str
    state: Literal["ready", "not_configured", "unavailable", "failed"]
    display_name: str | None = None
    reason_code: str | None = None


class SandboxVolumeResolver(Protocol):
    def validate_workspace_volume(
        self,
        *,
        workspace_id: str,
        user_id: str,
        sandbox_volume_id: str,
    ) -> SandboxVolumeBinding:
        ...
```

Add `DefaultSandboxVolumeResolver` that returns `not_configured` and does not claim ownership.

- [ ] **Step 4: Implement host-local validation**

Rules:

- `absolute_root` required.
- `sandbox_volume_id` rejected.
- expand `~`.
- require absolute path after expansion.
- reject symlink root with `candidate.is_symlink()` before resolving.
- resolve with `strict=False`, then require `.exists()` and `.is_dir()`.
- require containment under one configured allowed root using `resolve_safe_local_path`.
- return resolved path and display name.

Use explicit codes:

- `workspace_project_root_path_required`
- `workspace_project_root_not_absolute`
- `workspace_project_root_missing`
- `workspace_project_root_not_directory`
- `workspace_project_root_symlink`
- `workspace_project_root_outside_allowed_roots`

- [ ] **Step 5: Implement sandbox-volume validation**

Rules:

- `sandbox_volume_id` required.
- `absolute_root` rejected.
- id regex `^[A-Za-z0-9_.:-]{1,128}$`.
- default resolver returns `SandboxVolumeBinding(state="not_configured")`.
- if `strict_sandbox_validation` and state is not `ready`, raise `WorkspaceRootConfigurationError(code="workspace_sandbox_volume_resolver_unavailable")`.
- otherwise persist fail-closed with `root_state="attached"` and `sandbox_mount_state=binding.state`.

- [ ] **Step 6: Implement replacement/idempotency policy**

`attach_primary_workspace_root` should:

1. Load current primary with `db.get_workspace_primary_root(workspace_id)`.
2. Normalize request and target.
3. Resolve `root_id`:
   - supplied root id if valid.
   - current primary root id when current primary matches backend+target.
   - otherwise `"primary"`.
4. If current primary exists and does not match backend+target:
   - raise `WorkspaceRootConflictError` when `replace_existing` is false.
   - pass `replace_existing=True` to DB when true.
5. Call `db.upsert_workspace_primary_root` with:
   - `root_id`
   - `backend`
   - backend-specific target
   - normalized `display_name`
   - `root_state="attached"`
   - sandbox `sandbox_mount_state`
   - `expected_workspace_version`
   - `replace_existing`
6. Return the persisted DB root row.

Wrap DB exceptions at this service boundary so the endpoint keeps stable
Workspace-root error payloads:

```python
try:
    return db.upsert_workspace_primary_root(workspace_id, payload)
except ConflictError as exc:
    message = str(exc)
    code = "workspace_version_mismatch" if "version" in message.lower() else "workspace_primary_root_write_conflict"
    raise WorkspaceRootConflictError(message, code=code) from exc
except InputError as exc:
    raise WorkspaceRootInputError(str(exc), code="workspace_root_invalid_request") from exc
except CharactersRAGDBError:
    raise
```

- [ ] **Step 7: Run service tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Workspaces/root_binding_service.py tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py
git commit -m "feat: add workspace root binding service"
```

## Task 4: Fail-Closed Project Root Capabilities

**Files:**
- Modify: `tldw_Server_API/app/core/Workspaces/context.py:309-316`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_core_context.py`

- [ ] **Step 1: Write failing context test**

Add:

```python
def test_sandbox_volume_root_fails_closed_until_mount_ready() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "primary",
            "backend": "sandbox_volume",
            "root_state": "attached",
            "sandbox_volume_id": "volume-1",
            "sandbox_mount_state": "not_configured",
        },
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "sandbox": {"state": "available", "reason_code": None},
                "acp": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_sandbox": {"allowed": True, "reason_code": None},
                "use_acp_agents": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[],
    )

    assert context["allowed_actions"]["write_files"] == {
        "allowed": False,
        "reason_code": "sandbox_mount_not_configured",
    }
    assert context["allowed_actions"]["run_sandbox"] == {
        "allowed": False,
        "reason_code": "sandbox_mount_not_configured",
    }
    assert context["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "sandbox_mount_not_configured",
    }
```

- [ ] **Step 2: Run context tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_context.py -q
```

Expected: FAIL because current root readiness only checks `root_state`, `root_id`, and backend.

- [ ] **Step 3: Add sandbox mount readiness gate**

Update `_project_root_ready`:

```python
def _project_root_ready(profile: str, project_root: Mapping[str, Any]) -> tuple[bool, str]:
    if profile != "project":
        return False, "project_root_not_configured"
    if project_root.get("state") != "attached":
        return False, f"project_root_{project_root.get('state') or 'not_configured'}"
    if not project_root.get("root_id") or not project_root.get("backend"):
        return False, "project_root_unresolved"
    if project_root.get("backend") == "sandbox_volume":
        mount_state = str(project_root.get("sandbox_mount_state") or "not_configured")
        if mount_state not in {"ready", "mounted"}:
            return False, f"sandbox_mount_{mount_state}"
    return True, ""
```

Keep `"mounted"` as a compatibility alias because existing tests and prior slices use it.

- [ ] **Step 4: Run context tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_core_context.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Workspaces/context.py tldw_Server_API/tests/Workspaces/test_workspace_core_context.py
git commit -m "feat: fail closed for unready sandbox workspace roots"
```

## Task 5: API Schema And Endpoint

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py:18-30`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py:23-75`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py:978-1005`
- Test: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`

- [ ] **Step 1: Write failing API tests**

Add tests:

```python
@pytest.mark.integration
def test_attach_workspace_primary_host_local_root_returns_redacted_roots_response(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Rooted Workspace")

    from tldw_Server_API.app.core.Workspaces import root_binding_service

    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda **kwargs: (allowed,),
        raising=True,
    )
    response = _put_workspace_primary_root(
        workspace_fastapi_app,
        db,
        "ws-root",
        {"backend": "host_local", "absolute_root": str(project)},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_profile"] == "project"
    assert payload["primary_root"]["root_id"] == "primary"
    assert payload["primary_root"]["backend"] == "host_local"
    assert payload["primary_root"]["path_hint"] == "project"
    assert "absolute_root" not in payload["primary_root"]
```

Add helper similar to `_get_workspace_roots_response`, but using `WORKSPACES_WRITE_RATE_LIMIT`:

```python
def _put_workspace_primary_root(app: FastAPI, db_like: Any, workspace_id: str, payload: dict[str, Any]):
    async def _allow_rate_limit() -> None:
        return None

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            return client.put(f"/api/v1/workspaces/{workspace_id}/roots/primary", json=payload)
    finally:
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
        app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)
```

Add endpoint tests for:

- outside allowed root returns `403` and `detail.code == "workspace_project_root_outside_allowed_roots"`.
- no configured roots returns `503` and `detail.code == "workspace_project_roots_not_configured"`.
- different root without `replace_existing` returns `409` and `detail.code == "workspace_primary_root_exists"`.
- replacement with `replace_existing: true` returns new primary.
- sandbox attach returns `path_hint == sandbox_volume_id` and `sandbox_mount_state == "not_configured"`.
- stale `expected_workspace_version` returns `409` with
  `detail.code == "workspace_version_mismatch"`.

- [ ] **Step 2: Run API tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q
```

Expected: FAIL because schema/endpoint do not exist.

- [ ] **Step 3: Add Pydantic request schema**

In `workspace_schemas.py`, add:

```python
class WorkspacePrimaryRootAttachRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: WorkspaceProjectRootBackend
    root_id: str | None = None
    absolute_root: str | None = None
    sandbox_volume_id: str | None = None
    display_name: str | None = Field(default=None, max_length=120)
    replace_existing: StrictBool = False
    expected_workspace_version: int | None = Field(default=None, ge=1)
    strict_sandbox_validation: StrictBool = False
```

Keep field-specific backend validation in `root_binding_service`; this avoids duplicating domain errors in Pydantic and preserves stable service error codes.

- [ ] **Step 4: Add endpoint helpers**

In `workspaces.py`, import:

```python
WorkspacePrimaryRootAttachRequest,
attach_primary_workspace_root,
WorkspaceRootAttachRequest,
WorkspaceRootServiceError,
```

Add helper:

```python
def _workspace_roots_response(
    *,
    workspace_id: str,
    workspace: dict[str, Any],
    roots: list[dict[str, Any]],
) -> WorkspaceRootsResponse:
    primary_root = next((root for root in roots if bool(root.get("is_primary"))), None)
    return WorkspaceRootsResponse(
        workspace_id=workspace_id,
        workspace_profile=str(workspace.get("workspace_profile") or "research"),
        primary_root=_root_to_response(primary_root) if primary_root else None,
        roots=[_root_to_response(root) for root in roots],
    )
```

Refactor `list_workspace_roots` to use this helper.

- [ ] **Step 5: Add endpoint**

Add before the existing `GET /{workspace_id}/roots` route:

```python
@router.put(
    "/{workspace_id}/roots/primary",
    response_model=WorkspaceRootsResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Attach or replace the primary workspace project root",
)
async def attach_workspace_primary_root_endpoint(
    workspace_id: str,
    body: WorkspacePrimaryRootAttachRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceRootsResponse:
    try:
        _require_workspace(db, workspace_id)
        attach_primary_workspace_root(
            db=db,
            workspace_id=workspace_id,
            user_id=str(getattr(current_user, "id", "")),
            request=WorkspaceRootAttachRequest(**body.model_dump()),
        )
        workspace = _require_workspace(db, workspace_id)
        roots = db.list_workspace_project_roots(workspace_id)
        return _workspace_roots_response(workspace_id=workspace_id, workspace=workspace, roots=roots)
    except WorkspaceRootServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except HTTPException:
        raise
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to attach workspace primary root") from exc
```

- [ ] **Step 6: Run API tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py
git commit -m "feat: add workspace primary root attach endpoint"
```

## Task 6: Cross-Contract Regression Tests

**Files:**
- Modify: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
- Modify as needed from earlier tasks.

- [ ] **Step 1: Add read contract regression test**

After attaching a host-local root through the new endpoint, call:

- `GET /api/v1/workspaces/{workspace_id}/roots`
- `GET /api/v1/workspaces/{workspace_id}/capabilities`
- `GET /api/v1/workspaces/{workspace_id}/context`

Assert:

- `workspace_profile == "project"`.
- `project_root.path_hint` uses basename/display name only.
- serialized JSON does not contain the temp absolute path string.
- capability actions are conservative for file indexing/preview unless those states are explicitly ready.

- [ ] **Step 2: Add sandbox fail-closed API regression test**

Attach `sandbox_volume` with default resolver. Assert:

- root response includes `sandbox_mount_state == "not_configured"`.
- `/capabilities` and `/context` return `allowed_actions.use_sandbox.allowed is False`.
- `reason_code == "sandbox_mount_not_configured"` for project-root-gated sandbox/ACP actions.

- [ ] **Step 3: Run focused Workspace suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py \
  tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py \
  tldw_Server_API/tests/Workspaces/test_workspace_core_context.py \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tldw_Server_API/tests/Workspaces/test_workspaces_api.py
git commit -m "test: cover workspace root attach read contracts"
```

## Task 7: Final Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-2235 - Implement-Workspace-primary-root-attach-write-API-slice.md`
- Move to: `backlog/completed/task-2235 - Implement-Workspace-primary-root-attach-write-API-slice.md`

- [ ] **Step 1: Run focused tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Config/test_startup_validation.py \
  tldw_Server_API/tests/Workspaces/test_workspace_project_roots_db.py \
  tldw_Server_API/tests/Workspaces/test_workspace_root_binding_service.py \
  tldw_Server_API/tests/Workspaces/test_workspace_core_context.py \
  tldw_Server_API/tests/Workspaces/test_workspace_service_capabilities.py \
  tldw_Server_API/tests/Workspaces/test_workspaces_api.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run compile smoke**

```bash
source .venv/bin/activate && python -m compileall \
  tldw_Server_API/app/core/Workspaces \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/config.py
```

Expected: `0` exit.

- [ ] **Step 3: Run Bandit on touched backend scope**

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Workspaces \
  tldw_Server_API/app/api/v1/endpoints/workspaces.py \
  tldw_Server_API/app/api/v1/schemas/workspace_schemas.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/config.py \
  -f json -o /tmp/bandit_task_2235.json
```

Expected: no new findings in touched code.

- [ ] **Step 4: Run diff hygiene**

```bash
git diff --check
```

Expected: no output and `0` exit.

- [ ] **Step 5: Update Backlog task**

Record:

- tests run and results.
- compile smoke result.
- Bandit result.
- any known skips.
- final summary.

Then complete the task using Backlog MCP if available.

- [ ] **Step 6: Final commit**

```bash
git add backlog/tasks/task-2235\ -\ Implement-Workspace-primary-root-attach-write-API-slice.md backlog/completed/task-2235\ -\ Implement-Workspace-primary-root-attach-write-API-slice.md
git commit -m "chore: close workspace primary root attach task"
```

## Plan Review Notes

Self-review findings before implementation:

- The endpoint must not compare root identity by `root_id` alone. The service owns backend+target comparison so omitted `root_id` retries remain deterministic.
- `replace_existing: true` must handle replacement even when the current and requested root id are both `"primary"` but the backend changes.
- `expected_workspace_version` must be checked inside the DB write transaction. Service precheck alone is insufficient.
- DB conflicts from the attach write must be wrapped by the service so stale-version and root-write conflicts keep stable API `{code,message}` payloads instead of falling through the generic DB mapper.
- The default Sandbox resolver must not mark a volume ready. It may only persist a fail-closed `not_configured` binding unless strict validation rejects the request.
- Existing public response helpers already redact `absolute_root`; regression tests still need to assert the absolute temp path never appears in `/roots`, `/capabilities`, or `/context`.
- Keep `[WORKSPACES]` allowlists separate from ingestion roots. ACP roots are fallback only when Workspace-specific roots are absent, not an additive broadening source.
