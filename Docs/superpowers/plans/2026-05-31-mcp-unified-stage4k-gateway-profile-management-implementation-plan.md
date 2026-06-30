# MCP Unified Stage 4K Gateway Profile Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add package-owned gateway profile management so standalone MCP gateway users can list stored profiles, duplicate built-in presets, set/read a persisted default profile, and have no-profile JSON-RPC requests use that default.

**Architecture:** Introduce one `GatewayProfileManager` in `mcp_unified.gateway.profiles` and reuse it from CLI and FastAPI routes. Keep default-profile state in `ProfileAssignmentStore` via a shared assignment-aware resolver/helper so runtime, CLI, and HTTP management surfaces do not invent separate default storage. Keep route mounting explicit: management endpoints appear only when a manager/bootstrap is supplied or an explicit enable flag is paired with a valid manager source.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, package-local MCP gateway/profile/storage protocols, SQLite store already backed by SQLAlchemy, pytest, Ruff, Bandit.

---

## Source Design

- Spec: `Docs/superpowers/specs/2026-05-30-mcp-unified-stage4k-gateway-profile-management-design.md`
- Backlog: `TASK-570`
- Prior gateway slices to preserve:
  - `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4e-gateway-profile-runtime-plan.md`
  - `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4f-gateway-profile-bootstrap-plan.md`
  - `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4g-gateway-config-bootstrap-plan.md`
  - `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4i-gateway-cli-plan.md`
  - `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4j-preset-details-cli-plan.md`

## Scope Boundaries

Build only:

- Stored profile listing and inspection.
- Built-in preset duplication into the configured store.
- Gateway default profile get/set through `ProfileAssignmentStore`.
- Assignment-aware default resolution for no-profile JSON-RPC.
- CLI and FastAPI management surfaces for those operations.

Do not build:

- Arbitrary profile editing, deletion, disable APIs, approval-policy CRUD, credential CRUD, workspace binding CRUD, external server lifecycle APIs, audit viewer APIs, or WebUI changes.

## File Map

- Create: `mcp_unified/gateway/profiles.py`
  - Owns `GatewayProfileManager`, domain error types, store metadata helpers, JSON-safe payload builders, default assignment writes, and optional audit event emission.
- Create: `mcp_unified/profiles/defaults.py`
  - Owns `GATEWAY_DEFAULT_ASSIGNMENT_ID`, deterministic default-assignment selection, and async default-profile-id loading from a `ProfileAssignmentStore`.
- Modify: `mcp_unified/profiles/store.py`
  - Add `InMemoryProfileAssignmentStore` and assignment-store unavailable error primitives for memory-backed tests/dev configs.
- Modify: `mcp_unified/profiles/resolver.py`
  - Add assignment-aware resolver support without breaking `StoreBackedProfileResolver` callers.
- Modify: `mcp_unified/gateway/profile_runtime.py`
  - Continue accepting `profile_resolver`; default bootstrap/config paths should pass the new resolver, not a process-local-only default id.
- Modify: `mcp_unified/gateway/bootstrap.py`
  - Extend `GatewayProfileBootstrap` with `assignment_store`, `audit_store`, `profile_manager`, and `store_metadata`.
  - Build a shared resolver/manager against the same assignment store.
  - Keep runtime bootstrap seeding centralized so config profiles and `default_preset_id` behavior stay compatible.
- Modify: `mcp_unified/gateway/config.py`
  - Add a storage-bundle helper that creates profile, assignment, audit stores, and metadata from config.
  - Keep SQLite store construction lazy.
- Modify: `mcp_unified/gateway/cli.py`
  - Add config-aware management commands and deterministic JSON success/error output.
- Modify: `mcp_unified/gateway/fastapi.py`
  - Add optional management route mounting and exact request/response envelopes.
- Modify: `mcp_unified/gateway/__init__.py`
  - Export new public package helpers without eagerly importing FastAPI.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`
  - New manager, memory assignment store, default selection, audit, and copy-isolation tests.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py`
  - Assignment-aware resolver behavior and no-host-import boundary coverage for new profile package file.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`
  - CLI command envelopes, config selection, memory-store mutation rejection, and domain-error stderr.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
  - Route gating, HTTP management envelopes, status mapping, and runtime default changes without restart.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
  - Run unchanged as boundary regression.

## Public Contracts To Preserve

- Existing `create_gateway_app(runtime, prefix="/mcp")` continues to expose only status/request/ws and no profile management routes.
- Existing no-profile profile runtime behavior still fails closed unless bootstrap/config supplies a default id or default assignment.
- Explicit transport profile selectors still override stored defaults.
- New package files must not import `tldw_Server_API`.
- CLI `list-presets` and `show-preset` remain catalog-only and do not require config.

## Domain Shapes

Use these exact reason codes:

```python
PROFILE_REASON_STATUS = {
    "profile_not_found": 404,
    "preset_not_found": 404,
    "default_profile_not_configured": 404,
    "profile_disabled": 409,
    "profile_already_exists": 409,
    "invalid_profile_request": 422,
    "profile_store_unavailable": 503,
    "assignment_store_unavailable": 503,
}
```

Use this stable default assignment id:

```python
GATEWAY_DEFAULT_ASSIGNMENT_ID = "gateway-default"
```

Use these store metadata envelopes:

```json
{"kind": "sqlite", "persistent": true}
{"kind": "memory", "persistent": false}
```

For read-only memory-store CLI commands, include this exact nested payload:

```json
{"store": {"kind": "memory", "persistent": false}}
```

---

### Task 1: Manager And Memory Store RED Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py`

- [ ] **Step 1: Add manager tests for list/show and copy isolation**

Add tests that seed `InMemoryProfileStore` with two profiles and assert `GatewayProfileManager.list_profiles()` and `show_profile()` return deterministic, JSON-safe payloads.

Expected test skeleton:

```python
@pytest.mark.asyncio
async def test_gateway_profile_manager_lists_profiles_with_store_metadata() -> None:
    store = InMemoryProfileStore([
        MCPProfile(id="reviewer", name="Reviewer"),
        MCPProfile(id="architect", name="Architect"),
    ])
    assignment_store = InMemoryProfileAssignmentStore()
    manager = GatewayProfileManager(
        profile_store=store,
        assignment_store=assignment_store,
        store_metadata=GatewayProfileStoreMetadata(kind="memory", persistent=False),
    )

    payload = await manager.list_profiles()

    assert payload["ok"] is True
    assert [profile["id"] for profile in payload["profiles"]] == ["architect", "reviewer"]
    assert payload["store"] == {"kind": "memory", "persistent": False}
```

- [ ] **Step 2: Add preset duplication tests**

Cover:

- success from `project-researcher` using the preset id as default stored id;
- custom `profile_id` and custom `name`;
- unknown preset returns `preset_not_found`;
- collision returns `profile_already_exists`;
- returned profile mutation does not mutate the store.

Expected domain error assertion:

```python
with pytest.raises(GatewayProfileManagementError) as exc_info:
    await manager.duplicate_preset("missing-preset")

assert exc_info.value.reason_code == "preset_not_found"
assert exc_info.value.to_payload()["preset_id"] == "missing-preset"
```

- [ ] **Step 3: Add default profile tests**

Cover:

- `set_default_profile()` stores one enabled `ProfileAssignment` with id `gateway-default`;
- repeated set overwrites the same assignment id instead of accumulating defaults;
- `get_default_profile()` reads from assignment store first;
- missing default returns `default_profile_not_configured`;
- missing target profile returns `profile_not_found`;
- disabled target profile returns `profile_disabled`;
- multiple legacy defaults choose greatest `updated_at`, then assignment id ascending.

Use fixed aware UTC timestamps for deterministic ordering.

- [ ] **Step 4: Add optional audit tests**

Add a small in-memory audit store test double in the test file. Assert configured audit receives events for:

- `profile.duplicated_from_preset`;
- `profile.default_changed`;
- failed duplication due to collision;
- failed duplication due to unknown preset;
- failed default set due to missing profile;
- failed default set due to disabled profile.

Assert event payloads contain ids/reason codes and do not contain full profile documents.

- [ ] **Step 5: Add resolver RED tests**

In `test_profile_registry_resolver.py`, add tests for a new assignment-aware resolver:

```python
@pytest.mark.asyncio
async def test_assignment_backed_resolver_uses_stored_default_assignment() -> None:
    profile_store = InMemoryProfileStore([
        MCPProfile(id="default", name="Default"),
        MCPProfile(id="explicit", name="Explicit"),
    ])
    assignment_store = InMemoryProfileAssignmentStore([
        ProfileAssignment(id="gateway-default", profile_id="default", is_default=True),
    ])
    resolver = AssignmentBackedProfileResolver(
        profile_store,
        assignment_store=assignment_store,
        fallback_default_profile_id=None,
    )

    result = await resolver.resolve_profile_result(None)

    assert result.status == "resolved"
    assert result.profile is not None
    assert result.profile.id == "default"
    assert result.provenance["used_default_assignment"] is True
```

Also assert explicit `profile_id="explicit"` bypasses the stored default.

- [ ] **Step 6: Run RED tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py -q
```

Expected: failures for missing `mcp_unified.gateway.profiles`, `InMemoryProfileAssignmentStore`, and assignment-aware resolver.

### Task 2: Shared Default Assignment And Manager Implementation

**Files:**
- Create: `mcp_unified/profiles/defaults.py`
- Modify: `mcp_unified/profiles/store.py`
- Modify: `mcp_unified/profiles/resolver.py`
- Create: `mcp_unified/gateway/profiles.py`
- Modify: `mcp_unified/gateway/__init__.py`

- [ ] **Step 1: Add in-memory assignment store**

Implement `InMemoryProfileAssignmentStore` in `mcp_unified/profiles/store.py` with the same copy-isolation behavior as `InMemoryProfileStore`.

Required behavior:

```python
class InMemoryProfileAssignmentStore:
    def __init__(self, assignments: Iterable[ProfileAssignment | Mapping[str, Any]] | None = None) -> None: ...

    async def get_assignment(self, assignment_id: str) -> ProfileAssignment | None: ...

    async def list_assignments(
        self,
        *,
        profile_id: str | None = None,
        principal_id: str | None = None,
        workspace_id: str | None = None,
    ) -> list[ProfileAssignment]: ...

    async def upsert_assignment(self, assignment: ProfileAssignment | Mapping[str, Any]) -> ProfileAssignment: ...

    async def delete_assignment(self, assignment_id: str) -> bool: ...
```

Sort `list_assignments()` by id for deterministic tests. Filter only when a filter value is not `None`.

- [ ] **Step 2: Add default assignment helpers**

Create `mcp_unified/profiles/defaults.py`:

```python
GATEWAY_DEFAULT_ASSIGNMENT_ID = "gateway-default"

def select_gateway_default_assignment(assignments: Iterable[ProfileAssignment]) -> ProfileAssignment | None:
    enabled_defaults = [
        assignment
        for assignment in assignments
        if assignment.is_default and assignment.enabled
    ]
    if not enabled_defaults:
        return None
    max_updated_at = max(assignment.updated_at for assignment in enabled_defaults)
    newest = [assignment for assignment in enabled_defaults if assignment.updated_at == max_updated_at]
    return sorted(newest, key=lambda assignment: assignment.id)[0].model_copy(deep=True)
```

Also add:

```python
async def load_gateway_default_assignment(
    assignment_store: ProfileAssignmentStore,
) -> ProfileAssignment | None:
    assignments = await assignment_store.list_assignments()
    return select_gateway_default_assignment(assignments)
```

- [ ] **Step 3: Add assignment-aware resolver**

In `mcp_unified/profiles/resolver.py`, add `AssignmentBackedProfileResolver` without changing existing `StoreBackedProfileResolver` semantics.

Required constructor:

```python
class AssignmentBackedProfileResolver(StoreBackedProfileResolver):
    def __init__(
        self,
        profile_store: ProfileStore,
        *,
        assignment_store: ProfileAssignmentStore,
        fallback_default_profile_id: str | None = None,
    ) -> None: ...
```

Resolution order:

1. Explicit `profile_id`.
2. Stored default assignment from `assignment_store`.
3. `fallback_default_profile_id`.
4. `profile_required`.

Use provenance keys:

- `requested_profile_id`
- `resolved_profile_id`
- `used_default_assignment`
- `used_default_profile`
- `default_assignment_id`
- `resolver`

If the assignment store raises a known assignment-store unavailable error, return:

```python
ProfileResolutionResult(
    status="store_unavailable",
    reason_code="assignment_store_unavailable",
    provenance={...},
)
```

- [ ] **Step 4: Add manager error and metadata types**

In `mcp_unified/gateway/profiles.py`, implement:

```python
@dataclass(frozen=True, slots=True)
class GatewayProfileStoreMetadata:
    kind: Literal["memory", "sqlite"]
    persistent: bool

    def to_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "persistent": self.persistent}


class GatewayProfileManagementError(RuntimeError):
    def __init__(self, message: str, *, reason_code: str, profile_id: str | None = None, preset_id: str | None = None) -> None: ...

    def to_payload(self) -> dict[str, Any]:
        payload = {"ok": False, "error": str(self), "reason_code": self.reason_code}
        ...
```

Keep payload keys sorted naturally by `json.dumps(..., sort_keys=True)` at the CLI boundary, not by manually reordering dictionaries everywhere.

- [ ] **Step 5: Implement manager read operations**

Implement:

```python
class GatewayProfileManager:
    async def list_profiles(self) -> dict[str, Any]: ...
    async def show_profile(self, profile_id: str) -> dict[str, Any]: ...
```

Behavior:

- validate non-empty profile ids;
- call the profile store once per operation;
- return profile models with `model_dump(mode="json")`;
- include `"ok": True` and `"store": store_metadata.to_payload()`;
- convert expected store-unavailable exceptions into `profile_store_unavailable`.

- [ ] **Step 6: Implement preset duplication**

Implement:

```python
async def duplicate_preset(
    self,
    preset_id: str,
    *,
    profile_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]: ...
```

Behavior:

- trim ids/names and reject blank request values with `invalid_profile_request`;
- use `get_builtin_preset()`/`duplicate_builtin_preset()`;
- reject collisions with `profile_already_exists`;
- apply optional display name only to the duplicated stored profile;
- preserve `preset_id`, `preset_version`, created/updated preset provenance;
- `upsert_profile()` and return the stored profile.

- [ ] **Step 7: Implement default get/set**

Implement:

```python
async def get_default_profile(self) -> dict[str, Any]: ...
async def set_default_profile(self, profile_id: str) -> dict[str, Any]: ...
```

Behavior:

- `set_default_profile()` loads profile, rejects missing/disabled target, writes `ProfileAssignment(id=GATEWAY_DEFAULT_ASSIGNMENT_ID, profile_id=..., is_default=True, provenance={...})`.
- Preserve `created_at` when updating an existing `gateway-default` assignment if one already exists; always update `updated_at`.
- `get_default_profile()` uses `load_gateway_default_assignment()`, then fallback default id, then returns `default_profile_not_configured`.
- Both operations return the profile and assignment/default provenance in deterministic JSON.

- [ ] **Step 8: Add audit emission**

Add private helper:

```python
async def _append_audit_event(
    self,
    event_type: str,
    *,
    profile_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> None: ...
```

Emit append-only events only when `audit_store` is configured. Payloads should include ids, reason codes, and preset/default provenance only. Do not include full profile documents or secrets.

The implementation must emit expected-failure audit events before raising domain errors for missing profiles, disabled profiles, unknown presets, and id collisions. Keep unexpected infrastructure failures outside this expected-failure audit path.

- [ ] **Step 9: Export new helper types**

Update `mcp_unified/gateway/__init__.py` to export manager and error/metadata types without importing FastAPI.

- [ ] **Step 10: Run GREEN tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py -q
```

Expected: new manager/resolver tests pass.

- [ ] **Step 11: Commit Task 2**

Commit manager, default-helper, resolver, and tests:

```bash
git add mcp_unified/gateway/profiles.py mcp_unified/gateway/__init__.py mcp_unified/profiles/defaults.py mcp_unified/profiles/resolver.py mcp_unified/profiles/store.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py
git commit -m "feat: add gateway profile manager"
```

### Task 3: Bootstrap And Config Wiring

**Files:**
- Modify: `mcp_unified/gateway/bootstrap.py`
- Modify: `mcp_unified/gateway/config.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Add bootstrap/config RED tests**

Extend `test_gateway_fastapi_package.py` with tests that assert:

- `bootstrap_profile_gateway()` exposes `profile_manager`, `assignment_store`, `audit_store`, and `store_metadata`;
- memory bootstrap uses `InMemoryProfileAssignmentStore` and `"memory"/False` metadata;
- config bootstrap with SQLite uses the same `SQLiteMCPStore` instance for profile, assignment, and audit protocols;
- setting default through `bootstrap.profile_manager` affects no-profile JSON-RPC calls without recreating the app/runtime;
- explicit header profile still overrides the stored default.

Runtime default-change test shape:

```python
bootstrap = asyncio.run(bootstrap_profile_gateway(... profiles=[reviewer, architect]))
app = create_gateway_app(bootstrap.runtime, prefix="/mcp")

asyncio.run(bootstrap.profile_manager.set_default_profile("reviewer"))
first = client.post("/mcp/request", json={... tools/list ...})

asyncio.run(bootstrap.profile_manager.set_default_profile("architect"))
second = client.post("/mcp/request", json={... tools/list ...})

assert _tool_names(first) == ["echo.search"]
assert _tool_names(second) == ["admin.delete"]
```

- [ ] **Step 2: Run RED tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: new tests fail because bootstrap/config do not expose assignment stores or manager.

- [ ] **Step 3: Add storage bundle in config**

In `mcp_unified/gateway/config.py`, add:

```python
@dataclass(frozen=True, slots=True)
class GatewayProfileStorageBundle:
    profile_store: ProfileStore
    assignment_store: ProfileAssignmentStore
    audit_store: AuditStore | None
    metadata: GatewayProfileStoreMetadata
```

Implement `build_gateway_profile_storage()`:

- memory config returns `InMemoryProfileStore`, `InMemoryProfileAssignmentStore`, `audit_store=None`, metadata `{kind: "memory", persistent: False}`;
- sqlite config constructs one `SQLiteMCPStore` and reuses it as profile/assignment/audit store, metadata `{kind: "sqlite", persistent: True}`;
- injected stores override config where supplied, but if no assignment store is supplied, use memory assignment store for memory/injected tests;
- do not import `SQLiteMCPStore` at module import time.

- [ ] **Step 4: Extend bootstrap result**

In `mcp_unified/gateway/bootstrap.py`, extend `GatewayProfileBootstrap`:

```python
@dataclass(frozen=True, slots=True)
class GatewayProfileBootstrap:
    runtime: ProfileAwareGatewayRuntime
    profile_store: ProfileStore
    assignment_store: ProfileAssignmentStore
    audit_store: AuditStore | None
    profile_manager: GatewayProfileManager
    store_metadata: GatewayProfileStoreMetadata
    default_profile_id: str | None
    seeded_profile_ids: tuple[str, ...]
```

Do not remove existing fields.

- [ ] **Step 5: Wire assignment-aware resolver into bootstrap**

When `bootstrap_profile_gateway()` builds the runtime, create one shared assignment store and pass:

```python
resolver = AssignmentBackedProfileResolver(
    store,
    assignment_store=assignment_store,
    fallback_default_profile_id=resolved_default_profile_id,
)
runtime = ProfileAwareGatewayRuntime(backend, profile_resolver=resolver)
```

Also create:

```python
profile_manager = GatewayProfileManager(
    profile_store=store,
    assignment_store=assignment_store,
    audit_store=audit_store,
    fallback_default_profile_id=resolved_default_profile_id,
    store_metadata=store_metadata,
)
```

This is the core consistency requirement: runtime and manager share the same `assignment_store` object.

- [ ] **Step 6: Update config bootstrap to use storage bundle**

Have `bootstrap_profile_gateway_from_config()` call `build_gateway_profile_storage()` and pass profile/assignment/audit stores plus metadata into `bootstrap_profile_gateway()`.

Preserve existing caller behavior:

- default config still creates memory store;
- injected `profile_store` still wins for existing tests;
- `default_preset_id` still seeds the configured profile store.

- [ ] **Step 7: Run GREEN tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py -q
```

Expected: bootstrap/config/profile management tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add mcp_unified/gateway/bootstrap.py mcp_unified/gateway/config.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
git commit -m "feat: wire gateway profile defaults"
```

### Task 4: CLI Management Commands

**Files:**
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [ ] **Step 1: Add CLI RED tests for config source handling**

Add tests for:

- profile-management commands require `--config` or `MCP_UNIFIED_GATEWAY_CONFIG`;
- explicit `--config` wins over environment fallback;
- config loader failures return exit code `1` with JSON stderr;
- parse/request-shape failures return exit code `2` with JSON stderr.
- existing catalog-only `list-presets` and `show-preset` still work without config.

Use `monkeypatch.setenv("MCP_UNIFIED_GATEWAY_CONFIG", str(config_path))`.

- [ ] **Step 2: Add CLI RED tests for read-only commands**

Cover exact envelopes:

```json
{"ok": true, "profiles": [...], "store": {"kind": "memory", "persistent": false}}
{"ok": true, "profile": {...}, "store": {"kind": "memory", "persistent": false}}
{"ok": true, "profile": {...}, "assignment": {...}, "store": {"kind": "sqlite", "persistent": true}}
```

Commands:

- `list-profiles --config <path>`
- `show-profile <profile_id> --config <path>`
- `get-default-profile --config <path>`

Also cover a memory config with `default_preset_id="project-researcher"` so read-only commands can inspect config-seeded memory profiles and `get-default-profile` can return the fallback preset without an assignment record.

- [ ] **Step 3: Add CLI RED tests for mutating commands**

Cover:

- `duplicate-preset <preset_id> --config <sqlite-config>`;
- `duplicate-preset <preset_id> --profile-id <id> --name <name> --config <sqlite-config>`;
- `set-default-profile <profile_id> --config <sqlite-config>`;
- memory-store `duplicate-preset` and `set-default-profile` return exit code `1`, JSON stderr, and `reason_code="profile_store_unavailable"`.

Do not add a development override flag in this slice unless implementation discovers a strong need and updates this plan first.

- [ ] **Step 4: Implement parser commands**

Add subcommands:

- `list-profiles`
- `show-profile <profile_id>`
- `duplicate-preset <preset_id> [--profile-id <profile_id>] [--name <name>]`
- `get-default-profile`
- `set-default-profile <profile_id>`

Add common `--config` option to each profile-management subparser, not to catalog-only `list-presets`/`show-preset`.

- [ ] **Step 5: Add config-to-manager helper at CLI boundary**

Implement a private CLI helper:

```python
def _config_path_from_args(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config
    env_value = os.environ.get("MCP_UNIFIED_GATEWAY_CONFIG")
    if env_value and env_value.strip():
        return Path(env_value)
    raise _CliArgumentError("--config is required unless MCP_UNIFIED_GATEWAY_CONFIG is set")
```

Use `load_gateway_profile_bootstrap_config()` plus `build_gateway_profile_storage()` to create the manager without creating a runtime/backend.

Important seeding rule:

- For persistent stores, the CLI should operate on the configured store and must not silently seed profiles during read-only commands.
- For memory-store read-only commands, seed `config.profiles` and `default_preset_id` into the fresh memory profile store so tests and local development configs can inspect config-defined profiles.
- For memory-store mutating commands, reject the operation before mutation even if the config contains seed profiles.
- Use `config.default_profile_id or config.default_preset_id` as the manager fallback default id for memory read-only commands after seeding.

If this duplicates bootstrap seeding logic, extract a small package-private helper from `mcp_unified/gateway/bootstrap.py` and call it from both runtime bootstrap and the CLI helper. Do not create a fake runtime just to reuse `bootstrap_profile_gateway_from_config()`.

- [ ] **Step 6: Add async command runner**

CLI handlers are synchronous. Add:

```python
def _run_async(coro: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
    return asyncio.run(coro)
```

Close stores that expose `aclose()` in a `finally` block after each command.

- [ ] **Step 7: Add domain-error handling**

Catch `GatewayProfileManagementError` and emit `exc.to_payload()` to stderr with exit code `1`. Do not catch broad exceptions around manager operations except at config-loading boundaries already covered by existing CLI style.

- [ ] **Step 8: Reject mutating memory-store configs**

Before invoking `duplicate_preset` or `set_default_profile`, check bundle metadata:

```python
if not bundle.metadata.persistent:
    raise GatewayProfileManagementError(
        "Profile management mutation requires a persistent gateway store",
        reason_code="profile_store_unavailable",
    )
```

Keep read-only memory commands allowed and include exact memory store metadata.

- [ ] **Step 9: Run CLI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 10: Commit Task 4**

```bash
git add mcp_unified/gateway/cli.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
git commit -m "feat: add gateway profile management CLI"
```

### Task 5: FastAPI Management Routes

**Files:**
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Add route-gating RED tests**

Assert:

```python
app = create_gateway_app(runtime, prefix="/mcp")
response = client.get("/mcp/profiles")
assert response.status_code == 404
```

Then assert management routes mount when either:

- `create_gateway_app(runtime, prefix="/mcp", profile_manager=manager)`;
- `create_gateway_app(runtime, prefix="/mcp", profile_bootstrap=bootstrap)`;
- `enable_profile_management=True` is passed with a valid manager/bootstrap.

If `enable_profile_management=True` is passed without a manager/bootstrap, assert `ValueError` at app/router construction. This makes the app-factory route gate concrete.

- [ ] **Step 2: Add endpoint envelope RED tests**

Cover exact success payloads:

- `GET /profiles`
- `GET /profiles/{profile_id}`
- `POST /profiles/from-preset` with only `preset_id`;
- `POST /profiles/from-preset` with custom `profile_id` and `name`;
- `GET /profiles/default`;
- `PUT /profiles/default`.

Assert success payloads include `"ok": true` and `"store"`.

- [ ] **Step 3: Add error/status RED tests**

Assert reason-code mapping:

```python
assert client.get("/mcp/profiles/missing").status_code == 404
assert body["reason_code"] == "profile_not_found"
```

Cover:

- `profile_not_found` -> `404`
- `preset_not_found` -> `404`
- `default_profile_not_configured` -> `404`
- `profile_disabled` -> `409`
- `profile_already_exists` -> `409`
- `profile_store_unavailable` -> `503`
- `assignment_store_unavailable` -> `503`
- request body validation -> `422`

Use small failing test doubles for profile/assignment stores so unavailable-store domain errors are translated to JSON responses instead of leaking as generic `500` responses.

- [ ] **Step 4: Add runtime default-change test**

Assert `PUT /profiles/default` changes later no-profile JSON-RPC behavior without recreating the FastAPI app:

1. Build bootstrap with two profiles that expose different tool sets.
2. Create app with `profile_bootstrap=bootstrap`.
3. `PUT /mcp/profiles/default` to first profile and call `tools/list`.
4. `PUT /mcp/profiles/default` to second profile and call `tools/list`.
5. Assert the tool list changes.

- [ ] **Step 5: Add request models**

Inside `mcp_unified/gateway/fastapi.py`, add package-local Pydantic request models:

```python
class DuplicatePresetRequest(BaseModel):
    preset_id: str
    profile_id: str | None = None
    name: str | None = None


class SetDefaultProfileRequest(BaseModel):
    profile_id: str
```

Only `preset_id` is required for duplication.

- [ ] **Step 6: Add profile-management route mounting**

Change signatures compatibly:

```python
def create_gateway_router(
    runtime: GatewayRuntime,
    *,
    profile_manager: GatewayProfileManager | None = None,
    profile_bootstrap: GatewayProfileBootstrap | None = None,
    enable_profile_management: bool = False,
) -> APIRouter: ...
```

Route gate:

```python
manager = _resolve_profile_manager(
    profile_manager=profile_manager,
    profile_bootstrap=profile_bootstrap,
    enable_profile_management=enable_profile_management,
)
if manager is not None:
    _mount_profile_management_routes(router, manager)
```

`_resolve_profile_manager()` must:

- prefer explicit `profile_manager`;
- otherwise use `profile_bootstrap.profile_manager`;
- return `None` when not enabled and no manager/bootstrap was supplied;
- raise `ValueError` when `enable_profile_management=True` but no manager/bootstrap exists.

Mirror this through `create_gateway_app()`.

- [ ] **Step 7: Add error translation**

Use a helper:

```python
def _profile_error_status(exc: GatewayProfileManagementError) -> int:
    return PROFILE_REASON_STATUS.get(exc.reason_code, 500)
```

Return `JSONResponse(status_code=..., content=exc.to_payload())` for expected domain errors. Let unexpected exceptions propagate to FastAPI's normal error machinery.

- [ ] **Step 8: Run FastAPI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: all gateway FastAPI tests pass.

- [ ] **Step 9: Commit Task 5**

```bash
git add mcp_unified/gateway/fastapi.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
git commit -m "feat: expose gateway profile management API"
```

### Task 6: SQLite, Boundaries, And Compatibility Validation

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py` only if a gap is found in existing SQLite assignment/audit coverage.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py` only if boundary tests need new explicit file coverage.
- Modify: `Docs/superpowers/plans/2026-05-31-mcp-unified-stage4k-gateway-profile-management-implementation-plan.md`
- Modify: `backlog/tasks/task-571 - Implement-MCP-Unified-Stage-4K-gateway-profile-management.md`

- [x] **Step 1: Run focused SQLite/storage tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py -q
```

Result: `26 passed, 3 warnings`. No SQLite/storage coverage gap was found.

- [x] **Step 2: Run package boundary tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
```

Result: `43 passed, 3 warnings`. Existing package boundary scans cover the new files.

- [x] **Step 3: Run focused Stage 4K suite**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py -q
```

Result: `135 passed, 4 warnings`.

- [x] **Step 4: Run lint**

Run:

```bash
source .venv/bin/activate
python -m ruff check mcp_unified/gateway mcp_unified/profiles tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py
```

Result: `All checks passed!`

- [x] **Step 5: Run Bandit on touched production package scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r mcp_unified/gateway mcp_unified/profiles -f json -o /tmp/bandit_mcp_stage4k_profile_management.json
```

Result: `/tmp/bandit_mcp_stage4k_profile_management.json` reported `results: []` and `errors: []`.

- [x] **Step 6: Run whitespace check**

Run:

```bash
git diff --check
```

Result: no output and exit code `0`.

- [x] **Step 7: Update plan and Backlog task**

Record RED/GREEN results, focused verification outputs, any skips, and final summary in:

- `Docs/superpowers/plans/2026-05-31-mcp-unified-stage4k-gateway-profile-management-implementation-plan.md`
- `backlog/tasks/task-571 - Implement-MCP-Unified-Stage-4K-gateway-profile-management.md`

Check off acceptance criteria and Definition of Done when complete.

- [x] **Step 8: Commit final validation updates**

```bash
git add Docs/superpowers/plans/2026-05-31-mcp-unified-stage4k-gateway-profile-management-implementation-plan.md backlog/tasks/task-571\ -\ Implement-MCP-Unified-Stage-4K-gateway-profile-management.md
git commit -m "chore: validate gateway profile management implementation"
```

## Final Verification Command Set

Run this before claiming implementation complete:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py tldw_Server_API/app/core/MCP_unified/tests/test_sqlite_storage_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_storage_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
python -m ruff check mcp_unified/gateway mcp_unified/profiles tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_registry_resolver.py
python -m bandit -r mcp_unified/gateway mcp_unified/profiles -f json -o /tmp/bandit_mcp_stage4k_profile_management.json
git diff --check
```

Result: final combined pytest command reported `204 passed, 4 warnings`; ruff
reported `All checks passed!`; Bandit reported `results: []`; `git diff
--check` passed.

## Implementation Review Checklist

- [x] No new `tldw_Server_API` imports under `mcp_unified/gateway` or `mcp_unified/profiles`.
- [x] Default profile selection uses `ProfileAssignmentStore`; no process-local mutable default holder is added.
- [x] Runtime and manager share the same assignment store in bootstrap/config paths.
- [x] `PUT /profiles/default` affects later no-profile JSON-RPC requests without app/runtime restart.
- [x] Explicit header/query profile id still overrides stored default.
- [x] Management endpoints are absent from the default app/router.
- [x] CLI mutating commands reject memory-store configs.
- [x] Read-only memory CLI payload includes exactly `{"store": {"kind": "memory", "persistent": false}}`.
- [x] Domain failures emit JSON without tracebacks.
- [x] Audit payloads contain ids/reasons/provenance only, not full profile documents.
