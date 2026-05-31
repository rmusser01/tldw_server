# MCP Unified Stage 4L Editable Profile CRUD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manager-owned editable gateway profile create, limited patch, and guarded delete operations with matching FastAPI and CLI surfaces.

**Architecture:** Extend `GatewayProfileManager` as the only mutation boundary, then keep FastAPI and CLI thin over that manager. Add a narrow guarded-delete capability for persistent stores so profile assignment checks and profile deletion cannot race. Preserve Stage 4K response envelopes, compact audit posture, package boundaries, and persistent-store-only CLI mutations.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, package-local MCP profile/storage protocols, SQLAlchemy-backed SQLite store, pytest, Bandit.

---

## Source Design

- Spec: `Docs/superpowers/specs/2026-05-31-mcp-unified-stage4l-editable-profile-crud-design.md`
- Backlog: `TASK-574`
- Prior stage to preserve:
  - `Docs/superpowers/specs/2026-05-30-mcp-unified-stage4k-gateway-profile-management-design.md`
  - `Docs/superpowers/plans/2026-05-31-mcp-unified-stage4k-gateway-profile-management-implementation-plan.md`

## Scope Boundaries

Build only:

- Full-document stored profile create.
- Limited safe profile patch.
- Guarded hard delete.
- Manager-owned validation, default safety checks, assignment checks, and compact audit events.
- FastAPI endpoints for `POST /profiles`, `PATCH /profiles/{profile_id}`, and `DELETE /profiles/{profile_id}`.
- CLI commands for `create-profile`, `patch-profile`, and `delete-profile`.

Do not build:

- Profile assignment CRUD.
- Workspace or principal binding management.
- Approval policy editing.
- Path scope editing.
- External server grants.
- Credential grants.
- Storage schema migrations.
- Audit viewer APIs.
- Front-end UI changes.

## File Map

- Modify: `mcp_unified/gateway/profiles.py`
  - Add `create_profile()`, `patch_profile()`, and `delete_profile()`.
  - Add patch-shape validation helpers, effective-default helper, assignment guard helper, compact changed-field audit payloads, and guarded-delete dispatch.
- Modify: `mcp_unified/interfaces/storage.py`
  - Add a narrow optional protocol/capability for guarded profile deletion, for example `GuardedProfileDeleteStore`.
- Modify: `mcp_unified/profiles/store.py`
  - Add process-local locking or a guarded-delete method for `InMemoryProfileStore`/assignment-store combinations if the implementation chooses a shared memory helper.
- Modify: `mcp_unified/storage/sqlite.py`
  - Implement atomic persistent guarded delete using the existing SQLAlchemy backend and current tables, without schema migration.
- Modify: `mcp_unified/gateway/fastapi.py`
  - Add request/response models and management routes for create, patch, delete.
  - Extend `_PROFILE_MANAGEMENT_STATUS_CODES`.
- Modify: `mcp_unified/gateway/cli.py`
  - Add parser commands, JSON file/stdin loading helpers, manager adapters, and persistent-store enforcement for new mutations.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`
  - Manager CRUD, audit, default safety, assignment safety, and persistent guarded-delete tests.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`
  - HTTP route contracts and status mappings.
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`
  - CLI command contracts, stdin/file handling, malformed JSON, and memory-store mutation rejection.

## Public Contracts To Preserve

- Stage 4K profile management routes and CLI commands keep their current response shapes.
- Profile mutations remain manager-owned; FastAPI and CLI do not call stores directly.
- Package code in `mcp_unified` does not import `tldw_Server_API`.
- CLI mutating profile-management commands reject nonpersistent memory-store configs.
- Audit failures stay best effort and do not fail successful profile mutations.
- Full profile create may preserve valid `created_at`, but must set `updated_at` to the mutation time.
- Patch preserves omitted fields, rejects unsupported fields, and rejects no-op patch documents.
- Delete refuses current effective default and any assigned profile.

## Domain Shapes

Extend the FastAPI reason-code map with:

```python
_PROFILE_MANAGEMENT_STATUS_CODES = {
    "profile_not_found": 404,
    "preset_not_found": 404,
    "default_profile_not_configured": 404,
    "profile_disabled": 409,
    "profile_already_exists": 409,
    "invalid_profile_request": 422,
    "profile_store_unavailable": 503,
    "assignment_store_unavailable": 503,
    "profile_is_default": 409,
    "profile_has_assignments": 409,
    "invalid_profile_patch": 422,
}
```

Use reason codes consistently:

- `profile_already_exists`: create target already exists.
- `profile_is_default`: create, patch, or delete would make the effective gateway default missing or disabled.
- `profile_has_assignments`: delete target has non-default assignments or any assignment references.
- `invalid_profile_patch`: unsupported patch field, unsupported nested policy key, malformed patch shape after JSON parsing, or no-op patch.
- `invalid_profile_request`: malformed full-create request after JSON parsing or missing required text arguments.

Allowed patch fields:

```python
PROFILE_PATCH_FIELDS = {"name", "description", "enabled", "metadata", "policy_document"}
POLICY_PATCH_FIELDS = {
    "allowed_tools",
    "denied_tools",
    "capabilities",
    "denied_capabilities",
    "tool_patterns",
    "module_patterns",
    "risk_classes",
    "resource_constraints",
}
```

---

### Task 1: Manager RED Tests For Create, Patch, And Delete

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`

- [ ] **Step 1: Add create success and duplicate RED tests**

Append manager tests near the existing `duplicate_preset` tests:

```python
@pytest.mark.asyncio
async def test_create_profile_persists_valid_profile_and_audits_success() -> None:
    created_at = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    audit_store = InMemoryAuditStore()
    store = InMemoryProfileStore()
    manager = _manager(store, audit_store=audit_store)

    payload = await manager.create_profile(
        {
            "id": "custom-reviewer",
            "name": "Custom Reviewer",
            "metadata": {"owner": "qa"},
            "created_at": created_at.isoformat(),
            "updated_at": created_at.isoformat(),
        }
    )

    assert payload["ok"] is True
    assert payload["profile"]["id"] == "custom-reviewer"
    assert payload["profile"]["metadata"] == {"owner": "qa"}
    assert payload["profile"]["created_at"] == created_at.isoformat()
    assert payload["profile"]["updated_at"] != created_at.isoformat()
    assert await store.get_profile("custom-reviewer") is not None
    assert [event.event_type for event in audit_store.events] == ["profile.created"]
```

Add a duplicate test:

```python
@pytest.mark.asyncio
async def test_create_profile_rejects_duplicate_id() -> None:
    manager = _manager(InMemoryProfileStore([MCPProfile(id="existing", name="Existing")]))

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.create_profile({"id": "existing", "name": "Duplicate"})

    assert exc_info.value.reason_code == "profile_already_exists"
    assert exc_info.value.to_payload()["profile_id"] == "existing"
```

- [ ] **Step 2: Add effective-default disabled create RED tests**

Cover assignment-first resolution and fallback-only behavior:

```python
@pytest.mark.asyncio
async def test_create_profile_rejects_disabled_effective_default_assignment_id() -> None:
    assignment_store = InMemoryProfileAssignmentStore(
        [ProfileAssignment(id="gateway-default", profile_id="default", is_default=True)]
    )
    manager = _manager(InMemoryProfileStore(), assignment_store)

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.create_profile({"id": "default", "name": "Default", "enabled": False})

    assert exc_info.value.reason_code == "profile_is_default"
```

```python
@pytest.mark.asyncio
async def test_create_profile_allows_disabled_fallback_id_when_assignment_overrides_it() -> None:
    assignment_store = InMemoryProfileAssignmentStore(
        [ProfileAssignment(id="gateway-default", profile_id="assigned", is_default=True)]
    )
    manager = _manager(
        InMemoryProfileStore([MCPProfile(id="assigned", name="Assigned")]),
        assignment_store,
        fallback_default_profile_id="fallback",
    )

    payload = await manager.create_profile(
        {"id": "fallback", "name": "Fallback", "enabled": False}
    )

    assert payload["profile"]["id"] == "fallback"
    assert payload["profile"]["enabled"] is False
```

Also add a fallback-only rejection test:

```python
@pytest.mark.asyncio
async def test_create_profile_rejects_disabled_fallback_default_without_assignment() -> None:
    manager = _manager(
        InMemoryProfileStore(),
        fallback_default_profile_id="fallback",
    )

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.create_profile(
            {"id": "fallback", "name": "Fallback", "enabled": False}
        )

    assert exc_info.value.reason_code == "profile_is_default"
```

- [ ] **Step 3: Add patch allowed-field RED tests**

Add a test that patches `name`, `description`, `enabled`, `metadata`, and two `policy_document` fields while preserving omitted fields:

```python
@pytest.mark.asyncio
async def test_patch_profile_replaces_allowed_fields_and_preserves_omitted_fields() -> None:
    original = MCPProfile(
        id="reviewer",
        name="Reviewer",
        description="old",
        metadata={"old": True},
        policy_document={"allowed_tools": ["old.tool"], "denied_tools": ["old.deny"]},
    )
    manager = _manager(InMemoryProfileStore([original]))

    payload = await manager.patch_profile(
        "reviewer",
        {
            "name": "Senior Reviewer",
            "description": "new",
            "metadata": {"new": True},
            "policy_document": {"allowed_tools": ["new.tool"]},
        },
    )

    profile = payload["profile"]
    assert profile["name"] == "Senior Reviewer"
    assert profile["description"] == "new"
    assert profile["metadata"] == {"new": True}
    assert profile["policy_document"]["allowed_tools"] == ["new.tool"]
    assert profile["policy_document"]["denied_tools"] == ["old.deny"]
```

- [ ] **Step 4: Add patch rejection RED tests**

Cover:

- unsupported top-level field;
- unknown nested policy field;
- empty `{}`;
- `{"policy_document": {}}`;
- semantic no-op patch such as `{"name": existing_name}`;
- default-profile disable.

Expected assertion:

```python
with pytest.raises(GatewayProfileManagementError) as exc_info:
    await manager.patch_profile("reviewer", {"approval_policy": {"mode": "auto"}})
assert exc_info.value.reason_code == "invalid_profile_patch"
```

For semantic no-op, assert the manager does not update `updated_at`:

```python
before = (await store.get_profile("reviewer")).updated_at
with pytest.raises(GatewayProfileManagementError) as exc_info:
    await manager.patch_profile("reviewer", {"name": "Reviewer"})
after = (await store.get_profile("reviewer")).updated_at
assert exc_info.value.reason_code == "invalid_profile_patch"
assert after == before
```

- [ ] **Step 5: Add delete RED tests**

Cover:

- deleting a non-default unassigned profile returns `{"ok": True, "profile_id": "temporary"}`;
- missing profile returns `profile_not_found`;
- effective default deletion returns `profile_is_default`;
- assigned profile deletion returns `profile_has_assignments`;
- failure audit payloads remain compact and omit `policy_document`.
- expected failure audit events are emitted for unsupported patch fields,
  semantic no-op patches, default disable, default delete, and assignment delete
  protection.

- [ ] **Step 6: Add persistent guarded-delete RED test**

Use `SQLiteMCPStore` in a temp DB. Seed a profile and assignment, call the manager delete path, and assert it fails before cascade removes the assignment:

Add imports if they are not already present:

```python
from pathlib import Path

from mcp_unified.storage.sqlite import SQLiteMCPStore
```

```python
@pytest.mark.asyncio
async def test_delete_profile_sqlite_guard_preserves_assigned_profile(tmp_path: Path) -> None:
    store = SQLiteMCPStore(tmp_path / "mcp.db")
    await store.upsert_profile(MCPProfile(id="assigned", name="Assigned"))
    await store.upsert_assignment(
        ProfileAssignment(id="workspace-assignment", profile_id="assigned", workspace_id="ws")
    )
    manager = GatewayProfileManager(
        profile_store=store,
        assignment_store=store,
        store_metadata=GatewayProfileStoreMetadata(kind="sqlite", persistent=True),
    )

    with pytest.raises(GatewayProfileManagementError) as exc_info:
        await manager.delete_profile("assigned")

    assert exc_info.value.reason_code == "profile_has_assignments"
    assert await store.get_profile("assigned") is not None
    assert await store.get_assignment("workspace-assignment") is not None
    await store.aclose()
```

- [ ] **Step 7: Run RED manager tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py -q
```

Expected: new tests fail because `GatewayProfileManager` does not yet expose `create_profile`, `patch_profile`, or `delete_profile`.

### Task 2: Manager, Patch Validation, And Guarded Delete Implementation

**Files:**
- Modify: `mcp_unified/interfaces/storage.py`
- Modify: `mcp_unified/gateway/profiles.py`
- Modify: `mcp_unified/storage/sqlite.py`
- Modify: `mcp_unified/profiles/store.py` if needed for an in-memory guarded-delete helper
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`

- [ ] **Step 1: Add optional guarded-delete protocol**

In `mcp_unified/interfaces/storage.py`, add a focused protocol without changing the existing `ProfileStore` contract:

```python
class GuardedProfileDeleteStore(Protocol):
    """Store capability for atomically deleting unassigned non-default profiles."""

    async def delete_profile_if_unassigned(
        self,
        profile_id: str,
        *,
        effective_default_profile_id: str | None,
    ) -> str:
        """Return deleted, not_found, is_default, or has_assignments."""
        raise NotImplementedError
```

Use return strings for simple manager mapping:

- `"deleted"`
- `"not_found"`
- `"is_default"`
- `"has_assignments"`

- [ ] **Step 2: Implement SQLite guarded delete atomically**

In `SQLiteMCPStore`, add async wrapper and sync implementation:

```python
async def delete_profile_if_unassigned(
    self,
    profile_id: str,
    *,
    effective_default_profile_id: str | None,
) -> str:
    return await self._run_db(
        self._delete_profile_if_unassigned_sync,
        profile_id,
        effective_default_profile_id=effective_default_profile_id,
    )
```

Inside the sync method, use one transaction on `self._engine.begin()` and SQLAlchemy table objects:

1. if `profile_id == effective_default_profile_id`, return `"is_default"`;
2. select profile row by id; if absent return `"not_found"`;
3. select one assignment row with `profile_id`; if present return `"has_assignments"`;
4. delete profile row;
5. return `"deleted"` if rowcount is truthy, otherwise `"not_found"`.

Do not add tables or migrations.

- [ ] **Step 3: Add manager helper constants and validators**

In `mcp_unified/gateway/profiles.py`, add module constants:

```python
_PROFILE_PATCH_FIELDS = frozenset({"name", "description", "enabled", "metadata", "policy_document"})
_POLICY_PATCH_FIELDS = frozenset(
    {
        "allowed_tools",
        "denied_tools",
        "capabilities",
        "denied_capabilities",
        "tool_patterns",
        "module_patterns",
        "risk_classes",
        "resource_constraints",
    }
)
```

Add helpers:

```python
async def _effective_default_profile_id(self) -> str | None:
    assignment = await self._load_default_assignment()
    if assignment is not None:
        return assignment.profile_id
    return self.fallback_default_profile_id

def _validate_patch_document(self, patch: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(patch)
    # Validate keys and no-op shape here; see the tests in Task 1.
    return normalized

def _apply_profile_patch(
    self,
    profile: MCPProfile,
    patch: Mapping[str, Any],
) -> tuple[MCPProfile, tuple[str, ...]]:
    # Copy existing profile, replace only supplied allowed fields, validate model,
    # and compare the JSON-safe before/after payloads while ignoring updated_at.
    # If no semantic field changed, raise invalid_profile_patch.
    return profile.model_copy(deep=True), tuple(sorted(patch))
```

The validator must reject non-mapping input, unsupported top-level keys, unsupported nested policy keys, empty patch, and empty policy-only patch using `invalid_profile_patch`. `_apply_profile_patch()` must reject semantic no-ops using `invalid_profile_patch` after comparing the pre-patch and post-patch model dumps with `updated_at` excluded.

- [ ] **Step 4: Implement `create_profile()`**

Implementation shape:

```python
async def create_profile(self, profile_document: MCPProfile | Mapping[str, Any]) -> dict[str, Any]:
    try:
        profile = profile_document.model_copy(deep=True) if isinstance(profile_document, MCPProfile) else MCPProfile.model_validate(profile_document)
    except Exception as exc:
        raise self._error("Invalid profile request", reason_code="invalid_profile_request") from exc

    if await self._get_profile(profile.id) is not None:
        await self._audit_expected_failure(
            "profile.create_failed",
            reason_code="profile_already_exists",
            profile_id=profile.id,
            target_type="profile",
            target_id=profile.id,
        )
        raise self._error(
            f"Profile already exists: {profile.id}",
            reason_code="profile_already_exists",
            profile_id=profile.id,
        )

    effective_default_id = await self._effective_default_profile_id()
    if not profile.enabled and profile.id == effective_default_id:
        await self._audit_expected_failure(
            "profile.create_failed",
            reason_code="profile_is_default",
            profile_id=profile.id,
            target_type="profile",
            target_id=profile.id,
        )
        raise self._error(
            f"Profile is the effective default: {profile.id}",
            reason_code="profile_is_default",
            profile_id=profile.id,
        )

    now = datetime.now(timezone.utc)
    profile = profile.model_copy(update={"updated_at": now}, deep=True)
    stored = await self.profile_store.upsert_profile(profile)
    await self._append_audit_event(
        "profile.created",
        profile_id=stored.id,
        target_type="profile",
        target_id=stored.id,
        payload={"profile_id": stored.id},
    )
    return {"ok": True, "profile": self._dump_profile(stored), "store": self.store_metadata.to_payload()}
```

- [ ] **Step 5: Implement `patch_profile()`**

Implementation shape:

```python
async def patch_profile(self, profile_id: str, patch_document: Mapping[str, Any]) -> dict[str, Any]:
    normalized_profile_id = self._require_text(profile_id, field="profile_id")
    try:
        patch = self._validate_patch_document(patch_document)
    except GatewayProfileManagementError:
        await self._audit_expected_failure(
            "profile.patch_failed",
            reason_code="invalid_profile_patch",
            profile_id=normalized_profile_id,
            target_type="profile",
            target_id=normalized_profile_id,
        )
        raise
    profile = await self._get_profile(normalized_profile_id)
    if profile is None:
        await self._audit_expected_failure(
            "profile.patch_failed",
            reason_code="profile_not_found",
            profile_id=normalized_profile_id,
            target_type="profile",
            target_id=normalized_profile_id,
        )
        raise self._error(
            f"Profile not found: {normalized_profile_id}",
            reason_code="profile_not_found",
            profile_id=normalized_profile_id,
        )
    if patch.get("enabled") is False and normalized_profile_id == await self._effective_default_profile_id():
        await self._audit_expected_failure(
            "profile.patch_failed",
            reason_code="profile_is_default",
            profile_id=normalized_profile_id,
            target_type="profile",
            target_id=normalized_profile_id,
        )
        raise self._error(
            f"Profile is the effective default: {normalized_profile_id}",
            reason_code="profile_is_default",
            profile_id=normalized_profile_id,
        )
    try:
        updated, changed_fields = self._apply_profile_patch(profile, patch)
    except GatewayProfileManagementError:
        await self._audit_expected_failure(
            "profile.patch_failed",
            reason_code="invalid_profile_patch",
            profile_id=normalized_profile_id,
            target_type="profile",
            target_id=normalized_profile_id,
        )
        raise
    updated = updated.model_copy(update={"updated_at": datetime.now(timezone.utc)}, deep=True)
    stored = await self.profile_store.upsert_profile(updated)
    await self._append_audit_event(
        "profile.patched",
        profile_id=stored.id,
        target_type="profile",
        target_id=stored.id,
        payload={"profile_id": stored.id, "changed_fields": list(changed_fields)},
    )
    return {"ok": True, "profile": self._dump_profile(stored), "store": self.store_metadata.to_payload()}
```

- [ ] **Step 6: Implement `delete_profile()`**

Implementation shape:

```python
async def delete_profile(self, profile_id: str) -> dict[str, Any]:
    normalized_profile_id = self._require_text(profile_id, field="profile_id")
    effective_default_id = await self._effective_default_profile_id()
    if normalized_profile_id == effective_default_id:
        await self._audit_expected_failure(
            "profile.delete_failed",
            reason_code="profile_is_default",
            profile_id=normalized_profile_id,
            target_type="profile",
            target_id=normalized_profile_id,
        )
        raise self._error(
            f"Profile is the effective default: {normalized_profile_id}",
            reason_code="profile_is_default",
            profile_id=normalized_profile_id,
        )

    guarded_delete = getattr(self.profile_store, "delete_profile_if_unassigned", None)
    if callable(guarded_delete):
        result = await guarded_delete(
            normalized_profile_id,
            effective_default_profile_id=effective_default_id,
        )
    elif not self.store_metadata.persistent:
        result = await self._manager_guarded_delete(normalized_profile_id)
    else:
        raise self._error("Profile store unavailable", reason_code="profile_store_unavailable")

    if result == "deleted":
        await self._append_audit_event(
            "profile.deleted",
            profile_id=normalized_profile_id,
            target_type="profile",
            target_id=normalized_profile_id,
            payload={"profile_id": normalized_profile_id},
        )
        return {"ok": True, "profile_id": normalized_profile_id, "store": self.store_metadata.to_payload()}
    if result == "not_found":
        await self._audit_expected_failure(
            "profile.delete_failed",
            reason_code="profile_not_found",
            profile_id=normalized_profile_id,
            target_type="profile",
            target_id=normalized_profile_id,
        )
        raise self._error(
            f"Profile not found: {normalized_profile_id}",
            reason_code="profile_not_found",
            profile_id=normalized_profile_id,
        )
    if result == "has_assignments":
        await self._audit_expected_failure(
            "profile.delete_failed",
            reason_code="profile_has_assignments",
            profile_id=normalized_profile_id,
            target_type="profile",
            target_id=normalized_profile_id,
        )
        raise self._error(
            f"Profile has assignments: {normalized_profile_id}",
            reason_code="profile_has_assignments",
            profile_id=normalized_profile_id,
        )
    raise self._error(
        f"Profile is the effective default: {normalized_profile_id}",
        reason_code="profile_is_default",
        profile_id=normalized_profile_id,
    )
```

For result mapping:

- `"not_found"` -> `profile_not_found`
- `"is_default"` -> `profile_is_default`
- `"has_assignments"` -> `profile_has_assignments`

- [ ] **Step 7: Run manager tests to GREEN**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py -q
```

Expected: all tests in this file pass.

- [ ] **Step 8: Commit manager/storage slice**

```bash
git add mcp_unified/interfaces/storage.py mcp_unified/gateway/profiles.py mcp_unified/storage/sqlite.py mcp_unified/profiles/store.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py
git commit -m "feat: add gateway profile CRUD manager"
```

### Task 3: FastAPI CRUD Routes

**Files:**
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Extend manager double in FastAPI tests**

Add methods to `_ProfileManagementManagerDouble`:

```python
async def create_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
    self.calls.append(("create_profile", (profile,), {}))
    return {
        "ok": True,
        "profile": {"id": profile["id"], "name": profile["name"]},
        "store": {"kind": "memory", "persistent": False},
    }

async def patch_profile(self, profile_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    self.calls.append(("patch_profile", (profile_id, patch), {}))
    return {
        "ok": True,
        "profile": {"id": profile_id, "name": patch.get("name", f"Profile {profile_id}")},
        "store": {"kind": "memory", "persistent": False},
    }

async def delete_profile(self, profile_id: str) -> dict[str, Any]:
    self.calls.append(("delete_profile", (profile_id,), {}))
    return {
        "ok": True,
        "profile_id": profile_id,
        "store": {"kind": "memory", "persistent": False},
    }
```

Record calls in `self.calls` and return Stage 4L success envelopes.

- [ ] **Step 2: Add RED route tests**

Add tests for:

- `POST /mcp/profiles` forwards full body to `manager.create_profile()`;
- `PATCH /mcp/profiles/{profile_id}` forwards body to `manager.patch_profile()`;
- `DELETE /mcp/profiles/{profile_id}` forwards to `manager.delete_profile()`;
- `GatewayProfileManagementError` reason codes map as specified;
- malformed `POST /profiles` body returns FastAPI/Pydantic validation response;
- empty patch response from manager maps `invalid_profile_patch` to 422.

Expected route assertion:

```python
def test_gateway_profile_management_create_profile_route() -> None:
    runtime = _FakeGatewayRuntime()
    manager = _ProfileManagementManagerDouble()
    app = create_gateway_app(runtime, profile_manager=manager)
    client = TestClient(app)

    response = client.post("/mcp/profiles", json={"id": "custom", "name": "Custom"})

    assert response.status_code == 200
    assert response.json()["profile"]["id"] == "custom"
    assert manager.calls == [("create_profile", ({"id": "custom", "name": "Custom"},), {})]
```

- [ ] **Step 3: Implement FastAPI models and routes**

Add models near existing management models:

```python
from pydantic import BaseModel, ConfigDict

class CreateProfileRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str

class PatchProfileRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

class DeleteProfileResponse(BaseModel):
    ok: bool
    profile_id: str
    store: StoreMetadataResponse
```

Prefer route-level `request.model_dump(exclude_unset=True)` for patch so omitted fields stay omitted. For create, pass the full model dump to the manager.

Add:

```python
@router.post("/profiles", response_model=ProfileResponse)
async def create_profile(request: CreateProfileRequest) -> ProfileResponse | JSONResponse:
    try:
        return await manager.create_profile(request.model_dump(mode="json"))
    except GatewayProfileManagementError as exc:
        return _profile_management_error_response(exc)

@router.patch("/profiles/{profile_id}", response_model=ProfileResponse)
async def patch_profile(profile_id: str, request: PatchProfileRequest) -> ProfileResponse | JSONResponse:
    try:
        return await manager.patch_profile(
            profile_id,
            request.model_dump(mode="json", exclude_unset=True),
        )
    except GatewayProfileManagementError as exc:
        return _profile_management_error_response(exc)

@router.delete("/profiles/{profile_id}", response_model=DeleteProfileResponse)
async def delete_profile(profile_id: str) -> DeleteProfileResponse | JSONResponse:
    try:
        return await manager.delete_profile(profile_id)
    except GatewayProfileManagementError as exc:
        return _profile_management_error_response(exc)
```

Extend `_PROFILE_MANAGEMENT_STATUS_CODES` with Stage 4L codes.

- [ ] **Step 4: Run FastAPI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q
```

Expected: file passes.

- [ ] **Step 5: Commit FastAPI slice**

```bash
git add mcp_unified/gateway/fastapi.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
git commit -m "feat: expose gateway profile CRUD routes"
```

### Task 4: CLI CRUD Commands

**Files:**
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [ ] **Step 1: Add CLI RED tests for create and patch file input**

Use existing `_write_gateway_config()` and `_profile_payload()` helpers. Add tests for:

- `create-profile --profile-file profile.json --config gateway.json`;
- `create-profile --profile-file - --config gateway.json` using `monkeypatch.setattr(gateway_cli.sys, "stdin", io.StringIO(json.dumps(profile_payload)))`;
- `patch-profile reviewer --patch-file patch.json --config gateway.json`;
- `patch-profile reviewer --patch-file - --config gateway.json`;
- malformed JSON in profile/patch file returns exit code `2` and JSON stderr;
- memory-store create/patch/delete rejects with exit code `1`.

Add `import io` at the top of the test file.

- [ ] **Step 2: Add CLI RED tests for delete**

Cover:

- deleting an unassigned SQLite-backed profile succeeds and emits compact payload;
- deleting the effective default exits `1` with `profile_is_default`;
- deleting assigned profile exits `1` with `profile_has_assignments`.

- [ ] **Step 3: Implement parser commands**

In `_build_parser()`, add:

```python
create_profile = subparsers.add_parser(
    "create-profile",
    help="Create a profile from a JSON document in a persistent gateway profile store.",
)
create_profile.add_argument("--profile-file", required=True, type=Path)
_add_profile_config_argument(create_profile)
create_profile.set_defaults(handler=_handle_create_profile)

patch_profile = subparsers.add_parser(
    "patch-profile",
    help="Patch safe editable fields on a profile in a persistent gateway profile store.",
)
patch_profile.add_argument("profile_id")
patch_profile.add_argument("--patch-file", required=True, type=Path)
_add_profile_config_argument(patch_profile)
patch_profile.set_defaults(handler=_handle_patch_profile)

delete_profile = subparsers.add_parser(
    "delete-profile",
    help="Delete an unassigned non-default profile from a persistent gateway profile store.",
)
delete_profile.add_argument("profile_id")
_add_profile_config_argument(delete_profile)
delete_profile.set_defaults(handler=_handle_delete_profile)
```

- [ ] **Step 4: Implement JSON file/stdin helpers**

Add:

```python
def _load_json_argument_file(path: Path, *, label: str) -> dict[str, Any]:
    raw = sys.stdin.read() if str(path) == "-" else path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _CliArgumentError(f"Invalid {label} JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise _CliArgumentError(f"{label} JSON must be an object")
    return payload
```

Use `_CliArgumentError` so malformed input exits `2`.

- [ ] **Step 5: Implement handlers**

Handlers:

```python
def _handle_create_profile(args: argparse.Namespace) -> int:
    profile = _load_json_argument_file(args.profile_file, label="profile")
    return _handle_profile_management_command(
        args,
        lambda manager: manager.create_profile(profile),
        require_persistent=True,
    )
```

Similarly implement patch and delete. Use `_require_cli_text()` for `profile_id`.

- [ ] **Step 6: Run CLI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q
```

Expected: file passes.

- [ ] **Step 7: Commit CLI slice**

```bash
git add mcp_unified/gateway/cli.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
git commit -m "feat: add gateway profile CRUD CLI"
```

### Task 5: Integration Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-574 - Plan-MCP-Unified-Stage-4L-editable-profile-CRUD-implementation.md`
- Modify or create implementation task if execution starts from this plan.

- [ ] **Step 1: Run focused MCP Unified tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run package import boundary check**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
```

Expected: package boundary tests pass and no `mcp_unified` package file imports `tldw_Server_API`.

- [ ] **Step 3: Run Bandit on touched package files**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  mcp_unified/gateway/profiles.py \
  mcp_unified/gateway/fastapi.py \
  mcp_unified/gateway/cli.py \
  mcp_unified/interfaces/storage.py \
  mcp_unified/storage/sqlite.py \
  mcp_unified/profiles/store.py \
  -f json -o /tmp/bandit_mcp_stage4l_profile_crud.json
```

Expected: no new security findings in touched code.

- [ ] **Step 4: Run diff hygiene checks**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended implementation files are modified before final staging.

- [ ] **Step 5: Update Backlog task records**

If this plan-only task is still open, update `TASK-574` with final plan-review status and mark it done. If implementation begins, create a separate implementation task before code edits or update the approved implementation task with modified files and verification results.

- [ ] **Step 6: Commit final verification/backlog updates**

```bash
git add 'backlog/tasks/task-574 - Plan-MCP-Unified-Stage-4L-editable-profile-CRUD-implementation.md'
git commit -m "chore: close profile CRUD planning"
```

Only run this commit if the plan task is being finalized separately from implementation.
