# MCP Unified Stage 4L Editable Profile CRUD Design

## Summary

Stage 4L adds controlled profile create, patch, and delete operations to the
standalone MCP gateway profile-management surface. It builds directly on Stage
4K's stored profile listing, inspection, preset duplication, and default-profile
selection work.

The slice stays intentionally narrow. Full profile documents may be created, but
updates are limited to safe profile metadata, enablement, and basic policy
filter fields. Deletion is a guarded hard delete that refuses to remove profiles
that are still the gateway default or have assignments.

## Current Context

The merged Stage 4K gateway profile-management work provides:

- `GatewayProfileManager` in `mcp_unified.gateway.profiles`.
- FastAPI profile-management endpoints for listing profiles, showing a profile,
  duplicating a built-in preset, reading the gateway default profile, and setting
  the gateway default profile.
- CLI commands for the same management operations.
- `ProfileStore` and `ProfileAssignmentStore` protocols with create/update and
  delete primitives already available.
- `InMemoryProfileStore`, `InMemoryProfileAssignmentStore`, and
  `SQLiteMCPStore` implementations that already support the needed profile and
  assignment storage operations.
- Profile-aware runtime resolution that can use a stored gateway default
  profile assignment.

Stage 4K deliberately did not add arbitrary profile mutation, workspace binding
management, approval policy editing, external server grants, credential grants,
or UI changes. Stage 4L fills the next smallest product gap: editable stored
profiles for front ends that need to create and maintain gateway modes.

## Goals

- Add manager-owned create, limited patch, and guarded delete operations for
  stored gateway profiles.
- Keep the manager as the only business-logic mutation boundary for CLI and
  FastAPI.
- Accept full validated `MCPProfile` documents for create.
- Accept only approved safe fields for patch.
- Reject operations that would leave the gateway default pointing at a missing
  or disabled profile.
- Reject deletion of profiles that still have assignments.
- Preserve compact audit behavior without logging full policy documents or
  grant-like material.
- Keep package boundaries intact: package code must not import
  `tldw_Server_API`.

## Non-Goals

- Profile assignment CRUD.
- Workspace or principal binding management.
- Approval policy editing.
- Path scope editing.
- External MCP server grant editing.
- Credential grant editing.
- Storage schema migrations.
- Audit viewer APIs.
- Front-end UI changes.
- Changing legacy `tldw_server` MCP behavior outside the standalone gateway
  package.

## Architecture

Extend `mcp_unified.gateway.profiles.GatewayProfileManager` with three methods:

- `create_profile(profile_document)`
- `patch_profile(profile_id, patch_document)`
- `delete_profile(profile_id)`

FastAPI endpoints and CLI commands call these manager methods only. They do not
mutate profile stores directly and do not duplicate default-profile, assignment,
validation, or audit rules.

The manager continues to depend on the same package-owned protocols:

- `ProfileStore` for stored profile documents.
- `ProfileAssignmentStore` for gateway default and future assignment lookup.
- Optional `AuditStore` for profile lifecycle audit events.

No new storage protocol is required. `ProfileStore` already exposes
`get_profile`, `list_profiles`, `upsert_profile`, and `delete_profile`.
`ProfileAssignmentStore` already supports listing assignments by `profile_id`.

## Manager Contract

### `create_profile(profile_document)`

Create accepts a full `MCPProfile` document. The manager validates the input
against the existing `MCPProfile` model, rejects duplicate profile IDs, updates
timestamps, persists through `ProfileStore.upsert_profile()`, and emits a
compact audit event.

Create is not an upsert. If a profile with the requested ID already exists, the
operation fails with a deterministic `profile_already_exists` reason.

The manager should preserve a valid provided `created_at` value. `updated_at`
should be set to the current time during create so the stored document reflects
the mutation time.

### `patch_profile(profile_id, patch_document)`

Patch loads the existing profile, applies only approved fields, validates the
resulting `MCPProfile`, persists it, and emits a compact audit event.

Allowed top-level patch fields:

- `name`
- `description`
- `enabled`
- `metadata`

Allowed `policy_document` patch fields:

- `allowed_tools`
- `denied_tools`
- `capabilities`
- `denied_capabilities`
- `tool_patterns`
- `module_patterns`
- `risk_classes`
- `resource_constraints`

Omitted fields are preserved. Provided list and dict fields replace that
specific field; they are not merged. Providing `metadata` replaces the entire
metadata object.

Out-of-scope fields must be rejected rather than silently ignored. This includes
`id`, `schema_version`, `preset_id`, `preset_version`, `approval_policy`,
`path_scopes`, `external_server_grants`, `credential_grants`, `provenance`,
`created_at`, and `updated_at`.

If the patch sets `enabled=false` for the current gateway default profile, the
operation fails. Callers must move the default first.

On success, `created_at` is preserved and `updated_at` is set to the current
time.

### `delete_profile(profile_id)`

Delete is a guarded hard delete. The manager loads the profile, checks whether
it is the current gateway default, checks for any assignments referencing the
profile, then calls `ProfileStore.delete_profile()` only when safe.

Delete fails when:

- The profile does not exist.
- The profile is the current gateway default.
- Any assignment exists for the profile.

The manager should not rely on SQLite foreign-key cascade behavior for safety.
The user-visible contract is that the control plane refuses unsafe deletion
before storage mutation.

## FastAPI Contract

Add package-owned endpoints under the existing gateway management router:

- `POST /profiles`
- `PATCH /profiles/{profile_id}`
- `DELETE /profiles/{profile_id}`

`POST /profiles` accepts a full `MCPProfile` document. The success response
matches the existing profile response shape:

```json
{
  "ok": true,
  "profile": {},
  "store": {"kind": "sqlite", "persistent": true}
}
```

`PATCH /profiles/{profile_id}` accepts a constrained patch body:

```json
{
  "name": "Researcher",
  "enabled": true,
  "metadata": {"owner": "workspace-a"},
  "policy_document": {
    "allowed_tools": ["search.*"],
    "denied_tools": ["shell.exec"]
  }
}
```

The `profile_id` path parameter is authoritative. The body must not include a
profile ID.

`DELETE /profiles/{profile_id}` returns a small success payload:

```json
{
  "ok": true,
  "profile_id": "workspace-researcher",
  "store": {"kind": "sqlite", "persistent": true}
}
```

Expected HTTP status mapping:

- `200`: successful read or write.
- `404`: profile not found.
- `409`: duplicate profile ID, current default protection, or assignment
  protection.
- `422`: malformed request body or unsupported patch field.
- `503`: required profile or assignment store unavailable.

New reason codes should extend Stage 4K's deterministic mapping:

- `profile_is_default`
- `profile_has_assignments`
- `invalid_profile_patch`

Existing reason codes should be reused where they already fit, including
`profile_not_found`, `profile_already_exists`, `invalid_profile_request`,
`profile_store_unavailable`, and `assignment_store_unavailable`.

## CLI Contract

Add these commands to `mcp-unified-gateway`:

- `create-profile --profile-file <path|->`
- `patch-profile <profile_id> --patch-file <path|->`
- `delete-profile <profile_id>`

`-` means stdin for `--profile-file` and `--patch-file`.

All commands emit one JSON object. Success payloads include store metadata, as
in Stage 4K:

```json
{"kind": "sqlite", "persistent": true}
{"kind": "memory", "persistent": false}
```

Mutating commands remain persistent-store-only. They must reject nonpersistent
memory-store configs unless a future implementation plan explicitly adds and
tests a development-only override. This keeps the CLI consistent with Stage 4K:
read-only memory inspection may be useful in tests, but mutation should not
pretend to persist.

Expected success payloads:

```json
{"ok": true, "profile": {}, "store": {"kind": "sqlite", "persistent": true}}
{"ok": true, "profile": {}, "store": {"kind": "sqlite", "persistent": true}}
{"ok": true, "profile_id": "workspace-researcher", "store": {"kind": "sqlite", "persistent": true}}
```

CLI exit codes remain:

- `0`: success.
- `1`: domain error.
- `2`: argument parsing or request-shape error.

## Validation And Data Flow

Create flow:

1. FastAPI or CLI parses JSON input.
2. The manager validates the input as `MCPProfile`.
3. The manager checks for duplicate profile IDs.
4. The manager normalizes mutation timestamps.
5. The manager persists through `ProfileStore.upsert_profile()`.
6. The manager writes a compact audit event.

Patch flow:

1. FastAPI or CLI parses a constrained patch request.
2. The manager loads the existing profile.
3. The manager rejects unsupported fields.
4. The manager rejects `enabled=false` for the current gateway default profile.
5. The manager applies provided field replacements only.
6. The manager validates the resulting `MCPProfile`.
7. The manager persists and audits.

Delete flow:

1. The manager loads the profile.
2. The manager checks whether it is the current gateway default.
3. The manager calls `ProfileAssignmentStore.list_assignments(profile_id=...)`.
4. If any assignment exists, deletion is rejected.
5. Otherwise, the manager calls `ProfileStore.delete_profile()`.
6. The manager writes a compact audit event.

## Audit Behavior

Audit remains best effort and compact. Success and expected failure events should
include operation type, profile ID, result, reason code when applicable, and a
small set of changed field names. They should not include full profile policy
documents, resource constraints, grants, approval policies, or credential-like
material.

Expected failure audit examples:

- Duplicate ID during create.
- Unsupported patch field.
- Default-profile disable protection.
- Default-profile delete protection.
- Assignment delete protection.

Unexpected audit-store failures should be logged but should not make successful
profile mutations fail, preserving the Stage 4K audit posture.

## Testing Plan

Manager tests should cover:

- Creating a valid profile and auditing success.
- Rejecting duplicate profile IDs.
- Patching allowed fields and preserving omitted fields.
- Replacing `metadata` and individual policy fields rather than merging them.
- Rejecting unsupported patch fields.
- Rejecting `enabled=false` when the profile is the current gateway default.
- Deleting a non-default unassigned profile.
- Rejecting deletion of the current default profile.
- Rejecting deletion of a profile with assignments.
- Compact audit events for expected failures.

FastAPI tests should cover:

- `POST /profiles` success and duplicate failure.
- `PATCH /profiles/{profile_id}` success.
- Blocked default disable through `PATCH /profiles/{profile_id}`.
- `DELETE /profiles/{profile_id}` success.
- Blocked delete for default and assigned profiles.
- Deterministic reason-code to HTTP-status mapping.

CLI tests should cover:

- `create-profile --profile-file`.
- `create-profile --profile-file -`.
- `patch-profile <profile_id> --patch-file`.
- `patch-profile <profile_id> --patch-file -`.
- `delete-profile <profile_id>`.
- Guarded delete and default-disable domain errors.
- Persistent-store requirement for all mutating commands.

## Implementation Notes

Implementation should stay close to the Stage 4K patterns:

- Add request/response models near the existing FastAPI management models.
- Extend `_PROFILE_MANAGEMENT_STATUS_CODES` rather than introducing a second
  error mapper.
- Keep CLI JSON rendering and domain-error rendering consistent with current
  profile commands.
- Prefer manager-level helper methods for default detection and assignment
  checks so FastAPI and CLI do not duplicate safety rules.
- Preserve copy isolation when returning profile documents.
- Avoid storage-layer migrations unless tests reveal a current store contract
  mismatch.

## Acceptance Criteria

- The design is tracked by Backlog task `TASK-573`.
- A reviewed Stage 4L implementation plan can be written from this spec without
  reopening CRUD scope questions.
- The planned implementation keeps all profile mutations behind
  `GatewayProfileManager`.
- The planned implementation does not include assignment CRUD, approval policy
  editing, path scope editing, external server grants, credential grants, or UI
  changes.
