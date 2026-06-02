# MCP Gateway Config Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add versioned standalone gateway config import/export snapshots for profiles, default assignment, external servers, and credential-grant metadata.

**Architecture:** Introduce a package-owned snapshot model and manager that reads/writes through existing store protocols. Export produces secret-safe JSON snapshots by validating grant metadata/provenance plus external server command and URL fields for inline secret material. Import validates all references before the first write, supports dry-run planning, and applies changes in dependency order without destructive replace semantics; arbitrary injected stores are validate-first best-effort rather than transactionally atomic.

**Tech Stack:** Python, Pydantic, existing `mcp_unified` store protocols, existing SQLite store, CLI JSON workflows, pytest, Bandit.

**Backlog Task:** `TASK-592`

**Depends On:** `TASK-591`

---

## File Structure

- Create `mcp_unified/gateway/snapshots.py`: snapshot Pydantic models, validation, export/import manager, dry-run mutation plan.
- Modify `mcp_unified/gateway/config.py`: build snapshot manager from existing profile and external registry storage bundles.
- Modify `mcp_unified/gateway/cli.py`: add `export-config` and `import-config` commands.
- Reuse `mcp_unified/gateway/credential_grants.py`: share the secret-key detection helper from `TASK-591` rather than duplicating token lists.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_config_snapshots.py`: manager export/import/dry-run behavior.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`: CLI import/export behavior.

## Acceptance Criteria

- Exported snapshots include schema version, profiles, default assignment, external servers, and credential grants.
- Snapshot output contains no plaintext secrets and rejects secret-looking metadata/provenance, command arguments, URL userinfo, and sensitive URL query keys.
- Import dry-run validates references and reports planned mutations without writing.
- Import applies in safe order: profiles, default assignment, external servers, credential grants.
- Import validates the whole snapshot before the first write and reports partial write failures explicitly; arbitrary injected stores are not promised transactional rollback.
- Import defaults to upsert semantics and does not delete missing local records.
- A snapshot exported from one SQLite store can be imported into a fresh SQLite store and exported again with equivalent semantic content.

### Task 1: Add Snapshot Model Failing Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_config_snapshots.py`

- [ ] **Step 1: Write export shape test**

Create a SQLite-backed gateway store with one profile, default assignment, external server, and credential grant. Assert export returns a deterministic object with:

```json
{
  "schema": "mcp_unified.gateway.config_snapshot",
  "version": 1,
  "profiles": [],
  "default_assignment": null,
  "external_servers": [],
  "credential_grants": []
}
```

- [ ] **Step 2: Write dry-run test**

Assert `import_snapshot(..., dry_run=True)` returns a mutation plan and leaves target stores unchanged.

- [ ] **Step 3: Write secret-safety test**

Assert snapshot validation rejects metadata/provenance keys containing `secret`, `token`, `password`, `api_key`, `authorization`, `headers`, `env`, or `credential_value`.

- [ ] **Step 4: Write external-server inline secret tests**

Assert export/import validation rejects external server `command` args containing inline values such as `--token=abc`, `api_key=abc`, or `PASSWORD=abc`, URL userinfo such as `https://user:pass@example.test`, and URL query keys such as `?token=abc`. `env_allowlist` may still contain environment variable names because it does not contain values.

- [ ] **Step 5: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_config_snapshots.py -v
```

Expected: fail because `mcp_unified.gateway.snapshots` does not exist.

### Task 2: Implement Snapshot Models And Export

**Files:**
- Create: `mcp_unified/gateway/snapshots.py`

- [ ] **Step 1: Add snapshot models**

Define models:

```python
class GatewayConfigSnapshot(BaseModel):
    schema: Literal["mcp_unified.gateway.config_snapshot"]
    version: Literal[1]
    profiles: list[MCPProfile]
    default_assignment: ProfileAssignment | None = None
    external_servers: list[ExternalServerDefinition]
    credential_grants: list[CredentialGrant]
```

Use model validators for deterministic ordering and secret-key rejection. Reuse the secret-key helper from `mcp_unified.gateway.credential_grants` and add snapshot-specific validators for external server command and URL fields.

- [ ] **Step 2: Add manager dependencies**

Create `GatewayConfigSnapshotManager` with required stores: `profile_store`, `assignment_store`, `external_registry_store`, and `credential_grant_store`. Optionally accept `audit_store`.

- [ ] **Step 3: Implement export**

Read all records through protocol methods, load only the default assignment from `GATEWAY_DEFAULT_ASSIGNMENT_ID`, sort lists by id, validate the full snapshot for secret safety, and return JSON-safe model dumps.

- [ ] **Step 4: Run export tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_config_snapshots.py::test_export_snapshot_includes_expected_sections -v
```

Expected: pass.

### Task 3: Implement Import And Dry-Run

**Files:**
- Modify: `mcp_unified/gateway/snapshots.py`

- [ ] **Step 1: Add mutation plan payload**

Return planned actions such as `upsert_profile`, `set_default_assignment`, `upsert_external_server`, and `upsert_credential_grant` with target ids and counts. Include enough detail for failures to identify which action failed without echoing raw snapshot payloads.

- [ ] **Step 2: Add reference validation**

Validate:

- default assignment profile exists in incoming snapshot or current store.
- every grant profile exists in incoming snapshot or current store.
- every grant external server exists in incoming snapshot or current store when `external_server_id` is set.
- every referenced credential slot is present in the referenced external server definition when that server is known.
- all secret-safety checks pass before any write begins.

- [ ] **Step 3: Define atomicity behavior**

Document in code comments and docs that the generic manager is validate-first best-effort: it validates the full snapshot before writing, then reports applied and failed action ids if an injected store fails during writes. Do not claim transactionality for arbitrary stores. If all stores are the same `SQLiteMCPStore` and a small transaction helper already exists, it may be used, but it is not required for this slice.

- [ ] **Step 4: Apply import in safe order**

Upsert profiles first, then default assignment, then external servers, then credential grants. Do not delete local records absent from the snapshot.

- [ ] **Step 5: Add audit events**

Append best-effort audit events for import start/completion/failure when an audit store is available. Do not include snapshot raw payload in audit metadata.

- [ ] **Step 6: Run import tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_config_snapshots.py -v
```

Expected: pass.

### Task 4: Add Config Builder And CLI Commands

**Files:**
- Modify: `mcp_unified/gateway/config.py`
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [ ] **Step 1: Add builder helper**

Add `gateway_config_snapshot_manager_from_storage(...)` or equivalent helper that builds the manager from existing profile/external storage bundles.

- [ ] **Step 2: Write CLI tests**

Cover:

- `export-config --config <path>` writes JSON to stdout.
- `export-config --config <path> --output <file>` writes a file.
- `import-config --config <path> --snapshot-file <file> --dry-run` reports planned actions without mutation.
- `import-config --config <path> --snapshot-file <file>` applies upserts.
- import failure after validation reports applied and failed action ids without raw secret-bearing payloads.

- [ ] **Step 3: Implement CLI handlers**

Use existing `_load_json_argument_file`, `_emit_json`, and config path helpers. Reject memory stores because snapshots are meant for persistent standalone configuration.

- [ ] **Step 4: Run CLI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -v
```

Expected: pass.

### Task 5: Round-Trip Verification

**Files:**
- Modified files from prior tasks.
- Backlog task `TASK-592`.

- [ ] **Step 1: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_config_snapshots.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  -v
```

- [ ] **Step 2: Run Bandit**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  mcp_unified/gateway/snapshots.py \
  mcp_unified/gateway/config.py \
  mcp_unified/gateway/cli.py \
  -f json -o /tmp/bandit_mcp_gateway_config_snapshots.json
```

- [ ] **Step 3: Update Backlog**

Record touched files, verification commands, known skips, and final summary in `TASK-592`.
