# MCP Gateway Admin Auth And Credential Grants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a host-neutral standalone gateway admin auth seam plus credential-grant metadata management over CLI and FastAPI.

**Architecture:** Keep admin security package-owned and optional: management routes remain opt-in, and admin auth is injected into profile, external-server, runtime, and credential-grant routes without changing JSON-RPC request handling. Standalone config may enable admin auth and name the header/env var, but it must not persist the admin key itself. Credential grants are metadata only, backed by the existing `CredentialGrantStore` protocol and `SQLiteMCPStore`; the manager rejects secret-looking metadata/provenance and never persists credential values.

**Tech Stack:** Python, FastAPI dependencies, Pydantic models, existing `mcp_unified` storage protocols, existing SQLite store, pytest, Bandit.

**Backlog Task:** `TASK-591`

---

## File Structure

- Create `mcp_unified/gateway/admin_auth.py`: admin auth config, verifier protocol, FastAPI dependency helpers, and sanitized error payloads.
- Create `mcp_unified/gateway/credential_grants.py`: credential-grant manager, validation, secret-key detection, audit events, and JSON-safe payloads.
- Modify `mcp_unified/interfaces/storage.py`: add atomic credential-grant create capability and duplicate-id domain error.
- Modify `mcp_unified/storage/sqlite.py`: implement atomic credential-grant create with the existing async DB offload helper.
- Modify `mcp_unified/gateway/config.py`: add standalone admin auth bootstrap config, validate config loading, and expose credential-grant manager construction from resolved storage bundles.
- Modify `mcp_unified/gateway/fastapi.py`: apply optional admin dependencies to management routes and mount credential-grant routes.
- Modify `mcp_unified/gateway/cli.py`: add credential-grant CRUD commands using the persistent gateway store.
- Modify `mcp_unified/gateway/__init__.py`: export public admin/credential-grant helpers if local package exports follow that pattern.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_admin_auth.py`: route gating and JSON-RPC non-interference.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_credential_grants.py`: manager validation, secret rejection, audit behavior, and persistence semantics.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`: credential-grant HTTP routes and auth behavior.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`: credential-grant CLI commands.

## Acceptance Criteria

- Management routes can require standalone admin auth without importing `tldw_Server_API`.
- Standalone gateway config can enable admin auth, customize the header, and resolve the admin key from an environment variable without storing the key in the config file.
- JSON-RPC `/request` and `/ws` routes remain usable under their existing profile/runtime flow and are not accidentally gated by admin auth.
- Credential grants support list/show/create/patch/delete through manager, FastAPI, and CLI.
- Credential-grant create rejects duplicate ids atomically instead of silently replacing an existing grant.
- Credential grants persist only broker references, slots, scopes, metadata, and provenance; secret-looking values are rejected before persistence.
- External-server delete guards continue to block deletion when enabled grants reference the server.
- Focused tests and Bandit on touched package files are recorded.

### Task 1: Add Admin Auth Failing Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_admin_auth.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Write route-gating tests**

Build a gateway app with profile management enabled and admin auth configured. Assert `GET /mcp/profiles` returns unauthorized without the configured header and succeeds with it.

- [ ] **Step 2: Write non-interference test**

Use the same app and assert `GET /mcp/status` and JSON-RPC `/mcp/request` do not require the admin header.

- [ ] **Step 3: Write standalone config tests**

Add JSON and TOML config-loader tests that verify `admin_auth.enabled`, `admin_auth.header_name`, and `admin_auth.api_key_env_var` validate successfully. Also assert malformed config rejects blank header names and does not accept a plaintext `api_key` field.

- [ ] **Step 4: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_admin_auth.py -v
```

Expected: fail because `mcp_unified.gateway.admin_auth` and router auth wiring do not exist yet.

### Task 2: Implement Admin Auth Seam

**Files:**
- Create: `mcp_unified/gateway/admin_auth.py`
- Modify: `mcp_unified/gateway/fastapi.py`

- [ ] **Step 1: Add package auth config**

Implement a small config surface:

```python
@dataclass(frozen=True, slots=True)
class GatewayAdminAuthConfig:
    enabled: bool = False
    header_name: str = "X-MCP-Gateway-Admin-Key"
    api_key: str | None = None
    verifier: GatewayAdminVerifier | None = None
```

Use `secrets.compare_digest` for static API-key comparison. Validate that `enabled=True` has either `api_key` or `verifier`.

Add a separate bootstrap/config model:

```python
@dataclass(frozen=True, slots=True)
class GatewayAdminAuthBootstrapConfig:
    enabled: bool = False
    header_name: str = "X-MCP-Gateway-Admin-Key"
    api_key_env_var: str = "MCP_UNIFIED_GATEWAY_ADMIN_KEY"
```

Reject plaintext `api_key` in file-backed standalone config. Config loading should only name the env var; app construction resolves the actual secret from `os.environ`.

- [ ] **Step 2: Add FastAPI dependency factory**

Expose a helper that returns FastAPI route dependencies. If auth is disabled, return an empty list. If enabled, read the configured header, validate it, and return direct JSON error responses with stable `reason_code` values such as `admin_auth_required` and `admin_auth_invalid`. Do not rely on a default `HTTPException` shape that wraps the payload under `detail`.

- [ ] **Step 3: Wire only management routes**

Update route mounting helpers in `mcp_unified/gateway/fastapi.py` to accept `admin_dependencies` or `admin_auth`. Apply those dependencies to profile management, external registry management, external runtime management, and credential-grant management routes only. Update `create_gateway_app()` and `bootstrap_profile_gateway_from_config()` so standalone config can enable admin auth without host-app glue code.

- [ ] **Step 4: Run admin auth tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_admin_auth.py -v
```

Expected: pass.

### Task 3: Add Credential-Grant Manager Failing Tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_credential_grants.py`

- [ ] **Step 1: Write manager CRUD tests**

Use in-memory stores matching the existing external registry tests. Assert list/show/create/patch/delete return deterministic envelopes with `ok`, `grant`, `grants`, and `store`.

- [ ] **Step 2: Write secret rejection tests**

Assert create/patch rejects metadata or provenance containing keys such as `secret`, `token`, `password`, `api_key`, `authorization`, `headers`, `env`, or `credential_value`.

- [ ] **Step 3: Write reference validation tests**

When optional profile and external-server stores are provided, assert missing `profile_id` or `external_server_id` returns expected domain errors.

- [ ] **Step 4: Write duplicate create tests**

Assert creating a grant with an existing id returns `credential_grant_already_exists` and does not replace the existing grant.

- [ ] **Step 5: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_credential_grants.py -v
```

Expected: fail because the manager module does not exist yet.

### Task 4: Implement Credential-Grant Manager

**Files:**
- Create: `mcp_unified/gateway/credential_grants.py`
- Modify: `mcp_unified/interfaces/storage.py`
- Modify: `mcp_unified/storage/sqlite.py`
- Modify: `mcp_unified/gateway/config.py`

- [ ] **Step 1: Add domain error and metadata class**

Mirror existing gateway manager style with `GatewayCredentialGrantManagementError.to_payload()` and existing store metadata shape.

- [ ] **Step 2: Add atomic create store capability**

Add `CredentialGrantAlreadyExistsError` and `CredentialGrantStore.create_grant(...)` to `mcp_unified/interfaces/storage.py`. Implement it in `SQLiteMCPStore` with an insert that rejects duplicate ids, mirroring the profile/external-server create pattern.

- [ ] **Step 3: Implement validation**

Normalize required text fields: `id`, `profile_id`, `broker_id`, and `credential_slot`. Allow patch fields: `broker_id`, `credential_slot`, `external_server_id`, `scopes`, `metadata`, `provenance`, and `enabled`.

- [ ] **Step 4: Add secret scanning**

Reject secret-looking metadata/provenance keys recursively. Keep this helper reusable for the snapshot slice.

- [ ] **Step 5: Add CRUD methods**

Implement:

```python
async def list_grants(...)
async def show_grant(grant_id: str)
async def create_grant(grant_document: CredentialGrant | Mapping[str, Any])
async def patch_grant(grant_id: str, patch_document: Mapping[str, Any])
async def delete_grant(grant_id: str)
```

Use caller-owned model copies and append best-effort audit events.

- [ ] **Step 6: Add config builder**

Add a `credential_grant_manager_from_storage(...)` helper. It should accept explicit stores and, when convenient, a profile storage bundle plus external registry storage bundle so profile reference validation is possible without duplicating store construction.

- [ ] **Step 7: Run manager tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_credential_grants.py -v
```

Expected: pass.

### Task 5: Add FastAPI Credential-Grant Routes

**Files:**
- Modify: `mcp_unified/gateway/fastapi.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

- [ ] **Step 1: Write HTTP route tests**

Add tests for `GET /mcp/credential-grants`, `POST /mcp/credential-grants`, `GET/PATCH/DELETE /mcp/credential-grants/{grant_id}`, including auth behavior when admin auth is enabled.

- [ ] **Step 2: Add request/response models**

Add Pydantic models for create/patch/list/detail/delete responses without accepting extra secret fields.

- [ ] **Step 3: Mount routes**

Extend `create_gateway_router()` and `create_gateway_app()` with optional `credential_grant_manager` and `enable_credential_grant_management` arguments. Keep explicit injection as the default; only derive from bootstrap if the bootstrap already carries the manager by this point.

- [ ] **Step 4: Preserve error response shape**

Map credential-grant errors to direct JSON response payloads with stable status codes. Cover unauthorized, duplicate create, missing profile, missing external server, invalid patch, and store-unavailable cases.

- [ ] **Step 5: Run FastAPI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -v
```

Expected: pass.

### Task 6: Add CLI Credential-Grant Commands

**Files:**
- Modify: `mcp_unified/gateway/cli.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py`

- [ ] **Step 1: Write CLI tests**

Cover:

- `list-credential-grants`
- `show-credential-grant <grant_id>`
- `create-credential-grant --grant-file <path-or-dash>`
- `patch-credential-grant <grant_id> --patch-file <path-or-dash>`
- `delete-credential-grant <grant_id>`

- [ ] **Step 2: Implement CLI handlers**

Follow existing JSON output style and persistent-store checks from external-server commands.

- [ ] **Step 3: Run CLI tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -v
```

Expected: pass.

### Task 7: Final Verification

**Files:**
- Modified files from prior tasks.
- Backlog task `TASK-591`.

- [ ] **Step 1: Run focused package tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_admin_auth.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_credential_grants.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  -v
```

- [ ] **Step 2: Run Bandit on touched package files**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  mcp_unified/interfaces/storage.py \
  mcp_unified/storage/sqlite.py \
  mcp_unified/gateway/admin_auth.py \
  mcp_unified/gateway/credential_grants.py \
  mcp_unified/gateway/config.py \
  mcp_unified/gateway/fastapi.py \
  mcp_unified/gateway/cli.py \
  -f json -o /tmp/bandit_mcp_gateway_admin_auth_credential_grants.json
```

- [ ] **Step 3: Update Backlog**

Record touched files, verification commands, known skips, and final summary in `TASK-591`.
