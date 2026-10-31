# Authenticated User Scope Propagation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve already validated organization and team scope on the authenticated `User` returned by JWT and API-key authentication.

**Architecture:** Extend the existing compatibility model rather than changing sharing dependencies. Both authentication paths assign effective scope only after current membership validation and scoped-permission resolution, keeping `User`, `request.state`, and `AuthPrincipal` aligned.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, pytest, PostgreSQL live UAT

---

### Task 1: Add Failing Returned-User Contract Tests

**Files:**
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py`
- Modify: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py`
- Modify: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py`

- [x] Add assertions that the authenticated `request_user` exposes effective organization, team, and active scope.
- [x] Extend the valid JWT fixture with an additional current membership that is not present in the token claims; assert `request_user` remains exactly limited to the claimed organization/team and that both active IDs match `AuthPrincipal` and `request.state`.
- [x] For an organization-restricted API key, assert `request_user.org_ids` is exactly the selected organization, `request_user.team_ids` contains only teams in that organization, and both active IDs match `AuthPrincipal` and `request.state`.
- [x] For a team-restricted API key, assert `request_user.org_ids` and `request_user.team_ids` contain only the selected organization/team, excluding the user's other memberships, and both active IDs match `AuthPrincipal` and `request.state`.
- [x] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py::test_verify_jwt_accepts_valid_membership_claims` and verify it fails with missing `User.org_ids` before implementation.
- [x] Preserve existing stale-membership and invalid API-key scope assertions.

### Task 2: Propagate Effective Scope

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`

- [x] Add scope fields with empty/`None` defaults to `User`.
- [x] Assign JWT scope after `apply_scoped_permissions` succeeds.
- [x] Assign API-key scope after `apply_scoped_permissions` succeeds.
- [x] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py` and expect all tests to pass.

### Task 3: Verify Sharing and Live PostgreSQL UAT

**Files:**
- Verify: `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`

- [x] Run `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py tldw_Server_API/tests/Sharing/test_sharing_endpoints.py` and expect all selected tests to pass (PostgreSQL integration tests may skip only when the canonical fixture reports PostgreSQL unavailable).
- [x] Run `source .venv/bin/activate && python -m ruff check tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py` and expect exit code 0.
- [x] Run `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py -f json -o /tmp/bandit_task_12020_42.json`; confirm the three reported `B106` findings are pre-existing literal token-type false positives outside the modified hunks and that this task adds no findings.
- [x] Restart the isolated backend and repeat owner organization sharing.
- [x] Assert the recipient receives `{items,total}`, the expected workspace title,
      and matching shared-workspace metadata.
- [x] Resume TASK-12020.39 WebUI/CDP validation.

Live restart command (reuse the existing task-owned PostgreSQL container and
runtime files; run from the worktree root):

```bash
source /private/tmp/task-12020-39/runtime.env
AUTH_MODE=multi_user \
DATABASE_URL="postgresql://tldw_uat:${TASK_12020_39_PG_PASSWORD}@127.0.0.1:55439/task1202039" \
JWT_SECRET_KEY="$TASK_12020_39_JWT_SECRET" \
API_KEY_PEPPER="$TASK_12020_39_JWT_SECRET" \
SESSION_ENCRYPTION_KEY="$TASK_12020_39_SESSION_KEY" \
MCP_JWT_SECRET="$TASK_12020_39_MCP_JWT_SECRET" \
MCP_API_KEY_SALT="$TASK_12020_39_MCP_API_KEY_SALT" \
TLDW_ENV_FILE=/private/tmp/task-12020-39/.env \
TLDW_CONFIG_FILE=/private/tmp/task-12020-39/config.txt \
TLDW_CONFIG_DIR=/private/tmp/task-12020-39 \
USER_DB_BASE_DIR=/private/tmp/task-12020-39/user_databases \
JOBS_DB_PATH=/private/tmp/task-12020-39/jobs.db \
SCHEDULER_DATABASE_URL=sqlite:////private/tmp/task-12020-39/scheduler/scheduler.db \
SCHEDULER_BASE_PATH=/private/tmp/task-12020-39/scheduler \
CIRCUIT_BREAKER_REGISTRY_DB_PATH=/private/tmp/task-12020-39/circuit-breakers.db \
WATCHLIST_TEMPLATE_DIR=/private/tmp/task-12020-39/watchlist-templates \
MCP_MODULES_CONFIG=/private/tmp/task-12020-39/mcp-modules.yaml \
ALLOWED_ORIGINS=http://127.0.0.1:18240 \
CORS_ALLOW_CREDENTIALS=true \
CSRF_ENABLED=true \
HOME=/private/tmp/task-12020-39/home \
TMPDIR=/private/tmp/task-12020-39/tmp \
XDG_CACHE_HOME=/private/tmp/task-12020-39/cache \
.venv/bin/python -m uvicorn tldw_Server_API.app.main:app \
  --host 127.0.0.1 --port 18242
```

Run the production endpoint sequence against `http://127.0.0.1:18242`:

```bash
source /private/tmp/task-12020-39/runtime.env

curl -fsS -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode username=owner \
  --data-urlencode "password=$TASK_12020_39_OWNER_PASSWORD" \
  http://127.0.0.1:18242/api/v1/auth/login \
  -o /private/tmp/task-12020-39/owner-login-patched.json

curl -fsS -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode username=member \
  --data-urlencode "password=$TASK_12020_39_MEMBER_PASSWORD" \
  http://127.0.0.1:18242/api/v1/auth/login \
  -o /private/tmp/task-12020-39/member-login-patched.json

OWNER_TOKEN=$(jq -r .access_token /private/tmp/task-12020-39/owner-login-patched.json)
MEMBER_TOKEN=$(jq -r .access_token /private/tmp/task-12020-39/member-login-patched.json)

curl -fsS -H "Authorization: Bearer $OWNER_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"share_scope_type":"org","share_scope_id":1,"access_level":"view_chat","allow_clone":true}' \
  http://127.0.0.1:18242/api/v1/sharing/workspaces/task-12020-39-owner-workspace/share \
  -o /private/tmp/task-12020-39/share-patched.json

curl -fsS -H "Authorization: Bearer $MEMBER_TOKEN" \
  http://127.0.0.1:18242/api/v1/sharing/shared-with-me \
  -o /private/tmp/task-12020-39/shared-with-me-patched.json

SHARE_ID=$(jq -r '.items[] | select(.workspace_id == "task-12020-39-owner-workspace") | .share_id' \
  /private/tmp/task-12020-39/shared-with-me-patched.json)

curl -fsS -H "Authorization: Bearer $MEMBER_TOKEN" \
  "http://127.0.0.1:18242/api/v1/sharing/shared-with-me/$SHARE_ID/workspace" \
  -o /private/tmp/task-12020-39/shared-workspace-patched.json

jq -e --arg id task-12020-39-owner-workspace \
  '(.items | type) == "array" and .total == (.items | length) and .total >= 1 and any(.items[]; .workspace_id == $id and .workspace_name == "TASK-12020.39 Recipient Contract Workspace")' \
  /private/tmp/task-12020-39/shared-with-me-patched.json
jq -e --arg id task-12020-39-owner-workspace '.share.workspace_id == $id' \
  /private/tmp/task-12020-39/shared-workspace-patched.json
```

## Status

- Stage 1: Complete
- Stage 2: Complete
- Stage 3: Complete
