# Notification Authorization and Recovery UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make personal notifications authorized, recoverable, principal-scoped, and accessible in both WebUI and extension runtimes without terminal retry loops or automatic mutation replay.

**Architecture:** Add notification permissions and canonical role membership at the AuthNZ layer, then share one pure lifecycle classifier across two runtime adapters. The extension adapter persists server/principal-scoped state in safe storage; the WebUI adapter owns state in an authenticated provider consumed by its poll, toast bridge, inbox, and shared header.

**Tech Stack:** FastAPI/AuthNZ, SQLite, PostgreSQL/asyncpg, TypeScript, React, Next.js, WXT, Vitest, Playwright, pytest, Bandit.

**Spec:** `Docs/superpowers/specs/2026-07-10-chatbooks-residual-uat-remediation-design.md`

**Backlog:** `TASK-12098.4`; final browser dependency recorded in `TASK-12098.3`.

---

## Stage Overview

| Stage | Goal | Success Criteria | Status |
| --- | --- | --- | --- |
| 1 | Canonical authorization data | SQLite 090 and PostgreSQL fresh/backfill paths grant permissions and backfill effective role membership | In Progress |
| 2 | Shared lifecycle policy | Both transports preserve status; reads/SSE retry safely; mutations never replay automatically | Not Started |
| 3 | Runtime adapters | Extension and WebUI expose truthful principal-scoped lifecycle state | Not Started |
| 4 | Accessible recovery UX | Header and inbox distinguish active, degraded, auth-required, and unavailable states | Not Started |
| 5 | Acceptance and security | Focused browser tests, Bandit, typecheck, and UAT role assertions pass | Not Started |

## File Responsibilities

### Backend

- `tldw_Server_API/app/core/AuthNZ/migrations.py`: SQLite migration 090, permission grants, and legacy `users.role` to `user_roles` backfill.
- `tldw_Server_API/app/core/AuthNZ/rbac_seed.py`: fresh-install permission catalog and built-in role grants for both database backends.
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`: idempotent PostgreSQL permission and legacy membership backfill.
- `tldw_Server_API/app/core/AuthNZ/initialize.py`: invoke PostgreSQL backfill after baseline roles exist.
- `tldw_Server_API/app/services/registration_service.py`: write canonical role membership in the registration transaction.
- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`: remove request-time legacy role inference only after migrations and registration are covered.

### Shared Frontend

- Create `apps/packages/ui/src/services/notification-lifecycle.ts`: status normalization, state transitions, retry calculation, and scope-key construction.
- `apps/packages/ui/src/services/notifications.ts`: wire stream open/error callbacks and bounded reconnect policy to the lifecycle module.
- `apps/packages/ui/src/entries/shared/notification-subscription.ts`: extension adapter and scoped storage.
- `apps/packages/ui/src/components/Layouts/ChatHeader.tsx`: notification status button, accessible naming, badge semantics, and popover.
- `apps/packages/ui/src/components/Layouts/Header.tsx`: pass lifecycle props to `ChatHeader`.

### WebUI

- `apps/tldw-frontend/lib/sse.ts`: throw structured `ApiError` with HTTP status.
- `apps/tldw-frontend/lib/api/notifications.ts`: adapt WebUI SSE to the shared stream lifecycle.
- Create `apps/tldw-frontend/components/notifications/NotificationLifecycleProvider.tsx`: authenticated WebUI adapter/context.
- `apps/tldw-frontend/components/layout/WebLayout.tsx`: lifecycle owner for unread polling and header props.
- `apps/tldw-frontend/components/notifications/NotificationToastBridge.tsx`: terminal bootstrap suppression and stream state.
- `apps/tldw-frontend/pages/notifications.tsx`: page status, disabled actions, explicit mutation retry, and permission retry.

## Stage 0: Preflight

- [ ] Confirm the implementation branch contains current `origin/dev` and record both SHAs in TASK-12098.4:

```bash
cd "$(git rev-parse --show-toplevel)"
git fetch origin
git merge-base --is-ancestor origin/dev HEAD
git rev-parse origin/dev HEAD
git status --short --branch
```

Expected: ancestor check exits 0; only documented unrelated files may be untracked.

- [ ] Confirm migration 089 is still latest immediately before editing; if not, renumber every migration-090 reference in this plan before proceeding:

```bash
cd "$(git rev-parse --show-toplevel)"
tail -n 190 tldw_Server_API/app/core/AuthNZ/migrations.py
```

## Task 1: Seed Permissions and Backfill Existing Role Membership

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/rbac_seed.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/initialize.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_rbac_seed_helper.py`
- Create: `tldw_Server_API/tests/AuthNZ/integration/test_notification_permissions_postgres.py`

- [x] **Step 1: Write failing SQLite migration tests**

Add tests that migrate a version-089 database and assert:

```python
assert permission_names >= {"notifications.read", "notifications.control"}
assert role_permissions["user"] >= {"notifications.read", "notifications.control"}
assert role_permissions["admin"] >= {"notifications.read", "notifications.control"}
assert explicit_denies == {"notifications.control"}
assert user_role_rows == {(legacy_user_id, expected_role_id)}
```

Cover `moderator`, `reviewer`, and `viewer` only when the role exists. Add a user whose legacy `users.role` matches an existing custom role and prove the matching role is backfilled without changing custom grants.

- [x] **Step 2: Run the SQLite tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_rbac_seed_helper.py -v
```

Expected: FAIL because migration 090 and notification seed entries do not exist.

- [x] **Step 3: Implement SQLite migration 090 and fresh seed entries**

Use parameterized SQL and `INSERT OR IGNORE`. The migration must:

```python
permissions = (
    ("notifications.read", "Read personal notifications", "notifications"),
    ("notifications.control", "Manage personal notifications", "notifications"),
)
interactive_roles = ("admin", "user", "moderator", "reviewer", "viewer")
```

Backfill `user_roles` by joining `users.role` to an existing `roles.name`; never create a role from arbitrary legacy text. Do not modify `user_permissions`.

- [x] **Step 4: Write failing PostgreSQL fresh-seed and post-role backfill tests**

Test both sequences:

1. fresh initialization where roles are seeded after core tables; and
2. an existing installation with roles/users present but notification permissions or `user_roles` missing.

Re-run the new file and expect failure before implementation:

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/integration/test_notification_permissions_postgres.py -v
```

- [x] **Step 5: Implement PostgreSQL seed ordering and backfill**

Add notification permissions to `rbac_seed.py`. Add an idempotent helper in `pg_migrations_extra.py` that runs after baseline role seeding from `initialize.py`, grants every present interactive system role, and backfills matching legacy role membership. Keep the existing pre-role ensure harmless, but do not rely on it for grants.

- [x] **Step 6: Run backend authorization tests GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_rbac_seed_helper.py \
  tldw_Server_API/tests/AuthNZ/integration/test_notification_permissions_postgres.py -v
```

Expected: PASS, including idempotent second execution and explicit-deny preservation.

- [x] **Step 7: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
git add tldw_Server_API/app/core/AuthNZ/migrations.py tldw_Server_API/app/core/AuthNZ/rbac_seed.py tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py tldw_Server_API/app/core/AuthNZ/initialize.py tldw_Server_API/tests/AuthNZ/unit/test_rbac_seed_helper.py tldw_Server_API/tests/AuthNZ/integration/test_notification_permissions_postgres.py
git commit -m "fix(authnz): seed notification permissions and roles"
```

## Task 2: Make Registration Write Canonical Role Membership

**Files:**
- Modify: `tldw_Server_API/app/services/registration_service.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_registration_service_backend_selection.py`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_registration_default_role_membership.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py`

- [ ] **Step 1: Write failing registration membership tests**

Cover default registration, registration-code-selected role, rollback on unknown role, and both database backends. Assert the created user has a `user_roles` row before registration returns.

- [ ] **Step 2: Verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_registration_service_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_registration_default_role_membership.py -v
```

Expected: FAIL because current registration writes the legacy role field without canonical membership.

- [ ] **Step 3: Insert role membership inside the registration transaction**

Resolve the selected role by name and insert the membership before commit. Unknown role names fail registration with a descriptive error; they must not silently fall back.

- [ ] **Step 4: Remove request-time role inference and assert the UAT fixture**

Delete the `users.role` authorization fallback from `User_DB_Handling.py`. Update the Chatbooks UAT fixture test to query effective roles and permissions before server launch.

- [ ] **Step 5: Run focused tests GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_registration_service_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_registration_default_role_membership.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py -v
```

- [ ] **Step 6: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
git add tldw_Server_API/app/services/registration_service.py tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py tldw_Server_API/tests/AuthNZ/unit/test_registration_service_backend_selection.py tldw_Server_API/tests/AuthNZ/unit/test_registration_default_role_membership.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py
git commit -m "fix(authnz): persist registration role membership"
```

## Task 3: Implement the Shared Notification Lifecycle

**Files:**
- Create: `apps/packages/ui/src/services/notification-lifecycle.ts`
- Create: `apps/packages/ui/src/services/__tests__/notification-lifecycle.test.ts`
- Modify: `apps/packages/ui/src/services/notifications.ts`
- Modify: `apps/packages/ui/src/services/__tests__/notifications.test.ts`
- Modify: `apps/tldw-frontend/lib/sse.ts`
- Modify: `apps/tldw-frontend/lib/api/notifications.ts`
- Modify: `apps/tldw-frontend/lib/__tests__/notifications.test.ts`

- [ ] **Step 1: Write lifecycle and retry tests**

Define the public contract in tests:

```ts
export type NotificationLifecycleState =
  | "idle"
  | "connecting"
  | "active"
  | "degraded"
  | "auth-required"
  | "unavailable"

expect(classifyNotificationError({ status: 401 })).toEqual({ kind: "auth-required" })
expect(classifyNotificationError({ statusCode: 403 })).toEqual({ kind: "unavailable" })
expect(classifyNotificationError({ status: 503, retryAfter: 40 })).toMatchObject({ kind: "retry", delayMs: 40_000 })
```

Cover 408/425/429/5xx/network, other 4xx, abort, backoff cap, jitter injection, scope keys, and cursor retention.

- [ ] **Step 2: Run shared tests RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx vitest run -c vitest.extension.config.ts \
  ../packages/ui/src/services/__tests__/notification-lifecycle.test.ts \
  ../packages/ui/src/services/__tests__/notifications.test.ts
```

- [ ] **Step 3: Implement the pure lifecycle module**

Export only pure functions and types: `readHttpStatus`, `classifyNotificationError`, `nextReconnectDelay`, `reduceNotificationLifecycle`, and `buildNotificationScopeKey`. Reuse the normalized server/principal pattern from `chat-surface-scope.ts`.

- [ ] **Step 4: Preserve direct-WebUI SSE status and expose stream-open**

Make `openSSEStream` throw `ApiError` with `status`. Extend `NotificationStreamReader` with an `onOpen` callback fired after a successful response/body is acquired. Never infer active state from creation of an unsubscribe handle.

- [ ] **Step 5: Prove no automatic mutation replay**

Tests must show stream/read retries occur, while `markNotificationsRead`, dismiss, snooze, cancel-snooze, preference updates, and future mutations make one request per explicit call.

- [ ] **Step 6: Run shared and WebUI API tests GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx vitest run \
  lib/__tests__/notifications.test.ts
bunx vitest run -c vitest.extension.config.ts \
  ../packages/ui/src/services/__tests__/notification-lifecycle.test.ts \
  ../packages/ui/src/services/__tests__/notifications.test.ts
```

- [ ] **Step 7: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
git add apps/packages/ui/src/services/notification-lifecycle.ts apps/packages/ui/src/services/__tests__/notification-lifecycle.test.ts apps/packages/ui/src/services/notifications.ts apps/packages/ui/src/services/__tests__/notifications.test.ts apps/tldw-frontend/lib/sse.ts apps/tldw-frontend/lib/api/notifications.ts apps/tldw-frontend/lib/__tests__/notifications.test.ts
git commit -m "fix(notifications): classify terminal and transient failures"
```

## Task 4: Implement Principal-Scoped Runtime Adapters

**Files:**
- Modify: `apps/packages/ui/src/entries/shared/notification-subscription.ts`
- Modify: `apps/packages/ui/src/entries/shared/__tests__/notification-subscription.test.ts`
- Modify: `apps/packages/ui/src/hooks/useNotificationCount.ts`
- Modify: `apps/packages/ui/src/hooks/__tests__/useNotificationCount.test.tsx`
- Create: `apps/tldw-frontend/components/notifications/NotificationLifecycleProvider.tsx`
- Create: `apps/tldw-frontend/__tests__/components/notification-lifecycle-provider.test.tsx`
- Modify: `apps/tldw-frontend/components/layout/WebLayout.tsx`
- Modify: `apps/tldw-frontend/components/notifications/NotificationToastBridge.tsx`
- Modify: `apps/tldw-frontend/__tests__/components/notification-toast-bridge.test.tsx`

- [ ] **Step 1: Write extension adapter tests RED**

Assert server/principal-scoped keys, synchronous clearing before account switch render, `connecting -> active` only on `onOpen`, 401 stop/restart after auth success, 403 stop until explicit retry, and no passive polling.

- [ ] **Step 2: Run extension adapter tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx vitest run -c vitest.extension.config.ts \
  ../packages/ui/src/entries/shared/__tests__/notification-subscription.test.ts \
  ../packages/ui/src/hooks/__tests__/useNotificationCount.test.tsx
```

Expected: FAIL on missing scoped lifecycle state and restart behavior.

- [ ] **Step 3: Implement the extension adapter**

Store `{state, unreadCount, updatedAt}` under the scoped key. Logout and server/principal changes abort first, clear the rendered selector, then start the new scope. Keep `idle` internal-only.

- [ ] **Step 4: Write WebUI provider tests**

Cover the 30-second layout poll, toast bootstrap, terminal suppression, transient recovery, explicit permission retry, auth recovery, and mutation errors. Use fake timers and assert no repeated 401/403 requests.

- [ ] **Step 5: Run WebUI provider tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx vitest run \
  __tests__/components/notification-lifecycle-provider.test.tsx \
  __tests__/components/notification-toast-bridge.test.tsx \
  __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
```

Expected: FAIL because no WebUI lifecycle provider coordinates these initiators.

- [ ] **Step 6: Implement the WebUI provider and wire consumers**

`WebLayout` owns the provider. The toast bridge and inbox consume it; they must not create independent retry loops. A successful explicit permission refresh or `Try again` may leave unavailable; time alone may not.

- [ ] **Step 7: Run adapter tests GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx vitest run \
  __tests__/components/notification-lifecycle-provider.test.tsx \
  __tests__/components/notification-toast-bridge.test.tsx \
  __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
bunx vitest run -c vitest.extension.config.ts \
  ../packages/ui/src/entries/shared/__tests__/notification-subscription.test.ts \
  ../packages/ui/src/hooks/__tests__/useNotificationCount.test.tsx
```

- [ ] **Step 8: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
git add apps/packages/ui/src/entries/shared/notification-subscription.ts apps/packages/ui/src/entries/shared/__tests__/notification-subscription.test.ts apps/packages/ui/src/hooks/useNotificationCount.ts apps/packages/ui/src/hooks/__tests__/useNotificationCount.test.tsx apps/tldw-frontend/components/notifications/NotificationLifecycleProvider.tsx apps/tldw-frontend/__tests__/components/notification-lifecycle-provider.test.tsx apps/tldw-frontend/components/layout/WebLayout.tsx apps/tldw-frontend/components/notifications/NotificationToastBridge.tsx apps/tldw-frontend/__tests__/components/notification-toast-bridge.test.tsx
git commit -m "fix(notifications): scope lifecycle by server and principal"
```

## Task 5: Build Accessible Header and Inbox Recovery States

**Files:**
- Modify: `apps/packages/ui/src/components/Layouts/ChatHeader.tsx`
- Modify: `apps/packages/ui/src/components/Layouts/Header.tsx`
- Create: `apps/packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx`
- Modify: `apps/tldw-frontend/pages/notifications.tsx`
- Modify: `apps/tldw-frontend/__tests__/pages/notifications.test.tsx`

- [ ] **Step 1: Write failing header accessibility tests**

Assert:

- active label `Notifications, 3 unread` and an `aria-hidden` visual badge;
- no bell in internal idle;
- connecting/degraded/auth-required/unavailable names;
- enabled native status button, not `aria-disabled`;
- `aria-haspopup`, `aria-expanded`, `aria-controls`;
- Enter/Space open, Escape close, and focus return;
- one polite announcement per state transition; and
- explicit `Try again` makes exactly one request.

- [ ] **Step 2: Run header accessibility tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx vitest run -c vitest.extension.config.ts \
  ../packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx
```

Expected: FAIL because the header accepts only a count/callback and has no lifecycle UI.

- [ ] **Step 3: Implement the status button and popover**

Use existing Ant Design/Lucide primitives. Do not add repeated toasts. Use `Bell` for active and the existing Lucide unavailable/status icon where appropriate. Keep copy free of RBAC terminology.

- [ ] **Step 4: Write inbox state tests**

Direct `/notifications` navigation must render auth-required, unavailable, and degraded states. Suppress mutations that cannot succeed. Transient mutation errors remain inline and expose explicit retry.

- [ ] **Step 5: Run inbox tests and verify RED**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx vitest run __tests__/pages/notifications.test.tsx
```

Expected: FAIL on missing lifecycle states and explicit retry behavior.

- [ ] **Step 6: Implement inbox states and explicit retries**

Consume `NotificationLifecycleProvider`; do not introduce another timer or stream owner.

- [ ] **Step 7: Run component tests and typecheck GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx vitest run \
  __tests__/pages/notifications.test.tsx
bunx vitest run -c vitest.extension.config.ts \
  ../packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx
bun run typecheck
```

- [ ] **Step 8: Commit**

```bash
cd "$(git rev-parse --show-toplevel)"
git add apps/packages/ui/src/components/Layouts/ChatHeader.tsx apps/packages/ui/src/components/Layouts/Header.tsx apps/packages/ui/src/components/Layouts/__tests__/ChatHeader.notifications.test.tsx apps/tldw-frontend/pages/notifications.tsx apps/tldw-frontend/__tests__/pages/notifications.test.tsx
git commit -m "fix(webui): expose notification recovery states"
```

## Task 6: Add Browser Acceptance Coverage

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/tier-4-admin/notifications.spec.ts`
- Modify: `tldw_Server_API/tests/Notifications/test_notifications_api.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py`

- [ ] **Step 1: Add failing browser scenarios**

Cover standard-user list/count/control/SSE, restricted-role badge suppression and keyboard popover, one explicit retry, 401 sign-in recovery, later role grant plus `Try again`, account switch, and zero terminal request loops.

- [ ] **Step 2: Add backend API assertions**

Assert a standard user can call all notification endpoints and a restricted custom role remains 403. Include explicit-deny precedence.

- [ ] **Step 3: Run the new acceptance tests and verify RED**

Run both command blocks below before making integration fixes. Expected: at least one new scenario fails on missing final browser/UAT wiring; if all pass, document that earlier tasks already satisfied the acceptance test and do not manufacture a failure.

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Notifications/test_notifications_api.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py -v
```

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx playwright test e2e/workflows/tier-4-admin/notifications.spec.ts --project=tier-4 --reporter=line
```

- [ ] **Step 4: Implement only acceptance wiring gaps**

Keep production behavior in the modules from Tasks 1-5. This step may update fixtures, auth setup, and browser selectors, but must not duplicate lifecycle logic inside the E2E test.

- [ ] **Step 5: Run focused backend and browser tests GREEN**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Notifications/test_notifications_api.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py -v
```

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bunx playwright test e2e/workflows/tier-4-admin/notifications.spec.ts --project=tier-4 --reporter=line
```

Expected: all required scenarios pass with no skips.

- [ ] **Step 6: Run security and final notification gates**

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/rbac_seed.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/services/registration_service.py \
  tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py \
  -f json -o /tmp/bandit_task_12098_4.json
```

```bash
cd "$(git rev-parse --show-toplevel)"
cd apps/tldw-frontend
bun run typecheck
```

```bash
cd "$(git rev-parse --show-toplevel)"
git diff --check
```

- [ ] **Step 7: Update TASK-12098.4 and commit**

Record exact pass counts, Bandit result, browser report path, and any documented skips (the certification scenario itself permits none).

```bash
cd "$(git rev-parse --show-toplevel)"
git add apps/tldw-frontend/e2e/workflows/tier-4-admin/notifications.spec.ts tldw_Server_API/tests/Notifications/test_notifications_api.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py backlog/tasks/task-12098.4\ -\ Fix-notification-permissions-and-terminal-stream-retries.md
git commit -m "test(notifications): cover authorization recovery ux"
```

## Final Verification Checklist

- [ ] SQLite 090 migrates from 089 and is idempotent.
- [ ] PostgreSQL fresh install and existing-install backfill both grant after role seeding.
- [ ] Existing matching legacy roles are backfilled before request-time fallback is removed.
- [ ] Explicit user denies and custom grants are unchanged.
- [ ] WebUI and extension preserve structured 401/403 status.
- [ ] Only idempotent reads/bootstrap/poll/SSE retry automatically.
- [ ] State and unread counts never cross server or principal boundaries.
- [ ] Header and inbox meet keyboard, focus, naming, and live-region requirements.
- [ ] Standard, restricted, reauthentication, and role-grant browser scenarios pass without skips.
- [ ] Typecheck, targeted pytest/Vitest/Playwright, Bandit, and `git diff --check` pass.
