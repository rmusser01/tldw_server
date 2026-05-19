# Admin-UI Full Test Coverage + Documentation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring admin-ui test coverage from ~55% to ~95% and documentation from ~40% to ~90%.

**Architecture:** 4 parallel workstreams — backend pytest tests, frontend vitest tests, Playwright E2E specs, and Markdown documentation. Each workstream is independent (different files/directories). Within each workstream, tasks are ordered by dependency.

**Tech Stack:** pytest (backend), Vitest + RTL (frontend unit), Playwright (E2E), Markdown (docs)

---

## Workstream 1: Backend Unit Tests

### Task 1: admin_ops.py — Realtime Stats + Compliance + Billing

**Files:**
- Test: `tldw_Server_API/tests/Admin/test_admin_ops_new_endpoints.py`
- Reference: `tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py`

**Endpoints to test:**
1. `GET /admin/stats/realtime` — mock ACP session store, verify `active_sessions` + `tokens_today`
2. `GET /admin/compliance/posture` — mock DB pool, verify MFA % + key rotation %
3. `GET /admin/billing/analytics` — mock billing repo, verify MRR + subscriber counts
4. `GET /admin/incidents/metrics/sla` — create test incidents with timestamps, verify MTTA/MTTR math
5. `POST /admin/incidents/{id}/notify` — mock email service, verify per-recipient results + timeline event

**Test pattern:**
- Use `monkeypatch` to inject mock dependencies
- Each endpoint: happy path, auth required (no token → 401), admin required (non-admin → 403), error handling
- Run: `cd /path/to/worktree && python -m pytest tldw_Server_API/tests/Admin/test_admin_ops_new_endpoints.py -v`

**Commit:** `test(backend): add tests for realtime stats, compliance, billing, SLA, notifications`

---

### Task 2: admin_ops.py — Dependencies + Uptime + Email Deliveries

**Files:**
- Test: `tldw_Server_API/tests/Admin/test_admin_ops_dependencies.py`

**Endpoints to test:**
1. `GET /admin/dependencies` — mock health check functions, verify timeout handling, status mapping
2. `GET /admin/dependencies/{name}/uptime` — seed health history, verify uptime % calculation + sparkline
3. `GET /admin/email/deliveries` — seed delivery log, verify pagination + status filter

**Commit:** `test(backend): add tests for dependencies, uptime tracking, email deliveries`

---

### Task 3: admin_ops.py — Webhook CRUD + Report Schedules + Digests

**Files:**
- Test: `tldw_Server_API/tests/Admin/test_admin_ops_webhooks.py`
- Test: `tldw_Server_API/tests/Admin/test_admin_ops_reports.py`

**Endpoints to test (webhooks):**
1. `POST /admin/webhooks` — create, verify secret generated, events validated
2. `GET /admin/webhooks` — list, verify secrets redacted
3. `PATCH /admin/webhooks/{id}` — update URL/events/enabled
4. `DELETE /admin/webhooks/{id}` — delete, verify 404 after
5. `GET /admin/webhooks/{id}/deliveries` — list delivery history
6. `POST /admin/webhooks/{id}/test` — mock HTTP, verify delivery recorded

**Endpoints to test (reports + digests):**
1. Report schedule CRUD (4 endpoints) — frequency/format/recipients validation
2. `POST .../send-now` — mock compliance data + email, verify report generated
3. Digest preference get/set — per-user scoping

**Commit:** `test(backend): add tests for webhook CRUD, report schedules, digest preferences`

---

### Task 4: admin_acp_agents.py — Agent Metrics + Session Budget

**Files:**
- Test: `tldw_Server_API/tests/Admin/test_admin_acp_new_endpoints.py`

**Endpoints to test:**
1. `GET /admin/acp/agents/metrics` — mock session DB, verify GROUP BY results
2. `PATCH /admin/acp/sessions/{id}/budget` — set budget, verify fields updated
3. Budget enforcement — create session with budget, simulate token increment past budget, verify auto-termination

**Commit:** `test(backend): add tests for ACP agent metrics and session budget enforcement`

---

### Task 5: admin_user.py — Invitations

**Files:**
- Test: `tldw_Server_API/tests/Admin/test_admin_invitations.py`

**Endpoints to test:**
1. `POST /admin/users/invite` — mock email, verify token + expiry + delivery tracking
2. `GET /admin/users/invitations` — pagination, status filter (pending/accepted/expired)
3. `POST /admin/users/invitations/{id}/resend` — verify rate limit (3 max), token regen
4. `DELETE /admin/users/invitations/{id}` — revoke pending, reject non-pending

**Commit:** `test(backend): add tests for user invitation workflow`

---

### Task 6: Other Endpoints — Voice Validation, Error Breakdown, Rate Limits, Jobs SLA

**Files:**
- Test: `tldw_Server_API/tests/Admin/test_admin_misc_endpoints.py`

**Endpoints to test:**
1. `POST /voice/commands/{id}/validate` — mock VoiceCommandRouter, verify per-step validation
2. `GET /admin/errors/breakdown` — mock system service, verify grouping
3. `GET /admin/rate-limits/summary` — mock governor data
4. Jobs SLA policy CRUD + `GET /admin/jobs/sla-breaches`

**Commit:** `test(backend): add tests for voice validation, error breakdown, rate limits, jobs SLA`

---

## Workstream 2: Frontend Unit Tests

### Task 7: New Page Tests — Webhooks, AI Ops, Billing Analytics

**Files:**
- Test: `admin-ui/app/webhooks/__tests__/page.test.tsx`
- Test: `admin-ui/app/ai-ops/__tests__/page.test.tsx`
- Test: `admin-ui/app/billing/analytics/__tests__/page.test.tsx`

**Per page test:**
- Renders without error
- Shows loading skeleton then data
- Handles API error gracefully
- Key interactions (create dialog opens, export works, etc.)

**Run:** `cd admin-ui && npx vitest run app/webhooks/__tests__/ app/ai-ops/__tests__/ app/billing/analytics/__tests__/`

**Commit:** `test(admin-ui): add unit tests for webhooks, AI ops, billing analytics pages`

---

### Task 8: New Page Tests — Registration, Voice Validation, ACP Budget

**Files:**
- Test: `admin-ui/app/users/registration/__tests__/page.test.tsx`
- Test: `admin-ui/app/voice-commands/[id]/__tests__/page.test.tsx` (extend existing)
- Test: `admin-ui/app/acp-sessions/__tests__/page.test.tsx` (extend existing)
- Test: `admin-ui/app/acp-agents/__tests__/page.test.tsx` (extend existing)

**Commit:** `test(admin-ui): add unit tests for registration, voice validation, ACP budget pages`

---

### Task 9: Component Tests — TagInput, UptimeBar, SpendProgressBar, ErrorBreakdown

**Files:**
- Test: `admin-ui/components/ui/__tests__/tag-input.test.tsx`
- Test: `admin-ui/app/monitoring/components/__tests__/ErrorBreakdownPanel.test.tsx`
- Test: `admin-ui/app/dependencies/__tests__/UptimeBar.test.tsx`
- Test: `admin-ui/app/budgets/__tests__/SpendProgressBar.test.tsx`

**TagInput tests:** add tag via Enter, add via comma, remove tag, paste multi-tag, dedup
**UptimeBar tests:** renders SVG, color mapping (green/yellow/red), handles empty data
**SpendProgressBar tests:** 0%, 50%, 90%, 100%+, null data
**ErrorBreakdown tests:** renders table, handles empty, groups by endpoint

**Commit:** `test(admin-ui): add component tests for TagInput, UptimeBar, SpendProgressBar, ErrorBreakdown`

---

### Task 10: Extend Existing Page Tests — Resource Governor, Incidents SLA, Org Members

**Files:**
- Modify: `admin-ui/app/resource-governor/__tests__/page.test.tsx`
- Modify: `admin-ui/app/incidents/__tests__/page.test.tsx`
- Modify: `admin-ui/app/organizations/[id]/__tests__/page.test.tsx`

**Add tests for:** rate limit analytics section, SLA metric cards, member search + pagination, tab navigation

**Commit:** `test(admin-ui): extend tests for resource-governor, incidents SLA, org detail tabs`

---

## Workstream 3: E2E Tests

### Task 11: Webhook Management E2E

**Files:**
- Create: `admin-ui/tests/e2e/real-backend/webhooks.spec.ts`

**Scenarios:**
1. Navigate to /webhooks, verify empty state
2. Create webhook with URL + events, verify secret shown
3. Dismiss secret dialog, verify webhook in table
4. Test delivery, verify delivery history appears
5. Disable webhook, verify status changes
6. Delete webhook with confirmation

**Commit:** `test(e2e): add webhook management Playwright spec`

---

### Task 12: Compliance + Billing E2E

**Files:**
- Create: `admin-ui/tests/e2e/real-backend/compliance.spec.ts`
- Create: `admin-ui/tests/e2e/real-backend/billing-analytics.spec.ts`

**Compliance scenarios:** Navigate, verify score card, verify breakdown cards, schedule report
**Billing scenarios:** Navigate, verify MRR, subscribers, plan distribution

**Commit:** `test(e2e): add compliance and billing analytics Playwright specs`

---

### Task 13: Invitations + AI Ops + Dependencies E2E

**Files:**
- Create: `admin-ui/tests/e2e/real-backend/invitations.spec.ts`
- Create: `admin-ui/tests/e2e/real-backend/ai-ops.spec.ts`
- Create: `admin-ui/tests/e2e/real-backend/dependencies.spec.ts`

**Commit:** `test(e2e): add invitations, AI ops, dependencies Playwright specs`

---

### Task 14: ACP Budget + Incidents SLA E2E

**Files:**
- Create: `admin-ui/tests/e2e/real-backend/acp-budget.spec.ts`
- Create: `admin-ui/tests/e2e/real-backend/incidents-sla.spec.ts`

**Commit:** `test(e2e): add ACP budget and incidents SLA Playwright specs`

---

## Workstream 4: Documentation

### Task 15: README.md Complete Rewrite

**File:** `admin-ui/README.md`

**Sections:**
1. Overview + screenshots description
2. Complete feature list (30+ pages organized by category)
3. Architecture (App Router, proxy pattern, auth flow, state management)
4. Getting started (prerequisites, install, dev server, env vars)
5. Testing (unit, E2E, a11y, coverage commands)
6. Deployment (Docker build, compose, env config, health probes)
7. Contributing (code style, PR checklist, test requirements)

**Commit:** `docs(admin-ui): complete README rewrite with all features and guides`

---

### Task 16: API Reference Document

**File:** `admin-ui/docs/api-reference.md`

Document all admin proxy endpoints:
- Group by category (Dashboard, IAM, API Keys, ACP, Monitoring, Billing, Compliance, Webhooks)
- Per endpoint: method, path, auth required, request schema, response schema, example
- Include error responses

**Commit:** `docs(admin-ui): add comprehensive API reference for all admin endpoints`

---

### Task 17: Architecture + Deployment + Testing Guides

**Files:**
- Create: `admin-ui/docs/architecture.md`
- Create: `admin-ui/docs/deployment-guide.md`
- Create: `admin-ui/docs/testing-guide.md`

**Architecture:** Component tree, provider hierarchy, proxy pattern, auth flow, state management, design system
**Deployment:** Docker build args, runtime env, Kubernetes manifests, health probes, scaling
**Testing:** How to write backend tests, frontend unit tests, E2E tests, test utilities

**Commit:** `docs(admin-ui): add architecture, deployment, and testing guides`

---

### Task 18: Feature Guides + Troubleshooting

**Files:**
- Create: `admin-ui/docs/feature-guides/webhooks.md`
- Create: `admin-ui/docs/feature-guides/compliance.md`
- Create: `admin-ui/docs/feature-guides/ai-ops.md`
- Create: `admin-ui/docs/feature-guides/billing.md`
- Create: `admin-ui/docs/troubleshooting.md`

**Per feature guide:** Purpose, setup, usage walkthrough, configuration options, API endpoints used
**Troubleshooting:** Common issues (auth failures, proxy errors, missing env vars, test failures)

**Commit:** `docs(admin-ui): add feature guides and troubleshooting`

---

### Task 19: Release Checklist Update

**File:** `admin-ui/Release_Checklist.md`

Update with:
- Feature-specific validation items for each new capability
- New E2E test commands
- Regression check list for critical paths
- Snapshot review guidance

**Commit:** `docs(admin-ui): update release checklist with feature-specific validation`

---

## Verification

After all tasks:
- `cd admin-ui && npx vitest run` — all tests pass (target: 650+ tests)
- `cd admin-ui && npx playwright test` — E2E specs pass
- `python -m pytest tldw_Server_API/tests/Admin/ -v` — backend tests pass
- All documentation renders correctly in Markdown
- README accurately describes all features
