# Admin-UI Full Test Coverage + Documentation Push

**Date:** 2026-03-27
**Goal:** Bring test coverage from ~55% to ~95% and documentation from ~40% to ~90%
**Estimated effort:** ~174 hours across 4 parallel workstreams

---

## Workstream 1: Backend Unit Tests (~85h)

### Batch A: admin_ops.py (12 endpoints, ~40h)
- `GET /admin/stats/realtime` — mock ACP store, verify session count + token totals
- `GET /admin/compliance/posture` — mock DB queries, verify MFA + key rotation calcs
- `GET /admin/billing/analytics` — mock billing repo, verify MRR + subscriber counts
- `GET /admin/dependencies` — mock health checks, verify timeout handling
- `GET /admin/dependencies/{name}/uptime` — mock history store, verify uptime %
- `GET /admin/incidents/metrics/sla` — mock incidents, verify MTTA/MTTR math
- `POST /admin/incidents/{id}/notify` — mock email service, verify delivery tracking
- `GET /admin/email/deliveries` — mock store, verify pagination + status filter
- Webhook CRUD (4 endpoints) — create/list/update/delete, secret generation, event validation
- Report schedule CRUD (4 endpoints) — create/list/update/delete/send-now
- Digest preferences (2 endpoints) — get/set with user scoping

### Batch B: admin_acp_agents.py (3 endpoints, ~12h)
- `GET /admin/acp/agents/metrics` — mock session DB, verify GROUP BY aggregation
- `PATCH /admin/acp/sessions/{id}/budget` — set budget, verify enforcement trigger
- Budget auto-termination integration test

### Batch C: admin_user.py (3 endpoints, ~10h)
- `POST /admin/users/invite` — mock email, verify token + expiry
- `GET /admin/users/invitations` — pagination, status filter
- `POST /admin/users/invitations/{id}/resend` — rate limit (3 max), token regen

### Batch D: Other (5 endpoints, ~15h)
- Voice command validation — mock VoiceCommandRouter, verify per-step results
- Error breakdown — mock audit data, verify grouping
- Rate limit summary — mock governor events
- Jobs SLA policies CRUD + breach detection
- API key usage (verify existing tests comprehensive)

### Test pattern per endpoint:
1. Happy path with mocked dependencies
2. Auth required (401 without token)
3. Permission required (403 for non-admin)
4. Invalid input (400/422)
5. Not found (404 for missing resources)
6. Edge cases (empty data, boundary values)

---

## Workstream 2: Frontend Unit Tests (~40h)

### Pages (8 pages, ~25h)
- `/webhooks` — CRUD table, delivery panel, test action, secret display
- `/ai-ops` — KPI loading, agent table, session list, error state
- `/billing/analytics` — MRR card, subscriber card, plan distribution, billing-disabled state
- `/users/registration` — code list, create dialog, delete, copy
- `/voice-commands/[id]` — validation report, pass/fail display
- `/acp-agents` — metrics columns, budget fields in edit dialog
- `/acp-sessions` — budget progress bar, set budget dialog, cost column
- `/resource-governor` — rate limit analytics, throttle count

### Components (5 components, ~15h)
- `ErrorBreakdownPanel` — renders table, handles empty/error
- `TagInput` — add via Enter/comma, remove, paste, dedup
- `UptimeBar` — SVG rendering, color by health, downsampling
- `SpendProgressBar` — progress %, exhaustion label, color thresholds
- `OrgContextBanner` — renders for scoped users, null for unscoped

---

## Workstream 3: E2E Tests (~46h)

### Tier 1 — Critical (4 specs, ~24h)
- `webhooks.spec.ts` — create webhook, view in table, test delivery, check delivery log, delete
- `compliance.spec.ts` — view score, check breakdown cards, schedule report, send-now
- `billing-analytics.spec.ts` — view MRR, subscribers, plan distribution
- `user-invitations.spec.ts` — invite user, verify pending, resend, revoke

### Tier 2 — Operational (4 specs, ~22h)
- `ai-ops.spec.ts` — view KPIs, agent table, recent sessions
- `acp-budget.spec.ts` — set budget on session, verify progress bar updates
- `dependencies.spec.ts` — view system deps, check uptime badges
- `incidents-sla.spec.ts` — view SLA cards, notify stakeholders, check delivery

### E2E setup:
- Use `chromium` project for stubbed API tests
- Use `chromium-real-jwt` for real backend integration
- Page objects for new pages following existing pattern

---

## Workstream 4: Documentation (~49h)

### Core Updates (~9h)
- **README.md** — complete feature list (all 30+ pages), architecture overview, all endpoints, dev setup, test commands
- **Release_Checklist.md** — feature-specific validation items for each new capability

### New Docs (~40h)
- **docs/api-reference.md** (~10h) — all admin endpoints with req/res schemas, auth requirements, examples
- **docs/architecture.md** (~6h) — component tree, state management, proxy pattern, auth flow
- **docs/deployment-guide.md** (~6h) — Docker build, env vars, Kubernetes probes, monitoring setup
- **docs/testing-guide.md** (~4h) — how to write backend, frontend unit, and E2E tests
- **docs/troubleshooting.md** (~4h) — common issues (auth, CORS, proxy errors, test failures)
- **docs/feature-guides/** (~10h) — webhooks, compliance, AI ops, billing, invitations, ACP budgets

---

## Execution Order

```
Week 1-2: Backend tests Batch A + B (in parallel with docs core updates)
Week 2-3: Backend tests Batch C + D (in parallel with frontend unit tests)
Week 3-4: E2E Tier 1 specs (in parallel with API reference doc)
Week 4-5: E2E Tier 2 specs (in parallel with remaining docs)
```

All workstreams are parallelizable — backend tests, frontend tests, E2E, and docs touch different files.
