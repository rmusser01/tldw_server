# Admin-UI Tier 3: Post-Launch Polish — Phased Implementation Design

**Date:** 2026-03-27
**Scope:** All remaining post-launch polish items from the production readiness audit
**Approach:** Full backend + frontend implementation, dependency-driven phasing
**Total Estimated Effort:** ~54 hours across 6 weeks

---

## Context

Tiers 1-2 of the admin-ui production readiness plan have shipped (33 commits, 522 tests). The admin-ui is now deployable with security headers, Docker containerization, auth hardening, observability foundations, and critical UX safety fixes. Tier 3 covers polish items that improve quality, accessibility, and business intelligence but don't block production deployment.

---

## Phase A: Observability & Accessibility (Week 1)

**Goal:** Complete the observability stack and fix all accessibility issues.
**Effort:** ~12 hours | **Dependencies:** None (all frontend-only)

### A.1: Sentry Error Tracking Integration (4-6h)
- Install `@sentry/nextjs` (~100KB gzipped)
- Create `sentry.client.config.ts`, `sentry.server.config.ts`, `sentry.edge.config.ts`
- Wrap `next.config.js` with `withSentryConfig()` (chain with existing `withBundleAnalyzer`)
- Report errors from `app/error.tsx` and `app/global-error.tsx`
- Report proxy failures from `app/api/proxy/[...path]/route.ts`
- Disabled when `SENTRY_DSN` env var is absent
- **Files:** `package.json`, `next.config.js`, `sentry.*.config.ts` (new), `app/error.tsx`, `app/global-error.tsx`

### A.2: Client-Side Console→Logger Migration (3-4h)
- Replace ~124 `console.*` calls across ~38 client component files
- Use existing `lib/logger.ts` (works client-side)
- Add component context to each call
- Keep `console.error` in `error.tsx`/`global-error.tsx` as fallback
- **Files:** ~38 files in `app/*/page.tsx` and `components/`

### A.3: Chart Accessibility (2-3h) — REVIEW.md 11.1
- 5 Recharts components need `role="img"` + descriptive `aria-label`
- Add collapsible `<details>` data table fallback below each chart
- **Files:** `components/dashboard/ActivitySection.tsx`, `app/monitoring/components/MetricsChart.tsx`, plus 3 others

### A.4: ExportMenu Keyboard Accessibility (1-2h) — REVIEW.md 11.2
- Rewrite `components/ui/export-menu.tsx` using Radix `DropdownMenu` (v2.1.16 already in deps)
- Add `role="menu"`, `aria-haspopup`, keyboard arrow navigation, focus management
- **Files:** `components/ui/export-menu.tsx`

### A.5: Graceful Shutdown Documentation (0.5h)
- Verified: Next.js 15 standalone `server.js` on Node 20 handles SIGTERM correctly
- Document in README.md deployment section
- No code changes needed

**Phase A Verification:**
- Sentry receives test error within 30s
- `grep -r 'console\.' --include='*.ts' --include='*.tsx' | grep -v test | grep -v logger.ts` returns only error.tsx/global-error.tsx
- Charts pass axe-core accessibility audit
- ExportMenu navigable via keyboard Tab/Enter/Arrow keys

---

## Phase B: UX Refactoring (Week 2)

**Goal:** Break up monolithic page components and address remaining UX debt.
**Effort:** ~8 hours | **Dependencies:** Phase A (Sentry catches regressions)

### B.1: Org Detail Page Tabs (3-4h) — REVIEW.md 2.5
- Split 1132-line `app/organizations/[id]/page.tsx` into Radix Tabs:
  - **Members** — member table with search/pagination
  - **Teams** — teams list
  - **Keys & Secrets** — BYOK keys, watchlist settings
  - **Billing** — subscription/usage (conditional on billing enabled)
- Use `@radix-ui/react-tabs` (v1.1.13 already in deps)
- **Files:** `app/organizations/[id]/page.tsx` (refactor), potential tab component files

### B.2: Registration Code Relocation (3-4h) — REVIEW.md 2.10
- Extract ~140 lines (registration code CRUD) from `app/page.tsx` to `app/users/registration/page.tsx`
- Keep only a summary card + link on dashboard
- Reduces dashboard from ~1162 lines
- **Files:** `app/page.tsx` (reduce), `app/users/registration/page.tsx` (new)

### B.3: Quick Wins Batch (1h)
- Make API key hygiene cards clickable → wire `onClick` to table filters (REVIEW.md 3.2, 30min)
- Expand keyboard shortcuts to cover more pages (REVIEW.md 9.4, 30min)

**Phase B Verification:**
- Org detail page renders all tabs with correct content
- Registration codes page accessible via navigation
- Dashboard page is < 1050 lines
- Key hygiene cards filter the table on click

---

## Phase C: Backend + Frontend Operational Features (Weeks 3-4)

**Goal:** Add operational safety features that require backend API changes.
**Effort:** ~20 hours | **Dependencies:** None (parallel with Phase B)

### C.1: Dashboard KPI Cards — Active Sessions + Token Consumption (4h)
**Backend (2h):**
- New endpoint: `GET /api/v1/admin/stats/realtime`
- Returns: `{ active_sessions: int, tokens_today: { prompt: int, completion: int, total: int } }`
- Source: ACP sessions store (active count) + usage tracking (daily tokens)
- **Files:** `tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py`

**Frontend (2h):**
- Add 2 cards to `components/dashboard/StatsGrid.tsx`
- "Active Sessions" with live count
- "Tokens Today" with prompt/completion breakdown
- **Files:** `components/dashboard/StatsGrid.tsx`, `lib/dashboard-kpis.ts`, `lib/api-client.ts`

### C.2: ACP Session Token Budgets + Auto-Termination (10-14h) — REVIEW.md 4.6 (Critical)
**Backend (6-8h):**
- Add to ACP session schema: `token_budget: int | None`, `auto_terminate_at_budget: bool`
- Add DB columns: `token_budget INTEGER DEFAULT NULL`, `auto_terminate_at_budget BOOLEAN DEFAULT 0`
- Add to agent config: `default_token_budget` field
- Auto-termination logic: after each message, check `total_tokens >= token_budget` → close session
- `PATCH /admin/acp/sessions/{id}/budget` to set/update budget on existing sessions
- **Files:** `ACP_Sessions_DB.py`, `admin_acp_sessions_service.py`, `agent_client_protocol.py`, ACP schemas

**Frontend (4-6h):**
- Budget config field on agent edit form
- Progress bar on sessions table (tokens used / budget)
- "Set Budget" action on individual sessions
- Warning badge when > 80% of budget consumed
- Auto-refresh already in place (15s interval)
- **Files:** `app/acp-agents/page.tsx`, `app/acp-sessions/page.tsx`, `lib/api-client.ts`

### C.3: Compliance Posture Dashboard (6-8h) — REVIEW.md 6.11
**Backend (4-5h):**
- New endpoint: `GET /api/v1/admin/compliance/posture`
- Returns aggregated metrics:
  ```json
  {
    "mfa_adoption_pct": 78.5,
    "api_key_rotation_compliance_pct": 65.0,
    "keys_needing_rotation": 12,
    "retention_policy_active": true,
    "audit_log_retention_days": 90,
    "overall_score": 72
  }
  ```
- Source: users table (MFA status), API keys (age), retention policies
- **Files:** `tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py`, new compliance service

**Frontend (2-3h):**
- New page: `app/compliance/page.tsx`
- Score card (overall posture grade A-F)
- Breakdown cards: MFA adoption, key rotation, retention compliance, audit coverage
- Link to remediation actions (e.g., "View users without MFA")
- **Files:** `app/compliance/page.tsx` (new), `lib/api-client.ts`, `lib/navigation.ts`

**Phase C Verification:**
- Dashboard shows active sessions and token consumption KPIs
- ACP session with budget shows progress bar; auto-terminates when budget exceeded
- Compliance page shows posture score with breakdown

---

## Phase D: Business Intelligence (Weeks 5-6)

**Goal:** Add revenue analytics and advanced billing features.
**Effort:** ~14 hours | **Dependencies:** Billing APIs from backend

### D.1: Revenue Analytics Dashboard (10-12h) — REVIEW.md 7.9
**Backend (6-8h):**
- New endpoint: `GET /api/v1/admin/billing/analytics`
- Returns:
  ```json
  {
    "mrr_cents": 149970,
    "subscriber_count": 42,
    "plan_distribution": { "free": 20, "pro": 15, "enterprise": 7 },
    "churn_rate_30d_pct": 4.2,
    "trial_conversion_rate_pct": 35.0,
    "revenue_trend_30d": [{ "date": "2026-03-01", "mrr_cents": 140000 }, ...],
    "top_revenue_orgs": [{ "org_id": 1, "name": "Acme", "mrr_cents": 9999 }, ...]
  }
  ```
- Source: subscriptions table, plan pricing, historical snapshots
- **Files:** New billing analytics service, endpoint in admin billing endpoints

**Frontend (4h):**
- New page: `app/billing/analytics/page.tsx`
- MRR trend chart (Recharts AreaChart with accessible fallback)
- Subscriber distribution (by plan)
- Churn rate + trial conversion metrics
- Top revenue organizations table
- **Files:** `app/billing/analytics/page.tsx` (new), `lib/api-client.ts`, `lib/navigation.ts`

### D.2: Subscription Lifecycle + At-Risk Identification (4h) — REVIEW.md 7.4, 7.5
**Backend (2h):**
- Extend subscription list response with: `days_since_past_due`, `plan_changes_history`, `payment_failures_count`
- Add `at_risk` computed flag (past_due > 7 days, high usage near limit, or 3+ payment failures)

**Frontend (2h):**
- "Needs Attention" section at top of subscriptions page
- Lifecycle detail view (plan changes, payment history, status transitions)
- At-risk badge on subscription rows
- **Files:** `app/subscriptions/page.tsx`, `lib/api-client.ts`

**Phase D Verification:**
- Revenue analytics page shows MRR, churn, conversion with charts
- Subscription page highlights at-risk subscriptions
- Lifecycle detail shows plan change history

---

## Remaining REVIEW.md Items (Not in Phases A-D)

These are nice-to-have items that can be addressed opportunistically:

| ID | Finding | Effort | Notes |
|----|---------|--------|-------|
| 1.3 | Cache hit rate as KPI | 30min | Promote from ActivitySection |
| 1.5 | Activity chart error/latency overlays | 2h | Recharts multi-series |
| 1.7 | Job Queue in System Health | 30min | Add to DASHBOARD_SUBSYSTEMS |
| 1.8 | System Health error details | 1h | Show reason on degraded |
| 1.9 | RecentActivity severity filter | 1h | Add toggle + count badges |
| 2.1 | User table Created At + MFA batch | 2h | Needs bulk MFA endpoint |
| 2.3 | Dormant account indicator | 1h | Red badge if > 90d inactive |
| 2.6 | Org member search/pagination | 2h | Match users list pattern |
| 2.7 | Permission matrix search/grouping | 3h | Namespace-based grouping |
| 2.9 | Bulk org/team operations | 3h | Checkbox + bulk action bar |
| 3.4 | Create Key on hub page | 1h | Wizard or link |
| 3.7 | Proactive key expiration alerts | 2h | Needs notification system |
| 4.3 | ACP tool permissions picker | 2h | Multi-select from MCP tools |
| 4.15 | AI Operations summary dashboard | 4h | Aggregate page |
| 6.3 | Budget forecasting/trends | 3h | Sparklines + burn rate |
| 9.7 | Optimistic updates | 4h | For flag toggles, status changes |

**Total additional effort:** ~30 hours (can be done as sprint backlog items)

---

## Summary

| Phase | Timeline | Effort | Type | Key Deliverables |
|-------|----------|--------|------|------------------|
| **A** | Week 1 | ~12h | Frontend | Sentry, logger migration, chart a11y, ExportMenu a11y |
| **B** | Week 2 | ~8h | Frontend | Org tabs, registration relocation, quick wins |
| **C** | Weeks 3-4 | ~20h | Full stack | KPI cards, token budgets, compliance dashboard |
| **D** | Weeks 5-6 | ~14h | Full stack | Revenue analytics, subscription lifecycle |
| **Backlog** | Ongoing | ~30h | Mixed | 16 remaining REVIEW.md items |
| **Total** | 6 weeks | **~84h** | | |
