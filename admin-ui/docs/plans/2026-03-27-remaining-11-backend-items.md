# Remaining 11 Items — Backend Infrastructure Plan

## Context

57 commits have shipped, addressing ~65 of 71 REVIEW.md findings. The remaining 11 items all require backend infrastructure work. Key finding: **all 11 are extensions of existing systems** — no greenfield development needed. The backend already has email services, webhook DLQ, job metrics, health checks, MFA endpoints, and a full voice command pipeline.

**Estimated total:** ~38 days of focused work across 4 phases.

---

## Phase 1: Quick Backend Extensions (Week 1-2, ~8 days)

Items that extend existing endpoints with minimal new infrastructure.

### 2.2: Bulk MFA Status (~2.5 days)
**Backend:**
- Extend `GET /admin/users` to accept `include=mfa_status` query param
- Add `mfa_enabled` boolean to each user in list response (JOIN on `two_factor_enabled` column — already exists in users table)
- Add `mfa_enabled` filter param (true/false/all)
- **Files:** `tldw_Server_API/app/api/v1/endpoints/admin/admin_users.py`, user listing query
- **Reference:** Compliance posture endpoint already does aggregate MFA count via efficient SQL

**Frontend:**
- Remove N+1 `getUserMfaStatus()` calls in `admin-ui/app/users/page.tsx`
- Read MFA status from user list response directly
- Add MFA Status column with badge

### 10.1: Webhook Management CRUD (~2 days)
**Backend:**
- New table: `admin_webhooks` (id, url, secret, events[], enabled, created_at)
- CRUD endpoints: `GET/POST/PATCH/DELETE /admin/webhooks`
- Reuse existing `webhook.py` adapter for HMAC signing
- **Files:** New repo file, new endpoint file, reuse `app/core/Workflows/adapters/integration/webhook.py`
- **Reference:** Workflow webhook DLQ pattern at `workflows_webhook_dlq_service.py`

**Frontend:**
- New page: `admin-ui/app/webhooks/page.tsx` with CRUD table
- Add to navigation under integrations

### 10.6: User Invitation Workflow (~2 days)
**Backend:**
- Extend existing registration code system with email-based invites
- New endpoint: `POST /admin/users/invite` accepting email + role
- Send invite via existing email service (`email_service.py`)
- Track invite status: pending → accepted → expired
- **Files:** `admin_users.py` endpoint, `email_service.py` templates

**Frontend:**
- Add "Invite User" button to users page
- Invite dialog: email input + role selector
- Pending invitations table section

### 5.10: Jobs SLA Violation Alerting (~1.5 days)
**Backend:**
- Add SLA threshold config: `POST /admin/jobs/sla-policies` (max_processing_seconds, max_wait_seconds per job_type)
- Extend existing `jobs_metrics_service.py` `increment_sla_breach()` to check thresholds
- Wire to monitoring alert system when breached
- **Files:** `jobs_metrics_service.py`, `admin_ops.py` endpoints
- **Reference:** `increment_sla_breach()` already exists, just needs threshold logic

**Frontend:**
- Add SLA policy management section to jobs page
- Highlight breaching jobs in red
- Show breach count badge on SLA card

---

## Phase 2: Monitoring & Health Extensions (Week 3-4, ~9 days)

Items that extend the health/monitoring stack.

### 5.4: Dependencies Beyond LLM Providers (~3 days)
**Backend:**
- Create unified dependency health endpoint: `GET /admin/dependencies`
- Aggregate health from existing checks: Embeddings, RAG, TTS, STT, Database, Job Queue, Audit
- Each dependency: name, status (healthy/degraded/down), latency_ms, last_checked, error
- **Files:** New `admin_dependencies.py` endpoint, import from existing health checks in `health_checks.py`, `health_check.py`
- **Reference:** Health checks exist for each component; just need aggregation layer

**Frontend:**
- Expand `admin-ui/app/dependencies/page.tsx` to show all dependencies (not just LLM)
- Add status/latency columns for each

### 5.5: Historical Uptime Tracking (~3 days)
**Backend:**
- New table: `dependency_health_history` (dependency_name, status, latency_ms, checked_at)
- Periodic insert job (every 60s) storing health check results
- Aggregation endpoint: `GET /admin/dependencies/{name}/uptime?days=30`
- Returns: uptime_pct, downtime_minutes, incident_count, sparkline_data
- **Files:** New DB migration, scheduled job, aggregation query
- **Reference:** Schedulers exist (`scheduled_tasks_control_plane_service.py`)

**Frontend:**
- Add 7/30-day uptime percentage per dependency
- Sparkline for uptime history
- SLA-style availability badge

### 10.2: Webhook Delivery Status (~2 days)
**Backend:**
- Extend webhook CRUD (from Phase 1) with delivery log: `GET /admin/webhooks/{id}/deliveries`
- Store: attempt_at, status_code, response_time_ms, error, payload_hash
- Reuse DLQ retry pattern from `workflows_webhook_dlq_service.py`
- **Files:** Extend webhook repo, add delivery log table

**Frontend:**
- Add "Delivery History" tab to webhook detail view
- Show success/failure rates, recent attempts

### 4.12: Voice Command Dry-Run (~1 day)
**Backend:**
- Add `dry_run=true` query param to voice command test endpoint
- When dry_run: parse intent, validate action config, return validation report without executing
- Check: referenced MCP tool exists, config schema valid, required params present
- **Files:** Voice command test endpoint, `intent_parser.py`, `workflow_handler.py`
- **Reference:** Full pipeline exists; just skip the execution step

**Frontend:**
- Add "Dry Run" button next to existing "Test" on voice command detail page
- Show validation report: pass/fail per stage with error details

---

## Phase 3: Notification Infrastructure (Week 5-7, ~12 days)

Items requiring email delivery + scheduling. These share a common foundation.

### Foundation: Email Provider Adapter (~2 days, shared by all items below)
**Backend:**
- Abstract email provider interface (SMTP exists, add SendGrid/SES adapters)
- Provider selection via config: `EMAIL_PROVIDER=smtp|sendgrid|ses`
- Delivery tracking: store send attempts + status in DB
- **Files:** Extend `email_service.py`, new provider adapters

### 5.6: Incident Stakeholder Notifications (~3 days)
**Backend:**
- On incident status change, notify assigned users + configurable stakeholder list
- Channels: email (via provider), in-app notification
- Config: `POST /admin/incidents/{id}/notify` for manual, auto-notify on status transitions
- **Files:** `admin_ops.py` incident endpoints, notification service

**Frontend:**
- Add "Notify" button on incident detail
- Stakeholder list management (email addresses)
- Notification history on incident timeline

### 5.12: Scheduled Compliance Reports (~3 days)
**Backend:**
- Scheduled job: generate compliance report (PDF/HTML) on interval (daily/weekly/monthly)
- Reuse existing compliance posture data + audit export
- Deliver via email to configured recipients
- Store report history: `GET /admin/compliance/reports`
- **Files:** New scheduled job, report template, extend compliance endpoints
- **Reference:** `reading_digest_scheduler.py` pattern for scheduling

**Frontend:**
- Report schedule configuration (frequency, recipients, format)
- Report history list with download links

### 10.8: Scheduled Email Digests (~2 days)
**Backend:**
- Daily/weekly summary email: key metrics, alerts, incidents
- Reuse dashboard stats + monitoring data
- Configurable per admin user
- **Files:** New digest scheduler, email template
- **Reference:** `reading_digest_scheduler.py` already exists with similar pattern

**Frontend:**
- Digest preferences in user settings (frequency, enabled)
- Preview digest format

### 2.12: Resend Invite (~2 days, depends on 10.6 from Phase 1)
**Backend:**
- `POST /admin/users/invites/{id}/resend` endpoint
- Regenerate invite token, resend email
- Rate limit: max 3 resends per invite
- **Files:** Extend invitation endpoints from Phase 1

**Frontend:**
- Add "Resend" button to invitation rows
- Show resend count + last sent timestamp

---

## Phase 4: Remaining Items (Week 8, ~5 days)

### 3.6: Per-Key Cost/Usage Attribution (~3 days)
**Backend:**
- Track per-API-key usage: requests, tokens, estimated cost
- New table or extend usage tracking: `api_key_usage_daily` (key_id, date, request_count, tokens, cost_usd)
- Aggregation endpoint: `GET /admin/api-keys/{id}/usage?days=30`
- **Files:** New usage tracking middleware, aggregation query
- **Reference:** ACP session cost tracking pattern (pricing_catalog.py)

**Frontend:**
- Expandable row detail in API keys table showing usage breakdown
- Per-key usage chart (last 30 days)

### 10.4 + 10.5 + 10.7: Minor Info Architecture (~2 days)
These are partially frontend items but may need lightweight backend support:
- **10.4**: Error rate breakdown by endpoint — aggregate from existing logs
- **10.5**: Rate limit dashboard — aggregate from existing governor events
- **10.7**: Admin action filter — already partially done via audit filter

---

## Dependency Map

```
Phase 1 (Weeks 1-2)
├── 2.2  Bulk MFA ──────────── independent
├── 10.1 Webhook CRUD ─────── independent
├── 10.6 User Invitations ──── uses email_service
└── 5.10 Jobs SLA ─────────── independent

Phase 2 (Weeks 3-4)
├── 5.4  Dependencies ──────── independent
├── 5.5  Uptime History ────── depends on 5.4
├── 10.2 Webhook Delivery ──── depends on 10.1
└── 4.12 Voice Dry-Run ────── independent

Phase 3 (Weeks 5-7) ──── all depend on Email Provider Adapter
├── Foundation: Email Provider Adapter
├── 5.6  Incident Notifications
├── 5.12 Compliance Reports
├── 10.8 Email Digests
└── 2.12 Resend Invite ────── depends on 10.6

Phase 4 (Week 8)
├── 3.6  Per-Key Cost ──────── independent
└── 10.x Minor IA items ────── independent
```

## Parallelization

- **Phase 1**: All 4 items can be developed in parallel (different files/systems)
- **Phase 2**: 5.4, 4.12 in parallel; 5.5 after 5.4; 10.2 after 10.1
- **Phase 3**: All items after foundation adapter; 5.6, 5.12, 10.8 in parallel
- **Phase 4**: Both items in parallel

## Verification

After each phase:
- All existing tests pass (`npx vitest run` for frontend, `pytest` for backend)
- New endpoints return expected responses via curl
- Frontend pages display new data correctly
- E2E smoke tests cover critical paths

## Summary

| Phase | Timeline | Effort | Items |
|-------|----------|--------|-------|
| **1** | Weeks 1-2 | ~8 days | Bulk MFA, Webhook CRUD, User invites, Jobs SLA |
| **2** | Weeks 3-4 | ~9 days | Dependencies, uptime history, webhook delivery, voice dry-run |
| **3** | Weeks 5-7 | ~12 days | Email adapter, incident notifications, compliance reports, digests, resend invite |
| **4** | Week 8 | ~5 days | Per-key cost attribution, minor IA items |
| **Total** | 8 weeks | ~34 days | 11 items + foundation work |
