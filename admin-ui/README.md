# tldw Admin Panel

Administration dashboard for the tldw_server platform. This UI is intended for sysadmin and ops workflows and is separate from the end-user Next.js WebUI (`apps/tldw-frontend`).

## Features

### Identity & Access Management
- **Users** (`/users`) — User list with search, MFA status, dormant badges, bulk actions, inline role changes
- **User Detail** (`/users/[id]`) — Profile, role assignment, MFA reset, API keys per user
- **User API Keys** (`/users/[id]/api-keys`) — Per-user API key management
- **Registration Codes** (`/users/registration`) — Invite codes with max-uses, expiry, role-to-grant, toggle self-registration
- **Organizations** (`/organizations`) — Org listing with create dialog; tabbed detail view (Members, Teams, Keys, Billing)
- **Org Detail** (`/organizations/[id]`) — Tabbed org management (members, teams, API keys, billing)
- **Teams** (`/teams`) — Team management with all-orgs view for super-admins
- **Team Detail** (`/teams/[id]`) — Team members and settings
- **Roles** (`/roles`) — Role CRUD with permission assignment
- **Role Detail** (`/roles/[id]`) — Single role editing with permission toggles
- **Permission Matrix** (`/roles/matrix`) — Cross-role permission grid with search, group collapse, diff-only toggle, batch save
- **Role Comparison** (`/roles/compare`) — Side-by-side role permission comparison
- **API Keys** (`/api-keys`) — Unified key hub with hygiene cards, bulk revoke, per-key usage

### AI & Model Governance
- **Providers** (`/providers`) — LLM provider health, usage metrics, deprecated model detection, BYOK key management
- **BYOK** (`/byok`) — Bring-your-own-key management: shared org/team keys, per-user keys, validation runs, usage logs
- **ACP Agents** (`/acp-agents`) — Agent config CRUD with tool allow/deny lists, org/team scoping, enabled toggles
- **ACP Sessions** (`/acp-sessions`) — Live session list with 15s auto-refresh, token usage, cost, pause/terminate actions
- **AI Operations** (`/ai-ops`) — Unified dashboard: spend KPIs, agent metrics table, recent sessions at a glance
- **Voice Commands** (`/voice-commands`) — Phrase CRUD, dry-run validation, analytics charts (usage trends, top commands, active sessions)
- **Voice Command Detail** (`/voice-commands/[id]`) — Single command editing and test
- **MCP Servers** (`/mcp-servers`) — Module health checks, tool/resource/prompt counts, tool listings with schemas
- **Resource Governor** (`/resource-governor`) — Rate-limit policy CRUD (global/org/user/role scopes), throttle event log with metrics

### Operations & Monitoring
- **Dashboard** (`/`) — KPI cards (users, orgs, keys, providers, security, billing, uptime), activity chart with range/overlay controls, severity-filtered alerts, quick actions
- **Monitoring** (`/monitoring`) — System status, alert list with pagination, metrics chart, 60s auto-refresh
- **Dependencies** (`/dependencies`) — All system deps (LLM providers + infrastructure), health checks, uptime stats with sparklines
- **Incidents** (`/incidents`) — Incident CRUD with SLA metrics (MTTA/MTTR), stakeholder email notifications, action items, postmortem timeline, export
- **Jobs** (`/jobs`) — Background job queue management, SLA policies, breach detection
- **Audit Logs** (`/audit`) — Filterable audit trail with cross-reference to system logs via request ID
- **System Logs** (`/logs`) — Correlation filtering, request tracing, severity/source filters
- **Usage** (`/usage`) — Router analytics: status, quota, models, providers, access, network, conversations, request log

### Billing & Subscriptions
- **Plans** (`/plans`) — Plan CRUD with subscriber safety checks before deletion
- **Subscriptions** (`/subscriptions`) — Subscription list with at-risk identification, lifecycle status badges, org names, export
- **Revenue Analytics** (`/billing/analytics`) — MRR, active subscribers, plan distribution, trial conversion metrics
- **Budgets** (`/budgets`) — Per-org/user spend budgets with progress bars, exhaustion forecasting, alert thresholds, enforcement modes, export
- **Feature Registry** (`/feature-registry`) — Feature-to-plan matrix with inline toggle editing
- **Onboarding** (`/onboarding`) — Three-step org onboarding wizard (org details, plan selection, confirmation)

### Security & Compliance
- **Compliance** (`/compliance`) — Posture score (A-F grade), MFA adoption rate, key rotation status, report scheduling with email delivery
- **Security** (`/security`) — Risk score breakdown with weighted factors (MFA gaps, key age, failed logins, suspicious activity), remediation links
- **Data Ops** (`/data-ops`) — Backups, retention policies with visual escalation, data subject requests (DSR), exports, maintenance
- **Feature Flags** (`/flags`) — Flag CRUD with scope (global/org/user), rollout percentage, target users, maintenance mode toggle

### System Configuration
- **Config** (`/config`) — Read-only server configuration viewer with environment badge, provider summaries, feature flag overview
- **Debug** (`/debug`) — API key resolver (by key/user ID/key ID), permission resolver, JWT token validator

### Integrations
- **Webhooks** (`/webhooks`) — Webhook CRUD with event type selection, HMAC secret, test delivery action, expandable delivery history with status

## Architecture

- **Framework:** Next.js 15 (App Router), React 19, TypeScript 5.9
- **UI:** Radix UI primitives + Tailwind CSS, custom design tokens
- **Auth:** httpOnly JWT cookies; middleware token validation with LRU cache; API-key login path (disabled by default)
- **Data:** Server-side proxy (`/api/proxy/*`) to backend API; no direct browser-to-backend calls
- **State:** React state + URL search params for shareable, bookmarkable views
- **Observability:** Sentry (client, server, edge configs), structured JSON logger, X-Request-ID correlation via middleware

## Getting Started

### Prerequisites
- Node.js 20+ or Bun 1.3+
- tldw_server backend running on port 8000

### Install
```bash
cd admin-ui
bun install
```

### Development
```bash
bun run dev          # Start dev server on port 3001
```

Open `http://localhost:3001`.

### Environment Variables
```bash
cp .env.example .env.local
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | Yes | `http://localhost:8000` | Backend API URL |
| `NEXT_PUBLIC_API_VERSION` | No | `v1` | API version prefix |
| `NEXT_PUBLIC_DEFAULT_AUTH_MODE` | No | `password` | Auth mode (`password` or `apikey`) |
| `NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN` | No | `false` | Show API-key login tab in UI |
| `ADMIN_UI_ALLOW_API_KEY_LOGIN` | No | `false` | Server-side API-key login gate (requires `AUTH_MODE=single_user`) |
| `NEXT_PUBLIC_BILLING_ENABLED` | No | `false` | Enable billing features (plans, subscriptions, analytics, budgets) |
| `NEXT_PUBLIC_ADMIN_UI_ENABLE_UNSAFE_LOCAL_TOOLS` | No | `false` | Re-enable local-only DSR processing, backup scheduling, monitoring mutations |
| `NEXT_PUBLIC_SENTRY_DSN` | No | -- | Sentry error tracking DSN |
| `SENTRY_AUTH_TOKEN` | No | -- | Sentry release upload token |
| `JWT_SECRET_KEY` | No | -- | Local JWT verification (falls back to backend verification if absent) |
| `JWT_SECONDARY_SECRET` | No | -- | Rotated JWT secret for zero-downtime key rotation |
| `JWT_ALGORITHM` | No | `HS256` | JWT signing algorithm (HS256/384/512) |
| `SERVER_X_API_KEY` | No | -- | Server-side X-API-KEY for single-user mode proxy |

## Authentication

- **Multi-user mode** (default): Username/password login via Next.js route handlers that set httpOnly session cookies. Browser JavaScript never stores or reads admin bearer tokens.
- **Single-user mode**: API key login supported but disabled by default. The validated key is moved into an httpOnly session cookie after login. Requires both `NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN=true` and `ADMIN_UI_ALLOW_API_KEY_LOGIN=true`, plus `AUTH_MODE=single_user` on the backend.
- **Authenticated API calls**: The UI sends privileged requests through `/api/proxy/*` on the same origin. The proxy attaches the server-managed credential to backend requests.

## Testing

```bash
bun run test              # Unit tests (Vitest)
bun run test:a11y         # Accessibility tests (axe-core)
bun run test:smoke        # Playwright smoke tests (no backend required)
bun run test:real-backend # E2E against real backend (auto-starts backend instances)
bun run typecheck         # TypeScript type checking
bun run lint              # ESLint
bun run analyze           # Bundle size analysis (ANALYZE=true next build)
```

### Quality Gate

Run the full production gate before cutting a release or merging risky changes:

```bash
bun run lint
bun run typecheck
bun run test
bun run test:a11y
bun run build
bun run test:smoke
```

### Smoke Tests

`test:smoke` starts a local Next.js server by default, stubs `/api/auth/*` and `/api/proxy/*` in-browser, and verifies the hardened password+MFA login path plus privileged user actions. No live backend required.

To point at an already running server:
```bash
TLDW_ADMIN_UI_AUTOSTART=false \
TLDW_ADMIN_UI_URL=http://127.0.0.1:3001 \
bun run test:smoke
```

### Real-Backend E2E

```bash
bunx playwright install --with-deps chromium
bun run test:real-backend -- \
  --project=chromium-real-jwt \
  --project=chromium-real-single-user \
  --reporter=line
```

This lane auto-starts two auth-specific admin-ui servers plus matching backend instances:
- `chromium-real-jwt` — multi-user JWT admin flows
- `chromium-real-single-user` — single-user API-key admin flows

The suite runs with `--workers=1` intentionally since each auth-mode project shares one managed backend instance.

To reuse already-running backends:
```bash
TLDW_ADMIN_E2E_JWT_API_URL=http://127.0.0.1:8101 \
TLDW_ADMIN_E2E_SINGLE_USER_API_URL=http://127.0.0.1:8102 \
bun run test:real-backend -- \
  --project=chromium-real-jwt \
  --project=chromium-real-single-user \
  --reporter=line
```

Reused backends must have `ENABLE_ADMIN_E2E_TEST_MODE=true` for seed/reset/bootstrap helpers.

## Deployment

### Docker

```bash
docker build -f Dockerfiles/Dockerfile.admin-ui \
  --build-arg NEXT_PUBLIC_API_URL=https://api.example.com \
  --build-arg NEXT_PUBLIC_BILLING_ENABLED=true \
  -t tldw-admin:latest .

docker run -p 3001:3001 \
  -e JWT_SECRET_KEY=your-secret \
  tldw-admin:latest
```

### Docker Compose

```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.yml \
  -f Dockerfiles/docker-compose.admin-ui.yml \
  up -d --build
```

The admin-ui service depends on the `app` (backend) service being healthy before starting.

### Health Probes
- **Liveness:** `GET /api/health` — always 200 if process is running
- **Readiness:** `GET /api/health/ready` — 503 if backend is unreachable

## Project Structure

```
admin-ui/
├── app/                 # Next.js App Router pages and API routes
│   ├── api/             # Server-side proxy, auth, and health routes
│   ├── {page}/          # Feature pages (users, roles, incidents, etc.)
│   └── layout.tsx       # Root layout with sidebar navigation
├── components/          # Shared UI components (Radix-based primitives, guards, dialogs)
├── lib/                 # API client, auth helpers, formatters, analytics, export utilities
├── tests/               # Playwright test suites (smoke + real-backend E2E)
├── types/               # TypeScript type definitions
├── middleware.ts         # Auth validation, X-Request-ID injection, route protection
├── sentry.*.config.ts   # Sentry configuration (client, server, edge)
└── .env.example         # Environment variable template
```

## Security

- Content-Security-Policy with `frame-ancestors 'none'`
- HSTS, X-Frame-Options, X-Content-Type-Options headers
- Auth endpoint rate limiting (10 req/min per IP)
- httpOnly cookies for all credentials; no tokens in localStorage
- `PrivilegedActionDialog` confirmation for all destructive operations
- `PermissionGuard` component enforces RBAC on every route
- Maintenance mode allowlist for emergency admin access
