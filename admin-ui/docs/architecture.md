# Admin UI Architecture Guide

## 1. Overview

The Admin UI is a Next.js 15 application using the App Router. It provides a platform administration dashboard for the tldw_server backend. The browser never talks to the backend directly -- all API traffic flows through a server-side proxy route that forwards requests with credentials attached.

Key technology choices:
- **Next.js 15** with App Router and `output: 'standalone'` for containerized deployment
- **Radix UI** primitives with **Tailwind CSS** for styling
- **Vitest** for unit tests, **Playwright** for E2E tests
- **Bun** as the package manager (build-time), **Node.js 20** at runtime

## 2. Directory Structure

```
admin-ui/
  app/                    # Next.js App Router pages and layouts
    api/                  # Server-side route handlers
      proxy/[...path]/    # Catch-all proxy to backend
      health/             # Liveness and readiness probes
      auth/               # Login/logout server actions
    (feature)/            # Feature pages (users, orgs, jobs, etc.)
    layout.tsx            # Root layout
    providers.tsx         # Client-side provider tree
    page.tsx              # Dashboard home
  components/             # Shared React components
    ui/                   # Primitives (button, dialog, table, toast, etc.)
    dashboard/            # Dashboard-specific widgets
    data-ops/             # Backup/retention components
    api-keys/             # API key management components
    users/                # User management components
  lib/                    # Client-side utilities and hooks
    api-client.ts         # Typed API client (all backend calls)
    http.ts               # Low-level fetch wrapper with proxy routing
    api-config.ts         # Backend URL construction
    auth.ts               # Cookie/token management
    server-auth.ts        # Server-side auth header forwarding
    correlation-id.ts     # X-Request-Id generation
    logger.ts             # Structured logging
    env.ts                # Zod-validated environment schema
    use-paged-resource.ts # Generic paginated data hook
    use-resource-state.ts # Generic CRUD state hook
    use-url-state.ts      # URL search param sync hook
    scoped-storage.ts     # Namespaced localStorage wrapper
  types/                  # TypeScript type definitions
    index.ts              # Barrel export of all types
    incidents.ts          # Incident-related types
    webhooks.ts           # Webhook-related types
    email-deliveries.ts   # Email delivery types
  middleware.ts           # Auth gate (runs on every non-public route)
  next.config.js          # Build config, security headers, Sentry
  vitest.config.ts        # Unit test configuration
  playwright.config.ts    # E2E test configuration
```

## 3. Auth Flow

Authentication uses httpOnly cookies. The flow:

1. **Login page** (`/login`) -- user submits credentials (password or API key).
2. **Server action** (`/api/auth/...`) -- validates with backend, sets `access_token` (JWT) or `x_api_key` cookie.
3. **Middleware** (`middleware.ts`) -- runs on every non-public route. Checks for valid auth cookies or headers:
   - For JWTs: attempts local HMAC verification using `JWT_SECRET_KEY` (avoids backend round-trip). Falls back to `/users/me` API call if local verification is unavailable.
   - For API keys: validates via `/users/me` API call.
   - Results are cached in-memory (30s TTL for valid, 5s for invalid) with LRU eviction (max 500 entries).
4. **Redirect** -- unauthenticated requests are redirected to `/login?redirectTo=...`.
5. **Logout** -- invalidates the auth cache entry immediately, then clears the cookie.

The middleware matcher excludes `/login`, `/api/*`, `/_next/*`, and static files.

## 4. Proxy Pattern

All client-side API calls go through the Next.js server-side proxy. This keeps backend URLs and credentials out of the browser.

```
Browser                    Next.js Server              Backend
  |                             |                        |
  |  fetch(/api/proxy/admin/users)                       |
  |  ----cookie attached--->    |                        |
  |                             |  forward with auth     |
  |                             |  headers + X-Request-Id|
  |                             |  --------------------> |
  |                             |  <--- JSON response    |
  |  <--- proxied response      |                        |
```

Implementation details (`app/api/proxy/[...path]/route.ts`):
- Strips `/api/proxy` prefix, forwards to `NEXT_PUBLIC_API_URL/api/v1/{path}`
- Copies auth cookies into `Authorization` / `X-API-KEY` headers
- Adds `X-Request-Id` and `X-Forwarded-For` headers
- 30-second timeout with `AbortController`
- GET requests retry once on network failure (not on timeout)
- Returns 502 on backend unreachable, 504 on timeout

The client-side `lib/http.ts` module:
- `requestJson()`, `requestText()`, `requestBlob()` -- typed fetch wrappers
- Automatically prefixes endpoints with `/api/proxy`
- Sets `credentials: 'include'` so cookies are forwarded
- On 401: calls `logout()` and redirects to `/login`
- On 403 with CSRF message: throws a descriptive `ApiError`

## 5. State Management

The application avoids a global state library. State strategies:

- **React component state** -- for local UI state (form fields, open/close toggles).
- **URL search parameters** -- for filter state, pagination, and sort order. Uses `use-url-state.ts` for type-safe read/write of URL params. This makes views shareable and bookmarkable.
- **Scoped localStorage** -- `lib/scoped-storage.ts` provides namespaced `getItem`/`setItem` for persisting user preferences (collapsed sidebar, theme, etc.) without key collisions.
- **Custom hooks** -- `use-paged-resource.ts` provides a generic hook for fetching paginated lists with loading/error/refresh state. `use-resource-state.ts` provides CRUD lifecycle state.
- **Fetch-on-mount** pattern -- pages fetch data in `useEffect` via the `api` client. No query cache layer.

## 6. Provider Tree

The client-side provider tree wraps all pages. Defined in `app/providers.tsx`:

```
ErrorBoundary
  ToastProvider
    ConfirmProvider
      PrivilegedActionDialogProvider
        PermissionProvider
          OrgContextProvider
            KeyboardShortcutsProvider
              {children}
```

- **ErrorBoundary** -- catches unhandled React errors and renders a fallback UI.
- **ToastProvider** -- global toast notification system.
- **ConfirmProvider** -- `useConfirm()` hook for confirmation dialogs.
- **PrivilegedActionDialogProvider** -- modal requiring admin password re-entry for destructive actions (delete user, revoke sessions, etc.).
- **PermissionProvider** -- makes the current user's permissions available via `usePermissions()`. Guards features based on RBAC.
- **OrgContextProvider** -- tracks the currently-selected organization for org-scoped views. Provides `useOrgContext()`.
- **KeyboardShortcutsProvider** -- registers global keyboard shortcuts (navigation, quick actions).

## 7. Design System

The UI is built on **Radix UI** primitives styled with **Tailwind CSS**.

### Component Primitives (`components/ui/`)

| Component | Base |
|-----------|------|
| `button.tsx` | Radix Slot for `asChild` composition |
| `dialog.tsx` | Radix Dialog |
| `dropdown-menu.tsx` | Radix DropdownMenu |
| `select.tsx` | Radix Select |
| `tabs.tsx` | Radix Tabs |
| `checkbox.tsx` | Radix Checkbox |
| `label.tsx` | Radix Label |
| `toast.tsx` | Custom context-based toast |
| `table.tsx` | Semantic HTML table with Tailwind styling |
| `pagination.tsx` | Page navigation with ellipsis |
| `confirm-dialog.tsx` | Radix AlertDialog with confirm/cancel |
| `privileged-action-dialog.tsx` | Password re-entry modal for sensitive ops |
| `form.tsx` | Form layout with label/error scaffolding |
| `empty-state.tsx` | Placeholder for empty lists |
| `export-menu.tsx` | Download dropdown (CSV, JSON) |
| `status-indicator.tsx` | Colored dot + label for status values |
| `tag-input.tsx` | Multi-value tag input field |

### Patterns

- **Semantic color tokens** -- Tailwind classes like `bg-destructive`, `text-muted-foreground` map to CSS custom properties, enabling theme switching.
- **Responsive layout** -- `ResponsiveLayout.tsx` provides sidebar + main content with collapsible nav.
- **Breadcrumbs** -- `Breadcrumbs.tsx` renders based on the current route.
- **PlanGuard / PlanBadge** -- gating and labeling for billing-tier-locked features.

## 8. Testing Strategy

### Unit Tests (Vitest)

Configuration: `vitest.config.ts`

- Tests are colocated with source code as `*.test.ts` / `*.test.tsx` files.
- `lib/__tests__/` contains tests for utility modules.
- `components/__tests__/` contains component tests.
- `app/api/**/__tests__/` contains route handler tests.
- Mocking: `vi.mock()` for modules, `vi.fn()` for individual functions.

### E2E Tests (Playwright)

Configuration: `playwright.config.ts`

- Tests live alongside the feature pages or in a dedicated `e2e/` directory.
- Page objects encapsulate selectors and actions.
- Can run against a mock backend or a real backend (controlled by `TLDW_ADMIN_E2E_REAL_BACKEND` env var with port mapping in `api-config.ts`).

### Test Utilities

- `lib/normalize.ts` -- `normalizeListResponse()` and `normalizePagedResponse()` handle inconsistent backend response shapes, making tests deterministic.
- The `ApiError` class from `lib/http.ts` is re-exported by `lib/api-client.ts` for test assertions.
