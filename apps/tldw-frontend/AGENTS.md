# AGENTS.md - tldw Web UI (Next.js)

This file provides context for coding agents when working on the web UI codebase.

## Overview

The tldw Web UI is a **Next.js application** that serves as a thin wrapper around shared UI code from `packages/ui/src/`. It provides:

- Browser-based access to tldw features (chat, media, RAG, etc.)
- JWT authentication for multi-user mode
- Browser API shims for code shared with the extension

## Monorepo Architecture

This app is part of a bun workspace monorepo that shares UI code with the browser extension.

**Key directories:**
- `packages/ui/src/` — Shared components, hooks, services, routes, stores
- `tldw-frontend/` — This directory: Next.js wrapper, shims, web-only code
- `extension/` — WXT browser extension (separate build)

**For feature parity guidelines, see:** [../DEVELOPMENT.md](../DEVELOPMENT.md)

## Directory Structure

```
tldw-frontend/
├── pages/                 # Next.js pages (thin wrappers around shared routes)
├── extension/
│   └── shims/             # Browser API compatibility shims
│       ├── wxt-browser.ts     # localStorage-based browser.* shim
│       └── react-router-dom.tsx  # Next.js router shim for react-router-dom
├── hooks/                 # Web-only hooks (auth/config are shared — see @/services/tldw/TldwAuth)
├── lib/                   # Web-only utilities
│   ├── api.ts             # Fetch wrapper with auth
│   └── auth.ts            # Token management
├── components/            # Web-only components (rare)
├── types/                 # Web-specific TypeScript types
└── styles/                # Global styles, Tailwind imports
```

**Shared code (imported from `@/`):**
- `packages/ui/src/components/` — React components
- `packages/ui/src/hooks/` — React hooks
- `packages/ui/src/services/` — API calls, business logic
- `packages/ui/src/store/` — Zustand state stores
- `packages/ui/src/routes/` — Page-level components

## Import Aliases

| Alias | Resolves To | Usage |
|-------|-------------|-------|
| `@/` | `packages/ui/src/` | Shared components, hooks, services |
| `~/` | `packages/ui/src/` | Alternative shared path |
| `@tldw/ui` | `packages/ui/src/` | Explicit shared package import |
| `@web/*` | `tldw-frontend/*` | Web-only modules |

```typescript
// Shared UI (from packages/ui/src/)
import { MyComponent } from "@/components/MyComponent"
import { useMyHook } from "@/hooks/use-my-hook"
import { myService } from "@/services/my-service"
import { tldwAuth } from "@/services/tldw/TldwAuth"  // shared auth (live path)

// Web-only (from tldw-frontend/)
import { api } from "@web/lib/api"

// Browser APIs (automatically shimmed)
import { browser } from "wxt/browser"  // Uses localStorage shim
import { Link, useNavigate } from "react-router-dom"  // Uses Next.js router shim
```

## Web-Specific Patterns

### 1. Page Wrappers with SSR Disabled

All pages must disable SSR because shared code uses browser APIs:

```typescript
// pages/my-feature.tsx
import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/my-feature-page"), { ssr: false })
```

**Why:** Shared code uses IndexedDB (Dexie), browser storage APIs, and `window` which don't exist during SSR.

### 2. Browser API Shims

The `extension/shims/` directory provides compatibility layers:

**`wxt-browser.ts`** — Maps `browser.storage` to `localStorage`:
```typescript
// Shared code calls:
await browser.storage.local.set({ key: "value" })

// Shim translates to:
localStorage.setItem("key", JSON.stringify("value"))
```

**`react-router-dom.tsx`** — Maps react-router to Next.js router:
```typescript
// Shared code uses:
import { Link, useNavigate } from "react-router-dom"

// Shim provides:
// <Link to="/path"> → <NextLink href="/path">
// useNavigate().push() → useRouter().push()
```

### 3. Authentication

Auth state lives in the shared stack (`@/services/tldw/TldwAuth` + the `@/store/connection`
store), not a web-only hook. Pages stay thin wrappers; auth is resolved inside the shared route
component:

```typescript
// pages/protected-page.tsx
import dynamic from "next/dynamic"

// Auth is resolved inside the shared route via tldwAuth / the connection store.
export default dynamic(() => import("@/routes/protected-route"), { ssr: false })
```

### 4. Platform Detection in Shared Code

When behavior must differ:

```typescript
// packages/ui/src/utils/platform.ts
export const isExtension = typeof chrome !== "undefined" && chrome.runtime?.id

// Usage in shared code:
if (isExtension) {
  // Extension-specific behavior
} else {
  // Web-specific behavior
}
```

## Common Commands

```bash
# From apps/tldw-frontend/

# Development (use port 8080 for CORS compatibility)
npm run dev -- -p 8080
# or
bun run dev -- -p 8080

# Type checking
npm run lint
bun run --cwd ../packages/ui tsc --noEmit

# Tests
npm run test              # Unit tests (Vitest)
npm run test:integration  # Full integration tests
npm run smoke             # API connectivity check

# Production build
npm run build
```

## Environment Variables

Copy `.env.local.example` to `.env.local`:

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend URL | `http://127.0.0.1:8000` |
| `NEXT_PUBLIC_API_VERSION` | API version | `v1` |
| `NEXT_PUBLIC_X_API_KEY` | Single-user API key | — |
| `NEXT_PUBLIC_API_BEARER` | Bearer token for chat | — |

## Testing

```bash
# Unit tests
npm run test

# Integration tests (starts backend if needed)
npm run test:integration

# Smoke test (API connectivity)
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
NEXT_PUBLIC_X_API_KEY=your_key \
npm run smoke
```

## Common Pitfalls

### 1. Missing SSR Disable

**Wrong:**
```typescript
// pages/my-page.tsx
import MyRoute from "@/routes/my-route"
export default MyRoute  // Breaks SSR!
```

**Right:**
```typescript
import dynamic from "next/dynamic"
export default dynamic(() => import("@/routes/my-route"), { ssr: false })
```

### 2. Web-Only Imports in Shared Code

**Wrong:**
```typescript
// packages/ui/src/components/MyComponent.tsx
import { api } from "@web/lib/api"  // Breaks extension!
```

**Right:** Keep web-only imports in `tldw-frontend/pages/` wrappers only.

### 3. Direct Browser API Usage

**Wrong:**
```typescript
chrome.storage.local.get("key")  // No shim for chrome.*
```

**Right:**
```typescript
import { browser } from "wxt/browser"  // Uses shim
browser.storage.local.get("key")
```

## Key Dependencies

- `next` — React framework with SSR/SSG
- `@tanstack/react-query` — Server state management
- `zustand` — Client state management
- `tailwindcss` — Utility CSS
- `antd` — UI component library
- `axios` — HTTP client

## Related Documentation

- [../DEVELOPMENT.md](../DEVELOPMENT.md) — Feature parity workflows
- [../extension/AGENTS.md](../extension/AGENTS.md) — Extension development guide
- [README.md](./README.md) — Setup and deployment details
