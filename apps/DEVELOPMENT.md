# Development Guide: Extension & Web UI Parity

This guide helps developers maintain feature parity between the browser extension and web UI while allowing platform-specific implementations where necessary.

## Setup Modes

- Recommended user/self-hosting setup: `make quickstart`
- API-only Docker setup: `make quickstart-docker`
- Team/public deployment: `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- Local contributor setup: `make quickstart-install` for the API and `bun run --cwd apps/tldw-frontend dev` for the WebUI

Use the Docker paths when you want a stable instance. Use the local paths when you are actively changing code, debugging, or running frontend development workflows.

## Local Development Setup

### Local API

```bash
# from repo root
make quickstart-install

# already have the venv and deps?
make quickstart-local
```

### Local WebUI

```bash
# from repo root
cd apps/tldw-frontend
cp .env.local.example .env.local
bun install
bun run dev -- -p 8080
```

If Turbopack becomes unstable or its cache is corrupted, use:

```bash
cd apps/tldw-frontend
bun run dev:webpack
```

Related setup docs:
- `Docs/Getting_Started/Profile_Docker_Single_User.md`
- `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- `Docs/Getting_Started/Profile_Local_Single_User.md`

## Architecture Overview

### Monorepo Structure

```
apps/
├── extension/           # WXT browser extension (build config, manifests)
│   ├── entrypoints/     # Thin wrappers pointing to shared entries
│   └── wxt.config.ts    # Extension build configuration
├── packages/
│   └── ui/
│       └── src/         # Shared UI source (the core of both platforms)
│           ├── components/   # Shared React components
│           ├── hooks/        # Shared React hooks
│           ├── services/     # API calls, business logic
│           ├── store/        # Zustand state management
│           ├── routes/       # Page-level components
│           ├── entries/      # Extension entrypoints (background, sidepanel, etc.)
│           ├── types/        # Shared TypeScript types
│           ├── utils/        # Utility functions
│           ├── config/       # Configuration constants
│           ├── context/      # React context providers
│           ├── db/           # Dexie database schemas
│           └── assets/       # Fonts, icons, i18n locales
├── tldw-frontend/       # Next.js web application
│   ├── pages/           # Next.js page wrappers
│   ├── extension/
│   │   └── shims/       # Browser API compatibility shims
│   ├── hooks/           # Web-only hooks
│   ├── lib/             # Web-only utilities
│   └── components/      # Web-only components (layout, networking, UI primitives)
└── package.json         # Workspace root
```

### Import Alias Mapping

| Alias | Extension Resolution | Web Resolution | Usage |
|-------|---------------------|----------------|-------|
| `@/` | `packages/ui/src/` | `packages/ui/src/` | Shared components, hooks, services |
| `~/` | `packages/ui/src/` | `packages/ui/src/` | Alternative shared path |
| `@tldw/ui` | `packages/ui/src/` | `packages/ui/src/` | Explicit shared package |
| `@web/*` | N/A | `tldw-frontend/*` | Web-only modules |

## Code Organization Rules

### Where to Put New Code

| Feature Type | Location | Reason |
|--------------|----------|--------|
| Shared components | `packages/ui/src/components/` | Used by both platforms |
| Shared hooks | `packages/ui/src/hooks/` | Platform-agnostic logic |
| Shared services | `packages/ui/src/services/` | API calls, business logic |
| Shared stores | `packages/ui/src/store/` | Zustand state management |
| Shared routes | `packages/ui/src/routes/` | Page-level components |
| Shared types | `packages/ui/src/types/` | TypeScript interfaces |
| Extension entries | `packages/ui/src/entries/` | Background, content scripts, sidepanel |
| Web-only auth | `tldw-frontend/lib/` | JWT handling, session management |
| Web-only hooks | `tldw-frontend/hooks/` | `useAuth`, `useConfig`, etc. |
| Web-only pages | `tldw-frontend/pages/` | Next.js routing wrappers |

### Decision Tree

```
Is this code specific to the browser extension APIs?
├── Yes → packages/ui/src/entries/ (if entrypoint) or services with platform detection
└── No → Is this code specific to Next.js/web?
    ├── Yes → tldw-frontend/ (hooks, lib, or components)
    └── No → packages/ui/src/ (shared code)
```

## Feature Development Workflow

### Adding a New Feature (Step-by-Step)

1. **Define types** in `packages/ui/src/types/`
   ```typescript
   // packages/ui/src/types/my-feature.ts
   export interface MyFeatureData {
     id: string
     name: string
   }
   ```

2. **Implement core logic** in `packages/ui/src/services/`
   ```typescript
   // packages/ui/src/services/my-feature.ts
   import type { MyFeatureData } from "@/types/my-feature"

   export async function fetchMyFeature(): Promise<MyFeatureData[]> {
     // API call logic
   }
   ```

3. **Create shared hooks** in `packages/ui/src/hooks/`
   ```typescript
   // packages/ui/src/hooks/use-my-feature.ts
   import { useQuery } from "@tanstack/react-query"
   import { fetchMyFeature } from "@/services/my-feature"

   export function useMyFeature() {
     return useQuery({ queryKey: ["my-feature"], queryFn: fetchMyFeature })
   }
   ```

4. **Build shared components** in `packages/ui/src/components/`
   ```typescript
   // packages/ui/src/components/MyFeature/MyFeatureList.tsx
   import { useMyFeature } from "@/hooks/use-my-feature"

   export function MyFeatureList() {
     const { data } = useMyFeature()
     return <ul>{data?.map(item => <li key={item.id}>{item.name}</li>)}</ul>
   }
   ```

5. **Create route component** in `packages/ui/src/routes/`
   ```typescript
   // packages/ui/src/routes/my-feature-page.tsx
   import { MyFeatureList } from "@/components/MyFeature/MyFeatureList"

   export default function MyFeaturePage() {
     return (
       <div>
         <h1>My Feature</h1>
         <MyFeatureList />
       </div>
     )
   }
   ```

6. **Wire up extension** (if needed) in `packages/ui/src/entries/`

7. **Create Next.js page wrapper** in `tldw-frontend/pages/`
   ```typescript
   // tldw-frontend/pages/my-feature.tsx
   import dynamic from "next/dynamic"

   export default dynamic(() => import("@/routes/my-feature-page"), { ssr: false })
   ```

8. **Test both platforms**
   ```bash
   # Test in extension
   cd apps/extension && bun run dev

   # Test in web
   cd apps/tldw-frontend && bun run dev
   ```

## Platform Adaptation Patterns

### Browser API Abstraction

The extension uses real `browser.*` APIs while the web uses shims.

**Extension (real APIs):**
```typescript
// In shared code, import from wxt/browser
import { browser } from "wxt/browser"
await browser.storage.local.set({ key: "value" })
```

**Web (shimmed):**
The `tldw-frontend/extension/shims/wxt-browser.ts` provides a localStorage-based implementation:
```typescript
// Automatically resolved via tsconfig paths
import { browser } from "wxt/browser"  // Uses shim in web context
```

### Storage Abstraction

| Context | Primary Storage | Fallback |
|---------|-----------------|----------|
| Extension | `browser.storage` + Dexie | — |
| Web | `localStorage` + Dexie | — |

**Pattern:** Use Zustand stores in `packages/ui/src/store/` which internally handle storage differences:
```typescript
// packages/ui/src/store/settings.ts
import { create } from "zustand"
import { persist } from "zustand/middleware"

export const useSettingsStore = create(
  persist(
    (set) => ({
      theme: "light",
      setTheme: (theme: string) => set({ theme })
    }),
    { name: "settings-storage" }
  )
)
```

### Routing Abstraction

| Context | Router | Solution |
|---------|--------|----------|
| Extension | React Router | Native `react-router-dom` |
| Web | Next.js Router | Shim at `extension/shims/react-router-dom.tsx` |

**Pattern:** Always import from `react-router-dom` in shared code:
```typescript
// packages/ui/src/components/Navigation.tsx
import { Link, useNavigate, useLocation } from "react-router-dom"

// Works in both contexts - web uses shim automatically
```

The web shim maps:
- `<Link to="...">` → `<NextLink href="...">`
- `useNavigate()` → `useRouter().push/replace`
- `useLocation()` → `useRouter().asPath` parsing

### repo2txt Route Parity

The `repo2txt` feature is implemented as a shared options route in `packages/ui/src/routes/option-repo2txt.tsx`.

- Extension options route: `options.html#/repo2txt`
- Web wrapper route: `tldw-frontend/pages/repo2txt.tsx` (dynamic import of the shared route)
- Sidepanel behavior (V1): link-out only to options; no in-panel repo2txt surface

### Authentication Abstraction

| Context | Auth Method |
|---------|-------------|
| Extension | API key stored in extension storage |
| Web | JWT with session management |

**Web-only hooks** in `tldw-frontend/hooks/`:
- `useAuth` - JWT authentication state
- `useConfig` - Server configuration

**Pattern:** Check for auth in route components, use shared services for API calls:
```typescript
// packages/ui/src/services/api.ts
export async function fetchWithAuth(url: string) {
  // Uses stored credentials (works in both contexts)
}
```

## Running the Project

### Development Servers

```bash
# From repo root
cd apps

# Install all workspace dependencies
bun install

# Extension development (opens browser with extension loaded)
bun run --cwd extension dev

# Web development (runs Next.js dev server)
bun run --cwd tldw-frontend dev
```

### Production Builds

```bash
cd apps

# Extension production build (outputs to extension/build/)
bun run --cwd extension build

# Web production build
bun run --cwd tldw-frontend build
```

### Testing

```bash
cd apps

# Run extension tests
bun run --cwd extension test

# Run web tests
bun run --cwd tldw-frontend test
```

## Feature Parity Checklist

Use this template when adding new features:

```markdown
## Feature: [Name]

### Implementation Status
- [ ] Types defined in `packages/ui/src/types/`
- [ ] Core logic in `packages/ui/src/services/`
- [ ] Shared hooks in `packages/ui/src/hooks/`
- [ ] Components in `packages/ui/src/components/`
- [ ] Route created in `packages/ui/src/routes/`
- [ ] Extension entry updated (if browser API needed)
- [ ] Next.js page wrapper created in `tldw-frontend/pages/`

### Testing
- [ ] Tested in extension dev mode (Chrome/Firefox)
- [ ] Tested in web dev mode
- [ ] Works without browser APIs (for web)
- [ ] SSR disabled for dynamic imports

### Platform-Specific Notes
- Extension-specific behavior: [describe or N/A]
- Web-specific behavior: [describe or N/A]
```

## Common Pitfalls

### 1. Direct Browser API Usage

**Wrong:**
```typescript
// packages/ui/src/components/MyComponent.tsx
chrome.storage.local.get("key")  // Breaks on web
```

**Right:**
```typescript
// packages/ui/src/components/MyComponent.tsx
import { browser } from "wxt/browser"  // Uses shim on web
browser.storage.local.get("key")
```

### 2. Web-Only Code in Shared Package

**Wrong:**
```typescript
// packages/ui/src/components/MyComponent.tsx
import { useAuth } from "@web/hooks/useAuth"  // Breaks extension
```

**Right:**
```typescript
// tldw-frontend/pages/my-page.tsx (web-only wrapper)
import { useAuth } from "@web/hooks/useAuth"
import MyComponent from "@/routes/my-route"

export default function MyPage() {
  const { user } = useAuth()
  return user ? <MyComponent /> : <Login />
}
```

### 3. Missing Next.js Page Wrapper

**Wrong:** Only creating route in `packages/ui/src/routes/` without web wrapper

**Right:** Always create corresponding page in `tldw-frontend/pages/`:
```typescript
import dynamic from "next/dynamic"
export default dynamic(() => import("@/routes/my-route"), { ssr: false })
```

### 4. SSR Issues

All shared UI code must be imported with `{ ssr: false }` in Next.js pages because:
- Browser APIs don't exist during SSR
- Dexie requires browser environment
- Extension shims need `window`

**Pattern:**
```typescript
// tldw-frontend/pages/any-page.tsx
import dynamic from "next/dynamic"

// Always disable SSR for shared routes
export default dynamic(() => import("@/routes/some-route"), { ssr: false })
```

### 5. Forgetting Platform Detection

When behavior must differ between platforms:

```typescript
// packages/ui/src/utils/platform.ts
export const isExtension = typeof chrome !== "undefined" && chrome.runtime?.id

// packages/ui/src/services/storage.ts
import { isExtension } from "@/utils/platform"

export async function saveData(key: string, value: unknown) {
  if (isExtension) {
    await browser.storage.local.set({ [key]: value })
  } else {
    localStorage.setItem(key, JSON.stringify(value))
  }
}
```

## Quick Reference

### File Locations

| Need | Location |
|------|----------|
| Shared component | `packages/ui/src/components/[Feature]/` |
| Shared hook | `packages/ui/src/hooks/use-[feature].ts` |
| API service | `packages/ui/src/services/[feature].ts` |
| Zustand store | `packages/ui/src/store/[feature].ts` |
| Route/page component | `packages/ui/src/routes/[feature]-page.tsx` |
| Extension background | `packages/ui/src/entries/background.ts` |
| Extension sidepanel | `packages/ui/src/entries/sidepanel/` |
| Web page wrapper | `tldw-frontend/pages/[feature].tsx` |
| Web-only hook | `tldw-frontend/hooks/use[Feature].ts` |
| Browser shim | `tldw-frontend/extension/shims/` |

### Common Commands

```bash
# Navigate to apps workspace
cd apps

# Install dependencies
bun install

# Start extension dev server
bun run --cwd extension dev

# Start web dev server
bun run --cwd tldw-frontend dev

# Build extension for production
bun run --cwd extension build

# Build web for production
bun run --cwd tldw-frontend build

# Type check shared UI
bun run --cwd packages/ui tsc --noEmit

# Lint (if configured)
bun run lint
```

### Import Cheatsheet

```typescript
// Shared UI (both platforms)
import { Component } from "@/components/Feature/Component"
import { useMyHook } from "@/hooks/use-my-hook"
import { myService } from "@/services/my-service"
import { useMyStore } from "@/store/my-store"
import type { MyType } from "@/types/my-type"

// Browser APIs (shimmed on web)
import { browser } from "wxt/browser"

// Routing (shimmed on web)
import { Link, useNavigate, useLocation } from "react-router-dom"

// Web-only (only in tldw-frontend/)
import { useAuth } from "@web/hooks/useAuth"
import { api } from "@web/lib/api"
```

---

For questions about this architecture, see also:
- `apps/Shared_UI_Monorepo.md` - Overview of the monorepo setup
- `apps/extension/AGENTS.md` - Extension-specific development guide
- `tldw-frontend/README.md` - Web UI specific documentation
