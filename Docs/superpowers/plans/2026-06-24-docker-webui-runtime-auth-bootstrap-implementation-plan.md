# Docker WebUI Runtime Auth Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Docker single-user WebUI auth come from runtime container env instead of build-time `NEXT_PUBLIC_X_API_KEY`, while keeping setup writes authenticated and Docker quickstart local-only by default.

**Architecture:** Add a WebUI-local Next.js API route that exposes single-user runtime auth only under strict local guards. Update the existing browser runtime bootstrap to fetch that route before app auth state is calculated, write runtime-owned credentials without clobbering manual credentials, and set the in-memory runtime auth helper so stale baked keys lose precedence. Wire Docker compose runtime env and setup remote-write env, then document and test the new contract.

**Tech Stack:** Next.js `pages/api`, Vitest/jsdom, Plasmo safe storage shim, Docker Compose YAML, FastAPI/Pytest smoke coverage, Bandit for touched Python scope.

---

## File Structure

- Create `apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts`
  - WebUI-local runtime config endpoint with host, forwarding-header, auth-mode, exposure-flag, and placeholder-key guards.
- Create `apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts`
  - Direct Next API handler tests using the existing `createApiRequest` and `createApiResponse` helpers.
- Modify `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts`
  - Export `runtimeBootstrapReady`.
  - Fetch runtime config before build-time env seeding.
  - Track runtime-owned key metadata in `tldwRuntimeAuthMetadata`.
  - Call `setRuntimeApiKey()` when runtime auth wins.
- Modify `apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`
  - Await `runtimeBootstrapReady`.
  - Cover runtime key precedence, stale env-key replacement, manual-key preservation, metadata writes, and fetch failure fallback.
- Modify `apps/tldw-frontend/pages/_app.tsx`
  - Import `runtimeBootstrapReady` and await it before `getConfiguredAuthState()`.
- Modify `apps/tldw-frontend/__tests__/app/app-layout.test.tsx`
  - Mock and control `runtimeBootstrapReady` so auth state is not read before bootstrap completes.
- Modify `Dockerfiles/docker-compose.webui.yml`
  - Pass runtime `AUTH_MODE`, `SINGLE_USER_API_KEY`, and `TLDW_WEBUI_EXPOSE_RUNTIME_AUTH` into the WebUI container, with runtime-auth exposure disabled by default unless a local setup path explicitly opts in.
  - Keep host binding `127.0.0.1:8080:3000`.
- Modify `Dockerfiles/docker-compose.single-user.yml`
  - Set `TLDW_SETUP_ALLOW_REMOTE=${TLDW_SETUP_ALLOW_REMOTE:-1}` on `app`.
- Modify `apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts`
  - Add compose-file assertions for runtime env, loopback binding, internal API origin, and setup remote flag.
- Modify `apps/tldw-frontend/__tests__/pr-916-review-followups.test.ts`
  - Keep existing Docker build alignment assertions valid after runtime-auth env additions.
- Modify `apps/packages/ui/src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx`
  - Assert setup admin requests do not pass `noAuth: true`.
- Create `tldw_Server_API/tests/MCP_unified/test_packaging_shape.py`
  - Import-smoke the active in-tree MCP Unified package and assert this branch is not relying on a root `mcp_unified/` directory.
- Modify docs:
  - `README.md`
  - `Docs/Getting_Started/TROUBLESHOOTING.md`
  - `Docs/Getting_Started/Profile_Docker_Single_User.md`
  - `Docs/Published/Getting_Started/Profile_Docker_Single_User.md`
  - `Dockerfiles/README.md`
- Modify `backlog/tasks/task-2360 - Fix-Docker-single-user-WebUI-runtime-auth-bootstrap.md`
  - Link this plan and record verification results during execution.

---

### Task 1: Add WebUI Runtime Config Endpoint

**Files:**
- Create: `apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts`
- Create: `apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts`

- [ ] **Step 1: Write failing route tests**

Create `apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts`:

```ts
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import handler from "@web/pages/api/_tldw-webui/runtime-config"
import {
  createApiRequest,
  createApiResponse
} from "./test-utils"

const ORIGINAL_ENV = {
  AUTH_MODE: process.env.AUTH_MODE,
  SINGLE_USER_API_KEY: process.env.SINGLE_USER_API_KEY,
  TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH,
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
}

const restoreEnv = () => {
  for (const [key, value] of Object.entries(ORIGINAL_ENV)) {
    if (value === undefined) {
      delete process.env[key]
    } else {
      process.env[key] = value
    }
  }
}

const configureRuntimeAuth = () => {
  process.env.AUTH_MODE = "single_user"
  process.env.SINGLE_USER_API_KEY = "runtime-single-user-key"
  process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH = "1"
  process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
}

const callRuntimeConfig = async (headers: Record<string, string> = {}) => {
  const req = createApiRequest({
    method: "GET",
    url: "/api/_tldw-webui/runtime-config",
    headers: {
      host: "127.0.0.1:8080",
      ...headers
    }
  })
  const res = createApiResponse()
  await handler(req, res)
  return res
}

describe("WebUI runtime config API", () => {
  beforeEach(() => {
    restoreEnv()
    configureRuntimeAuth()
  })

  afterEach(() => {
    restoreEnv()
  })

  it("returns runtime single-user auth for local quickstart requests", async () => {
    const res = await callRuntimeConfig()

    expect(res.statusCode).toBe(200)
    expect(res.headers["cache-control"]).toContain("no-store")
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: true,
        authMode: "single-user",
        apiKey: "runtime-single-user-key"
      },
      networking: {
        deploymentMode: "quickstart"
      }
    })
  })

  it.each([
    ["disabled exposure", { TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: "0" }],
    ["multi-user auth mode", { AUTH_MODE: "multi_user" }],
    ["placeholder key", { SINGLE_USER_API_KEY: "change-me" }],
    ["blank key", { SINGLE_USER_API_KEY: "   " }]
  ])("returns unavailable for %s", async (_name, envPatch) => {
    Object.assign(process.env, envPatch)

    const res = await callRuntimeConfig()

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })

  it.each([
    ["api.example.test"],
    ["192.168.1.50:8080"],
    ["0.0.0.0:8080"]
  ])("returns unavailable for non-loopback host %s", async (host) => {
    const res = await callRuntimeConfig({ host })

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
  })

  it.each([
    ["forwarded", "for=203.0.113.10;host=localhost:8080"],
    ["x-forwarded-for", "203.0.113.10"],
    ["x-forwarded-host", "localhost:8080"],
    ["x-real-ip", "203.0.113.10"]
  ])("returns unavailable when %s is present", async (header, value) => {
    const res = await callRuntimeConfig({ [header]: value })

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
  })

  it("rejects non-GET methods without exposing auth", async () => {
    const req = createApiRequest({
      method: "POST",
      headers: { host: "127.0.0.1:8080" }
    })
    const res = createApiResponse()

    await handler(req, res)

    expect(res.statusCode).toBe(405)
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })
})
```

- [ ] **Step 2: Run route tests to verify they fail**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/pages/api/runtime-config.test.ts
```

Expected: FAIL with a module resolution error for `@web/pages/api/_tldw-webui/runtime-config`.

- [ ] **Step 3: Implement the runtime config route**

Create `apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts`:

```ts
import type { NextApiRequest, NextApiResponse } from "next"

type RuntimeConfigResponse = {
  runtimeAuth: {
    available: boolean
    authMode?: "single-user"
    apiKey?: string
    reason?: string
  }
  networking: {
    deploymentMode: string
    serverUrl: string
  }
}

const FORWARDED_HEADER_NAMES = [
  "forwarded",
  "x-forwarded-for",
  "x-forwarded-host",
  "x-real-ip"
]

const PLACEHOLDER_KEYS = new Set([
  "change-me",
  "changeme",
  "your-api-key",
  "your_api_key",
  "placeholder",
  "replace-me",
  "replace_me"
])

const normalizeEnvValue = (value?: string): string => String(value || "").trim()

const isSingleUserMode = (value?: string): boolean => {
  const normalized = normalizeEnvValue(value).toLowerCase().replace(/-/g, "_")
  return normalized === "single_user"
}

const isEnabled = (value?: string): boolean => {
  const normalized = normalizeEnvValue(value).toLowerCase()
  return normalized === "1" || normalized === "true" || normalized === "yes"
}

const isUsableApiKey = (value?: string): value is string => {
  const normalized = normalizeEnvValue(value)
  if (!normalized) return false
  if (/\s/.test(normalized)) return false
  return !PLACEHOLDER_KEYS.has(normalized.toLowerCase())
}

const extractHostname = (hostHeader?: string | string[]): string => {
  const host = Array.isArray(hostHeader) ? hostHeader[0] : hostHeader
  const normalized = normalizeEnvValue(host).toLowerCase()
  if (!normalized) return ""
  if (normalized.startsWith("[") && normalized.includes("]")) {
    return normalized.slice(1, normalized.indexOf("]"))
  }
  if (normalized === "::1") return normalized
  const colonCount = (normalized.match(/:/g) || []).length
  if (colonCount > 1) return normalized
  return normalized.split(":")[0] || ""
}

const isLoopbackHost = (hostHeader?: string | string[]): boolean => {
  const hostname = extractHostname(hostHeader)
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1"
}

const hasForwardingHeaders = (req: NextApiRequest): boolean =>
  FORWARDED_HEADER_NAMES.some((name) => {
    const value = req.headers[name]
    return Array.isArray(value) ? value.length > 0 : Boolean(value)
  })

const unavailable = (reason: string): RuntimeConfigResponse => ({
  runtimeAuth: {
    available: false,
    reason
  },
  networking: {
    deploymentMode: normalizeEnvValue(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE) || "quickstart",
    serverUrl: ""
  }
})

export default function handler(
  req: NextApiRequest,
  res: NextApiResponse<RuntimeConfigResponse | { error: string }>
) {
  res.setHeader("Cache-Control", "no-store, max-age=0")

  if (req.method !== "GET") {
    res.setHeader("Allow", "GET")
    res.status(405).json({ error: "Method not allowed" })
    return
  }

  if (!isSingleUserMode(process.env.AUTH_MODE)) {
    res.status(200).json(unavailable("auth-mode"))
    return
  }

  if (!isEnabled(process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH)) {
    res.status(200).json(unavailable("disabled"))
    return
  }

  if (!isLoopbackHost(req.headers.host)) {
    res.status(200).json(unavailable("host"))
    return
  }

  if (hasForwardingHeaders(req)) {
    res.status(200).json(unavailable("forwarded"))
    return
  }

  const apiKey = normalizeEnvValue(process.env.SINGLE_USER_API_KEY)
  if (!isUsableApiKey(apiKey)) {
    res.status(200).json(unavailable("api-key"))
    return
  }

  res.status(200).json({
    runtimeAuth: {
      available: true,
      authMode: "single-user",
      apiKey
    },
    networking: {
      deploymentMode: normalizeEnvValue(process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE) || "quickstart",
      serverUrl: ""
    }
  })
}
```

- [ ] **Step 4: Run route tests to verify they pass**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/pages/api/runtime-config.test.ts
```

Expected: PASS for all runtime-config route tests.

- [ ] **Step 5: Commit Task 1**

```bash
git add apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts
git commit -m "feat: add webui runtime auth config endpoint"
```

---

### Task 2: Implement Runtime Bootstrap Precedence

**Files:**
- Modify: `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts`
- Modify: `apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`

- [ ] **Step 1: Add failing bootstrap tests**

Append these tests inside the existing `describe("runtime-bootstrap chrome shim", () => { ... })` block in `apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`:

```ts
  it("seeds runtime auth before stale build-time env auth", async () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "stale-build-key"
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        runtimeAuth: {
          available: true,
          authMode: "single-user",
          apiKey: "runtime-key"
        },
        networking: {
          deploymentMode: "quickstart",
          serverUrl: ""
        }
      })
    }))
    vi.stubGlobal("fetch", fetchMock)

    const mod = await import("@web/extension/shims/runtime-bootstrap")
    await mod.runtimeBootstrapReady

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    const metadata = readStoredValue("tldwRuntimeAuthMetadata") as Record<string, unknown>

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/_tldw-webui/runtime-config",
      expect.objectContaining({
        credentials: "same-origin",
        cache: "no-store"
      })
    )
    expect(nextConfig.apiKey).toBe("runtime-key")
    expect(nextConfig.authMode).toBe("single-user")
    expect(metadata.source).toBe("webui-runtime")
  })

  it("replaces a stale key that was seeded from NEXT_PUBLIC_X_API_KEY", async () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "stale-build-key"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "stale-build-key",
        serverUrl: window.location.origin
      })
    )
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({
        runtimeAuth: {
          available: true,
          authMode: "single-user",
          apiKey: "runtime-key"
        },
        networking: {
          deploymentMode: "quickstart",
          serverUrl: ""
        }
      })
    })))

    const mod = await import("@web/extension/shims/runtime-bootstrap")
    await mod.runtimeBootstrapReady

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(nextConfig.apiKey).toBe("runtime-key")
  })

  it("does not replace a user-managed key", async () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "stale-build-key"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "manual-user-key",
        serverUrl: window.location.origin
      })
    )
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({
        runtimeAuth: {
          available: true,
          authMode: "single-user",
          apiKey: "runtime-key"
        },
        networking: {
          deploymentMode: "quickstart",
          serverUrl: ""
        }
      })
    })))

    const mod = await import("@web/extension/shims/runtime-bootstrap")
    await mod.runtimeBootstrapReady

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(nextConfig.apiKey).toBe("manual-user-key")
    expect(readStoredValue("tldwRuntimeAuthMetadata")).toBeNull()
  })

  it("replaces a previous runtime-owned key when the runtime key changes", async () => {
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "old-runtime-key",
        serverUrl: window.location.origin
      })
    )
    localStorage.setItem(
      "tldwRuntimeAuthMetadata",
      JSON.stringify({
        source: "webui-runtime",
        authMode: "single-user",
        fingerprint: "old"
      })
    )
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({
        runtimeAuth: {
          available: true,
          authMode: "single-user",
          apiKey: "new-runtime-key"
        },
        networking: {
          deploymentMode: "quickstart",
          serverUrl: ""
        }
      })
    })))

    const mod = await import("@web/extension/shims/runtime-bootstrap")
    await mod.runtimeBootstrapReady

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(nextConfig.apiKey).toBe("new-runtime-key")
  })

  it("falls back to existing env bootstrap when runtime config fetch fails", async () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "env-key"
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new Error("offline")
    }))

    const mod = await import("@web/extension/shims/runtime-bootstrap")
    await mod.runtimeBootstrapReady

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(nextConfig.apiKey).toBe("env-key")
  })
```

Also update the test file globals:

```ts
const originalXApiKey = process.env.NEXT_PUBLIC_X_API_KEY

afterEach(() => {
  vi.unstubAllGlobals()
  if (originalXApiKey === undefined) {
    delete process.env.NEXT_PUBLIC_X_API_KEY
  } else {
    process.env.NEXT_PUBLIC_X_API_KEY = originalXApiKey
  }
})
```

Keep the existing restore logic for `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE`.

- [ ] **Step 2: Run bootstrap tests to verify they fail**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/runtime-bootstrap.test.ts
```

Expected: FAIL because `runtimeBootstrapReady` is not exported and runtime fetch is not implemented.

- [ ] **Step 3: Implement runtime bootstrap helpers**

Modify `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts`.

Add this import near the top:

```ts
import { setRuntimeApiKey } from "@web/lib/authStorage"
```

Add these types and constants after `DEFAULT_TLDW_SERVER_URL`:

```ts
type RuntimeConfigPayload = {
  runtimeAuth?: {
    available?: boolean
    authMode?: "single-user"
    apiKey?: string
  }
  networking?: {
    deploymentMode?: string
    serverUrl?: string
  }
}

type RuntimeAuthMetadata = {
  source: "webui-runtime"
  authMode: "single-user"
  fingerprint: string
}

const RUNTIME_CONFIG_ENDPOINT = "/api/_tldw-webui/runtime-config"
const RUNTIME_AUTH_METADATA_KEY = "tldwRuntimeAuthMetadata"
```

Add these helpers before `seedTldwConfigFromEnv`:

```ts
const normalizeApiKey = (value?: string | null): string | null => {
  const normalized = String(value || "").trim()
  if (!normalized || /\s/.test(normalized)) return null
  return normalized
}

const fingerprintRuntimeKey = async (apiKey: string): Promise<string> => {
  try {
    const subtle = globalThis.crypto?.subtle
    if (!subtle) return `len:${apiKey.length}`
    const data = new TextEncoder().encode(apiKey)
    const digest = await subtle.digest("SHA-256", data)
    return Array.from(new Uint8Array(digest))
      .slice(0, 8)
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("")
  } catch {
    return `len:${apiKey.length}`
  }
}

const fetchRuntimeConfig = async (): Promise<RuntimeConfigPayload | null> => {
  if (typeof window === "undefined" || typeof fetch !== "function") return null
  try {
    const response = await fetch(RUNTIME_CONFIG_ENDPOINT, {
      credentials: "same-origin",
      cache: "no-store"
    })
    if (!response.ok) return null
    const payload = await response.json()
    return isRecord(payload) ? (payload as RuntimeConfigPayload) : null
  } catch {
    return null
  }
}

const shouldReplaceWithRuntimeKey = (
  existingKey: string | null,
  runtimeKey: string,
  metadata: RuntimeAuthMetadata | null,
  buildTimeKey: string | null
): boolean => {
  if (!existingKey) return true
  if (existingKey === runtimeKey) return true
  if (metadata?.source === "webui-runtime") return true
  if (buildTimeKey && existingKey === buildTimeKey) return true
  const lower = existingKey.toLowerCase()
  return lower === "change-me" || lower === "changeme" || lower === "your_api_key"
}

const seedTldwConfigFromRuntime = async (): Promise<boolean> => {
  if (typeof window === "undefined") return false
  const payload = await fetchRuntimeConfig()
  const runtimeKey = normalizeApiKey(payload?.runtimeAuth?.apiKey)
  if (!payload?.runtimeAuth?.available || payload.runtimeAuth.authMode !== "single-user" || !runtimeKey) {
    return false
  }

  const storage = createSafeStorage()
  const existing = (await storage.get<TldwConfig>("tldwConfig").catch(() => null)) || null
  const metadata =
    (await storage.get<RuntimeAuthMetadata>(RUNTIME_AUTH_METADATA_KEY).catch(() => null)) || null
  const buildTimeKey = normalizeApiKey(process.env.NEXT_PUBLIC_X_API_KEY)
  const existingKey = normalizeApiKey(existing?.apiKey)

  if (!shouldReplaceWithRuntimeKey(existingKey, runtimeKey, metadata, buildTimeKey)) {
    return false
  }

  const quickstartWebUiServerUrl = getQuickstartWebUiServerUrl()
  const next: TldwConfig = {
    ...(existing || {}),
    authMode: "single-user",
    apiKey: runtimeKey,
    serverUrl: quickstartWebUiServerUrl || existing?.serverUrl || ""
  }
  const nextMetadata: RuntimeAuthMetadata = {
    source: "webui-runtime",
    authMode: "single-user",
    fingerprint: await fingerprintRuntimeKey(runtimeKey)
  }

  await storage.set("tldwConfig", next)
  await storage.set(RUNTIME_AUTH_METADATA_KEY, nextMetadata)
  if (next.serverUrl) {
    await storage.set("tldwServerUrl", next.serverUrl)
  }
  setRuntimeApiKey(runtimeKey)
  window.dispatchEvent(new CustomEvent("tldw:config-updated"))
  return true
}
```

Replace the current bottom call:

```ts
void seedTldwConfigFromEnv()
```

with:

```ts
export const runtimeBootstrapReady = (async () => {
  const seededRuntimeAuth = await seedTldwConfigFromRuntime().catch(() => false)
  if (!seededRuntimeAuth) {
    await seedTldwConfigFromEnv().catch(() => undefined)
  }
})()
```

- [ ] **Step 4: Run bootstrap tests to verify they pass**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/extension/runtime-bootstrap.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add apps/tldw-frontend/extension/shims/runtime-bootstrap.ts apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts
git commit -m "feat: bootstrap webui auth from runtime config"
```

---

### Task 3: Await Bootstrap Before App Auth State

**Files:**
- Modify: `apps/tldw-frontend/pages/_app.tsx`
- Modify: `apps/tldw-frontend/__tests__/app/app-layout.test.tsx`

- [x] **Step 1: Write failing app auth-order test**

In `apps/tldw-frontend/__tests__/app/app-layout.test.tsx`, replace the side-effect runtime bootstrap behavior with a controllable mock before importing `App`:

```ts
let resolveRuntimeBootstrap: (() => void) | null = null
let runtimeBootstrapReady: Promise<void> = Promise.resolve()

const resetRuntimeBootstrap = (deferred = false) => {
  if (!deferred) {
    runtimeBootstrapReady = Promise.resolve()
    resolveRuntimeBootstrap = null
    return
  }
  runtimeBootstrapReady = new Promise<void>((resolve) => {
    resolveRuntimeBootstrap = resolve
  })
}

vi.mock("@web/extension/shims/runtime-bootstrap", () => ({
  get runtimeBootstrapReady() {
    return runtimeBootstrapReady
  }
}))
```

Call `resetRuntimeBootstrap()` in the existing `beforeEach`.

Add this test inside `describe("App layout routing", () => { ... })`:

```ts
  it("waits for runtime bootstrap before reading configured auth state", async () => {
    resetRuntimeBootstrap(true)
    currentConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "runtime-key"
    }

    renderApp("/media")

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(mockGetConfig).not.toHaveBeenCalled()

    await act(async () => {
      resolveRuntimeBootstrap?.()
      await runtimeBootstrapReady
    })

    await waitFor(() => {
      expect(mockGetConfig).toHaveBeenCalled()
    })
  })
```

- [x] **Step 2: Run app layout test to verify it fails**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/app/app-layout.test.tsx
```

Expected: FAIL because `_app.tsx` does not import or await `runtimeBootstrapReady`.

- [x] **Step 3: Await runtime bootstrap in `_app.tsx`**

Modify the import in `apps/tldw-frontend/pages/_app.tsx`.

Replace:

```ts
import "@web/extension/shims/runtime-bootstrap"
```

with:

```ts
import { runtimeBootstrapReady } from "@web/extension/shims/runtime-bootstrap"
```

Update `refreshAuthState`:

```ts
    const refreshAuthState = async () => {
      await runtimeBootstrapReady.catch(() => undefined)
      const envAuthed = hasEnvApiAuth()
      const configuredAuth = await getConfiguredAuthState()
      const authed = configuredAuth.hasConfig
        ? configuredAuth.authMode === "multi-user"
          ? configuredAuth.isAuthenticated
          : configuredAuth.isAuthenticated || envAuthed
        : envAuthed

      if (!cancelled) {
        setIsAuthenticated(authed)
        setAuthResolved(true)
      }
    }
```

- [x] **Step 4: Run app layout test to verify it passes**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/app/app-layout.test.tsx
```

Expected: PASS.

- [x] **Step 5: Commit Task 3**

```bash
git add apps/tldw-frontend/pages/_app.tsx apps/tldw-frontend/__tests__/app/app-layout.test.tsx
git commit -m "fix: wait for webui runtime auth bootstrap"
```

---

### Task 4: Wire Docker Runtime Auth And Setup Remote Flag

**Files:**
- Modify: `Dockerfiles/docker-compose.webui.yml`
- Modify: `Dockerfiles/docker-compose.single-user.yml`
- Modify: `apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts`
- Modify: `apps/tldw-frontend/__tests__/pr-916-review-followups.test.ts`

- [x] **Step 1: Add failing compose assertions**

In `apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts`, add:

```ts
const webuiComposePath = path.join(repoRoot, "Dockerfiles", "docker-compose.webui.yml")
const singleUserComposePath = path.join(repoRoot, "Dockerfiles", "docker-compose.single-user.yml")
```

Add these tests inside `describe("frontend quickstart networking", () => { ... })`:

```ts
  it("passes runtime auth env to the WebUI container without changing the loopback port binding", () => {
    const compose = readFileSync(webuiComposePath, "utf8")

    expect(compose).toContain("- AUTH_MODE=${AUTH_MODE:-single_user}")
    expect(compose).toContain("- SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}")
    expect(compose).toContain("- TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=${TLDW_WEBUI_EXPOSE_RUNTIME_AUTH:-0}")
    expect(compose).toContain('"127.0.0.1:8080:3000"')
    expect(compose).toContain("- TLDW_INTERNAL_API_ORIGIN=${TLDW_INTERNAL_API_ORIGIN:-http://app:8000}")
  })

  it("enables remote setup writes for the Docker single-user app service", () => {
    const compose = readFileSync(singleUserComposePath, "utf8")

    expect(compose).toContain("- TLDW_SETUP_ALLOW_REMOTE=${TLDW_SETUP_ALLOW_REMOTE:-1}")
    expect(compose).toContain("- AUTH_MODE=${AUTH_MODE:-single_user}")
    expect(compose).toContain("- SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}")
  })
```

In `apps/tldw-frontend/__tests__/pr-916-review-followups.test.ts`, extend the first test:

```ts
    expect(compose).toContain("- AUTH_MODE=${AUTH_MODE:-single_user}")
    expect(compose).toContain("- SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}")
    expect(compose).toContain("- TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=${TLDW_WEBUI_EXPOSE_RUNTIME_AUTH:-0}")
```

- [x] **Step 2: Run compose assertions to verify they fail**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/frontend-quickstart-networking.test.ts __tests__/pr-916-review-followups.test.ts
```

Expected: FAIL because the compose env entries do not exist yet.

- [x] **Step 3: Update Docker compose files**

Modify `Dockerfiles/docker-compose.webui.yml` service `webui.environment` to include:

```yaml
      - AUTH_MODE=${AUTH_MODE:-single_user}
      - SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}
      - TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=${TLDW_WEBUI_EXPOSE_RUNTIME_AUTH:-0}
```

The environment block should become:

```yaml
    environment:
      - NODE_ENV=production
      - HOSTNAME=0.0.0.0
      - PORT=3000
      - AUTH_MODE=${AUTH_MODE:-single_user}
      - SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}
      - TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=${TLDW_WEBUI_EXPOSE_RUNTIME_AUTH:-0}
      - NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL:-}
      - NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=${NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE:-quickstart}
      - TLDW_INTERNAL_API_ORIGIN=${TLDW_INTERNAL_API_ORIGIN:-http://app:8000}
```

Modify `Dockerfiles/docker-compose.single-user.yml` service `app.environment` to include:

```yaml
      - TLDW_SETUP_ALLOW_REMOTE=${TLDW_SETUP_ALLOW_REMOTE:-1}
```

Place it after `SINGLE_USER_API_KEY`:

```yaml
      - AUTH_MODE=${AUTH_MODE:-single_user}
      - SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}
      - TLDW_SETUP_ALLOW_REMOTE=${TLDW_SETUP_ALLOW_REMOTE:-1}
      - tldw_production=${tldw_production:-false}
```

- [x] **Step 4: Validate compose syntax**

Run from repo root:

```bash
docker compose -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml config >/tmp/tldw_single_webui_runtime_auth_compose.yml
```

Expected: exit code 0. Inspect `/tmp/tldw_single_webui_runtime_auth_compose.yml` only if the command fails.

- [x] **Step 5: Run compose assertions to verify they pass**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/frontend-quickstart-networking.test.ts __tests__/pr-916-review-followups.test.ts
```

Expected: PASS.

- [x] **Step 6: Commit Task 4**

```bash
git add Dockerfiles/docker-compose.webui.yml Dockerfiles/docker-compose.single-user.yml apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts apps/tldw-frontend/__tests__/pr-916-review-followups.test.ts
git commit -m "fix: pass docker webui auth at runtime"
```

---

### Task 5: Add Setup Auth And MCP Packaging Regression Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx`
- Create: `tldw_Server_API/tests/MCP_unified/test_packaging_shape.py`

- [x] **Step 1: Add failing setup auth regression assertion**

In `apps/packages/ui/src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx`, add a helper near existing test helpers:

```ts
const expectSetupAdminRequestsAuthenticated = (calls: Array<[Record<string, unknown>]>) => {
  const setupCalls = calls.filter(([init]) =>
    String(init?.path || "").startsWith("/api/v1/setup/admin/")
  )
  expect(setupCalls.length).toBeGreaterThan(0)
  for (const [init] of setupCalls) {
    expect(init).not.toHaveProperty("noAuth", true)
  }
}
```

In the existing provisioning test after the `bgRequest` expectations, add:

```ts
    expectSetupAdminRequestsAuthenticated(mocks.bgRequest.mock.calls)
```

In the existing verification test after the `bgRequest` expectations, add:

```ts
    expectSetupAdminRequestsAuthenticated(mocks.bgRequest.mock.calls)
```

- [x] **Step 2: Add MCP packaging smoke**

Create `tldw_Server_API/tests/MCP_unified/test_packaging_shape.py`:

```py
from importlib import import_module
from pathlib import Path


def test_mcp_unified_is_packaged_in_tree():
    module = import_module("tldw_Server_API.app.core.MCP_unified")

    assert module.__name__ == "tldw_Server_API.app.core.MCP_unified"
    assert getattr(module, "__version__", None)


def test_active_branch_does_not_depend_on_root_mcp_unified_package():
    repo_root = Path(__file__).resolve().parents[3]

    assert not (repo_root / "mcp_unified").exists()
```

- [x] **Step 3: Run setup and MCP tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run ../packages/ui/src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx
```

Expected: PASS.

Run from repo root:

```bash
.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_packaging_shape.py -v
```

Expected: PASS.

- [x] **Step 4: Commit Task 5**

```bash
git add apps/packages/ui/src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx tldw_Server_API/tests/MCP_unified/test_packaging_shape.py
git commit -m "test: cover setup auth and mcp packaging shape"
```

---

### Task 6: Update Docker Runtime Auth Documentation

**Files:**
- Modify: `README.md`
- Modify: `Docs/Getting_Started/TROUBLESHOOTING.md`
- Modify: `Docs/Getting_Started/Profile_Docker_Single_User.md`
- Modify: `Docs/Published/Getting_Started/Profile_Docker_Single_User.md`
- Modify: `Dockerfiles/README.md`

- [x] **Step 1: Update README WebUI auth copy**

Find the README section that says Docker/WebUI rebuilds require exporting `NEXT_PUBLIC_X_API_KEY`. Replace that wording with:

```md
For the Docker single-user WebUI quickstart, the WebUI container reads `AUTH_MODE` and `SINGLE_USER_API_KEY` at runtime and exposes the key to the browser only through the local runtime bootstrap path. You do not need to rebuild the WebUI image when the single-user key changes.

`NEXT_PUBLIC_X_API_KEY` remains available for advanced/static WebUI builds where the operator deliberately wants to bake a public browser credential into the client bundle. Do not use it for the normal Docker quickstart path.
```

- [x] **Step 2: Update troubleshooting auth section**

In `Docs/Getting_Started/TROUBLESHOOTING.md`, update the `"NEXT_PUBLIC_X_API_KEY" confusion` section to include:

```md
Docker single-user WebUI quickstart uses runtime auth bootstrap. Check these when the WebUI returns `401`:

- `make setup-docker-single` or the equivalent `tldw-setup init --profile docker-single-webui` command wrote `TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1` to `tldw_Server_API/Config_Files/.env`; the shared WebUI compose overlay defaults that flag to `0` without this explicit local quickstart opt-in.
- The WebUI is opened through the default loopback URL, `http://127.0.0.1:8080`.
- `SINGLE_USER_API_KEY` is not `change-me`.
- The backend `app` service has the same `SINGLE_USER_API_KEY`.

Only advanced/static deployments should set `NEXT_PUBLIC_X_API_KEY` before building the WebUI.
```

- [x] **Step 3: Update Docker single-user profile docs**

In both `Docs/Getting_Started/Profile_Docker_Single_User.md` and `Docs/Published/Getting_Started/Profile_Docker_Single_User.md`, add this note near the WebUI startup instructions:

```md
The WebUI image is not tied to a specific single-user API key. In the default local compose profile, the WebUI receives `SINGLE_USER_API_KEY` at container runtime and bootstraps browser auth from the local WebUI origin. Keep the default `127.0.0.1:8080:3000` binding unless you are intentionally configuring an advanced remote deployment.
```

- [x] **Step 4: Update Dockerfiles README**

In `Dockerfiles/README.md`, add:

```md
Single-user WebUI auth is runtime-configured. The `webui` service receives `AUTH_MODE`, `SINGLE_USER_API_KEY`, and `TLDW_WEBUI_EXPOSE_RUNTIME_AUTH`; the browser does not require `NEXT_PUBLIC_X_API_KEY` for the default local quickstart. The reusable overlay defaults runtime-auth exposure to disabled unless local setup writes `TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1`.
```

- [x] **Step 5: Run doc guard tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run __tests__/pr-916-review-followups.test.ts
```

Expected: PASS.

- [x] **Step 6: Commit Task 6**

```bash
git add README.md Docs/Getting_Started/TROUBLESHOOTING.md Docs/Getting_Started/Profile_Docker_Single_User.md Docs/Published/Getting_Started/Profile_Docker_Single_User.md Dockerfiles/README.md apps/tldw-frontend/__tests__/pr-916-review-followups.test.ts
git commit -m "docs: document docker webui runtime auth"
```

---

### Task 7: Final Verification And Task Finalization

**Files:**
- Modify: `backlog/tasks/task-2360 - Fix-Docker-single-user-WebUI-runtime-auth-bootstrap.md`

- [x] **Step 1: Run focused frontend tests**

Run from `apps/tldw-frontend`:

```bash
bunx vitest run \
  __tests__/pages/api/runtime-config.test.ts \
  __tests__/extension/runtime-bootstrap.test.ts \
  __tests__/app/app-layout.test.tsx \
  __tests__/frontend-quickstart-networking.test.ts \
  __tests__/pr-916-review-followups.test.ts \
  ../packages/ui/src/components/Option/Setup/__tests__/AudioInstallerPanel.test.tsx
```

Expected: PASS.

- [x] **Step 2: Run focused backend smoke**

Run from repo root:

```bash
.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_packaging_shape.py tldw_Server_API/tests/Security/test_setup_access_guard.py -v
```

Expected: PASS.

- [x] **Step 3: Validate compose config**

Run from repo root:

```bash
docker compose -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml config >/tmp/tldw_single_webui_runtime_auth_compose.yml
```

Expected: exit code 0.

- [x] **Step 4: Run Bandit for touched Python scope**

Run from repo root:

```bash
.venv/bin/python -m bandit -r tldw_Server_API/tests/MCP_unified/test_packaging_shape.py -f json -o /tmp/bandit_docker_webui_runtime_auth.json
```

Expected: exit code 0 and no new findings for the touched Python test file.

- [x] **Step 5: Update Backlog task final notes**

Update `backlog/tasks/task-2360 - Fix-Docker-single-user-WebUI-runtime-auth-bootstrap.md`:

```md
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-24 implementation: added the WebUI-local runtime config endpoint, runtime auth bootstrap precedence, Docker compose runtime auth env, setup remote-write env, setup auth regression coverage, MCP in-tree packaging smoke, and Docker quickstart docs.
Verification: [paste exact commands and pass/fail results from Steps 1-4].
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Docker single-user WebUI auth now comes from runtime container env through a local-only Next.js runtime config endpoint. Runtime auth takes precedence over stale baked public env keys without overwriting manual credentials. Docker compose now enables authenticated setup writes from the WebUI container and docs describe `NEXT_PUBLIC_X_API_KEY` as advanced/static-build compatibility rather than the quickstart path. Active MCP Unified packaging is verified as in-tree under `tldw_Server_API`.
<!-- SECTION:FINAL_SUMMARY:END -->
```

Mark each acceptance criterion complete after verification:

```md
- [x] #1 Runtime-auth WebUI bootstrap works without requiring Docker users to bake `NEXT_PUBLIC_X_API_KEY` into the WebUI image.
- [x] #2 Runtime auth takes precedence over stale build-time public env auth and does not overwrite user-managed credentials.
- [x] #3 Docker single-user compose enables authenticated setup writes from the WebUI container via `TLDW_SETUP_ALLOW_REMOTE=1`.
- [x] #4 Docker WebUI compose passes runtime auth env to the WebUI service while preserving loopback host port binding.
- [x] #5 Setup onboarding write calls remain authenticated; no unauthenticated `noAuth` regression is introduced.
- [x] #6 Docs clarify runtime auth bootstrap as the Docker quickstart default and `NEXT_PUBLIC_X_API_KEY` as advanced/static-build compatibility.
- [x] #7 Stale root `mcp_unified` Docker guidance is handled with branch-accurate verification rather than an unconditional nonexistent package copy.
- [x] #8 Focused tests or verification cover runtime endpoint guards, bootstrap precedence, compose wiring, and active MCP package/import shape.
```

- [x] **Step 6: Commit final task updates**

```bash
git add "backlog/tasks/task-2360 - Fix-Docker-single-user-WebUI-runtime-auth-bootstrap.md"
git commit -m "chore: finalize docker webui runtime auth task"
```

---

## Self-Review

### Spec Coverage

- Runtime-auth endpoint with loopback, forwarding-header, auth-mode, exposure-flag, and placeholder guards: Task 1.
- Runtime key precedence over stale build-time `NEXT_PUBLIC_X_API_KEY`: Task 2.
- `_app.tsx` waits for bootstrap before auth state: Task 3.
- Docker WebUI runtime env and loopback binding: Task 4.
- Backend setup remote-write flag: Task 4.
- Setup requests remain authenticated: Task 5.
- Stale root `mcp_unified` item handled by branch-accurate active packaging smoke: Task 5.
- Docs clarify runtime auth versus advanced/static `NEXT_PUBLIC_X_API_KEY`: Task 6.
- Verification and Bandit: Task 7.

### Placeholder Scan

The plan contains no forbidden placeholder markers or unspecified test commands. Each code-changing task includes concrete code snippets and exact commands.

### Type Consistency

- Runtime route response shape uses `runtimeAuth.available`, `runtimeAuth.authMode`, `runtimeAuth.apiKey`, and `networking.deploymentMode`.
- Bootstrap tests and implementation use the same `tldwRuntimeAuthMetadata` storage key and `runtimeBootstrapReady` export.
- `_app.tsx` imports the named `runtimeBootstrapReady` export introduced in Task 2.
