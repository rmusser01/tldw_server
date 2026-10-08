import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  HEADER_SHORTCUT_SELECTION_SETTING
} from "@/services/settings/ui-settings"

type GlobalWithExtensionRuntime = typeof globalThis & {
  browser?: Record<string, unknown>
  chrome?: Record<string, unknown>
}

const chromeDescriptor = Object.getOwnPropertyDescriptor(globalThis, "chrome")
const browserDescriptor = Object.getOwnPropertyDescriptor(globalThis, "browser")
const originalApiUrl = process.env.NEXT_PUBLIC_API_URL
const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
const originalXApiKey = process.env.NEXT_PUBLIC_X_API_KEY
const originalWindowLocation = window.location
let beforeManualSessionCredentialLock: (() => void) | null = null

const setWindowLocation = (href: string) => {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: new URL(href)
  })
}

const setGlobal = (key: "chrome" | "browser", value: unknown) => {
  Object.defineProperty(globalThis, key, {
    value,
    writable: true,
    configurable: true
  })
}

const restoreGlobal = (
  key: "chrome" | "browser",
  descriptor?: PropertyDescriptor
) => {
  if (descriptor) {
    Object.defineProperty(globalThis, key, descriptor)
    return
  }
  delete (globalThis as Record<string, unknown>)[key]
}

const readStoredValue = (key: string): unknown => {
  const raw = localStorage.getItem(key)
  if (raw == null) return null

  let next: unknown = raw
  while (typeof next === "string") {
    try {
      next = JSON.parse(next)
    } catch {
      break
    }
  }

  return next
}

const importAndAwaitBootstrap = async () => {
  const mod = await import("@web/extension/shims/runtime-bootstrap")
  await mod.runtimeBootstrapReady
  return mod
}

const stubCookieRuntimeFetch = ({
  bootstrapOk = true,
  probeOk = true
}: {
  bootstrapOk?: boolean
  probeOk?: boolean
} = {}) => {
  const fetchMock = vi.fn(async (
    input: RequestInfo | URL,
    _init?: RequestInit
  ) => {
    const url = String(input)
    if (url === "/api/_tldw-webui/runtime-config") {
      return {
        ok: true,
        json: async () => ({
          runtimeAuth: {
            available: true,
            authMode: "single-user",
            transport: "cookie-session"
          },
          networking: {
            deploymentMode: "quickstart",
            serverUrl: ""
          }
        })
      }
    }
    if (url === "/api/_tldw-webui/session") {
      return { ok: bootstrapOk }
    }
    if (url === "/api/v1/users/me/profile") {
      return { ok: probeOk }
    }
    throw new Error(`Unexpected request: ${url}`)
  })
  vi.stubGlobal("fetch", fetchMock)
  return fetchMock
}

describe("runtime-bootstrap chrome shim", () => {
  beforeEach(() => {
    vi.resetModules()
    localStorage.clear()
    sessionStorage.clear()
    beforeManualSessionCredentialLock = null
    vi.stubGlobal(
      "navigator",
      Object.create(window.navigator, {
        locks: {
          value: {
            request: async (_name: string, work: () => unknown) => {
              beforeManualSessionCredentialLock?.()
              return await work()
            }
          }
        }
      })
    )
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    restoreGlobal("chrome", chromeDescriptor)
    restoreGlobal("browser", browserDescriptor)
    if (originalApiUrl === undefined) {
      delete process.env.NEXT_PUBLIC_API_URL
    } else {
      process.env.NEXT_PUBLIC_API_URL = originalApiUrl
    }
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
    if (originalXApiKey === undefined) {
      delete process.env.NEXT_PUBLIC_X_API_KEY
    } else {
      process.env.NEXT_PUBLIC_X_API_KEY = originalXApiKey
    }
    Object.defineProperty(window, "location", {
      configurable: true,
      value: originalWindowLocation
    })
    vi.unstubAllGlobals()
    localStorage.clear()
    sessionStorage.clear()
  })

  it("creates browser/chrome shims when globals are absent", async () => {
    restoreGlobal("chrome", undefined)
    restoreGlobal("browser", undefined)

    await importAndAwaitBootstrap()

    const globalScope = globalThis as GlobalWithExtensionRuntime
    expect(typeof globalScope.browser?.storage).toBe("object")
    expect(typeof globalScope.chrome?.storage).toBe("object")
    expect(typeof globalScope.chrome?.runtime).toBe("object")
    expect(typeof globalScope.chrome?.storage?.local).toBe("object")
  })

  it("augments pre-existing chrome objects with missing extension APIs", async () => {
    const existingChromeGet = vi.fn()
    setGlobal("chrome", {
      app: { isInstalled: true },
      storage: {
        local: {
          get: existingChromeGet
        }
      }
    })
    restoreGlobal("browser", undefined)

    await importAndAwaitBootstrap()

    const globalScope = globalThis as GlobalWithExtensionRuntime
    const chromeGlobal = globalScope.chrome

    expect(chromeGlobal?.app).toEqual({ isInstalled: true })
    expect(chromeGlobal?.storage?.local?.get).toBe(existingChromeGet)
    expect(typeof chromeGlobal?.storage?.local?.set).toBe("function")
    expect(typeof chromeGlobal?.runtime?.getURL).toBe("function")
    expect(typeof globalScope.browser?.storage?.local?.get).toBe("function")
  })

  it("backfills document workspace into persisted web header shortcuts", async () => {
    localStorage.setItem(
      HEADER_SHORTCUT_SELECTION_SETTING.key,
      JSON.stringify(["chat", "media"])
    )

    await importAndAwaitBootstrap()

    const nextRaw = localStorage.getItem(HEADER_SHORTCUT_SELECTION_SETTING.key)
    expect(nextRaw).toBeTruthy()
    const nextSelection = JSON.parse(String(nextRaw))
    expect(Array.isArray(nextSelection)).toBe(true)
    expect(nextSelection).toContain("chat")
    expect(nextSelection).toContain("media")
    expect(nextSelection).toContain("document-workspace")
    expect(
      localStorage.getItem(
        "tldw:web-defaults:header-shortcuts-document-workspace:v1"
      )
    ).toBe("true")
  })

  it("backfills MCP Hub into persisted web header shortcuts", async () => {
    localStorage.setItem(
      HEADER_SHORTCUT_SELECTION_SETTING.key,
      JSON.stringify(["chat", "media"])
    )

    await importAndAwaitBootstrap()

    const nextRaw = localStorage.getItem(HEADER_SHORTCUT_SELECTION_SETTING.key)
    expect(nextRaw).toBeTruthy()
    const nextSelection = JSON.parse(String(nextRaw))
    expect(Array.isArray(nextSelection)).toBe(true)
    expect(nextSelection).toContain("chat")
    expect(nextSelection).toContain("media")
    expect(nextSelection).toContain("mcp-hub")
    expect(
      localStorage.getItem("tldw:web-defaults:header-shortcuts-mcp-hub:v1")
    ).toBe("true")
  })

  it("ignores a stale WebUI page-origin host in advanced mode", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
    localStorage.setItem("tldw-api-host", window.location.origin)
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "frontend-key",
        serverUrl: "http://127.0.0.1:8000"
      })
    )

    await importAndAwaitBootstrap()

    expect(localStorage.getItem("tldw-api-host")).toBe("http://127.0.0.1:8000")
    await vi.waitFor(() => {
      expect(readStoredValue("tldwServerUrl")).toBe("http://127.0.0.1:8000")

      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      expect(nextConfig.serverUrl).toBe("http://127.0.0.1:8000")
      expect(nextConfig.apiKey).toBe("frontend-key")
    })
  })

  it("preserves manual multi-user tokens without replacement runtime auth", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
    delete process.env.NEXT_PUBLIC_X_API_KEY
    delete process.env.NEXT_PUBLIC_API_BEARER
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "multi-user",
        accessToken: "user-token",
        refreshToken: "user-refresh",
        serverUrl: "http://127.0.0.1:8000/"
      })
    )

    await importAndAwaitBootstrap()

    expect(readStoredValue("tldwConfig")).toMatchObject({
      authMode: "multi-user",
      accessToken: "user-token",
      refreshToken: "user-refresh",
      serverUrl: "http://127.0.0.1:8000"
    })
  })

  it("clears manual multi-user tokens when the target server changes", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
    process.env.NEXT_PUBLIC_API_URL = "https://new-api.example.test"
    delete process.env.NEXT_PUBLIC_X_API_KEY
    delete process.env.NEXT_PUBLIC_API_BEARER
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "multi-user",
        accessToken: "old-access-token",
        refreshToken: "old-refresh-token",
        serverUrl: "https://old-api.example.test"
      })
    )

    await importAndAwaitBootstrap()

    expect(readStoredValue("tldwConfig")).toEqual(
      expect.objectContaining({
        authMode: "multi-user",
        serverUrl: "https://new-api.example.test"
      })
    )
    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(nextConfig.accessToken).toBeUndefined()
    expect(nextConfig.refreshToken).toBeUndefined()
  })

  it("keeps a fresh advanced URL hint without creating authentication configuration", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
    process.env.NEXT_PUBLIC_API_URL = "https://initial-api.example.test"
    delete process.env.NEXT_PUBLIC_X_API_KEY
    delete process.env.NEXT_PUBLIC_API_BEARER
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false }))
    await importAndAwaitBootstrap()
    expect(readStoredValue("tldwServerUrl")).toBe("https://initial-api.example.test")
    expect(localStorage.getItem("tldwConfig")).toBeNull()
  })

  it("repairs a stale env LAN host to the current browser host during bootstrap", async () => {
    process.env.NEXT_PUBLIC_API_URL = "http://192.168.5.184:8000"

    await importAndAwaitBootstrap()

    expect(localStorage.getItem("tldw-api-host")).toBe("http://localhost:8000")
    await vi.waitFor(() => {
      expect(readStoredValue("tldwServerUrl")).toBe("http://localhost:8000")

      expect(localStorage.getItem("tldwConfig")).toBeNull()
    })
  })

  it("repairs a stale explicit web host to the current browser host during bootstrap", async () => {
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
    localStorage.setItem("tldw-api-host", "http://192.168.5.186:8000")
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "frontend-key",
        serverUrl: "http://192.168.5.186:8000"
      })
    )

    await importAndAwaitBootstrap()

    expect(localStorage.getItem("tldw-api-host")).toBe("http://localhost:8000")
    await vi.waitFor(() => {
      expect(readStoredValue("tldwServerUrl")).toBe("http://localhost:8000")

      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      expect(nextConfig.serverUrl).toBe("http://localhost:8000")
      expect(nextConfig.apiKey).toBe("frontend-key")
    })
  })

  it("prefers an explicit web host over the env default when syncing config", async () => {
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
    localStorage.setItem("tldw-api-host", "http://localhost:18001")
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "frontend-key",
        serverUrl: "http://192.168.5.186:8000"
      })
    )

    await importAndAwaitBootstrap()

    expect(localStorage.getItem("tldw-api-host")).toBe("http://localhost:18001")
    await vi.waitFor(() => {
      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      expect(nextConfig.serverUrl).toBe("http://localhost:18001")
      expect(readStoredValue("tldwServerUrl")).toBe("http://localhost:18001")
    })
  })

  it("bootstraps cookie auth, probes it, and stores no runtime secret", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "stale-public-key"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "legacy-runtime-key",
        serverUrl: "http://127.0.0.1:8000"
      })
    )
    localStorage.setItem("apiKey", "legacy-bridge-key")
    localStorage.setItem(
      "tldwRuntimeAuthMetadata",
      JSON.stringify({ source: "webui-runtime", keyFingerprint: "legacy" })
    )
    sessionStorage.setItem(
      "tldwManualSessionApiKey",
      JSON.stringify({ apiKey: "ambiguous-session-key" })
    )
    const fetchMock = stubCookieRuntimeFetch()

    await importAndAwaitBootstrap()

    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/api/_tldw-webui/session",
      expect.objectContaining({ method: "POST", credentials: "include" })
    )
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "/api/v1/users/me/profile",
      expect.objectContaining({ method: "GET", credentials: "same-origin" })
    )
    expect(readStoredValue("tldwCookieSessionConfig")).toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    expect(readStoredValue("tldwConfig")).toEqual({
      authMode: "single-user",
      serverUrl: "http://127.0.0.1:8000"
    })
    expect(localStorage.getItem("apiKey")).toBeNull()
    expect(localStorage.getItem("tldwRuntimeAuthMetadata")).toBeNull()
    expect(sessionStorage.getItem("tldwManualSessionApiKey")).toBeNull()
    expect(JSON.stringify([...Array(localStorage.length)].map((_, index) => {
      const key = localStorage.key(index)
      return key ? [key, localStorage.getItem(key)] : null
    }))).not.toContain("stale-public-key")
    const { getApiKey } = await import("@web/lib/authStorage")
    expect(getApiKey()).toBeNull()

    const signals = fetchMock.mock.calls.map(([, init]) => init?.signal)
    expect(signals).toHaveLength(3)
    expect(signals.every((signal) => signal instanceof AbortSignal)).toBe(true)
    expect(new Set(signals)).toHaveLength(3)
  })

  it.each([
    ["runtime config", "/api/_tldw-webui/runtime-config"],
    ["session bootstrap", "/api/_tldw-webui/session"],
    ["profile probe", "/api/v1/users/me/profile"]
  ])(
    "bounds a never-settling %s request and preserves manual configuration",
    async (_label, stalledUrl) => {
      vi.useFakeTimers()
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      const existing = {
        authMode: "single-user",
        apiKey: "manual-key",
        credentialSource: "manual",
        apiKeyPersistence: "device",
        apiKeyServerOrigin: "https://remote.example.test",
        serverUrl: "https://remote.example.test"
      }
      localStorage.setItem("tldwConfig", JSON.stringify(existing))
      localStorage.setItem("apiKey", "legacy-key")
      localStorage.setItem(
        "tldwCookieSessionConfig",
        JSON.stringify({
          authMode: "single-user",
          authSource: "cookie-session",
          serverUrl: window.location.origin
        })
      )

      const fetchMock = vi.fn(
        async (input: RequestInfo | URL, _init?: RequestInit) => {
          const url = String(input)
          if (url === stalledUrl) {
            return await new Promise<Response>(() => undefined)
          }
          if (url === "/api/_tldw-webui/runtime-config") {
            return {
              ok: true,
              json: async () => ({
                runtimeAuth: {
                  available: true,
                  authMode: "single-user",
                  transport: "cookie-session"
                }
              })
            } as Response
          }
          return { ok: true } as Response
        }
      )
      vi.stubGlobal("fetch", fetchMock)

      const mod = await import("@web/extension/shims/runtime-bootstrap")
      await vi.advanceTimersByTimeAsync(0)
      expect(fetchMock).toHaveBeenCalledWith(
        stalledUrl,
        expect.objectContaining({ signal: expect.any(AbortSignal) })
      )

      let settled = false
      void mod.runtimeBootstrapReady.then(() => {
        settled = true
      })
      await vi.advanceTimersByTimeAsync(7_999)
      await Promise.resolve()
      expect(settled).toBe(false)

      await vi.advanceTimersByTimeAsync(1)
      await mod.runtimeBootstrapReady

      expect(readStoredValue("tldwConfig")).toEqual(existing)
      expect(localStorage.getItem("apiKey")).toBe("legacy-key")
      expect(readStoredValue("tldwCookieSessionConfig")).toBeNull()
    }
  )

  it("preserves manual config and legacy slots when session bootstrap fails", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const existing = {
      authMode: "single-user",
      apiKey: "manual-key",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://remote.example.test",
      serverUrl: "https://remote.example.test"
    }
    localStorage.setItem("tldwConfig", JSON.stringify(existing))
    localStorage.setItem("apiKey", "legacy-key")
    const fetchMock = stubCookieRuntimeFetch({ bootstrapOk: false })

    await importAndAwaitBootstrap()

    expect(readStoredValue("tldwConfig")).toEqual(existing)
    expect(localStorage.getItem("apiKey")).toBe("legacy-key")
    expect(fetchMock).not.toHaveBeenCalledWith(
      "/api/v1/users/me/profile",
      expect.anything()
    )
  })

  it.each([
    [
      "device persistence",
      {
        authMode: "single-user",
        credentialSource: "manual",
        apiKeyPersistence: "device",
        apiKeyServerOrigin: "https://remote.example.test",
        serverUrl: "https://remote.example.test"
      }
    ],
    [
      "cookie auth",
      {
        authMode: "single-user",
        authSource: "cookie-session",
        serverUrl: "https://remote.example.test"
      }
    ],
    [
      "incomplete manual metadata",
      {
        authMode: "single-user",
        credentialSource: "manual",
        serverUrl: "https://remote.example.test"
      }
    ],
    [
      "different-origin metadata",
      {
        authMode: "single-user",
        credentialSource: "manual",
        apiKeyPersistence: "session",
        apiKeyServerOrigin: "https://other.example.test",
        serverUrl: "https://remote.example.test"
      }
    ],
    [
      "noncanonical origin metadata",
      {
        authMode: "single-user",
        credentialSource: "manual",
        apiKeyPersistence: "session",
        apiKeyServerOrigin: "https://remote.example.test/path",
        serverUrl: "https://remote.example.test"
      }
    ]
  ])("scrubs a session secret beside %s", async (_label, config) => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    localStorage.setItem("tldwConfig", JSON.stringify(config))
    sessionStorage.setItem(
      "tldwManualSessionApiKey",
      JSON.stringify({
        credentialSource: "manual",
        apiKeyPersistence: "session",
        apiKeyServerOrigin: "https://remote.example.test",
        apiKey: "manual-session-key"
      })
    )
    stubCookieRuntimeFetch()

    await importAndAwaitBootstrap()

    expect(sessionStorage.getItem("tldwManualSessionApiKey")).toBeNull()
  })

  it("scrubs a complete session secret for the active quickstart origin", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const origin = window.location.origin
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        credentialSource: "manual",
        apiKeyPersistence: "session",
        apiKeyServerOrigin: origin,
        serverUrl: origin
      })
    )
    sessionStorage.setItem(
      "tldwManualSessionApiKey",
      JSON.stringify({
        credentialSource: "manual",
        apiKeyPersistence: "session",
        apiKeyServerOrigin: origin,
        apiKey: "same-origin-session-key"
      })
    )
    stubCookieRuntimeFetch()

    await importAndAwaitBootstrap()

    expect(sessionStorage.getItem("tldwManualSessionApiKey")).toBeNull()
  })

  it("preserves a complete manual session connection beside the active cookie session", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        credentialSource: "manual",
        apiKeyPersistence: "session",
        apiKeyServerOrigin: "https://remote.example.test",
        serverUrl: "https://remote.example.test/api"
      })
    )
    sessionStorage.setItem(
      "tldwManualSessionApiKey",
      JSON.stringify({
        credentialSource: "manual",
        apiKeyPersistence: "session",
        apiKeyServerOrigin: "https://remote.example.test",
        apiKey: "manual-session-key"
      })
    )
    stubCookieRuntimeFetch()

    await importAndAwaitBootstrap()

    const manualConfig = {
      authMode: "single-user",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://remote.example.test",
      serverUrl: "https://remote.example.test/api"
    }
    expect(readStoredValue("tldwConfig")).toEqual(manualConfig)
    expect(readStoredValue("tldwCookieSessionConfig")).toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })
    expect(sessionStorage.getItem("tldwManualSessionApiKey")).toContain(
      "manual-session-key"
    )
    const { resolveManualCredential } = await import(
      "@/services/tldw/single-user-credential"
    )
    await expect(
      resolveManualCredential(manualConfig, {
        session: {
          get: async (key: string) => {
            const raw = sessionStorage.getItem(key)
            return raw ? JSON.parse(raw) : null
          },
          set: async () => undefined,
          remove: async () => undefined
        }
      })
    ).resolves.toBe("manual-session-key")
    const { getApiKey } = await import("@web/lib/authStorage")
    expect(getApiKey()).toBeNull()

    vi.resetModules()
    stubCookieRuntimeFetch()
    await importAndAwaitBootstrap()

    expect(readStoredValue("tldwConfig")).toEqual(manualConfig)
  })

  it("preserves a newly written matching session record while bootstrap waits for the credential lock", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        credentialSource: "manual",
        apiKeyPersistence: "session",
        apiKeyServerOrigin: "https://stale.example.test",
        serverUrl: "https://stale.example.test"
      })
    )
    sessionStorage.setItem(
      "tldwManualSessionApiKey",
      JSON.stringify({
        credentialSource: "manual",
        apiKeyPersistence: "session",
        apiKeyServerOrigin: "https://stale.example.test",
        apiKey: "stale-session-key"
      })
    )
    beforeManualSessionCredentialLock = () => {
      localStorage.setItem(
        "tldwConfig",
        JSON.stringify({
          authMode: "single-user",
          credentialSource: "manual",
          apiKeyPersistence: "session",
          apiKeyServerOrigin: "https://remote.example.test",
          serverUrl: "https://remote.example.test"
        })
      )
      sessionStorage.setItem(
        "tldwManualSessionApiKey",
        JSON.stringify({
          credentialSource: "manual",
          apiKeyPersistence: "session",
          apiKeyServerOrigin: "https://remote.example.test",
          apiKey: "new-session-key"
        })
      )
    }
    stubCookieRuntimeFetch()

    await importAndAwaitBootstrap()

    expect(sessionStorage.getItem("tldwManualSessionApiKey")).toContain(
      "new-session-key"
    )
  })

  it("preserves manual config and legacy slots when the cookie probe fails", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const existing = {
      authMode: "single-user",
      apiKey: "manual-key",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://remote.example.test",
      serverUrl: "https://remote.example.test"
    }
    localStorage.setItem("tldwConfig", JSON.stringify(existing))
    localStorage.setItem("apiKey", "legacy-key")
    stubCookieRuntimeFetch({ probeOk: false })

    await importAndAwaitBootstrap()

    expect(readStoredValue("tldwConfig")).toEqual(existing)
    expect(localStorage.getItem("apiKey")).toBe("legacy-key")
  })

  it("does not fall back to public runtime key storage when capability is unavailable", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "public-runtime-key"
    const existing = {
      authMode: "single-user",
      apiKey: "manual-key",
      serverUrl: "https://remote.example.test"
    }
    localStorage.setItem("tldwConfig", JSON.stringify(existing))
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ runtimeAuth: { available: false } })
      }))
    )

    await importAndAwaitBootstrap()

    expect(readStoredValue("tldwConfig")).toEqual(existing)
    expect(JSON.stringify(readStoredValue("tldwConfig"))).not.toContain(
      "public-runtime-key"
    )
  })

  it("invalidates a stale cookie marker in memory when persistent removal fails", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const manualConfig = {
      authMode: "single-user" as const,
      apiKey: "manual-device-key",
      credentialSource: "manual" as const,
      apiKeyPersistence: "device" as const,
      apiKeyServerOrigin: "https://remote.example.test",
      serverUrl: "https://remote.example.test"
    }
    localStorage.setItem("tldwConfig", JSON.stringify(manualConfig))
    localStorage.setItem(
      "tldwCookieSessionConfig",
      JSON.stringify({
        authMode: "single-user",
        authSource: "cookie-session",
        serverUrl: window.location.origin
      })
    )
    const { Storage } = await import("@plasmohq/storage")
    vi.spyOn(Storage.prototype, "remove").mockRejectedValueOnce(
      new Error("storage unavailable")
    )
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ runtimeAuth: { available: false } })
      }))
    )

    const { TldwApiClient } = await import("@/services/tldw/TldwApiClient")
    const client = new TldwApiClient()
    expect(await client.getConfig()).toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: window.location.origin
    })

    await importAndAwaitBootstrap()
    const config = await client.getConfig()

    expect(config).toEqual(manualConfig)
    expect(readStoredValue("tldwConfig")).toEqual(manualConfig)
    expect(readStoredValue("tldwCookieSessionConfig")).not.toBeNull()
  })

  it.each([
    ["runtime config", "/api/_tldw-webui/runtime-config"],
    ["session bootstrap", "/api/_tldw-webui/session"],
    ["profile probe", "/api/v1/users/me/profile"]
  ])(
    "recovers the same page after a cold %s timeout",
    async (_label, stalledUrl) => {
      vi.useFakeTimers()
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      process.env.NEXT_PUBLIC_API_URL = "/api"
      setWindowLocation("http://localhost:19769/chat")
      const fetchMock = stubCookieRuntimeFetch()
      const healthyFetch = fetchMock.getMockImplementation()!
      let cold = true
      fetchMock.mockImplementation(async (input, init) =>
        cold && String(input) === stalledUrl
          ? await new Promise<Response>(() => undefined)
          : await healthyFetch(input, init)
      )
      const mod = await import("@web/extension/shims/runtime-bootstrap")
      await vi.advanceTimersByTimeAsync(8_000)
      await mod.runtimeBootstrapReady
      expect(readStoredValue("tldwCookieSessionConfig")).toBeNull()
      cold = false
      const { TldwApiClient } = await import("@/services/tldw/TldwApiClient")
      const client = new TldwApiClient()
      await client.initialize()
      expect(await client.getConfig()).toEqual({
        authMode: "single-user",
        authSource: "cookie-session",
        serverUrl: "http://localhost:19769"
      })
      expect(
        fetchMock.mock.calls.filter(
          ([url]) => String(url) === "/api/_tldw-webui/runtime-config"
        )
      ).toHaveLength(2)
      const count = fetchMock.mock.calls.length
      await client.initialize()
      expect(fetchMock).toHaveBeenCalledTimes(count)
    }
  )

  it("connection checking preserves failed bootstrap recovery for account Retry", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_API_URL = "/api"
    setWindowLocation("http://localhost:19769/chat")
    const fetchMock = stubCookieRuntimeFetch()
    const healthyFetch = fetchMock.getMockImplementation()!
    fetchMock.mockImplementation(async () => {
      throw new TypeError("offline")
    })
    await importAndAwaitBootstrap()
    const { useConnectionStore } = await import("@/store/connection")
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(readStoredValue("tldwConfig")).toBeNull()
    expect(useConnectionStore.getState().state).toMatchObject({
      isConnected: false,
      serverUrl: "http://localhost:19769",
      configStep: "auth"
    })

    fetchMock.mockImplementation(healthyFetch)
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    expect(await tldwClient.getConfig()).toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: "http://localhost:19769"
    })
  })

  it.each([
    ["serialized null", null],
    ["URL-only setup hint", { serverUrl: "http://localhost:19769" }],
    [
      "legacy cookie hint",
      {
        authMode: "single-user",
        authSource: "cookie-session",
        serverUrl: "http://localhost:19769"
      }
    ]
  ])("recovers after a failed bootstrap with %s", async (_label, stored) => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_API_URL = "/api"
    setWindowLocation("http://localhost:19769/chat")
    localStorage.setItem("tldwConfig", JSON.stringify(stored))
    const fetchMock = stubCookieRuntimeFetch()
    const healthyFetch = fetchMock.getMockImplementation()!
    fetchMock.mockImplementation(async () => {
      throw new TypeError("offline")
    })
    await importAndAwaitBootstrap()
    fetchMock.mockImplementation(healthyFetch)
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    expect(await tldwClient.getConfig()).toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: "http://localhost:19769"
    })
    expect(readStoredValue("tldwCookieSessionConfig")).toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: "http://localhost:19769"
    })
    expect(readStoredValue("tldwConfig")).toEqual(stored)
  })

  it.each([
    [
      "manual metadata",
      { serverUrl: "http://localhost:19769", credentialSource: "manual" }
    ],
    [
      "manual key",
      {
        serverUrl: "http://localhost:19769",
        apiKey: "manual-key",
        credentialSource: "manual",
        apiKeyPersistence: "device",
        apiKeyServerOrigin: "http://localhost:19769"
      }
    ],
    [
      "multi-user tokens",
      {
        serverUrl: "http://localhost:19769",
        authMode: "multi-user",
        accessToken: "manual-token"
      }
    ],
    ["foreign URL", { serverUrl: "https://other.example" }],
    [
      "foreign cookie",
      {
        serverUrl: "https://other.example",
        authMode: "single-user",
        authSource: "cookie-session"
      }
    ],
    [
      "unknown field",
      { serverUrl: "http://localhost:19769", futureAuthority: true }
    ],
    [
      "manual source",
      { serverUrl: "http://localhost:19769", authSource: "manual" }
    ],
    ["path URL", { serverUrl: "http://localhost:19769/chat" }],
    ["array", []],
    ["primitive", "legacy hint"],
    ...[
      "apiKey",
      "apiBearer",
      "accessToken",
      "refreshToken",
      "credentialSource",
      "apiKeyPersistence",
      "apiKeyServerOrigin"
    ].map((key) => [
      `empty ${key}`,
      { serverUrl: "http://localhost:19769", [key]: "" }
    ])
  ])(
    "does not recover over %s after a failed bootstrap",
    async (_label, stored) => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      process.env.NEXT_PUBLIC_API_URL = "/api"
      setWindowLocation("http://localhost:19769/chat")
      localStorage.setItem("tldwConfig", JSON.stringify(stored))
      const fetchMock = stubCookieRuntimeFetch()
      const healthyFetch = fetchMock.getMockImplementation()!
      fetchMock.mockImplementation(async () => {
        throw new TypeError("offline")
      })
      await importAndAwaitBootstrap()
      const count = fetchMock.mock.calls.length
      fetchMock.mockImplementation(healthyFetch)
      const { tldwClient } = await import("@/services/tldw/TldwApiClient")
      await tldwClient.getConfig()
      expect(fetchMock).toHaveBeenCalledTimes(count)
      expect(readStoredValue("tldwCookieSessionConfig")).toBeNull()
    }
  )

  const beginFailedPageRecovery = async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_API_URL = "/api"
    setWindowLocation("http://localhost:19769/chat")
    const fetchMock = stubCookieRuntimeFetch()
    const healthyFetch = fetchMock.getMockImplementation()!
    fetchMock.mockImplementation(async () => {
      throw new TypeError("offline")
    })
    await importAndAwaitBootstrap()
    const { TldwApiClient } = await import("@/services/tldw/TldwApiClient")
    const client = new TldwApiClient()
    let enter!: () => void
    let release!: () => void
    const entered = new Promise<void>((resolve) => {
      enter = resolve
    })
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    fetchMock.mockImplementation(async (input, init) => {
      if (String(input) === "/api/_tldw-webui/runtime-config") {
        enter()
        await held
      }
      return await healthyFetch(input, init)
    })
    return { client, fetchMock, entered, release }
  }

  it("explicit same-value URL configuration retires held recovery", async () => {
    const { client, fetchMock, entered, release } =
      await beginFailedPageRecovery()
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({ serverUrl: "http://localhost:19769" })
    )
    const pending = client.initialize()
    await entered
    await client.updateConfig({ serverUrl: "http://localhost:19769" })
    release()
    await pending
    expect(readStoredValue("tldwCookieSessionConfig")).toBeNull()
    expect(
      fetchMock.mock.calls.filter(
        ([url]) => String(url) === "/api/_tldw-webui/session"
      )
    ).toHaveLength(0)
  })

  it("shares recovery between simultaneous client initializers", async () => {
    const { client, fetchMock, entered, release } =
      await beginFailedPageRecovery()
    const first = client.initialize()
    const second = client.initialize()
    await entered
    release()
    await Promise.all([first, second])
    expect(
      fetchMock.mock.calls.filter(
        ([url]) => String(url) === "/api/_tldw-webui/session"
      )
    ).toHaveLength(1)
    expect((await client.getConfig())?.authSource).toBe("cookie-session")
  })

  it.each(["logout", "manual ABA", "foreign account ABA"])(
    "retires held recovery on %s",
    async (boundary) => {
      const { client, fetchMock, entered, release } =
        await beginFailedPageRecovery()
      const pending = client.initialize()
      await entered
      if (boundary === "logout") await client.clearCookieSingleUserSession()
      else if (boundary === "manual ABA") {
        await client.updateConfig({
          authMode: "single-user",
          serverUrl: "https://new.example.test"
        })
        await client.updateConfig({
          authMode: "single-user",
          serverUrl: "http://localhost:19769"
        })
      } else {
        window.dispatchEvent(
          new StorageEvent("storage", {
            key: "tldwConfig",
            oldValue: null,
            newValue: JSON.stringify({
              serverUrl: "https://other.example.test"
            })
          })
        )
        window.dispatchEvent(
          new StorageEvent("storage", {
            key: "tldwConfig",
            oldValue: JSON.stringify({
              serverUrl: "https://other.example.test"
            }),
            newValue: null
          })
        )
      }
      release()
      await pending
      expect(readStoredValue("tldwCookieSessionConfig")).toBeNull()
      expect(
        fetchMock.mock.calls.filter(
          ([url]) => String(url) === "/api/_tldw-webui/session"
        )
      ).toHaveLength(0)
    }
  )

  it("requires acknowledged cookie storage before activating recovered auth", async () => {
    const { client, entered, release } = await beginFailedPageRecovery()
    const { Storage } = await import("@plasmohq/storage")
    const set = Storage.prototype.set
    vi.spyOn(Storage.prototype, "set").mockImplementation(
      async function (key, value) {
        if (key === "tldwCookieSessionConfig")
          throw new Error("storage unavailable")
        return await set.call(this, key, value)
      }
    )
    const pending = client.initialize()
    await entered
    release()
    await pending
    expect(await client.getConfig()).toBeNull()
  })

  it.each(["healthy", "failed"])(
    "does not reactivate after logout during the initial %s bootstrap",
    async (outcome) => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      process.env.NEXT_PUBLIC_API_URL = "/api"
      setWindowLocation("http://localhost:19769/chat")
      const fetchMock = stubCookieRuntimeFetch()
      const healthyFetch = fetchMock.getMockImplementation()!
      let enter!: () => void
      let release!: () => void
      const entered = new Promise<void>((resolve) => {
        enter = resolve
      })
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      fetchMock.mockImplementation(async (input, init) => {
        if (String(input) === "/api/_tldw-webui/runtime-config") {
          enter()
          await held
          if (outcome === "failed") throw new TypeError("offline")
        }
        return await healthyFetch(input, init)
      })
      const mod = await import("@web/extension/shims/runtime-bootstrap")
      await entered
      const { TldwApiClient } = await import("@/services/tldw/TldwApiClient")
      const client = new TldwApiClient()
      await client.clearCookieSingleUserSession()
      release()
      await mod.runtimeBootstrapReady
      fetchMock.mockImplementation(healthyFetch)
      await client.initialize()
      expect(await client.getConfig()).toBeNull()
      expect(
        fetchMock.mock.calls.filter(
          ([url]) => String(url) === "/api/_tldw-webui/session"
        )
      ).toHaveLength(0)
    }
  )

  it.each(["logout", "manual replacement"])(
    "ignores a cookie marker write acknowledged after %s",
    async (boundary) => {
      const { client, entered, release } = await beginFailedPageRecovery()
      const { Storage } = await import("@plasmohq/storage")
      const set = Storage.prototype.set
      let enterWrite!: () => void
      let releaseWrite!: () => void
      const enteredWrite = new Promise<void>((resolve) => {
        enterWrite = resolve
      })
      const heldWrite = new Promise<void>((resolve) => {
        releaseWrite = resolve
      })
      vi.spyOn(Storage.prototype, "set").mockImplementation(
        async function (key, value) {
          if (key === "tldwCookieSessionConfig") {
            enterWrite()
            await heldWrite
          }
          return await set.call(this, key, value)
        }
      )
      const pending = client.initialize()
      await entered
      release()
      await enteredWrite
      if (boundary === "logout") await client.clearCookieSingleUserSession()
      else
        await client.updateConfig({
          authMode: "single-user",
          serverUrl: "https://new.example.test"
        })
      releaseWrite()
      await pending
      const { getEffectiveStoredTldwConfig } =
        await import("@web/lib/authStorage")
      expect(getEffectiveStoredTldwConfig()).toEqual(await client.getConfig())
      expect(getEffectiveStoredTldwConfig()?.authSource).not.toBe(
        "cookie-session"
      )
    }
  )

  it("a retired credential scrub preserves a newer acknowledged manual session key", async () => {
    const { client, entered, release } = await beginFailedPageRecovery()
    let firstLock = true
    Object.defineProperty(navigator.locks, "request", {
      value:
        // Declared lock boundary: a newer credential operation finishes before the waiting scrub enters.
        async (_name: string, work: () => unknown) => {
          if (firstLock) {
            firstLock = false
            await client.clearCookieSingleUserSession()
            await client.saveManualSingleUserCredential({
              serverUrl: "http://localhost:19769",
              apiKey: "new-session-key",
              persistence: "session"
            })
          }
          return await work()
        }
    })
    const pending = client.initialize()
    await entered
    release()
    await pending
    expect(
      JSON.parse(sessionStorage.getItem("tldwManualSessionApiKey") || "null")
    ).toMatchObject({
      apiKey: "new-session-key",
      apiKeyServerOrigin: "http://localhost:19769"
    })
    expect((await client.getConfig())?.apiKey).toBe("new-session-key")
  })

  it("actual account Retry recovers configuration and preserves an unowned draft", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_API_URL = "/api"
    setWindowLocation("http://localhost:19769/chat")
    const fetchMock = stubCookieRuntimeFetch()
    const healthyFetch = fetchMock.getMockImplementation()!
    fetchMock.mockImplementation(async () => {
      throw new TypeError("offline")
    })
    await importAndAwaitBootstrap()
    const React = await import("react")
    const { render, fireEvent, screen, waitFor, cleanup } =
      await import("@testing-library/react/pure")
    const { useChatDraftOwner } = await import("@/hooks/useChatDraftOwner")
    const clearDraft = vi.fn()
    const initialText = "Mira, explore the observatory.  \nKeep this draft."
    const OwnerRetry = () => {
      const [draft, setDraft] = React.useState(initialText)
      const owner = useChatDraftOwner(() => {
        clearDraft()
        setDraft("")
      })
      return React.createElement(
        "div",
        null,
        React.createElement("textarea", {
          "aria-label": "Draft",
          value: draft,
          onChange: (e) => setDraft(e.currentTarget.value)
        }),
        React.createElement(
          "output",
          { "aria-label": "Account state" },
          owner.verificationState
        ),
        React.createElement(
          "button",
          { onClick: owner.retryVerification },
          "Retry account verification"
        )
      )
    }
    try {
      render(React.createElement(OwnerRetry))
      await waitFor(() =>
        expect(screen.getByLabelText("Account state")).toHaveTextContent(
          "error"
        )
      )
      fetchMock.mockImplementation(healthyFetch)
      fireEvent.click(
        screen.getByRole("button", { name: "Retry account verification" })
      )
      await waitFor(() =>
        expect(screen.getByLabelText("Account state")).toHaveTextContent(
          "ready"
        )
      )
      expect(screen.getByLabelText("Draft")).toHaveValue(initialText)
      expect(clearDraft).not.toHaveBeenCalled()
      expect(readStoredValue("tldwCookieSessionConfig")).not.toBeNull()
    } finally {
      cleanup()
    }
  })

  it("does not fetch runtime config for extension protocols", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    setWindowLocation("chrome-extension://extension-id/options.html")
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)

    await importAndAwaitBootstrap()

    expect(fetchMock).not.toHaveBeenCalled()
  })
})
