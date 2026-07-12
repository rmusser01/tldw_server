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

  it("repairs a stale env LAN host to the current browser host during bootstrap", async () => {
    process.env.NEXT_PUBLIC_API_URL = "http://192.168.5.184:8000"

    await importAndAwaitBootstrap()

    expect(localStorage.getItem("tldw-api-host")).toBe("http://localhost:8000")
    await vi.waitFor(() => {
      expect(readStoredValue("tldwServerUrl")).toBe("http://localhost:8000")

      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      expect(nextConfig.serverUrl).toBe("http://localhost:8000")
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

  it("does not fetch runtime config for extension protocols", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    setWindowLocation("chrome-extension://extension-id/options.html")
    const fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)

    await importAndAwaitBootstrap()

    expect(fetchMock).not.toHaveBeenCalled()
  })
})
