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

const stubRuntimeConfigFetch = (apiKey: string) => {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    json: async () => ({
      runtimeAuth: {
        available: true,
        authMode: "single-user",
        apiKey
      },
      networking: {
        deploymentMode: "quickstart",
        serverUrl: ""
      }
    })
  }))
  vi.stubGlobal("fetch", fetchMock)
  return fetchMock
}

describe("runtime-bootstrap chrome shim", () => {
  beforeEach(() => {
    vi.resetModules()
    localStorage.clear()
  })

  afterEach(() => {
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

  it("canonicalizes quickstart webui bootstrap to the current page origin", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    delete process.env.NEXT_PUBLIC_API_URL
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "frontend-key",
        serverUrl: "http://127.0.0.1:8000"
      })
    )

    await importAndAwaitBootstrap()

    expect(localStorage.getItem("tldw-api-host")).toBe(window.location.origin)
    await vi.waitFor(() => {
      expect(readStoredValue("tldwServerUrl")).toBe(window.location.origin)

      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      expect(nextConfig.serverUrl).toBe(window.location.origin)
      expect(nextConfig.apiKey).toBeUndefined()
      expect(localStorage.getItem("tldwConfig")).not.toContain("frontend-key")
    })
  })

  it("seeds first-run quickstart config from the public single-user API key", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "quickstart-api-key"
    delete process.env.NEXT_PUBLIC_API_URL

    await import("@web/extension/shims/runtime-bootstrap")

    expect(localStorage.getItem("tldw-api-host")).toBe(window.location.origin)
    await vi.waitFor(() => {
      expect(readStoredValue("tldwServerUrl")).toBe(window.location.origin)

      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      expect(nextConfig).toMatchObject({
        authMode: "single-user",
        serverUrl: window.location.origin
      })
      expect(nextConfig.apiKey).toBeUndefined()
      expect(localStorage.getItem("tldwConfig")).not.toContain("quickstart-api-key")
    })
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

    await import("@web/extension/shims/runtime-bootstrap")

    expect(localStorage.getItem("tldw-api-host")).toBe("http://127.0.0.1:8000")
    await vi.waitFor(() => {
      expect(readStoredValue("tldwServerUrl")).toBe("http://127.0.0.1:8000")

      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      expect(nextConfig.serverUrl).toBe("http://127.0.0.1:8000")
      expect(nextConfig.apiKey).toBeUndefined()
      expect(localStorage.getItem("tldwConfig")).not.toContain("frontend-key")
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
      expect(nextConfig.apiKey).toBeUndefined()
      expect(localStorage.getItem("tldwConfig")).not.toContain("frontend-key")
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

  it("seeds runtime auth before stale build-time env auth", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "stale-build-key"
    const fetchMock = stubRuntimeConfigFetch("runtime-key")

    await importAndAwaitBootstrap()
    const { getApiKey } = await import("@web/lib/authStorage")

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    const metadata = readStoredValue("tldwRuntimeAuthMetadata") as Record<string, unknown>

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/_tldw-webui/runtime-config",
      expect.objectContaining({
        credentials: "same-origin",
        cache: "no-store"
      })
    )
    expect(getApiKey()).toBe("runtime-key")
    expect(nextConfig.authMode).toBe("single-user")
    expect(nextConfig.apiKey).toBeUndefined()
    expect(nextConfig.serverUrl).toBe(window.location.origin)
    expect(localStorage.getItem("tldwConfig")).not.toContain("runtime-key")
    expect(readStoredValue("tldwServerUrl")).toBe(window.location.origin)
    expect(metadata.source).toBe("webui-runtime")
    expect(metadata.version).toBe(1)
    expect(typeof metadata.keyFingerprint).toBe("string")
  })

  it("scrubs a manual stored key while runtime auth wins request precedence", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "stale-build-key"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "manual-user-key",
        serverUrl: "http://127.0.0.1:8000"
      })
    )
    stubRuntimeConfigFetch("runtime-key")

    await importAndAwaitBootstrap()
    const { getApiKey } = await import("@web/lib/authStorage")

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(getApiKey()).toBe("runtime-key")
    expect(nextConfig.authMode).toBe("single-user")
    expect(nextConfig.apiKey).toBeUndefined()
    expect(nextConfig.serverUrl).toBe(window.location.origin)
    expect(localStorage.getItem("tldwConfig")).not.toContain("manual-user-key")
    expect(readStoredValue("tldwServerUrl")).toBe(window.location.origin)
    expect(readStoredValue("tldwRuntimeAuthMetadata")).toBeNull()
  })

  it("uses runtime auth for shared requests after scrubbing a manual single-user key", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "manual-user-key",
        serverUrl: "http://127.0.0.1:8000"
      })
    )
    stubRuntimeConfigFetch("runtime-key")

    await importAndAwaitBootstrap()
    const { tldwRequest } = await import("@/services/tldw/request-core")
    const requestFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    )

    await tldwRequest(
      {
        path: "/api/v1/config/docs-info",
        method: "GET"
      },
      {
        getConfig: async () =>
          readStoredValue("tldwConfig") as Record<string, unknown>,
        fetchFn: requestFetch
      }
    )

    expect(requestFetch).toHaveBeenCalledWith(
      "/api/v1/config/docs-info",
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-API-KEY": "runtime-key"
        })
      })
    )
  })

  it("scrubs manual multi-user credentials while runtime auth wins request precedence", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "multi-user",
        accessToken: "manual-token",
        serverUrl: "http://127.0.0.1:8000"
      })
    )
    stubRuntimeConfigFetch("runtime-key")

    await importAndAwaitBootstrap()
    const { getApiKey } = await import("@web/lib/authStorage")

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(getApiKey()).toBe("runtime-key")
    expect(nextConfig.authMode).toBe("single-user")
    expect(nextConfig.accessToken).toBeUndefined()
    expect(nextConfig.apiKey).toBeUndefined()
    expect(nextConfig.serverUrl).toBe(window.location.origin)
    expect(localStorage.getItem("tldwConfig")).not.toContain("manual-token")
    expect(readStoredValue("tldwServerUrl")).toBe(window.location.origin)
    expect(readStoredValue("tldwRuntimeAuthMetadata")).toBeNull()
  })

  it("uses runtime auth for shared requests after scrubbing manual multi-user credentials", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "multi-user",
        accessToken: "manual-token",
        serverUrl: "http://127.0.0.1:8000"
      })
    )
    stubRuntimeConfigFetch("runtime-key")

    await importAndAwaitBootstrap()
    const { tldwRequest } = await import("@/services/tldw/request-core")
    const requestFetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    )

    await tldwRequest(
      {
        path: "/api/v1/config/docs-info",
        method: "GET"
      },
      {
        getConfig: async () =>
          readStoredValue("tldwConfig") as Record<string, unknown>,
        fetchFn: requestFetch
      }
    )

    expect(requestFetch).toHaveBeenCalledWith(
      "/api/v1/config/docs-info",
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-API-KEY": "runtime-key"
        })
      })
    )
    const requestHeaders = requestFetch.mock.calls[0]?.[1]?.headers as Record<
      string,
      string
    >
    expect(requestHeaders.Authorization).toBeUndefined()
  })

  it("does not mark a matching manual key as runtime-owned without prior metadata", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "runtime-key",
        serverUrl: "http://127.0.0.1:8000"
      })
    )
    stubRuntimeConfigFetch("runtime-key")

    await importAndAwaitBootstrap()
    const { getApiKey } = await import("@web/lib/authStorage")

    let nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(getApiKey()).toBe("runtime-key")
    expect(nextConfig.apiKey).toBeUndefined()
    expect(nextConfig.serverUrl).toBe(window.location.origin)
    expect(localStorage.getItem("tldwConfig")).not.toContain("runtime-key")
    expect(readStoredValue("tldwRuntimeAuthMetadata")).toBeNull()

    vi.resetModules()
    stubRuntimeConfigFetch("rotated-runtime-key")
    await importAndAwaitBootstrap()

    const authStorage = await import("@web/lib/authStorage")
    nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(authStorage.getApiKey()).toBe("rotated-runtime-key")
    expect(nextConfig.apiKey).toBeUndefined()
    expect(localStorage.getItem("tldwConfig")).not.toContain("rotated-runtime-key")
    expect(readStoredValue("tldwRuntimeAuthMetadata")).toMatchObject({
      source: "webui-runtime",
      version: 1,
      authMode: "single-user"
    })
  })

  it.each(["CHANGE_ME_TO_SECURE_API_KEY", "test-key"])(
    "replaces persisted placeholder key %s with runtime auth metadata",
    async (placeholderKey) => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      localStorage.setItem(
        "tldwConfig",
        JSON.stringify({
          authMode: "single-user",
          apiKey: placeholderKey,
          serverUrl: "http://127.0.0.1:8000"
        })
      )
      stubRuntimeConfigFetch("runtime-key")

      await importAndAwaitBootstrap()

      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      const metadata = readStoredValue("tldwRuntimeAuthMetadata") as Record<string, unknown>
      expect(nextConfig.apiKey).toBeUndefined()
      expect(localStorage.getItem("tldwConfig")).not.toContain("runtime-key")
      expect(metadata).toMatchObject({
        source: "webui-runtime",
        version: 1,
        authMode: "single-user"
      })
    }
  )

  it("replaces a previous runtime-owned key when the runtime key changes", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    stubRuntimeConfigFetch("old-runtime-key")

    await importAndAwaitBootstrap()
    const oldMetadata = readStoredValue("tldwRuntimeAuthMetadata") as Record<string, unknown>

    vi.resetModules()
    stubRuntimeConfigFetch("new-runtime-key")
    await importAndAwaitBootstrap()

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    const newMetadata = readStoredValue("tldwRuntimeAuthMetadata") as Record<string, unknown>

    expect(nextConfig.apiKey).toBeUndefined()
    expect(localStorage.getItem("tldwConfig")).not.toContain("new-runtime-key")
    expect(newMetadata.source).toBe("webui-runtime")
    expect(newMetadata.keyFingerprint).not.toBe(oldMetadata.keyFingerprint)
  })

  it("writes runtime auth metadata without storing a duplicate secret", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    stubRuntimeConfigFetch("runtime-key")

    await importAndAwaitBootstrap()

    const rawMetadata = localStorage.getItem("tldwRuntimeAuthMetadata")
    const metadata = readStoredValue("tldwRuntimeAuthMetadata") as Record<string, unknown>

    expect(metadata).toMatchObject({
      source: "webui-runtime",
      version: 1,
      authMode: "single-user"
    })
    expect(rawMetadata).not.toContain("runtime-key")
  })

  it("falls back to existing env bootstrap when runtime config fetch fails", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.NEXT_PUBLIC_X_API_KEY = "env-key"
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("offline")
      })
    )

    await importAndAwaitBootstrap()
    const { getApiKey } = await import("@web/lib/authStorage")

    const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
    expect(getApiKey()).toBe("env-key")
    expect(nextConfig.apiKey).toBeUndefined()
    expect(localStorage.getItem("tldwConfig")).not.toContain("env-key")
    expect(readStoredValue("tldwRuntimeAuthMetadata")).toBeNull()
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
