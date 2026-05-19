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
    localStorage.clear()
  })

  it("creates browser/chrome shims when globals are absent", async () => {
    restoreGlobal("chrome", undefined)
    restoreGlobal("browser", undefined)

    await import("@web/extension/shims/runtime-bootstrap")

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

    await import("@web/extension/shims/runtime-bootstrap")

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

    await import("@web/extension/shims/runtime-bootstrap")

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

    await import("@web/extension/shims/runtime-bootstrap")

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

    await import("@web/extension/shims/runtime-bootstrap")

    expect(localStorage.getItem("tldw-api-host")).toBe(window.location.origin)
    await vi.waitFor(() => {
      expect(readStoredValue("tldwServerUrl")).toBe(window.location.origin)

      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      expect(nextConfig.serverUrl).toBe(window.location.origin)
      expect(nextConfig.apiKey).toBe("frontend-key")
    })
  })

  it("repairs a stale env LAN host to the current browser host during bootstrap", async () => {
    process.env.NEXT_PUBLIC_API_URL = "http://192.168.5.184:8000"

    await import("@web/extension/shims/runtime-bootstrap")

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

    await import("@web/extension/shims/runtime-bootstrap")

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

    await import("@web/extension/shims/runtime-bootstrap")

    expect(localStorage.getItem("tldw-api-host")).toBe("http://localhost:18001")
    await vi.waitFor(() => {
      const nextConfig = readStoredValue("tldwConfig") as Record<string, unknown>
      expect(nextConfig.serverUrl).toBe("http://localhost:18001")
      expect(readStoredValue("tldwServerUrl")).toBe("http://localhost:18001")
    })
  })
})
