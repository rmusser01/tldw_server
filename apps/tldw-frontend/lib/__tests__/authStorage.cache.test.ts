/** @vitest-environment jsdom */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  __clearAuthStorageCacheForTests,
  clearRuntimeAuth,
  getApiBearer,
  getApiKey,
  getSessionAccessToken
} from "../authStorage"
import { AUTH_CREDENTIALS_CHANGED_EVENT } from "@web/lib/auth-events"
import { REFRESH_ROTATION_KEY } from "@/services/tldw/single-user-credential"

const originalApiKey = process.env.NEXT_PUBLIC_X_API_KEY
const originalApiBearer = process.env.NEXT_PUBLIC_API_BEARER
const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

const clearEnv = () => {
  delete process.env.NEXT_PUBLIC_X_API_KEY
  delete process.env.NEXT_PUBLIC_API_BEARER
  delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
}

const restoreEnv = () => {
  if (originalApiKey === undefined) delete process.env.NEXT_PUBLIC_X_API_KEY
  else process.env.NEXT_PUBLIC_X_API_KEY = originalApiKey
  if (originalApiBearer === undefined) delete process.env.NEXT_PUBLIC_API_BEARER
  else process.env.NEXT_PUBLIC_API_BEARER = originalApiBearer
  if (originalDeploymentMode === undefined) {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  } else {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
  }
}

const singleUserConfigWithKey = (apiKey: string) => ({
  authMode: "single-user",
  serverUrl: "https://api.example.test/v1",
  apiKey,
  credentialSource: "manual",
  apiKeyPersistence: "device",
  apiKeyServerOrigin: "https://api.example.test"
})

async function loadApiHelpers() {
  process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
  return import("../api")
}

describe("authStorage read caching", () => {
  beforeEach(() => {
    clearEnv()
    localStorage.clear()
    sessionStorage.clear()
    clearRuntimeAuth()
    __clearAuthStorageCacheForTests()
  })

  afterEach(() => {
    localStorage.clear()
    sessionStorage.clear()
    clearRuntimeAuth()
    restoreEnv()
  })

  it("reads stored auth state once across a 50-call header burst", async () => {
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "multi-user",
        authSource: "manual",
        serverUrl: "https://api.example.test/v1",
        orgId: 7,
        accessToken: "access-0",
        refreshToken: "refresh-0"
      })
    )
    const apiModule = await loadApiHelpers()
    const getItemSpy = vi.spyOn(Storage.prototype, "getItem")

    apiModule.buildAuthHeaders("GET")
    const readsAfterFirstCall = getItemSpy.mock.calls.length
    expect(readsAfterFirstCall).toBeGreaterThan(0)

    for (let i = 0; i < 49; i += 1) apiModule.buildAuthHeaders("GET")
    expect(getItemSpy.mock.calls.length).toBe(readsAfterFirstCall)
  })

  it("reads the effective config once across sequential getter calls", async () => {
    await loadApiHelpers()
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "multi-user",
        authSource: "manual",
        serverUrl: "https://api.example.test/v1",
        orgId: 7,
        accessToken: "access-0",
        refreshToken: "refresh-0"
      })
    )
    const getItemSpy = vi.spyOn(Storage.prototype, "getItem")

    const { getEffectiveStoredTldwConfig } = await import("../authStorage")
    getEffectiveStoredTldwConfig()
    const readsAfterFirstCall = getItemSpy.mock.calls.length
    expect(readsAfterFirstCall).toBeGreaterThan(0)

    for (let i = 0; i < 49; i += 1) getEffectiveStoredTldwConfig()
    expect(getItemSpy.mock.calls.length).toBe(readsAfterFirstCall)
  })

  it("drops the cache when a same-tab config write reaches the storage-instance watch", async () => {
    localStorage.setItem("tldwConfig", JSON.stringify(singleUserConfigWithKey("key-a")))
    expect(getApiKey()).toBe("key-a")

    const { Storage } = await import("@plasmohq/storage")
    await new Storage({ area: "local" }).set(
      "tldwConfig",
      singleUserConfigWithKey("key-b")
    )
    expect(getApiKey()).toBe("key-b")
  })

  it("drops the cache on the cross-tab storage event", () => {
    localStorage.setItem("tldwConfig", JSON.stringify(singleUserConfigWithKey("key-a")))
    expect(getApiKey()).toBe("key-a")

    localStorage.setItem("tldwConfig", JSON.stringify(singleUserConfigWithKey("key-b")))
    window.dispatchEvent(
      new StorageEvent("storage", { key: "tldwConfig", storageArea: window.localStorage })
    )
    expect(getApiKey()).toBe("key-b")
  })

  it("drops the cache on the tldw:config-updated event", () => {
    localStorage.setItem("tldwConfig", JSON.stringify(singleUserConfigWithKey("key-a")))
    expect(getApiKey()).toBe("key-a")

    localStorage.setItem("tldwConfig", JSON.stringify(singleUserConfigWithKey("key-b")))
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
    expect(getApiKey()).toBe("key-b")
  })

  it("drops the cache on auth-credentials-changed for the legacy session token", () => {
    localStorage.setItem("access_token", "legacy-session-token")
    expect(getSessionAccessToken()).toBe("legacy-session-token")

    localStorage.removeItem("access_token")
    window.dispatchEvent(
      new CustomEvent(AUTH_CREDENTIALS_CHANGED_EVENT, { detail: { authenticated: false } })
    )
    expect(getSessionAccessToken()).toBeNull()
  })

  it("drops the cache on window focus", () => {
    localStorage.setItem("tldwConfig", JSON.stringify(singleUserConfigWithKey("key-a")))
    expect(getApiKey()).toBe("key-a")

    localStorage.setItem("tldwConfig", JSON.stringify(singleUserConfigWithKey("key-b")))
    window.dispatchEvent(new Event("focus"))
    expect(getApiKey()).toBe("key-b")
  })

  it("drops the cache when runtime auth is cleared", () => {
    localStorage.setItem("tldwConfig", JSON.stringify(singleUserConfigWithKey("key-a")))
    expect(getApiKey()).toBe("key-a")

    clearRuntimeAuth()
    localStorage.setItem("tldwConfig", JSON.stringify(singleUserConfigWithKey("key-b")))
    expect(getApiKey()).toBe("key-b")
  })

  it("re-reads a rotated credential after a same-tab rotation write", async () => {
    const base = {
      authMode: "multi-user",
      authSource: "manual",
      serverUrl: "https://api.example.test/v1",
      orgId: 7,
      accessToken: "access-0",
      refreshToken: "refresh-0"
    }
    const rotationRecord = (accessToken: string, refreshToken: string) => ({
      version: 1,
      serverUrl: base.serverUrl,
      authMode: base.authMode,
      authSource: base.authSource,
      orgId: base.orgId,
      sourceAccessToken: "access-0",
      sourceRefreshToken: "refresh-0",
      accessToken,
      refreshToken
    })
    localStorage.setItem("tldwConfig", JSON.stringify(base))
    localStorage.setItem(
      REFRESH_ROTATION_KEY,
      JSON.stringify(rotationRecord("access-1", "refresh-1"))
    )
    expect(getApiBearer()).toBe("access-1")

    const { Storage } = await import("@plasmohq/storage")
    await new Storage({ area: "local" }).set(
      REFRESH_ROTATION_KEY,
      rotationRecord("access-2", "refresh-2")
    )
    expect(getApiBearer()).toBe("access-2")
  })
})
