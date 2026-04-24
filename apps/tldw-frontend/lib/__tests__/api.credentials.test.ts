/** @vitest-environment jsdom */

import { afterEach, beforeEach, describe, expect, it } from "vitest"

import {
  clearRuntimeAuth,
  hasEnvApiAuth,
  setRuntimeApiBearer,
  setRuntimeApiKey
} from "../authStorage"

async function loadShouldIncludeBrowserCredentials() {
  process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
  const apiModule = await import("../api")
  return apiModule.shouldIncludeBrowserCredentials
}

async function loadApiHelpers() {
  process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
  return import("../api")
}

describe("shouldIncludeBrowserCredentials", () => {
  const originalApiKey = process.env.NEXT_PUBLIC_X_API_KEY
  const originalApiBearer = process.env.NEXT_PUBLIC_API_BEARER

  beforeEach(() => {
    localStorage.clear()
    clearRuntimeAuth()
    delete process.env.NEXT_PUBLIC_X_API_KEY
    delete process.env.NEXT_PUBLIC_API_BEARER
  })

  afterEach(() => {
    localStorage.clear()
    clearRuntimeAuth()
    if (originalApiKey === undefined) {
      delete process.env.NEXT_PUBLIC_X_API_KEY
    } else {
      process.env.NEXT_PUBLIC_X_API_KEY = originalApiKey
    }
    if (originalApiBearer === undefined) {
      delete process.env.NEXT_PUBLIC_API_BEARER
    } else {
      process.env.NEXT_PUBLIC_API_BEARER = originalApiBearer
    }
  })

  it("disables browser credentials when single-user API key auth is active", () => {
    setRuntimeApiKey("test-api-key")

    return loadShouldIncludeBrowserCredentials().then((shouldIncludeBrowserCredentials) => {
      expect(shouldIncludeBrowserCredentials()).toBe(false)
    })
  })

  it("disables browser credentials when explicit bearer auth is active without a JWT session", () => {
    setRuntimeApiBearer("api-bearer")

    return loadShouldIncludeBrowserCredentials().then((shouldIncludeBrowserCredentials) => {
      expect(shouldIncludeBrowserCredentials()).toBe(false)
    })
  })

  it("keeps browser credentials enabled when a JWT session is active", () => {
    setRuntimeApiKey("test-api-key")
    localStorage.setItem("access_token", "jwt-token")

    return loadShouldIncludeBrowserCredentials().then((shouldIncludeBrowserCredentials) => {
      expect(shouldIncludeBrowserCredentials()).toBe(true)
    })
  })

  it("keeps browser credentials enabled before login so CSRF/session cookies can be used", () => {
    return loadShouldIncludeBrowserCredentials().then((shouldIncludeBrowserCredentials) => {
      expect(shouldIncludeBrowserCredentials()).toBe(true)
    })
  })

  it("normalizes env API auth detection through a shared helper", () => {
    expect(hasEnvApiAuth()).toBe(false)

    process.env.NEXT_PUBLIC_X_API_KEY = "   "
    process.env.NEXT_PUBLIC_API_BEARER = "  Bearer env-token  "

    expect(hasEnvApiAuth()).toBe(true)
  })

  it("reuses stored single-user api keys from localStorage config for browser requests", () => {
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        authMode: "single-user",
        apiKey: "stored-api-key"
      })
    )

    return loadApiHelpers().then((apiModule) => {
      expect(apiModule.shouldIncludeBrowserCredentials()).toBe(false)
      expect(apiModule.buildAuthHeaders("GET")).toMatchObject({
        "X-API-KEY": "stored-api-key"
      })
    })
  })

  it("treats whitespace-only env auth values as unset", () => {
    process.env.NEXT_PUBLIC_X_API_KEY = "   "
    process.env.NEXT_PUBLIC_API_BEARER = "  "

    return loadApiHelpers().then((apiModule) => {
      expect(apiModule.hasEnvAuthConfigured()).toBe(false)
    })
  })

  it("keeps auth recovery routes out of the unauthorized login redirect loop", () => {
    return loadApiHelpers().then((apiModule) => {
      expect(apiModule.shouldRedirectUnauthorizedToLogin("/login")).toBe(false)
      expect(apiModule.shouldRedirectUnauthorizedToLogin("/settings")).toBe(false)
      expect(apiModule.shouldRedirectUnauthorizedToLogin("/settings/tldw")).toBe(false)
      expect(apiModule.shouldRedirectUnauthorizedToLogin("/settings/health/")).toBe(false)
      expect(apiModule.shouldRedirectUnauthorizedToLogin("/auth/reset-password")).toBe(false)
      expect(apiModule.shouldRedirectUnauthorizedToLogin("/chat")).toBe(true)
    })
  })
})
