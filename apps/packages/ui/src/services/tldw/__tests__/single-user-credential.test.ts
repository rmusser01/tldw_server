import { describe, expect, it } from "vitest"

import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  MANUAL_SESSION_KEY,
  REFRESH_ROTATION_KEY,
  clearManualCredentials,
  hasNewerCurrentAccessToken,
  hasNewerCurrentRefreshRotation,
  normalizeServerOrigin,
  resolveEffectiveTldwConfig,
  resolveManualCredential,
  storeRefreshRotationIfCurrent,
  toPersistedTldwConfig,
  waitForNewerCurrentAccessToken,
  type CredentialStorage
} from "@/services/tldw/single-user-credential"

class MemoryStorage implements CredentialStorage {
  values = new Map<string, unknown>()

  async get<T>(key: string): Promise<T | undefined> {
    return this.values.get(key) as T | undefined
  }

  async set<T>(key: string, value: T): Promise<void> {
    this.values.set(key, value)
  }

  async remove(key: string): Promise<void> {
    this.values.delete(key)
  }
}

const deviceConfig = {
  authMode: "single-user",
  serverUrl: "https://api.example.test/v1",
  apiKey: "secret",
  credentialSource: "manual",
  apiKeyPersistence: "device",
  apiKeyServerOrigin: "https://api.example.test"
} satisfies TldwConfig

const multiUserConfig = {
  authMode: "multi-user",
  authSource: "manual",
  serverUrl: "https://api.example.test/v1",
  orgId: 7,
  accessToken: "access-0",
  refreshToken: "refresh-0"
} satisfies TldwConfig
const jwtForUser = (userId: string | number): string =>
  `header.${btoa(JSON.stringify({ sub: String(userId) }))}.signature`

describe("manual single-user credential policy", () => {
  it("hydrates an exact-origin session key without persisting it", async () => {
    const persistent = new MemoryStorage()
    const session = new MemoryStorage()
    const config: TldwConfig = {
      authMode: "single-user",
      serverUrl: "https://api.example.test/path",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    }
    await persistent.set("tldwConfig", config)
    await session.set(MANUAL_SESSION_KEY, {
      apiKey: "session-secret",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    })
    await persistent.set(REFRESH_ROTATION_KEY, {
      accessToken: "stale-access",
      refreshToken: "stale-refresh"
    })
    await persistent.set(REFRESH_ROTATION_KEY, {
      accessToken: "stale-access",
      refreshToken: "stale-refresh"
    })

    await expect(
      resolveEffectiveTldwConfig({ persistent, session })
    ).resolves.toEqual({ ...config, apiKey: "session-secret" })
    await expect(persistent.get("tldwConfig")).resolves.toEqual(config)

    await session.set(MANUAL_SESSION_KEY, {
      apiKey: "other-secret",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://other.test"
    })
    await expect(
      resolveEffectiveTldwConfig({ persistent, session })
    ).resolves.toEqual(config)
  })

  it("prefers only an exact-origin cookie session and removes explicit auth", async () => {
    const persistent = new MemoryStorage()
    const session = new MemoryStorage()
    await persistent.set("tldwConfig", deviceConfig)
    const cookieSession: TldwConfig = {
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: "https://api.example.test",
      apiKey: "stale-api-key",
      apiBearer: "stale-api-bearer",
      accessToken: "stale-access-token",
      refreshToken: "stale-refresh-token",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://api.example.test"
    }

    await expect(
      resolveEffectiveTldwConfig(
        { persistent, session },
        { cookieSession, expectedCookieOrigin: "https://api.example.test" }
      )
    ).resolves.toEqual({
      authMode: "single-user",
      authSource: "cookie-session",
      serverUrl: "https://api.example.test"
    })

    await expect(
      resolveEffectiveTldwConfig(
        { persistent, session },
        {
          cookieSession: {
            ...cookieSession,
            serverUrl: "https://other.test"
          },
          expectedCookieOrigin: "https://api.example.test"
        }
      )
    ).resolves.toEqual(deviceConfig)
  })

  it("fails closed when credential storage is unreadable", async () => {
    const unreadable: CredentialStorage = {
      get: async () => {
        throw new Error("unreadable")
      },
      set: async () => undefined,
      remove: async () => undefined
    }
    const session = new MemoryStorage()

    await expect(
      resolveEffectiveTldwConfig({ persistent: unreadable, session })
    ).resolves.toBeNull()

    const persistent = new MemoryStorage()
    const config: TldwConfig = {
      authMode: "single-user",
      serverUrl: "https://api.example.test",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    }
    await persistent.set("tldwConfig", config)
    await expect(
      resolveEffectiveTldwConfig({ persistent, session: unreadable })
    ).resolves.toEqual(config)
  })

  it("accepts a complete device credential only for its exact origin", async () => {
    const session = new MemoryStorage()

    await expect(resolveManualCredential(deviceConfig, { session })).resolves.toBe(
      "secret"
    )
    await expect(
      resolveManualCredential(
        { ...deviceConfig, serverUrl: "https://other.test" },
        { session }
      )
    ).resolves.toBeNull()
  })

  it("accepts a complete session credential only for its exact origin", async () => {
    const session = new MemoryStorage()
    await session.set(MANUAL_SESSION_KEY, {
      apiKey: "session-secret",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    })
    const config: TldwConfig = {
      authMode: "single-user",
      serverUrl: "https://api.example.test/path",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    }

    await expect(resolveManualCredential(config, { session })).resolves.toBe(
      "session-secret"
    )
    await expect(
      resolveManualCredential(
        { ...config, serverUrl: "https://other.test" },
        { session }
      )
    ).resolves.toBeNull()
  })

  it("rejects incomplete or reclassified session records", async () => {
    const session = new MemoryStorage()
    const config: TldwConfig = {
      authMode: "single-user",
      serverUrl: "https://api.example.test",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    }

    await session.set(MANUAL_SESSION_KEY, {
      apiKey: "runtime-secret",
      credentialSource: "cookie-session",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    })
    await expect(resolveManualCredential(config, { session })).resolves.toBeNull()
  })

  it("strips runtime, session, and incomplete keys from persisted tldwConfig", () => {
    expect(
      toPersistedTldwConfig({
        ...deviceConfig,
        apiKey: "runtime",
        credentialSource: "cookie-session"
      })
    ).not.toHaveProperty("apiKey")
    expect(
      toPersistedTldwConfig({
        ...deviceConfig,
        apiKey: "session",
        apiKeyPersistence: "session"
      })
    ).not.toHaveProperty("apiKey")
    expect(
      toPersistedTldwConfig({
        authMode: "single-user",
        serverUrl: "https://api.example.test",
        apiKey: "ambiguous"
      })
    ).not.toHaveProperty("apiKey")
    expect(toPersistedTldwConfig(deviceConfig)).toHaveProperty("apiKey", "secret")
  })

  it("clears device and session secrets while preserving connection metadata", async () => {
    const persistent = new MemoryStorage()
    const session = new MemoryStorage()
    await persistent.set("tldwConfig", deviceConfig)
    await session.set(MANUAL_SESSION_KEY, {
      apiKey: "session-secret",
      credentialSource: "manual",
      apiKeyPersistence: "session",
      apiKeyServerOrigin: "https://api.example.test"
    })

    await clearManualCredentials(persistent, session)

    expect(await persistent.get<TldwConfig>("tldwConfig")).toEqual({
      authMode: "single-user",
      serverUrl: "https://api.example.test/v1"
    })
    expect(await session.get(MANUAL_SESSION_KEY)).toBeUndefined()
    expect(await persistent.get(REFRESH_ROTATION_KEY)).toBeUndefined()
  })

  it("surfaces a persistent read failure after attempting session clearing", async () => {
    const persistent: CredentialStorage = {
      get: async () => {
        throw new Error("persistent read unavailable")
      },
      set: async () => undefined,
      remove: async () => undefined
    }
    const session = new MemoryStorage()
    await session.set(MANUAL_SESSION_KEY, { apiKey: "session-secret" })

    await expect(clearManualCredentials(persistent, session)).rejects.toThrow(
      "persistent read unavailable"
    )
    expect(await session.get(MANUAL_SESSION_KEY)).toBeUndefined()
  })

  it("surfaces a persistent sanitize failure after clearing session storage", async () => {
    const persistent: CredentialStorage = {
      get: async () => deviceConfig,
      set: async () => {
        throw new Error("persistent sanitize unavailable")
      },
      remove: async () => undefined
    }
    const session = new MemoryStorage()
    await session.set(MANUAL_SESSION_KEY, { apiKey: "session-secret" })

    await expect(clearManualCredentials(persistent, session)).rejects.toThrow(
      "persistent sanitize unavailable"
    )
    expect(await session.get(MANUAL_SESSION_KEY)).toBeUndefined()
  })

  it("surfaces session removal failure after sanitizing persistent storage", async () => {
    const persistent = new MemoryStorage()
    await persistent.set("tldwConfig", deviceConfig)
    const session: CredentialStorage = {
      get: async () => ({ apiKey: "session-secret" }),
      set: async () => undefined,
      remove: async () => {
        throw new Error("session remove unavailable")
      }
    }

    await expect(clearManualCredentials(persistent, session)).rejects.toThrow(
      "session remove unavailable"
    )
    expect(await persistent.get<TldwConfig>("tldwConfig")).toEqual({
      authMode: "single-user",
      serverUrl: "https://api.example.test/v1"
    })
  })

  it("normalizes only HTTP server origins", () => {
    expect(normalizeServerOrigin("https://API.example.test/path")).toBe(
      "https://api.example.test"
    )
    expect(normalizeServerOrigin("ftp://api.example.test")).toBeNull()
    expect(normalizeServerOrigin("not a URL")).toBeNull()
  })
})

describe("scoped refresh-token rotation", () => {
  it("applies a guarded rotation when advanced transport leaves serverUrl unset", async () => {
    const persistent = new MemoryStorage()
    const session = new MemoryStorage()
    const advancedConfig = {
      authMode: "multi-user" as const,
      accessToken: "access-0",
      refreshToken: "refresh-0"
    }
    await persistent.set("tldwConfig", advancedConfig)

    await expect(storeRefreshRotationIfCurrent(
      persistent,
      advancedConfig as never,
      "refresh-0",
      { accessToken: "access-1", refreshToken: "refresh-1" }
    )).resolves.toBe(true)

    await expect(
      resolveEffectiveTldwConfig({ persistent, session })
    ).resolves.toEqual({
      ...advancedConfig,
      accessToken: "access-1",
      refreshToken: "refresh-1"
    })
  })

  it("overlays a rotation only while its target and source token still match", async () => {
    const persistent = new MemoryStorage()
    const session = new MemoryStorage()
    await persistent.set("tldwConfig", multiUserConfig)
    await persistent.set(REFRESH_ROTATION_KEY, {
      version: 1,
      serverUrl: multiUserConfig.serverUrl,
      authMode: multiUserConfig.authMode,
      authSource: multiUserConfig.authSource,
      orgId: multiUserConfig.orgId,
      sourceAccessToken: "access-0",
      sourceRefreshToken: "refresh-0",
      accessToken: "access-1",
      refreshToken: "refresh-1"
    })

    await expect(
      resolveEffectiveTldwConfig({ persistent, session })
    ).resolves.toEqual({
      ...multiUserConfig,
      accessToken: "access-1",
      refreshToken: "refresh-1"
    })
    await expect(persistent.get("tldwConfig")).resolves.toEqual(multiUserConfig)

    const replacementAccount = {
      ...multiUserConfig,
      accessToken: "other-access",
      refreshToken: "other-refresh"
    }
    await persistent.set("tldwConfig", replacementAccount)
    await expect(
      resolveEffectiveTldwConfig({ persistent, session })
    ).resolves.toEqual(replacementAccount)

    const replacementTarget = {
      ...multiUserConfig,
      serverUrl: "https://other.example.test",
      accessToken: "target-access",
      refreshToken: "refresh-0"
    }
    await persistent.set("tldwConfig", replacementTarget)
    await expect(
      resolveEffectiveTldwConfig({ persistent, session })
    ).resolves.toEqual(replacementTarget)
  })

  it("preserves the original source token across consecutive rotations", async () => {
    const persistent = new MemoryStorage()
    const session = new MemoryStorage()
    const target = {
      serverUrl: multiUserConfig.serverUrl,
      authMode: multiUserConfig.authMode,
      authSource: multiUserConfig.authSource,
      orgId: multiUserConfig.orgId
    }
    await persistent.set("tldwConfig", multiUserConfig)

    await expect(storeRefreshRotationIfCurrent(
      persistent,
      target,
      "refresh-0",
      { accessToken: "access-1", refreshToken: "refresh-1" }
    )).resolves.toBe(true)
    await expect(hasNewerCurrentRefreshRotation(
      persistent,
      target,
      "access-0"
    )).resolves.toBe(true)
    await expect(storeRefreshRotationIfCurrent(
      persistent,
      target,
      "refresh-1",
      { accessToken: "access-2", refreshToken: "refresh-2" }
    )).resolves.toBe(true)
    await expect(hasNewerCurrentRefreshRotation(
      persistent,
      target,
      "access-1"
    )).resolves.toBe(true)
    await expect(hasNewerCurrentRefreshRotation(
      persistent,
      target,
      "access-2"
    )).resolves.toBe(false)

    await expect(persistent.get(REFRESH_ROTATION_KEY)).resolves.toMatchObject({
      sourceAccessToken: "access-0",
      sourceRefreshToken: "refresh-0",
      accessToken: "access-2",
      refreshToken: "refresh-2"
    })
    await expect(
      resolveEffectiveTldwConfig({ persistent, session })
    ).resolves.toEqual({
      ...multiUserConfig,
      accessToken: "access-2",
      refreshToken: "refresh-2"
    })

    await persistent.set("tldwConfig", {
      ...multiUserConfig,
      accessToken: "other-access",
      refreshToken: "other-refresh"
    })
    await expect(hasNewerCurrentRefreshRotation(
      persistent,
      target,
      "access-2"
    )).resolves.toBe(false)
  })

  it("allows another refresh when the current access token expires without refresh rotation", async () => {
    const persistent = new MemoryStorage()
    const target = {
      serverUrl: multiUserConfig.serverUrl,
      authMode: multiUserConfig.authMode,
      authSource: multiUserConfig.authSource,
      orgId: multiUserConfig.orgId
    }
    await persistent.set("tldwConfig", multiUserConfig)
    await expect(storeRefreshRotationIfCurrent(
      persistent,
      target,
      "refresh-0",
      { accessToken: "access-1", refreshToken: "refresh-0" }
    )).resolves.toBe(true)

    await expect(hasNewerCurrentRefreshRotation(
      persistent,
      target,
      "access-1"
    )).resolves.toBe(false)
  })

  it("observes a newer rotation written by another request context", async () => {
    const persistent = new MemoryStorage()
    const target = {
      serverUrl: multiUserConfig.serverUrl,
      authMode: multiUserConfig.authMode,
      authSource: multiUserConfig.authSource,
      orgId: multiUserConfig.orgId
    }
    await persistent.set("tldwConfig", multiUserConfig)
    const waiting = waitForNewerCurrentAccessToken(
      persistent,
      target,
      "access-0",
      { timeoutMs: 100, pollIntervalMs: 1 }
    )
    setTimeout(() => {
      void storeRefreshRotationIfCurrent(
        persistent,
        target,
        "refresh-0",
        { accessToken: "access-1", refreshToken: "refresh-1" }
      )
    }, 5)

    await expect(waiting).resolves.toBe(true)
  })

  it("observes newer raw credentials written by an ordinary refresh", async () => {
    const persistent = new MemoryStorage()
    const capturedAccess = jwtForUser(42)
    const target = {
      serverUrl: multiUserConfig.serverUrl,
      authMode: multiUserConfig.authMode,
      authSource: multiUserConfig.authSource,
      orgId: multiUserConfig.orgId
    }
    await persistent.set("tldwConfig", {
      ...multiUserConfig,
      accessToken: `${jwtForUser(42)}-rotated`,
      refreshToken: "ordinary-refresh"
    })

    await expect(hasNewerCurrentAccessToken(
      persistent,
      target,
      capturedAccess
    )).resolves.toBe(true)
  })

  it("does not treat unverifiable raw token drift as a refresh winner", async () => {
    const persistent = new MemoryStorage()
    await persistent.set("tldwConfig", {
      ...multiUserConfig,
      accessToken: "other-account-access",
      refreshToken: "other-account-refresh"
    })

    await expect(hasNewerCurrentAccessToken(
      persistent,
      multiUserConfig,
      "access-0"
    )).resolves.toBe(false)
  })

  it("cannot overwrite a target change racing the rotation-record write", async () => {
    const replacement = {
      ...multiUserConfig,
      serverUrl: "https://replacement.example.test",
      accessToken: "replacement-access",
      refreshToken: "replacement-refresh"
    }
    class RacingStorage extends MemoryStorage {
      override async set<T>(key: string, value: T): Promise<void> {
        if (key === REFRESH_ROTATION_KEY) {
          this.values.set("tldwConfig", replacement)
        }
        await super.set(key, value)
      }
    }
    const persistent = new RacingStorage()
    const session = new MemoryStorage()
    await persistent.set("tldwConfig", multiUserConfig)

    await expect(storeRefreshRotationIfCurrent(
      persistent,
      {
        serverUrl: multiUserConfig.serverUrl,
        authMode: multiUserConfig.authMode,
        authSource: multiUserConfig.authSource,
        orgId: multiUserConfig.orgId
      },
      "refresh-0",
      { accessToken: "access-1", refreshToken: "refresh-1" }
    )).resolves.toBe(true)

    await expect(persistent.get("tldwConfig")).resolves.toEqual(replacement)
    await expect(
      resolveEffectiveTldwConfig({ persistent, session })
    ).resolves.toEqual(replacement)
  })

  it("makes a rotation inert when raw access changes during the record write", async () => {
    const replacementAccount = {
      ...multiUserConfig,
      accessToken: "account-b-access"
    }
    class RacingStorage extends MemoryStorage {
      override async set<T>(key: string, value: T): Promise<void> {
        if (key === REFRESH_ROTATION_KEY) {
          this.values.set("tldwConfig", replacementAccount)
        }
        await super.set(key, value)
      }
    }
    const persistent = new RacingStorage()
    const session = new MemoryStorage()
    await persistent.set("tldwConfig", multiUserConfig)

    await expect(storeRefreshRotationIfCurrent(
      persistent,
      { ...multiUserConfig, accessToken: "access-0" },
      "refresh-0",
      { accessToken: "access-1", refreshToken: "refresh-1" }
    )).resolves.toBe(true)

    await expect(
      resolveEffectiveTldwConfig({ persistent, session })
    ).resolves.toEqual(replacementAccount)
    await expect(persistent.get(REFRESH_ROTATION_KEY)).resolves.toMatchObject({
      sourceAccessToken: "access-0"
    })
  })

  it("rejects same-target account drift even when the refresh token is unchanged", async () => {
    const persistent = new MemoryStorage()
    await persistent.set("tldwConfig", {
      ...multiUserConfig,
      accessToken: "account-b-access"
    })

    await expect(storeRefreshRotationIfCurrent(
      persistent,
      {
        serverUrl: multiUserConfig.serverUrl,
        authMode: multiUserConfig.authMode,
        authSource: multiUserConfig.authSource,
        orgId: multiUserConfig.orgId,
        accessToken: "account-a-access"
      },
      "refresh-0",
      { accessToken: "access-1", refreshToken: "refresh-1" }
    )).resolves.toBe(false)
    await expect(persistent.get(REFRESH_ROTATION_KEY)).resolves.toBeUndefined()
  })
})
