import { describe, expect, it } from "vitest"

import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  MANUAL_SESSION_KEY,
  clearManualCredentials,
  normalizeServerOrigin,
  resolveManualCredential,
  toPersistedTldwConfig,
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

describe("manual single-user credential policy", () => {
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
  })

  it("normalizes only HTTP server origins", () => {
    expect(normalizeServerOrigin("https://API.example.test/path")).toBe(
      "https://api.example.test"
    )
    expect(normalizeServerOrigin("ftp://api.example.test")).toBeNull()
    expect(normalizeServerOrigin("not a URL")).toBeNull()
  })
})
