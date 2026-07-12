import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  testApiKey: vi.fn(),
  saveManualSingleUserCredential: vi.fn(),
  login: vi.fn(),
  verifyMagicLink: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    saveManualSingleUserCredential: mocks.saveManualSingleUserCredential
  }
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    testApiKey: mocks.testApiKey,
    login: mocks.login,
    verifyMagicLink: mocks.verifyMagicLink
  }
}))

import {
  categorizeConnectionError,
  commitManualServerTransition,
  validateApiKey
} from "../validation"

describe("onboarding validation error classification", () => {
  beforeEach(() => {
    mocks.testApiKey.mockReset()
    mocks.saveManualSingleUserCredential.mockReset()
    mocks.login.mockReset()
    mocks.verifyMagicLink.mockReset()
    mocks.saveManualSingleUserCredential.mockResolvedValue("device")
  })

  it("classifies browser fetch network errors as cors/network blocked", () => {
    const kind = categorizeConnectionError(
      0,
      "NetworkError when attempting to fetch resource. (GET /api/v1/users/me/profile)"
    )

    expect(kind).toBe("cors_blocked")
  })

  it("returns a blocking network diagnostic for API key validation network failures", async () => {
    mocks.testApiKey.mockRejectedValueOnce(
      new Error("NetworkError when attempting to fetch resource. (GET /api/v1/users/me/profile)")
    )

    const result = await validateApiKey(
      "http://192.168.5.186:8000",
      "real-key",
      ((key: string, fallback: string) => fallback || key) as any
    )

    expect(result.success).toBe(false)
    expect(result.errorKind).toBe("cors_blocked")
  })

  it("classifies failed fetches against loopback setup URLs as refused", async () => {
    mocks.testApiKey.mockRejectedValueOnce(new Error("Failed to fetch"))

    const result = await validateApiKey(
      "http://127.0.0.1:65535",
      "real-key",
      ((key: string, fallback: string) => fallback || key) as any
    )

    expect(result.success).toBe(false)
    expect(result.errorKind).toBe("refused")
  })

  it("keeps invalid key classification for explicit auth failures", async () => {
    mocks.testApiKey.mockResolvedValueOnce(false)

    const result = await validateApiKey(
      "http://192.168.5.186:8000",
      "bad-key",
      ((key: string, fallback: string) => fallback || key) as any
    )

    expect(result.success).toBe(false)
    expect(result.errorKind).toBe("auth_invalid")
    expect(mocks.saveManualSingleUserCredential).not.toHaveBeenCalled()
  })

  it("keeps the previous configuration when a candidate-origin probe fails", async () => {
    const oldConfig = {
      serverUrl: "https://old.example.test",
      apiKey: "old-key"
    }
    let currentConfig = oldConfig
    mocks.testApiKey.mockResolvedValueOnce(false)
    mocks.saveManualSingleUserCredential.mockImplementationOnce(async (input) => {
      currentConfig = input
      return "device"
    })

    await expect(
      commitManualServerTransition({
        serverUrl: "https://new.example.test",
        apiKey: "new-key",
        persistence: "device"
      })
    ).rejects.toThrow(/validation failed/i)

    expect(currentConfig).toBe(oldConfig)
    expect(mocks.saveManualSingleUserCredential).not.toHaveBeenCalled()
  })

  it("rejects an invalid candidate URL without invoking the probe", async () => {
    await expect(
      commitManualServerTransition({
        serverUrl: "not-a-url",
        apiKey: "new-key",
        persistence: "device"
      })
    ).rejects.toThrow(/invalid server url/i)

    expect(mocks.testApiKey).not.toHaveBeenCalled()
    expect(mocks.saveManualSingleUserCredential).not.toHaveBeenCalled()
  })

  it("commits only after a successful candidate-origin probe", async () => {
    mocks.testApiKey.mockResolvedValueOnce(true)

    await expect(
      commitManualServerTransition({
        serverUrl: "https://new.example.test/path",
        apiKey: "new-key",
        persistence: "device"
      })
    ).resolves.toBe("device")

    expect(mocks.testApiKey).toHaveBeenCalledWith(
      "https://new.example.test/path",
      "new-key"
    )
    expect(mocks.saveManualSingleUserCredential).toHaveBeenCalledWith({
      serverUrl: "https://new.example.test/path",
      apiKey: "new-key",
      persistence: "device"
    })
    expect(mocks.testApiKey.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.saveManualSingleUserCredential.mock.invocationCallOrder[0]
    )
  })

  it("commits the selected session scope and reports the achieved scope", async () => {
    mocks.testApiKey.mockResolvedValueOnce(true)
    mocks.saveManualSingleUserCredential.mockResolvedValueOnce("session")

    const result = await validateApiKey(
      "https://new.example.test",
      "new-key",
      ((key: string, fallback: string) => fallback || key) as any,
      "session"
    )

    expect(result).toMatchObject({ success: true, persistence: "session" })
    expect(mocks.saveManualSingleUserCredential).toHaveBeenCalledWith({
      serverUrl: "https://new.example.test",
      apiKey: "new-key",
      persistence: "session"
    })
  })
})
