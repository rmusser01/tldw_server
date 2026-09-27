import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: vi.fn()
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: vi.fn(async () => null),
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"

describe("TldwApiClient character delete", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("deletes with an explicit expected_version when provided", async () => {
    mocks.bgRequest.mockResolvedValue(undefined)

    const client = new TldwApiClient()
    await client.deleteCharacter("123", 7)

    const calls = mocks.bgRequest.mock.calls.map(([request]) => request as { path?: string; method?: string })
    expect(
      calls.some(
        (request) =>
          request.path === "/api/v1/characters/123?expected_version=7" &&
          request.method === "DELETE"
      )
    ).toBe(true)
    expect(
      calls.some(
        (request) => request.path === "/api/v1/characters/123" && request.method === "GET"
      )
    ).toBe(false)
  })

  it("fetches the latest version before delete when expected_version is omitted", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.path === "/api/v1/characters/123" && request.method === "GET") {
        return { id: "123", version: 5 }
      }
      return undefined
    })

    const client = new TldwApiClient()
    await client.deleteCharacter("123")

    const calls = mocks.bgRequest.mock.calls.map(([request]) => request as { path?: string; method?: string })
    expect(
      calls.some(
        (request) => request.path === "/api/v1/characters/123" && request.method === "GET"
      )
    ).toBe(true)
    expect(
      calls.some(
        (request) =>
          request.path === "/api/v1/characters/123?expected_version=5" &&
          request.method === "DELETE"
      )
    ).toBe(true)
  })

  it("fails fast when character version cannot be resolved", async () => {
    mocks.bgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (request.path === "/api/v1/characters/123" && request.method === "GET") {
        return { id: "123" }
      }
      return undefined
    })

    const client = new TldwApiClient()
    await expect(client.deleteCharacter("123")).rejects.toThrow(
      "Character delete failed: missing expected version"
    )

    const calls = mocks.bgRequest.mock.calls.map(([request]) => request as { path?: string; method?: string })
    expect(
      calls.some(
        (request) => request.path === "/api/v1/characters/123" && request.method === "GET"
      )
    ).toBe(true)
    expect(
      calls.some((request) => request.method === "DELETE" && request.path?.includes("/api/v1/characters/123"))
    ).toBe(false)
  })

  it("adds expected_version to updates only when the caller supplies the loaded version", async () => {
    mocks.bgRequest.mockResolvedValue({ id: "123" })

    const client = new TldwApiClient()
    await client.updateCharacter("123", { name: "Saved from editor" }, 0)
    await client.updateCharacter("124", { name: "No loaded version" })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/characters/123?expected_version=0",
        method: "PUT"
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/characters/124",
        method: "PUT"
      })
    )
  })
})
