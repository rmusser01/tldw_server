import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
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

import {
  TldwApiClient,
  TldwApiClientBase
} from "@/services/tldw/TldwApiClient"

describe("TldwApiClient Wave 5 boundary slices", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("keeps listAdminUsers on the mixed admin domain path after class cleanup", async () => {
    mocks.bgRequest.mockResolvedValue({
      users: [],
      total: 0,
      page: 1,
      limit: 20,
      pages: 0
    })

    const client = new TldwApiClient()
    await client.listAdminUsers({ limit: 20 })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/admin/users?limit=20",
        method: "GET"
      })
    )
    expect(Object.getOwnPropertyNames(TldwApiClientBase.prototype)).not.toContain(
      "listAdminUsers"
    )
  })

  it("keeps listSkills on the mixed workspace-api domain path after class cleanup", async () => {
    mocks.bgRequest.mockResolvedValue({
      skills: [],
      count: 0,
      total: 0,
      limit: 10,
      offset: 0
    })

    const client = new TldwApiClient()
    client.resolveApiPath = vi
      .fn(async () => "/api/v1/skills") as typeof client.resolveApiPath

    await client.listSkills({ limit: 10 })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/skills?limit=10",
        method: "GET"
      })
    )
    expect(Object.getOwnPropertyNames(TldwApiClientBase.prototype)).not.toContain(
      "listSkills"
    )
  })

  it("keeps presentation methods on the mixed presentations domain paths after class cleanup", async () => {
    const client = new TldwApiClient()
    client.ensureConfigForRequest = vi
      .fn(async () => ({ serverUrl: "http://127.0.0.1:8000" })) as typeof client.ensureConfigForRequest
    client.request = vi
      .fn()
      .mockResolvedValueOnce({
        id: "pres-1",
        title: "Deck",
        theme: "black",
        slides: [],
        version: 1,
        created_at: "2026-04-17T00:00:00Z"
      })
      .mockResolvedValueOnce({
        id: "pres-1",
        title: "Deck",
        theme: "black",
        slides: [],
        version: 1,
        created_at: "2026-04-17T00:00:00Z",
        last_modified: "2026-04-17T00:00:00Z"
      })
      .mockResolvedValueOnce({
        data: new ArrayBuffer(0)
      }) as typeof client.request

    await client.generateSlidesFromMedia(7, { titleHint: "Deck" })
    await client.getPresentation("pres-1")
    await client.exportPresentation("pres-1", "pdf")

    expect(client.request).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/slides/generate/from-media",
        method: "POST"
      })
    )
    expect(client.request).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/slides/presentations/pres-1",
        method: "GET"
      })
    )
    expect(client.request).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        path: "/api/v1/slides/presentations/pres-1/export?format=pdf",
        method: "GET"
      })
    )
    const baseMethodNames = Object.getOwnPropertyNames(TldwApiClientBase.prototype)
    expect(baseMethodNames).not.toContain("generateSlidesFromMedia")
    expect(baseMethodNames).not.toContain("getPresentation")
    expect(baseMethodNames).not.toContain("exportPresentation")
  })
})
