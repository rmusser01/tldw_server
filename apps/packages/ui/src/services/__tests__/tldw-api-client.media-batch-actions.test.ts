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

describe("TldwApiClient media batch and lifecycle actions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("calls bulk keyword update endpoint with mode and ids", async () => {
    mocks.bgRequest.mockResolvedValue({ updated: 2, failed: 0, results: [] })

    const client = new TldwApiClient()
    await client.bulkUpdateMediaKeywords({
      media_ids: [1, 2],
      keywords: ["ai"],
      mode: "add"
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/bulk/keyword-update",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: {
          media_ids: [1, 2],
          keywords: ["ai"],
          mode: "add"
        }
      })
    )
  })

  it("falls back to per-item keyword updates when bulk endpoint is unavailable", async () => {
    mocks.bgRequest
      .mockRejectedValueOnce({ status: 404 })
      .mockResolvedValueOnce({ media_id: 1, keywords: ["ai"] })
      .mockResolvedValueOnce({ media_id: 2, keywords: ["ai"] })

    const client = new TldwApiClient()
    const result = await client.bulkUpdateMediaKeywords({
      media_ids: [1, 2],
      keywords: ["ai"]
    })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/media/1/keywords",
        method: "PATCH"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        path: "/api/v1/media/2/keywords",
        method: "PATCH"
      })
    )
    expect(result).toMatchObject({
      endpoint: "fallback",
      updated: 2,
      failed: 0
    })
  })

  it("passes backend-unavailable suppression to single-item keyword updates", async () => {
    mocks.bgRequest.mockResolvedValue({ media_id: 3, keywords: ["ai"] })

    const client = new TldwApiClient()
    await client.updateMediaKeywords(
      3,
      {
        keywords: ["ai"],
        mode: "set"
      },
      { suppressBackendUnavailableEvent: true }
    )

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/3/keywords",
        method: "PATCH",
        suppressBackendUnavailableEvent: true
      })
    )
  })

  it("passes backend-unavailable suppression to media detail requests", async () => {
    mocks.bgRequest.mockResolvedValue({ id: 3 })

    const client = new TldwApiClient()
    await client.getMediaDetails(3, {
      include_content: false,
      suppressBackendUnavailableEvent: true
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: expect.stringContaining("/api/v1/media/3?"),
        method: "GET",
        suppressBackendUnavailableEvent: true
      })
    )
  })

  it("calls soft delete endpoint for media item", async () => {
    mocks.bgRequest.mockResolvedValue(undefined)

    const client = new TldwApiClient()
    await client.deleteMedia(7)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/7",
        method: "DELETE"
      })
    )
  })

  it("calls restore endpoint for media item", async () => {
    mocks.bgRequest.mockResolvedValue({ id: 7 })

    const client = new TldwApiClient()
    await client.restoreMedia(7)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/7/restore",
        method: "POST"
      })
    )
  })

  it("calls permanent delete endpoint for media item", async () => {
    mocks.bgRequest.mockResolvedValue(undefined)

    const client = new TldwApiClient()
    await client.permanentlyDeleteMedia(7)

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/7/permanent",
        method: "DELETE"
      })
    )
  })

  it("calls reprocess endpoint for a media item", async () => {
    mocks.bgRequest.mockResolvedValue({ media_id: 7, status: "queued" })

    const client = new TldwApiClient()
    await client.reprocessMedia(7, { include_embeddings: true })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/7/reprocess",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { include_embeddings: true }
      })
    )
  })

  it("calls media statistics endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({ total_items: 42 })

    const client = new TldwApiClient()
    await client.getMediaStatistics()

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/statistics",
        method: "GET"
      })
    )
  })
})
