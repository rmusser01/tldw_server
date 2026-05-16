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

describe("TldwApiClient media ingest contract", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("uses multipart /media/add fields inferred from URL", async () => {
    mocks.bgUpload.mockResolvedValue({ results: [] })

    const client = new TldwApiClient()
    await client.addMedia("https://example.com/article")

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/add",
        method: "POST",
        fields: expect.objectContaining({
          media_type: "document",
          urls: ["https://example.com/article"]
        })
      })
    )
  })

  it("infers video media_type for youtube URLs", async () => {
    mocks.bgUpload.mockResolvedValue({ results: [] })

    const client = new TldwApiClient()
    await client.addMedia("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          media_type: "video",
          urls: ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
        })
      })
    )
  })

  it("keeps explicit media_type overrides", async () => {
    mocks.bgUpload.mockResolvedValue({ results: [] })

    const client = new TldwApiClient()
    await client.addMedia("https://example.com/file.mp4", {
      media_type: "audio"
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        fields: expect.objectContaining({
          media_type: "audio",
          urls: ["https://example.com/file.mp4"]
        })
      })
    )
  })

  it("forwards timeout and ingest options with urls list", async () => {
    mocks.bgUpload.mockResolvedValue({ results: [] })

    const client = new TldwApiClient()
    await client.addMedia("https://example.com/video.mp4", {
      timeoutMs: 45000,
      perform_analysis: false,
      perform_chunking: true,
      overwrite_existing: false
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        timeoutMs: 45000,
        fields: expect.objectContaining({
          media_type: "video",
          urls: ["https://example.com/video.mp4"],
          perform_analysis: false,
          perform_chunking: true,
          overwrite_existing: false
        })
      })
    )
  })

  it("submits media ingest jobs via multipart fields", async () => {
    mocks.bgUpload.mockResolvedValue({ batch_id: "batch-1", jobs: [] })

    const client = new TldwApiClient()
    await client.submitMediaIngestJobs({
      media_type: "video",
      urls: ["https://example.com/video.mp4"],
      perform_analysis: true,
      timeoutMs: 120000
    })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs",
        method: "POST",
        timeoutMs: 120000,
        fields: expect.objectContaining({
          media_type: "video",
          urls: ["https://example.com/video.mp4"],
          perform_analysis: true
        })
      })
    )
  })

  it("fetches ingest job detail and batch list endpoints", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ id: 12, status: "queued" })
    mocks.bgRequest.mockResolvedValueOnce({ batch_id: "b1", jobs: [] })

    const client = new TldwApiClient()
    await client.getMediaIngestJob(12)
    await client.listMediaIngestJobs({ batch_id: "b1", limit: 25 })

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs/12",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/media/ingest/jobs?batch_id=b1&limit=25",
        method: "GET"
      })
    )
  })

  it("preflights playlist URLs through the metadata-only media endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({ playlist_id: "PLtest", items: [] })

    const client = new TldwApiClient()
    await client.preflightPlaylist({
      url: "https://www.youtube.com/playlist?list=PLtest",
      max_items: 34
    })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/media/playlists/preflight",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: {
          url: "https://www.youtube.com/playlist?list=PLtest",
          max_items: 34
        }
      })
    )
  })

  it("uploads character imports using binary payloads", async () => {
    mocks.bgUpload.mockResolvedValue({
      id: 123,
      name: "Imported Character",
      message: "Character imported successfully"
    })

    const client = new TldwApiClient()
    ;(client as any).ensureConfigForRequest = vi.fn(async () => ({ ok: true }))
    const rawBuffer = new Uint8Array([0x89, 0x50, 0x4e, 0x47]).buffer
    const file = {
      name: "card.png",
      type: "image/png",
      arrayBuffer: vi.fn(async () => rawBuffer)
    } as unknown as File

    await client.importCharacterFile(file, { allowImageOnly: true })

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/characters/import",
        method: "POST",
        fileFieldName: "character_file",
        fields: { allow_image_only: true },
        file: expect.objectContaining({
          name: "card.png",
          type: "image/png"
        })
      })
    )

    const callArg = mocks.bgUpload.mock.calls[0][0] as {
      file?: { data?: unknown }
    }
    expect(callArg.file?.data).toBe(rawBuffer)
    expect(Array.isArray(callArg.file?.data)).toBe(false)
  })

  it("uploads yaml character imports through the same endpoint contract", async () => {
    mocks.bgUpload.mockResolvedValue({
      id: 124,
      name: "Imported YAML Character",
      message: "Character imported successfully"
    })

    const client = new TldwApiClient()
    ;(client as any).ensureConfigForRequest = vi.fn(async () => ({ ok: true }))
    const rawBuffer = new TextEncoder().encode("name: YAML Client Test").buffer
    const file = {
      name: "card.yaml",
      type: "text/yaml",
      arrayBuffer: vi.fn(async () => rawBuffer)
    } as unknown as File

    await client.importCharacterFile(file)

    expect(mocks.bgUpload).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/characters/import",
        method: "POST",
        fileFieldName: "character_file",
        file: expect.objectContaining({
          name: "card.yaml",
          type: "text/yaml"
        })
      })
    )

    const callArg = mocks.bgUpload.mock.calls.at(-1)?.[0] as {
      fields?: Record<string, unknown>
    }
    expect(callArg.fields).toBeUndefined()
  })
})
