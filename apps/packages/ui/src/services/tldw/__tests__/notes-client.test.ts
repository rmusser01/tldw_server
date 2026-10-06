import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: vi.fn(),
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

describe("TldwApiClient notes methods", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.bgRequest.mockResolvedValue([])
  })

  it("maps page/results_per_page to limit/offset for listNotes", async () => {
    const client = new TldwApiClient()
    await client.listNotes({ page: 2, results_per_page: 200, include_keywords: false })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/?limit=200&offset=200&include_keywords=false",
        method: "GET"
      })
    )
  })

  it("uses GET query params for searchNotes", async () => {
    const client = new TldwApiClient()
    await client.searchNotes("cell biology")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/search/?query=cell+biology",
        method: "GET"
      })
    )
  })

  it("falls back to listNotes when search query is empty", async () => {
    const client = new TldwApiClient()
    await client.searchNotes("   ")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/",
        method: "GET"
      })
    )
  })

  it("returns search results from the requested Notes page and carries cancellation", async () => {
    const client = new TldwApiClient()
    const controller = new AbortController()
    mocks.bgRequest.mockImplementation(
      async (request: { path: string; abortSignal?: AbortSignal }) =>
        request.path ===
          "/api/v1/notes/search/?query=biology&limit=50&offset=200" &&
        request.abortSignal === controller.signal
          ? { notes: [{ id: "note-207", title: "Page five note" }], total: 250 }
          : { notes: [] },
    )
    expect(
      await client.searchNotes(
        "biology",
        { limit: 50, offset: 200 },
        { signal: controller.signal },
      ),
    ).toEqual({
      notes: [{ id: "note-207", title: "Page five note" }],
      total: 250,
    })
  })

  it("loads note folders from the public notes folder endpoint", async () => {
    const client = new TldwApiClient()
    await client.listNoteFolders()

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/folders/",
        method: "GET"
      })
    )
  })

  it("creates note folders by normalized path", async () => {
    const client = new TldwApiClient()
    await client.createNoteFolder("Inbox/Captured Articles")

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/folders/",
        method: "POST",
        body: { path: "Inbox/Captured Articles" }
      })
    )
  })
})
