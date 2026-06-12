import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  searchMedia: vi.fn(),
  searchNotes: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    searchMedia: mocks.searchMedia,
    searchNotes: mocks.searchNotes
  }
}))

import { explainerApi } from "../explainerApi"

describe("explainerApi.searchSources", () => {
  beforeEach(() => {
    mocks.searchMedia.mockReset()
    mocks.searchNotes.mockReset()
  })

  it("returns combined media and note candidates", async () => {
    mocks.searchMedia.mockResolvedValue({
      media: [{ media_id: 1, title: "A video" }]
    })
    mocks.searchNotes.mockResolvedValue([{ id: "n1", title: "A note" }])

    const results = await explainerApi.searchSources("attention")

    expect(results.map((item) => item.sourceType)).toEqual(["media", "note"])
  })

  it("still returns note results when media search fails", async () => {
    mocks.searchMedia.mockRejectedValue(new Error("media backend down"))
    mocks.searchNotes.mockResolvedValue([{ id: "n1", title: "A note" }])

    const results = await explainerApi.searchSources("attention")

    expect(results).toHaveLength(1)
    expect(results[0]).toMatchObject({ sourceId: "n1", sourceType: "note" })
  })

  it("still returns media results when notes search fails", async () => {
    mocks.searchMedia.mockResolvedValue({
      media: [{ media_id: 1, title: "A video" }]
    })
    mocks.searchNotes.mockRejectedValue(new Error("notes backend down"))

    const results = await explainerApi.searchSources("attention")

    expect(results).toHaveLength(1)
    expect(results[0]).toMatchObject({ sourceId: "1", sourceType: "media" })
  })
})
