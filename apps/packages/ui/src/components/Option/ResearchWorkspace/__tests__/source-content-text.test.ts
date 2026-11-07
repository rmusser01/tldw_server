import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  MAX_SOURCE_CONTENT_CACHE_ENTRIES,
  extractSourceContentText,
  resolveSourceContentText
} from "../source-content-text"
import { useWorkspaceStore } from "@/store/workspace"

const getMediaDetailsMock = vi.hoisted(() => vi.fn())

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: (...args: unknown[]) => getMediaDetailsMock(...args)
  }
}))

const detailWithText = (text: string) => ({
  content: { text },
  source: { title: "Fetched title" }
})

const resetCache = () => {
  useWorkspaceStore.setState({ sourceContentCache: {} })
}

describe("resolveSourceContentText (WP-45 store reuse)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetCache()
  })

  it("pins no-network on the second ask for the same (mediaId, version)", async () => {
    getMediaDetailsMock.mockResolvedValue(detailWithText("first fetch text"))

    const first = await resolveSourceContentText(7)
    expect(first).toEqual({
      text: "first fetch text",
      title: "Fetched title",
      fromCache: false
    })
    expect(getMediaDetailsMock).toHaveBeenCalledTimes(1)
    expect(getMediaDetailsMock).toHaveBeenCalledWith(
      7,
      expect.objectContaining({
        include_content: true,
        include_versions: false,
        include_version_content: false
      })
    )

    const second = await resolveSourceContentText(7)
    expect(second).toEqual({
      text: "first fetch text",
      title: "Fetched title",
      fromCache: true
    })
    expect(getMediaDetailsMock).toHaveBeenCalledTimes(1)
  })

  it("fetches only misses", async () => {
    getMediaDetailsMock.mockImplementation(async (mediaId: number) =>
      detailWithText(`text for ${mediaId}`)
    )

    await resolveSourceContentText(7)
    await resolveSourceContentText(8)
    await resolveSourceContentText(7)

    expect(getMediaDetailsMock).toHaveBeenCalledTimes(2)
    expect(getMediaDetailsMock).toHaveBeenCalledWith(8, expect.anything())
  })

  it("does not serve a versioned cache entry to an unversioned ask", async () => {
    useWorkspaceStore.getState().setSourceContentText(7, 3, "versioned text")
    getMediaDetailsMock.mockResolvedValue(detailWithText("unversioned text"))

    const unversioned = await resolveSourceContentText(7)
    expect(unversioned.text).toBe("unversioned text")
    expect(unversioned.fromCache).toBe(false)

    // The unversioned ask misses (version mismatch) and then overwrites the
    // entry under its own identity.
    expect(useWorkspaceStore.getState().getSourceContentText(7, null)).toBe(
      "unversioned text"
    )
    expect(useWorkspaceStore.getState().getSourceContentText(7, 3)).toBeNull()
  })

  it("propagates fetch failures without caching", async () => {
    getMediaDetailsMock.mockRejectedValue(new Error("detail fetch failed"))

    await expect(resolveSourceContentText(42)).rejects.toThrow(
      "detail fetch failed"
    )
    expect(useWorkspaceStore.getState().getSourceContentText(42, null)).toBeNull()
  })

  it("does not cache empty extractions", async () => {
    getMediaDetailsMock.mockResolvedValue({ content: {} })

    const resolved = await resolveSourceContentText(9)
    expect(resolved.text).toBe("")
    expect(resolved.fromCache).toBe(false)
    expect(useWorkspaceStore.getState().getSourceContentText(9, null)).toBeNull()
  })

  it("bounds the in-memory cache to the entry cap", () => {
    const store = useWorkspaceStore.getState()
    for (let mediaId = 1; mediaId <= MAX_SOURCE_CONTENT_CACHE_ENTRIES + 1; mediaId += 1) {
      store.setSourceContentText(mediaId, null, `text ${mediaId}`)
    }

    const cache = useWorkspaceStore.getState().sourceContentCache
    expect(Object.keys(cache)).toHaveLength(MAX_SOURCE_CONTENT_CACHE_ENTRIES)
    expect(useWorkspaceStore.getState().getSourceContentText(1, null)).toBeNull()
    expect(useWorkspaceStore.getState().getSourceContentText(
      MAX_SOURCE_CONTENT_CACHE_ENTRIES + 1,
      null
    )).toBe(`text ${MAX_SOURCE_CONTENT_CACHE_ENTRIES + 1}`)
  })
})

describe("extractSourceContentText", () => {
  it("reads the first supported text candidate", () => {
    expect(extractSourceContentText({ content: { text: "a" } })).toBe("a")
    expect(extractSourceContentText({ content: "direct string" })).toBe(
      "direct string"
    )
    expect(extractSourceContentText({ content: { transcript: "t" } })).toBe("t")
    expect(extractSourceContentText({ processing: { analysis: "p" } })).toBe("p")
    expect(extractSourceContentText({ content: { text: "   " } })).toBe("")
    expect(extractSourceContentText(null)).toBe("")
  })
})
