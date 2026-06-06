import { beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(),
    getMediaDetails: vi.fn(),
    searchMedia: vi.fn()
  }
}))

import { tldwClient } from "@/services/tldw/TldwApiClient"

import {
  extractContentFromMediaDetail,
  extractMediaId,
  normalizeMediaSearchResults,
  toPinnedResult,
  withFullMediaTextIfAvailable,
  type RagResult
} from "../useKnowledgeSearch"

describe("useKnowledgeSearch helpers", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("normalizes media search responses into knowledge result cards", () => {
    const payload = {
      items: [
        {
          id: 42,
          title: "Quarterly Report",
          type: "pdf",
          url: "/api/v1/media/42"
        }
      ]
    }

    const results = normalizeMediaSearchResults(payload)

    expect(results).toHaveLength(1)
    expect(results[0].metadata?.media_id).toBe(42)
    expect(results[0].metadata?.title).toBe("Quarterly Report")
    expect(results[0].metadata?.type).toBe("pdf")
    expect(results[0].content).toContain("Library item")
  })

  it("extracts media id and carries it into pinned result metadata", () => {
    const result: RagResult = {
      content: "Snippet text",
      metadata: {
        media_id: "17",
        title: "Research Note",
        type: "note"
      }
    }

    expect(extractMediaId(result)).toBe(17)

    const pinned = toPinnedResult(result)
    expect(pinned.mediaId).toBe(17)
    expect(pinned.title).toBe("Research Note")
    expect(pinned.type).toBe("note")
  })

  it("extracts full text from nested media detail content objects", () => {
    const detail = {
      content: {
        text: "Full media transcript"
      }
    }

    expect(extractContentFromMediaDetail(detail)).toBe("Full media transcript")
  })

  it("falls back to latest_version and data content fields", () => {
    const latestVersionDetail = {
      latest_version: {
        content: "Latest version text"
      }
    }
    const dataDetail = {
      data: {
        raw_text: "Data-level text"
      }
    }

    expect(extractContentFromMediaDetail(latestVersionDetail)).toBe(
      "Latest version text"
    )
    expect(extractContentFromMediaDetail(dataDetail)).toBe("Data-level text")
  })

  it("marks normalized media-library rows and carries the marker into pinned results", () => {
    const [result] = normalizeMediaSearchResults({
      items: [
        {
          id: 42,
          title: "Quarterly Report",
          type: "pdf",
          url: "/api/v1/media/42"
        }
      ]
    })

    expect(result.metadata?.media_id).toBe(42)
    expect(result.metadata?.origin).toBe("media-library")
    expect(toPinnedResult(result).contextOrigin).toBe("media-library")
  })

  it("fetches full text only for pinned media-library results", async () => {
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      content: { text: "Full media body content" }
    })

    const pinned = await withFullMediaTextIfAvailable({
      id: "media-42",
      title: "Quarterly Report",
      snippet: "Library item: Quarterly Report",
      mediaId: 42,
      contextOrigin: "media-library"
    })

    expect(pinned.snippet).toBe("Full media body content")
    expect(tldwClient.getMediaDetails).toHaveBeenCalledWith(42, {
      include_content: true,
      include_versions: false,
      include_version_content: false
    })
  })

  it("does not fetch full media details for chunk-scoped pinned results", async () => {
    const pinned = await withFullMediaTextIfAvailable({
      id: "chunk-42",
      title: "Chunk",
      snippet: "Retrieved chunk only",
      mediaId: 42
    })

    expect(pinned.snippet).toBe("Retrieved chunk only")
    expect(tldwClient.getMediaDetails).not.toHaveBeenCalled()
  })

  it("keeps fallback snippet when full media detail fetch fails or returns empty content", async () => {
    vi.mocked(tldwClient.getMediaDetails).mockRejectedValueOnce(
      new Error("media detail unavailable")
    )

    const failedFetch = await withFullMediaTextIfAvailable({
      id: "media-42",
      title: "Quarterly Report",
      snippet: "Library item: Quarterly Report",
      mediaId: 42,
      contextOrigin: "media-library"
    })

    expect(failedFetch.snippet).toBe("Library item: Quarterly Report")

    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      content: { text: "" }
    })

    const emptyContent = await withFullMediaTextIfAvailable({
      id: "media-42",
      title: "Quarterly Report",
      snippet: "Library item: Quarterly Report",
      mediaId: 42,
      contextOrigin: "media-library"
    })

    expect(emptyContent.snippet).toBe("Library item: Quarterly Report")
  })
})
