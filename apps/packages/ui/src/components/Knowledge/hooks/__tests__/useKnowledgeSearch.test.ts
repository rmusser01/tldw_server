import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(),
    getMediaDetails: vi.fn(),
    searchMedia: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false, vi.fn()] as const
}))

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useStoreMessageOption } from "@/store/option"

import {
  extractContentFromMediaDetail,
  extractMediaId,
  normalizeMediaSearchResults,
  toPinnedResult,
  useKnowledgeSearch,
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

describe("useKnowledgeSearch action paths", () => {
  const baseSettings = {
    ...DEFAULT_RAG_SETTINGS
  }

  const mediaResult = (overrides?: { id?: number; title?: string }) =>
    normalizeMediaSearchResults({
      items: [
        {
          id: overrides?.id ?? 42,
          title: overrides?.title ?? "Quarterly Report",
          type: "pdf",
          url: `/api/v1/media/${overrides?.id ?? 42}`
        }
      ]
    })[0]

  const chunkResult = (): RagResult => ({
    content: "Retrieved chunk only",
    metadata: {
      media_id: 42,
      title: "Chunk Title",
      type: "pdf"
    }
  })

  const createHook = () => {
    const applySettings = vi.fn()
    const onInsert = vi.fn()
    const onAsk = vi.fn()

    const hook = renderHook(() =>
      useKnowledgeSearch({
        resolvedQuery: "quarterly report",
        draftSettings: baseSettings,
        applySettings,
        onInsert,
        onAsk
      })
    )

    return { ...hook, applySettings, onInsert, onAsk }
  }

  const deferredMediaDetails = (text: string) => {
    let resolveDetail: () => void = () => {}
    const pending = new Promise((resolve) => {
      resolveDetail = () => resolve({ content: { text } })
    })
    vi.mocked(tldwClient.getMediaDetails).mockReturnValue(
      pending as ReturnType<typeof tldwClient.getMediaDetails>
    )
    return resolveDetail
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(tldwClient.initialize).mockResolvedValue(undefined)
    useStoreMessageOption.setState({
      ragPinnedResults: [],
      ragMediaIds: null
    })
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: vi.fn()
      }
    })
  })

  it("asks with full media-library content and ignores pinned results", async () => {
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      content: { text: "Full media body content" }
    })
    const { result, onAsk } = createHook()

    act(() => {
      result.current.handleAsk(mediaResult())
    })

    await waitFor(() => {
      expect(onAsk).toHaveBeenCalledWith(
        expect.stringContaining("Full media body content"),
        { ignorePinnedResults: true }
      )
    })
  })

  it("pins full media-library content and preserves media id scoping", async () => {
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      content: { text: "Full media body content" }
    })
    const { result } = createHook()

    act(() => {
      result.current.handlePin(mediaResult())
    })

    await waitFor(() => {
      expect(useStoreMessageOption.getState().ragPinnedResults[0]?.snippet).toBe(
        "Full media body content"
      )
    })
    expect(useStoreMessageOption.getState().ragMediaIds).toEqual([42])
  })

  it("preserves separate media pins resolved from overlapping requests", async () => {
    vi.mocked(tldwClient.getMediaDetails)
      .mockResolvedValueOnce({ content: { text: "Full first content" } })
      .mockResolvedValueOnce({ content: { text: "Full second content" } })
    const { result } = createHook()

    act(() => {
      result.current.handlePin(mediaResult({ id: 42, title: "First Report" }))
      result.current.handlePin(mediaResult({ id: 43, title: "Second Report" }))
    })

    await waitFor(() => {
      expect(useStoreMessageOption.getState().ragPinnedResults).toHaveLength(2)
    })
    expect(useStoreMessageOption.getState().ragPinnedResults).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          mediaId: 42,
          snippet: "Full first content"
        }),
        expect.objectContaining({
          mediaId: 43,
          snippet: "Full second content"
        })
      ])
    )
    expect(useStoreMessageOption.getState().ragMediaIds).toEqual([42, 43])
  })

  it("does not re-add a pending media pin after clear all", async () => {
    const resolveDetail = deferredMediaDetails("Full media body content")
    const { result } = createHook()

    act(() => {
      result.current.handlePin(mediaResult())
    })

    await waitFor(() => {
      expect(tldwClient.getMediaDetails).toHaveBeenCalledTimes(1)
    })
    expect(useStoreMessageOption.getState().ragPinnedResults).toEqual([])

    await act(async () => {
      result.current.handleClearPins()
      resolveDetail()
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(useStoreMessageOption.getState().ragPinnedResults).toEqual([])
    expect(useStoreMessageOption.getState().ragMediaIds).toBeNull()
  })

  it("copies full media-library content as markdown", async () => {
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      content: { text: "Full media body content" }
    })
    const { result } = createHook()

    await act(async () => {
      await result.current.copyResult(mediaResult(), "markdown")
    })

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining("Full media body content")
    )
  })

  it("keeps chunk-scoped media results unexpanded for ask, pin, and copy", async () => {
    const { result, onAsk } = createHook()

    act(() => {
      result.current.handleAsk(chunkResult())
    })

    await waitFor(() => {
      expect(onAsk).toHaveBeenCalledWith(
        expect.stringContaining("Retrieved chunk only"),
        { ignorePinnedResults: true }
      )
    })
    expect(tldwClient.getMediaDetails).not.toHaveBeenCalled()

    act(() => {
      result.current.handlePin(chunkResult())
    })

    await waitFor(() => {
      expect(useStoreMessageOption.getState().ragPinnedResults[0]?.snippet).toBe(
        "Retrieved chunk only"
      )
    })
    expect(tldwClient.getMediaDetails).not.toHaveBeenCalled()

    await act(async () => {
      await result.current.copyResult(chunkResult(), "markdown")
    })

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining("Retrieved chunk only")
    )
    expect(tldwClient.getMediaDetails).not.toHaveBeenCalled()
  })
})
