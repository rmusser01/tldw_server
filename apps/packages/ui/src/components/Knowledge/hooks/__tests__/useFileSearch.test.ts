import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useFileSearch } from "../useFileSearch"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(),
    searchMedia: vi.fn(),
    getMediaDetails: vi.fn()
  }
}))

describe("useFileSearch", () => {
  const baseSettings = {
    ...DEFAULT_RAG_SETTINGS
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(tldwClient.initialize).mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: vi.fn()
      }
    })
  })

  const createHook = (resolvedQuery = "knowledge query") => {
    const applySettings = vi.fn()
    const onInsert = vi.fn()
    const onPin = vi.fn()

    const hook = renderHook(() =>
      useFileSearch({
        resolvedQuery,
        draftSettings: baseSettings,
        applySettings,
        onInsert,
        pinnedResults: [],
        onPin
      })
    )

    return { ...hook, applySettings, onInsert, onPin }
  }

  it("runs file search with selected media filters and apply-first behavior", async () => {
    vi.mocked(tldwClient.searchMedia).mockResolvedValue({
      items: [
        {
          id: 42,
          title: "Quarterly Report",
          type: "pdf",
          url: "/api/v1/media/42",
          snippet: "Quarterly report summary"
        }
      ]
    })

    const { result, applySettings } = createHook("quarterly report")

    act(() => {
      result.current.setMediaTypes(["pdf", "note"])
    })

    await act(async () => {
      await result.current.runSearch({ applyFirst: true })
    })

    expect(applySettings).toHaveBeenCalledTimes(1)
    expect(tldwClient.searchMedia).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "quarterly report",
        fields: ["title", "content"],
        sort_by: "relevance",
        media_types: ["pdf", "note"]
      }),
      { page: 1, results_per_page: 50 }
    )
    expect(result.current.results).toHaveLength(1)
    expect(result.current.results[0].metadata?.title).toBe("Quarterly Report")
    expect(result.current.results[0].metadata?.media_id).toBe(42)
    expect(result.current.results[0].metadata?.origin).toBe("media-library")
  })

  it("sets query error and skips search when resolved query is empty", async () => {
    const { result } = createHook("   ")

    await act(async () => {
      await result.current.runSearch()
    })

    expect(result.current.queryError).toBe("Enter a query to search.")
    expect(tldwClient.searchMedia).not.toHaveBeenCalled()
  })

  it("attaches full media text and tracks attached media ids", async () => {
    vi.mocked(tldwClient.searchMedia).mockResolvedValue({
      items: [
        {
          id: 42,
          title: "Quarterly Report",
          type: "pdf",
          url: "/api/v1/media/42",
          snippet: "Short snippet"
        }
      ]
    })
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      content: {
        text: "Full media body content"
      }
    })

    const { result, onInsert } = createHook("quarterly report")

    await act(async () => {
      await result.current.runSearch()
    })

    await act(async () => {
      result.current.handleAttach(result.current.results[0])
    })

    await waitFor(() => {
      expect(onInsert).toHaveBeenCalledWith(
        expect.stringContaining("Full media body content")
      )
    })
    expect(onInsert).toHaveBeenCalledWith(
      expect.stringContaining("**Quarterly Report**")
    )
    expect(result.current.attachedMediaIds.has(42)).toBe(true)
    expect(tldwClient.getMediaDetails).toHaveBeenCalledWith(
      42,
      expect.objectContaining({ include_content: true })
    )
  })

  it("copies full media-library content as markdown", async () => {
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({
      content: {
        text: "Full media body content"
      }
    })

    const { result } = createHook("quarterly report")
    const mediaResult = {
      content: "Short snippet",
      metadata: {
        id: 42,
        media_id: 42,
        title: "Quarterly Report",
        type: "pdf",
        url: "/api/v1/media/42",
        origin: "media-library"
      }
    }

    await act(async () => {
      await result.current.copyResult(mediaResult, "markdown")
    })

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      expect.stringContaining("Full media body content")
    )
  })

  it("marks timedOut when media search fails with timeout", async () => {
    vi.mocked(tldwClient.searchMedia).mockRejectedValue(
      new Error("request timed out")
    )

    const { result } = createHook("slow query")

    await act(async () => {
      await result.current.runSearch()
    })

    expect(result.current.timedOut).toBe(true)
    expect(result.current.results).toEqual([])
    expect(result.current.loading).toBe(false)
  })
})
