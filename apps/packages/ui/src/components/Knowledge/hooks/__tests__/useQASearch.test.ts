import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import {
  DEFAULT_RAG_SETTINGS,
  buildRagSearchRequest
} from "@/services/rag/unified-rag"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useQASearch } from "../useQASearch"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(),
    ragSearch: vi.fn()
  }
}))

vi.mock("@/services/rag/unified-rag", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/rag/unified-rag")
  >("@/services/rag/unified-rag")
  return {
    ...actual,
    buildRagSearchRequest: vi.fn()
  }
})

describe("useQASearch", () => {
  const baseSettings = {
    ...DEFAULT_RAG_SETTINGS
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(tldwClient.initialize).mockResolvedValue(undefined)
    vi.mocked(buildRagSearchRequest).mockImplementation((settings) => ({
      query: settings.query || "default query",
      options: { strategy: settings.strategy },
      timeoutMs: 1000
    }))
  })

  const createHook = (resolvedQuery = "knowledge query") => {
    const applySettings = vi.fn()
    const onInsert = vi.fn()
    const onPin = vi.fn()

    const hook = renderHook(() =>
      useQASearch({
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

  it("builds RAG request payload and applies settings when requested", async () => {
    vi.mocked(buildRagSearchRequest).mockReturnValue({
      query: "built query",
      options: { strategy: "agentic", rerankTopK: 15 },
      timeoutMs: 4500
    })
    vi.mocked(tldwClient.ragSearch).mockResolvedValue({})

    const { result, applySettings } = createHook("user supplied query")

    await act(async () => {
      await result.current.runQASearch({ applyFirst: true })
    })

    expect(applySettings).toHaveBeenCalledTimes(1)
    expect(buildRagSearchRequest).toHaveBeenCalledWith(
      expect.objectContaining({ query: "user supplied query" })
    )
    expect(tldwClient.ragSearch).toHaveBeenCalledWith(
      "built query",
      expect.objectContaining({
        strategy: "agentic",
        rerankTopK: 15,
        timeoutMs: 4500
      })
    )
  })

  it("normalizes responses that return documents in `results`", async () => {
    vi.mocked(tldwClient.ragSearch).mockResolvedValue({
      query: "normalized query",
      generated_answer: "Generated answer",
      results: [
        {
          id: "doc-1",
          content: "Result document content",
          score: "0.87",
          relevance: 0.51,
          media_id: "123",
          metadata: { title: "Document Title" }
        }
      ],
      citations: [{ source: "doc-1", score: "0.42", chunk_index: "2" }],
      timings: { retrieval: "0.23" },
      total_time: "1.7",
      cache_hit: true,
      feedback_id: "fb-1",
      errors: ["warning"],
      expanded_queries: ["knowledge query variant"]
    })

    const { result } = createHook()

    await act(async () => {
      await result.current.runQASearch()
    })

    expect(result.current.response).not.toBeNull()
    expect(result.current.response?.query).toBe("normalized query")
    expect(result.current.response?.generatedAnswer).toBe("Generated answer")
    expect(result.current.response?.documents).toHaveLength(1)
    expect(result.current.response?.documents[0].content).toBe(
      "Result document content"
    )
    expect(result.current.response?.documents[0].score).toBe(0.87)
    expect(result.current.response?.documents[0].media_id).toBe(123)
    expect(result.current.response?.citations[0].score).toBe(0.42)
    expect(result.current.response?.citations[0].chunk_index).toBe(2)
    expect(result.current.response?.timings).toEqual({ retrieval: 0.23 })
    expect(result.current.response?.totalTime).toBe(1.7)
    expect(result.current.response?.cacheHit).toBe(true)
    expect(result.current.response?.feedbackId).toBe("fb-1")
    expect(result.current.response?.expandedQueries).toEqual([
      "knowledge query variant"
    ])
  })

  it("falls back to `docs` and legacy answer/cache field names", async () => {
    vi.mocked(tldwClient.ragSearch).mockResolvedValue({
      documents: [],
      docs: [{ chunk: "Chunk from docs payload", relevance: "0.66" }],
      answer: "Legacy answer field",
      cacheHit: true,
      totalTime: 2.4,
      feedbackId: "feedback-legacy"
    })

    const { result } = createHook()

    await act(async () => {
      await result.current.runQASearch()
    })

    expect(result.current.response?.generatedAnswer).toBe("Legacy answer field")
    expect(result.current.response?.documents).toHaveLength(1)
    expect(result.current.response?.documents[0].chunk).toBe(
      "Chunk from docs payload"
    )
    expect(result.current.response?.documents[0].relevance).toBe(0.66)
    expect(result.current.response?.cacheHit).toBe(true)
    expect(result.current.response?.totalTime).toBe(2.4)
    expect(result.current.response?.feedbackId).toBe("feedback-legacy")
  })

  it("keeps `documents` precedence when multiple document arrays are present", async () => {
    vi.mocked(tldwClient.ragSearch).mockResolvedValue({
      documents: [{ content: "From documents" }],
      results: [{ content: "From results" }],
      docs: [{ content: "From docs" }]
    })

    const { result } = createHook()

    await act(async () => {
      await result.current.runQASearch()
    })

    expect(result.current.response?.documents).toHaveLength(1)
    expect(result.current.response?.documents[0].content).toBe("From documents")
  })

  it("returns safe defaults for partial payloads", async () => {
    vi.mocked(tldwClient.ragSearch).mockResolvedValue({})

    const { result } = createHook("fallback query")

    await act(async () => {
      await result.current.runQASearch()
    })

    expect(result.current.response).toEqual({
      generatedAnswer: null,
      documents: [],
      citations: [],
      academicCitations: [],
      timings: {},
      totalTime: 0,
      cacheHit: false,
      feedbackId: null,
      errors: [],
      query: "fallback query",
      expandedQueries: []
    })
  })

  it("sets timedOut for timeout failures and clears response", async () => {
    vi.mocked(tldwClient.ragSearch).mockRejectedValue(
      new Error("request timed out")
    )

    const { result } = createHook()

    await act(async () => {
      await result.current.runQASearch()
    })

    expect(result.current.response).toBeNull()
    expect(result.current.timedOut).toBe(true)
    expect(result.current.loading).toBe(false)
  })

  it("handles non-timeout failures without marking timedOut", async () => {
    vi.mocked(tldwClient.ragSearch).mockRejectedValue(
      new Error("backend unavailable")
    )

    const { result } = createHook()

    await act(async () => {
      await result.current.runQASearch()
    })

    expect(result.current.response).toBeNull()
    expect(result.current.timedOut).toBe(false)
    expect(result.current.loading).toBe(false)
  })

  it("supports answer and chunk actions (copy/insert/pin)", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText }
    })

    const doc = {
      id: "doc-9",
      content: "Chunk body",
      score: 0.88,
      media_id: 77,
      metadata: {
        title: "Doc title",
        source: "knowledge base"
      }
    }

    vi.mocked(tldwClient.ragSearch).mockResolvedValue({
      generated_answer: "Answer text",
      documents: [doc]
    })

    const { result, onInsert, onPin } = createHook("question")

    await act(async () => {
      await result.current.runQASearch()
    })

    await act(async () => {
      await result.current.copyAnswer()
    })
    expect(writeText).toHaveBeenCalledWith("Answer text")

    act(() => {
      result.current.insertAnswer()
    })
    expect(onInsert).toHaveBeenCalledWith("Answer text")

    act(() => {
      result.current.insertChunk(doc)
    })
    expect(onInsert).toHaveBeenCalledWith(
      expect.stringContaining("**Doc title**")
    )
    expect(onInsert).toHaveBeenCalledWith(expect.stringContaining("Chunk body"))

    await act(async () => {
      await result.current.copyChunk(doc, "text")
    })
    expect(writeText).toHaveBeenLastCalledWith(
      expect.stringContaining("Chunk body")
    )

    act(() => {
      result.current.pinChunk(doc)
    })
    expect(onPin).toHaveBeenCalledWith(
      expect.objectContaining({
        content: "Chunk body",
        metadata: expect.objectContaining({
          title: "Doc title",
          media_id: 77,
          id: "doc-9"
        }),
        score: 0.88
      })
    )
  })

  // CC-10 (#3101), decision [2]: Search & Context lists sources first and
  // generates an answer only when the user asks for one.
  it("searches without generating an answer, even when settings enable it", async () => {
    vi.mocked(tldwClient.ragSearch).mockResolvedValue({
      documents: [{ content: "Source chunk" }]
    })
    const { result } = renderHook(() =>
      useQASearch({
        resolvedQuery: "knowledge query",
        draftSettings: { ...baseSettings, enable_generation: true },
        applySettings: vi.fn(),
        onInsert: vi.fn(),
        pinnedResults: [],
        onPin: vi.fn()
      })
    )

    await act(async () => {
      await result.current.runQASearch()
    })

    expect(buildRagSearchRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "knowledge query",
        enable_generation: false
      })
    )
    expect(result.current.response?.documents).toHaveLength(1)
  })

  it("generates an answer for the last search on request and keeps the sources meanwhile", async () => {
    vi.mocked(tldwClient.ragSearch).mockResolvedValueOnce({
      documents: [{ content: "Source chunk" }]
    })
    const { result, rerender } = createHook("first query")

    await act(async () => {
      await result.current.runQASearch()
    })

    let resolveAnswer: (value: unknown) => void = () => {}
    vi.mocked(tldwClient.ragSearch).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveAnswer = resolve
      }) as ReturnType<typeof tldwClient.ragSearch>
    )
    rerender()
    let pending: Promise<void> = Promise.resolve()
    act(() => {
      pending = result.current.generateAnswer()
    })

    expect(result.current.answerLoading).toBe(true)
    expect(result.current.response?.documents[0].content).toBe("Source chunk")

    await act(async () => {
      resolveAnswer({
        generated_answer: "An answer from the sources",
        documents: [{ content: "Source chunk" }]
      })
      await pending
    })

    expect(buildRagSearchRequest).toHaveBeenLastCalledWith(
      expect.objectContaining({
        query: "first query",
        enable_generation: true
      })
    )

    expect(result.current.answerLoading).toBe(false)
    expect(result.current.response?.generatedAnswer).toBe(
      "An answer from the sources"
    )
    expect(result.current.answerError).toBeNull()
  })

  it("keeps the sources and reports an error when answer generation fails", async () => {
    vi.mocked(tldwClient.ragSearch).mockResolvedValueOnce({
      documents: [{ content: "Source chunk" }]
    })
    const { result } = createHook("first query")

    await act(async () => {
      await result.current.runQASearch()
    })
    vi.mocked(tldwClient.ragSearch).mockRejectedValueOnce(new Error("boom"))

    await act(async () => {
      await result.current.generateAnswer()
    })

    expect(result.current.response?.documents[0].content).toBe("Source chunk")
    expect(result.current.response?.generatedAnswer).toBeNull()
    expect(result.current.answerError).toBe(
      "Couldn't generate an answer. Try again, or check the model in Settings."
    )
  })
})
