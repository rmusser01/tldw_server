import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"

const ragSearchMock = vi.fn()
const ragSearchStreamMock = vi.fn()
const messageOpenMock = vi.fn()
const trackMetricMock = vi.fn()
const mockTldwClient = vi.hoisted(() => ({
  initialize: vi.fn().mockResolvedValue(undefined),
  fetchWithAuth: vi.fn().mockResolvedValue({
    ok: false,
    json: async () => [],
    text: async () => "",
  }),
  normalizeRagQuery: vi.fn((query: string) => query),
  ragSearch: vi.fn((...args: unknown[]) => ragSearchMock(...args)),
  ragSearchStream: vi.fn(async function* (
    this: { normalizeRagQuery: (query: string) => string },
    ...args: unknown[]
  ) {
    yield* ragSearchStreamMock.apply(this, args as [])
  }),
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [undefined],
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: messageOpenMock,
  }),
}))

vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: (...args: unknown[]) => trackMetricMock(...args),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: mockTldwClient,
}))

let latestContext: ReturnType<typeof useKnowledgeQA> | null = null

function ContextProbe() {
  latestContext = useKnowledgeQA()
  return null
}

describe("KnowledgeQAProvider streaming search", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    latestContext = null
    mockTldwClient.normalizeRagQuery.mockImplementation((query: string) => query)
    trackMetricMock.mockResolvedValue(undefined)
    ragSearchMock.mockResolvedValue({
      results: [{ id: "fallback-doc" }],
      answer: "Fallback answer",
    })
  })

  it("applies streamed contexts/deltas incrementally and finalizes answer", async () => {
    let releaseFinalDelta: (() => void) | null = null
    ragSearchStreamMock.mockImplementation(async function* () {
      yield {
        type: "contexts",
        contexts: [
          {
            id: "doc-1",
            title: "Doc One",
            score: 0.92,
            url: "https://example.com/doc-1",
            source: "media_db",
          },
        ],
        why: {
          topicality: 0.88,
          diversity: 0.42,
          freshness: null,
        },
        source_status: {
          media_db: { status: "searched", count: 1 },
          prompts: { status: "empty", count: 0, reason: "no_matching_entries" },
        },
      }
      yield { type: "delta", text: "Hello" }
      await new Promise<void>((resolve) => {
        releaseFinalDelta = resolve
      })
      yield { type: "delta", text: " world" }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-stream-test")
    })

    act(() => {
      latestContext!.setQuery("stream this")
    })

    let searchPromise: Promise<void> | null = null
    act(() => {
      searchPromise = latestContext!.search()
    })

    await waitFor(() => expect(latestContext!.results.length).toBe(1))
    await waitFor(() => {
      expect(latestContext!.answer).toBe("Hello")
      expect(latestContext!.isSearching).toBe(true)
    })

    act(() => {
      releaseFinalDelta?.()
    })

    await act(async () => {
      await searchPromise
    })

    expect(latestContext!.answer).toBe("Hello world")
    expect(latestContext!.isSearching).toBe(false)
    expect(latestContext!.searchDetails).toEqual(
      expect.objectContaining({
        rerankingEnabled: true,
        averageRelevance: 0.92,
        whyTheseSources: expect.objectContaining({
          topicality: 0.88,
          diversity: 0.42,
        }),
        sourceStatus: {
          media_db: { status: "searched", count: 1 },
          prompts: { status: "empty", count: 0, reason: "no_matching_entries" },
        },
      })
    )
    expect(ragSearchStreamMock).toHaveBeenCalledTimes(1)
    expect(ragSearchMock).not.toHaveBeenCalled()
    expect(trackMetricMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "search_complete",
        used_streaming: true,
        has_answer: true,
      })
    )
  })

  it("invokes ragSearchStream with the client binding intact", async () => {
    mockTldwClient.normalizeRagQuery.mockImplementation((query: string) =>
      query.toUpperCase()
    )
    ragSearchStreamMock.mockImplementation(async function* (
      this: { normalizeRagQuery: (query: string) => string },
      query: string
    ) {
      const normalized = this.normalizeRagQuery(query)
      yield {
        type: "contexts",
        contexts: [
          {
            id: "doc-bound",
            title: "Bound Doc",
            score: 0.77,
            source: "media_db",
          },
        ],
      }
      yield { type: "delta", text: normalized }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-stream-this-binding")
    })

    act(() => {
      latestContext!.setQuery("bound stream")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(mockTldwClient.normalizeRagQuery).toHaveBeenCalledWith("bound stream")
    expect(ragSearchMock).not.toHaveBeenCalled()
    expect(latestContext!.answer).toBe("BOUND STREAM")
  })

  it("falls back to non-stream rag search when stream path fails", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      throw new Error("stream endpoint unavailable")
    })
    ragSearchMock.mockResolvedValue({
      results: [{ id: "fallback-doc", metadata: { title: "Fallback" } }],
      answer: "Fallback answer",
      expanded_queries: ["fallback path alt"],
      faithfulness: {
        faithfulness_score: "87",
        total_claims: "5",
        supported_claims: "4",
        unsupported_claims: "1",
      },
      verification_report: {
        total_claims: "5",
        verified_count: "4",
        verification_rate: 80,
        coverage: "75",
      },
      metadata: {
        source_status: {
          media_db: { status: "searched", count: 1 },
          world_books: {
            status: "unavailable",
            count: 0,
            reason: "no_retriever_configured",
          },
        },
        web_fallback: {
          triggered: true,
          engine_used: "duckduckgo",
        },
        retrieval_metrics: {
          documents_considered: "25",
          also_considered: [
            { id: "cand-1", title: "Near miss", score: 41, reason: "below threshold" },
          ],
        },
      },
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-stream-fallback")
    })

    act(() => {
      latestContext!.setQuery("fallback path")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(ragSearchStreamMock).toHaveBeenCalledTimes(1)
    expect(ragSearchMock).toHaveBeenCalledTimes(1)
    expect(latestContext!.answer).toBe("Fallback answer")
    expect(latestContext!.isSearching).toBe(false)
    expect(latestContext!.searchDetails).toEqual(
      expect.objectContaining({
        expandedQueries: ["fallback path alt"],
        webFallbackTriggered: true,
        webFallbackEngine: "duckduckgo",
        faithfulnessScore: 0.87,
        faithfulnessTotalClaims: 5,
        faithfulnessSupportedClaims: 4,
        faithfulnessUnsupportedClaims: 1,
        verificationRate: 0.8,
        verificationCoverage: 0.75,
        verificationReportAvailable: true,
        sourceStatus: {
          media_db: { status: "searched", count: 1 },
          world_books: {
            status: "unavailable",
            count: 0,
            reason: "no_retriever_configured",
          },
        },
        candidatesConsidered: 25,
        candidatesReturned: 1,
        candidatesRejected: 24,
        alsoConsidered: [
          expect.objectContaining({
            id: "cand-1",
            title: "Near miss",
            score: 0.41,
            reason: "below threshold",
          }),
        ],
      })
    )
    expect(trackMetricMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "search_complete",
        used_streaming: false,
        has_answer: true,
      })
    )
  })

  it("normalizes whitespace-only non-stream answers to null", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      throw new Error("stream endpoint unavailable")
    })
    ragSearchMock.mockResolvedValue({
      results: [{ id: "blank-answer-doc", metadata: { title: "Blank Answer Doc" } }],
      answer: "   ",
      metadata: {},
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-blank-answer-test")
    })

    act(() => {
      latestContext!.setQuery("blank answer query")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(ragSearchMock).toHaveBeenCalledTimes(1)
    expect(latestContext!.results).toHaveLength(1)
    expect(latestContext!.answer).toBeNull()
    expect(latestContext!.isSearching).toBe(false)
  })

  it("removes out-of-scope local results without remapping surviving citation indexes", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      throw new Error("stream endpoint unavailable")
    })
    ragSearchMock.mockResolvedValue({
      results: [
        {
          id: "excluded-doc",
          content: "Excluded evidence.",
          metadata: { source_type: "media_db", source_id: "99" },
          score: 0.92,
        },
        {
          id: "allowed-doc",
          content: "Allowed evidence.",
          metadata: { source_type: "media_db", source_id: "42" },
          score: 0.91,
        },
      ],
      answer: "The excluded claim cites [1], while the scoped claim cites [2].",
      metadata: {},
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-scope-validation")
    })

    act(() => {
      latestContext!.updateSetting("sources", ["media_db"])
      latestContext!.updateSetting("include_media_ids", [42])
      latestContext!.updateSetting("enable_web_fallback", false)
      latestContext!.setQuery("only selected source")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(latestContext!.results).toHaveLength(1)
    expect(latestContext!.results[0].id).toBe("allowed-doc")
    expect(latestContext!.results[0].metadata?.original_result_index).toBe(1)
    expect(latestContext!.citations).toEqual([
      expect.objectContaining({ index: 2, documentId: "allowed-doc" }),
    ])
    expect(latestContext!.queryWarning).toBe(
      "Some returned sources were hidden because they were outside the selected source scope."
    )
  })

  it("surfaces query-length warning when a submitted query exceeds backend limits", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      throw new Error("stream endpoint unavailable")
    })
    ragSearchMock.mockResolvedValue({
      results: [{ id: "fallback-doc", metadata: { title: "Fallback" } }],
      answer: "Fallback answer",
      metadata: {},
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-long-query-warning")
    })

    act(() => {
      latestContext!.setQuery("x".repeat(21000))
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(latestContext!.queryWarning).toBe(
      "Query exceeded 20,000 characters and was shortened before search."
    )

    act(() => {
      latestContext!.setQuery("short query")
    })
    expect(latestContext!.queryWarning).toBeNull()
  })

  it("merges pinned source filters into rag search options", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      throw new Error("stream endpoint unavailable")
    })
    ragSearchMock.mockResolvedValue({
      results: [],
      answer: "Pinned filter check",
      metadata: {},
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-pinned-filter-test")
    })

    act(() => {
      latestContext!.updateSetting("include_media_ids", [7])
      ;(latestContext as any).updateSetting("include_note_ids", [
        "note-base-uuid",
      ])
      ;(latestContext as any).setPinnedSourceFilters({
        mediaIds: [42],
        noteIds: ["note-pinned-uuid"],
      })
      latestContext!.setQuery("pinned filters query")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(ragSearchMock).toHaveBeenCalledTimes(1)
    const ragOptions = ragSearchMock.mock.calls[0]?.[1] as Record<string, unknown>
    expect(ragOptions.include_media_ids).toEqual(
      expect.arrayContaining([7, 42])
    )
    expect(ragOptions.include_note_ids).toEqual(
      expect.arrayContaining(["note-base-uuid", "note-pinned-uuid"])
    )
  })
})
