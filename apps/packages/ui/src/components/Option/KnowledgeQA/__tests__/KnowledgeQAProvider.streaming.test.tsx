import "./knowledgeQaAuthorityFixture"
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

const completeEvent = (outputEmitted: boolean) => ({
  schema_version: 1,
  type: "complete",
  code: "complete",
  upstream_dispatched: true,
  output_emitted: outputEmitted,
  allow_non_stream_fallback: false,
  message: "Search completed.",
})

const preDispatchTransportError = {
  schema_version: 1,
  type: "error",
  code: "stream_transport_unavailable",
  upstream_dispatched: false,
  output_emitted: false,
  allow_non_stream_fallback: true,
  message: "Streaming is unavailable before provider dispatch.",
}

function ContextProbe() {
  const context = useKnowledgeQA()
  React.useLayoutEffect(() => {
    latestContext = context
  }, [context])
  return null
}

describe("KnowledgeQAProvider streaming search", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    sessionStorage.clear()
    latestContext = null
    mockTldwClient.normalizeRagQuery.mockImplementation((query: string) => query)
    trackMetricMock.mockResolvedValue(undefined)
    mockTldwClient.fetchWithAuth.mockResolvedValue({ ok: false, json: async () => [], text: async () => "" })
    ragSearchMock.mockResolvedValue({
      results: [{ id: "fallback-doc" }],
      answer: "Fallback answer",
    })
  })

  it("contains timeout feedback without a console overlay and permits recovery", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    ragSearchStreamMock.mockImplementationOnce(() => { throw new Error("RAG search timed out. private-upstream-sentinel") })
      .mockImplementation(async function* () {
        yield { type: "contexts", contexts: [{ id: "cedar", excerpt: "Cedar opens in January." }] }
        yield { type: "delta", text: "Cedar opens in January [1]." }
        yield completeEvent(true)
      })
    try {
      render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
      await waitFor(() => expect(latestContext).not.toBeNull())
      await act(async () => { await latestContext!.selectThread("local-timeout-recovery") })
      act(() => { latestContext!.setQuery("When does Cedar open?") })
      await act(async () => { await latestContext!.search() })
      expect(latestContext!.error).toMatch(/timed out/i)
      expect(latestContext!.query).toBe("When does Cedar open?")
      expect(latestContext!.isSearching).toBe(false)
      expect(ragSearchMock).not.toHaveBeenCalled()
      expect(consoleError).not.toHaveBeenCalled()
      expect(JSON.stringify(consoleWarn.mock.calls)).not.toContain("private-upstream-sentinel")
      await act(async () => { await latestContext!.search() })
      expect(latestContext!.error).toBeNull()
      expect(latestContext!.answer).toBe("Cedar opens in January [1].")
    } finally { consoleError.mockRestore(); consoleWarn.mockRestore() }
  })

  it.each([true, false])("records completed generation=%s despite controls changing while waiting", async (requested) => {
    let release!: () => void
    const pending = new Promise<void>((resolve) => { release = resolve })
    ragSearchStreamMock.mockImplementation(async function* () {
      await pending
      yield { type: "contexts", contexts: [{ id: "cedar", excerpt: "Cedar opens in January." }] }
      yield completeEvent(false)
    })
    ragSearchMock.mockImplementation(async () => {
      await pending
      return { results: [{ id: "cedar", content: "Cedar opens in January." }], answer: null }
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => { await latestContext!.selectThread("local-generation-snapshot") })
    act(() => { latestContext!.setQuery("When does Cedar open?"); latestContext!.updateSetting("enable_generation", requested) })
    let search!: Promise<void>
    act(() => { search = latestContext!.search() })
    await waitFor(() => expect(requested ? ragSearchStreamMock : ragSearchMock).toHaveBeenCalled())
    act(() => { latestContext!.updateSetting("enable_generation", !requested); release() })
    await act(async () => { await search })
    expect(latestContext!.completedGenerationEnabled).toBe(requested)
    expect(latestContext!.searchHistory[0]?.settingsSnapshot?.enable_generation).toBe(requested)
    expect(latestContext!.settings.enable_generation).toBe(!requested)
    expect(latestContext!.results).toHaveLength(1)
    expect(latestContext!.answer).toBeNull()
  })

  it.each([true, false, undefined])("restores generation intent %s from a saved result independently of controls", async (enabled) => {
    mockTldwClient.fetchWithAuth.mockResolvedValue({
      ok: true,
      json: async () => [{
        id: "saved-answer", role: "assistant", content: "",
        rag_context: {
          search_query: "When does Cedar open?", generated_answer: "",
          settings_snapshot: enabled === undefined ? {} : { enable_generation: enabled },
          retrieved_documents: [{ id: "cedar", excerpt: "Cedar opens in January." }],
        },
      }],
      text: async () => "",
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => { await latestContext!.selectThread("remote-empty-answer") })
    expect(latestContext!.completedGenerationEnabled).toBe(enabled ?? null)
    act(() => { latestContext!.updateSetting("enable_generation", enabled !== true) })
    expect(latestContext!.completedGenerationEnabled).toBe(enabled ?? null)
    expect(latestContext!.results).toHaveLength(1)
  })

  it("sends only completed question-answer pairs as follow-up context", async () => {
    ragSearchStreamMock
      .mockImplementationOnce(() => { throw new Error("Provider unavailable") })
      .mockImplementation(async function* () {
        yield { type: "delta", text: "Project Juniper launches on 18 October." }
        yield completeEvent(true)
      })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => { await latestContext!.selectThread("local-context-filter") })
    for (const query of ["Failed question", "When does Project Juniper launch?", "Who owns it?"]) {
      act(() => { latestContext!.setQuery(query) })
      await act(async () => { await latestContext!.search() })
    }
    expect(ragSearchStreamMock.mock.calls[2][1].chat_history).toEqual([
      { role: "user", content: "When does Project Juniper launch?" },
      { role: "assistant", content: "Project Juniper launches on 18 October." },
    ])
  })

  it("recovers from invalid RAG defaults using an explicit local provider and model", async () => {
    ragSearchStreamMock.mockImplementationOnce(async function* () {
      yield { ...completeEvent(false), type: "error", code: "provider_configuration_invalid", message: "Invalid config" }
    }).mockImplementation(async function* () {
      yield { type: "contexts", contexts: [{ id: "media-42-chunk-7", source: "media_db", source_type: "media_db", source_id: "42", chunk_id: "7", evidence_origin: "local_library", source_status: "searched", title: "Project Juniper", excerpt: "Project Juniper launches on 18 October 2026; Mira Chen owns it." }] }
      yield { type: "delta", text: "Project Juniper launches on 18 October 2026, owned by Mira Chen [1]." }
      yield completeEvent(true)
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => { await latestContext!.selectThread("local-provider-recovery") })
    act(() => { latestContext!.setQuery("When does Project Juniper launch, and who owns it? Cite the source.") })
    await act(async () => { await latestContext!.search() })
    expect(latestContext!.error).toContain("Choose an answer provider and model")
    act(() => {
      latestContext!.updateSetting("generation_provider", "llama")
      latestContext!.updateSetting("generation_model", "local-model.gguf")
    })
    await act(async () => { await latestContext!.search() })
    expect(ragSearchStreamMock.mock.calls[1][1]).toMatchObject({ generation_provider: "llama.cpp", generation_model: "local-model.gguf" })
    expect(latestContext!.error).toBeNull()
    expect(latestContext!.answer).toContain("18 October 2026")
    expect(latestContext!.citations).toHaveLength(1)
    expect(latestContext!.results[0]).toMatchObject({
      sourceId: "42",
      chunkId: "7",
      evidenceOrigin: "local_library",
      sourceStatus: "searched",
      content: "Project Juniper launches on 18 October 2026; Mira Chen owns it.",
      metadata: { source_type: "media_db" },
    })
    expect(latestContext!.citations[0]).toMatchObject({
      documentId: "media-42-chunk-7",
      excerpt: "Project Juniper launches on 18 October 2026; Mira Chen owns it.",
    })
  })

  it("does not invent citations from source prose or invalid numbered references", async () => {
    const answer = "Source: Cedar (media_db). References [Source 1], [0], [99], [abc]."
    ragSearchStreamMock.mockImplementation(async function* () {
      yield { type: "contexts", contexts: [{ id: "cedar", source: "media_db", title: "Cedar", excerpt: "Cedar launches in November." }] }
      yield { type: "delta", text: answer }
      yield completeEvent(true)
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    act(() => { latestContext!.setQuery("When does Cedar launch? Cite the source.") })
    await act(async () => { await latestContext!.search() })
    expect(latestContext!.answer).toBe(answer)
    expect(latestContext!.citations).toEqual([])
  })

  it.each([
    { excluded: 1, retained: 0, warning: "Security settings excluded all retrieved sources" },
    { excluded: 1, retained: 1, warning: "Some retrieved sources were excluded by security settings" },
    { excluded: 0, retained: 0, warning: null },
  ])("distinguishes security exclusion from empty retrieval ($excluded/$retained)", async ({ excluded, retained, warning }) => {
    ragSearchStreamMock.mockImplementation(async function* () {
      yield {
        type: "contexts",
        contexts: retained ? [{ id: "public", excerpt: "Public release notice", source: "media_db" }] : [],
        security_filter: { excluded_count: excluded, retained_count: retained },
      }
      if (retained) yield { type: "delta", text: "Public release notice [1]." }
      yield completeEvent(Boolean(retained))
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => { await latestContext!.selectThread("local-security-outcome") })
    act(() => { latestContext!.setQuery("When is the release?") })
    await act(async () => { await latestContext!.search() })
    if (warning) expect(latestContext!.queryWarning).toContain(warning)
    else expect(latestContext!.queryWarning).toBeNull()
    expect(latestContext!.results).toHaveLength(retained)
    expect(latestContext!.answer).toBe(retained ? "Public release notice [1]." : null)
    expect(ragSearchMock).not.toHaveBeenCalled()
  })

  it("explains security exclusion for a retrieval-only response", async () => {
    ragSearchMock.mockResolvedValue({ documents: [], metadata: { security_filter: { excluded_count: 1, retained_count: 0 } } })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    act(() => {
      latestContext!.setQuery("When is the release?")
      latestContext!.updateSetting("enable_generation", false)
    })
    await act(async () => { await latestContext!.search() })
    expect(latestContext!.queryWarning).toContain("Security settings excluded all retrieved sources")
    expect(latestContext!.answer).toBeNull()
  })

  it("keeps the live Cedar RRF score for ranking without degrading a cited answer", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      yield { type: "contexts", contexts: [{ id: "late_chunk:2:0", title: "Cedar public launch brief", score: 0.004838709677419355, source: "media_db", source_id: "2", chunk_id: "late_chunk:2:0", excerpt: "Project Cedar launches on 22 November 2026. The project lead is Mira Chen." }] }
      yield { type: "delta", text: "Project Cedar launches on 22 November 2026, and the project lead is Mira Chen [1]." }
      yield completeEvent(true)
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => { await latestContext!.selectThread("local-cedar-rank") })
    act(() => { latestContext!.setQuery("When does Project Cedar launch, and who owns it? Cite the source.") })
    await act(async () => { await latestContext!.search() })
    expect(latestContext!.results[0].score).toBe(0.004838709677419355)
    expect(latestContext!.searchDetails?.averageRelevance).toBeNull()
    expect(latestContext!.answerTrustReasonCodes).not.toContain("low_relevance")
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
            score_kind: "relevance_probability",
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
      yield completeEvent(true)
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
      yield completeEvent(true)
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

  it("falls back only for a certified pre-dispatch stream transport error", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      yield preDispatchTransportError
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
            { id: "cand-1", title: "Near miss", score: 25, reason: "below threshold" },
            { id: "cand-negative", title: "Negative rank", score: -2.5 },
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
            score: 25,
            reason: "below threshold",
          }),
          expect.objectContaining({ id: "cand-negative", score: -2.5 }),
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

  it.each([
    ["a trailing delta", { type: "delta", text: "late output" }],
    ["a second terminal", completeEvent(false)],
  ])(
    "fails closed when a certified replay terminal is followed by %s",
    async (_name, trailingEvent) => {
      ragSearchStreamMock.mockImplementation(async function* () {
        yield preDispatchTransportError
        yield trailingEvent
      })

      render(
        <KnowledgeQAProvider>
          <ContextProbe />
        </KnowledgeQAProvider>
      )

      await waitFor(() => expect(latestContext).not.toBeNull())
      await act(async () => {
        await latestContext!.selectThread("local-trailing-terminal-event")
      })
      act(() => {
        latestContext!.setQuery("do not replay this stream")
      })

      await act(async () => {
        await latestContext!.search()
      })

      expect(ragSearchMock).not.toHaveBeenCalled()
      expect(latestContext!.error).toBe("Invalid RAG terminal stream event.")
    }
  )

  it("does not replay a clean empty stream completion", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      yield {
        type: "contexts",
        contexts: [],
        source_status: {
          notes: { status: "empty", count: 0, reason: "no_matching_entries" },
        },
      }
      yield completeEvent(false)
    })
    ragSearchMock.mockResolvedValue({
      results: [
        {
          id: "note-scoped",
          content: "Scoped note answers must stay inside the selected source.",
          metadata: {
            title: "Knowledge QA UAT Scoped Note",
            source_type: "notes",
            source_id: "note-scoped",
          },
          score: 0.91,
        },
      ],
      generated_answer:
        "Scoped note answers must stay inside the selected source [1].",
      metadata: {
        source_status: {
          notes: { status: "searched", count: 1 },
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
      await latestContext!.selectThread("local-empty-stream-fallback")
    })

    act(() => {
      latestContext!.updateSetting("sources", ["notes"])
      latestContext!.updateSetting("include_note_ids", ["note-scoped"])
      latestContext!.setQuery("What does the selected note say?")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(ragSearchStreamMock).toHaveBeenCalledTimes(1)
    expect(ragSearchMock).not.toHaveBeenCalled()
    expect(latestContext!.results).toHaveLength(0)
    expect(latestContext!.answer).toBeNull()
    expect(latestContext!.isSearching).toBe(false)
    expect(trackMetricMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "search_complete",
        used_streaming: true,
        has_answer: false,
      })
    )
  })

  it("does not replay a terminal provider credential error", async () => {
    const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {})
    ragSearchStreamMock.mockImplementation(async function* () {
      yield {
        schema_version: 1,
        type: "error",
        code: "provider_authentication_failed",
        upstream_dispatched: true,
        output_emitted: false,
        allow_non_stream_fallback: false,
        message: "The selected provider credentials could not be authenticated.",
      }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-terminal-credential-error")
    })

    act(() => {
      latestContext!.setQuery("use my provider")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(ragSearchMock).not.toHaveBeenCalled()
    expect(latestContext!.error).toBe(
      "The selected provider credentials could not be authenticated."
    )
    expect(latestContext!.answerTrustState).toBe("failed_search")
    expect(trackMetricMock).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "search_complete" })
    )
    expect(consoleWarn).toHaveBeenCalledWith(
      "Search failed:",
      "provider_authentication_failed"
    )
    consoleWarn.mockRestore()
  })

  it.each([
    [
      502,
      "provider_unavailable",
      "The selected provider is currently unavailable.",
    ],
    [
      503,
      "credential_store_unavailable",
      "Provider credential storage is temporarily unavailable.",
    ],
  ])(
    "stops after certified stream fallback and sanitized HTTP %i failure",
    async (status, code, expectedMessage) => {
      const sentinel = `sk-nonstream-${status}-/Users/private/provider.log`
      const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {})
      localStorage.setItem("tldwConfig", "keep-api-key-config")
      localStorage.setItem("access_token", "keep-access-token")
      sessionStorage.setItem("tldwManualSessionApiKey", "keep-session-api-key")
      ragSearchStreamMock.mockImplementation(async function* () {
        yield preDispatchTransportError
      })
      ragSearchMock.mockRejectedValue(
        Object.assign(new Error(sentinel), { code, status })
      )

      render(
        <KnowledgeQAProvider>
          <ContextProbe />
        </KnowledgeQAProvider>
      )

      await waitFor(() => expect(latestContext).not.toBeNull())
      await act(async () => {
        await latestContext!.selectThread(`local-nonstream-${status}`)
      })
      act(() => {
        latestContext!.setQuery(`provider failure ${status}`)
      })

      await act(async () => {
        await latestContext!.search()
      })

      expect(ragSearchStreamMock).toHaveBeenCalledTimes(1)
      expect(ragSearchMock).toHaveBeenCalledTimes(1)
      expect(latestContext!.error).toBe(expectedMessage)
      expect(localStorage.getItem("tldwConfig")).toBe("keep-api-key-config")
      expect(localStorage.getItem("access_token")).toBe("keep-access-token")
      expect(sessionStorage.getItem("tldwManualSessionApiKey")).toBe(
        "keep-session-api-key"
      )
      expect(consoleWarn.mock.calls.filter(([label]) => label === "Search failed:")).toHaveLength(1)
      const logged = JSON.stringify(consoleWarn.mock.calls)
      expect(logged).toContain(code)
      expect(logged).not.toContain(sentinel)
      consoleWarn.mockRestore()
    }
  )

  it("sanitizes an unknown terminal stream error without replay", async () => {
    const sentinel = "sk-stream-secret-/Users/private/provider.log"
    const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {})
    ragSearchStreamMock.mockImplementation(async function* () {
      yield {
        schema_version: 1,
        type: "error",
        code: "unknown_provider_failure",
        status_code: 502,
        upstream_dispatched: true,
        output_emitted: false,
        allow_non_stream_fallback: false,
        message: sentinel,
      }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-unknown-stream-error")
    })
    act(() => {
      latestContext!.setQuery("unknown stream failure")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(ragSearchStreamMock).toHaveBeenCalledTimes(1)
    expect(ragSearchMock).not.toHaveBeenCalled()
    expect(latestContext!.error).toBe(
      "The selected provider is currently unavailable."
    )
    const logged = JSON.stringify(consoleWarn.mock.calls)
    expect(logged).toContain("provider_unavailable")
    expect(logged).not.toContain("unknown_provider_failure")
    expect(logged).not.toContain(sentinel)
    consoleWarn.mockRestore()
  })

  it("sanitizes an unknown non-stream error after certified fallback", async () => {
    const sentinel = "sk-nonstream-secret-/Users/private/provider.log"
    const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {})
    ragSearchStreamMock.mockImplementation(async function* () {
      yield preDispatchTransportError
    })
    ragSearchMock.mockRejectedValue(
      Object.assign(new Error(sentinel), {
        code: "unknown_provider_failure",
        status: 503,
      })
    )

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-unknown-nonstream-error")
    })
    act(() => {
      latestContext!.setQuery("unknown non-stream failure")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(ragSearchStreamMock).toHaveBeenCalledTimes(1)
    expect(ragSearchMock).toHaveBeenCalledTimes(1)
    expect(latestContext!.error).toBe("RAG search failed due to a server error.")
    const logged = JSON.stringify(consoleWarn.mock.calls)
    expect(logged).not.toContain("unknown_provider_failure")
    expect(logged).not.toContain(sentinel)
    consoleWarn.mockRestore()
  })

  it("does not replay after partial provider output", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      yield { type: "delta", text: "Partial answer" }
      yield {
        schema_version: 1,
        type: "error",
        code: "provider_unavailable",
        upstream_dispatched: true,
        output_emitted: true,
        allow_non_stream_fallback: false,
        message: "The selected provider is currently unavailable.",
      }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-partial-terminal-error")
    })

    act(() => {
      latestContext!.setQuery("partial failure")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(ragSearchMock).not.toHaveBeenCalled()
    expect(latestContext!.error).toBe(
      "The selected provider is currently unavailable."
    )
    expect(latestContext!.answerTrustState).toBe("failed_search")
  })

  it("fails closed for malformed terminal events", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      yield {
        schema_version: 2,
        type: "error",
        code: "stream_transport_unavailable",
        upstream_dispatched: false,
        output_emitted: false,
        allow_non_stream_fallback: true,
        message: "Unknown contract version.",
      }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await act(async () => {
      await latestContext!.selectThread("local-malformed-terminal-error")
    })
    act(() => {
      latestContext!.setQuery("malformed terminal")
    })

    await act(async () => {
      await latestContext!.search()
    })

    expect(ragSearchMock).not.toHaveBeenCalled()
    expect(latestContext!.error).toBe("Invalid RAG terminal stream event.")
  })

  it("normalizes whitespace-only non-stream answers to null", async () => {
    ragSearchStreamMock.mockImplementation(async function* () {
      yield preDispatchTransportError
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
      yield preDispatchTransportError
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
      yield preDispatchTransportError
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
      yield preDispatchTransportError
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
