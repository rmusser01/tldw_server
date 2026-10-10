import "./knowledgeQaAuthorityFixture"
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { beforeEach, describe, expect, it, vi } from "vitest"
import knowledgeEn from "@/assets/locale/en/knowledge.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"
import { AnswerPanel } from "../AnswerPanel"
import { ExportDialog } from "../ExportDialog"
import { readKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance"

const ragSearchMock = vi.fn()
const ragSearchStreamMock = vi.fn()
const messageOpenMock = vi.fn()
const trackMetricMock = vi.fn()
const queuePrefillMock = vi.fn()
const mockTldwClient = vi.hoisted(() => ({
  initialize: vi.fn().mockResolvedValue(undefined),
  fetchWithAuth: vi.fn().mockResolvedValue({
    ok: false,
    json: async () => [],
    text: async () => "",
  }),
  normalizeRagQuery: vi.fn((query: string) => query),
  createNote: vi.fn().mockResolvedValue({ id: "partial-scope-receipt" }),
  addChatMessage: vi.fn().mockResolvedValue({ id: "partial-message" }),
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

vi.mock("@/hooks/useHomeMilestoneScope", () => ({
  useHomeMilestoneScope: () => "test-owner",
}))

vi.mock("@/utils/research-workspace-prefill", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/research-workspace-prefill")>()),
  queueResearchWorkspacePrefill: (...args: unknown[]) => queuePrefillMock(...args),
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
    ragSearchStreamMock.mockReset()
    localStorage.clear()
    sessionStorage.clear()
    latestContext = null
    mockTldwClient.normalizeRagQuery.mockImplementation((query: string) => query)
    trackMetricMock.mockResolvedValue(undefined)
    queuePrefillMock.mockResolvedValue(undefined)
    mockTldwClient.fetchWithAuth.mockResolvedValue({ ok: false, json: async () => [], text: async () => "" })
    ragSearchMock.mockResolvedValue({
      results: [{ id: "fallback-doc" }],
      answer: "Fallback answer",
    })
  })

  it("retains received scoped answer and citations when the iterator fails before its queued flush", async () => {
    ragSearchStreamMock.mockImplementationOnce(async function* () {
      yield { type: "contexts", contexts: [{ id: "note-a", source_type: "notes", source_id: "note-a", excerpt: "A evidence." }] }
      yield { type: "delta", text: "Received answer [1]" }
      throw new Error("Synthetic stream parser failure")
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await waitFor(() => expect(latestContext).not.toBeNull())
    act(() => { latestContext!.setQuery("retain received evidence") })
    await act(async () => { await latestContext!.search() })
    expect(latestContext!.answer).toBe("Received answer [1]")
    expect(latestContext!.citations).toEqual([expect.objectContaining({ index: 1, documentId: "note-a" })])
    expect(latestContext!.answerTrustState).toBe("failed_search")
    expect(latestContext!.isSearching).toBe(false)
    expect(ragSearchMock).not.toHaveBeenCalled()
    expect(ragSearchStreamMock).toHaveBeenCalledTimes(1)
  })

  it.each(["cancel", "clear", "supersede"] as const)("drops queued output before a held generator settles after %s", async action => {
    let queued!: () => void
    let finishOld!: () => void
    let startedNew!: () => void
    let finishNew!: () => void
    const oldQueued = new Promise<void>(resolve => { queued = resolve })
    const oldEnd = new Promise<void>(resolve => { finishOld = resolve })
    const newStarted = new Promise<void>(resolve => { startedNew = resolve })
    const newEnd = new Promise<void>(resolve => { finishNew = resolve })
    let oldSignal!: AbortSignal
    ragSearchStreamMock.mockImplementationOnce(async function* (_query: string, options: { signal: AbortSignal }) {
      oldSignal = options.signal
      yield { type: "contexts", contexts: [{ id: "note-a", source_type: "notes", source_id: "note-a", excerpt: "A evidence." }] }
      yield { type: "delta", text: "Queued old answer [1]." }
      queued()
      await oldEnd
      yield completeEvent(true)
    }).mockImplementationOnce(async function* () {
      yield { type: "contexts", contexts: [{ id: "note-b", source_type: "notes", source_id: "note-b", excerpt: "B evidence." }] }
      startedNew()
      await newEnd
      yield completeEvent(false)
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await act(async () => { await latestContext!.selectThread("local-held-stream-owner") })
    act(() => {
      latestContext!.setQuery("Old question")
      latestContext!.updateSetting("sources", ["notes"])
      latestContext!.updateSetting("include_note_ids", ["note-a", "note-b"])
      latestContext!.updateSetting("enable_web_fallback", false)
    })
    vi.useFakeTimers()
    let oldRequest!: Promise<void>
    let newRequest: Promise<void> | undefined
    try {
      const baselineTimers = vi.getTimerCount()
      await act(async () => { oldRequest = latestContext!.search(); await oldQueued })
      expect(vi.getTimerCount()).toBe(baselineTimers + 1)
      expect(latestContext!.answer).toBeNull()
      if (action === "supersede") {
        act(() => { latestContext!.setQuery("New question") })
        await act(async () => { newRequest = latestContext!.search(); await newStarted })
      } else {
        act(() => {
          if (action === "cancel") latestContext!.cancelSearch()
          else latestContext!.clearResults()
        })
      }
      expect(oldSignal.aborted).toBe(true)
      const retained = {
        answer: latestContext!.answer, results: latestContext!.results,
        resultQuery: latestContext!.resultQuery, lastSearchScope: latestContext!.lastSearchScope,
        citations: latestContext!.citations, isSearching: latestContext!.isSearching,
        queryStage: latestContext!.queryStage,
      }
      expect(vi.getTimerCount()).toBe(baselineTimers)
      await act(async () => { await vi.advanceTimersByTimeAsync(80) })
      expect(latestContext).toMatchObject(retained)
      expect(latestContext!.answer || "").not.toContain("Queued old answer")
      expect(vi.getTimerCount()).toBe(baselineTimers)
      expect(ragSearchMock).not.toHaveBeenCalled()
    } finally {
      finishOld()
      finishNew()
      await act(async () => { await oldRequest; await newRequest })
      vi.useRealTimers()
    }
  })

  it.each(["markdown", "pdf", "note", "handoff"])("keeps original citation [2] bound to B after excluding C in %s", async (destination) => {
    let finish!: () => void
    const end = new Promise<void>(resolve => { finish = resolve })
    ragSearchStreamMock.mockImplementationOnce(async function* () {
      yield { type: "contexts", contexts: [
        { id: "note-c", source_type: "notes", source_id: "note-c", title: "C", excerpt: "Excluded C evidence." },
        { id: "note-b", source_type: "notes", source_id: "note-b", title: "B", excerpt: "B evidence." },
        { id: "note-d", source_type: "notes", source_id: "note-d", title: "D", excerpt: "D evidence." },
      ] }
      yield { type: "delta", text: "B claim [2]." }
      await end
      throw Object.assign(new Error("Cancelled partial output"), { name: "AbortError" })
    })
    mockTldwClient.fetchWithAuth.mockResolvedValue({ ok: true, json: async () => [], text: async () => "" })
    const i18n = createInstance().use(ICUWithInterpolation)
    await i18n.init({ lng: "en", fallbackLng: "en", defaultNS: "knowledge", resources: { en: { knowledge: knowledgeEn } } })
    render(<I18nextProvider i18n={i18n}><MemoryRouter><KnowledgeQAProvider><ContextProbe /><AnswerPanel /><ExportDialog open onClose={vi.fn()} /></KnowledgeQAProvider></MemoryRouter></I18nextProvider>)
    await act(async () => { await latestContext!.selectThread("remote-citation-identity") })
    act(() => {
      latestContext!.setQuery("B question")
      latestContext!.updateSetting("sources", ["notes"])
      latestContext!.updateSetting("include_note_ids", ["note-b", "note-d"])
      latestContext!.updateSetting("enable_web_fallback", false)
    })
    let pending!: Promise<void>
    act(() => { pending = latestContext!.search() })
    try {
      await waitFor(() => expect(latestContext!.answer).toBe("B claim [2]."))
      expect(latestContext!.results.map(result => [result.id, result.metadata?.original_result_index])).toEqual([["note-b", 1], ["note-d", 2]])
      expect(latestContext!.citations.map(citation => [citation.index, citation.documentId])).toEqual([[2, "note-b"]])
      await act(async () => { latestContext!.cancelSearch(); finish(); await pending })
      expect(latestContext!.queryStage).toBe("cancelled")
      if (destination === "handoff") {
        fireEvent.click(screen.getByRole("button", { name: "Continue in Research Workspace" }))
        await waitFor(() => expect(queuePrefillMock).toHaveBeenCalledTimes(1))
        const prefill = queuePrefillMock.mock.calls[0][0]
        expect(prefill.citations).toEqual([2])
        expect(prefill.sources.map(source => [source.originalId, source.citationIndex])).toEqual([["note-b", 2], ["note-d", undefined]])
        expect(prefill.scope.include_note_ids).toEqual(["note-b", "note-d"])
      } else if (destination === "note") {
        fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
        await screen.findByRole("link", { name: "Open saved note" })
        const [body, fields, request] = mockTldwClient.createNote.mock.calls[0]
        expect(fields.knowledge_provenance.sources.map(source => [source.originalId, source.citationIndex])).toEqual([["note-b", 2], ["note-d", undefined]])
        expect(readKnowledgeNoteProvenance(body)?.sources).toEqual(fields.knowledge_provenance.sources)
        expect(body).toContain("- [2] B maps to Source 2.")
        expect(body).not.toContain("Excluded C evidence.")
        expect(request.idempotencyKey).toBeTruthy()
      } else {
        if (destination === "pdf") fireEvent.click(screen.getByRole("button", { name: /PDF/i }))
        fireEvent.click(screen.getByRole("button", { name: "Export" }))
        await screen.findByText("Preview")
        const preview = screen.getByText((_, element) => element?.tagName === "PRE").textContent!
        expect(preview).toContain("- [2] B maps to Source 2.")
        expect(preview).toContain("### [2] B")
        expect(preview).toContain("### [3] D")
        expect(preview).toMatch(/## Bibliography[\s\S]*\[2\] B[\s\S]*\[3\] D/)
        expect(preview).not.toContain("Excluded C evidence.")
      }
    } finally {
      finish()
      await act(async () => { await pending })
    }
  })

  it.each([
    { firstEvent: "contexts", includeAllowed: false },
    { firstEvent: "delta", includeAllowed: false },
    { firstEvent: "contexts", includeAllowed: true },
    { firstEvent: "delta", includeAllowed: true },
  ])("validates every partial publication before cancellation: %j", async ({ firstEvent, includeAllowed }) => {
    let publishSecond!: () => void
    let finish!: () => void
    const second = new Promise<void>(resolve => { publishSecond = resolve })
    const end = new Promise<void>(resolve => { finish = resolve })
    const draft = includeAllowed ? "Excluded claim [1]; selected claim [2]." : "Excluded claim [1]."
    ragSearchStreamMock.mockImplementationOnce(async function* () {
      const contexts = { type: "contexts", contexts: [
        { id: "note-c", source_type: "notes", source_id: "note-c", excerpt: "Excluded evidence C." },
        ...(includeAllowed ? [{ id: "note-b", source_type: "notes", source_id: "note-b", excerpt: "Selected evidence B." }] : []),
      ] }
      const delta = { type: "delta", text: draft }
      yield firstEvent === "contexts" ? contexts : delta
      await second
      yield firstEvent === "contexts" ? delta : contexts
      await end
      throw Object.assign(new Error("Aborted after partial output"), { name: "AbortError" })
    })
    mockTldwClient.fetchWithAuth.mockResolvedValue({ ok: true, json: async () => [], text: async () => "" })
    const i18n = createInstance().use(ICUWithInterpolation)
    await i18n.init({ lng: "en", fallbackLng: "en", defaultNS: "knowledge", resources: { en: { knowledge: knowledgeEn } } })
    render(<I18nextProvider i18n={i18n}><MemoryRouter><KnowledgeQAProvider><ContextProbe /><AnswerPanel /><ExportDialog open onClose={vi.fn()} /></KnowledgeQAProvider></MemoryRouter></I18nextProvider>)
    await act(async () => { await latestContext!.selectThread("remote-partial-validation") })
    act(() => {
      latestContext!.setQuery("Selected note B?")
      latestContext!.updateSetting("sources", ["notes"])
      latestContext!.updateSetting("include_note_ids", ["note-b"])
      latestContext!.updateSetting("enable_web_fallback", false)
    })
    let pending!: Promise<void>
    act(() => { pending = latestContext!.search() })
    try {
      await waitFor(() => expect(latestContext!.resultQuery).toBe("Selected note B?"))
      expect(latestContext!.results.map(result => result.id)).toEqual(firstEvent === "contexts" && includeAllowed ? ["note-b"] : [])
      act(() => { publishSecond() })
      await waitFor(() => expect(latestContext!.answer).toBe(draft))
      await waitFor(() => expect(latestContext!.queryStage).toBe(firstEvent === "contexts" ? "generating" : "ranking"))
      expect(latestContext!.results.map(result => result.id)).toEqual(includeAllowed ? ["note-b"] : [])
      expect(latestContext!.citations.map(citation => [citation.index, citation.documentId])).toEqual(includeAllowed ? [[2, "note-b"]] : [])
      expect(latestContext!.answerTrustState).toBe(includeAllowed ? "cited_answer" : "no_results")
      expect(latestContext!.answerTrustReasonCodes).toEqual(includeAllowed ? [] : ["no_evidence"])
      expect(latestContext!.queryWarning).toContain("outside the selected source scope")
      act(() => { latestContext!.cancelSearch(); finish() })
      await act(async () => { await pending })
      expect(latestContext!.lastSearchScope).toMatchObject({ sources: ["notes"], includeNoteIds: ["note-b"], webFallback: false })
      expect(latestContext!.results.map(result => result.id)).toEqual(includeAllowed ? ["note-b"] : [])
      expect(latestContext!.answer).toBe(draft)
      expect(latestContext!.queryStage).toBe("cancelled")
      expect(ragSearchMock).not.toHaveBeenCalled()
      if (includeAllowed) {
        fireEvent.click(screen.getByRole("button", { name: "Continue in Research Workspace" }))
        await waitFor(() => expect(queuePrefillMock).toHaveBeenCalledTimes(1))
        expect(queuePrefillMock.mock.calls[0][0]).toMatchObject({
          scope: { sources: ["notes"], include_note_ids: ["note-b"] },
          sources: [expect.objectContaining({ originalId: "note-b", excerpt: "Selected evidence B." })],
        })
        fireEvent.click(screen.getByLabelText("Settings snapshot"))
        fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
        expect(await screen.findByRole("link", { name: "Open saved note" })).toHaveAttribute("href", "/notes?source_ref_id=partial-scope-receipt")
        const [body, fields] = mockTldwClient.createNote.mock.calls[0]
        expect(body).not.toContain("Excluded evidence C.")
        expect(fields.knowledge_provenance).toMatchObject({
          scope: { sources: ["notes"], include_note_ids: ["note-b"] },
          sources: [expect.objectContaining({ originalId: "note-b" })],
        })
      }
    } finally {
      act(() => { latestContext!.cancelSearch(); publishSecond(); finish() })
      await act(async () => { await pending })
    }
  })

  it.each([false, true])("fences buffered events yielded after abort (prior partial: %s)", async (publishBeforeCancel) => {
    let drain!: () => void
    const buffered = new Promise<void>(resolve => { drain = resolve })
    let observedAbort = false
    ragSearchStreamMock.mockImplementationOnce(async function* () {
      yield { type: "contexts", contexts: [{ id: "note-a", source_type: "notes", source_id: "note-a", excerpt: "A evidence." }] }
      yield { type: "delta", text: "A answer [1]." }
      yield completeEvent(true)
    }).mockImplementationOnce(async function* (_query: string, options: { signal: AbortSignal }) {
      if (publishBeforeCancel) {
        yield { type: "contexts", contexts: [{ id: "note-b", source_type: "notes", source_id: "note-b", excerpt: "B evidence." }] }
        yield { type: "delta", text: "B partial [1]." }
      }
      await buffered
      observedAbort = options.signal.aborted
      yield { type: "contexts", contexts: [{ id: "buffered-b", source_type: "notes", source_id: "note-b", excerpt: "Buffered B evidence." }] }
      yield { type: "delta", text: "Buffered late answer [1]." }
      yield completeEvent(true)
      throw Object.assign(new Error("Aborted after draining"), { name: "AbortError" })
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await act(async () => { await latestContext!.selectThread("local-buffered-cancel") })
    act(() => {
      latestContext!.setQuery("A question")
      latestContext!.updateSetting("sources", ["notes"])
      latestContext!.updateSetting("include_note_ids", ["note-a"])
    })
    await act(async () => { await latestContext!.search() })
    act(() => {
      latestContext!.setQuery("B question")
      latestContext!.updateSetting("include_note_ids", ["note-b"])
    })
    let pending!: Promise<void>
    act(() => { pending = latestContext!.search() })
    try {
      await waitFor(() => expect(ragSearchStreamMock).toHaveBeenCalledTimes(2))
      if (publishBeforeCancel) await waitFor(() => expect(latestContext!.answer).toBe("B partial [1]."))
      const retained = { answer: latestContext!.answer, results: latestContext!.results, citations: latestContext!.citations, resultQuery: latestContext!.resultQuery, lastSearchScope: latestContext!.lastSearchScope, answerTrustState: latestContext!.answerTrustState }
      act(() => { latestContext!.cancelSearch(); drain() })
      await act(async () => { await pending })
      expect(observedAbort).toBe(true)
      expect(latestContext).toMatchObject(retained)
      expect(latestContext!.isSearching).toBe(false)
      expect(latestContext!.queryStage).toBe("cancelled")
      expect(ragSearchMock).not.toHaveBeenCalled()
    } finally {
      act(() => { latestContext!.cancelSearch(); drain() })
      await act(async () => { await pending })
    }
  })

  it.each(["contexts", "delta"])("hands off the replacement scope after cancellation when %s publishes first", async (firstEvent) => {
    let publishFirst!: () => void
    let publishSecond!: () => void
    let finish!: () => void
    const first = new Promise<void>(resolve => { publishFirst = resolve })
    const second = new Promise<void>(resolve => { publishSecond = resolve })
    const end = new Promise<void>(resolve => { finish = resolve })
    ragSearchStreamMock.mockImplementationOnce(async function* () {
      yield { type: "contexts", contexts: [{ id: "note-a", source: "notes", source_id: "note-a", excerpt: "A opens in January." }] }
      yield { type: "delta", text: "A opens in January [1]." }
      yield completeEvent(true)
    }).mockImplementationOnce(async function* (_query: string, options: { signal: AbortSignal }) {
      const contexts = { type: "contexts", contexts: [{ id: "note-b", source: "notes", source_id: "note-b", excerpt: "B opens in February." }] }
      const delta = { type: "delta", text: "B opens in February [1]." }
      await first
      yield firstEvent === "contexts" ? contexts : delta
      await second
      yield firstEvent === "contexts" ? delta : contexts
      await end
      if (options.signal.aborted) throw Object.assign(new Error("Aborted"), { name: "AbortError" })
      yield completeEvent(true)
    })
    const i18n = createInstance().use(ICUWithInterpolation)
    await i18n.init({ lng: "en", fallbackLng: "en", defaultNS: "knowledge", resources: { en: { knowledge: knowledgeEn } } })
    render(<I18nextProvider i18n={i18n}><MemoryRouter><KnowledgeQAProvider><ContextProbe /><AnswerPanel /><ExportDialog open onClose={vi.fn()} /></KnowledgeQAProvider></MemoryRouter></I18nextProvider>)
    await act(async () => { await latestContext!.selectThread("local-scope-replacement") })
    act(() => {
      latestContext!.setQuery("When does A open?")
      latestContext!.updateSetting("sources", ["notes"])
      latestContext!.updateSetting("include_note_ids", ["note-a"])
      latestContext!.updateSetting("enable_web_fallback", false)
    })
    await act(async () => { await latestContext!.search() })
    const priorScope = latestContext!.lastSearchScope
    act(() => {
      latestContext!.setQuery("When does B open?")
      latestContext!.updateSetting("sources", ["notes", "media_db"])
      latestContext!.updateSetting("include_note_ids", ["note-b"])
      latestContext!.updateSetting("include_media_ids", [84])
      latestContext!.updateSetting("collection_id", 7)
      latestContext!.updateSetting("keyword_filter", "b-topic")
      latestContext!.updateSetting("enable_web_fallback", true)
      latestContext!.setPreset("thorough")
    })
    let pending!: Promise<void>
    act(() => { pending = latestContext!.search() })
    try {
      await waitFor(() => expect(ragSearchStreamMock).toHaveBeenCalledTimes(2))
      expect(latestContext!.answer).toBe("A opens in January [1].")
      expect(latestContext!.lastSearchScope).toEqual(priorScope)
      act(() => {
        latestContext!.updateSetting("sources", ["media_db"])
        latestContext!.updateSetting("include_note_ids", ["note-c"])
        latestContext!.updateSetting("include_media_ids", [99])
        latestContext!.updateSetting("collection_id", 9)
        latestContext!.updateSetting("keyword_filter", "c-topic")
        latestContext!.updateSetting("enable_web_fallback", false)
        publishFirst()
      })
      await waitFor(() => expect(latestContext!.resultQuery).toBe("When does B open?"))
      expect(latestContext!.lastSearchScope).toEqual({ preset: "thorough", sources: ["notes", "media_db"], includeNoteIds: ["note-b"], includeMediaIds: [84], collectionId: 7, keywordFilter: "b-topic", webFallback: true })
      act(() => { publishSecond() })
      await waitFor(() => expect(latestContext!.answer).toBe("B opens in February [1]."))
      await waitFor(() => expect(latestContext!.results.map(result => result.id)).toEqual(["note-b"]))
      act(() => { latestContext!.cancelSearch(); finish() })
      await act(async () => { await pending })
      expect(latestContext!.queryStage).toBe("cancelled")
      fireEvent.click(screen.getByRole("button", { name: "Continue in Research Workspace" }))
      await waitFor(() => expect(queuePrefillMock).toHaveBeenCalledTimes(1))
      expect(queuePrefillMock.mock.calls[0]).toEqual([
        expect.objectContaining({
          query: "When does B open?", answer: "B opens in February [1].",
          scope: { sources: ["notes", "media_db"], include_note_ids: ["note-b"], include_media_ids: [84], collection_id: 7, keyword_filter: "b-topic", enable_web_fallback: true },
          sources: [expect.objectContaining({ originalId: "note-b", excerpt: "B opens in February." })],
        }), "test-owner",
      ])
      fireEvent.click(screen.getByLabelText("I understand this unsupported draft will be labeled in the export."))
      fireEvent.click(screen.getByLabelText("Settings snapshot"))
      fireEvent.click(screen.getByRole("button", { name: "Save to Notes" }))
      expect(await screen.findByRole("link", { name: "Open saved note" })).toHaveAttribute("href", "/notes?source_ref_id=partial-scope-receipt")
      const [body, fields] = mockTldwClient.createNote.mock.calls[0]
      const snapshot = JSON.parse(body.match(/## Settings Used\n\n```json\n([\s\S]*?)\n```/)![1])
      const scope = { sources: ["notes", "media_db"], include_note_ids: ["note-b"], include_media_ids: [84], collection_id: 7, keyword_filter: "b-topic", enable_web_fallback: true }
      expect(snapshot).toMatchObject({ preset: "thorough", settings: scope })
      expect(fields.knowledge_provenance).toMatchObject({ question: "When does B open?", scope })
      expect(readKnowledgeNoteProvenance(body)).toMatchObject({ question: "When does B open?", scope })
      expect(ragSearchMock).not.toHaveBeenCalled()
    } finally {
      act(() => { latestContext!.cancelSearch(); publishFirst(); publishSecond(); finish() })
      await act(async () => { await pending })
    }
  })

  it("keeps the completed answer scope when a replacement is cancelled before publication", async () => {
    let finish!: () => void
    const end = new Promise<void>(resolve => { finish = resolve })
    ragSearchStreamMock.mockImplementationOnce(async function* () {
      yield { type: "contexts", contexts: [{ id: "note-a", source: "notes", source_id: "note-a", excerpt: "A opens in January." }] }
      yield { type: "delta", text: "A opens in January [1]." }
      yield completeEvent(true)
    }).mockImplementationOnce(async function* (_query: string, options: { signal: AbortSignal }) {
      await end
      if (options.signal.aborted) throw Object.assign(new Error("Aborted"), { name: "AbortError" })
      yield { type: "delta", text: "B opens in February." }
      yield completeEvent(true)
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await act(async () => { await latestContext!.selectThread("local-before-publication") })
    act(() => {
      latestContext!.setQuery("When does A open?")
      latestContext!.updateSetting("sources", ["notes"])
      latestContext!.updateSetting("include_note_ids", ["note-a"])
    })
    await act(async () => { await latestContext!.search() })
    const priorScope = latestContext!.lastSearchScope
    act(() => {
      latestContext!.setQuery("When does B open?")
      latestContext!.updateSetting("include_note_ids", ["note-b"])
    })
    let pending!: Promise<void>
    act(() => { pending = latestContext!.search() })
    try {
      await waitFor(() => expect(ragSearchStreamMock).toHaveBeenCalledTimes(2))
      act(() => { latestContext!.cancelSearch(); finish() })
      await act(async () => { await pending })
      expect(latestContext!.queryStage).toBe("cancelled")
      expect(latestContext!.resultQuery).toBe("When does A open?")
      expect(latestContext!.answer).toBe("A opens in January [1].")
      expect(latestContext!.lastSearchScope).toEqual(priorScope)
      expect(ragSearchMock).not.toHaveBeenCalled()
    } finally {
      act(() => { latestContext!.cancelSearch(); finish() })
      await act(async () => { await pending })
    }
  })

  it("keeps received answer and evidence when the user cancels a partial stream", async () => {
    ragSearchStreamMock.mockImplementation(async function* (_query: string, options: { signal: AbortSignal }) {
      yield { type: "contexts", contexts: [{ id: "cedar", excerpt: "Cedar opens in January." }] }
      yield { type: "delta", text: "Cedar opens in January [1]." }
      await new Promise<void>((_resolve, reject) => {
        options.signal.addEventListener("abort", () => {
          reject(Object.assign(new Error("Aborted"), { name: "AbortError" }))
        }, { once: true })
      })
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await act(async () => { await latestContext!.selectThread("local-partial-cancellation") })
    act(() => { latestContext!.setQuery("When does Cedar open?") })
    let pending!: Promise<void>
    act(() => { pending = latestContext!.search() })
    await waitFor(() => expect(latestContext!.answer).toBe("Cedar opens in January [1]."))
    const receivedTrust = latestContext!.answerTrustState
    act(() => { latestContext!.cancelSearch() })
    await act(async () => { await pending })
    expect(latestContext!.queryStage).toBe("cancelled")
    expect(latestContext!.error).toBeNull()
    expect(latestContext!.answer).toBe("Cedar opens in January [1].")
    expect(latestContext!.results.map(result => result.id)).toEqual(["cedar"])
    expect(latestContext!.answerTrustState).toBe(receivedTrust)
    expect(ragSearchMock).not.toHaveBeenCalled()
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
