import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"
import { InlineRecentSessions } from "../empty/InlineRecentSessions"
import { ExportDialog } from "../ExportDialog"
import { SourceViewerModal } from "../SourceViewerModal"
import { SourceList } from "../SourceList"
import { getKnowledgeQaHistoryStorageKey } from "../historyStorage"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

const harness = vi.hoisted(() => ({
  config: { serverUrl: "http://alice-server", authMode: "multi-user" as const, accessToken: "alice" },
  loading: false,
  connected: true,
  snapshot: {} as ServicePromptSnapshot,
  fetch: vi.fn(),
  search: vi.fn(),
  stream: vi.fn(),
  create: vi.fn(),
  add: vi.fn(),
  remove: vi.fn(),
  message: vi.fn(),
  share: vi.fn(),
  createShare: vi.fn(), exportChatbook: vi.fn(), download: vi.fn(), clipboard: vi.fn(),
  settings: undefined as unknown,
  feedback: vi.fn(),
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({ config: harness.config, loading: harness.loading, authorityLoading: harness.loading }),
}))
vi.mock("@/store/connection", () => ({
  useConnectionStore: Object.assign((select: (store: { state: { isConnected: boolean; mode: string } }) => unknown) => select({ state: { isConnected: harness.connected, mode: "normal" } }), {
    getState: () => ({ state: { isConnected: harness.connected, mode: "normal" } }),
  }),
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: vi.fn(async () => harness.snapshot),
}))
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: (key: string) => [key === "ragSearchSettingsV2" ? harness.settings : undefined] }))
vi.mock("@/hooks/useAntdMessage", () => ({ useAntdMessage: () => ({ open: harness.message }) }))
vi.mock("@/utils/knowledge-qa-search-metrics", () => ({ trackKnowledgeQaSearchMetric: vi.fn() }))
vi.mock("@/services/feedback", () => ({
  getFeedbackSessionId: () => "qa-test-session",
  submitExplicitFeedback: (...args: unknown[]) => harness.feedback(...args),
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: vi.fn().mockResolvedValue(undefined),
  fetchWithAuth: (...args: unknown[]) => harness.fetch(...args),
  searchCharacters: vi.fn().mockResolvedValue([]),
  listCharacters: vi.fn().mockResolvedValue([]),
  ragSourceHealth: vi.fn().mockResolvedValue({}),
  createChat: (...args: unknown[]) => harness.create(...args),
  getChat: vi.fn().mockResolvedValue({ version: 1 }),
  addChatMessage: (...args: unknown[]) => harness.add(...args),
  ragSearch: (...args: unknown[]) => harness.search(...args),
  ragSearchStream: (...args: unknown[]) => harness.stream(...args),
  deleteChat: (...args: unknown[]) => harness.remove(...args),
  resolveConversationShareLink: (...args: unknown[]) => harness.share(...args),
  createConversationShareLink: (...args: unknown[]) => harness.createShare(...args),
  exportChatbook: (...args: unknown[]) => harness.exportChatbook(...args),
  downloadChatbookExport: (...args: unknown[]) => harness.download(...args),
} }))

let current: ReturnType<typeof useKnowledgeQA>
function Recent({ sourceFeedback = false }: { sourceFeedback?: boolean }) {
  const context = useKnowledgeQA()
  React.useLayoutEffect(() => { current = context }, [context])
  const [exportOpen, setExportOpen] = React.useState(false)
  const [previewOpen, setPreviewOpen] = React.useState(false)
  return <><InlineRecentSessions items={context.searchHistory} onRestore={context.restoreFromHistory} />
    <output aria-label="Answer">{context.answer}</output>
    <output aria-label="Query">{context.query}</output>
    <button onClick={() => setExportOpen(true)}>Open export</button>
    <button onClick={() => setPreviewOpen(true)}>Preview source</button>
    <SourceViewerModal open={previewOpen} result={context.results[0] ?? null} index={1} onClose={() => setPreviewOpen(false)} />
    {sourceFeedback && <SourceList />}
    {exportOpen && <ExportDialog open onClose={() => setExportOpen(false)} />}
  </>
}
const view = (sourceFeedback = false) => <KnowledgeQAProvider><Recent sourceFeedback={sourceFeedback} /></KnowledgeQAProvider>
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail })
  return { promise, resolve, reject }
}
const response = (data: unknown) => ({ ok: true, status: 200, json: async () => data })
const aliceHistory = {
  id: "alice-history", query: "Alice Cedar question", timestamp: "2026-09-15T11:00:00Z",
  conversationId: "alice-thread", keywords: ["__knowledge_QA__"], sourcesCount: 1,
  hasAnswer: true, citationCount: 1, trustState: "cited_answer",
}
const serverHistory = [{ id: "alice-thread", title: "Alice Cedar question", message_count: 2,
  keywords: ["__knowledge_QA__"], last_modified: "2026-09-15T11:00:00Z" }]
function account(name: string) {
  harness.config = { ...harness.config, accessToken: name }
  const controller = new AbortController()
  harness.snapshot = {
    capability: "unchecked", definitions: {},
    scopeKey: `verified-${name}`, requestScope: { config: harness.config, userId: name },
    scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal,
    release: vi.fn(),
  }
  return controller
}

describe("Knowledge QA verified account boundary", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    harness.loading = false
    harness.connected = true
    harness.settings = undefined
    account("alice")
    harness.fetch.mockResolvedValue(response([]))
    harness.create.mockResolvedValue({ id: "alice-thread", version: 1 })
    harness.add.mockResolvedValue({ id: "saved-message" })
    harness.remove.mockResolvedValue(undefined)
    harness.search.mockResolvedValue({ results: [], generated_answer: null, metadata: {} })
    harness.stream.mockImplementation(async function* () { yield { type: "done", status: "success" } })
  })

  it("does not expose unowned legacy questions through the real Recent buttons", async () => {
    localStorage.setItem("knowledge_qa_history", JSON.stringify([aliceHistory]))
    account("bob")
    render(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    expect(screen.queryByRole("button", { name: /Alice Cedar question/ })).not.toBeInTheDocument()
    expect(current.searchHistory).toEqual([])
  })

  it("masks prior Recent and active state immediately when verified authority is unresolved", async () => {
    harness.fetch.mockResolvedValue(response(serverHistory))
    const rendered = render(view())
    await screen.findByRole("button", { name: /Alice Cedar question/ })
    act(() => current.setQuery("Alice private draft"))
    harness.loading = true
    rendered.rerender(view())
    expect(screen.queryByRole("button", { name: /Alice Cedar question/ })).not.toBeInTheDocument()
    expect(screen.getByLabelText("Query")).toHaveTextContent("")
  })

  it("ignores delayed Alice history even after Alice to Bob to Alice", async () => {
    const pending = deferred<unknown>()
    harness.fetch.mockReturnValueOnce(pending.promise)
    const original = harness.snapshot
    const rendered = render(view())
    await waitFor(() => expect(harness.fetch).toHaveBeenCalled())
    account("bob")
    rendered.rerender(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    account("alice")
    rendered.rerender(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    await act(async () => { pending.resolve(response(serverHistory)); await pending.promise })
    expect(current.searchHistory).toEqual([])
    expect(original.release).toHaveBeenCalled()
  })

  it("rejects a delayed Recent restore after account replacement", async () => {
    const pending = deferred<unknown>()
    harness.fetch.mockImplementation((path: string) => path.includes("messages-with-context")
      ? pending.promise : Promise.resolve(response(serverHistory)))
    const rendered = render(view())
    fireEvent.click(await screen.findByRole("button", { name: /Alice Cedar question/ }))
    await waitFor(() => expect(harness.fetch.mock.calls.some(([path]) => path.includes("messages-with-context"))).toBe(true))
    account("bob")
    harness.fetch.mockResolvedValue(response([]))
    rendered.rerender(view())
    await act(async () => { pending.resolve(response([{ id: "answer", role: "assistant", content: "Alice private answer" }])); await pending.promise })
    expect(screen.getByLabelText("Answer")).not.toHaveTextContent("Alice private answer")
    expect(current.messages).toEqual([])
    expect(current.currentThreadId).toBeNull()
  })

  it("does not continue a pending thread creation with the next account's credentials", async () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {})
    const pending = deferred<unknown>()
    harness.create.mockReturnValue(pending.promise)
    const rendered = render(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    act(() => current.setQuery("Alice private pending question"))
    let search!: Promise<void>
    act(() => { search = current.search() })
    await waitFor(() => expect(harness.create).toHaveBeenCalled())
    account("bob")
    rendered.rerender(view())
    await act(async () => { pending.resolve({ id: "alice-created", version: 1 }); await search })
    expect(harness.add).not.toHaveBeenCalled()
    expect(harness.stream).not.toHaveBeenCalled()
    expect(current.messages).toEqual([])
    expect(errors).not.toHaveBeenCalled()
  })

  it("discards a delayed stream and its persistence after Alice to Bob to Alice", async () => {
    const pending = deferred<void>()
    harness.stream.mockImplementation(async function* () {
      yield { type: "contexts", contexts: [{ id: "alice-source", title: "Alice source", source: "media_db", excerpt: "Alice private evidence" }] }
      yield { type: "delta", text: "Alice partial answer [1]" }
      await pending.promise
      yield { type: "delta", text: " and private completion" }
      yield { schema_version: 1, type: "complete", code: "complete", upstream_dispatched: true, output_emitted: true, allow_non_stream_fallback: false, message: "Search completed." }
    })
    const rendered = render(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    act(() => current.setQuery("Alice streamed question"))
    let search!: Promise<void>
    act(() => { search = current.search() })
    await waitFor(() => expect(current.answer).toContain("Alice partial answer"))
    expect(current.results).toHaveLength(1)
    fireEvent.click(screen.getByRole("button", { name: "Preview source" }))
    expect(screen.getByRole("dialog")).toHaveTextContent("Alice private evidence")
    account("bob")
    rendered.rerender(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    account("alice")
    rendered.rerender(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    await act(async () => { pending.resolve(); await search })
    expect(current.answer).toBeNull()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(screen.queryByText("Alice private evidence")).not.toBeInTheDocument()
    expect(current.results).toEqual([])
    expect(current.citations).toEqual([])
    expect(current.searchHistory).toEqual([])
    expect(harness.add.mock.calls.map(([, payload]) => payload.role)).toEqual(["user"])
    expect(harness.fetch.mock.calls.some(([path]) => path.includes("/rag-context"))).toBe(false)
    expect(localStorage.getItem(getKnowledgeQaHistoryStorageKey(harness.snapshot.requestScope))).toBeNull()
  })

  it("does not populate the next owner with a delayed public share read", async () => {
    const pending = deferred<unknown>()
    harness.share.mockReturnValue(pending.promise)
    const rendered = render(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    let restore!: ReturnType<typeof current.selectSharedThread>
    act(() => { restore = current.selectSharedThread("public-token") })
    await waitFor(() => expect(harness.share).toHaveBeenCalled())
    account("bob")
    rendered.rerender(view())
    await act(async () => {
      pending.resolve({ conversation_id: "public-thread", messages: [{ id: "public-answer", role: "assistant", content: "Previously selected answer" }] })
      await restore
    })
    expect(current.messages).toEqual([])
    expect(current.answer).toBeNull()
  })

  it("binds actual SourceList feedback to its owner and ignores a late failure after replacement", async () => {
    const pending = deferred<unknown>()
    harness.feedback.mockReturnValue(pending.promise)
    harness.stream.mockImplementation(async function* () {
      yield { type: "contexts", contexts: [{ id: "alice-source", title: "Alice source", source: "media_db", excerpt: "Alice private evidence" }] }
      yield { type: "delta", text: "Alice answer [1]" }
      yield { schema_version: 1, type: "complete", code: "complete", upstream_dispatched: true, output_emitted: true, allow_non_stream_fallback: false, message: "Search completed." }
    })
    const rendered = render(view(true))
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    act(() => current.setQuery("Alice feedback question"))
    await act(async () => { await current.search() })
    fireEvent.click(screen.getByRole("button", { name: "Yes" }))
    await waitFor(() => expect(harness.feedback).toHaveBeenCalled())
    expect(harness.feedback).toHaveBeenCalledWith(
      expect.objectContaining({ query: "Alice feedback question", document_ids: ["alice-source"] }),
      expect.objectContaining({ requestScope: harness.snapshot.requestScope, signal: expect.any(AbortSignal) }),
    )
    account("bob")
    rendered.rerender(view(true))
    harness.message.mockClear()
    await act(async () => { pending.reject(new Error("Late failed feedback")); await pending.promise.catch(() => undefined) })
    expect(harness.message).not.toHaveBeenCalled()
    expect(screen.queryByRole("button", { name: "Retry feedback" })).not.toBeInTheDocument()
  })

  it("does not replay an old delete action after Alice to Bob to Alice", async () => {
    harness.fetch.mockResolvedValue(response(serverHistory))
    const rendered = render(view())
    await screen.findByRole("button", { name: /Alice Cedar question/ })
    const oldDelete = current.deleteHistoryItem
    account("bob")
    harness.fetch.mockResolvedValue(response([]))
    rendered.rerender(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    account("alice")
    rendered.rerender(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    await act(async () => { await oldDelete("alice-thread") })
    expect(harness.remove).not.toHaveBeenCalled()
  })

  it("preserves the owner's local Recent history and a typed draft through benign re-render", async () => {
    localStorage.setItem(getKnowledgeQaHistoryStorageKey(harness.snapshot.requestScope), JSON.stringify([aliceHistory]))
    const rendered = render(view())
    await screen.findByRole("button", { name: /Alice Cedar question/ })
    act(() => current.setQuery("Own unsent draft"))
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } })))
    rendered.rerender(view())
    expect(current.query).toBe("Own unsent draft")
    rendered.unmount()
    render(view())
    await screen.findByRole("button", { name: /Alice Cedar question/ })
  })

  it("does not share persisted history when different verified targets have a colliding display scope key", async () => {
    localStorage.setItem(getKnowledgeQaHistoryStorageKey(harness.snapshot.requestScope), JSON.stringify([aliceHistory]))
    account("bob")
    harness.snapshot = { ...harness.snapshot, scopeKey: "verified-alice" }
    render(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    expect(current.searchHistory).toEqual([])
  })

  it("keeps restored owner history bounded to the existing 100-entry limit", async () => {
    localStorage.setItem(getKnowledgeQaHistoryStorageKey(harness.snapshot.requestScope), JSON.stringify(Array.from({ length: 120 }, (_, i) => ({ ...aliceHistory, id: `entry-${i}` }))))
    render(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    expect(current.searchHistory).toHaveLength(100)
  })

  it("does not adopt source IDs from unowned global settings into the next account", async () => {
    harness.settings = { include_media_ids: [1], include_note_ids: ["alice-private-note"], top_k: 8 }
    account("bob")
    render(view())
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    expect(current.settings.include_media_ids).toEqual([])
    expect(current.settings.include_note_ids).toEqual([])
    expect(current.settings.top_k).toBe(8)
  })

  it("does not read or write source filters before authority is verified", async () => {
    const reads = vi.spyOn(window.localStorage, "getItem")
    const writes = vi.spyOn(window.localStorage, "setItem")
    harness.loading = true
    render(view(true))
    expect(reads.mock.calls.filter(([key]) => key.startsWith("knowledge_qa_source_filters:"))).toEqual([])
    expect(writes.mock.calls.filter(([key]) => key.startsWith("knowledge_qa_source_filters:"))).toEqual([])
  })

  it("rejects legacy source filters and restores the verified owner's filters on reload", async () => {
    const legacyFilters = JSON.stringify({ sortMode: "title", sourceType: "all", contentFacet: "all", dateFilter: "all", keyword: "Alice private filter" })
    localStorage.setItem("knowledge_qa_source_filters:global", legacyFilters)
    localStorage.setItem("knowledge_qa_source_filters:alice-thread", legacyFilters)
    account("bob")
    harness.stream.mockImplementation(async function* () {
      yield { type: "contexts", contexts: Array.from({ length: 4 }, (_, i) => ({ id: `own-${i}`, title: `Own source ${i}`, source: "media_db", excerpt: `Own evidence ${i}` })) }
      yield { type: "delta", text: "Own answer [1]" }
      yield { schema_version: 1, type: "complete", code: "complete", upstream_dispatched: true, output_emitted: true, allow_non_stream_fallback: false, message: "Search completed." }
    })
    const rendered = render(view(true))
    await waitFor(() => expect(current.historyHydrated).toBe(true))
    act(() => current.setQuery("Own source-filter question"))
    await act(async () => { await current.search() })
    expect(screen.getByLabelText("Filter sources by keyword")).toHaveValue("")
    fireEvent.change(screen.getByLabelText("Filter sources by keyword"), { target: { value: "Own evidence" } })
    const messages = current.messages
    rendered.unmount()
    harness.fetch.mockImplementation(async (path: string) => response(path.includes("messages-with-context") ? messages : []))
    render(view(true))
    fireEvent.click(await screen.findByRole("button", { name: /Own source-filter question/ }))
    await waitFor(() => expect(screen.getByLabelText("Filter sources by keyword")).toHaveValue("Own evidence"))
  })

  it("clears the prior answer when the actual Recent target is missing without a console error overlay", async () => {
    const errors = vi.spyOn(console, "error").mockImplementation(() => {})
    const missing = { ...serverHistory[0], id: "missing", title: "Deleted owned question" }
    harness.fetch.mockImplementation(async (path: string) => path.includes("/missing/messages-with-context")
      ? { ok: false, status: 404, json: async () => ({ detail: "Not found" }) }
      : response(path.includes("messages-with-context") ? [
        { id: "u", role: "user", content: "Alice Cedar question" },
        { id: "a", role: "assistant", content: "Prior answer" },
      ] : [...serverHistory, missing]))
    render(view())
    fireEvent.click(await screen.findByRole("button", { name: /Alice Cedar question/ }))
    await waitFor(() => expect(current.answer).toBe("Prior answer"))
    fireEvent.click(screen.getByRole("button", { name: /Deleted owned question/ }))
    await waitFor(() => expect(current.error).toMatch(/Unable to load/))
    expect(current.answer).toBeNull()
    expect(current.results).toEqual([])
    expect(errors).not.toHaveBeenCalled()
  })

  it("keeps public shared conversation reading available without authenticated history", async () => {
    harness.connected = false
    harness.share.mockResolvedValue({ conversation_id: "public-thread", messages: [
      { id: "public-user", role: "user", content: "Public question" },
      { id: "public-answer", role: "assistant", content: "Public answer" },
    ] })
    render(view())
    await act(async () => { await current.selectSharedThread("public-token") })
    expect(current.messages).toHaveLength(2)
    expect(harness.fetch).not.toHaveBeenCalled()
    expect(current.searchHistory).toEqual([])
  })

  it.each(["share", "export"])("does not copy or download a delayed %s after account replacement", async (kind) => {
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: harness.clipboard } })
    const pending = deferred<unknown>()
    harness.createShare.mockReturnValue(pending.promise)
    harness.exportChatbook.mockReturnValue(pending.promise)
    harness.fetch.mockImplementation((path: string) => Promise.resolve(response(path.includes("messages-with-context") ? [
      { id: "question", role: "user", content: "Alice Cedar question" },
      { id: "answer", role: "assistant", content: "Alice answer" },
    ] : serverHistory)))
    const rendered = render(view())
    fireEvent.click(await screen.findByRole("button", { name: /Alice Cedar question/ }))
    await waitFor(() => expect(current.currentThreadId).toBe("alice-thread"))
    fireEvent.click(screen.getByRole("button", { name: "Open export" }))
    if (kind === "share") fireEvent.click(screen.getByRole("button", { name: "Create share link" }))
    else {
      fireEvent.click(screen.getByRole("button", { name: /Chatbook/i }))
      fireEvent.click(screen.getByRole("button", { name: "Export" }))
    }
    await waitFor(() => expect(kind === "share" ? harness.createShare : harness.exportChatbook).toHaveBeenCalled())
    account("bob")
    harness.fetch.mockResolvedValue(response([]))
    rendered.rerender(view())
    await act(async () => { pending.resolve({ share_id: "private-share", share_path: "/knowledge/shared/private", expires_at: "tomorrow", success: true, job_id: "private-export" }); await pending.promise })
    expect(harness.clipboard).not.toHaveBeenCalled()
    expect(harness.download).not.toHaveBeenCalled()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })
})
