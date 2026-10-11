import "./knowledgeQaAuthorityFixture"
import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"

const ragSearchMock = vi.fn()
const messageOpenMock = vi.fn()
const trackMetricMock = vi.fn()
const createChatMock = vi.fn()
const deleteChatMock = vi.fn()
const addChatMessageMock = vi.fn()
const searchCharactersMock = vi.fn()
const listCharactersMock = vi.fn()
const fetchWithAuthMock = vi.fn()

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
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    fetchWithAuth: (...args: unknown[]) => fetchWithAuthMock(...args),
    ragSearch: (...args: unknown[]) => ragSearchMock(...args),
    createChat: (...args: unknown[]) => createChatMock(...args),
    deleteChat: (...args: unknown[]) => deleteChatMock(...args),
    addChatMessage: (...args: unknown[]) => addChatMessageMock(...args),
    searchCharacters: (...args: unknown[]) => searchCharactersMock(...args),
    listCharacters: (...args: unknown[]) => listCharactersMock(...args),
    getChat: vi.fn().mockResolvedValue({ version: 1 }),
  },
}))

let latestContext: ReturnType<typeof useKnowledgeQA> | null = null

function ContextProbe() {
  const context = useKnowledgeQA()
  React.useEffect(() => { latestContext = context }, [context])
  return null
}

describe("KnowledgeQAProvider search cancellation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    latestContext = null
    fetchWithAuthMock.mockResolvedValue({ ok: false, json: async () => [], text: async () => "" })
    trackMetricMock.mockResolvedValue(undefined)
    createChatMock.mockResolvedValue({ id: "thread-default", version: 1 })
    deleteChatMock.mockResolvedValue(undefined)
    addChatMessageMock.mockResolvedValue({ id: "msg-default" })
    searchCharactersMock.mockResolvedValue([
      { id: 7, name: "Helpful AI Assistant" },
    ])
    listCharactersMock.mockResolvedValue([
      { id: 7, name: "Helpful AI Assistant" },
    ])
    ragSearchMock.mockImplementation((_query: string, options: { signal?: AbortSignal }) => {
      return new Promise((_resolve, reject) => {
        options.signal?.addEventListener("abort", () => {
          const abortError = new Error("Aborted")
          ;(abortError as Error & { name: string }).name = "AbortError"
          reject(abortError)
        })
      })
    })
  })

  it("does not activate cancelled A after its delayed tagging settles behind completed B", async () => {
    let finishTagA!: () => void
    const tagA = new Promise<void>(resolve => { finishTagA = resolve })
    fetchWithAuthMock.mockImplementation(async (path: string, options?: { method?: string }) => {
      if (path.endsWith("/thread-a") && options?.method === "PATCH") await tagA
      return { ok: true, json: async () => ({ keywords: [] }), text: async () => "" }
    })
    createChatMock.mockResolvedValueOnce({ id: "thread-a", version: 1 })
      .mockResolvedValueOnce({ id: "thread-b", version: 1 })
    ragSearchMock.mockResolvedValue({
      results: [{ id: "note-b", content: "B evidence", metadata: { source_type: "notes", note_id: "note-b" } }],
      answer: "B answer [1].",
    })
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    act(() => {
      latestContext!.setQuery("A question")
      latestContext!.updateSetting("sources", ["notes"])
      latestContext!.updateSetting("include_note_ids", ["note-b"])
      latestContext!.updateSetting("enable_web_fallback", false)
    })
    let requestA!: Promise<void>
    let requestB: Promise<void> | undefined
    act(() => { requestA = latestContext!.search() })
    try {
      await waitFor(() => expect(fetchWithAuthMock).toHaveBeenCalledWith(
        "/api/v1/chat/conversations/thread-a", expect.objectContaining({ method: "PATCH" })
      ))
      act(() => { latestContext!.cancelSearch(); latestContext!.setQuery("B question") })
      await act(async () => { requestB = latestContext!.search(); await requestB })
      expect(latestContext!.currentThreadId).toBe("thread-b")
      expect(latestContext!.messages.map(message => message.content)).toEqual(["B question", "B answer [1]."])
      expect(latestContext!.queryStage).toBe("complete")
      const retained = {
        currentThreadId: latestContext!.currentThreadId,
        messages: latestContext!.messages,
        results: latestContext!.results,
        answer: latestContext!.answer,
        resultQuery: latestContext!.resultQuery,
        lastSearchScope: latestContext!.lastSearchScope,
        citations: latestContext!.citations,
      }
      await act(async () => { finishTagA(); await requestA })
      expect(latestContext).toMatchObject(retained)
      expect(deleteChatMock).toHaveBeenCalledWith("thread-a", expect.objectContaining({
        requestScope: expect.objectContaining({ userId: "test-owner" }),
        signal: expect.any(AbortSignal),
      }))
      expect(ragSearchMock).toHaveBeenCalledTimes(1)
      expect(latestContext!.queryStage).toBe("complete")
    } finally {
      finishTagA()
      await act(async () => { await requestA; await requestB })
    }
  })

  it("settles cancellation synchronously without an old completion clearing a newer request", async () => {
    let finishOld!: (value: Record<string, unknown>) => void
    let finishNew!: (value: Record<string, unknown>) => void
    ragSearchMock.mockImplementation((query: string) => new Promise(resolve => {
      if (query === "Old question") finishOld = resolve
      else finishNew = resolve
    }))
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await act(async () => { await latestContext!.selectThread("local-cancel-owner") })
    act(() => { latestContext!.setQuery("Old question") })
    let oldRequest!: Promise<void>
    let newRequest: Promise<void> | undefined
    act(() => { oldRequest = latestContext!.search() })
    try {
      await waitFor(() => expect(ragSearchMock).toHaveBeenCalledTimes(1))
      act(() => { latestContext!.cancelSearch() })
      expect(latestContext!.isSearching).toBe(false)
      expect(latestContext!.queryStage).toBe("cancelled")
      act(() => { latestContext!.setQuery("New question") })
      act(() => { newRequest = latestContext!.search() })
      await waitFor(() => expect(ragSearchMock).toHaveBeenCalledTimes(2))
      await act(async () => { finishOld({ results: [], answer: "Late old answer" }); await oldRequest })
      expect(latestContext!.isSearching).toBe(true)
      expect(latestContext!.queryStage).toBe("ranking")
      expect(latestContext!.answer).toBeNull()
      act(() => { latestContext!.cancelSearch() })
      expect((ragSearchMock.mock.calls[1][1] as { signal: AbortSignal }).signal.aborted).toBe(true)
      await act(async () => { finishNew({ results: [], answer: "Late new answer" }); await newRequest })
      expect(latestContext!.isSearching).toBe(false)
      expect(latestContext!.queryStage).toBe("cancelled")
      expect(latestContext!.answer).toBeNull()
    } finally {
      act(() => { latestContext!.cancelSearch(); finishOld({ results: [] }); finishNew?.({ results: [] }) })
      await act(async () => { await oldRequest; await newRequest })
    }
  })

  it("passes AbortSignal to ragSearch and supports cancelSearch", async () => {
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectThread("local-test-thread")
    })

    act(() => {
      latestContext!.setQuery("cancel this query")
    })

    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(ragSearchMock).toHaveBeenCalledTimes(1))
    expect(trackMetricMock).toHaveBeenCalledWith({
      type: "search_submit",
      query_length: "cancel this query".length,
    })

    const ragSearchOptions = ragSearchMock.mock.calls[0][1] as {
      signal?: AbortSignal
    }
    expect(ragSearchOptions.signal).toBeInstanceOf(AbortSignal)

    act(() => {
      latestContext!.cancelSearch()
    })

    await waitFor(() => {
      expect(latestContext!.isSearching).toBe(false)
      expect(latestContext!.error).toBeNull()
      expect(latestContext!.queryStage).toBe("cancelled")
      expect(latestContext!.hasSearched).toBe(true)
    })
    expect(latestContext!.answerTrustState).not.toBe("failed_search")
    expect(latestContext!.extensionFailureState).not.toBe("search_failed")

    expect(messageOpenMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "info",
        content: "Search cancelled.",
      })
    )
    expect(trackMetricMock).toHaveBeenCalledWith({ type: "search_cancel" })

    ragSearchMock.mockResolvedValueOnce({
      results: [{ id: "recovered", content: "Recovered source" }],
      answer: "Recovered answer",
    })
    await act(async () => { await latestContext!.search() })
    expect(latestContext!.queryStage).toBe("complete")
    expect(latestContext!.error).toBeNull()
    expect(latestContext!.answer).toBe("Recovered answer")
  })

  it("keeps an unexpected transport abort on the failure path", async () => {
    ragSearchMock.mockRejectedValueOnce(Object.assign(new Error("Transport aborted"), { name: "AbortError" }))
    render(<KnowledgeQAProvider><ContextProbe /></KnowledgeQAProvider>)
    await act(async () => { await latestContext!.selectThread("local-abort-failure") })
    act(() => { latestContext!.setQuery("A query that the transport aborts") })
    await act(async () => { await latestContext!.search() })
    expect(latestContext!.queryStage).toBe("error")
    expect(latestContext!.answerTrustState).toBe("failed_search")
    expect(latestContext!.error).not.toBe("Search cancelled")

    act(() => { void latestContext!.search() })
    await waitFor(() => expect(ragSearchMock).toHaveBeenCalledTimes(2))
    act(() => { latestContext!.cancelSearch() })
    await waitFor(() => expect(latestContext!.queryStage).toBe("cancelled"))
    expect(latestContext!.error).toBeNull()
    expect(latestContext!.answerTrustState).not.toBe("failed_search")
    expect(latestContext!.extensionFailureState).not.toBe("search_failed")
  })

  it("tracks clear-full actions and keeps clear aborts status-neutral", async () => {
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectThread("local-test-thread")
    })

    act(() => {
      latestContext!.setQuery("query to clear")
    })

    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(ragSearchMock).toHaveBeenCalledTimes(1))

    act(() => {
      latestContext!.clearResults()
    })

    await waitFor(() => {
      expect(latestContext!.isSearching).toBe(false)
      expect(latestContext!.error).toBeNull()
    })

    expect(trackMetricMock).toHaveBeenCalledWith({ type: "search_clear_full" })
    expect(trackMetricMock).not.toHaveBeenCalledWith({ type: "search_cancel" })
  })

  it("ignores stale search completions when an older request resolves after a newer search", async () => {
    let resolveFirstSearch: ((value: Record<string, unknown>) => void) | null = null
    let resolveSecondSearch: ((value: Record<string, unknown>) => void) | null = null
    ragSearchMock
      .mockImplementationOnce(
        () =>
          new Promise<Record<string, unknown>>((resolve) => {
            resolveFirstSearch = resolve
          })
      )
      .mockImplementationOnce(
        () =>
          new Promise<Record<string, unknown>>((resolve) => {
            resolveSecondSearch = resolve
          })
      )

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectThread("local-concurrency-thread")
    })

    act(() => {
      latestContext!.setQuery("first query")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(ragSearchMock).toHaveBeenCalledTimes(1))

    act(() => {
      latestContext!.setQuery("second query")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(ragSearchMock).toHaveBeenCalledTimes(2))

    resolveSecondSearch?.({
      results: [{ id: "doc-second", content: "Second source" }],
      answer: "Second answer",
    })

    await waitFor(() => {
      expect(latestContext!.answer).toBe("Second answer")
      expect(latestContext!.results.map((result) => result.id)).toEqual(["doc-second"])
    })

    resolveFirstSearch?.({
      results: [{ id: "doc-first", content: "First source" }],
      answer: "First answer",
    })

    await act(async () => {
      await Promise.resolve()
    })

    expect(latestContext!.answer).toBe("Second answer")
    expect(latestContext!.results.map((result) => result.id)).toEqual(["doc-second"])
  })

  it("ignores late search completions after clearResults resets the session", async () => {
    let resolveSearch: ((value: Record<string, unknown>) => void) | null = null
    ragSearchMock.mockImplementationOnce(
      () =>
        new Promise<Record<string, unknown>>((resolve) => {
          resolveSearch = resolve
        })
    )

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectThread("local-clear-race-thread")
    })

    act(() => {
      latestContext!.setQuery("query to clear before completion")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(ragSearchMock).toHaveBeenCalledTimes(1))

    act(() => {
      latestContext!.clearResults()
    })

    await waitFor(() => {
      expect(latestContext!.answer).toBeNull()
      expect(latestContext!.results).toEqual([])
      expect(latestContext!.currentThreadId).toBeNull()
    })

    resolveSearch?.({
      results: [{ id: "doc-late", content: "Late source" }],
      answer: "Late answer",
    })

    await act(async () => {
      await Promise.resolve()
    })

    expect(latestContext!.answer).toBeNull()
    expect(latestContext!.results).toEqual([])
    expect(latestContext!.currentThreadId).toBeNull()
  })

  it("keeps the newer thread selected when an older empty-state thread creation resolves late", async () => {
    let resolveFirstCreateChat: ((value: Record<string, unknown>) => void) | null = null
    createChatMock
      .mockImplementationOnce(
        () =>
          new Promise<Record<string, unknown>>((resolve) => {
            resolveFirstCreateChat = resolve
          })
      )
      .mockResolvedValueOnce({ id: "thread-second", version: 1 })
    addChatMessageMock
      .mockResolvedValueOnce({ id: "msg-second-user" })
      .mockResolvedValueOnce({ id: "msg-second-assistant" })
    ragSearchMock
      .mockResolvedValueOnce({
        results: [{ id: "doc-second" }],
        answer: "Second answer",
      })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    act(() => {
      latestContext!.setQuery("first empty-state query")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(createChatMock).toHaveBeenCalledTimes(1))

    act(() => {
      latestContext!.setQuery("second empty-state query")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(createChatMock).toHaveBeenCalledTimes(2))
    await waitFor(() => {
      expect(latestContext!.currentThreadId).toBe("thread-second")
      expect(latestContext!.answer).toBe("Second answer")
    })

    resolveFirstCreateChat?.({ id: "thread-first", version: 1 })

    await act(async () => {
      await Promise.resolve()
    })

    expect(latestContext!.currentThreadId).toBe("thread-second")
    expect(latestContext!.answer).toBe("Second answer")
  })

  it("deletes stale remote threads created by superseded empty-state searches", async () => {
    let resolveFirstCreateChat: ((value: Record<string, unknown>) => void) | null = null
    createChatMock
      .mockImplementationOnce(
        () =>
          new Promise<Record<string, unknown>>((resolve) => {
            resolveFirstCreateChat = resolve
          })
      )
      .mockResolvedValueOnce({ id: "thread-second", version: 1 })
    addChatMessageMock
      .mockResolvedValueOnce({ id: "msg-second-user" })
      .mockResolvedValueOnce({ id: "msg-second-assistant" })
    ragSearchMock.mockResolvedValueOnce({
      results: [{ id: "doc-second" }],
      answer: "Second answer",
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    act(() => {
      latestContext!.setQuery("first empty-state query")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(createChatMock).toHaveBeenCalledTimes(1))

    act(() => {
      latestContext!.setQuery("second empty-state query")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(createChatMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(latestContext!.currentThreadId).toBe("thread-second"))

    resolveFirstCreateChat?.({ id: "thread-first", version: 1 })

    await waitFor(() => {
      expect(deleteChatMock).toHaveBeenCalledWith("thread-first", expect.objectContaining({ requestScope: expect.objectContaining({ userId: "test-owner" }) }))
    })
  })
})
