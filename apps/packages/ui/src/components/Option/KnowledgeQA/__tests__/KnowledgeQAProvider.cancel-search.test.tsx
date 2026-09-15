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
    fetchWithAuth: vi.fn().mockResolvedValue({
      ok: false,
      json: async () => [],
      text: async () => "",
    }),
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
  latestContext = useKnowledgeQA()
  return null
}

describe("KnowledgeQAProvider search cancellation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    latestContext = null
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
      expect(latestContext!.error).toBe("Search cancelled")
    })

    expect(messageOpenMock).toHaveBeenCalledWith(
      expect.objectContaining({
        type: "info",
        content: "Search cancelled.",
      })
    )
    expect(trackMetricMock).toHaveBeenCalledWith({ type: "search_cancel" })
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
