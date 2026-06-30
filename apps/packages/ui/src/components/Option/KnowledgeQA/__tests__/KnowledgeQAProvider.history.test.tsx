import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"
import type { SearchHistoryItem } from "../types"

const fetchWithAuthMock = vi.fn()
const ragSearchMock = vi.fn()
const addChatMessageMock = vi.fn()
const searchCharactersMock = vi.fn()
const listCharactersMock = vi.fn()
const createChatMock = vi.fn()
const getChatMock = vi.fn()
const messageOpenMock = vi.fn()
const trackMetricMock = vi.fn()

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
    addChatMessage: (...args: unknown[]) => addChatMessageMock(...args),
    searchCharacters: (...args: unknown[]) => searchCharactersMock(...args),
    listCharacters: (...args: unknown[]) => listCharactersMock(...args),
    createChat: (...args: unknown[]) => createChatMock(...args),
    getChat: (...args: unknown[]) => getChatMock(...args),
  },
}))

let latestContext: ReturnType<typeof useKnowledgeQA> | null = null

function ContextProbe() {
  latestContext = useKnowledgeQA()
  return null
}

const baseHistoryItem: SearchHistoryItem = {
  id: "history-1",
  query: "Hydrate this thread",
  timestamp: "2026-02-18T10:00:00.000Z",
  sourcesCount: 1,
  hasAnswer: true,
  preset: "fast",
  keywords: ["__knowledge_QA__"],
  conversationId: "remote-thread-1",
}

describe("KnowledgeQAProvider history hydration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    latestContext = null
    trackMetricMock.mockResolvedValue(undefined)
    addChatMessageMock.mockResolvedValue(null)
    searchCharactersMock.mockResolvedValue([])
    listCharactersMock.mockResolvedValue([])
    createChatMock.mockResolvedValue({ id: "thread-1", version: 1 })
    getChatMock.mockResolvedValue({ version: 1 })
    ragSearchMock.mockResolvedValue({
      results: [],
      generated_answer: null,
      metadata: {},
    })
    localStorage.clear()

    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/messages-with-context")) {
        return {
          ok: true,
          status: 200,
          json: async () => [],
          text: async () => "",
        }
      }
      return {
        ok: false,
        status: 404,
        json: async () => [],
        text: async () => "",
      }
    })
  })

  it("hydrates persisted local history on mount without wiping storage first", async () => {
    localStorage.setItem("knowledge_qa_history", JSON.stringify([baseHistoryItem]))

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await waitFor(() => {
      expect(latestContext!.searchHistory).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "history-1",
            conversationId: "remote-thread-1",
          }),
        ])
      )
    })

    expect(localStorage.getItem("knowledge_qa_history")).not.toBeNull()
  })

  it("does not crash when clearing empty history and storage removal is blocked", async () => {
    const removeItemSpy = vi
      .spyOn(Storage.prototype, "removeItem")
      .mockImplementation(() => {
        throw new DOMException("Blocked", "SecurityError")
      })
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    expect(() =>
      render(
        <KnowledgeQAProvider>
          <ContextProbe />
        </KnowledgeQAProvider>
      )
    ).not.toThrow()

    await waitFor(() => expect(latestContext).not.toBeNull())
    expect(latestContext!.searchHistory).toEqual([])

    removeItemSpy.mockRestore()
    consoleErrorSpy.mockRestore()
  })

  it("still hydrates server history when local history storage is malformed", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    localStorage.setItem("knowledge_qa_history", "{not-valid-json")
    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/api/v1/chat/conversations?")) {
        return {
          ok: true,
          status: 200,
          json: async () => [
            {
              id: "server-thread-1",
              title: "Recovered server thread",
              message_count: 4,
              last_modified: "2026-02-19T10:00:00.000Z",
              keywords: ["__knowledge_QA__"],
            },
          ],
          text: async () => "",
        }
      }
      if (path.includes("/messages-with-context")) {
        return {
          ok: true,
          status: 200,
          json: async () => [],
          text: async () => "",
        }
      }
      return {
        ok: false,
        status: 404,
        json: async () => [],
        text: async () => "",
      }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await waitFor(() => {
      expect(latestContext!.searchHistory).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "server-thread-1",
            conversationId: "server-thread-1",
            query: "Recovered server thread",
          }),
        ])
      )
    })

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      "Failed to parse Knowledge QA local history:",
      expect.any(SyntaxError)
    )
    consoleErrorSpy.mockRestore()
  })

  it("restores query, answer, sources, and citations when selecting history", async () => {
    localStorage.setItem("knowledge_qa_history", JSON.stringify([baseHistoryItem]))
    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/messages-with-context")) {
        return {
          ok: true,
          status: 200,
          json: async () => [
            {
              id: "msg-user",
              role: "user",
              content: "How did findings change?",
              created_at: "2026-02-18T10:00:00.000Z",
            },
            {
              id: "msg-assistant",
              role: "assistant",
              content: "Fallback answer",
              created_at: "2026-02-18T10:00:02.000Z",
              rag_context: {
                search_query: "How did findings change?",
                generated_answer: "Final answer [1]",
                settings_snapshot: {
                  sources: ["media_db", "notes"],
                  include_media_ids: [42],
                  include_note_ids: ["note-7"],
                  enable_web_fallback: false,
                },
                retrieved_documents: [
                  {
                    id: "doc-1",
                    title: "Spec",
                    source_type: "media_db",
                    excerpt: "Evidence snippet",
                    score: 0.91,
                    page_number: 2,
                  },
                ],
              },
            },
          ],
          text: async () => "",
        }
      }
      return {
        ok: false,
        status: 404,
        json: async () => [],
        text: async () => "",
      }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.restoreFromHistory(baseHistoryItem)
    })

    await waitFor(() => {
      expect(latestContext!.preset).toBe("fast")
      expect(latestContext!.query).toBe("How did findings change?")
      expect(latestContext!.answer).toBe("Final answer [1]")
      expect(latestContext!.results).toHaveLength(1)
      expect(latestContext!.settings.sources).toEqual(["media_db", "notes"])
      expect(latestContext!.settings.include_media_ids).toEqual([42])
      expect(latestContext!.settings.include_note_ids).toEqual(["note-7"])
      expect(latestContext!.settings.enable_web_fallback).toBe(false)
      expect(latestContext!.citations).toEqual(
        expect.arrayContaining([expect.objectContaining({ index: 1 })])
      )
      expect(latestContext!.messages).toHaveLength(2)
    })
  })

  it("hydrates partial payloads without failing and clears stale results", async () => {
    localStorage.setItem("knowledge_qa_history", JSON.stringify([baseHistoryItem]))
    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/messages-with-context")) {
        return {
          ok: true,
          status: 200,
          json: async () => [
            {
              id: "msg-user-2",
              role: "user",
              content: "Question with no context payload",
              created_at: "2026-02-18T10:01:00.000Z",
            },
            {
              id: "msg-assistant-2",
              role: "assistant",
              content: "Answer without rag context",
              created_at: "2026-02-18T10:01:02.000Z",
            },
          ],
          text: async () => "",
        }
      }
      return {
        ok: false,
        status: 404,
        json: async () => [],
        text: async () => "",
      }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectThread("remote-thread-1")
    })

    await waitFor(() => {
      expect(latestContext!.query).toBe("Question with no context payload")
      expect(latestContext!.answer).toBe("Answer without rag context")
      expect(latestContext!.results).toHaveLength(0)
      expect(latestContext!.citations).toHaveLength(0)
    })
  })

  it("preserves the active thread state when loading a different thread fails", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})

    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/remote-thread-1/")) {
        return {
          ok: true,
          status: 200,
          json: async () => [
            {
              id: "msg-user-1",
              role: "user",
              content: "Primary thread question",
              created_at: "2026-02-18T10:00:00.000Z",
            },
            {
              id: "msg-assistant-1",
              role: "assistant",
              content: "Primary thread answer [1]",
              created_at: "2026-02-18T10:00:02.000Z",
              rag_context: {
                search_query: "Primary thread question",
                generated_answer: "Primary thread answer [1]",
                retrieved_documents: [
                  {
                    id: "doc-primary",
                    title: "Primary source",
                    source_type: "media_db",
                    excerpt: "Primary evidence",
                    score: 0.97,
                  },
                ],
              },
            },
          ],
          text: async () => "",
        }
      }
      if (path.includes("/remote-thread-2/")) {
        return {
          ok: false,
          status: 503,
          json: async () => ({ detail: "temporary failure" }),
          text: async () => "temporary failure",
        }
      }
      return {
        ok: false,
        status: 404,
        json: async () => [],
        text: async () => "",
      }
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectThread("remote-thread-1")
    })

    await waitFor(() => {
      expect(latestContext!.currentThreadId).toBe("remote-thread-1")
      expect(latestContext!.query).toBe("Primary thread question")
      expect(latestContext!.answer).toBe("Primary thread answer [1]")
      expect(latestContext!.results).toHaveLength(1)
      expect(latestContext!.messages).toHaveLength(2)
    })

    await act(async () => {
      await latestContext!.selectThread("remote-thread-2")
    })

    await waitFor(() => {
      expect(latestContext!.currentThreadId).toBe("remote-thread-1")
      expect(latestContext!.query).toBe("Primary thread question")
      expect(latestContext!.answer).toBe("Primary thread answer [1]")
      expect(latestContext!.results).toHaveLength(1)
      expect(latestContext!.messages).toHaveLength(2)
      expect(latestContext!.error).toBe("Unable to load this conversation right now.")
    })

    consoleErrorSpy.mockRestore()
  })

  it("keeps the newer thread selected when an older thread load resolves late", async () => {
    let resolveFirstThread: ((value: { ok: boolean; status: number; json: () => Promise<unknown[]>; text: () => Promise<string> }) => void) | null =
      null

    fetchWithAuthMock.mockImplementation((path: string) => {
      if (path.includes("/remote-thread-1/")) {
        return new Promise((resolve) => {
          resolveFirstThread = resolve
        })
      }
      if (path.includes("/remote-thread-2/")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => [
            {
              id: "msg-user-2",
              role: "user",
              content: "Newer thread question",
              created_at: "2026-02-18T11:00:00.000Z",
            },
            {
              id: "msg-assistant-2",
              role: "assistant",
              content: "Newer thread answer [1]",
              created_at: "2026-02-18T11:00:02.000Z",
              rag_context: {
                search_query: "Newer thread question",
                generated_answer: "Newer thread answer [1]",
                retrieved_documents: [
                  {
                    id: "doc-newer",
                    title: "Newer source",
                    source_type: "media_db",
                    excerpt: "Newer evidence",
                    score: 0.93,
                  },
                ],
              },
            },
          ],
          text: async () => "",
        })
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        json: async () => [],
        text: async () => "",
      })
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    act(() => {
      void latestContext!.selectThread("remote-thread-1")
    })

    await waitFor(() => expect(fetchWithAuthMock).toHaveBeenCalledTimes(2))

    act(() => {
      void latestContext!.selectThread("remote-thread-2")
    })

    await waitFor(() => {
      expect(latestContext!.currentThreadId).toBe("remote-thread-2")
      expect(latestContext!.query).toBe("Newer thread question")
      expect(latestContext!.answer).toBe("Newer thread answer [1]")
    })

    resolveFirstThread?.({
      ok: true,
      status: 200,
      json: async () => [
        {
          id: "msg-user-1",
          role: "user",
          content: "Older thread question",
          created_at: "2026-02-18T10:00:00.000Z",
        },
        {
          id: "msg-assistant-1",
          role: "assistant",
          content: "Older thread answer [1]",
          created_at: "2026-02-18T10:00:02.000Z",
          rag_context: {
            search_query: "Older thread question",
            generated_answer: "Older thread answer [1]",
            retrieved_documents: [
              {
                id: "doc-older",
                title: "Older source",
                source_type: "media_db",
                excerpt: "Older evidence",
                score: 0.91,
              },
            ],
          },
        },
      ],
      text: async () => "",
    })

    await act(async () => {
      await Promise.resolve()
    })

    expect(latestContext!.currentThreadId).toBe("remote-thread-2")
    expect(latestContext!.query).toBe("Newer thread question")
    expect(latestContext!.answer).toBe("Newer thread answer [1]")
  })

  it("toggles pin state and persists to localStorage", async () => {
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    localStorage.setItem("knowledge_qa_history", JSON.stringify([baseHistoryItem]))
    await act(async () => {
      await latestContext!.loadSearchHistory()
    })
    await waitFor(() => expect(latestContext!.searchHistory).toHaveLength(1))

    act(() => {
      latestContext!.toggleHistoryPin("history-1")
    })

    await waitFor(() => {
      expect(latestContext!.searchHistory[0]?.pinned).toBe(true)
      const persisted = JSON.parse(localStorage.getItem("knowledge_qa_history") || "[]")
      expect(persisted[0]?.pinned).toBe(true)
    })
  })

  it("preserves newer pin changes when a stale history load resolves late", async () => {
    let resolveServerHistory:
      | ((
          value: {
            ok: boolean
            status: number
            json: () => Promise<unknown[]>
            text: () => Promise<string>
          }
        ) => void)
      | null = null

    localStorage.setItem("knowledge_qa_history", JSON.stringify([baseHistoryItem]))
    fetchWithAuthMock.mockImplementation((path: string) => {
      if (path.includes("/api/v1/chat/conversations?")) {
        return new Promise((resolve) => {
          resolveServerHistory = resolve
        })
      }
      if (path.includes("/messages-with-context")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => [],
          text: async () => "",
        })
      }
      return Promise.resolve({
        ok: false,
        status: 404,
        json: async () => [],
        text: async () => "",
      })
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    await waitFor(() => expect(latestContext!.searchHistory).toHaveLength(1))

    act(() => {
      latestContext!.toggleHistoryPin("history-1")
    })

    await waitFor(() => {
      expect(latestContext!.searchHistory[0]?.pinned).toBe(true)
    })

    resolveServerHistory?.({
      ok: true,
      status: 200,
      json: async () => [
        {
          id: "remote-thread-1",
          title: "Hydrate this thread",
          message_count: 3,
          last_modified: "2026-02-18T10:00:00.000Z",
          keywords: ["__knowledge_QA__"],
        },
      ],
      text: async () => "",
    })

    await act(async () => {
      await Promise.resolve()
    })

    expect(latestContext!.searchHistory).toHaveLength(1)
    expect(latestContext!.searchHistory[0]?.id).toBe("history-1")
    expect(latestContext!.searchHistory[0]?.pinned).toBe(true)
  })

  it("re-runs the saved query when restoring a legacy entry without conversation id", async () => {
    const legacyHistoryItem: SearchHistoryItem = {
      ...baseHistoryItem,
      id: "legacy-history",
      query: "Re-run this saved query",
      conversationId: undefined,
    }
    ragSearchMock.mockResolvedValue({
      results: [
        {
          id: "doc-legacy-1",
          content: "Legacy source snippet",
          metadata: { title: "Legacy Source" },
          score: 0.87,
        },
      ],
      generated_answer: "Recovered answer [1]",
      metadata: {},
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.restoreFromHistory(legacyHistoryItem)
    })

    await waitFor(() => {
      expect(ragSearchMock).toHaveBeenCalledTimes(1)
      expect(latestContext!.answer).toBe("Recovered answer [1]")
      expect(latestContext!.results).toHaveLength(1)
      expect(latestContext!.citations).toEqual(
        expect.arrayContaining([expect.objectContaining({ index: 1 })])
      )
    })
  })

  it("re-runs the saved query when restoring a local-only conversation id", async () => {
    const localHistoryItem: SearchHistoryItem = {
      ...baseHistoryItem,
      id: "local-history",
      query: "Recover from local-only session",
      conversationId: "local-1234",
      settingsSnapshot: {
        sources: ["notes"],
        include_note_ids: ["note-local-1"],
        enable_web_fallback: false,
      },
    }
    ragSearchMock.mockResolvedValue({
      results: [
        {
          id: "doc-local-1",
          content: "Local source snippet",
          metadata: { title: "Local Source" },
          score: 0.73,
        },
      ],
      generated_answer: "Recovered local answer [1]",
      metadata: {},
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.restoreFromHistory(localHistoryItem)
    })

    await waitFor(() => {
      expect(ragSearchMock).toHaveBeenCalledTimes(1)
      expect(ragSearchMock).toHaveBeenCalledWith(
        "Recover from local-only session",
        expect.objectContaining({
          sources: ["notes"],
          include_note_ids: ["note-local-1"],
          enable_web_fallback: false,
        })
      )
      expect(latestContext!.answer).toBe("Recovered local answer [1]")
      expect(latestContext!.results).toHaveLength(1)
      expect(latestContext!.settings.sources).toEqual(["notes"])
      expect(latestContext!.settings.include_note_ids).toEqual(["note-local-1"])
      expect(latestContext!.settings.enable_web_fallback).toBe(false)
    })
  })

  it("stores degraded trust state for answer payloads without citations", async () => {
    searchCharactersMock.mockResolvedValue([
      { id: 7, name: "Helpful AI Assistant" },
    ])
    addChatMessageMock.mockImplementation(async (_threadId: string, payload: any) => ({
      id: payload?.role === "assistant" ? "assistant-remote" : "user-remote",
      created_at: "2026-02-18T10:05:00.000Z",
    }))
    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/rag-context")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ success: true }),
          text: async () => "",
        }
      }
      if (path.includes("/api/v1/chat/conversations/thread-1")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ keywords: [], version: 1 }),
          text: async () => "",
        }
      }
      return {
        ok: false,
        status: 404,
        json: async () => [],
        text: async () => "",
      }
    })
    ragSearchMock.mockResolvedValue({
      results: [
        {
          id: "doc-uncited",
          content: "Evidence snippet",
          metadata: { title: "Evidence" },
          score: 0.83,
        },
      ],
      generated_answer: "Answer without inline citation.",
      metadata: {},
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    act(() => {
      latestContext!.setQuery("Explain the uncited result")
    })
    await waitFor(() => expect(latestContext!.query).toBe("Explain the uncited result"))

    await act(async () => {
      await latestContext!.search()
    })

    await waitFor(() => {
      expect(latestContext!.answer).toBe("Answer without inline citation.")
      expect(latestContext!.answerTrustState).toBe("uncited_degraded_answer")
      expect(latestContext!.searchHistory[0]).toEqual(
        expect.objectContaining({
          query: "Explain the uncited result",
          trustState: "uncited_degraded_answer",
        })
      )
    })
  })

  it("marks failed searches as failed trust state in current state and history", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    ragSearchMock.mockRejectedValue(new Error("network offline"))

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    act(() => {
      latestContext!.setQuery("Search while offline")
    })
    await waitFor(() => expect(latestContext!.query).toBe("Search while offline"))

    await act(async () => {
      await latestContext!.search()
    })

    await waitFor(() => {
      expect(latestContext!.error).toBeTruthy()
      expect(latestContext!.answerTrustState).toBe("failed_search")
      expect(latestContext!.searchHistory[0]).toEqual(
        expect.objectContaining({
          query: "Search while offline",
          hasAnswer: false,
          sourcesCount: 0,
          trustState: "failed_search",
        })
      )
    })

    consoleErrorSpy.mockRestore()
  })

  it("preserves unsynced local trust over a cited answer when thread persistence falls back locally", async () => {
    ragSearchMock.mockResolvedValue({
      results: [
        {
          id: "doc-local-unsynced",
          content: "Local fallback evidence",
          metadata: { title: "Local fallback" },
          score: 0.88,
        },
      ],
      generated_answer: "Cited answer [1]",
      metadata: {},
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())
    act(() => {
      latestContext!.setQuery("Search in a local-only thread")
    })
    await waitFor(() => expect(latestContext!.query).toBe("Search in a local-only thread"))

    await act(async () => {
      await latestContext!.search()
    })

    await waitFor(() => {
      expect(latestContext!.isLocalOnlyThread).toBe(true)
      expect(latestContext!.answer).toBe("Cited answer [1]")
      expect(latestContext!.citations).toEqual(
        expect.arrayContaining([expect.objectContaining({ index: 1 })])
      )
      expect(latestContext!.answerTrustState).toBe("unsynced_local_result")
      expect(latestContext!.searchHistory[0]).toEqual(
        expect.objectContaining({
          query: "Search in a local-only thread",
          trustState: "unsynced_local_result",
        })
      )
    })
  })
})
