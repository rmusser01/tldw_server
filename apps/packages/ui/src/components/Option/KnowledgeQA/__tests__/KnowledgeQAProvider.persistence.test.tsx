import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"

const ragSearchMock = vi.fn()
const addChatMessageMock = vi.fn()
const createChatMock = vi.fn()
const deleteChatMock = vi.fn()
const fetchWithAuthMock = vi.fn()
const messageOpenMock = vi.fn()
const searchCharactersMock = vi.fn()
const listCharactersMock = vi.fn()
const trackMetricMock = vi.fn()
let storedPresetValue: unknown = undefined
let storedSettingsValue: unknown = undefined
let storedStreamingFlagValue: unknown = undefined

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string) => {
    if (key === "ragSearchPreset") return [storedPresetValue]
    if (key === "ragSearchSettingsV2") return [storedSettingsValue]
    if (key === "ff_knowledgeQaStreaming") return [storedStreamingFlagValue]
    return [undefined]
  },
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
    createChat: (...args: unknown[]) => createChatMock(...args),
    deleteChat: (...args: unknown[]) => deleteChatMock(...args),
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

describe("KnowledgeQAProvider persistence safeguards", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    latestContext = null
    storedPresetValue = undefined
    storedSettingsValue = undefined
    storedStreamingFlagValue = undefined
    trackMetricMock.mockResolvedValue(undefined)
    ragSearchMock.mockResolvedValue({
      results: [],
      generated_answer: null,
    })
    createChatMock.mockResolvedValue({ id: "thread-1", version: 1 })
    addChatMessageMock.mockResolvedValue({ id: "msg-1" })
    deleteChatMock.mockResolvedValue(undefined)
    searchCharactersMock.mockResolvedValue([])
    listCharactersMock.mockResolvedValue([])
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

  it("switches into local-only mode when thread creation falls back to local ids", async () => {
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    act(() => {
      latestContext!.setQuery("query that triggers local fallback")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => {
      expect(latestContext!.isSearching).toBe(false)
      expect(latestContext!.currentThreadId).toMatch(/^local-/)
      expect(latestContext!.isLocalOnlyThread).toBe(true)
    })
  })

  it("shows persistence warning only on first chat message save failure", async () => {
    addChatMessageMock.mockRejectedValue(new Error("save failed"))

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectThread("remote-thread")
    })
    await waitFor(() => expect(latestContext!.isLocalOnlyThread).toBe(false))

    act(() => {
      latestContext!.setQuery("first attempt")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(latestContext!.isSearching).toBe(false))

    act(() => {
      latestContext!.setQuery("second attempt")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(latestContext!.isSearching).toBe(false))

    const warningCalls = messageOpenMock.mock.calls.filter((args) => {
      const payload = args[0] as { type?: string; content?: string }
      return (
        payload?.type === "warning" &&
        payload?.content ===
          "Unable to save conversation. Results are available but may not persist."
      )
    })
    expect(warningCalls).toHaveLength(1)
  })

  it("keeps successful search results visible when backend message sync fails", async () => {
    searchCharactersMock.mockResolvedValue([
      { id: 7, name: "Helpful AI Assistant" },
    ])
    addChatMessageMock.mockRejectedValue(new Error("message sync failed"))
    ragSearchMock.mockResolvedValue({
      results: [
        {
          id: "doc-unsynced-1",
          content: "Unsynced evidence",
          metadata: { title: "Unsynced source" },
          score: 0.9,
        },
      ],
      generated_answer: "Unsynced answer [1]",
      metadata: {},
    })

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    act(() => {
      latestContext!.setQuery("question with sync failure")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => {
      expect(latestContext!.isSearching).toBe(false)
      expect(latestContext!.answer).toBe("Unsynced answer [1]")
      expect(latestContext!.results).toHaveLength(1)
      expect(latestContext!.isLocalOnlyThread).toBe(true)
      expect(latestContext!.answerTrustState).toBe("unsynced_local_result")
      expect(latestContext!.extensionFailureState).toBe(
        "search_succeeded_sync_failed"
      )
      expect(latestContext!.retrySync).toBeTypeOf("function")
    })

    addChatMessageMock.mockReset()
    addChatMessageMock
      .mockResolvedValueOnce({ id: "msg-user-synced" })
      .mockResolvedValueOnce({ id: "msg-assistant-synced" })
    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/rag-context")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ success: true }),
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

    await act(async () => {
      await expect(latestContext!.retrySync()).resolves.toBe(true)
    })

    await waitFor(() => {
      expect(ragSearchMock).toHaveBeenCalledTimes(1)
      expect(latestContext!.extensionFailureState).toBeNull()
      expect(latestContext!.isLocalOnlyThread).toBe(false)
      expect(latestContext!.answerTrustState).toBe("cited_answer")
    })
  })

  it("persists materialized evidence identifiers in RAG context", async () => {
    let persistedRagContextBody: Record<string, any> | null = null
    searchCharactersMock.mockResolvedValue([
      { id: 7, name: "Helpful AI Assistant" },
    ])
    addChatMessageMock
      .mockResolvedValueOnce({ id: "msg-user-1" })
      .mockResolvedValueOnce({ id: "msg-assistant-1" })
    ragSearchMock.mockResolvedValue({
      results: [
        {
          id: "media:42:chunk:7",
          excerpt: "Visible matched excerpt.",
          sourceId: "42",
          sourceType: "media_db",
          chunkId: "7",
          evidenceOrigin: "local_library",
          sourceStatus: "searched",
          unavailableReason: null,
          metadata: {
            title: "Grounded source",
            source_type: "media_db",
            source_id: "42",
            chunk_id: "7",
            evidence_origin: "local_library",
            source_status: "searched",
            unavailable_reason: null,
          },
          score: 0.91,
        },
      ],
      generated_answer: "Grounded answer [1]",
      metadata: {
        knowledge_trust: {
          state: "no_answer_insufficient_evidence",
          reason_codes: ["missing_inspectable_evidence"],
          evidence_origin: "local_library",
        },
      },
    })
    fetchWithAuthMock.mockImplementation(async (path: string, init?: RequestInit) => {
      if (path.includes("/rag-context")) {
        persistedRagContextBody = JSON.parse(String(init?.body || "{}"))
        return {
          ok: true,
          status: 200,
          json: async () => ({ success: true }),
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

    act(() => {
      latestContext!.setQuery("materialized evidence")
    })
    act(() => {
      void latestContext!.search()
    })

    await waitFor(() => expect(persistedRagContextBody).not.toBeNull())
    expect(latestContext!.answerTrustState).toBe("no_answer_insufficient_evidence")
    expect(latestContext!.answerTrustReasonCodes).toEqual([
      "missing_inspectable_evidence",
    ])
    expect(latestContext!.answerEvidenceOrigin).toBe("local_library")
    expect(persistedRagContextBody!.rag_context).toMatchObject({
      trust_state: "no_answer_insufficient_evidence",
      trust_reason_codes: ["missing_inspectable_evidence"],
      trust_evidence_origin: "local_library",
    })
    const [document] = persistedRagContextBody!.rag_context.retrieved_documents
    expect(document).toMatchObject({
      id: "media:42:chunk:7",
      source_id: "42",
      source_type: "media_db",
      chunk_id: "7",
      excerpt: "Visible matched excerpt.",
      evidence_origin: "local_library",
      source_status: "searched",
      score: 0.91,
    })
    expect(document).not.toHaveProperty("unavailable_reason")
  })

  it("starts a fresh topic with cleared visible state", async () => {
    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/remote-thread/")) {
        return {
          ok: true,
          status: 200,
          json: async () => [
            {
              id: "msg-user-remote",
              role: "user",
              content: "Prior topic question",
              created_at: "2026-03-16T11:00:00.000Z",
            },
            {
              id: "msg-assistant-remote",
              role: "assistant",
              content: "Prior topic answer [1]",
              created_at: "2026-03-16T11:00:02.000Z",
              rag_context: {
                search_query: "Prior topic question",
                generated_answer: "Prior topic answer [1]",
                retrieved_documents: [
                  {
                    id: "doc-prior-1",
                    title: "Prior source",
                    source_type: "media_db",
                    excerpt: "Prior evidence",
                    score: 0.91,
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
      await latestContext!.selectThread("remote-thread")
    })

    await waitFor(() => {
      expect(latestContext!.currentThreadId).toBe("remote-thread")
      expect(latestContext!.query).toBe("Prior topic question")
      expect(latestContext!.answer).toBe("Prior topic answer [1]")
      expect(latestContext!.results).toHaveLength(1)
      expect(latestContext!.messages).toHaveLength(2)
    })

    const freshTopicAction = (latestContext as unknown as {
      startNewTopic?: () => Promise<string>
    }).startNewTopic

    expect(freshTopicAction).toBeTypeOf("function")

    if (typeof freshTopicAction === "function") {
      await act(async () => {
        await freshTopicAction()
      })
    }

    await waitFor(() => {
      expect(latestContext!.currentThreadId).not.toBe("remote-thread")
      expect(latestContext!.query).toBe("")
      expect(latestContext!.answer).toBeNull()
      expect(latestContext!.results).toEqual([])
      expect(latestContext!.messages).toEqual([])
    })
  })

  it("does not let a stale fresh-topic creation overwrite a later thread selection", async () => {
    let resolveFreshTopicCreate: ((value: Record<string, unknown>) => void) | null = null
    searchCharactersMock.mockResolvedValue([{ id: 7, name: "Helpful AI Assistant" }])
    listCharactersMock.mockResolvedValue([{ id: 7, name: "Helpful AI Assistant" }])
    createChatMock.mockImplementationOnce(
      () =>
        new Promise<Record<string, unknown>>((resolve) => {
          resolveFreshTopicCreate = resolve
        })
    )

    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/remote-thread-2/")) {
        return {
          ok: true,
          status: 200,
          json: async () => [
            {
              id: "msg-user-remote-2",
              role: "user",
              content: "Selected thread question",
              created_at: "2026-03-16T13:00:00.000Z",
            },
            {
              id: "msg-assistant-remote-2",
              role: "assistant",
              content: "Selected thread answer [1]",
              created_at: "2026-03-16T13:00:02.000Z",
              rag_context: {
                search_query: "Selected thread question",
                generated_answer: "Selected thread answer [1]",
                retrieved_documents: [
                  {
                    id: "doc-remote-2",
                    title: "Selected source",
                    source_type: "media_db",
                    excerpt: "Selected evidence",
                    score: 0.95,
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

    const freshTopicAction = (latestContext as unknown as {
      startNewTopic?: () => Promise<string>
    }).startNewTopic

    expect(freshTopicAction).toBeTypeOf("function")

    if (typeof freshTopicAction === "function") {
      act(() => {
        void freshTopicAction()
      })
    }

    await waitFor(() => expect(createChatMock).toHaveBeenCalledTimes(1))

    await act(async () => {
      await latestContext!.selectThread("remote-thread-2")
    })

    await waitFor(() => {
      expect(latestContext!.currentThreadId).toBe("remote-thread-2")
      expect(latestContext!.query).toBe("Selected thread question")
      expect(latestContext!.answer).toBe("Selected thread answer [1]")
    })

    resolveFreshTopicCreate?.({ id: "fresh-topic-stale", version: 1 })

    await act(async () => {
      await Promise.resolve()
    })

    expect(latestContext!.currentThreadId).toBe("remote-thread-2")
    expect(latestContext!.query).toBe("Selected thread question")
    expect(latestContext!.answer).toBe("Selected thread answer [1]")
    expect(deleteChatMock).toHaveBeenCalledWith("fresh-topic-stale")
  })

  it("clears the active session after deleting the currently open remote thread", async () => {
    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/remote-thread/")) {
        return {
          ok: true,
          status: 200,
          json: async () => [
            {
              id: "msg-user-remote",
              role: "user",
              content: "Active remote question",
              created_at: "2026-03-16T12:00:00.000Z",
            },
            {
              id: "msg-assistant-remote",
              role: "assistant",
              content: "Active remote answer [1]",
              created_at: "2026-03-16T12:00:02.000Z",
              rag_context: {
                search_query: "Active remote question",
                generated_answer: "Active remote answer [1]",
                retrieved_documents: [
                  {
                    id: "doc-remote-1",
                    title: "Remote source",
                    source_type: "media_db",
                    excerpt: "Remote evidence",
                    score: 0.94,
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
    localStorage.setItem(
      "knowledge_qa_history",
      JSON.stringify([
        {
          id: "history-remote-thread",
          query: "Remote thread title",
          timestamp: "2026-03-16T12:00:00.000Z",
          sourcesCount: 1,
          hasAnswer: true,
          conversationId: "remote-thread",
          keywords: ["__knowledge_QA__"],
        },
      ])
    )

    await act(async () => {
      await latestContext!.loadSearchHistory()
    })

    await waitFor(() =>
      expect(latestContext!.searchHistory).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: "history-remote-thread",
            conversationId: "remote-thread",
          }),
        ])
      )
    )

    await act(async () => {
      await latestContext!.selectThread("remote-thread")
    })

    await waitFor(() => {
      expect(latestContext!.currentThreadId).toBe("remote-thread")
      expect(latestContext!.query).toBe("Active remote question")
      expect(latestContext!.answer).toBe("Active remote answer [1]")
      expect(latestContext!.results).toHaveLength(1)
      expect(latestContext!.messages).toHaveLength(2)
    })

    await act(async () => {
      await latestContext!.deleteHistoryItem("history-remote-thread")
    })

    await waitFor(() => {
      expect(deleteChatMock).toHaveBeenCalledWith("remote-thread")
      expect(latestContext!.currentThreadId).toBeNull()
      expect(latestContext!.query).toBe("")
      expect(latestContext!.answer).toBeNull()
      expect(latestContext!.results).toEqual([])
      expect(latestContext!.messages).toEqual([])
      expect(latestContext!.searchHistory).toEqual([])
    })
  })
})
