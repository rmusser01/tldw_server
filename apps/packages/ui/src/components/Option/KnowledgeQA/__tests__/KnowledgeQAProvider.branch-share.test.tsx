import "./knowledgeQaAuthorityFixture"
import React from "react"
import { act, render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { KnowledgeQAProvider, useKnowledgeQA } from "../KnowledgeQAProvider"

const fetchWithAuthMock = vi.fn()
const addChatMessageMock = vi.fn()
const createChatMock = vi.fn()
const deleteChatMock = vi.fn()
const resolveConversationShareLinkMock = vi.fn()
const messageOpenMock = vi.fn()

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [undefined],
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: messageOpenMock,
  }),
}))

vi.mock("@/utils/knowledge-qa-search-metrics", () => ({
  trackKnowledgeQaSearchMetric: vi.fn(),
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn().mockResolvedValue(undefined),
    fetchWithAuth: (...args: unknown[]) => fetchWithAuthMock(...args),
    ragSearch: vi.fn(),
    ragSearchStream: vi.fn(),
    addChatMessage: (...args: unknown[]) => addChatMessageMock(...args),
    createChat: (...args: unknown[]) => createChatMock(...args),
    deleteChat: (...args: unknown[]) => deleteChatMock(...args),
    getChat: vi.fn().mockResolvedValue({ version: 1 }),
    searchCharacters: vi.fn().mockResolvedValue([{ id: 10, name: "Helpful AI Assistant" }]),
    listCharacters: vi.fn().mockResolvedValue([{ id: 10, name: "Helpful AI Assistant" }]),
    resolveConversationShareLink: (...args: unknown[]) =>
      resolveConversationShareLinkMock(...args),
  },
}))

let latestContext: ReturnType<typeof useKnowledgeQA> | null = null

function ContextProbe() {
  latestContext = useKnowledgeQA()
  return null
}

describe("KnowledgeQAProvider branch/share actions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    latestContext = null
    localStorage.clear()

    createChatMock.mockResolvedValue({ id: "branch-thread-1", version: 1 })
    deleteChatMock.mockResolvedValue(undefined)
    addChatMessageMock
      .mockResolvedValueOnce({ id: "branch-u1", created_at: "2026-02-19T10:00:01.000Z" })
      .mockResolvedValueOnce({ id: "branch-a1", created_at: "2026-02-19T10:00:02.000Z" })
    resolveConversationShareLinkMock.mockResolvedValue({
      conversation_id: "shared-thread-9",
      permission: "view",
      shared_by_user_id: "1",
      expires_at: "2026-02-20T10:00:00.000Z",
      messages: [
        {
          id: "shared-u1",
          role: "user",
          content: "Shared question",
          created_at: "2026-02-19T09:00:00.000Z",
        },
        {
          id: "shared-a1",
          role: "assistant",
          content: "Shared answer [1]",
          created_at: "2026-02-19T09:00:02.000Z",
          rag_context: {
            search_query: "Shared question",
            generated_answer: "Shared answer [1]",
            retrieved_documents: [{ id: "doc-1", title: "Shared source", excerpt: "Evidence" }],
          },
        },
      ],
    })
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
          json: async () => [
            {
              id: "u1",
              role: "user",
              content: "Branch source question",
              created_at: "2026-02-19T08:00:00.000Z",
            },
            {
              id: "a1",
              role: "assistant",
              content: "Branch source answer [1]",
              created_at: "2026-02-19T08:00:02.000Z",
              rag_context: {
                search_query: "Branch source question",
                generated_answer: "Branch source answer [1]",
                retrieved_documents: [{ id: "doc-a1", title: "Source A", excerpt: "Alpha" }],
              },
            },
            {
              id: "u2",
              role: "user",
              content: "Latest question",
              created_at: "2026-02-19T08:01:00.000Z",
            },
            {
              id: "a2",
              role: "assistant",
              content: "Latest answer",
              created_at: "2026-02-19T08:01:02.000Z",
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
  })

  it("branches from a historical turn and seeds a new branch thread", async () => {
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectThread("source-thread-1")
    })

    await waitFor(() => expect(latestContext!.messages).toHaveLength(4))

    await act(async () => {
      await latestContext!.branchFromTurn("u1")
    })

    await waitFor(() => {
      expect(createChatMock).toHaveBeenCalled()
      expect(addChatMessageMock).toHaveBeenCalledTimes(2)
      expect(latestContext!.currentThreadId).toBe("branch-thread-1")
      expect(latestContext!.messages).toHaveLength(2)
      expect(latestContext!.messages[0]?.content).toBe("Branch source question")
      expect(latestContext!.messages[1]?.content).toContain("Branch source answer")
      expect(latestContext!.answer).toBe("Branch source answer [1]")
    })
  })

  it("does not restore a stale branch after the user clears the session", async () => {
    let resolveBranchThread: ((value: Record<string, unknown>) => void) | null = null
    createChatMock.mockImplementationOnce(
      () =>
        new Promise<Record<string, unknown>>((resolve) => {
          resolveBranchThread = resolve
        })
    )

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectThread("source-thread-1")
    })

    await waitFor(() => expect(latestContext!.messages).toHaveLength(4))

    act(() => {
      void latestContext!.branchFromTurn("u1")
    })

    await waitFor(() => expect(createChatMock).toHaveBeenCalledTimes(1))

    act(() => {
      latestContext!.clearResults()
    })

    resolveBranchThread?.({ id: "branch-thread-stale", version: 1 })

    await act(async () => {
      await Promise.resolve()
    })

    expect(latestContext!.currentThreadId).toBeNull()
    expect(latestContext!.messages).toEqual([])
    expect(latestContext!.answer).toBeNull()
    expect(addChatMessageMock).not.toHaveBeenCalled()
    expect(deleteChatMock).toHaveBeenCalledWith("branch-thread-stale", expect.objectContaining({ requestScope: expect.objectContaining({ userId: "test-owner" }) }))
  })

  it("hydrates shared conversations through tokenized links", async () => {
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectSharedThread("token-abc")
    })

    await waitFor(() => {
      expect(resolveConversationShareLinkMock).toHaveBeenCalledWith("token-abc")
      expect(latestContext!.currentThreadId).toBeNull()
      expect(latestContext!.messages).toHaveLength(2)
      expect(latestContext!.query).toBe("Shared question")
      expect(latestContext!.answer).toBe("Shared answer [1]")
      expect(latestContext!.results).toHaveLength(1)
    })
  })

  it("handles expired share links with a safe recovery state", async () => {
    resolveConversationShareLinkMock.mockRejectedValueOnce(
      new Error("HTTP 410 expired")
    )

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectSharedThread("token-expired")
    })

    await waitFor(() => {
      expect(resolveConversationShareLinkMock).toHaveBeenCalledWith(
        "token-expired"
      )
      expect(latestContext!.currentThreadId).toBeNull()
      expect(latestContext!.messages).toEqual([])
      expect(latestContext!.results).toEqual([])
      expect(latestContext!.answer).toBeNull()
      expect(latestContext!.error).toBe(
        "Unable to open this shared conversation link."
      )
    })
  })

  it("handles revoked share links with a safe recovery state", async () => {
    resolveConversationShareLinkMock.mockRejectedValueOnce(
      new Error("HTTP 403 revoked")
    )

    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )

    await waitFor(() => expect(latestContext).not.toBeNull())

    await act(async () => {
      await latestContext!.selectSharedThread("token-revoked")
    })

    await waitFor(() => {
      expect(resolveConversationShareLinkMock).toHaveBeenCalledWith(
        "token-revoked"
      )
      expect(latestContext!.currentThreadId).toBeNull()
      expect(latestContext!.messages).toEqual([])
      expect(latestContext!.results).toEqual([])
      expect(latestContext!.answer).toBeNull()
      expect(latestContext!.error).toBe(
        "Unable to open this shared conversation link."
      )
    })
  })
})
