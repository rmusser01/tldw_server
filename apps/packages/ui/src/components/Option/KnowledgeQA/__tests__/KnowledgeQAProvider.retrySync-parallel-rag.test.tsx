// Must be imported before the provider so the authority mock registers first.
import "./knowledgeQaAuthorityFixture"
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
    createChat: (...args: unknown[]) => createChatMock(...args),
    deleteChat: (...args: unknown[]) => deleteChatMock(...args),
    searchCharacters: (...args: unknown[]) => searchCharactersMock(...args),
    listCharacters: (...args: unknown[]) => listCharactersMock(...args),
    getChat: vi.fn().mockResolvedValue({ version: 1 }),
  },
}))

let latestContext: ReturnType<typeof useKnowledgeQA> | null = null

function ContextProbe() {
  const context = useKnowledgeQA()
  React.useEffect(() => {
    latestContext = context
  }, [context])
  return null
}

const okJson = (payload: unknown) => ({
  ok: true,
  status: 200,
  json: async () => payload,
  text: async () => "",
})

describe("KnowledgeQAProvider retrySync rag-context write parallelism (WP-48)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    latestContext = null
    trackMetricMock.mockResolvedValue(undefined)
    ragSearchMock.mockResolvedValue({
      results: [],
      generated_answer: null,
    })
    createChatMock.mockResolvedValue({ id: "thread-1", version: 1 })
    addChatMessageMock.mockResolvedValue({ id: "msg-x" })
    deleteChatMock.mockResolvedValue(undefined)
    searchCharactersMock.mockResolvedValue([])
    listCharactersMock.mockResolvedValue([])
    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/rag-context")) {
        return okJson({ success: true })
      }
      if (path.includes("/messages-with-context")) {
        return okJson([])
      }
      return okJson([])
    })
  })

  it("keeps the message chain moving while a rag-context write is in flight", async () => {
    // Two searches accumulate user+assistant pairs (each assistant carries a
    // rag context), giving retrySync a 4-message tail with 2 rag writes.
    ragSearchMock.mockResolvedValue({
      results: [],
      generated_answer: "Answer one",
    })
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )
    await waitFor(() => expect(latestContext).not.toBeNull())

    act(() => {
      latestContext!.setQuery("first question")
    })
    await act(async () => {
      await latestContext!.search()
    })

    ragSearchMock.mockResolvedValue({
      results: [],
      generated_answer: "Answer two",
    })
    act(() => {
      latestContext!.setQuery("second question")
    })
    await act(async () => {
      await latestContext!.search()
    })

    const issuedMessageWrites: string[] = []
    const issuedRagContextWrites: string[] = []
    let releaseHangingRagWrite: (() => void) | null = null
    let hangNextRagWrite = true

    addChatMessageMock.mockImplementation(async () => {
      const id = `retry-msg-${issuedMessageWrites.length + 1}`
      issuedMessageWrites.push(id)
      return { id }
    })

    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/rag-context")) {
        issuedRagContextWrites.push(path)
        if (hangNextRagWrite) {
          hangNextRagWrite = false
          await new Promise<void>((resolve) => {
            releaseHangingRagWrite = resolve
          })
        }
        return okJson({ success: true })
      }
      if (path.includes("/messages-with-context")) {
        return okJson([])
      }
      return okJson([])
    })

    let retryPromise: Promise<boolean> | null = null
    await act(async () => {
      retryPromise = latestContext!.retrySync()
      // Let the microtask chain run without awaiting completion.
      await Promise.resolve()
    })

    // While the first rag-context write hangs, every remaining message write
    // must already be issued: rag-context writes never block the chain.
    await waitFor(
      () => {
        expect(issuedMessageWrites).toHaveLength(4)
      },
      { timeout: 3000 }
    )
    expect(issuedRagContextWrites).toHaveLength(2)

    releaseHangingRagWrite!()
    await act(async () => {
      await expect(retryPromise).resolves.toBe(true)
    })
  }, 15000)

  it("still marks the sync failed when a deferred rag-context write reports failure", async () => {
    ragSearchMock.mockResolvedValue({
      results: [],
      generated_answer: "Answer one",
    })
    render(
      <KnowledgeQAProvider>
        <ContextProbe />
      </KnowledgeQAProvider>
    )
    await waitFor(() => expect(latestContext).not.toBeNull())
    act(() => {
      latestContext!.setQuery("one question")
    })
    await act(async () => {
      await latestContext!.search()
    })

    let ragWriteCount = 0
    fetchWithAuthMock.mockImplementation(async (path: string) => {
      if (path.includes("/rag-context")) {
        ragWriteCount += 1
        return okJson({ success: false })
      }
      if (path.includes("/messages-with-context")) {
        return okJson([])
      }
      return { ok: false, status: 404, json: async () => [], text: async () => "" }
    })

    let retryOutcome: boolean | null = null
    await act(async () => {
      retryOutcome = await latestContext!.retrySync()
    })

    // The retry tail carries at least one assistant rag write; its failure
    // must surface as a failed sync, not a silent success.
    expect(ragWriteCount).toBeGreaterThanOrEqual(1)
    expect(retryOutcome).toBe(false)
    expect(latestContext!.extensionFailureState).not.toBeNull()
  }, 15000)
})
