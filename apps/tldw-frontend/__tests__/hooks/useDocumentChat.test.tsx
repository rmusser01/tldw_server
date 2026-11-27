import { describe, it, expect, beforeEach, vi } from "vitest"
import { renderHook, act, waitFor } from "@testing-library/react"
import { DEFAULT_RAG_SETTINGS } from "@/services/rag/unified-rag"
import { useStoreMessageOption } from "@/store/option"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { useDocumentChat } from "@/hooks/document-workspace/useDocumentChat"

const operation: QuickIngestOperation = {
  authorityKey: "verified-account",
  authorityRevision: 1,
  requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" },
  signal: new AbortController().signal,
  isCurrent: () => true,
  assertCurrent: () => {}
}

let connectionState = {
  state: { isConnected: true, mode: "normal" }
}

vi.mock("@/store/connection", () => ({
  useConnectionStore: (selector: (state: typeof connectionState) => unknown) =>
    selector(connectionState)
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    requestScope: operation.requestScope,
    scopeSignal: operation.signal,
    scopeInvalidatedSignal: operation.signal,
    release: () => {}
  })
}))
vi.mock("@/db/dexie/helpers", async (original) => ({
  ...await original<typeof import("@/db/dexie/helpers")>(),
  getAllHistoriesByDocId: async () => []
}))

const resetChatStore = (
  overrides: Partial<ReturnType<typeof useStoreMessageOption.getState>> = {}
) => {
  useStoreMessageOption.setState({
    messages: [],
    history: [],
    historyId: null,
    isFirstMessage: true,
    serverChatId: null,
    serverChatTitle: null,
    serverChatCharacterId: null,
    serverChatMetaLoaded: false,
    serverChatState: null,
    serverChatVersion: null,
    serverChatTopic: null,
    serverChatClusterId: null,
    serverChatSource: null,
    serverChatExternalRef: null,
    selectedKnowledge: null,
    replyTarget: null,
    actionInfo: null,
    ragMediaIds: null,
    ragSources: DEFAULT_RAG_SETTINGS.sources,
    ...overrides
  })
}

describe("useDocumentChat", () => {
  beforeEach(() => {
    connectionState = { state: { isConnected: true, mode: "normal" } }
    resetChatStore()
  })

  it("swaps chat drafts per document and restores when revisiting", async () => {
    const { rerender, result } = renderHook(
      ({ mediaId }) => useDocumentChat(mediaId, operation),
      { initialProps: { mediaId: 1 } }
    )

    expect(useStoreMessageOption.getState().ragMediaIds).toEqual([1])

    await waitFor(() => expect(result.current.isChatReady).toBe(true))
    act(() => result.current.setInputValue("Doc 1 draft"))

    act(() => {
      rerender({ mediaId: 2 })
    })

    expect(useStoreMessageOption.getState().ragMediaIds).toEqual([2])
    await waitFor(() => expect(result.current.isChatReady).toBe(true))
    expect(result.current.inputValue).toBe("")
    act(() => result.current.setInputValue("Doc 2 draft"))

    act(() => {
      rerender({ mediaId: 1 })
    })

    expect(result.current.inputValue).toBe("Doc 1 draft")
  })

  it("toggles RAG per document and restores ragSources on disable", async () => {
    const baselineSources = ["media_db", "notes"]
    resetChatStore({ ragSources: baselineSources })

    const { result } = renderHook(() => useDocumentChat(1, operation))

    await waitFor(() => expect(result.current.isChatReady).toBe(true))
    expect(result.current.ragEnabled).toBe(false)

    act(() => {
      result.current.setRagEnabled(true)
    })

    const stateAfterEnable = useStoreMessageOption.getState()
    expect(stateAfterEnable.selectedKnowledge?.id).toBe("document:1")
    expect(stateAfterEnable.ragSources).toEqual(["media_db"])

    act(() => {
      result.current.setRagEnabled(false)
    })

    const stateAfterDisable = useStoreMessageOption.getState()
    expect(stateAfterDisable.selectedKnowledge).toBeNull()
    expect(stateAfterDisable.ragSources).toEqual(baselineSources)
  })

  it("restores baseline chat state and rag settings on unmount", () => {
    const baselineMessages = [
      {
        isBot: false,
        name: "You",
        message: "Baseline message",
        sources: []
      }
    ]
    const baselineHistory = [{ role: "user", content: "Baseline message" }]
    const baselineRagSources = ["notes"]

    resetChatStore({
      messages: baselineMessages,
      history: baselineHistory,
      historyId: "baseline",
      ragMediaIds: [99],
      ragSources: baselineRagSources
    })

    const { unmount } = renderHook(() => useDocumentChat(1, operation))

    act(() => {
      useStoreMessageOption.setState({
        messages: [
          {
            isBot: false,
            name: "You",
            message: "Doc message",
            sources: []
          }
        ],
        historyId: "doc-1",
        ragMediaIds: [1],
        ragSources: ["media_db"]
      })
    })

    unmount()

    const restored = useStoreMessageOption.getState()
    expect(restored.messages).toEqual(baselineMessages)
    expect(restored.history).toEqual(baselineHistory)
    expect(restored.historyId).toBe("baseline")
    expect(restored.ragMediaIds).toEqual([99])
    expect(restored.ragSources).toEqual(baselineRagSources)
  })
})
