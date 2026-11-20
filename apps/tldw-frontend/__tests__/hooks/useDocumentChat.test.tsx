import { describe, it, expect, beforeEach, vi } from "vitest"
import { renderHook, act } from "@testing-library/react"
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

  it("swaps chat sessions per document and restores when revisiting", () => {
    const { rerender } = renderHook(
      ({ mediaId }) => useDocumentChat(mediaId, operation),
      { initialProps: { mediaId: 1 } }
    )

    expect(useStoreMessageOption.getState().ragMediaIds).toEqual([1])

    const doc1Message = {
      isBot: false,
      name: "You",
      message: "Doc 1 message",
      sources: []
    }

    act(() => {
      useStoreMessageOption.setState({
        messages: [doc1Message],
        history: [{ role: "user", content: "Doc 1 message" }],
        historyId: "doc-1",
        isFirstMessage: false
      })
    })

    act(() => {
      rerender({ mediaId: 2 })
    })

    expect(useStoreMessageOption.getState().ragMediaIds).toEqual([2])
    expect(useStoreMessageOption.getState().messages).toEqual([])

    const doc2Message = {
      isBot: false,
      name: "You",
      message: "Doc 2 message",
      sources: []
    }

    act(() => {
      useStoreMessageOption.setState({
        messages: [doc2Message],
        history: [{ role: "user", content: "Doc 2 message" }],
        historyId: "doc-2",
        isFirstMessage: false
      })
    })

    act(() => {
      rerender({ mediaId: 1 })
    })

    expect(useStoreMessageOption.getState().messages).toEqual([doc1Message])
    expect(useStoreMessageOption.getState().historyId).toBe("doc-1")
  })

  it("toggles RAG per document and restores ragSources on disable", () => {
    const baselineSources = ["media_db", "notes"]
    resetChatStore({ ragSources: baselineSources })

    const { result } = renderHook(() => useDocumentChat(1, operation))

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
