// @vitest-environment jsdom
/**
 * G19 follow-up (Wave 6): the full-page Playground offers Stop on the reply
 * bubble a request is filling before streaming starts, as the side panel does.
 * The bubble counts as pending only once the last row is the bot "▋"
 * placeholder and the request's controller is live; a previous finished
 * reply never shows Stop.
 */
import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PageAssistContext } from "@/context"
import { PlaygroundChat } from "../PlaygroundChat"

const useMessageOptionState = vi.hoisted(() => ({
  value: {
    messages: [] as any[],
    setMessages: vi.fn(),
    streaming: false,
    isProcessing: false,
    regenerateLastMessage: vi.fn(),
    isSearchingInternet: false,
    editMessage: vi.fn(),
    deleteMessage: vi.fn(),
    toggleMessagePinned: vi.fn(),
    ttsEnabled: false,
    onSubmit: vi.fn(),
    actionInfo: null,
    messageSteeringMode: "none",
    setMessageSteeringMode: vi.fn(),
    messageSteeringForceNarrate: false,
    setMessageSteeringForceNarrate: vi.fn(),
    clearMessageSteering: vi.fn(),
    createChatBranch: vi.fn(),
    createCompareBranch: vi.fn(),
    temporaryChat: false,
    serverChatId: null as string | null,
    serverChatCharacterId: null as string | null,
    stopStreamingRequest: vi.fn(),
    isEmbedding: false,
    compareMode: false,
    compareFeatureEnabled: false,
    compareSelectionByCluster: {},
    setCompareSelectionForCluster: vi.fn(),
    compareActiveModelsByCluster: {},
    setCompareActiveModelsForCluster: vi.fn(),
    setCompareSelectedModels: vi.fn(),
    historyId: "history-1",
    setSelectedModel: vi.fn(),
    setCompareMode: vi.fn(),
    sendPerModelReply: vi.fn(),
    compareCanonicalByCluster: {},
    setCompareCanonicalForCluster: vi.fn(),
    compareContinuationModeByCluster: {},
    setCompareContinuationModeForCluster: vi.fn(),
    setCompareParentForHistory: vi.fn(),
    compareSplitChats: {},
    setCompareSplitChat: vi.fn(),
    compareMaxModels: 3,
    selectedAssistant: null as unknown
  }
}))

vi.mock("@/hooks/chat/useHistorySelection", async (original) => ({
  ...(await original<typeof import("@/hooks/chat/useHistorySelection")>()),
  useHistorySelectionContext: () => null
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key
  })
}))

vi.mock("@tanstack/react-query", async (original) => {
  const actual = await original<typeof import("@tanstack/react-query")>()
  const queryClient = new actual.QueryClient()
  return {
    ...actual,
    useQueryClient: () => queryClient,
    useQuery: () => ({ data: [] })
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false]
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => useMessageOptionState.value
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/components/Common/ChatGreetingPicker", () => ({
  ChatGreetingPicker: () => <div data-testid="chat-greeting-picker" />
}))

vi.mock("./PlaygroundEmpty", () => ({
  PlaygroundEmpty: () => <div data-testid="playground-empty" />
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: { message: string; isBot: boolean; isPending?: boolean; isStreaming?: boolean }) => (
    <div
      data-testid="playground-message-mock"
      data-pending={String(Boolean(props.isPending))}>
      {props.message}
    </div>
  )
}))

const pendingRows = (controller: AbortController | null, lastMessage: string) => {
  Object.assign(useMessageOptionState.value, {
    streaming: false,
    isProcessing: true,
    messages: [
      { id: "u1", role: "user", isBot: false, name: "You", message: "Earlier question" },
      { id: "a1", role: "assistant", isBot: true, name: "Assistant", message: "Earlier answer" },
      { id: "u2", role: "user", isBot: false, name: "You", message: "Next question" },
      { id: "a2", role: "assistant", isBot: true, name: "Assistant", message: lastMessage }
    ]
  })
  render(
    <PageAssistContext.Provider
      value={{ controller, setController: vi.fn(), embeddingController: null, setEmbeddingController: vi.fn() }}>
      <PlaygroundChat />
    </PageAssistContext.Provider>
  )
  return screen.getAllByTestId("playground-message-mock").map((row) => row.getAttribute("data-pending"))
}

describe("PlaygroundChat pending reply Stop (G19)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("marks only the reply bubble a request is filling before streaming starts", () => {
    expect(pendingRows(new AbortController(), "▋")).toEqual(["false", "false", "false", "true"])
  })

  it("does not mark the previous finished reply while the request is still being prepared", () => {
    expect(pendingRows(new AbortController(), "Earlier finished answer")).toEqual(["false", "false", "false", "false"])
  })

  it("stops marking the reply once the request is stopped", () => {
    const stopped = new AbortController()
    stopped.abort()
    expect(pendingRows(stopped, "▋")).toEqual(["false", "false", "false", "false"])
  })

  it("marks nothing without a request in flight", () => {
    expect(pendingRows(null, "▋")).toEqual(["false", "false", "false", "false"])
  })
})
