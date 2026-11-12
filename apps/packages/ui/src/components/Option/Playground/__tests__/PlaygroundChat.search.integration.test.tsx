// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundChat } from "../PlaygroundChat"

const useMessageOptionState = vi.hoisted(() => ({
  value: {
    messages: [
      {
        id: "m-user",
        role: "user",
        isBot: false,
        name: "You",
        message: "Find this in search"
      },
      {
        id: "m-assistant",
        role: "assistant",
        isBot: true,
        name: "Model",
        message: "Assistant response for search highlighting"
      }
    ],
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
    serverChatId: "chat-1",
    serverChatCharacterId: null,
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
    compareMaxModels: 3
  }
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

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null]
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
  PlaygroundMessage: (props: {
    message: string
    searchQuery?: string
    searchMatch?: "active" | "match" | null
  }) => (
    <div
      data-testid="playground-message-mock"
      data-search-query={props.searchQuery || ""}
      data-search-match={props.searchMatch || ""}
    >
      {props.message}
    </div>
  )
}))

describe("PlaygroundChat search integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("passes search query and active match metadata to message cards", () => {
    render(
      <PlaygroundChat
        searchQuery="search"
        matchedMessageIndices={new Set([1])}
        activeSearchMessageIndex={1}
      />
    )

    const cards = screen.getAllByTestId("playground-message-mock")
    expect(cards).toHaveLength(2)
    expect(cards[0]).toHaveAttribute("data-search-query", "search")
    expect(cards[0]).toHaveAttribute("data-search-match", "")
    expect(cards[1]).toHaveAttribute("data-search-query", "search")
    expect(cards[1]).toHaveAttribute("data-search-match", "active")
  })
})
