// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundChat } from "../PlaygroundChat"

const useMessageOptionState = vi.hoisted(() => ({
  value: {
    messages: [],
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
    serverChatLoadState: "failed",
    serverChatLoadError: "Failed to load conversation.",
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

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: [] })
}))

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
  ChatGreetingPicker: () => null
}))

vi.mock("../PlaygroundEmpty", () => ({
  PlaygroundEmpty: () => <div data-testid="playground-empty" />
}))

vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: () => null
}))

describe("PlaygroundChat selected server chat load state", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("uses the tighter empty-state top spacing when no messages are present", () => {
    useMessageOptionState.value = {
      ...useMessageOptionState.value,
      serverChatLoadState: "idle",
      serverChatLoadError: null
    }

    render(<PlaygroundChat />)

    const emptyState = screen.getByTestId("playground-empty")
    expect(emptyState.parentElement).toHaveClass("mt-4")
    expect(emptyState.parentElement).not.toHaveClass("mt-8")
    expect(emptyState.parentElement?.parentElement).toHaveClass("pt-8")
    expect(emptyState.parentElement?.parentElement).not.toHaveClass("pt-16")
  })

  it("omits the starter deck when the parent chat surface disallows it", () => {
    useMessageOptionState.value = {
      ...useMessageOptionState.value,
      serverChatLoadState: "idle",
      serverChatLoadError: null
    }

    render(<PlaygroundChat showStarterDeck={false} />)

    expect(screen.queryByTestId("playground-empty")).not.toBeInTheDocument()
  })

  it("shows a selected-chat load failure state instead of the empty state", () => {
    useMessageOptionState.value = {
      ...useMessageOptionState.value,
      serverChatLoadState: "failed",
      serverChatLoadError: "Failed to load conversation."
    }

    render(<PlaygroundChat />)

    expect(screen.getByText("Failed to load conversation.")).toBeInTheDocument()
    expect(screen.queryByTestId("playground-empty")).not.toBeInTheDocument()
  })
})
