// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import { PlaygroundChat } from "../PlaygroundChat"

const queryState = vi.hoisted(() => ({
  chatModels: [] as any[],
  chatModelsFetched: true,
  providersStatus: undefined as
    | undefined
    | { any_configured: boolean; providers: any[] }
}))

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
  useQuery: ({ queryKey }: { queryKey?: unknown[] }) => {
    const key = Array.isArray(queryKey) ? queryKey[0] : undefined
    if (key === "playground:chatModels") {
      return {
        data: queryState.chatModels,
        isFetched: queryState.chatModelsFetched,
        refetch: vi.fn()
      }
    }
    if (key === "playground:providersStatus") {
      return {
        data: queryState.providersStatus,
        refetch: vi.fn()
      }
    }
    return { data: [] }
  }
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false]
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useIsConnected: () => true
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
    queryState.chatModels = []
    queryState.chatModelsFetched = true
    queryState.providersStatus = undefined
    useMessageOptionState.value = {
      ...useMessageOptionState.value,
      messages: [],
      serverChatId: "chat-1",
      serverChatLoadState: "failed",
      serverChatLoadError: "Failed to load conversation."
    }
  })

  it("uses the tighter empty-state top spacing when no messages are present", () => {
    queryState.chatModels = [
      {
        api_name: "ollama",
        model: "gemma3:1b",
        provider: "ollama",
        is_configured: true
      }
    ]
    queryState.providersStatus = {
      any_configured: true,
      providers: [{ name: "ollama", configured: true, requires_api_key: false }]
    }
    useMessageOptionState.value = {
      ...useMessageOptionState.value,
      serverChatLoadState: "idle",
      serverChatLoadError: null
    }

    render(
      <MemoryRouter>
        <PlaygroundChat />
      </MemoryRouter>
    )

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

    render(
      <MemoryRouter>
        <PlaygroundChat />
      </MemoryRouter>
    )

    expect(screen.getByText("Failed to load conversation.")).toBeInTheDocument()
    expect(screen.queryByTestId("playground-empty")).not.toBeInTheDocument()
  })

  it("does not show the no-provider setup banner when usable chat models are present", () => {
    queryState.chatModels = [
      {
        api_name: "ollama",
        model: "gemma3:1b",
        provider: "ollama",
        is_configured: true
      }
    ]
    queryState.providersStatus = {
      any_configured: false,
      providers: [{ name: "ollama", configured: true, requires_api_key: false }]
    }
    useMessageOptionState.value = {
      ...useMessageOptionState.value,
      serverChatLoadState: "idle",
      serverChatLoadError: null
    }

    render(
      <MemoryRouter>
        <PlaygroundChat />
      </MemoryRouter>
    )

    expect(
      screen.queryByText("No LLM provider configured")
    ).not.toBeInTheDocument()
    expect(screen.getByTestId("playground-empty")).toBeInTheDocument()
  })

  it("shows the no-provider setup banner when catalog rows are provider-unconfigured", () => {
    queryState.chatModels = [
      {
        api_name: "openai",
        model: "tldw:gpt-4o",
        provider: "openai"
      }
    ]
    queryState.providersStatus = {
      any_configured: false,
      providers: [{ name: "openai", configured: false, requires_api_key: true }]
    }
    useMessageOptionState.value = {
      ...useMessageOptionState.value,
      serverChatLoadState: "idle",
      serverChatLoadError: null
    }

    render(
      <MemoryRouter>
        <PlaygroundChat />
      </MemoryRouter>
    )

    expect(screen.getByText("No LLM provider configured")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open Settings" })).toHaveAttribute(
      "href",
      "/settings/tldw"
    )
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument()
    expect(screen.queryByTestId("playground-empty")).not.toBeInTheDocument()
  })

  it("keeps model setup recovery primary when no usable models are available", () => {
    queryState.chatModels = []
    queryState.providersStatus = {
      any_configured: true,
      providers: [{ name: "openai", configured: true, requires_api_key: true }]
    }
    useMessageOptionState.value = {
      ...useMessageOptionState.value,
      serverChatLoadState: "idle",
      serverChatLoadError: null
    }

    render(
      <MemoryRouter>
        <PlaygroundChat />
      </MemoryRouter>
    )

    expect(screen.getByText("No AI models available")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "refresh models" })
    ).toBeInTheDocument()
    expect(screen.queryByTestId("playground-empty")).not.toBeInTheDocument()
  })
})
