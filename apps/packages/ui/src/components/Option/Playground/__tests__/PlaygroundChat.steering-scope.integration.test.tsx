// @vitest-environment jsdom
/**
 * CM-09 (#3101, G17): role-play steering belongs to character chats. A
 * character chat's last reply offers "Continue as user", "Impersonate user"
 * and "Force narrate" and keeps an armed intent; once the same Playground shows
 * a plain chat, the controls go away and any armed intent is dropped, because
 * every send applies it and the plain chat has no control to clear it.
 */
import React from "react"
import { act, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useStoreMessageOption } from "@/store/option"
import { PlaygroundChat } from "../PlaygroundChat"

const CHARACTER = {
  kind: "character",
  id: "5",
  name: "Robot",
  metadata: { selectionMode: "tracked" }
}

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

// The real overflow menu is covered by MessageActionsBar.overflow-groups; here
// only what the chat hands each reply matters.
vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: (props: {
    message: string
    isBot: boolean
    onMessageSteeringModeChange?: unknown
    onMessageSteeringForceNarrateChange?: unknown
    onClearMessageSteering?: unknown
  }) => (
    <div
      data-testid="playground-message-mock"
      data-bot={String(props.isBot)}
      data-steering-offered={String(
        Boolean(
          props.onMessageSteeringModeChange &&
            props.onMessageSteeringForceNarrateChange &&
            props.onClearMessageSteering
        )
      )}>
      {props.message}
    </div>
  )
}))

const lastReply = () =>
  screen
    .getAllByTestId("playground-message-mock")
    .filter((card) => card.getAttribute("data-bot") === "true")
    .at(-1)!

const steering = () => {
  const state = useStoreMessageOption.getState()
  return {
    mode: state.messageSteeringMode,
    forceNarrate: state.messageSteeringForceNarrate
  }
}

describe("PlaygroundChat role-play steering scope", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useStoreMessageOption.getState().clearMessageSteering()
    Object.assign(useMessageOptionState.value, {
      selectedAssistant: CHARACTER,
      serverChatId: null,
      serverChatCharacterId: null,
      compareMode: false,
      compareFeatureEnabled: false,
      messages: [
        { id: "u1", role: "user", isBot: false, name: "You", message: "Hello there" },
        { id: "a1", role: "assistant", isBot: true, name: "Robot", message: "Beep, hello." }
      ]
    })
  })

  it("offers steering on a character chat reply and keeps an armed intent", () => {
    render(<PlaygroundChat />)

    act(() => {
      useStoreMessageOption.getState().setMessageSteeringForceNarrate(true)
    })

    expect(lastReply()).toHaveAttribute("data-steering-offered", "true")
    expect(steering()).toEqual({ mode: "none", forceNarrate: true })
  })

  it("offers steering on a saved character chat once its character is known", () => {
    Object.assign(useMessageOptionState.value, {
      serverChatId: "chat-robot",
      serverChatCharacterId: "5"
    })

    render(<PlaygroundChat />)

    expect(lastReply()).toHaveAttribute("data-steering-offered", "true")
  })

  it("drops steering armed in a character chat when a plain chat opens", () => {
    const view = render(<PlaygroundChat />)
    act(() => {
      const state = useStoreMessageOption.getState()
      state.setMessageSteeringMode("impersonate_user")
      state.setMessageSteeringForceNarrate(true)
    })

    useMessageOptionState.value.selectedAssistant = null
    useMessageOptionState.value.messages = [
      { id: "u2", role: "user", isBot: false, name: "You", message: "Plain question" },
      { id: "a2", role: "assistant", isBot: true, name: "Assistant", message: "Plain answer" }
    ]
    view.rerender(<PlaygroundChat />)

    expect(lastReply()).toHaveAttribute("data-steering-offered", "false")
    expect(steering()).toEqual({ mode: "none", forceNarrate: false })
  })
})
