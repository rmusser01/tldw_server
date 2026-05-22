// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  dispatchOpenAssistantSelect: vi.fn(),
  updateSettings: vi.fn(async () => null),
  clearChat: vi.fn(),
  selectServerChat: vi.fn(),
  setSelectedAssistant: vi.fn(async () => undefined)
}))

const state = vi.hoisted(() => ({
  selectedAssistant: null as Record<string, unknown> | null,
  settings: {
    assistantOverlay: {
      kind: "persona",
      id: "persona-1",
      name: "Guide Persona",
      avatar_url: null,
      system_prompt_snapshot: "Persona prompt",
      updatedAt: "2026-05-22T12:00:00.000Z"
    }
  } as Record<string, unknown> | null,
  option: {
    historyId: "history-1",
    serverChatId: "chat-1",
    serverChatAssistantKind: null,
    serverChatAssistantId: null,
    serverChatCharacterId: null
  },
  history: [
    {
      id: "tracked-chat-1",
      title: "Captain Mira",
      assistant_kind: "character",
      character_id: "char-1",
      created_at: "2026-05-22T12:00:00.000Z"
    },
    {
      id: "plain-chat-1",
      title: "Plain chat",
      assistant_kind: null,
      character_id: null,
      created_at: "2026-05-21T12:00:00.000Z"
    }
  ] as Array<Record<string, unknown>>
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [
    state.selectedAssistant,
    mocks.setSelectedAssistant,
    { isLoading: false, setRenderValue: vi.fn() }
  ]
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: state.settings,
    updateSettings: mocks.updateSettings,
    chatKey: "server:chat-1"
  })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (input: typeof state.option) => unknown) =>
    selector(state.option)
}))

vi.mock("@/utils/assistant-select-events", () => ({
  dispatchOpenAssistantSelect: mocks.dispatchOpenAssistantSelect
}))

vi.mock("@/hooks/chat/useClearChat", () => ({
  useClearChat: () => mocks.clearChat
}))

vi.mock("@/hooks/chat/useSelectServerChat", () => ({
  useSelectServerChat: () => mocks.selectServerChat
}))

vi.mock("@/hooks/useServerChatHistory", () => ({
  useServerChatHistory: () => ({
    data: state.history,
    total: state.history.length
  })
}))

import { CharacterControlRail } from "../CharacterControlRail"

describe("CharacterControlRail", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    state.selectedAssistant = null
    state.settings = {
      assistantOverlay: {
        kind: "persona",
        id: "persona-1",
        name: "Guide Persona",
        avatar_url: null,
        system_prompt_snapshot: "Persona prompt",
        updatedAt: "2026-05-22T12:00:00.000Z"
      }
    }
    state.option = {
      historyId: "history-1",
      serverChatId: "chat-1",
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatCharacterId: null
    }
    state.history = [
      {
        id: "tracked-chat-1",
        title: "Captain Mira",
        assistant_kind: "character",
        character_id: "char-1",
        created_at: "2026-05-22T12:00:00.000Z"
      },
      {
        id: "plain-chat-1",
        title: "Plain chat",
        assistant_kind: null,
        character_id: null,
        created_at: "2026-05-21T12:00:00.000Z"
      }
    ]
  })

  it("renders the current mode summary and tracked sessions separately", () => {
    render(<CharacterControlRail />)

    expect(screen.getByText("Overlay personality")).toBeInTheDocument()
    expect(screen.getByText("Guide Persona")).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Tracked sessions" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Captain Mira" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Plain chat" })).toBeNull()
  })

  it("opens the assistant picker with explicit overlay intent", async () => {
    const user = userEvent.setup()
    render(<CharacterControlRail />)

    await user.click(screen.getByRole("button", { name: "Change overlay" }))

    expect(mocks.dispatchOpenAssistantSelect).toHaveBeenCalledWith({
      tab: "persona",
      applyAs: "overlay",
      source: "character-control-rail"
    })
    expect(mocks.updateSettings).not.toHaveBeenCalled()
  })

  it("clears overlay settings without resetting the conversation", async () => {
    const user = userEvent.setup()
    render(<CharacterControlRail />)

    await user.click(screen.getByRole("button", { name: "Clear overlay" }))

    expect(mocks.updateSettings).toHaveBeenCalledWith({
      assistantOverlay: null
    })
    expect(mocks.clearChat).not.toHaveBeenCalled()
  })

  it("keeps tracked-start actions separate from overlay actions", async () => {
    const user = userEvent.setup()
    render(<CharacterControlRail />)

    await user.click(
      screen.getByRole("button", { name: "Start tracked character chat" })
    )

    expect(mocks.clearChat).toHaveBeenCalledTimes(1)
    expect(mocks.updateSettings).toHaveBeenCalledWith({
      assistantOverlay: null
    })
    expect(mocks.dispatchOpenAssistantSelect).toHaveBeenCalledWith({
      tab: "character",
      applyAs: "tracked",
      source: "character-control-rail"
    })
  })

  it("does not expose overlay actions when the current chat is already tracked", () => {
    state.settings = {
      assistantOverlay: {
        kind: "persona",
        id: "persona-1",
        name: "Guide Persona",
        avatar_url: null,
        system_prompt_snapshot: "Persona prompt",
        updatedAt: "2026-05-22T12:00:00.000Z"
      }
    }
    state.option = {
      historyId: "history-1",
      serverChatId: "chat-1",
      serverChatAssistantKind: "character",
      serverChatAssistantId: null,
      serverChatCharacterId: "char-7"
    }

    render(<CharacterControlRail />)

    expect(screen.getByText("Tracked character chat")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Apply overlay" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Change overlay" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Clear overlay" })).toBeNull()
  })
})
