// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  dispatchOpenAssistantSelect: vi.fn(),
  updateSettings: vi.fn(async () => null),
  clearChat: vi.fn(),
  surfaceResetChat: vi.fn(),
  selectServerChat: vi.fn(),
  setServerChatAssistantKind: vi.fn(),
  setServerChatAssistantId: vi.fn(),
  setServerChatCharacterId: vi.fn(),
  setServerChatPersonaMemoryMode: vi.fn(),
  setServerChatMetaLoaded: vi.fn(),
  setSelectedAssistant: vi.fn(async () => undefined),
  beforeTrackedStart: vi.fn(async () => undefined),
  onRequestClose: vi.fn(),
  tabsCreate: vi.fn(async () => undefined),
  windowOpen: vi.fn(),
  assistantSelect: vi.fn((props: Record<string, unknown>) => props)
}))

const state = vi.hoisted(() => ({
  runtime: null as {
    id?: string
    getURL?: (path: string) => string
  } | null,
  selectedAssistant: null as Record<string, unknown> | null,
  settings: {
    assistantOverlay: null
  } as Record<string, unknown> | null,
  option: {
    historyId: "history-1",
    serverChatId: "chat-1",
    serverChatAssistantKind: null as string | null,
    serverChatAssistantId: null as string | null,
    serverChatCharacterId: null as string | null,
    setServerChatAssistantKind: mocks.setServerChatAssistantKind,
    setServerChatAssistantId: mocks.setServerChatAssistantId,
    setServerChatCharacterId: mocks.setServerChatCharacterId,
    setServerChatPersonaMemoryMode: mocks.setServerChatPersonaMemoryMode,
    setServerChatMetaLoaded: mocks.setServerChatMetaLoaded
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

vi.mock("@/components/Common/AssistantSelect", () => ({
  AssistantSelect: (props: Record<string, unknown>) => {
    mocks.assistantSelect(props)
    return (
      <button
        type="button"
        onClick={() => {
          const onSelectionComplete = props.onSelectionComplete as
            | ((selection: Record<string, unknown>) => void | Promise<void>)
            | undefined
          void onSelectionComplete?.({
            kind: "character",
            id: "char-2",
            name: "Tracked Character"
          })
        }}
      >
        {String(
          props.variant === "inline"
            ? "Choose tracked assistant"
            : props.labelOverride ?? "AssistantSelect"
        )}
      </button>
    )
  }
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

vi.mock("@/utils/browser-runtime", () => ({
  getBrowserRuntime: () => state.runtime,
  isExtensionRuntime: (
    runtime?: {
      id?: string
    } | null
  ) => Boolean(runtime?.id)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    tabs: {
      create: mocks.tabsCreate
    }
  }
}))

import { CharacterControlsSheet } from "../CharacterControlsSheet"

describe("CharacterControlsSheet", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.open = mocks.windowOpen as unknown as typeof window.open
    state.runtime = null
    state.selectedAssistant = null
    state.settings = {
      assistantOverlay: null
    }
    state.option = {
      historyId: "history-1",
      serverChatId: "chat-1",
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatCharacterId: null,
      setServerChatAssistantKind: mocks.setServerChatAssistantKind,
      setServerChatAssistantId: mocks.setServerChatAssistantId,
      setServerChatCharacterId: mocks.setServerChatCharacterId,
      setServerChatPersonaMemoryMode: mocks.setServerChatPersonaMemoryMode,
      setServerChatMetaLoaded: mocks.setServerChatMetaLoaded
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

  it("shows overlay and tracked actions separately for plain chats", () => {
    render(
      <CharacterControlsSheet
        beforeTrackedStart={mocks.beforeTrackedStart}
        onRequestClose={mocks.onRequestClose}
      />
    )

    expect(screen.getByTestId("chat-character-controls-sheet")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Apply overlay" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Start tracked character chat" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Start tracked persona chat" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Captain Mira" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Plain chat" })).toBeNull()
  })

  it("shows clear overlay only when the current chat is in overlay mode", () => {
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

    render(<CharacterControlsSheet />)

    expect(screen.getByRole("button", { name: "Change overlay" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Clear overlay" })).toBeInTheDocument()
  })

  it("hides overlay actions when the current chat is already tracked", () => {
    state.option = {
      historyId: "history-1",
      serverChatId: "chat-1",
      serverChatAssistantKind: "character",
      serverChatAssistantId: null,
      serverChatCharacterId: "char-7",
      setServerChatAssistantKind: mocks.setServerChatAssistantKind,
      setServerChatAssistantId: mocks.setServerChatAssistantId,
      setServerChatCharacterId: mocks.setServerChatCharacterId,
      setServerChatPersonaMemoryMode: mocks.setServerChatPersonaMemoryMode,
      setServerChatMetaLoaded: mocks.setServerChatMetaLoaded
    }
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

    render(<CharacterControlsSheet />)

    expect(screen.getByText("Tracked character chat")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Apply overlay" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Change overlay" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Clear overlay" })).toBeNull()
    expect(screen.getByRole("button", { name: "Start tracked character chat" })).toBeInTheDocument()
  })

  it("clears overlay state before starting a tracked chat", async () => {
    const user = userEvent.setup()
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

    render(
      <CharacterControlsSheet
        beforeTrackedStart={mocks.beforeTrackedStart}
        onRequestClose={mocks.onRequestClose}
        resetChat={mocks.surfaceResetChat}
      />
    )

    await user.click(
      screen.getByRole("button", { name: "Start tracked character chat" })
    )

    expect(mocks.beforeTrackedStart).toHaveBeenCalledTimes(1)
    expect(mocks.updateSettings).toHaveBeenCalledWith({
      assistantOverlay: null
    })
    expect(mocks.setSelectedAssistant).toHaveBeenCalledWith(null)
    expect(mocks.onRequestClose).not.toHaveBeenCalled()
    expect(mocks.surfaceResetChat).not.toHaveBeenCalled()
    expect(mocks.clearChat).not.toHaveBeenCalled()
    expect(mocks.dispatchOpenAssistantSelect).not.toHaveBeenCalled()
    expect(mocks.assistantSelect).toHaveBeenLastCalledWith(
      expect.objectContaining({
        variant: "inline",
        selectionModePreference: "tracked",
        initialTab: "character",
        onSelectionComplete: expect.any(Function)
      })
    )

    await user.click(
      screen.getByRole("button", { name: "Choose tracked assistant" })
    )

    expect(mocks.onRequestClose).toHaveBeenCalledTimes(1)
    expect(mocks.surfaceResetChat).toHaveBeenCalledTimes(1)
    expect(mocks.clearChat).not.toHaveBeenCalled()
    expect(mocks.setServerChatAssistantKind).toHaveBeenCalledWith("character")
    expect(mocks.setServerChatAssistantId).toHaveBeenCalledWith("char-2")
    expect(mocks.setServerChatCharacterId).toHaveBeenCalledWith("char-2")
    expect(mocks.setServerChatPersonaMemoryMode).toHaveBeenCalledWith(null)
    expect(mocks.setServerChatMetaLoaded).toHaveBeenCalledWith(false)
    expect(mocks.surfaceResetChat.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.setServerChatAssistantKind.mock.invocationCallOrder[0]
    )
    expect(mocks.setServerChatMetaLoaded.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.onRequestClose.mock.invocationCallOrder[0]
    )
  })

  it("opens tracked sessions through the standard server-chat selection path", async () => {
    const user = userEvent.setup()
    render(
      <CharacterControlsSheet
        beforeTrackedStart={mocks.beforeTrackedStart}
        onRequestClose={mocks.onRequestClose}
      />
    )

    await user.click(screen.getByRole("button", { name: "Captain Mira" }))

    expect(mocks.selectServerChat).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "tracked-chat-1",
        title: "Captain Mira"
      })
    )
    expect(mocks.onRequestClose).toHaveBeenCalledTimes(1)
  })

  it("opens the expression editor route in web fallback mode", async () => {
    const user = userEvent.setup()
    state.option = {
      historyId: "history-1",
      serverChatId: "chat-1",
      serverChatAssistantKind: "character",
      serverChatAssistantId: null,
      serverChatCharacterId: "42",
      setServerChatAssistantKind: mocks.setServerChatAssistantKind,
      setServerChatAssistantId: mocks.setServerChatAssistantId,
      setServerChatCharacterId: mocks.setServerChatCharacterId,
      setServerChatPersonaMemoryMode: mocks.setServerChatPersonaMemoryMode,
      setServerChatMetaLoaded: mocks.setServerChatMetaLoaded
    }

    render(<CharacterControlsSheet />)

    await user.click(
      screen.getByRole("button", { name: "Edit character expressions" })
    )

    expect(mocks.windowOpen).toHaveBeenCalledWith(
      "/characters?from=sidepanel-character-controls&focus=expressions&characterId=42",
      "_blank"
    )
  })

  it("opens the expression editor in an extension options tab", async () => {
    const user = userEvent.setup()
    state.runtime = {
      id: "extension-id",
      getURL: (path) => `chrome-extension://extension-id${path}`
    }
    state.option = {
      historyId: "history-1",
      serverChatId: "chat-1",
      serverChatAssistantKind: "character",
      serverChatAssistantId: null,
      serverChatCharacterId: "42",
      setServerChatAssistantKind: mocks.setServerChatAssistantKind,
      setServerChatAssistantId: mocks.setServerChatAssistantId,
      setServerChatCharacterId: mocks.setServerChatCharacterId,
      setServerChatPersonaMemoryMode: mocks.setServerChatPersonaMemoryMode,
      setServerChatMetaLoaded: mocks.setServerChatMetaLoaded
    }

    render(<CharacterControlsSheet />)

    await user.click(
      screen.getByRole("button", { name: "Edit character expressions" })
    )

    expect(mocks.tabsCreate).toHaveBeenCalledWith({
      url: "chrome-extension://extension-id/options.html#/characters?from=sidepanel-character-controls&focus=expressions&characterId=42"
    })
    expect(mocks.windowOpen).not.toHaveBeenCalled()
  })
})
