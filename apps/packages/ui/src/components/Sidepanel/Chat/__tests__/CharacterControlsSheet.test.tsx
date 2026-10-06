// @vitest-environment jsdom
import React from "react"
import { act, render, screen } from "@testing-library/react"
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
  onTrackedSelection: vi.fn(async () => true),
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
        onClick={async () => {
          const onTrackedSelection = props.onTrackedSelection as
            | ((selection: Record<string, unknown>, isCurrent: () => boolean) => Promise<boolean>)
            | undefined
          const selected = { kind: "character", id: "char-2", name: "Tracked Character" }
          if (onTrackedSelection && await onTrackedSelection(selected, () => true) === false) return
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
  useStoreMessageOption: Object.assign((selector: (input: typeof state.option) => unknown) =>
    selector(state.option), { getState: () => state.option })
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
    mocks.onTrackedSelection.mockResolvedValue(true)
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

  it.each(["tracked choice", "close", "source changes"])("retires an older held overlay clear after %s", async intent => {
    state.settings = { assistantOverlay: { kind: "persona", id: "p", name: "Source overlay", system_prompt_snapshot: "Source prompt" } }
    let finish!: (value: null) => void
    mocks.updateSettings.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const user = userEvent.setup()
    const mounted = render(<CharacterControlsSheet onTrackedSelection={mocks.onTrackedSelection} />)
    await user.click(screen.getByRole("button", { name: "Clear overlay" }))
    if (intent === "tracked choice") {
      await user.click(screen.getByRole("button", { name: "Start tracked character chat" }))
      await user.click(screen.getByRole("button", { name: "Choose tracked assistant" }))
      expect(mocks.onTrackedSelection).toHaveBeenCalledOnce()
    } else if (intent === "source changes") state.option = { ...state.option, historyId: "new-source" }
    else mounted.unmount()
    await act(async () => { finish(null) })
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
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

  it("opens a tracked picker without changing the source assistant or overlay", async () => {
    const user = userEvent.setup()
    state.settings = { assistantOverlay: { kind: "persona", id: "persona-1", name: "Guide Persona" } }
    render(<CharacterControlsSheet onTrackedSelection={mocks.onTrackedSelection} onRequestClose={mocks.onRequestClose} />)
    await user.click(screen.getByRole("button", { name: "Start tracked character chat" }))
    expect(mocks.updateSettings).not.toHaveBeenCalled()
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.beforeTrackedStart).not.toHaveBeenCalled()
    expect(mocks.assistantSelect).toHaveBeenLastCalledWith(expect.objectContaining({
      variant: "inline", selectionModePreference: "tracked", initialTab: "character", onTrackedSelection: expect.any(Function)
    }))
    await user.click(screen.getByRole("button", { name: "Choose tracked assistant" }))
    expect(mocks.onTrackedSelection).toHaveBeenCalledWith(expect.objectContaining({ id: "char-2" }), expect.any(Function))
    expect(mocks.onRequestClose).toHaveBeenCalledOnce()
    expect(mocks.clearChat).not.toHaveBeenCalled()
    expect(mocks.setServerChatCharacterId).not.toHaveBeenCalled()
  })

  it("leaves the picker and source intact after a declined tracked selection", async () => {
    const user = userEvent.setup()
    mocks.onTrackedSelection.mockResolvedValue(false)
    render(<CharacterControlsSheet onTrackedSelection={mocks.onTrackedSelection} onRequestClose={mocks.onRequestClose} />)
    await user.click(screen.getByRole("button", { name: "Start tracked persona chat" }))
    await user.click(screen.getByRole("button", { name: "Choose tracked assistant" }))
    expect(screen.getByTestId("tracked-assistant-picker")).toBeInTheDocument()
    expect(mocks.onRequestClose).not.toHaveBeenCalled()
    expect(mocks.updateSettings).not.toHaveBeenCalled()
    expect(mocks.setSelectedAssistant).not.toHaveBeenCalled()
    expect(mocks.setServerChatAssistantKind).not.toHaveBeenCalled()
  })

  it("opens tracked sessions through the standard server-chat selection path", async () => {
    const user = userEvent.setup()
    render(
      <CharacterControlsSheet
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
