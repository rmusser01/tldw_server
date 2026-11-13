import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Header } from "../Header"
import { resetServerChatSaveStatus } from "@/store/server-chat-save-status"

const clearChatMock = vi.fn()
const setTemporaryChatMock = vi.fn()
const setHeaderShortcutsExpandedMock = vi.fn().mockResolvedValue(undefined)
const setSelectedCharacterMock = vi.fn()
const navigateMock = vi.fn()
let locationPathname = "/chat"

const messageOptionState = {
  clearChat: clearChatMock,
  historyId: "history-1",
  temporaryChat: false,
  setTemporaryChat: setTemporaryChatMock,
  serverChatId: null as string | null
}

let selectedCharacter: { id: string; name: string } | null = null
const forkParent = vi.hoisted(() => ({ value: null as { id: string; title: string } | null, requested: [] as unknown[] }))
const chatFiles = vi.hoisted(() => ({ exportServerChat: vi.fn(), success: vi.fn(), error: vi.fn() }))
vi.mock("@/services/chat-transcript-export", () => ({
  exportServerChat: (...args: unknown[]) => chatFiles.exportServerChat(...args)
}))
vi.mock("@/components/Sidepanel/Chat/BulkFolderPickerModal", () => ({
  default: ({ open, conversationIds }: { open: boolean; conversationIds: string[] }) =>
    open ? <div data-testid="folder-picker" data-ids={conversationIds.join(",")} /> : null
}))

const mockT = (
  key: string,
  fallback?: string,
  values?: Record<string, unknown>
) => {
  if (!fallback) return key
  if (!values) return fallback
  return Object.entries(values).reduce((acc, [name, value]) => {
    return acc.replaceAll(`{{${name}}}`, String(value))
  }, fallback)
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mockT
  })
}))

vi.mock("react-router-dom", () => ({
  useLocation: () => ({ pathname: locationPathname }),
  useNavigate: () => navigateMock
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Modal: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div data-testid="share-modal">{children}</div> : null,
  Button: ({ children, onClick, disabled, ...rest }: any) => (
    <button type="button" onClick={onClick} disabled={disabled} {...rest}>
      {children}
    </button>
  ),
  Input: ({ value, onChange, placeholder }: any) => (
    <input value={value} onChange={onChange} placeholder={placeholder} />
  ),
  message: { success: chatFiles.success, error: chatFiles.error },
  InputNumber: ({ value, onChange }: any) => (
    <input
      type="number"
      value={value}
      onChange={(event) => onChange?.(Number(event.target.value))}
    />
  )
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [false, setHeaderShortcutsExpandedMock]
}))

vi.mock("@/hooks/useDarkmode", () => ({
  useDarkMode: () => ({ mode: "dark", toggleDarkMode: vi.fn() })
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [selectedCharacter, setSelectedCharacterMock]
}))

vi.mock("~/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    ...messageOptionState,
    effectiveAssistantState: selectedCharacter
      ? { mode: "tracked_character", kind: "character", displayName: selectedCharacter.name }
      : { mode: "plain", kind: null, displayName: null }
  })
}))

vi.mock("@/db", () => ({
  getTitleById: vi.fn().mockResolvedValue("Saved title"),
  updateHistory: vi.fn().mockResolvedValue(undefined)
}))

vi.mock("@/hooks/chat/useForkParentChat", () => ({
  useForkParentChat: (chatId: unknown) => {
    forkParent.requested.push(chatId)
    return forkParent.value
  }
}))

vi.mock("../ChatHeader", () => ({
  ChatHeader: ({
    parentChatTitle,
    onOpenParentChat,
    onExportChat,
    onMoveToFolder,
    onOpenCommandPalette,
    onStartSavedChat,
    onStartTemporaryChat,
    onStartCharacterChat,
    activeCharacterName, persistenceKind
  }: {
    onOpenCommandPalette?: () => void
    onStartSavedChat?: () => void
    onStartTemporaryChat?: () => void
    onStartCharacterChat?: () => void
    persistenceKind?: string
    activeCharacterName?: string | null
    parentChatTitle?: string | null
    onOpenParentChat?: () => void
    onExportChat?: (format: "markdown" | "json") => void
    onMoveToFolder?: () => void
  }) => (
    <div>
      {onExportChat ? (
        <button type="button" onClick={() => onExportChat("json")}>Export JSON</button>
      ) : null}
      {onMoveToFolder ? (
        <button type="button" onClick={() => onMoveToFolder()}>Move to folder</button>
      ) : null}
      {onOpenParentChat ? (
        <button type="button" onClick={() => onOpenParentChat()}>
          Back to {parentChatTitle}
        </button>
      ) : null}
      <output data-testid="header-persistence">{persistenceKind}</output>
      <button type="button" onClick={() => onOpenCommandPalette?.()}>
        Open command palette
      </button>
      {onStartSavedChat ? (
        <button type="button" onClick={() => onStartSavedChat()}>
          New saved chat
        </button>
      ) : null}
      {onStartTemporaryChat ? (
        <button type="button" onClick={() => onStartTemporaryChat()}>
          Temporary chat
        </button>
      ) : null}
      {onStartCharacterChat ? (
        <button type="button" onClick={() => onStartCharacterChat()}>
          Character chat
        </button>
      ) : null}
      <div data-testid="active-character">
        {activeCharacterName || "none"}
      </div>
    </div>
  )
}))

vi.mock("@/components/Sidepanel/Chat/TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => null
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listConversationShareLinks: vi.fn(async () => ({ links: [] })),
    createConversationShareLink: vi.fn(),
    revokeConversationShareLink: vi.fn()
  }
}))

describe("Header branch lineage (CM-12)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    messageOptionState.serverChatId = "branch-chat"
    messageOptionState.temporaryChat = false
    forkParent.value = null
    forkParent.requested = []
    locationPathname = "/chat"
  })

  it("leads from a branch back to the chat it came from through the chat's address", () => {
    forkParent.value = { id: "trip", title: "Trip plan" }
    render(<Header />)
    expect(forkParent.requested.at(-1)).toBe("branch-chat")
    fireEvent.click(screen.getByRole("button", { name: "Back to Trip plan" }))
    expect(navigateMock).toHaveBeenCalledWith("/chat?chat=trip")
  })

  it("looks up no lineage off the chat page or in a temporary chat", () => {
    locationPathname = "/notes"
    render(<Header />)
    expect(forkParent.requested.at(-1)).toBeNull()
  })

  it("offers no way back for a chat that isn't a branch", () => {
    render(<Header />)
    expect(screen.queryByRole("button", { name: /^Back to/ })).not.toBeInTheDocument()
  })
})

describe("Header per-chat Export and Move to folder (CS-09)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    messageOptionState.serverChatId = "saved-chat"
    messageOptionState.temporaryChat = false
    forkParent.value = null
    locationPathname = "/chat"
  })

  it("exports the open saved chat from the server", async () => {
    chatFiles.exportServerChat.mockResolvedValue(undefined)
    render(<Header />)
    fireEvent.click(screen.getByRole("button", { name: "Export JSON" }))
    await vi.waitFor(() =>
      expect(chatFiles.exportServerChat).toHaveBeenCalledWith("saved-chat", "json", expect.any(Object))
    )
    await vi.waitFor(() => expect(chatFiles.success).toHaveBeenCalledWith("Chat exported."))
  })

  it("says when the export fails", async () => {
    chatFiles.exportServerChat.mockRejectedValue(new Error("HTTP 503"))
    render(<Header />)
    fireEvent.click(screen.getByRole("button", { name: "Export JSON" }))
    await vi.waitFor(() => expect(chatFiles.error).toHaveBeenCalledWith("Couldn't export this chat. Try again."))
  })

  it("moves the open saved chat to a folder", async () => {
    render(<Header />)
    fireEvent.click(screen.getByRole("button", { name: "Move to folder" }))
    expect(await screen.findByTestId("folder-picker")).toHaveAttribute("data-ids", "saved-chat")
  })

  it("offers neither for a chat that isn't saved on the server", () => {
    messageOptionState.serverChatId = null
    render(<Header />)
    expect(screen.queryByRole("button", { name: "Export JSON" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Move to folder" })).toBeNull()
  })
})

describe("Header character mode sequencing", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetServerChatSaveStatus()
    messageOptionState.serverChatId = null
    selectedCharacter = null
    locationPathname = "/chat"
  })

  it("does not call cold server-linked history saved without current acknowledgement", () => {
    messageOptionState.serverChatId = "cold-unconfirmed-chat"
    render(<Header />)
    expect(screen.getByTestId("header-persistence")).toHaveTextContent("serverUnconfirmed")
  })

  it("enters character mode and opens selection before scene controls when no character is active", () => {
    const actorListener = vi.fn()
    const assistantListener = vi.fn()
    const characterModeListener = vi.fn()
    window.addEventListener("tldw:open-actor-settings", actorListener)
    window.addEventListener("tldw:open-assistant-select", assistantListener)
    window.addEventListener("tldw:character-chat-mode-intent", characterModeListener)

    try {
      render(<Header />)

      fireEvent.click(screen.getByRole("button", { name: "Character chat" }))

      expect(setTemporaryChatMock).not.toHaveBeenCalled()
      expect(clearChatMock).not.toHaveBeenCalled()
      expect(actorListener).not.toHaveBeenCalled()
      expect(characterModeListener).toHaveBeenCalledWith(
        expect.objectContaining({
          detail: { source: "chat-header", characterId: null }
        })
      )
      expect(assistantListener).toHaveBeenCalledWith(
        expect.objectContaining({
          detail: { tab: "character", applyAs: "tracked", source: "chat-header" }
        })
      )
    } finally {
      window.removeEventListener("tldw:open-actor-settings", actorListener)
      window.removeEventListener("tldw:open-assistant-select", assistantListener)
      window.removeEventListener("tldw:character-chat-mode-intent", characterModeListener)
    }
  })

  it("keeps an active character selected and preserves the current chat", () => {
    selectedCharacter = { id: "char-1", name: "Rin" }
    const actorListener = vi.fn()
    const assistantListener = vi.fn()
    const focusListener = vi.fn()
    window.addEventListener("tldw:open-actor-settings", actorListener)
    window.addEventListener("tldw:open-assistant-select", assistantListener)
    window.addEventListener("tldw:focus-composer", focusListener)

    try {
      render(<Header />)

      expect(screen.getByTestId("active-character")).toHaveTextContent("Rin")
      fireEvent.click(screen.getByRole("button", { name: "Character chat" }))

      expect(setTemporaryChatMock).not.toHaveBeenCalled()
      expect(clearChatMock).not.toHaveBeenCalled()
      expect(actorListener).not.toHaveBeenCalled()
      expect(assistantListener).not.toHaveBeenCalled()
      expect(focusListener).toHaveBeenCalledTimes(1)
    } finally {
      window.removeEventListener("tldw:open-actor-settings", actorListener)
      window.removeEventListener("tldw:open-assistant-select", assistantListener)
      window.removeEventListener("tldw:focus-composer", focusListener)
    }
  })

  it("omits chat session actions on non-chat surfaces", () => {
    locationPathname = "/settings/characters"
    selectedCharacter = { id: "char-1", name: "Rin" }
    render(<Header />)

    expect(
      screen.queryByRole("button", { name: "New saved chat" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Temporary chat" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Character chat" })
    ).not.toBeInTheDocument()
    expect(setTemporaryChatMock).not.toHaveBeenCalled()
    expect(clearChatMock).not.toHaveBeenCalled()
    expect(navigateMock).not.toHaveBeenCalled()
  })

  it("preserves identity and temporary mode when the clear action is declined", () => {
    selectedCharacter = { id: "char-1", name: "Rin" }
    clearChatMock.mockReturnValueOnce(false).mockReturnValueOnce(false)
    render(<Header />)
    fireEvent.click(screen.getByRole("button", { name: "New saved chat" }))
    fireEvent.click(screen.getByRole("button", { name: "Temporary chat" }))
    expect(setSelectedCharacterMock).not.toHaveBeenCalled()
    expect(setTemporaryChatMock).not.toHaveBeenCalled()
  })

  it("clears character state when switching back to saved or temporary chat", () => {
    selectedCharacter = { id: "char-1", name: "Rin" }
    render(<Header />)

    fireEvent.click(screen.getByRole("button", { name: "New saved chat" }))
    fireEvent.click(screen.getByRole("button", { name: "Temporary chat" }))

    expect(setSelectedCharacterMock).toHaveBeenCalledTimes(2)
    expect(setSelectedCharacterMock).toHaveBeenCalledWith(null)
    expect(setTemporaryChatMock).toHaveBeenCalledWith(false)
    expect(setTemporaryChatMock).toHaveBeenCalledWith(true)
    expect(clearChatMock).toHaveBeenCalledTimes(2)
  })

  it("dispatches the global command palette event from the header search trigger", () => {
    const openListener = vi.fn()
    window.addEventListener("tldw:open-command-palette", openListener)

    try {
      render(<Header />)

      fireEvent.click(screen.getByRole("button", { name: "Open command palette" }))

      expect(openListener).toHaveBeenCalledWith(expect.any(CustomEvent))
    } finally {
      window.removeEventListener("tldw:open-command-palette", openListener)
    }
  })
})
