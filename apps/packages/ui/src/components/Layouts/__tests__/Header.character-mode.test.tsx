import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { Header } from "../Header"

const clearChatMock = vi.fn()
const setTemporaryChatMock = vi.fn()
const setHeaderShortcutsExpandedMock = vi.fn().mockResolvedValue(undefined)
const setSelectedCharacterMock = vi.fn()
const navigateMock = vi.fn()

const messageOptionState = {
  clearChat: clearChatMock,
  historyId: "history-1",
  temporaryChat: false,
  setTemporaryChat: setTemporaryChatMock,
  serverChatId: null
}

let selectedCharacter: { id: string; name: string } | null = null

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
  useLocation: () => ({ pathname: "/chat" }),
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
  useMessageOption: () => messageOptionState
}))

vi.mock("@/db", () => ({
  getTitleById: vi.fn().mockResolvedValue("Saved title"),
  updateHistory: vi.fn().mockResolvedValue(undefined)
}))

vi.mock("../ChatHeader", () => ({
  ChatHeader: ({
    onStartSavedChat,
    onStartTemporaryChat,
    onStartCharacterChat,
    activeCharacterName
  }: {
    onStartSavedChat?: () => void
    onStartTemporaryChat?: () => void
    onStartCharacterChat?: () => void
    activeCharacterName?: string | null
  }) => (
    <div>
      <button type="button" onClick={() => onStartSavedChat?.()}>
        New saved chat
      </button>
      <button type="button" onClick={() => onStartTemporaryChat?.()}>
        Temporary chat
      </button>
      <button type="button" onClick={() => onStartCharacterChat?.()}>
        Character chat
      </button>
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

describe("Header character mode sequencing", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    selectedCharacter = null
  })

  it("opens character selection before scene controls when no character is active", () => {
    const actorListener = vi.fn()
    const assistantListener = vi.fn()
    window.addEventListener("tldw:open-actor-settings", actorListener)
    window.addEventListener("tldw:open-assistant-select", assistantListener)

    render(<Header />)

    fireEvent.click(screen.getByRole("button", { name: "Character chat" }))

    expect(setTemporaryChatMock).toHaveBeenCalledWith(false)
    expect(clearChatMock).toHaveBeenCalledTimes(1)
    expect(actorListener).not.toHaveBeenCalled()
    expect(assistantListener).toHaveBeenCalledWith(
      expect.objectContaining({
        detail: { tab: "character", source: "chat-header" }
      })
    )

    window.removeEventListener("tldw:open-actor-settings", actorListener)
    window.removeEventListener("tldw:open-assistant-select", assistantListener)
  })

  it("keeps an active character selected when starting a fresh character chat", () => {
    selectedCharacter = { id: "char-1", name: "Rin" }
    const actorListener = vi.fn()
    const assistantListener = vi.fn()
    window.addEventListener("tldw:open-actor-settings", actorListener)
    window.addEventListener("tldw:open-assistant-select", assistantListener)

    render(<Header />)

    expect(screen.getByTestId("active-character")).toHaveTextContent("Rin")
    fireEvent.click(screen.getByRole("button", { name: "Character chat" }))

    expect(setTemporaryChatMock).toHaveBeenCalledWith(false)
    expect(clearChatMock).toHaveBeenCalledTimes(1)
    expect(actorListener).not.toHaveBeenCalled()
    expect(assistantListener).not.toHaveBeenCalled()

    window.removeEventListener("tldw:open-actor-settings", actorListener)
    window.removeEventListener("tldw:open-assistant-select", assistantListener)
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
})
