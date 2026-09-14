import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import commonEn from "@/assets/locale/en/common.json"
import { Header } from "../Header"

const listConversationShareLinksMock = vi.fn()
const createConversationShareLinkMock = vi.fn()
const revokeConversationShareLinkMock = vi.fn()
const setHeaderShortcutsExpandedMock = vi.fn().mockResolvedValue(undefined)
const toggleDarkModeMock = vi.fn()
const setSelectedCharacterMock = vi.fn()
const navigateMock = vi.fn()
const mockT = vi.fn(
  (
    key: string,
    fallback?: string,
    values?: Record<string, unknown>
  ) => {
    if (key === "common:loading.title") return commonEn.loading.title
    if (!fallback) return key
    if (!values) return fallback
    return Object.entries(values).reduce((acc, [name, value]) => {
      return acc.replaceAll(`{{${name}}}`, String(value))
    }, fallback)
  }
)

const messageOptionState = {
  clearChat: vi.fn(),
  historyId: "history-1",
  temporaryChat: false,
  setTemporaryChat: vi.fn(),
  serverChatId: "server-chat-1"
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
  Button: ({
    children,
    onClick,
    disabled,
    loading: _loading,
    danger: _danger,
    ...rest
  }: any) => (
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
  useDarkMode: () => ({ mode: "dark", toggleDarkMode: toggleDarkModeMock })
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, setSelectedCharacterMock]
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
    onOpenShareModal,
    shareStatusLabel,
    shareButtonDisabled
  }: {
    onOpenShareModal?: () => void
    shareStatusLabel?: string | null
    shareButtonDisabled?: boolean
  }) => (
    <div>
      <button
        type="button"
        onClick={() => onOpenShareModal?.()}
        disabled={shareButtonDisabled}
      >
        Open share modal
      </button>
      <div data-testid="header-share-status">{shareStatusLabel || "none"}</div>
    </div>
  )
}))

vi.mock("@/components/Sidepanel/Chat/TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => null
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listConversationShareLinks: (...args: unknown[]) =>
      listConversationShareLinksMock(...args),
    createConversationShareLink: (...args: unknown[]) =>
      createConversationShareLinkMock(...args),
    revokeConversationShareLink: (...args: unknown[]) =>
      revokeConversationShareLinkMock(...args)
  }
}))

describe("Header share links integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: vi.fn().mockResolvedValue(undefined)
      },
      configurable: true
    })
  })

  it("loads share links on modal open and publishes active share status to the header", async () => {
    listConversationShareLinksMock.mockResolvedValue({
      conversation_id: "server-chat-1",
      links: [
        {
          id: "active",
          permission: "view",
          created_at: "2026-02-20T10:00:00.000Z",
          expires_at: "2126-02-20T10:00:00.000Z",
          revoked_at: null,
          share_path: "/knowledge/shared/active"
        },
        {
          id: "expired",
          permission: "view",
          created_at: "2026-02-19T10:00:00.000Z",
          expires_at: "2000-02-20T10:00:00.000Z",
          revoked_at: null,
          share_path: "/knowledge/shared/expired"
        }
      ]
    })

    render(<Header />)

    fireEvent.click(screen.getByRole("button", { name: "Open share modal" }))

    await waitFor(() => {
      expect(listConversationShareLinksMock).toHaveBeenCalledWith("server-chat-1")
    })
    await waitFor(() => {
      expect(screen.getByTestId("header-share-status")).toHaveTextContent(
        "1 active link(s)"
      )
    })
  })

  it("uses a scalar translation key while share links are loading", async () => {
    listConversationShareLinksMock.mockImplementation(
      () => new Promise(() => undefined)
    )

    render(<Header />)
    fireEvent.click(screen.getByRole("button", { name: "Open share modal" }))

    await waitFor(() => {
      expect(listConversationShareLinksMock).toHaveBeenCalledWith("server-chat-1")
    })
    await waitFor(() => {
      expect(screen.getByText(commonEn.loading.title)).toBeInTheDocument()
    })
    expect(mockT).toHaveBeenCalledWith("common:loading.title", "Loading...")
  })

  it("creates and revokes share links with ttl controls", async () => {
    let links: Array<Record<string, unknown>> = []
    listConversationShareLinksMock.mockImplementation(async () => ({
      conversation_id: "server-chat-1",
      links
    }))

    createConversationShareLinkMock.mockImplementation(async () => {
      links = [
        {
          id: "new-link",
          permission: "view",
          created_at: "2026-02-20T11:00:00.000Z",
          expires_at: "2126-02-20T11:00:00.000Z",
          revoked_at: null,
          share_path: "/knowledge/shared/new-link"
        }
      ]
      return {
        share_id: "new-link",
        permission: "view",
        created_at: "2026-02-20T11:00:00.000Z",
        expires_at: "2126-02-20T11:00:00.000Z",
        token: "token-new-link",
        share_path: "/knowledge/shared/new-link"
      }
    })
    revokeConversationShareLinkMock.mockImplementation(async () => {
      links = [
        {
          id: "new-link",
          permission: "view",
          created_at: "2026-02-20T11:00:00.000Z",
          expires_at: "2126-02-20T11:00:00.000Z",
          revoked_at: "2026-02-20T12:00:00.000Z",
          share_path: null
        }
      ]
      return {
        success: true,
        share_id: "new-link"
      }
    })

    render(<Header />)
    fireEvent.click(screen.getByRole("button", { name: "Open share modal" }))

    const ttlInput = screen.getByDisplayValue("24")
    fireEvent.change(ttlInput, { target: { value: "2" } })
    fireEvent.click(screen.getByTestId("chat-share-create-button"))

    await waitFor(() => {
      expect(createConversationShareLinkMock).toHaveBeenCalledWith(
        "server-chat-1",
        expect.objectContaining({
          permission: "view",
          ttl_seconds: 7200
        })
      )
    })

    await waitFor(() => {
      expect(screen.getByTestId("chat-share-revoke-new-link")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("chat-share-revoke-new-link"))

    await waitFor(() => {
      expect(revokeConversationShareLinkMock).toHaveBeenCalledWith(
        "server-chat-1",
        "new-link"
      )
    })
  })

  it("shows read-only role scope and opens workflow automation shortcut", async () => {
    listConversationShareLinksMock.mockResolvedValue({
      conversation_id: "server-chat-1",
      links: []
    })

    render(<Header />)
    fireEvent.click(screen.getByRole("button", { name: "Open share modal" }))

    expect(screen.getByTestId("chat-share-role-scope")).toHaveTextContent(
      "Read-only viewer"
    )
    expect(screen.getByTestId("chat-share-role-scope")).toHaveTextContent(
      "cannot send, edit, or delete"
    )

    fireEvent.click(screen.getByTestId("chat-share-open-workflows"))

    expect(navigateMock).toHaveBeenCalledWith(
      "/workflow-editor?source=chat-share&conversationId=server-chat-1"
    )
  })
})
