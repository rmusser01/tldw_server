// @vitest-environment jsdom
import React from "react"
import { render } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { Header } from "../Header"

const setHeaderShortcutsExpandedMock = vi.fn().mockResolvedValue(undefined)
const ttsClipsDrawerMock = vi.fn(() => null)
const mockT = (_key: string, fallback?: string) => fallback ?? _key

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mockT
  })
}))

vi.mock("react-router-dom", () => ({
  useLocation: () => ({ pathname: "/chat" }),
  useNavigate: () => vi.fn()
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Modal: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div>{children}</div> : null,
  Button: ({ children, ...props }: any) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
  Input: (props: any) => <input {...props} />,
  InputNumber: (props: any) => <input type="number" {...props} />
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [false, setHeaderShortcutsExpandedMock]
}))

vi.mock("@/hooks/useDarkmode", () => ({
  useDarkMode: () => ({ mode: "dark", toggleDarkMode: vi.fn() })
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn()]
}))

vi.mock("~/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    clearChat: vi.fn(),
    historyId: "temp",
    temporaryChat: true,
    setTemporaryChat: vi.fn(),
    serverChatId: null
  })
}))

vi.mock("@/db", () => ({
  getTitleById: vi.fn(),
  updateHistory: vi.fn()
}))

vi.mock("../ChatHeader", () => ({
  ChatHeader: () => <div data-testid="chat-header" />
}))

vi.mock("@/components/Sidepanel/Chat/TtsClipsDrawer", () => ({
  TtsClipsDrawer: (props: unknown) =>
    (ttsClipsDrawerMock as (props: unknown) => null)(props)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    listConversationShareLinks: vi.fn(),
    createConversationShareLink: vi.fn(),
    revokeConversationShareLink: vi.fn()
  }
}))

describe("Header tts clips mounting", () => {
  it("only mounts the TtsClipsDrawer when the drawer is open", () => {
    render(<Header />)

    expect(ttsClipsDrawerMock).not.toHaveBeenCalled()
  })
})
