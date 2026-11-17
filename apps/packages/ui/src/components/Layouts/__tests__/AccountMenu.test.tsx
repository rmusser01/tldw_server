import React from "react"
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import type { UserInfo } from "@/services/tldw/TldwAuth"

import { AccountMenu } from "../AccountMenu"
import { Header } from "../Header"

const mocks = vi.hoisted(() => ({
  profile: vi.fn(),
  logout: vi.fn(),
  navigate: vi.fn(),
  config: null as TldwConfig | null,
  authorityLoading: false
}))

const translate = (
  key: string,
  fallback?: string | { defaultValue?: string }
) => (typeof fallback === "string" ? fallback : fallback?.defaultValue || key)

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: () => mocks.profile(),
    logout: () => mocks.logout()
  }
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: mocks.config,
    loading: mocks.authorityLoading,
    authorityLoading: mocks.authorityLoading
  })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: translate
  })
}))
vi.mock("react-router-dom", () => ({
  useLocation: () => ({ pathname: "/chat" }),
  useNavigate: () => mocks.navigate,
  Link: ({
    to,
    children,
    onClick,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { to: string }) => (
    <a
      {...props}
      href={to}
      onClick={(event) => {
        event.preventDefault()
        onClick?.(event)
      }}>
      {children}
    </a>
  )
}))
vi.mock("antd", async (importOriginal) => ({
  ...await importOriginal<typeof import("antd")>(),
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Modal: () => null,
  Button: ({ children }: { children: React.ReactNode }) => (
    <button type="button">{children}</button>
  ),
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input {...props} />
  ),
  InputNumber: () => null
}))
vi.mock("wxt/browser", () => ({
  browser: {
    storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } }
  }
}))
vi.mock("@/hooks/useSetting", () => ({ useSetting: () => [false, vi.fn()] }))
vi.mock("@/hooks/useDarkmode", () => ({
  useDarkMode: () => ({ mode: "dark", toggleDarkMode: vi.fn() })
}))
vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn()]
}))
vi.mock("~/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    historyId: null,
    serverChatId: null,
    temporaryChat: false
  })
}))
vi.mock("@/hooks/useActiveChatTitle", () => ({
  useActiveChatTitle: () => ({
    title: "Chat",
    renameTitle: vi.fn(),
    owner: {},
    ready: true
  })
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))
vi.mock("@/components/Sidepanel/Chat/TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => null
}))
vi.mock("~/assets/icon.png", () => ({ default: "icon.png" }))
vi.mock("../HeaderShortcuts", () => ({ HeaderShortcuts: () => null }))

const currentUser: UserInfo = {
  id: 7,
  username: "current-user",
  email: "current-user@example.com",
  is_active: true
}
const accountConfig: TldwConfig = {
  serverUrl: "https://api.example.test",
  authMode: "multi-user",
  accessToken: "session-token"
}
const props = () => ({
  onClose: vi.fn(),
  onOpenNotifications: vi.fn(),
  notificationCount: 3,
  notificationState: "active" as const,
  onRetryNotifications: vi.fn()
})

beforeEach(() => {
  vi.clearAllMocks()
  mocks.config = accountConfig
  mocks.authorityLoading = false
  mocks.profile.mockReset().mockResolvedValue(currentUser)
  mocks.logout.mockReset().mockResolvedValue(undefined)
})

describe("AccountMenu", () => {
  it("shows server identity without a photo and uses working account and settings destinations", async () => {
    const callbacks = props()
    render(<AccountMenu {...callbacks} />)
    expect(await screen.findByText("current-user@example.com")).toBeVisible()
    expect(screen.getByText("current-user")).toBeVisible()
    expect(screen.queryByRole("img")).not.toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Account settings" })
    ).toHaveAttribute("href", "/settings/tldw")
    expect(screen.getByRole("link", { name: "Settings" })).toHaveAttribute(
      "href",
      "/settings"
    )
    fireEvent.click(screen.getByRole("link", { name: "Settings" }))
    expect(callbacks.onClose).toHaveBeenCalledOnce()
  })

  it("preserves notification navigation and retry", async () => {
    const callbacks = props()
    const user = userEvent.setup()
    render(<AccountMenu {...callbacks} notificationState="unavailable" />)
    await user.click(screen.getByRole("button", { name: "Try again" }))
    expect(callbacks.onRetryNotifications).toHaveBeenCalledOnce()
    await user.click(
      screen.getByRole("button", { name: /Notifications.*3 unread/ })
    )
    expect(callbacks.onOpenNotifications).toHaveBeenCalledOnce()
    expect(callbacks.onClose).toHaveBeenCalledOnce()
  })

  it.each([
    ["connecting", "Notifications are connecting."],
    ["degraded", "Notifications are reconnecting."],
    ["auth-required", "Sign in again to resume personal notifications."],
    ["unavailable", "Notifications unavailable for this account."]
  ] as const)(
    "keeps %s notification status visible",
    (notificationState, text) => {
      render(<AccountMenu {...props()} notificationState={notificationState} />)
      expect(screen.getByText(text)).toBeVisible()
    }
  )

  it("keeps account navigation and sign out available when identity cannot load", async () => {
    mocks.profile.mockRejectedValue(new Error("offline"))
    render(<AccountMenu {...props()} />)
    expect(
      await screen.findByText("Account details unavailable")
    ).toHaveAttribute("role", "status")
    expect(screen.getByRole("link", { name: "Account settings" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Sign out" })).toBeEnabled()
  })

  it.each(["tldw:auth-principal-changed", "storage"])(
    "clears identity and ignores late responses after %s",
    async (event) => {
      const callbacks = props()
      let resolveProfile!: (user: UserInfo) => void
      mocks.profile.mockReturnValue(
        new Promise<UserInfo>((resolve) => {
          resolveProfile = resolve
        })
      )
      render(<AccountMenu {...callbacks} />)
      act(() =>
        window.dispatchEvent(
          event === "storage"
            ? new StorageEvent("storage", { key: "tldwConfig" })
            : new Event(event)
        )
      )
      await act(async () => resolveProfile(currentUser))
      expect(callbacks.onClose).toHaveBeenCalledOnce()
      expect(
        screen.queryByText("current-user@example.com")
      ).not.toBeInTheDocument()
    }
  )

  it("uses existing sign out once and keeps it disabled while pending", async () => {
    const callbacks = props()
    let finishLogout!: () => void
    mocks.logout.mockReturnValue(
      new Promise<void>((resolve) => {
        finishLogout = resolve
      })
    )
    render(<AccountMenu {...callbacks} />)
    fireEvent.click(screen.getByRole("button", { name: "Sign out" }))
    expect(
      screen.getByRole("button", { name: "Signing out..." })
    ).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Signing out..." }))
    expect(mocks.logout).toHaveBeenCalledOnce()
    await act(async () => finishLogout())
    expect(callbacks.onClose).toHaveBeenCalledOnce()
  })

  it("shows a sign out failure without discarding account actions", async () => {
    mocks.logout.mockRejectedValue(new Error("storage failed"))
    render(<AccountMenu {...props()} />)
    fireEvent.click(screen.getByRole("button", { name: "Sign out" }))
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not sign out. Try again."
    )
    expect(screen.getByRole("button", { name: "Sign out" })).toBeEnabled()
  })

  it("does not close a newer account again when an earlier sign out finishes", async () => {
    const callbacks = props()
    let finishLogout!: () => void
    mocks.logout.mockReturnValue(
      new Promise<void>((resolve) => {
        finishLogout = resolve
      })
    )
    render(<AccountMenu {...callbacks} />)
    fireEvent.click(screen.getByRole("button", { name: "Sign out" }))
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
    await act(async () => finishLogout())
    expect(callbacks.onClose).toHaveBeenCalledOnce()
  })
})

describe("Header account integration", () => {
  it("handles the real logout principal boundary before the service promise resolves", async () => {
    mocks.logout.mockImplementation(async () => {
      mocks.config = { ...accountConfig, accessToken: undefined }
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "logout" }
        })
      )
    })
    const user = userEvent.setup()
    render(<Header {...props()} />)
    await user.click(screen.getByRole("button", { name: "Account menu" }))
    await screen.findByText("current-user@example.com")
    await user.click(screen.getByRole("button", { name: "Sign out" }))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(
      screen.queryByText("current-user@example.com")
    ).not.toBeInTheDocument()
    expect(screen.getByTestId("chat-header-notifications-bell")).toBeVisible()
    expect(mocks.logout).toHaveBeenCalledOnce()
  })

  it("opens the verified identity and actions from the account trigger", async () => {
    const user = userEvent.setup()
    render(<Header {...props()} />)
    await user.click(screen.getByRole("button", { name: "Account menu" }))
    const dialog = screen.getByRole("dialog", { name: "Account menu" })
    expect(
      await within(dialog).findByText("current-user@example.com")
    ).toBeVisible()
    expect(
      within(dialog).getByRole("link", { name: "Account settings" })
    ).toHaveFocus()
  })

  it.each([
    { label: "anonymous", config: null, loading: false },
    {
      label: "single-user",
      config: { ...accountConfig, authMode: "single-user" as const },
      loading: false
    },
    {
      label: "logged-out",
      config: { ...accountConfig, accessToken: undefined },
      loading: false
    },
    { label: "rehydrating", config: accountConfig, loading: true }
  ])(
    "keeps the notification control for $label sessions",
    ({ config, loading }) => {
      mocks.config = config
      mocks.authorityLoading = loading
      render(<Header {...props()} />)
      expect(
        screen.queryByRole("button", { name: "Account menu" })
      ).not.toBeInTheDocument()
      expect(screen.getByTestId("chat-header-notifications-bell")).toBeVisible()
      expect(mocks.profile).not.toHaveBeenCalled()
    }
  )

  it("closes and clears the prior account after an identity boundary", async () => {
    const user = userEvent.setup()
    render(<Header {...props()} />)
    const trigger = screen.getByRole("button", { name: "Account menu" })
    await user.click(trigger)
    await screen.findByText("current-user@example.com")
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    )
    expect(
      screen.queryByText("current-user@example.com")
    ).not.toBeInTheDocument()
  })

  it("ignores the prior server's delayed identity after reopening on another account", async () => {
    const pending: Array<(user: UserInfo) => void> = []
    mocks.profile.mockImplementation(
      () => new Promise<UserInfo>((resolve) => pending.push(resolve))
    )
    const user = userEvent.setup()
    const view = render(<Header {...props()} />)
    await user.click(screen.getByRole("button", { name: "Account menu" }))
    mocks.config = {
      ...accountConfig,
      serverUrl: "https://other.example.test",
      accessToken: "other-token"
    }
    act(() =>
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: true }
        })
      )
    )
    view.rerender(<Header {...props()} />)
    await user.click(screen.getByRole("button", { name: "Account menu" }))
    await act(async () =>
      pending[1]({
        ...currentUser,
        id: 8,
        username: "next-user",
        email: "next@example.com"
      })
    )
    await act(async () => pending[0](currentUser))
    expect(screen.getByText("next@example.com")).toBeVisible()
    expect(
      screen.queryByText("current-user@example.com")
    ).not.toBeInTheDocument()
  })
})
