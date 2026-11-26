import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { Modal } from "antd"
import { createInstance } from "i18next"
import { I18nextProvider } from "react-i18next"
import { beforeEach, describe, expect, it, vi } from "vitest"

import commonEn from "@/assets/locale/en/common.json"
import settingsEn from "@/assets/locale/en/settings.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import {
  listAuthSessions,
  revokeAuthSession,
  revokeOtherAuthSessions,
  type AuthSession
} from "@/services/auth-sessions"
import { ActiveSessionsSettings } from "../ActiveSessionsSettings"

const mocks = vi.hoisted(() => ({ authMode: "multi-user" }))

vi.mock("@/services/auth-sessions", () => ({
  listAuthSessions: vi.fn(),
  revokeAuthSession: vi.fn(),
  revokeOtherAuthSessions: vi.fn(),
  isInteractiveRequiredError: (e: unknown) =>
    (e as { errorCode?: string })?.errorCode === "interactive_session_required"
}))
vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: { authMode: mocks.authMode },
    loading: false,
    authorityLoading: false
  })
}))

const session = (over: Partial<AuthSession>): AuthSession => ({
  id: 1,
  ip_address: "203.0.113.7",
  user_agent: null,
  device_label: "Chrome on macOS",
  created_at: "2026-11-25T14:03:00Z",
  last_activity: "2026-11-25T14:41:00Z",
  expires_at: "2026-12-25T14:03:00Z",
  is_current: false,
  same_install: false,
  ...over
})
const current = session({ id: 1, is_current: true })
const other = session({
  id: 2,
  device_label: "Safari on iOS",
  ip_address: "198.51.100.9",
  created_at: "2026-11-24T09:12:00Z",
  last_activity: "2026-11-24T18:30:00Z"
})

const renderPage = async () => {
  const i18n = createInstance()
  await i18n.use(ICUWithInterpolation).init({
    lng: "en",
    fallbackLng: false,
    ns: ["settings", "common"],
    defaultNS: "settings",
    resources: { en: { common: commonEn, settings: settingsEn } },
    interpolation: { escapeValue: false }
  })
  return render(
    <I18nextProvider i18n={i18n}>
      <ActiveSessionsSettings />
    </I18nextProvider>
  )
}

const list = vi.mocked(listAuthSessions)

beforeEach(() => {
  vi.clearAllMocks()
  mocks.authMode = "multi-user"
  list.mockResolvedValue([current, other])
  vi.mocked(revokeAuthSession).mockResolvedValue(undefined)
  vi.mocked(revokeOtherAuthSessions).mockResolvedValue({ sessions_revoked: 1 })
})

describe("ActiveSessionsSettings", () => {
  it("renders the loading state, then one row per session with This device on the current row only", async () => {
    await renderPage()
    expect(screen.getByText("Loading your sessions...")).toBeInTheDocument()
    expect(await screen.findByText("Safari on iOS")).toBeInTheDocument()
    expect(screen.getByText("Chrome on macOS")).toBeInTheDocument()
    expect(screen.getAllByText("This device")).toHaveLength(1)
    const row = screen.getByText("Chrome on macOS").closest("tr") as HTMLElement
    expect(within(row).getByText("This device")).toBeInTheDocument()
  })

  it("the current row has no Sign out button; other rows have one labelled with device and time", async () => {
    await renderPage()
    await screen.findByText("Safari on iOS")
    const buttons = screen.getAllByRole("button", { name: /^Sign out Safari on iOS, signed in / })
    expect(buttons).toHaveLength(1)
    expect(screen.queryByRole("button", { name: /^Sign out Chrome on macOS/ })).toBeNull()
  })

  it('Sign out calls revokeAuthSession with that id, reloads, announces "Signed out Safari on iOS." in the status region and moves focus to the heading', async () => {
    const user = userEvent.setup()
    await renderPage()
    await screen.findByText("Safari on iOS")
    list.mockResolvedValue([current])
    await user.click(screen.getByRole("button", { name: /^Sign out Safari on iOS/ }))
    expect(revokeAuthSession).toHaveBeenCalledWith(2)
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent(
        "Signed out Safari on iOS. 0 other sessions remain."
      )
    )
    expect(list).toHaveBeenCalledTimes(2)
    expect(screen.getByRole("heading", { level: 1, name: "Active sessions" })).toHaveFocus()
  })

  it("Sign out all other sessions asks for confirmation with the count; Cancel calls nothing", async () => {
    const user = userEvent.setup()
    await renderPage()
    await screen.findByText("Safari on iOS")
    await user.click(screen.getByRole("button", { name: "Sign out all other sessions" }))
    const dialog = await screen.findByRole("dialog", { name: "Sign out all other sessions?" })
    expect(screen.getByText(/1 other session will be signed out\. This device stays signed in\./)).toBeInTheDocument()
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    expect(revokeOtherAuthSessions).not.toHaveBeenCalled()
  })

  it("confirming calls revokeOtherAuthSessions, reloads and announces the result", async () => {
    const user = userEvent.setup()
    const confirm = vi.spyOn(Modal, "confirm")
    await renderPage()
    await screen.findByText("Safari on iOS")
    await user.click(screen.getByRole("button", { name: "Sign out all other sessions" }))
    list.mockResolvedValue([current])
    await user.click(await screen.findByRole("button", { name: "Sign out other sessions" }))
    await waitFor(() => expect(revokeOtherAuthSessions).toHaveBeenCalledTimes(1))
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("Signed out 1 other session.")
    )
    // jsdom never finishes antd's close animation, so afterClose is invoked by hand: this pins
    // that the heading takes focus only once the modal has closed (after its focus restore).
    const heading = screen.getByRole("heading", { level: 1, name: "Active sessions" })
    expect(heading).not.toHaveFocus()
    confirm.mock.calls[0][0].afterClose?.()
    expect(heading).toHaveFocus()
  })

  it("Sign out all other sessions is disabled when only this device is listed", async () => {
    list.mockResolvedValue([current])
    await renderPage()
    await screen.findByText("Chrome on macOS")
    expect(screen.getByRole("button", { name: "Sign out all other sessions" })).toBeDisabled()
  })

  it("shows the read-only notice and disables sign-out when no row is current", async () => {
    list.mockResolvedValue([other])
    await renderPage()
    await screen.findByText("Safari on iOS")
    expect(screen.getByText("Sessions are read-only on this connection")).toBeInTheDocument()
    expect(screen.getByText(/This connection uses an API key\. Sign in with the browser to sign sessions out\./)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /^Sign out Safari on iOS/ })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Sign out all other sessions" })).toBeDisabled()
  })

  it("a failed reload after a successful sign-out shows the load error and no remaining-count claim", async () => {
    const user = userEvent.setup()
    await renderPage()
    await screen.findByText("Safari on iOS")
    list.mockRejectedValue(new Error("down"))
    await user.click(screen.getByRole("button", { name: /^Sign out Safari on iOS/ }))
    expect(await screen.findByText("Could not load your sessions")).toBeInTheDocument()
    expect(screen.getByRole("status")).toHaveTextContent(/^Signed out Safari on iOS\.$/)
  })

  it("shows the browser-sign-in warning when the server answers interactive_session_required", async () => {
    const user = userEvent.setup()
    vi.mocked(revokeAuthSession).mockRejectedValue({ errorCode: "interactive_session_required" })
    await renderPage()
    await screen.findByText("Safari on iOS")
    await user.click(screen.getByRole("button", { name: /^Sign out Safari on iOS/ }))
    expect(await screen.findByText("Sign in from a browser to sign sessions out.")).toBeInTheDocument()
  })

  it("shows a generic error when sign-out fails otherwise", async () => {
    const user = userEvent.setup()
    vi.mocked(revokeAuthSession).mockRejectedValue(new Error("boom"))
    await renderPage()
    await screen.findByText("Safari on iOS")
    await user.click(screen.getByRole("button", { name: /^Sign out Safari on iOS/ }))
    expect(await screen.findByText("Could not sign out that session. Try again.")).toBeInTheDocument()
  })

  it("shows the error alert with Try again, and Try again reloads", async () => {
    const user = userEvent.setup()
    list.mockRejectedValueOnce(new Error("down"))
    await renderPage()
    expect(await screen.findByText("Could not load your sessions")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Try again" }))
    expect(await screen.findByText("Safari on iOS")).toBeInTheDocument()
    expect(list).toHaveBeenCalledTimes(2)
  })

  it("shows the empty state for an empty list", async () => {
    list.mockResolvedValue([])
    await renderPage()
    expect(await screen.findByText("No sessions to show")).toBeInTheDocument()
    expect(screen.getByText("Sessions appear here for browsers signed in to this account.")).toBeInTheDocument()
  })

  it('shows "If you did not sign in, rotate your API key." in single-user mode and not in multi-user mode', async () => {
    mocks.authMode = "single-user"
    const view = await renderPage()
    await screen.findByText("Safari on iOS")
    expect(screen.getByText("If you did not sign in, rotate your API key.")).toBeInTheDocument()
    view.unmount()
    mocks.authMode = "multi-user"
    await renderPage()
    await screen.findByText("Safari on iOS")
    expect(screen.queryByText("If you did not sign in, rotate your API key.")).toBeNull()
  })
})
