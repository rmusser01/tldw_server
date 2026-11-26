import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  push: vi.fn(),
  pathname: "/chat",
  listNotifications: vi.fn(),
  getUnreadCount: vi.fn(),
  markNotificationsRead: vi.fn(),
  dismissNotification: vi.fn(),
  subscribeNotificationsStream: vi.fn(),
  listAuthSessions: vi.fn(),
  override: null as any
}))

vi.mock("next/router", () => ({
  useRouter: () => ({ push: mocks.push, pathname: mocks.pathname })
}))

vi.mock("@web/lib/api/notifications", () => ({
  getUnreadCount: (...a: unknown[]) => mocks.getUnreadCount(...a),
  listNotifications: (...a: unknown[]) => mocks.listNotifications(...a),
  markNotificationsRead: (...a: unknown[]) => mocks.markNotificationsRead(...a),
  dismissNotification: (...a: unknown[]) => mocks.dismissNotification(...a),
  subscribeNotificationsStream: (...a: unknown[]) => mocks.subscribeNotificationsStream(...a)
}))

vi.mock("@/services/auth-sessions", async () => {
  const actual = await vi.importActual<any>("@/services/auth-sessions")
  return { ...actual, listAuthSessions: (...a: unknown[]) => mocks.listAuthSessions(...a) }
})

vi.mock("@web/lib/api", () => ({ getApiBaseUrl: () => "https://api.example.test/api/v1" }))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({ phase: "connected", isConnected: true, mode: "normal", offlineBypass: false })
}))
vi.mock("@web/lib/authStorage", () => ({
  getApiBearer: () => null,
  getApiKey: () => "test-api-key",
  getSessionAccessToken: () => null,
  getEffectiveStoredTldwConfig: () => null
}))

vi.mock("@web/components/notifications/NotificationLifecycleProvider", async () => {
  const actual = await vi.importActual<any>("@web/components/notifications/NotificationLifecycleProvider")
  return {
    ...actual,
    useOptionalNotificationLifecycle: () => {
      const real = actual.useOptionalNotificationLifecycle()
      return mocks.override ?? real
    }
  }
})

import {
  NotificationLifecycleProvider,
  useOptionalNotificationLifecycle
} from "@web/components/notifications/NotificationLifecycleProvider"
import { NewSignInBanner } from "@web/components/notifications/NewSignInBanner"

const alertItem = (id: number, over: Record<string, unknown> = {}) => ({
  id,
  kind: "new_sign_in",
  title: "New sign-in to your account",
  message: `Chrome on macOS signed in from 203.0.113.${id}. If this was not you, sign out your other sessions.`,
  severity: "warning",
  source_job_id: id * 10,
  created_at: `2026-11-25T14:0${id}:00Z`,
  read_at: null,
  dismissed_at: null,
  ...over
})

const session = (id: number, over: Record<string, unknown> = {}) => ({
  id,
  ip_address: "203.0.113.1",
  user_agent: "UA",
  device_label: "Chrome on macOS",
  created_at: "",
  last_activity: "",
  expires_at: "",
  is_current: false,
  same_install: false,
  ...over
})

const lifecycle = (over: Record<string, unknown> = {}) => ({
  scopeKey: "s",
  lifecycleEpoch: 1,
  state: "active",
  connectionVerified: true,
  events: [] as any[],
  ...over
})

const setItems = (...items: any[]) =>
  mocks.listNotifications.mockResolvedValue({ items, total: items.length })

const flush = () => act(async () => { await Promise.resolve(); await Promise.resolve() })

const renderSettled = async () => {
  const utils = await renderBanner()
  await waitFor(() => expect(mocks.listAuthSessions).toHaveBeenCalled())
  await flush()
  return utils
}

const renderBanner = async () => {
  const utils = render(<NewSignInBanner />)
  await flush()
  return utils
}

describe("NewSignInBanner", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.override = lifecycle()
    mocks.pathname = "/chat"
    window.localStorage.clear()
    setItems(alertItem(1))
    mocks.listAuthSessions.mockResolvedValue([
      session(5, { is_current: true }),
      session(10)
    ])
    mocks.dismissNotification.mockResolvedValue({ dismissed: true })
    mocks.getUnreadCount.mockResolvedValue({ unread_count: 1 })
    mocks.subscribeNotificationsStream.mockImplementation(() => vi.fn())
  })

  it("shows a role=alert banner for an unread new_sign_in from another session", async () => {
    await renderBanner()
    const banner = await screen.findByRole("alert")
    expect(banner).toHaveTextContent("New sign-in to your account")
    expect(banner).toHaveTextContent("203.0.113.1")
  })

  it("asks the server for kind=new_sign_in only", async () => {
    await renderBanner()
    expect(mocks.listNotifications).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "new_sign_in", limit: 100, offset: 0, include_archived: false })
    )
  })

  it("does not show it when the source session is the current session", async () => {
    mocks.listAuthSessions.mockResolvedValue([session(10, { is_current: true })])
    await renderSettled()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("does not show it when the source session has same_install true", async () => {
    mocks.listAuthSessions.mockResolvedValue([session(5, { is_current: true }), session(10, { same_install: true })])
    await renderSettled()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("shows it when same_install is false even though the source and current sessions share IP and device label", async () => {
    mocks.listAuthSessions.mockResolvedValue([session(5, { is_current: true }), session(10, { same_install: false })])
    await renderBanner()
    expect(await screen.findByRole("alert")).toBeInTheDocument()
  })

  it("does not show read, dismissed or other-kind notifications", async () => {
    setItems(
      alertItem(1, { read_at: "2026-11-25T15:00:00Z" }),
      alertItem(2, { dismissed_at: "2026-11-25T15:00:00Z" }),
      alertItem(3, { kind: "job_failed" })
    )
    await renderSettled()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("a streamed new_sign_in notification triggers a reload and shows the banner", async () => {
    setItems()
    const { rerender } = await renderBanner()
    expect(screen.queryByRole("alert")).toBeNull()
    setItems(alertItem(1))
    mocks.override = lifecycle({
      events: [{ sequence: 1, event: { event: "notification", payload: { kind: "new_sign_in", notification_id: 1 } } }]
    })
    rerender(<NewSignInBanner />)
    expect(await screen.findByRole("alert")).toBeInTheDocument()
  })

  it("a notifications_coalesced event reloads the list and shows an alert that was not in the stream", async () => {
    setItems()
    const { rerender } = await renderBanner()
    setItems(alertItem(1))
    mocks.override = lifecycle({
      events: [{ sequence: 1, event: { event: "notifications_coalesced", payload: { count: 60 } } }]
    })
    rerender(<NewSignInBanner />)
    expect(await screen.findByRole("alert")).toBeInTheDocument()
  })

  it("a reset_required event reloads the list", async () => {
    const { rerender } = await renderBanner()
    const before = mocks.listNotifications.mock.calls.length
    mocks.override = lifecycle({ events: [{ sequence: 1, event: { event: "reset_required" } }] })
    rerender(<NewSignInBanner />)
    await waitFor(() => expect(mocks.listNotifications.mock.calls.length).toBeGreaterThan(before))
  })

  it("a lifecycle epoch change reloads the list", async () => {
    const { rerender } = await renderBanner()
    const before = mocks.listNotifications.mock.calls.length
    mocks.override = lifecycle({ lifecycleEpoch: 2 })
    rerender(<NewSignInBanner />)
    await waitFor(() => expect(mocks.listNotifications.mock.calls.length).toBeGreaterThan(before))
  })

  it("still shows the alert when the session list cannot be loaded", async () => {
    mocks.listAuthSessions.mockRejectedValue(new Error("api key"))
    await renderBanner()
    expect(await screen.findByRole("alert")).toBeInTheDocument()
  })

  it("Review sessions navigates to /settings/sessions and does not mark read", async () => {
    await renderBanner()
    fireEvent.click(await screen.findByRole("button", { name: "Review sessions" }))
    expect(mocks.push).toHaveBeenCalledWith("/settings/sessions")
    expect(mocks.markNotificationsRead).not.toHaveBeenCalled()
  })

  it("Dismiss calls dismissNotification and removes the banner", async () => {
    await renderBanner()
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }))
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
    expect(mocks.dismissNotification).toHaveBeenCalledWith(1)
  })

  const refuse = () =>
    mocks.dismissNotification.mockRejectedValue({ status: 403, errorCode: "new_sign_in_protected" })

  it("Dismiss answered with new_sign_in_protected keeps the banner, shows the explanation and offers Hide on this browser", async () => {
    refuse()
    await renderBanner()
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }))
    const banner = await screen.findByText(
      "This alert can be dismissed from a browser that was signed in before it, or after 24 hours."
    )
    expect(banner).toBeInTheDocument()
    expect(screen.getByRole("alert")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Hide on this browser" })).toBeInTheDocument()
  })

  it("shows a retry message for any other dismiss failure", async () => {
    mocks.dismissNotification.mockRejectedValue(new Error("boom"))
    await renderBanner()
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }))
    expect(await screen.findByText("Could not dismiss the alert. Try again.")).toBeInTheDocument()
  })

  it("Hide on this browser removes the banner, writes the localStorage key and makes no server call; the banner stays hidden after a remount", async () => {
    refuse()
    const { unmount } = await renderBanner()
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }))
    mocks.dismissNotification.mockClear()
    fireEvent.click(await screen.findByRole("button", { name: "Hide on this browser" }))
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
    expect(window.localStorage.getItem("tldw:new-sign-in-hidden:1")).toBe("1")
    expect(mocks.dismissNotification).not.toHaveBeenCalled()
    expect(mocks.markNotificationsRead).not.toHaveBeenCalled()
    unmount()
    await renderBanner()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("Hide on this browser still hides for the page's lifetime when localStorage throws", async () => {
    refuse()
    await renderBanner()
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }))
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("denied") })
    const getItem = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("denied") })
    try {
      fireEvent.click(await screen.findByRole("button", { name: "Hide on this browser" }))
      await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
    } finally {
      setItem.mockRestore()
      getItem.mockRestore()
    }
  })

  it("lifecycle state stays active after a refused dismiss", async () => {
    mocks.override = null
    mocks.subscribeNotificationsStream.mockImplementation((o: { onOpen?: () => void }) => {
      o.onOpen?.()
      return vi.fn()
    })
    refuse()
    let seen: any = null
    const Probe = () => {
      seen = useOptionalNotificationLifecycle()
      return null
    }
    render(
      <NotificationLifecycleProvider scopeKey="notifications:server-a:user-a">
        <Probe />
        <NewSignInBanner />
      </NotificationLifecycleProvider>
    )
    await flush()
    fireEvent.click(await screen.findByRole("button", { name: "Dismiss" }))
    await screen.findByText(/can be dismissed from a browser/)
    expect(seen.state).toBe("active")
    expect(seen.mutationError).toBeNull()
    expect(screen.getByRole("alert")).toBeInTheDocument()
  })

  it("is not rendered on /settings/sessions", async () => {
    mocks.pathname = "/settings/sessions"
    await renderBanner()
    await waitFor(() => expect(mocks.listNotifications).toHaveBeenCalled())
    await flush()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it.each([
    ["connecting", { state: "connecting" }],
    ["auth-required", { state: "auth-required" }],
    ["unverified", { connectionVerified: false }]
  ])("is not rendered, and fetches nothing, when the lifecycle is %s", async (_n, over) => {
    mocks.override = lifecycle(over)
    await renderBanner()
    expect(screen.queryByRole("alert")).toBeNull()
    expect(mocks.listNotifications).not.toHaveBeenCalled()
    expect(mocks.listAuthSessions).not.toHaveBeenCalled()
  })

  it("is not rendered, and fetches nothing, outside a lifecycle provider", async () => {
    mocks.override = null
    await renderBanner()
    expect(screen.queryByRole("alert")).toBeNull()
    expect(mocks.listNotifications).not.toHaveBeenCalled()
  })

  it("puts the opaque surface on the element wrapping the alert (class only; pixels are not checked)", async () => {
    await renderBanner()
    const banner = await screen.findByRole("alert")
    expect(banner.parentElement).toHaveClass("bg-surface")
    expect(banner).not.toHaveClass("bg-surface")
  })

  it("reloads when the route changes, so an alert handled in the inbox disappears", async () => {
    const { rerender } = await renderBanner()
    expect(await screen.findByRole("alert")).toBeInTheDocument()
    setItems(alertItem(1, { read_at: "2026-11-25T15:00:00Z" }))
    mocks.pathname = "/notifications"
    rerender(<NewSignInBanner />)
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
  })

  it("two unread alerts show the newest and \"1 more new sign-in.\"", async () => {
    setItems(alertItem(1), alertItem(2))
    await renderBanner()
    const banner = await screen.findByRole("alert")
    expect(banner).toHaveTextContent("203.0.113.2")
    expect(banner).not.toHaveTextContent("203.0.113.1.")
    expect(banner).toHaveTextContent("1 more new sign-in.")
  })
})
