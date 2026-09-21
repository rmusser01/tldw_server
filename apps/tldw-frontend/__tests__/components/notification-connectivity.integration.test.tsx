import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  connection: { phase: "error", isConnected: false, mode: "normal", offlineBypass: false, errorKind: "unreachable" },
  config: { serverUrl: "https://notifications.test", authMode: "single-user" },
  bearer: null as string | null,
  getUnreadCount: vi.fn(), listNotifications: vi.fn(), subscribeNotificationsStream: vi.fn(),
  getNotificationPreferences: vi.fn(), updateNotificationPreferences: vi.fn(),
  markNotificationsRead: vi.fn(), dismissNotification: vi.fn(), snoozeNotification: vi.fn(),
  cancelNotificationSnooze: vi.fn(), push: vi.fn(), show: vi.fn()
}))
vi.mock("@/hooks/useConnectionState", () => ({ useConnectionState: () => mocks.connection }))
vi.mock("@web/lib/api", () => ({ getApiBaseUrl: () => "https://notifications.test/api/v1" }))
vi.mock("@web/lib/authStorage", () => ({
  getApiBearer: () => mocks.bearer, getSessionAccessToken: () => mocks.bearer, getApiKey: () => "synthetic-key",
  getEffectiveStoredTldwConfig: () => mocks.config
}))
vi.mock("next/router", () => ({ useRouter: () => ({ push: mocks.push }) }))
vi.mock("@web/components/ui/ToastProvider", () => ({ useToast: () => ({ show: mocks.show }) }))
vi.mock("@web/lib/api/notifications", () => ({
  getUnreadCount: (...args: unknown[]) => mocks.getUnreadCount(...args),
  listNotifications: (...args: unknown[]) => mocks.listNotifications(...args),
  subscribeNotificationsStream: (...args: unknown[]) => mocks.subscribeNotificationsStream(...args),
  getNotificationPreferences: (...args: unknown[]) => mocks.getNotificationPreferences(...args),
  updateNotificationPreferences: (...args: unknown[]) => mocks.updateNotificationPreferences(...args),
  markNotificationsRead: (...args: unknown[]) => mocks.markNotificationsRead(...args),
  dismissNotification: (...args: unknown[]) => mocks.dismissNotification(...args),
  snoozeNotification: (...args: unknown[]) => mocks.snoozeNotification(...args),
  cancelNotificationSnooze: (...args: unknown[]) => mocks.cancelNotificationSnooze(...args)
}))

import { NotificationLifecycleProvider, useNotificationLifecycle } from "@web/components/notifications/NotificationLifecycleProvider"
import NotificationsRoute from "@web/components/notifications/NotificationsRoute"

function StateProbe() {
  const value = useNotificationLifecycle()
  return <output data-testid="connection-state">{value.state}:{String(value.connectionVerified)}</output>
}
const tree = (scopeKey?: string) => <NotificationLifecycleProvider scopeKey={scopeKey}><StateProbe /><NotificationsRoute /></NotificationLifecycleProvider>

describe("actual notification provider and inbox connection boundary", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.connection = { phase: "error", isConnected: false, mode: "normal", offlineBypass: false, errorKind: "unreachable" }
    mocks.config = { serverUrl: "https://notifications.test", authMode: "single-user" }
    mocks.bearer = null
    mocks.getUnreadCount.mockResolvedValue({ unread_count: 1 })
    mocks.listNotifications.mockResolvedValue({ items: [], total: 0 })
    mocks.subscribeNotificationsStream.mockImplementation(({ onOpen }) => { onOpen(); return vi.fn() })
  })
  afterEach(() => vi.useRealTimers())

  it("shows reconnecting on the first network-unverified render without dispatching private inbox work", async () => {
    vi.useFakeTimers()
    render(tree())
    expect(screen.getByTestId("connection-state")).toHaveTextContent("degraded:false")
    expect(screen.getByText("Notifications are reconnecting")).toBeInTheDocument()
    expect(screen.queryByText("Sign in again to view notifications")).not.toBeInTheDocument()
    expect(screen.queryByText(/Last updated before the connection was lost/)).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Preferences" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Try again" }))
    await act(async () => { await vi.advanceTimersByTimeAsync(60_000) })
    for (const call of [mocks.getUnreadCount, mocks.listNotifications, mocks.subscribeNotificationsStream,
      mocks.getNotificationPreferences, mocks.updateNotificationPreferences, mocks.markNotificationsRead,
      mocks.dismissNotification, mocks.snoozeNotification, mocks.cancelNotificationSnooze]) {
      expect(call).not.toHaveBeenCalled()
    }
  })

  it("resumes the real inbox and stream when the same configured key becomes verified", async () => {
    const view = render(tree())
    expect(mocks.listNotifications).not.toHaveBeenCalled()
    mocks.connection = { ...mocks.connection, phase: "connected", isConnected: true, errorKind: "none" }
    view.rerender(tree())
    await waitFor(() => expect(screen.getByTestId("connection-state")).toHaveTextContent("active:true"))
    expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1)
    expect(mocks.listNotifications).toHaveBeenCalledTimes(3)
    expect(screen.getByRole("button", { name: "Refresh" })).toBeEnabled()
  })

  it("updates network to genuine auth failure while verification stays false", async () => {
    const view = render(tree())
    expect(screen.getByTestId("connection-state")).toHaveTextContent("degraded:false")
    mocks.connection = { ...mocks.connection, errorKind: "auth" }
    view.rerender(tree())
    expect(screen.getByTestId("connection-state")).toHaveTextContent("auth-required:false")
    expect(screen.getByText("Sign in again to view notifications")).toBeInTheDocument()
    expect(mocks.listNotifications).not.toHaveBeenCalled()
  })

  it("gates the first child effect when configured multi-user credentials are missing", async () => {
    mocks.config = { ...mocks.config, authMode: "multi-user" }
    mocks.connection = { ...mocks.connection, phase: "connected", isConnected: true, errorKind: "none" }
    render(tree())
    await act(async () => { await Promise.resolve() })
    expect(screen.getByText("Sign in again to view notifications")).toBeInTheDocument()
    expect(mocks.listNotifications).not.toHaveBeenCalled()
    expect(mocks.getUnreadCount).not.toHaveBeenCalled()
  })

  it("does not claim a last successful inbox update when verified-core notification reads all fail", async () => {
    mocks.connection = { ...mocks.connection, phase: "connected", isConnected: true, errorKind: "none" }
    mocks.getUnreadCount.mockRejectedValue(new TypeError("Failed to fetch"))
    mocks.listNotifications.mockRejectedValue(new TypeError("Failed to fetch"))
    render(tree())
    await screen.findByText("Notifications are reconnecting")
    expect(screen.queryByText(/Last updated before the connection was lost/)).not.toBeInTheDocument()
  })

  it("keeps the successful inbox time through failures and clears it when ownership changes", async () => {
    vi.useFakeTimers({ toFake: ["Date"] })
    vi.setSystemTime(new Date("2026-09-16T00:00:00Z"))
    mocks.connection = { ...mocks.connection, phase: "connected", isConnected: true, errorKind: "none" }
    mocks.listNotifications.mockResolvedValue({ items: [{
      id: 71, title: "Retained source", message: "Saved notification", created_at: "2026-09-16T00:00:00Z",
      read_at: null, dismissed_at: null
    }], total: 1 })
    const view = render(tree("owner-a"))
    await screen.findByText("Retained source")
    await waitFor(() => expect(screen.getByTestId("connection-state")).toHaveTextContent("active:true"))
    vi.setSystemTime(new Date("2026-09-16T00:05:00Z"))
    mocks.listNotifications.mockRejectedValue(new TypeError("Failed to fetch"))
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }))
    await screen.findByText("Notifications are reconnecting")
    expect(screen.getByText(/Last updated before the connection was lost/)).toHaveTextContent("5 minutes ago")
    expect(screen.getByText("Retained source")).toBeInTheDocument()

    mocks.getUnreadCount.mockRejectedValue(new TypeError("Failed to fetch"))
    view.rerender(tree("owner-b"))
    await screen.findByText("Notifications are reconnecting")
    expect(screen.queryByText(/Last updated before the connection was lost/)).not.toBeInTheDocument()
    expect(screen.queryByText("Retained source")).not.toBeInTheDocument()
  })

  it.each(["none", "outage", "owner", "unmount", "outage-recovered", "owner-returned"])(
    "checks View authority after marking read: %s",
    async (transition) => {
      mocks.connection = { ...mocks.connection, phase: "connected", isConnected: true, errorKind: "none" }
      mocks.listNotifications.mockResolvedValue({ items: [{
        id: 71, title: "Private source", message: "Open saved source", created_at: "2026-09-16T00:00:00Z",
        read_at: null, dismissed_at: null, link_url: "/media/71", link_type: "media"
      }], total: 1 })
      let finish!: (value: { updated: number }) => void
      mocks.markNotificationsRead.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
      const view = render(tree("owner-a"))
      fireEvent.click(await screen.findByRole("button", { name: "View" }))
      expect(mocks.markNotificationsRead).toHaveBeenCalledTimes(1)

      if (transition.startsWith("outage")) {
        mocks.connection = { ...mocks.connection, phase: "error", isConnected: false, errorKind: "unreachable" }
        view.rerender(tree("owner-a"))
        expect(screen.getByRole("button", { name: "View" })).toBeDisabled()
        if (transition === "outage-recovered") {
          mocks.connection = { ...mocks.connection, phase: "connected", isConnected: true, errorKind: "none" }
          view.rerender(tree("owner-a"))
        }
      } else if (transition.startsWith("owner")) {
        view.rerender(tree("owner-b"))
        if (transition === "owner-returned") view.rerender(tree("owner-a"))
      } else if (transition === "unmount") {
        view.unmount()
      }

      await act(async () => finish({ updated: 1 }))
      if (transition === "none") expect(mocks.push).toHaveBeenCalledWith("/media/71")
      else expect(mocks.push).not.toHaveBeenCalled()
      expect(mocks.markNotificationsRead).toHaveBeenCalledTimes(1)
    }
  )

  it.each(["same-event", "same-owner-refresh", "server-aba", "principal-aba", "removed-restored"])(
    "checks pending View against synchronously observed authority: %s",
    async (transition) => {
      const token = (sub: string, revision: number) => `header.${btoa(JSON.stringify({ sub, revision }))}.signature`
      mocks.config = { ...mocks.config, authMode: "multi-user" }
      mocks.bearer = token("alice", 1)
      mocks.connection = { ...mocks.connection, phase: "connected", isConnected: true, errorKind: "none" }
      mocks.listNotifications.mockResolvedValue({ items: [{
        id: 71, title: "Private source", message: "Open saved source", created_at: "2026-09-16T00:00:00Z",
        read_at: null, dismissed_at: null, link_url: "/media/71", link_type: "media"
      }], total: 1 })
      let finish!: (value: { updated: number }) => void
      mocks.markNotificationsRead.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
      render(tree())
      fireEvent.click(await screen.findByRole("button", { name: "View" }))
      act(() => {
        if (transition === "server-aba") {
          mocks.config = { ...mocks.config, serverUrl: "https://other-notifications.test" }
          window.dispatchEvent(new Event("tldw:config-updated"))
          mocks.config = { ...mocks.config, serverUrl: "https://notifications.test" }
        } else if (transition === "principal-aba") {
          mocks.bearer = token("bob", 1)
          window.dispatchEvent(new Event("tldw:config-updated"))
          mocks.bearer = token("alice", 1)
        } else if (transition === "same-owner-refresh") {
          mocks.bearer = token("alice", 2)
        } else if (transition === "removed-restored") {
          mocks.bearer = null
          window.dispatchEvent(new CustomEvent("tldw:auth-credentials-changed", { detail: { authenticated: false } }))
          mocks.bearer = token("alice", 1)
        }
        window.dispatchEvent(new Event("tldw:config-updated"))
      })
      await act(async () => finish({ updated: 1 }))
      if (transition === "same-event" || transition === "same-owner-refresh") {
        expect(mocks.push).toHaveBeenCalledWith("/media/71")
      } else {
        expect(mocks.push).not.toHaveBeenCalled()
      }
      expect(mocks.markNotificationsRead).toHaveBeenCalledTimes(1)
    }
  )
})
