import React from "react"
import { act, render } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getUnreadCount: vi.fn(),
  listNotifications: vi.fn(),
  subscribeNotificationsStream: vi.fn(),
  showToast: vi.fn()
}))

vi.mock("@web/lib/api/notifications", () => ({
  getUnreadCount: (...args: unknown[]) => mocks.getUnreadCount(...args),
  listNotifications: (...args: unknown[]) => mocks.listNotifications(...args),
  subscribeNotificationsStream: (...args: unknown[]) =>
    mocks.subscribeNotificationsStream(...args)
}))

vi.mock("@web/lib/api", () => ({
  getApiBaseUrl: () => "https://api.example.test/api/v1"
}))

vi.mock("@web/lib/authStorage", () => ({
  getApiBearer: () => null,
  getApiKey: () => "test-api-key"
}))

vi.mock("@web/components/ui/ToastProvider", () => ({
  useToast: () => ({ show: mocks.showToast })
}))

import { NotificationLifecycleProvider } from "@web/components/notifications/NotificationLifecycleProvider"
import { NotificationToastBridge } from "@web/components/notifications/NotificationToastBridge"

describe("NotificationToastBridge", () => {
  beforeEach(() => {
    vi.useRealTimers()
    vi.clearAllMocks()
    mocks.getUnreadCount.mockResolvedValue({ unread_count: 1 })
    mocks.listNotifications.mockResolvedValue({
      items: [{ id: 50, title: "Existing" }],
      total: 1
    })
    mocks.subscribeNotificationsStream.mockImplementation(() => vi.fn())
  })

  it("does no work when rendered outside the lifecycle provider", async () => {
    render(<NotificationToastBridge />)
    await act(async () => Promise.resolve())

    expect(mocks.getUnreadCount).not.toHaveBeenCalled()
    expect(mocks.listNotifications).not.toHaveBeenCalled()
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled()
    expect(mocks.showToast).not.toHaveBeenCalled()
  })

  it("toasts provider-owned events without creating another bootstrap or stream", async () => {
    vi.useFakeTimers()
    let onEvent: ((event: { event: string; id?: number; payload?: unknown }) => void) | undefined
    mocks.subscribeNotificationsStream.mockImplementation(
      (options: { onEvent: typeof onEvent }) => {
        onEvent = options.onEvent
        return vi.fn()
      }
    )

    render(
      <NotificationLifecycleProvider scopeKey="notifications:server-a:user-a">
        <NotificationToastBridge />
      </NotificationLifecycleProvider>
    )
    await act(async () => vi.advanceTimersByTimeAsync(0))

    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(1)
    expect(mocks.listNotifications).toHaveBeenCalledTimes(1)
    expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1)
    expect(mocks.subscribeNotificationsStream).toHaveBeenCalledWith(
      expect.objectContaining({ after: 50 })
    )

    act(() =>
      onEvent?.({
        event: "notification",
        id: 51,
        payload: {
          notification_id: 51,
          kind: "deep_research_completed",
          title: "Deep research completed",
          message: "Open the report in Deep Research.",
          severity: "info",
          created_at: "2026-03-08T01:00:00Z"
        }
      })
    )
    await act(async () => vi.advanceTimersByTimeAsync(800))

    expect(mocks.showToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Deep research completed",
        description: "Open the report in Deep Research.",
        variant: "info"
      })
    )
    expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1)
    vi.useRealTimers()
  })

  it("coalesces a provider-owned burst into one summary toast", async () => {
    vi.useFakeTimers()
    let onEvent: ((event: { event: string; id?: number; payload?: unknown }) => void) | undefined
    mocks.subscribeNotificationsStream.mockImplementation(
      (options: { onEvent: typeof onEvent }) => {
        onEvent = options.onEvent
        return vi.fn()
      }
    )
    render(
      <NotificationLifecycleProvider scopeKey="notifications:server-a:user-a">
        <NotificationToastBridge />
      </NotificationLifecycleProvider>
    )
    await act(async () => vi.advanceTimersByTimeAsync(0))

    act(() =>
      onEvent?.({
        event: "notifications_coalesced",
        id: 52,
        payload: { count: 3 }
      })
    )
    await act(async () => vi.advanceTimersByTimeAsync(800))

    expect(mocks.showToast).toHaveBeenCalledTimes(1)
    expect(mocks.showToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "3 new notifications",
        description: "Your inbox has been updated."
      })
    )
    vi.useRealTimers()
  })
})
