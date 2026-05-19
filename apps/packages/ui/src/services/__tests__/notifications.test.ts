import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
}))

import {
  buildNotificationsQuery,
  cancelNotificationSnooze,
  createNotificationStreamSubscription,
  dismissNotification,
  getNotificationPreferences,
  getUnreadCount,
  listNotifications,
  markNotificationsRead,
  parseNotificationStreamEvent,
  snoozeNotification,
  updateNotificationPreferences
} from "../notifications"

describe("notifications service", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("lists notifications through shared UI services", async () => {
    mocks.bgRequest.mockResolvedValue({
      items: [{ id: 1, title: "Inbox item" }],
      total: 1
    })

    const result = await listNotifications({ limit: 20, offset: 0 })

    expect(result.items[0]?.id).toBe(1)
  })

  it("marks notifications read through shared UI services", async () => {
    mocks.bgRequest.mockResolvedValue({ updated: 1 })

    await expect(markNotificationsRead([1])).resolves.toEqual({ updated: 1 })
  })

  it("gets unread count through shared UI services", async () => {
    mocks.bgRequest.mockResolvedValue({ unread_count: 7 })

    await expect(getUnreadCount()).resolves.toEqual({ unread_count: 7 })
  })

  it("gets and updates notification preferences through shared UI services", async () => {
    const initialPreferences = {
      user_id: "user-1",
      reminder_enabled: true,
      job_completed_enabled: true,
      job_failed_enabled: true,
      updated_at: "2026-04-02T00:00:00Z"
    }
    const updatedPreferences = {
      ...initialPreferences,
      job_completed_enabled: false,
      updated_at: "2026-04-02T00:01:00Z"
    }
    mocks.bgRequest.mockResolvedValueOnce(initialPreferences)
    mocks.bgRequest.mockResolvedValueOnce(updatedPreferences)

    await expect(getNotificationPreferences()).resolves.toEqual(initialPreferences)
    await expect(
      updateNotificationPreferences({ job_completed_enabled: false })
    ).resolves.toEqual(updatedPreferences)

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(1, {
      path: "/api/v1/notifications/preferences",
      method: "GET"
    })
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(2, {
      path: "/api/v1/notifications/preferences",
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: { job_completed_enabled: false }
    })
  })

  it("shares notification query serialization for parity-sensitive paths", () => {
    expect(
      buildNotificationsQuery({ limit: 20, offset: 0, include_archived: false })
    ).toBe("?limit=20&offset=0&include_archived=false")
  })

  it("normalizes notification stream lines through shared helpers", () => {
    expect(
      parseNotificationStreamEvent(
        '{"event_id":12,"kind":"deep_research_completed","title":"Done","message":"Ready"}'
      )
    ).toEqual({
      event: "notification",
      id: 12,
      payload: {
        event_id: 12,
        kind: "deep_research_completed",
        title: "Done",
        message: "Ready"
      }
    })
  })

  it("retries the shared notification stream runner after an error and advances the cursor", async () => {
    vi.useFakeTimers()
    try {
      const onEvent = vi.fn()
      const onError = vi.fn()
      const readStream = vi.fn(
        async (_signal: AbortSignal, cursor: number, emit: (event: { event: string; id?: number; payload?: unknown }) => void) => {
          if (readStream.mock.calls.length === 1) {
            emit({
              event: "notification",
              id: 5,
              payload: {
                notification_id: 5,
                kind: "job_failed",
                title: "First",
                message: "First notification"
              }
            })
            const error = new Error("stream failed once") as Error & { cursor?: number }
            error.cursor = 5
            throw error
          }

          emit({
            event: "notification",
            id: 9,
            payload: {
              notification_id: 9,
              kind: "job_completed",
              title: "Second",
              message: "Second notification"
            }
          })
          await new Promise<void>(() => {})
          return Math.max(cursor, 9)
        }
      )

      const unsubscribe = createNotificationStreamSubscription({
        after: 0,
        reconnectDelayMs: 250,
        onEvent,
        onError,
        readStream
      })

      await Promise.resolve()
      expect(readStream).toHaveBeenCalledTimes(1)
      expect(onEvent).toHaveBeenCalledWith(
        expect.objectContaining({ event: "notification", id: 5 })
      )

      await vi.advanceTimersByTimeAsync(250)
      await Promise.resolve()

      expect(readStream).toHaveBeenCalledTimes(2)
      expect(readStream.mock.calls[0]?.[1]).toBe(0)
      expect(readStream.mock.calls[1]?.[1]).toBe(5)
      expect(onEvent).toHaveBeenCalledWith(
        expect.objectContaining({ event: "notification", id: 9 })
      )
      expect(onError).toHaveBeenCalledTimes(1)

      unsubscribe()
    } finally {
      vi.useRealTimers()
    }
  })

  it("throttles reconnects after a graceful stream close", async () => {
    vi.useFakeTimers()
    try {
      const readStream = vi.fn(async (_signal: AbortSignal, cursor: number) => cursor + 1)

      const unsubscribe = createNotificationStreamSubscription({
        after: 0,
        reconnectDelayMs: 250,
        onEvent: vi.fn(),
        readStream
      })

      await Promise.resolve()
      expect(readStream).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(249)
      await Promise.resolve()
      expect(readStream).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(1)
      await Promise.resolve()
      expect(readStream).toHaveBeenCalledTimes(2)

      unsubscribe()
    } finally {
      vi.useRealTimers()
    }
  })

  it("dismisses, cancels snoozes, and snoozes notifications through shared UI services", async () => {
    mocks.bgRequest.mockResolvedValueOnce({ dismissed: true })
    mocks.bgRequest.mockResolvedValueOnce({ cancelled: true, deleted_tasks: 1 })
    mocks.bgRequest.mockResolvedValueOnce({
      task_id: "task-123",
      run_at: "2026-03-20T00:15:00Z"
    })

    await dismissNotification(1)
    await cancelNotificationSnooze(1)
    await snoozeNotification(1, 15)

    expect(mocks.bgRequest).toHaveBeenCalledTimes(3)
  })
})
