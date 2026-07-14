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
  runNotificationMutation,
  snoozeNotification,
  subscribeNotificationsStream,
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

  it("reports stream acquisition before the first notification line", async () => {
    mocks.bgStream.mockImplementation(async function* (options: { onOpen?: () => void }) {
      options.onOpen?.()
    })
    const onOpen = vi.fn()

    const unsubscribe = subscribeNotificationsStream({
      onEvent: vi.fn(),
      onOpen
    })

    await vi.waitFor(() => expect(onOpen).toHaveBeenCalledTimes(1))
    unsubscribe()
  })

  it("retries the shared notification stream runner after an error and advances the cursor", async () => {
    vi.useFakeTimers()
    try {
      const onEvent = vi.fn()
      const onError = vi.fn()
      const onOpen = vi.fn()
      const readStream = vi.fn(
        async (
          _signal: AbortSignal,
          cursor: number,
          emit: (event: { event: string; id?: number; payload?: unknown }) => void,
          markOpen: () => void
        ) => {
          markOpen()
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
        reconnectJitter: 0.5,
        onEvent,
        onError,
        onOpen,
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
      expect(onOpen).toHaveBeenCalledTimes(2)

      unsubscribe()
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not report active until the reader confirms the stream is open", async () => {
    const onOpen = vi.fn()
    let confirmOpen: (() => void) | undefined
    const readStream = vi.fn(
      async (
        _signal: AbortSignal,
        cursor: number,
        _onEvent: (event: { event: string }) => void,
        markOpen: () => void
      ) => {
        confirmOpen = markOpen
        await new Promise<void>(() => {})
        return cursor
      }
    )

    const unsubscribe = createNotificationStreamSubscription({
      onEvent: vi.fn(),
      onOpen,
      readStream
    })

    expect(onOpen).not.toHaveBeenCalled()
    await Promise.resolve()
    expect(onOpen).not.toHaveBeenCalled()

    confirmOpen?.()
    expect(onOpen).toHaveBeenCalledTimes(1)

    unsubscribe()
  })

  it("stops reconnecting after a terminal stream response", async () => {
    vi.useFakeTimers()
    try {
      const terminalError = Object.assign(new Error("Forbidden"), {
        status: 403
      })
      const readStream = vi.fn(async () => {
        throw terminalError
      })
      const onError = vi.fn()

      const unsubscribe = createNotificationStreamSubscription({
        reconnectDelayMs: 250,
        onEvent: vi.fn(),
        onError,
        readStream
      })

      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(60_000)

      expect(readStream).toHaveBeenCalledTimes(1)
      expect(onError).toHaveBeenCalledTimes(1)

      unsubscribe()
    } finally {
      vi.useRealTimers()
    }
  })

  it("samples reconnect jitter when no deterministic override is provided", async () => {
    vi.useFakeTimers()
    const random = vi.spyOn(Math, "random").mockReturnValue(1)
    try {
      const readStream = vi.fn(async () => {
        throw Object.assign(new Error("service unavailable"), { status: 503 })
      })

      const unsubscribe = createNotificationStreamSubscription({
        reconnectDelayMs: 250,
        onEvent: vi.fn(),
        readStream
      })

      await Promise.resolve()

      expect(random).toHaveBeenCalledTimes(1)
      unsubscribe()
    } finally {
      random.mockRestore()
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
        reconnectJitter: 0.5,
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

  it("backs off repeated stream closes before acquisition", async () => {
    vi.useFakeTimers()
    try {
      const readStream = vi.fn(
        async (
          _signal: AbortSignal,
          cursor: number,
          _onEvent: (event: { event: string }) => void
        ) => {
          return cursor
        }
      )

      const unsubscribe = createNotificationStreamSubscription({
        reconnectDelayMs: 250,
        reconnectJitter: 0.5,
        onEvent: vi.fn(),
        readStream
      })

      await Promise.resolve()
      expect(readStream).toHaveBeenCalledTimes(1)

      await vi.advanceTimersByTimeAsync(250)
      expect(readStream).toHaveBeenCalledTimes(2)

      await vi.advanceTimersByTimeAsync(250)
      expect(readStream).toHaveBeenCalledTimes(2)

      await vi.advanceTimersByTimeAsync(250)
      expect(readStream).toHaveBeenCalledTimes(3)

      unsubscribe()
    } finally {
      vi.useRealTimers()
    }
  })

  it("resets reconnect backoff when a quiet stream opens successfully", async () => {
    vi.useFakeTimers()
    try {
      const readStream = vi.fn(
        async (
          _signal: AbortSignal,
          cursor: number,
          _onEvent: (event: { event: string }) => void,
          markOpen: () => void
        ) => {
          if (readStream.mock.calls.length === 1) {
            throw Object.assign(new Error("offline"), { status: 503 })
          }
          markOpen()
          return cursor
        }
      )

      const unsubscribe = createNotificationStreamSubscription({
        reconnectDelayMs: 250,
        reconnectJitter: 0.5,
        onEvent: vi.fn(),
        readStream
      })

      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(250)
      expect(readStream).toHaveBeenCalledTimes(2)

      await vi.advanceTimersByTimeAsync(250)
      expect(readStream).toHaveBeenCalledTimes(3)

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

  it("never automatically replays explicit notification mutations", async () => {
    const failure = Object.assign(new Error("temporary outage"), { status: 503 })
    mocks.bgRequest.mockRejectedValue(failure)

    await expect(markNotificationsRead([1])).rejects.toBe(failure)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)

    await expect(dismissNotification(1)).rejects.toBe(failure)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)

    await expect(snoozeNotification(1, 15)).rejects.toBe(failure)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(3)

    await expect(cancelNotificationSnooze(1)).rejects.toBe(failure)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(4)

    await expect(
      updateNotificationPreferences({ reminder_enabled: false })
    ).rejects.toBe(failure)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(5)
  })

  it("keeps future mutation helpers single-attempt", async () => {
    const failure = new Error("try again explicitly")
    const request = vi.fn().mockRejectedValue(failure)

    await expect(runNotificationMutation(request)).rejects.toBe(failure)

    expect(request).toHaveBeenCalledTimes(1)
  })
})
