import React, { useEffect } from "react"
import { act, render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getUnreadCount: vi.fn(),
  listNotifications: vi.fn(),
  subscribeNotificationsStream: vi.fn()
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

import {
  NotificationLifecycleProvider,
  useNotificationLifecycle,
  type NotificationLifecycleContextValue
} from "@web/components/notifications/NotificationLifecycleProvider"
import { AUTH_CREDENTIALS_CHANGED_EVENT } from "@web/lib/auth-events"

function Probe({ onValue }: { onValue: (value: NotificationLifecycleContextValue) => void }) {
  const value = useNotificationLifecycle()
  useEffect(() => onValue(value), [onValue, value])
  return (
    <output data-testid="lifecycle">
      {value.state}:{value.unreadCount}:{value.eventSequence}
    </output>
  )
}

const renderProvider = (scopeKey = "notifications:server-a:user-a") => {
  let latest: NotificationLifecycleContextValue | null = null
  const onValue = (value: NotificationLifecycleContextValue) => {
    latest = value
  }
  const view = render(
    <NotificationLifecycleProvider scopeKey={scopeKey}>
      <Probe onValue={onValue} />
    </NotificationLifecycleProvider>
  )
  return {
    ...view,
    latest: () => latest as NotificationLifecycleContextValue,
    rerenderScope: (nextScope: string) =>
      view.rerender(
        <NotificationLifecycleProvider scopeKey={nextScope}>
          <Probe onValue={onValue} />
        </NotificationLifecycleProvider>
      )
  }
}

const jwtFor = (subject: string): string => {
  const payload = window.btoa(JSON.stringify({ sub: subject }))
  return `eyJhbGciOiJub25lIn0.${payload.replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "")}.signature`
}

describe("NotificationLifecycleProvider", () => {
  beforeEach(() => {
    vi.useRealTimers()
    vi.clearAllMocks()
    window.localStorage.clear()
    mocks.getUnreadCount.mockResolvedValue({ unread_count: 5 })
    mocks.listNotifications.mockResolvedValue({
      items: [{ id: 10, title: "Existing" }],
      total: 1
    })
    mocks.subscribeNotificationsStream.mockImplementation(() => vi.fn())
  })

  it("does not start and abort a throwaway bootstrap during the Strict Mode probe mount", async () => {
    const signals: AbortSignal[] = []
    mocks.getUnreadCount.mockImplementation(
      async (options?: { signal?: AbortSignal }) => {
        if (options?.signal) signals.push(options.signal)
        return { unread_count: 5 }
      }
    )

    const view = render(
      <React.StrictMode>
        <NotificationLifecycleProvider scopeKey="notifications:server-a:user-a">
          <output>notifications</output>
        </NotificationLifecycleProvider>
      </React.StrictMode>
    )

    await waitFor(() => expect(mocks.getUnreadCount).toHaveBeenCalledTimes(1))
    expect(signals).toHaveLength(1)
    expect(signals[0].aborted).toBe(false)
    view.unmount()
  })

  it("owns one bootstrap, one stream, and one 30-second unread poll", async () => {
    vi.useFakeTimers()
    let streamOptions: Record<string, unknown> | undefined
    mocks.subscribeNotificationsStream.mockImplementation((options: Record<string, unknown>) => {
      streamOptions = options
      return vi.fn()
    })
    const view = renderProvider()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(1)
    expect(mocks.listNotifications).toHaveBeenCalledTimes(1)
    expect(mocks.subscribeNotificationsStream).toHaveBeenCalledWith(
      expect.objectContaining({ after: 10 })
    )
    expect(view.latest().state).toBe("connecting")

    act(() => (streamOptions?.onOpen as (() => void) | undefined)?.())
    expect(view.latest().state).toBe("active")
    expect(view.latest().unreadCount).toBe(5)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000)
    })
    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(2)
    expect(mocks.listNotifications).toHaveBeenCalledTimes(1)
    expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1)

    view.unmount()
    vi.useRealTimers()
  })

  it("does not overlap unread polls when a prior poll is still pending", async () => {
    vi.useFakeTimers()
    let resolvePoll: ((value: { unread_count: number }) => void) | undefined
    mocks.getUnreadCount
      .mockResolvedValueOnce({ unread_count: 5 })
      .mockImplementationOnce(
        () =>
          new Promise<{ unread_count: number }>((resolve) => {
            resolvePoll = resolve
          })
      )
      .mockResolvedValueOnce({ unread_count: 7 })
    const view = renderProvider()
    await act(async () => vi.advanceTimersByTimeAsync(0))

    await act(async () => vi.advanceTimersByTimeAsync(30_000))
    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(2)

    await act(async () => vi.advanceTimersByTimeAsync(30_000))
    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(2)

    await act(async () => resolvePoll?.({ unread_count: 6 }))
    await act(async () => vi.advanceTimersByTimeAsync(30_000))
    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(3)

    view.unmount()
    vi.useRealTimers()
  })

  it.each([
    [401, "auth-required"],
    [403, "unavailable"]
  ] as const)("stops all work after terminal bootstrap HTTP %s", async (status, state) => {
    vi.useFakeTimers()
    mocks.getUnreadCount.mockRejectedValue(Object.assign(new Error("terminal"), { status }))
    const view = renderProvider()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(view.latest().state).toBe(state)
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5 * 60_000)
    })
    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(1)
    expect(mocks.listNotifications).not.toHaveBeenCalled()
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled()

    view.unmount()
    vi.useRealTimers()
  })

  it("recovers unavailable only after explicit retry and leaves a failed retry unavailable", async () => {
    mocks.getUnreadCount
      .mockRejectedValueOnce(Object.assign(new Error("forbidden"), { status: 403 }))
      .mockRejectedValueOnce(Object.assign(new Error("still forbidden"), { status: 403 }))
      .mockResolvedValueOnce({ unread_count: 7 })
    const view = renderProvider()
    await waitFor(() => expect(view.latest().state).toBe("unavailable"))

    await act(async () => view.latest().tryAgain())
    expect(view.latest().state).toBe("unavailable")
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled()

    await act(async () => view.latest().refreshPermissions())
    expect(view.latest().state).toBe("connecting")
    expect(view.latest().unreadCount).toBe(7)
    expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1)
  })

  it("recovers auth-required after an authenticated scope remount and clears stale state immediately", async () => {
    let resolveNext: ((value: { unread_count: number }) => void) | undefined
    mocks.getUnreadCount
      .mockRejectedValueOnce(Object.assign(new Error("expired"), { status: 401 }))
      .mockImplementationOnce(
        () =>
          new Promise<{ unread_count: number }>((resolve) => {
            resolveNext = resolve
          })
      )
    const view = renderProvider("notifications:server-a:user-a")
    await waitFor(() => expect(view.latest().state).toBe("auth-required"))

    view.rerenderScope("notifications:server-a:user-b")
    expect(view.latest().state).toBe("connecting")
    expect(view.latest().unreadCount).toBe(0)

    await act(async () => resolveNext?.({ unread_count: 2 }))
    await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))
  })

  it("aborts an in-flight bootstrap before switching scopes", async () => {
    let firstSignal: AbortSignal | undefined
    mocks.getUnreadCount
      .mockImplementationOnce((options?: { signal?: AbortSignal }) => {
        firstSignal = options?.signal
        return new Promise(() => undefined)
      })
      .mockResolvedValueOnce({ unread_count: 2 })
    const view = renderProvider("notifications:server-a:user-a")
    await waitFor(() => expect(mocks.getUnreadCount).toHaveBeenCalledTimes(1))

    view.rerenderScope("notifications:server-a:user-b")

    expect(firstSignal?.aborted).toBe(true)
    await waitFor(() => expect(mocks.getUnreadCount).toHaveBeenCalledTimes(2))
  })

  it("restarts auth-required work after same-principal credentials rotate", async () => {
    mocks.getUnreadCount
      .mockRejectedValueOnce(Object.assign(new Error("expired"), { status: 401 }))
      .mockResolvedValueOnce({ unread_count: 3 })
    const view = renderProvider("notifications:server-a:user-a")
    await waitFor(() => expect(view.latest().state).toBe("auth-required"))

    act(() => {
      window.dispatchEvent(
        new CustomEvent(AUTH_CREDENTIALS_CHANGED_EVENT, {
          detail: { authenticated: true }
        })
      )
    })

    await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))
    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(2)
    expect(view.latest().unreadCount).toBe(3)
  })

  it("stops and clears scoped work when credentials are removed", async () => {
    const unsubscribe = vi.fn()
    mocks.subscribeNotificationsStream.mockReturnValue(unsubscribe)
    const view = renderProvider()
    await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))

    act(() => {
      window.dispatchEvent(
        new CustomEvent(AUTH_CREDENTIALS_CHANGED_EVENT, {
          detail: { authenticated: false }
        })
      )
    })

    expect(unsubscribe).toHaveBeenCalledTimes(1)
    expect(view.latest().state).toBe("auth-required")
    expect(view.latest().unreadCount).toBe(0)
  })

  it("does not restart unavailable work after same-principal credentials rotate", async () => {
    mocks.getUnreadCount.mockRejectedValue(
      Object.assign(new Error("forbidden"), { status: 403 })
    )
    const view = renderProvider("notifications:server-a:user-a")
    await waitFor(() => expect(view.latest().state).toBe("unavailable"))

    act(() => {
      window.dispatchEvent(
        new CustomEvent(AUTH_CREDENTIALS_CHANGED_EVENT, {
          detail: { authenticated: true }
        })
      )
    })
    await act(async () => Promise.resolve())

    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(1)
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled()
    expect(view.latest().state).toBe("unavailable")
  })

  it("does not restart unavailable work after cross-tab same-principal token rotation", async () => {
    mocks.getUnreadCount.mockRejectedValue(
      Object.assign(new Error("forbidden"), { status: 403 })
    )
    const view = renderProvider("notifications:server-a:user-a")
    await waitFor(() => expect(view.latest().state).toBe("unavailable"))

    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "access_token",
          oldValue: "old-token",
          newValue: "rotated-token"
        })
      )
    })
    await act(async () => Promise.resolve())

    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(1)
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled()
    expect(view.latest().state).toBe("unavailable")
  })

  it("clears unavailable state and restarts after a cross-tab principal change", async () => {
    window.localStorage.setItem("access_token", jwtFor("user-a"))
    mocks.getUnreadCount
      .mockRejectedValueOnce(Object.assign(new Error("forbidden"), { status: 403 }))
      .mockResolvedValueOnce({ unread_count: 2 })
    let latest: NotificationLifecycleContextValue | null = null
    render(
      <NotificationLifecycleProvider>
        <Probe onValue={(value) => { latest = value }} />
      </NotificationLifecycleProvider>
    )
    await waitFor(() => expect(latest?.state).toBe("unavailable"))

    const nextToken = jwtFor("user-b")
    act(() => {
      window.localStorage.setItem("access_token", nextToken)
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "access_token",
          oldValue: jwtFor("user-a"),
          newValue: nextToken
        })
      )
    })

    expect(latest?.state).toBe("connecting")
    expect(latest?.unreadCount).toBe(0)
    await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))
    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(2)
  })

  it("shows transient stream failures as degraded and recovers only on onOpen", async () => {
    let streamOptions: Record<string, unknown> | undefined
    mocks.subscribeNotificationsStream.mockImplementation((options: Record<string, unknown>) => {
      streamOptions = options
      return vi.fn()
    })
    const view = renderProvider()
    await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))

    act(() => (streamOptions?.onOpen as (() => void) | undefined)?.())
    expect(view.latest().state).toBe("active")
    act(() =>
      (streamOptions?.onError as ((error: unknown) => void) | undefined)?.(
        Object.assign(new Error("offline"), { status: 503 })
      )
    )
    expect(view.latest().state).toBe("degraded")
    act(() => (streamOptions?.onOpen as (() => void) | undefined)?.())
    expect(view.latest().state).toBe("active")
  })

  it("recovers an open stream after a transient unread poll succeeds", async () => {
    vi.useFakeTimers()
    let streamOptions: Record<string, unknown> | undefined
    mocks.subscribeNotificationsStream.mockImplementation((options: Record<string, unknown>) => {
      streamOptions = options
      return vi.fn()
    })
    mocks.getUnreadCount
      .mockResolvedValueOnce({ unread_count: 5 })
      .mockRejectedValueOnce(Object.assign(new Error("offline"), { status: 503 }))
      .mockResolvedValueOnce({ unread_count: 8 })
    const view = renderProvider()
    await act(async () => vi.advanceTimersByTimeAsync(0))
    act(() => (streamOptions?.onOpen as (() => void) | undefined)?.())

    await act(async () => vi.advanceTimersByTimeAsync(30_000))
    expect(view.latest().state).toBe("degraded")

    await act(async () => vi.advanceTimersByTimeAsync(30_000))
    expect(view.latest().state).toBe("active")
    expect(view.latest().unreadCount).toBe(8)
    vi.useRealTimers()
  })

  it("cancels polling after a terminal stream error", async () => {
    vi.useFakeTimers()
    const unsubscribe = vi.fn()
    let streamOptions: Record<string, unknown> | undefined
    mocks.subscribeNotificationsStream.mockImplementation((options: Record<string, unknown>) => {
      streamOptions = options
      return unsubscribe
    })
    const view = renderProvider()
    await act(async () => vi.advanceTimersByTimeAsync(0))

    act(() =>
      (streamOptions?.onError as ((error: unknown) => void) | undefined)?.(
        Object.assign(new Error("forbidden"), { status: 403 })
      )
    )
    expect(view.latest().state).toBe("unavailable")
    expect(unsubscribe).toHaveBeenCalledTimes(1)

    await act(async () => vi.advanceTimersByTimeAsync(5 * 60_000))
    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(1)
    vi.useRealTimers()
  })

  it("does not install polling when a terminal error fires during stream creation", async () => {
    vi.useFakeTimers()
    const unsubscribe = vi.fn()
    mocks.subscribeNotificationsStream.mockImplementation(
      (options: { onError?: (error: unknown) => void }) => {
        options.onError?.(Object.assign(new Error("forbidden"), { status: 403 }))
        return unsubscribe
      }
    )
    const view = renderProvider()
    await act(async () => vi.advanceTimersByTimeAsync(0))

    expect(view.latest().state).toBe("unavailable")
    expect(unsubscribe).toHaveBeenCalledTimes(1)

    await act(async () => vi.advanceTimersByTimeAsync(5 * 60_000))
    expect(mocks.getUnreadCount).toHaveBeenCalledTimes(1)
    vi.useRealTimers()
  })

  it("keeps an open stream degraded until a transient unread bootstrap recovers", async () => {
    vi.useFakeTimers()
    let streamOptions: Record<string, unknown> | undefined
    mocks.getUnreadCount
      .mockRejectedValueOnce(Object.assign(new Error("offline"), { status: 503 }))
      .mockResolvedValueOnce({ unread_count: 4 })
    mocks.subscribeNotificationsStream.mockImplementation((options: Record<string, unknown>) => {
      streamOptions = options
      return vi.fn()
    })
    const view = renderProvider()
    await act(async () => vi.advanceTimersByTimeAsync(0))

    act(() => (streamOptions?.onOpen as (() => void) | undefined)?.())
    expect(view.latest().state).toBe("degraded")

    await act(async () => vi.advanceTimersByTimeAsync(30_000))
    expect(view.latest().state).toBe("active")
    expect(view.latest().unreadCount).toBe(4)
    vi.useRealTimers()
  })

  it("retries a transient cursor bootstrap before opening the stream", async () => {
    vi.useFakeTimers()
    mocks.listNotifications
      .mockRejectedValueOnce(Object.assign(new Error("offline"), { status: 503 }))
      .mockResolvedValueOnce({ items: [{ id: 22 }], total: 1 })
    const view = renderProvider()
    await act(async () => vi.advanceTimersByTimeAsync(0))

    expect(view.latest().state).toBe("degraded")
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled()

    await act(async () => vi.advanceTimersByTimeAsync(30_000))
    expect(mocks.subscribeNotificationsStream).toHaveBeenCalledWith(
      expect.objectContaining({ after: 22 })
    )
    vi.useRealTimers()
  })

  it("publishes provider-owned events and updates unread count", async () => {
    let streamOptions: Record<string, unknown> | undefined
    mocks.subscribeNotificationsStream.mockImplementation((options: Record<string, unknown>) => {
      streamOptions = options
      return vi.fn()
    })
    const view = renderProvider()
    await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))

    act(() =>
      (streamOptions?.onEvent as ((event: unknown) => void) | undefined)?.({
        event: "notification",
        id: 11,
        payload: { notification_id: 11, title: "New" }
      })
    )

    expect(view.latest().eventSequence).toBe(1)
    expect(view.latest().latestEvent).toMatchObject({ id: 11 })
    expect(view.latest().unreadCount).toBe(6)
  })

  it("adds a validated coalesced count to the lifecycle unread total", async () => {
    let streamOptions: Record<string, unknown> | undefined
    mocks.subscribeNotificationsStream.mockImplementation((options: Record<string, unknown>) => {
      streamOptions = options
      return vi.fn()
    })
    const view = renderProvider()
    await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))

    act(() =>
      (streamOptions?.onEvent as ((event: unknown) => void) | undefined)?.({
        event: "notifications_coalesced",
        id: 14,
        payload: { count: 3 }
      })
    )

    expect(view.latest().unreadCount).toBe(8)
  })

  it.each([0, -1, Number.POSITIVE_INFINITY, Number.NaN])(
    "ignores invalid coalesced unread count %s",
    async (count) => {
      let streamOptions: Record<string, unknown> | undefined
      mocks.subscribeNotificationsStream.mockImplementation((options: Record<string, unknown>) => {
        streamOptions = options
        return vi.fn()
      })
      const view = renderProvider()
      await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))

      act(() =>
        (streamOptions?.onEvent as ((event: unknown) => void) | undefined)?.({
          event: "notifications_coalesced",
          id: 14,
          payload: { count }
        })
      )

      expect(view.latest().unreadCount).toBe(5)
    }
  )

  it("retains every stream event delivered in one React batch", async () => {
    let streamOptions: Record<string, unknown> | undefined
    mocks.subscribeNotificationsStream.mockImplementation((options: Record<string, unknown>) => {
      streamOptions = options
      return vi.fn()
    })
    const view = renderProvider()
    await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))

    act(() => {
      const onEvent = streamOptions?.onEvent as ((event: unknown) => void) | undefined
      onEvent?.({ event: "notification", id: 11, payload: { notification_id: 11 } })
      onEvent?.({ event: "notification", id: 12, payload: { notification_id: 12 } })
    })

    expect(view.latest().events.map(({ event }) => event.id)).toEqual([11, 12])
    expect(view.latest().eventSequence).toBe(2)
  })

  it("classifies mutation failures once without replaying the mutation", async () => {
    const view = renderProvider()
    await waitFor(() => expect(mocks.subscribeNotificationsStream).toHaveBeenCalledTimes(1))
    const mutation = vi.fn()
    const transient = Object.assign(new Error("try again"), { status: 503 })

    act(() => view.latest().reportMutationError(transient))

    expect(view.latest().state).toBe("degraded")
    expect(view.latest().mutationError).toBe(transient)
    expect(mutation).not.toHaveBeenCalled()

    act(() =>
      view.latest().reportMutationError(Object.assign(new Error("forbidden"), { status: 403 }))
    )
    expect(view.latest().state).toBe("unavailable")
  })

  it("performs no notification work while disabled", async () => {
    render(
      <NotificationLifecycleProvider scopeKey="demo" enabled={false}>
        <Probe onValue={vi.fn()} />
      </NotificationLifecycleProvider>
    )
    await act(async () => Promise.resolve())

    expect(mocks.getUnreadCount).not.toHaveBeenCalled()
    expect(mocks.listNotifications).not.toHaveBeenCalled()
    expect(mocks.subscribeNotificationsStream).not.toHaveBeenCalled()
  })
})
