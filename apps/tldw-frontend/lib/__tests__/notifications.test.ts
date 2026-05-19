/** @vitest-environment node */

import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
  apiDelete: vi.fn(),
  apiPatch: vi.fn(),
  buildAuthHeaders: vi.fn(),
  hasExplicitAuthHeaders: vi.fn(),
  getApiBaseUrl: vi.fn(),
  streamStructuredSSE: vi.fn(),
  bgRequest: vi.fn(),
  bgStream: vi.fn()
}))

vi.mock("@web/lib/api", () => ({
  apiClient: {
    get: (...args: unknown[]) => mocks.apiGet(...args),
    post: (...args: unknown[]) => mocks.apiPost(...args),
    delete: (...args: unknown[]) => mocks.apiDelete(...args),
    patch: (...args: unknown[]) => mocks.apiPatch(...args)
  },
  buildAuthHeaders: (...args: unknown[]) => mocks.buildAuthHeaders(...args),
  hasExplicitAuthHeaders: (...args: unknown[]) =>
    mocks.hasExplicitAuthHeaders(...args),
  getApiBaseUrl: (...args: unknown[]) => mocks.getApiBaseUrl(...args)
}))

vi.mock("@web/lib/sse", () => ({
  streamStructuredSSE: (...args: unknown[]) => mocks.streamStructuredSSE(...args)
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
}))

import {
  cancelNotificationSnooze,
  dismissNotification,
  getNotificationPreferences,
  getUnreadCount,
  listNotifications,
  markNotificationsRead,
  snoozeNotification,
  subscribeNotificationsStream,
  updateNotificationPreferences
} from "../api/notifications"

describe("web notifications adapter", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.buildAuthHeaders.mockReturnValue({
      Authorization: "Bearer web-token",
      "X-CSRF-Token": "csrf-token"
    })
    mocks.hasExplicitAuthHeaders.mockReturnValue(true)
    mocks.getApiBaseUrl.mockReturnValue("http://example.test/api/v1")
    mocks.apiGet.mockResolvedValue({ items: [], total: 0 })
    mocks.apiPost.mockResolvedValue({ updated: 1, dismissed: true, task_id: "task-1", run_at: "2026-03-20T00:15:00Z" })
    mocks.apiDelete.mockResolvedValue({ cancelled: true, deleted_tasks: 1 })
    mocks.apiPatch.mockResolvedValue({
      user_id: "user-1",
      reminder_enabled: true,
      job_completed_enabled: true,
      job_failed_enabled: true,
      updated_at: "2026-03-20T00:00:00Z"
    })
    mocks.streamStructuredSSE.mockImplementation(async (_url, _options, onEvent) => {
      onEvent({
        event: "notification",
        id: 99,
        payload: {
          notification_id: 99,
          kind: "info",
          title: "Stream item",
          message: "From web SSE",
          created_at: "2026-03-20T00:00:00Z"
        }
      })
      const error = new Error("stream failed once") as Error & { cursor?: number }
      error.cursor = 99
      throw error
    })
    mocks.bgRequest.mockResolvedValue({ items: [], total: 0 })
    mocks.bgStream.mockImplementation(async function* () {
      yield undefined
      await new Promise(() => {})
    })
  })

  it("uses the web apiClient transport for inbox CRUD", async () => {
    await listNotifications({ limit: 20, offset: 0 })
    await listNotifications({ limit: 25, offset: 5, include_archived: true, only_snoozed: true })
    await getUnreadCount()
    await markNotificationsRead([1])
    await dismissNotification(1)
    await cancelNotificationSnooze(1)
    await snoozeNotification(1, 15)
    await getNotificationPreferences()
    await updateNotificationPreferences({ reminder_enabled: false })

    expect(mocks.apiGet).toHaveBeenCalledWith(
      "/notifications?limit=20&offset=0&include_archived=false",
      { withCredentials: false }
    )
    expect(mocks.apiGet).toHaveBeenCalledWith(
      "/notifications?limit=25&offset=5&include_archived=true&only_snoozed=true",
      { withCredentials: false }
    )
    expect(mocks.apiGet).toHaveBeenCalledWith("/notifications/unread-count", {
      withCredentials: false
    })
    expect(mocks.apiPost).toHaveBeenCalledWith("/notifications/mark-read", {
      ids: [1]
    }, { withCredentials: false })
    expect(mocks.apiPost).toHaveBeenCalledWith("/notifications/1/dismiss", undefined, {
      withCredentials: false
    })
    expect(mocks.apiDelete).toHaveBeenCalledWith("/notifications/1/snooze", {
      withCredentials: false
    })
    expect(mocks.apiPost).toHaveBeenCalledWith("/notifications/1/snooze", {
      minutes: 15
    }, { withCredentials: false })
    expect(mocks.apiGet).toHaveBeenCalledWith("/notifications/preferences", {
      headers: expect.objectContaining({
        Authorization: "Bearer web-token",
        "X-CSRF-Token": "csrf-token"
      }),
      withCredentials: false
    })
    expect(mocks.apiPatch).toHaveBeenCalledWith("/notifications/preferences", {
      reminder_enabled: false
    }, {
      headers: expect.objectContaining({
        Authorization: "Bearer web-token",
        "X-CSRF-Token": "csrf-token"
      }),
      withCredentials: false
    })
  })

  it("uses the web apiClient transport for notification preferences", async () => {
    const initialPreferences = {
      user_id: "user-1",
      reminder_enabled: true,
      job_completed_enabled: true,
      job_failed_enabled: true,
      updated_at: "2026-04-02T00:00:00Z"
    }
    const updatedPreferences = {
      ...initialPreferences,
      job_failed_enabled: false,
      updated_at: "2026-04-02T00:01:00Z"
    }
    mocks.apiGet.mockResolvedValueOnce(initialPreferences)
    mocks.apiPatch.mockResolvedValueOnce(updatedPreferences)

    await expect(getNotificationPreferences()).resolves.toEqual(initialPreferences)
    await expect(
      updateNotificationPreferences({ job_failed_enabled: false })
    ).resolves.toEqual(updatedPreferences)

    expect(mocks.buildAuthHeaders).toHaveBeenCalledWith("GET")
    expect(mocks.buildAuthHeaders).toHaveBeenCalledWith("PATCH")
    expect(mocks.apiGet).toHaveBeenCalledWith("/notifications/preferences", {
      headers: {
        Authorization: "Bearer web-token",
        "X-CSRF-Token": "csrf-token"
      },
      withCredentials: false
    })
    expect(mocks.apiPatch).toHaveBeenCalledWith("/notifications/preferences", {
      job_failed_enabled: false
    }, {
      headers: {
        Authorization: "Bearer web-token",
        "X-CSRF-Token": "csrf-token"
      },
      withCredentials: false
    })
  })

  it("omits cookie credentials for the notification stream when header auth is present", async () => {
    const unsubscribe = subscribeNotificationsStream({
      after: 42,
      onEvent: vi.fn()
    })

    await Promise.resolve()

    expect(mocks.buildAuthHeaders).toHaveBeenCalledWith("GET")
    expect(mocks.getApiBaseUrl).toHaveBeenCalled()
    expect(mocks.streamStructuredSSE).toHaveBeenCalledWith(
      "http://example.test/api/v1/notifications/stream?after=42",
      expect.objectContaining({
        method: "GET",
        credentials: "omit",
        signal: expect.any(AbortSignal),
        headers: expect.objectContaining({
          Authorization: "Bearer web-token",
          "X-CSRF-Token": "csrf-token"
        })
      }),
      expect.any(Function)
    )

    unsubscribe()
  })

  it("keeps cookie credentials enabled when there is no header-based auth", async () => {
    mocks.buildAuthHeaders.mockReturnValue({})
    mocks.hasExplicitAuthHeaders.mockReturnValue(false)
    mocks.streamStructuredSSE.mockImplementationOnce(async (_url, options, _onEvent) => {
      expect(options).toEqual(
        expect.objectContaining({
          credentials: "include"
        })
      )
    })

    const unsubscribe = subscribeNotificationsStream({
      after: 1,
      onEvent: vi.fn()
    })

    await Promise.resolve()

    unsubscribe()
  })

  it("builds relative notification SSE URLs when the api base is relative", async () => {
    mocks.getApiBaseUrl.mockReturnValue("/api/v1")
    mocks.streamStructuredSSE.mockImplementationOnce(async (url, _options, _onEvent) => {
      expect(url).toBe("/api/v1/notifications/stream?after=12")
      return
    })

    const unsubscribe = subscribeNotificationsStream({
      after: 12,
      onEvent: vi.fn(),
      onError: vi.fn()
    })

    await Promise.resolve()

    expect(mocks.streamStructuredSSE).toHaveBeenCalledTimes(1)

    unsubscribe()
  })
})
