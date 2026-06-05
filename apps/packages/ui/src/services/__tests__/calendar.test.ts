import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

import {
  copyCalendarItemIntoTldw,
  createCalDavAccount,
  createCalendarItem,
  deleteCalDavAccount,
  discoverExternalCalendars,
  getCalendarAgenda,
  listCalendars,
  revokeCalDavAccount,
  triggerCalendarSync,
  updateCalendarItem,
  verifyCalDavAccount,
  type CalendarAgendaQuery,
  type CalendarItemCreateRequest
} from "@/services/calendar"

describe("calendar service contract", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
  })

  it("lists calendars through the calendar collection endpoint", async () => {
    mocks.bgRequest.mockResolvedValue({ items: [], total: 0 })

    const response = await listCalendars()

    expect(response.total).toBe(0)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "GET",
        path: "/api/v1/calendar/calendars"
      })
    )
  })

  it("creates calendar items with the typed payload", async () => {
    const payload: CalendarItemCreateRequest = {
      calendar_id: 7,
      kind: "event",
      title: "Planning review",
      description: "Review calendar MVP",
      start_at: "2026-06-05T09:00:00+00:00",
      end_at: "2026-06-05T10:00:00+00:00",
      timezone: "UTC",
      all_day: false,
      local_tags: ["planning"],
      metadata: { source: "test" }
    }
    mocks.bgRequest.mockResolvedValue({
      id: 12,
      calendar_id: 7,
      kind: "event",
      source_owner: "tldw",
      provider_owned: false,
      title: "Planning review",
      all_day: false,
      status: "confirmed",
      local_tags: ["planning"],
      metadata: { source: "test" },
      created_at: "2026-06-05T09:00:00+00:00",
      updated_at: "2026-06-05T09:00:00+00:00"
    })

    const response = await createCalendarItem(payload)

    expect(response.source_owner).toBe("tldw")
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "POST",
        path: "/api/v1/calendar/items",
        body: payload
      })
    )
  })

  it("rejects provider-owned item updates before sending a request", async () => {
    await expect(
      updateCalendarItem(42, {
        title: "Local edit",
        source_owner: "provider"
      })
    ).rejects.toThrow("Provider-owned calendar items are read-only")

    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })

  it("builds encoded agenda queries and requires a bounded window", async () => {
    mocks.bgRequest.mockResolvedValue({
      start_at: "2026-06-05T09:00:00+00:00",
      end_at: "2026-06-05T17:00:00+00:00",
      items: []
    })
    const query: CalendarAgendaQuery = {
      start_at: "2026-06-05T09:00:00+00:00",
      end_at: "2026-06-05T17:00:00+00:00",
      calendar_ids: [1, 42],
      include_scheduled_tasks: false
    }

    await expect(getCalendarAgenda({ end_at: query.end_at } as CalendarAgendaQuery)).rejects.toThrow(
      "start_at and end_at are required"
    )
    await expect(getCalendarAgenda({ start_at: query.start_at } as CalendarAgendaQuery)).rejects.toThrow(
      "start_at and end_at are required"
    )
    const response = await getCalendarAgenda(query)

    expect(response.items).toEqual([])
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        method: "GET",
        path:
          "/api/v1/calendar/views/agenda?start_at=2026-06-05T09%3A00%3A00%2B00%3A00&end_at=2026-06-05T17%3A00%3A00%2B00%3A00&calendar_ids=1&calendar_ids=42&include_scheduled_tasks=false"
      })
    )
  })

  it("only sends external calendar secrets to create and verify account endpoints", async () => {
    mocks.bgRequest
      .mockResolvedValueOnce({ id: 3, provider: "caldav", display_name: "Fastmail" })
      .mockResolvedValueOnce({ account_id: 3, verified: true })
      .mockResolvedValueOnce({ items: [] })
      .mockResolvedValueOnce({ revoked: true })
      .mockResolvedValueOnce({ deleted: true })
      .mockResolvedValueOnce({ binding_id: 11, queued: false, status: "not_implemented" })
      .mockResolvedValueOnce({ id: 99, source_owner: "tldw" })

    await createCalDavAccount({
      display_name: "Fastmail",
      server_url: "https://caldav.fastmail.com/dav/calendars",
      username: "reader@example.com",
      password: "create-secret"
    })
    await verifyCalDavAccount(3, { password: "verify-secret" })
    await discoverExternalCalendars(3, { password: "ignored-secret" } as unknown as never)
    await revokeCalDavAccount(3, { token: "ignored-secret" } as unknown as never)
    await deleteCalDavAccount(3, { token: "ignored-secret" } as unknown as never)
    await triggerCalendarSync(11, { password: "ignored-secret" } as unknown as never)
    await copyCalendarItemIntoTldw(55, {
      target_calendar_id: 7,
      title: "Local copy",
      password: "ignored-secret"
    } as unknown as never)

    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        method: "POST",
        path: "/api/v1/calendar/external/accounts",
        body: expect.objectContaining({ password: "create-secret" })
      })
    )
    expect(mocks.bgRequest).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        method: "POST",
        path: "/api/v1/calendar/external/accounts/3/verify",
        body: expect.objectContaining({ password: "verify-secret" })
      })
    )

    for (const call of mocks.bgRequest.mock.calls.slice(2)) {
      expect(JSON.stringify(call[0])).not.toContain("ignored-secret")
    }
  })
})
