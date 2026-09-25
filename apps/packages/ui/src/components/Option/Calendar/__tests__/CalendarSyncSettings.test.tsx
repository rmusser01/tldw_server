import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  listCalDavAccounts: vi.fn(),
  createCalDavAccount: vi.fn(),
  verifyCalDavAccount: vi.fn(),
  discoverExternalCalendars: vi.fn(),
  createExternalCalendarBinding: vi.fn(),
  listExternalCalendarBindings: vi.fn(),
  triggerCalendarSync: vi.fn(),
  revokeCalDavAccount: vi.fn(),
  deleteCalDavAccount: vi.fn()
}))

vi.mock("@/services/calendar", () => ({
  listCalDavAccounts: (...args: unknown[]) => mocks.listCalDavAccounts(...args),
  createCalDavAccount: (...args: unknown[]) => mocks.createCalDavAccount(...args),
  verifyCalDavAccount: (...args: unknown[]) => mocks.verifyCalDavAccount(...args),
  discoverExternalCalendars: (...args: unknown[]) => mocks.discoverExternalCalendars(...args),
  createExternalCalendarBinding: (...args: unknown[]) => mocks.createExternalCalendarBinding(...args),
  listExternalCalendarBindings: (...args: unknown[]) => mocks.listExternalCalendarBindings(...args),
  triggerCalendarSync: (...args: unknown[]) => mocks.triggerCalendarSync(...args),
  revokeCalDavAccount: (...args: unknown[]) => mocks.revokeCalDavAccount(...args),
  deleteCalDavAccount: (...args: unknown[]) => mocks.deleteCalDavAccount(...args)
}))

import { CalendarSyncSettings } from "../CalendarSyncSettings"

const calendars = [
  {
    id: 7,
    tenant_id: "default",
    owner_user_id: 1,
    org_id: null,
    name: "Research",
    color: "#2563eb",
    timezone: "UTC",
    visibility: "private",
    created_at: "2026-06-01T00:00:00Z",
    updated_at: "2026-06-01T00:00:00Z"
  }
]

const account = {
  id: 3,
  tenant_id: "default",
  user_id: 1,
  provider: "caldav",
  display_name: "Fastmail",
  account_metadata: {
    server_url: "https://caldav.fastmail.com/dav/calendars",
    username: "reader@example.test"
  },
  status: "active",
  created_at: "2026-06-01T00:00:00Z",
  updated_at: "2026-06-01T00:00:00Z"
}

const binding = {
  id: 10,
  account_id: 3,
  calendar_id: 7,
  remote_calendar_id: "https://caldav.fastmail.com/calendars/user/work/",
  remote_display_name: "Work",
  sync_enabled: true,
  sync_interval_minutes: 60,
  lookback_days: 30,
  lookahead_days: 120,
  provider_capabilities: { sync_strategy: "bounded_polling" },
  last_sync_at: "2026-06-01T12:00:00Z",
  next_scan_at: "2026-06-01T13:00:00Z",
  last_error: "provider down",
  created_at: "2026-06-01T00:00:00Z",
  updated_at: "2026-06-01T00:00:00Z"
}

const renderSettings = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <CalendarSyncSettings calendars={calendars} onChanged={vi.fn()} />
    </QueryClientProvider>
  )
}

describe("CalendarSyncSettings", () => {
  beforeEach(() => {
    for (const mock of Object.values(mocks)) {
      mock.mockReset()
    }
    mocks.listCalDavAccounts.mockResolvedValue({ items: [account], total: 1 })
    mocks.listExternalCalendarBindings.mockResolvedValue({ items: [binding], total: 1 })
    mocks.discoverExternalCalendars.mockResolvedValue({
      items: [
        {
          remote_calendar_id: "https://caldav.fastmail.com/calendars/user/work/",
          remote_display_name: "Work",
          provider_capabilities: {
            supports_vevent: true,
            sync_strategy: "bounded_polling"
          }
        }
      ]
    })
    mocks.createCalDavAccount.mockResolvedValue({ ...account, id: 4, display_name: "Personal Fastmail" })
    mocks.verifyCalDavAccount.mockResolvedValue({ account_id: 4, verified: true, status: "ok" })
    mocks.createExternalCalendarBinding.mockResolvedValue({ ...binding, id: 11 })
    mocks.triggerCalendarSync.mockResolvedValue({ binding_id: 10, queued: true, status: "queued", job_id: 99 })
    mocks.revokeCalDavAccount.mockResolvedValue({ revoked: true })
    mocks.deleteCalDavAccount.mockResolvedValue({ deleted: true })
  })

  it("adds a CalDAV account and sends the password only to create and verify calls", async () => {
    const user = userEvent.setup()
    renderSettings()

    await user.click(await screen.findByRole("button", { name: "Add CalDAV account" }))
    await user.type(screen.getByRole("textbox", { name: "Account name" }), "Personal Fastmail")
    await user.type(screen.getByRole("textbox", { name: "Server URL" }), "https://caldav.fastmail.com/dav/calendars")
    await user.type(screen.getByRole("textbox", { name: "Username" }), "reader@example.test")
    await user.type(screen.getByLabelText("Password"), "app-password")
    await user.click(screen.getByRole("button", { name: "Save and verify account" }))

    await waitFor(() => {
      expect(mocks.createCalDavAccount).toHaveBeenCalledWith(
        expect.objectContaining({
          display_name: "Personal Fastmail",
          server_url: "https://caldav.fastmail.com/dav/calendars",
          username: "reader@example.test",
          password: "app-password"
        })
      )
      expect(mocks.verifyCalDavAccount).toHaveBeenCalledWith(4, {
        password: "app-password"
      })
    })
    expect(mocks.discoverExternalCalendars).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ password: expect.any(String) })
    )
  })

  it("discovers remote calendars, binds one with sync windows, queues sync, and confirms delete", async () => {
    const user = userEvent.setup()
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true)
    renderSettings()

    const card = await screen.findByRole("region", { name: "Fastmail" })
    expect(within(card).getByText("Sync error: provider down")).toBeTruthy()
    expect(within(card).getByText("bounded polling")).toBeTruthy()

    await user.click(within(card).getByRole("button", { name: "Discover calendars" }))
    const discovery = await screen.findByRole("region", { name: "Discovered calendars" })
    expect(within(discovery).getByText("Work")).toBeTruthy()
    expect(within(discovery).getByText("Bounded polling")).toBeTruthy()

    await user.clear(within(discovery).getByRole("spinbutton", { name: "Lookback days" }))
    await user.type(within(discovery).getByRole("spinbutton", { name: "Lookback days" }), "30")
    await user.clear(within(discovery).getByRole("spinbutton", { name: "Lookahead days" }))
    await user.type(within(discovery).getByRole("spinbutton", { name: "Lookahead days" }), "120")
    await user.click(within(discovery).getByRole("button", { name: "Bind Work" }))

    await waitFor(() => {
      expect(mocks.createExternalCalendarBinding).toHaveBeenCalledWith(
        expect.objectContaining({
          account_id: 3,
          calendar_id: 7,
          remote_calendar_id: "https://caldav.fastmail.com/calendars/user/work/",
          lookback_days: 30,
          lookahead_days: 120
        })
      )
    })

    await user.click(within(card).getByRole("button", { name: "Sync now" }))
    await waitFor(() => {
      expect(mocks.triggerCalendarSync).toHaveBeenCalledWith(10, { reason: "manual" })
    })

    await user.click(within(card).getByRole("button", { name: "Delete account" }))
    await waitFor(() => {
      expect(confirmSpy).toHaveBeenCalled()
      expect(mocks.deleteCalDavAccount).toHaveBeenCalledWith(3)
    })
    confirmSpy.mockRestore()
  })
})
