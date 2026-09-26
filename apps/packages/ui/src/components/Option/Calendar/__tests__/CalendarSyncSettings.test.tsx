import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  messageError: vi.fn(),
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

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: { ...actual.message, error: mocks.messageError }
  }
})

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

const renderSettings = (onChanged = vi.fn()) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <CalendarSyncSettings calendars={calendars} onChanged={onChanged} />
    </QueryClientProvider>
  )
}

const setupDraft = {
  "Account name": "Personal Fastmail",
  "Server URL": "https://caldav.fastmail.com/dav/calendars",
  Username: "reader@example.test",
  Password: " app-password "
}

const fillAccountDraft = async (
  user: ReturnType<typeof userEvent.setup>,
  draft = setupDraft
) => {
  await user.click(await screen.findByRole("button", { name: "Add CalDAV account" }))
  for (const [label, value] of Object.entries(draft)) {
    fireEvent.change(screen.getByLabelText(label), { target: { value } })
  }
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

  it("adds a CalDAV account with atomic verification in the create request", async () => {
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
          password: "app-password",
          verify_before_create: true
        })
      )
    })
    await waitFor(() => expect(screen.queryByLabelText("Password")).not.toBeInTheDocument())
    expect(mocks.discoverExternalCalendars).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ password: expect.any(String) })
    )
  })

  it("does not call post-create verification or resend setup credentials", async () => {
    const user = userEvent.setup()
    const onChanged = vi.fn()
    renderSettings(onChanged)
    await fillAccountDraft(user)
    await user.click(screen.getByRole("button", { name: "Save and verify account" }))

    await waitFor(() => expect(onChanged).toHaveBeenCalledTimes(1))
    expect(mocks.createCalDavAccount).toHaveBeenCalledTimes(1)
    expect(mocks.verifyCalDavAccount).not.toHaveBeenCalled()
    expect(mocks.createCalDavAccount).toHaveBeenCalledWith({
      display_name: setupDraft["Account name"],
      server_url: setupDraft["Server URL"],
      username: setupDraft.Username,
      password: setupDraft.Password,
      verify_before_create: true
    })
  })

  it.each([
    ["Account name", ""],
    ["Account name", "   "],
    ["Server URL", ""],
    ["Server URL", "   "],
    ["Username", ""],
    ["Username", "   "],
    ["Password", ""],
    ["Password", "   "]
  ])("rejects incomplete setup when %s is '%s' without creating an account", async (label, value) => {
    const user = userEvent.setup()
    const onChanged = vi.fn()
    renderSettings(onChanged)
    const draft = { ...setupDraft, [label]: value }
    await fillAccountDraft(user, draft)
    await user.click(screen.getByRole("button", { name: "Save and verify account" }))

    await waitFor(() => expect(mocks.messageError).toHaveBeenCalledWith(
      "Account name, server URL, username, and password are required"
    ))
    expect(mocks.createCalDavAccount).not.toHaveBeenCalled()
    expect(mocks.verifyCalDavAccount).not.toHaveBeenCalled()
    expect(onChanged).not.toHaveBeenCalled()
    for (const [field, fieldValue] of Object.entries(draft)) {
      expect(screen.getByLabelText(field)).toHaveValue(fieldValue)
    }
  })

  it.each(["Unable to verify CalDAV account", "Unable to connect to CalDAV server"])(
    "retains the draft and drawer after failed creation: %s",
    async (errorMessage) => {
      const user = userEvent.setup()
      const onChanged = vi.fn()
      mocks.listCalDavAccounts.mockResolvedValue({ items: [], total: 0 })
      mocks.createCalDavAccount.mockRejectedValueOnce(new Error(errorMessage))
      renderSettings(onChanged)
      await fillAccountDraft(user)
      await user.click(screen.getByRole("button", { name: "Save and verify account" }))

      await waitFor(() => expect(mocks.messageError).toHaveBeenCalledWith(errorMessage))
      for (const [label, value] of Object.entries(setupDraft)) {
        expect(screen.getByLabelText(label)).toHaveValue(value)
      }
      expect(screen.queryByRole("region", { name: "Personal Fastmail" })).not.toBeInTheDocument()
      expect(mocks.listCalDavAccounts).toHaveBeenCalledTimes(1)
      expect(onChanged).not.toHaveBeenCalled()
      expect(mocks.verifyCalDavAccount).not.toHaveBeenCalled()
      expect(mocks.createCalDavAccount).toHaveBeenCalledWith({
        display_name: setupDraft["Account name"],
        server_url: setupDraft["Server URL"],
        username: setupDraft.Username,
        password: setupDraft.Password,
        verify_before_create: true
      })

      await user.click(screen.getByRole("button", { name: "Save and verify account" }))
      await waitFor(() => expect(onChanged).toHaveBeenCalledTimes(1))
      expect(mocks.createCalDavAccount).toHaveBeenCalledTimes(2)
      expect(mocks.verifyCalDavAccount).not.toHaveBeenCalled()
    }
  )

  it.each([true, false])("submits integer sync days after decimal input (blur=%s)", async (blur) => {
    const user = userEvent.setup()
    renderSettings()
    await user.click(await screen.findByRole("button", { name: "Discover calendars" }))
    const discovery = await screen.findByRole("region", { name: "Discovered calendars" })
    const lookback = within(discovery).getByRole("spinbutton", { name: "Lookback days" })
    const lookahead = within(discovery).getByRole("spinbutton", { name: "Lookahead days" })
    await user.clear(lookback)
    await user.type(lookback, "30.7")
    await user.clear(lookahead)
    await user.type(lookahead, "120.2")
    const bind = within(discovery).getByRole("button", { name: "Bind Work" })
    if (blur) {
      await user.tab()
      expect(lookback).toHaveValue("31")
      expect(lookahead).toHaveValue("120")
      await user.click(bind)
    } else {
      // Submit while a decimal is still being edited, without relying on blur rounding.
      fireEvent.click(bind)
    }

    await waitFor(() => expect(mocks.createExternalCalendarBinding).toHaveBeenCalledWith(
      expect.objectContaining({ lookback_days: 31, lookahead_days: 120 })
    ))
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
          sync_interval_minutes: 60,
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
