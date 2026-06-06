import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  useCanonicalConnectionConfig: vi.fn(),
  listCalendars: vi.fn(),
  getCalendarAgenda: vi.fn(),
  getCalendarWeek: vi.fn(),
  createCalendarItem: vi.fn(),
  updateCalendarItem: vi.fn(),
  deleteCalendarItem: vi.fn(),
  createCalendarAnnotation: vi.fn(),
  createCalendarLink: vi.fn(),
  copyCalendarItemIntoTldw: vi.fn(),
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

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: (...args: unknown[]) =>
    mocks.useCanonicalConnectionConfig(...args)
}))

vi.mock("@/services/calendar", () => ({
  listCalendars: (...args: unknown[]) => mocks.listCalendars(...args),
  getCalendarAgenda: (...args: unknown[]) => mocks.getCalendarAgenda(...args),
  getCalendarWeek: (...args: unknown[]) => mocks.getCalendarWeek(...args),
  createCalendarItem: (...args: unknown[]) => mocks.createCalendarItem(...args),
  updateCalendarItem: (...args: unknown[]) => mocks.updateCalendarItem(...args),
  deleteCalendarItem: (...args: unknown[]) => mocks.deleteCalendarItem(...args),
  createCalendarAnnotation: (...args: unknown[]) => mocks.createCalendarAnnotation(...args),
  createCalendarLink: (...args: unknown[]) => mocks.createCalendarLink(...args),
  copyCalendarItemIntoTldw: (...args: unknown[]) => mocks.copyCalendarItemIntoTldw(...args),
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

import { CalendarPage } from "../CalendarPage"

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const expectPresent = (element: Element | null): Element => {
  expect(element).not.toBeNull()
  return element as Element
}

const calendars = [
  {
    id: 1,
    tenant_id: "default",
    owner_user_id: 1,
    org_id: null,
    name: "Research",
    color: "#2563eb",
    timezone: "UTC",
    visibility: "private",
    created_at: "2026-06-01T00:00:00Z",
    updated_at: "2026-06-01T00:00:00Z"
  },
  {
    id: 2,
    tenant_id: "default",
    owner_user_id: 1,
    org_id: 8,
    name: "Lab",
    color: "#047857",
    timezone: "UTC",
    visibility: "org",
    created_at: "2026-06-01T00:00:00Z",
    updated_at: "2026-06-01T00:00:00Z"
  }
]

const agendaItems = [
  {
    id: "item-1",
    calendar_item_id: 1,
    calendar_id: 1,
    kind: "event",
    title: "Paper review",
    source_owner: "tldw",
    start_at: "2026-06-05T09:00:00Z",
    end_at: "2026-06-05T10:00:00Z",
    due_at: null,
    all_day: false,
    status: "confirmed",
    local_tags: [],
    metadata: {}
  },
  {
    id: "item-2",
    calendar_item_id: 2,
    calendar_id: 1,
    kind: "todo",
    title: "Tag figures",
    source_owner: "tldw",
    start_at: "2026-06-05T17:00:00Z",
    end_at: null,
    due_at: "2026-06-05T17:00:00Z",
    all_day: false,
    status: "needs_action",
    local_tags: ["figures"],
    metadata: {}
  },
  {
    id: "item-3",
    calendar_item_id: 3,
    calendar_id: 2,
    kind: "event",
    title: "Provider planning sync",
    source_owner: "provider",
    start_at: "2026-06-06T16:00:00Z",
    end_at: "2026-06-06T16:30:00Z",
    due_at: null,
    all_day: false,
    status: "confirmed",
    read_only_reason: "Managed by CalDAV",
    local_tags: [],
    metadata: { provider: "caldav" }
  },
  {
    id: "watchlist-job:17",
    calendar_item_id: null,
    calendar_id: null,
    kind: "event",
    title: "Daily source digest",
    source_owner: "linked_projection",
    start_at: "2026-06-07T14:00:00Z",
    end_at: null,
    due_at: null,
    all_day: false,
    status: "scheduled",
    read_only_reason: "Managed by Watchlists",
    local_tags: [],
    link: {
      target_type: "watchlist_job",
      target_id: "17",
      label: "Manage in Watchlists",
      url: "/watchlists?tab=jobs",
      metadata: {}
    },
    metadata: {}
  }
]

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

  return render(
    <MemoryRouter initialEntries={["/calendar"]}>
      <QueryClientProvider client={queryClient}>
        <CalendarPage />
      </QueryClientProvider>
    </MemoryRouter>
  )
}

describe("CalendarPage", () => {
  beforeEach(() => {
    for (const mock of Object.values(mocks)) {
      mock.mockReset()
    }
    mocks.useCanonicalConnectionConfig.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      },
      loading: false
    })
    mocks.listCalendars.mockResolvedValue({ items: calendars, total: calendars.length })
    mocks.getCalendarAgenda.mockResolvedValue({
      start_at: "2026-06-05T00:00:00.000Z",
      end_at: "2026-06-12T00:00:00.000Z",
      items: agendaItems
    })
    mocks.getCalendarWeek.mockResolvedValue({
      start_at: "2026-06-01T00:00:00.000Z",
      end_at: "2026-06-08T00:00:00.000Z",
      items: agendaItems
    })
    mocks.listCalDavAccounts.mockResolvedValue({ items: [], total: 0 })
    mocks.listExternalCalendarBindings.mockResolvedValue({ items: [], total: 0 })
    mocks.createCalDavAccount.mockResolvedValue({ id: 3, provider: "caldav", display_name: "Fastmail" })
    mocks.verifyCalDavAccount.mockResolvedValue({ account_id: 3, verified: true })
    mocks.discoverExternalCalendars.mockResolvedValue({ items: [] })
    mocks.createExternalCalendarBinding.mockResolvedValue({ id: 8 })
    mocks.triggerCalendarSync.mockResolvedValue({ binding_id: 8, queued: true, status: "queued" })
    mocks.revokeCalDavAccount.mockResolvedValue({ revoked: true })
    mocks.deleteCalDavAccount.mockResolvedValue({ deleted: true })
    fetchMock.mockReset()
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/calendar/calendars": {}
        }
      })
    })
  })

  it("loads calendars and agenda data into the workspace", async () => {
    renderPage()

    expectPresent(await screen.findByRole("heading", { name: "Calendar" }))
    expectPresent(await screen.findByText("Research"))
    expectPresent(await screen.findByText("Paper review"))

    expect(mocks.listCalendars).toHaveBeenCalledTimes(1)
    expect(mocks.getCalendarAgenda).toHaveBeenCalledWith(
      expect.objectContaining({
        start_at: expect.any(String),
        end_at: expect.any(String)
      })
    )
  })

  it("shows RecoveryCallout when the connected backend does not advertise calendar support", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ paths: {} })
    })

    renderPage()

    expectPresent(
      await screen.findByRole("heading", {
        name: "Calendar is unavailable on this server"
      })
    )
    expect(screen.getByLabelText("Diagnostics").textContent).toContain(
      "/api/v1/calendar/calendars"
    )
    expect(mocks.listCalendars).not.toHaveBeenCalled()
  })

  it("renders local, provider-owned, and linked projection ownership labels distinctly", async () => {
    renderPage()

    expectPresent(await screen.findByText("Paper review"))
    expectPresent(screen.getByText("Tag figures"))
    expectPresent(screen.getByText("Provider planning sync"))
    expectPresent(screen.getByText("Daily source digest"))

    const agenda = screen.getByRole("region", { name: "Agenda" })
    expect(within(agenda).getAllByText("Local")).toHaveLength(2)
    expectPresent(within(agenda).getByText("Provider"))
    expectPresent(within(agenda).getByText("Linked"))
    expectPresent(within(agenda).getByText("Manage in Watchlists"))
  })

  it("uses backend item kind for todo filtering when due-only todos are normalized into start_at", async () => {
    const user = userEvent.setup()
    renderPage()

    expectPresent(await screen.findByText("Tag figures"))
    await user.click(screen.getByLabelText("Events"))

    expect(screen.queryByText("Paper review")).toBeNull()
    expectPresent(screen.getByText("Tag figures"))
  })

  it("allows clearing every calendar filter without falling back to all calendars", async () => {
    const user = userEvent.setup()
    renderPage()

    expectPresent(await screen.findByText("Paper review"))
    await user.click(screen.getByLabelText("Research"))
    await user.click(screen.getByLabelText("Lab"))

    expect(screen.queryByText("Paper review")).toBeNull()
    expect(screen.queryByText("Provider planning sync")).toBeNull()
  })

  it("shows a degraded RecoveryCallout for partial calendar data", async () => {
    mocks.getCalendarAgenda.mockResolvedValueOnce({
      start_at: "2026-06-05T00:00:00.000Z",
      end_at: "2026-06-12T00:00:00.000Z",
      items: agendaItems,
      partial: true,
      warnings: ["CalDAV binding 7 fell back to bounded polling"]
    })

    renderPage()

    expectPresent(
      await screen.findByRole("heading", {
        name: "Calendar data is partially available"
      })
    )
    expect(screen.getByLabelText("Diagnostics").textContent).toContain(
      "CalDAV binding 7 fell back to bounded polling"
    )
  })

  it("creates a local event through the drawer form", async () => {
    const user = userEvent.setup()
    mocks.createCalendarItem.mockResolvedValue({
      id: 9,
      calendar_id: 1,
      kind: "event",
      source_owner: "tldw",
      provider_owned: false,
      title: "Lab review",
      all_day: false,
      status: "confirmed",
      local_tags: [],
      metadata: {},
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })

    renderPage()

    await user.click(await screen.findByRole("button", { name: "New item" }))
    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Lab review")
    await user.type(screen.getByRole("textbox", { name: "Start" }), "2026-06-05T11:00")
    await user.type(screen.getByRole("textbox", { name: "End" }), "2026-06-05T12:00")
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => {
      expect(mocks.createCalendarItem).toHaveBeenCalledWith(
        expect.objectContaining({
          calendar_id: 1,
          kind: "event",
          title: "Lab review",
          start_at: "2026-06-05T11:00",
          end_at: "2026-06-05T12:00"
        })
      )
    })
  })

  it("creates a local todo through the drawer form", async () => {
    const user = userEvent.setup()
    mocks.createCalendarItem.mockResolvedValue({
      id: 10,
      calendar_id: 1,
      kind: "todo",
      source_owner: "tldw",
      provider_owned: false,
      title: "Send notes",
      all_day: false,
      status: "needs_action",
      local_tags: [],
      metadata: {},
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })

    renderPage()

    await user.click(await screen.findByRole("button", { name: "New item" }))
    await user.click(screen.getByRole("radio", { name: "Todo" }))
    await user.type(await screen.findByRole("textbox", { name: "Title" }), "Send notes")
    await user.type(screen.getByRole("textbox", { name: "Due" }), "2026-06-05T17:00")
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => {
      expect(mocks.createCalendarItem).toHaveBeenCalledWith(
        expect.objectContaining({
          calendar_id: 1,
          kind: "todo",
          title: "Send notes",
          due_at: "2026-06-05T17:00"
        })
      )
    })
  })
})
