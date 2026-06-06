import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  createCalendarItem: vi.fn(),
  updateCalendarItem: vi.fn(),
  deleteCalendarItem: vi.fn(),
  createCalendarAnnotation: vi.fn(),
  createCalendarLink: vi.fn(),
  copyCalendarItemIntoTldw: vi.fn()
}))

vi.mock("@/services/calendar", () => ({
  createCalendarItem: (...args: unknown[]) => mocks.createCalendarItem(...args),
  updateCalendarItem: (...args: unknown[]) => mocks.updateCalendarItem(...args),
  deleteCalendarItem: (...args: unknown[]) => mocks.deleteCalendarItem(...args),
  createCalendarAnnotation: (...args: unknown[]) => mocks.createCalendarAnnotation(...args),
  createCalendarLink: (...args: unknown[]) => mocks.createCalendarLink(...args),
  copyCalendarItemIntoTldw: (...args: unknown[]) => mocks.copyCalendarItemIntoTldw(...args)
}))

import { CalendarItemDrawer } from "../CalendarItemDrawer"

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
  }
]

const providerItem = {
  id: "provider-42",
  calendar_item_id: 42,
  calendar_id: 1,
  title: "External sync review",
  source_owner: "provider",
  start_at: "2026-06-05T16:00:00Z",
  end_at: "2026-06-05T16:30:00Z",
  due_at: null,
  all_day: false,
  status: "confirmed",
  read_only_reason: "Managed by CalDAV",
  link: null,
  metadata: { provider: "caldav" }
}

const linkedProjection = {
  id: "watchlist-job:17",
  calendar_item_id: null,
  calendar_id: null,
  title: "Daily source digest",
  source_owner: "linked_projection",
  start_at: "2026-06-07T14:00:00Z",
  end_at: null,
  due_at: null,
  all_day: false,
  status: "scheduled",
  read_only_reason: "Managed by Watchlists",
  link: {
    target_type: "watchlist_job",
    target_id: "17",
    label: "Manage in Watchlists",
    url: "/watchlists?tab=jobs",
    metadata: {}
  },
  metadata: {}
}

const localItem = {
  id: "local-7",
  calendar_item_id: 7,
  calendar_id: 1,
  title: "Draft notes",
  source_owner: "tldw",
  start_at: "2026-06-05T09:00:00Z",
  end_at: "2026-06-05T10:00:00Z",
  due_at: null,
  all_day: false,
  status: "confirmed",
  link: {
    target_type: "media",
    target_id: "abc",
    label: "Source clip",
    url: "/media/abc",
    metadata: {}
  },
  metadata: { local_tags: ["draft"] }
}

const renderDrawer = (
  item: React.ComponentProps<typeof CalendarItemDrawer>["item"],
  extraProps: Partial<React.ComponentProps<typeof CalendarItemDrawer>> = {}
) =>
  render(
    <CalendarItemDrawer
      open
      item={item}
      calendars={calendars}
      onClose={vi.fn()}
      onSaved={vi.fn()}
      {...extraProps}
    />
  )

describe("CalendarItemDrawer", () => {
  beforeEach(() => {
    for (const mock of Object.values(mocks)) {
      mock.mockReset()
    }
  })

  it("disables provider-owned field editing and offers copy into tldw", async () => {
    const user = userEvent.setup()
    mocks.copyCalendarItemIntoTldw.mockResolvedValue({
      id: 99,
      calendar_id: 1,
      kind: "event",
      source_owner: "tldw",
      provider_owned: false,
      title: "External sync review",
      all_day: false,
      status: "confirmed",
      local_tags: [],
      metadata: {},
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })

    renderDrawer(providerItem)

    const title = await screen.findByDisplayValue("External sync review")
    expect(title).toHaveProperty("disabled", true)
    expectPresent(screen.getByText("Managed by CalDAV"))
    expect(screen.queryByRole("button", { name: "Save item" })).toBeNull()

    await user.click(screen.getByRole("button", { name: "Copy into tldw" }))

    await waitFor(() => {
      expect(mocks.copyCalendarItemIntoTldw).toHaveBeenCalledWith(42, {
        target_calendar_id: 1
      })
    })
  })

  it("opens linked projection manage URL and hides local edit controls", async () => {
    renderDrawer(linkedProjection)

    expectPresent(await screen.findByText("Managed by Watchlists"))
    expect(screen.getByRole("link", { name: "Manage in Watchlists" }).getAttribute("href")).toBe(
      "/watchlists?tab=jobs"
    )
    expect(screen.queryByRole("button", { name: "Save item" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Delete item" })).toBeNull()
    expect(screen.queryByLabelText("Title")).toBeNull()
  })

  it("saves edits, annotations, and links for local items", async () => {
    const user = userEvent.setup()
    mocks.updateCalendarItem.mockResolvedValue({
      id: 7,
      calendar_id: 1,
      kind: "event",
      source_owner: "tldw",
      provider_owned: false,
      title: "Draft notes updated",
      all_day: false,
      status: "confirmed",
      local_tags: ["draft", "review"],
      metadata: {},
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })
    mocks.createCalendarAnnotation.mockResolvedValue({
      id: 1,
      calendar_item_id: 7,
      author_user_id: 1,
      body: "Check citations",
      tags: [],
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })
    mocks.createCalendarLink.mockResolvedValue({
      id: 2,
      calendar_item_id: 7,
      target_type: "note",
      target_id: "note-9",
      label: "Research note",
      url: "/notes/note-9",
      metadata: {},
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })

    renderDrawer(localItem)

    const title = await screen.findByRole("textbox", { name: "Title" })
    await user.clear(title)
    await user.type(title, "Draft notes updated")
    await user.type(screen.getByRole("textbox", { name: "Tags" }), ", review")
    await user.type(screen.getByRole("textbox", { name: "Annotation" }), "Check citations")
    await user.type(screen.getByRole("textbox", { name: "Link label" }), "Research note")
    await user.type(screen.getByRole("textbox", { name: "Link URL" }), "/notes/note-9")
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => {
      expect(mocks.updateCalendarItem).toHaveBeenCalledWith(
        7,
        expect.objectContaining({
          title: "Draft notes updated",
          local_tags: ["draft", "review"],
          source_owner: "tldw",
          provider_owned: false
        })
      )
      expect(mocks.createCalendarAnnotation).toHaveBeenCalledWith(7, {
        body: "Check citations",
        tags: []
      })
      expect(mocks.createCalendarLink).toHaveBeenCalledWith(
        7,
        expect.objectContaining({
          label: "Research note",
          url: "/notes/note-9"
        })
      )
    })
  })
})
