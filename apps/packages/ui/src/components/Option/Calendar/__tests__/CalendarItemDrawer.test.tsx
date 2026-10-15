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
  listCalendarLinks: vi.fn(),
  deleteCalendarLink: vi.fn(),
  updateCalendarLocalTags: vi.fn(),
  copyCalendarItemIntoTldw: vi.fn()
}))

vi.mock("@/services/calendar", () => ({
  createCalendarItem: (...args: unknown[]) => mocks.createCalendarItem(...args),
  updateCalendarItem: (...args: unknown[]) => mocks.updateCalendarItem(...args),
  deleteCalendarItem: (...args: unknown[]) => mocks.deleteCalendarItem(...args),
  createCalendarAnnotation: (...args: unknown[]) => mocks.createCalendarAnnotation(...args),
  createCalendarLink: (...args: unknown[]) => mocks.createCalendarLink(...args),
  listCalendarLinks: (...args: unknown[]) => mocks.listCalendarLinks(...args),
  deleteCalendarLink: (...args: unknown[]) => mocks.deleteCalendarLink(...args),
  updateCalendarLocalTags: (...args: unknown[]) => mocks.updateCalendarLocalTags(...args),
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
  kind: "event",
  title: "External sync review",
  source_owner: "provider",
  start_at: "2026-06-05T16:00:00Z",
  end_at: "2026-06-05T16:30:00Z",
  due_at: null,
  all_day: false,
  status: "confirmed",
  read_only_reason: "Managed by CalDAV",
  local_tags: ["remote"],
  link: null,
  metadata: { provider: "caldav" }
}

const linkedProjection = {
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

const localItem = {
  id: "local-7",
  calendar_item_id: 7,
  calendar_id: 1,
  kind: "event",
  title: "Draft notes",
  source_owner: "tldw",
  start_at: "2026-06-05T09:00:00Z",
  end_at: "2026-06-05T10:00:00Z",
  due_at: null,
  all_day: false,
  status: "confirmed",
  local_tags: ["draft"],
  link: {
    target_type: "media",
    target_id: "abc",
    label: "Source clip",
    url: "/media/abc",
    metadata: {}
  },
  metadata: {}
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
  it("lets the item timezone resolve edited dates across DST instead of reusing the old offset", async () => {
    const user = userEvent.setup()
    renderDrawer({ ...localItem,
      start_at: "2026-10-31T09:00:00-07:00", end_at: "2026-10-31T10:00:00-07:00",
      metadata: { timezone: "America/Los_Angeles" }
    })
    for (const [label, value] of [["Start", "2026-11-02T09:00"], ["End", "2026-11-02T10:00"]]) {
      await user.clear(screen.getByRole("textbox", { name: label }))
      await user.type(screen.getByRole("textbox", { name: label }), value)
    }
    await user.click(screen.getByRole("button", { name: "Save item" }))
    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalled())
    expect(mocks.updateCalendarItem.mock.calls[0][1]).toMatchObject({
      start_at: "2026-11-02T09:00", end_at: "2026-11-02T10:00"
    })
  })
  beforeEach(() => {
    for (const mock of Object.values(mocks)) {
      mock.mockReset()
    }
    mocks.listCalendarLinks.mockResolvedValue({ items: [], total: 0 })
  })

  it.each([
    ["HTTPS", "https://example.test/digest", "https://example.test/digest"],
    ["relative", "/watchlists?tab=jobs", "/watchlists?tab=jobs"],
    ["normalized HTTPS", " \thttps://example.test/digest\n", "https://example.test/digest"]
  ])("keeps safe %s projection navigation", (_case, url, expected) => {
    renderDrawer({ ...linkedProjection, link: { ...linkedProjection.link, url } })
    expect(screen.getByRole("link", { name: "Manage in Watchlists" }).getAttribute("href")).toBe(expected)
  })

  it.each([
    "javascript:alert(1)",
    "java\tscript:alert(1)",
    "java\nscript:alert(1)",
    "java\rscript:alert(1)",
    " \u0000JaVa\u007fScRiPt:alert(1)"
  ])("does not expose unsafe projection URL %j as a navigation target", (url) => {
    renderDrawer({ ...linkedProjection, link: { ...linkedProjection.link, url } })
    expect(screen.queryByRole("link", { name: "Manage in Watchlists" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Manage in Watchlists" })).toBeNull()
  })

  it.each([
    { kind: "event", field: "Start", blank: "" },
    { kind: "event", field: "Start", blank: "   " },
    { kind: "todo", field: "Due", blank: "" },
    { kind: "todo", field: "Due", blank: "   " }
  ])("retains a new $kind draft and rejects blank $field (%j) before any mutation", async ({ kind, field, blank }) => {
    const user = userEvent.setup()
    const onSaved = vi.fn()
    const onClose = vi.fn()
    renderDrawer(null, { onSaved, onClose })
    if (kind === "todo") await user.click(screen.getByRole("radio", { name: "Todo" }))
    await user.type(screen.getByRole("textbox", { name: "Title" }), "Unscheduled draft")
    await user.type(screen.getByRole("textbox", { name: "Description" }), "Keep these notes")
    await user.type(screen.getByRole("textbox", { name: "Tags" }), "research, draft")
    if (kind === "event") await user.type(screen.getByRole("textbox", { name: "End" }), "2026-06-05T10:00")
    const control = screen.getByRole("textbox", { name: field })
    if (blank) await user.type(control, blank)
    await user.click(screen.getByRole("button", { name: "Save item" }))

    for (const name of ["createCalendarItem", "updateCalendarItem", "updateCalendarLocalTags", "createCalendarAnnotation", "createCalendarLink"] as const) {
      expect(mocks[name]).not.toHaveBeenCalled()
    }
    expect(control.getAttribute("aria-invalid")).toBe("true")
    expect(control.getAttribute("aria-describedby")).toBe(screen.getByRole("alert").id)
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveProperty("value", "Unscheduled draft")
    expect(screen.getByRole("textbox", { name: "Description" })).toHaveProperty("value", "Keep these notes")
    expect(screen.getByRole("textbox", { name: "Tags" })).toHaveProperty("value", "research, draft")
    expect(onSaved).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()

    await user.clear(control)
    await user.type(control, "2026-06-05T09:00")
    expect(control.getAttribute("aria-invalid")).not.toBe("true")
    await user.click(screen.getByRole("button", { name: "Save item" }))
    await waitFor(() => expect(mocks.createCalendarItem).toHaveBeenCalledTimes(1))
    expect(mocks.createCalendarItem).toHaveBeenCalledWith(expect.objectContaining({
      kind, title: "Unscheduled draft", description: "Keep these notes", local_tags: ["research", "draft"],
      start_at: kind === "event" ? "2026-06-05T09:00" : null,
      due_at: kind === "todo" ? "2026-06-05T09:00" : null
    }))
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1))
  })

  it.each(["event", "todo"])("rejects clearing the only schedule of an existing %s without losing local context", async (kind) => {
    const user = userEvent.setup()
    const onSaved = vi.fn()
    const onClose = vi.fn()
    renderDrawer({ ...localItem, kind,
      start_at: kind === "event" ? localItem.start_at : null,
      due_at: kind === "todo" ? localItem.start_at : null,
      end_at: kind === "event" ? localItem.end_at : null
    }, { onSaved, onClose })
    const control = screen.getByRole("textbox", { name: kind === "event" ? "Start" : "Due" })
    await user.clear(control)
    await user.type(screen.getByRole("textbox", { name: "Title" }), " revised")
    await user.type(screen.getByRole("textbox", { name: "Annotation" }), "Keep this annotation")
    await user.click(screen.getByRole("button", { name: "Save item" }))
    expect(mocks.updateCalendarItem).not.toHaveBeenCalled()
    expect(mocks.createCalendarAnnotation).not.toHaveBeenCalled()
    expect(control.getAttribute("aria-invalid")).toBe("true")
    expect(screen.getByRole("textbox", { name: "Annotation" })).toHaveProperty("value", "Keep this annotation")
    expect(onSaved).not.toHaveBeenCalled()
    expect(onClose).not.toHaveBeenCalled()
  })

  it.each([{}, { item_start_at: null }])("does not mistake a due-only view's synthesized start for a persisted start (%j)", async (metadata) => {
    const user = userEvent.setup()
    renderDrawer({ ...localItem, kind: "todo", start_at: localItem.start_at,
      due_at: localItem.start_at, end_at: null, metadata })
    await user.clear(screen.getByRole("textbox", { name: "Due" }))
    await user.click(screen.getByRole("button", { name: "Save item" }))
    expect(mocks.updateCalendarItem).not.toHaveBeenCalled()
    expect(screen.getByRole("textbox", { name: "Due" }).getAttribute("aria-invalid")).toBe("true")
  })

  it("preserves the persisted todo start after a temporary event start edit", async () => {
    const user = userEvent.setup()
    renderDrawer({ ...localItem, kind: "todo", end_at: null,
      metadata: { item_start_at: localItem.start_at } })
    await user.click(screen.getByRole("radio", { name: "Event" }))
    const start = screen.getByRole("textbox", { name: "Start" })
    await user.clear(start)
    await user.type(start, "2026-06-06T12:00")
    await user.click(screen.getByRole("radio", { name: "Todo" }))
    await user.click(screen.getByRole("button", { name: "Save item" }))
    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalledTimes(1))
    expect(mocks.updateCalendarItem.mock.calls[0][1]).not.toHaveProperty("start_at")
    expect(mocks.updateCalendarItem.mock.calls[0][1]).not.toHaveProperty("kind")
  })

  it("can clear a todo due time when canonical metadata proves an equal persisted start", async () => {
    const user = userEvent.setup()
    renderDrawer({ ...localItem, kind: "todo", due_at: localItem.start_at, end_at: null,
      metadata: { item_start_at: localItem.start_at } })
    await user.clear(screen.getByRole("textbox", { name: "Due" }))
    await user.click(screen.getByRole("button", { name: "Save item" }))
    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalledWith(7, expect.objectContaining({ due_at: null })))
    expect(mocks.updateCalendarItem.mock.calls[0][1]).not.toHaveProperty("start_at")
  })

  it("does not count an event's discarded start as a schedule when converting it to a todo", async () => {
    const user = userEvent.setup()
    renderDrawer(localItem)
    await user.click(screen.getByRole("radio", { name: "Todo" }))
    await user.click(screen.getByRole("button", { name: "Save item" }))
    expect(mocks.updateCalendarItem).not.toHaveBeenCalled()
    expect(screen.getByRole("textbox", { name: "Due" }).getAttribute("aria-invalid")).toBe("true")
    await user.click(screen.getByRole("radio", { name: "Event" }))
    expect(screen.queryByRole("alert")).toBeNull()
    await user.click(screen.getByRole("button", { name: "Save item" }))
    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalledTimes(1))
    expect(mocks.updateCalendarItem.mock.calls[0][1]).not.toHaveProperty("start_at")
  })

  it.each([
    { kind: "event", recurrence_id: 3, occurrence_index: 0 },
    { kind: "todo", recurrence_id: 3, occurrence_index: 2 },
    { kind: "todo" }
  ])("keeps text-only edits working with an unchanged missing schedule (%j)", async (extra) => {
    const user = userEvent.setup()
    renderDrawer({ ...localItem, ...extra, start_at: null, end_at: null, due_at: null })
    await user.type(screen.getByRole("textbox", { name: "Title" }), " revised")
    await user.click(screen.getByRole("button", { name: "Save item" }))
    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalledWith(
      7, expect.objectContaining({ title: "Draft notes revised" })
    ))
    for (const field of ["kind", "start_at", "end_at", "due_at", "recurrence"]) {
      expect(mocks.updateCalendarItem.mock.calls[0][1]).not.toHaveProperty(field)
    }
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it.each(["provider", "tldw"])("does not require a schedule for readonly %s local-context edits", async (source_owner) => {
    const user = userEvent.setup()
    renderDrawer({ ...providerItem, source_owner, start_at: null, end_at: null, due_at: null })
    expect(screen.getByRole("textbox", { name: "Start" })).toHaveProperty("disabled", true)
    await user.type(screen.getByRole("textbox", { name: "Annotation" }), "Local notes")
    await user.click(screen.getByRole("button", { name: "Save context" }))
    await waitFor(() => expect(mocks.createCalendarAnnotation).toHaveBeenCalledWith(42, { body: "Local notes", tags: [] }))
    expect(mocks.updateCalendarItem).not.toHaveBeenCalled()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("disables calendar moves for existing items but permits selection for new items", () => {
    const { unmount } = renderDrawer(localItem)
    expect(screen.getByRole("combobox", { name: "Calendar" })).toHaveProperty("disabled", true)
    unmount()
    renderDrawer(null)
    expect(screen.getByRole("combobox", { name: "Calendar" })).toHaveProperty("disabled", false)
  })

  it.each([
    { kind: "event", timezone: "America/Los_Angeles", changeSelection: false },
    { kind: "todo", timezone: "Asia/Kolkata", changeSelection: false },
    { kind: "event", timezone: "America/Los_Angeles", changeSelection: true },
    { kind: "todo", timezone: "Asia/Kolkata", changeSelection: true }
  ])("creates $kind in the selected calendar timezone $timezone (changeSelection=$changeSelection)", async ({ kind, timezone, changeSelection }) => {
    const user = userEvent.setup()
    const selectedCalendar = { ...calendars[0], id: 2, name: "Local research", timezone }
    renderDrawer(null, {
      calendars: changeSelection ? [...calendars, selectedCalendar] : [selectedCalendar]
    })
    if (changeSelection) {
      await user.click(screen.getByRole("combobox", { name: "Calendar" }))
      await user.click(await screen.findByText("Local research"))
    }
    await user.type(screen.getByRole("textbox", { name: "Title" }), "Local appointment")
    if (kind === "todo") {
      await user.click(screen.getByRole("radio", { name: "Todo" }))
      await user.type(screen.getByRole("textbox", { name: "Due" }), "2026-11-02T17:00")
    } else {
      await user.type(screen.getByRole("textbox", { name: "Start" }), "2026-11-02T09:00")
      await user.type(screen.getByRole("textbox", { name: "End" }), "2026-11-02T10:00")
    }
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => expect(mocks.createCalendarItem).toHaveBeenCalledWith(
      expect.objectContaining({
        calendar_id: 2,
        kind,
        timezone,
        start_at: kind === "event" ? "2026-11-02T09:00" : null,
        end_at: kind === "event" ? "2026-11-02T10:00" : null,
        due_at: kind === "todo" ? "2026-11-02T17:00" : null
      })
    ))
    expect(mocks.updateCalendarItem).not.toHaveBeenCalled()
  })

  it.each([
    { kind: "event", start_at: "2026-06-05T09:00:35.123-07:00", end_at: "2026-06-05T10:00:45-07:00", due_at: null },
    { kind: "todo", start_at: "2026-06-05T17:00:35+05:30", end_at: null, due_at: "2026-06-05T17:00:35+05:30" },
    { kind: "event", start_at: "2026-06-05", end_at: "2026-06-08", due_at: null, all_day: true }
  ])("omits unchanged temporal fields when editing $kind text", async (temporal) => {
    const user = userEvent.setup()
    renderDrawer({ ...localItem, ...temporal })
    await user.type(screen.getByRole("textbox", { name: "Title" }), " revised")
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalled())
    const updates = mocks.updateCalendarItem.mock.calls[0][1]
    for (const field of ["start_at", "end_at", "due_at", "kind", "all_day", "timezone", "recurrence"]) {
      expect(updates).not.toHaveProperty(field)
    }
  })

  it.each(["event", "todo"])("locks later %s occurrence timing/kind while allowing series text edits", async (kind) => {
    const user = userEvent.setup()
    renderDrawer({
      ...localItem,
      id: "calendar_item:7:occurrence:2:2026-06-12T09:00:00-07:00",
      kind,
      start_at: "2026-06-12T09:00:00-07:00",
      end_at: kind === "event" ? "2026-06-12T10:00:00-07:00" : null,
      due_at: kind === "todo" ? "2026-06-12T09:00:00-07:00" : null,
      recurrence_id: 3,
      occurrence_index: 2
    })
    for (const label of kind === "event" ? ["Start", "End"] : ["Due"]) {
      expect(screen.getByRole("textbox", { name: label })).toHaveProperty("disabled", true)
    }
    for (const label of ["Event", "Todo"]) {
      expect(screen.getByRole("radio", { name: label })).toHaveProperty("disabled", true)
    }
    expect(screen.getByRole("textbox", { name: "Title" })).toHaveProperty("disabled", false)
    await user.type(screen.getByRole("textbox", { name: "Title" }), " revised")
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalledWith(
      7, expect.objectContaining({ title: "Draft notes revised" })
    ))
    const updates = mocks.updateCalendarItem.mock.calls[0][1]
    for (const field of ["start_at", "end_at", "due_at", "kind", "recurrence"]) {
      expect(updates).not.toHaveProperty(field)
    }
  })

  it("locks the first occurrence even when its occurrence index is zero", () => {
    renderDrawer({ ...localItem, recurrence_id: 3, occurrence_index: 0 })
    expect(screen.getByRole("textbox", { name: "Start" })).toHaveProperty("disabled", true)
  })

  it.each([
    { label: "Start", field: "start_at", original: "2026-06-05T09:00:00-07:00", input: "2026-06-05T09:30", expected: "2026-06-05T09:30-07:00" },
    { label: "End", field: "end_at", original: "2026-06-05T10:00:00+05:30", input: "2026-06-05T10:30", expected: "2026-06-05T10:30+05:30" },
    { label: "Due", field: "due_at", original: "2026-06-05T17:00:00Z", input: "2026-06-05T17:30", expected: "2026-06-05T17:30Z" },
    { label: "Start", field: "start_at", original: "2026-06-05T09:00:00-07:00", input: "2026-06-05T09:30+02:00", expected: "2026-06-05T09:30+02:00" },
    { label: "Due", field: "due_at", original: "2026-06-05T17:00:00-07:00", input: "", expected: null }
  ])("preserves offsets or explicit clearing for $label edits ($input)", async ({ label, field, original, input, expected }) => {
    const user = userEvent.setup()
    renderDrawer({ ...localItem, kind: label === "Due" ? "todo" : "event", [field]: original })
    const control = screen.getByRole("textbox", { name: label })
    await user.clear(control)
    if (input) await user.type(control, input)
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalled())
    const updates = mocks.updateCalendarItem.mock.calls[0][1]
    expect(updates[field]).toBe(expected)
    for (const other of ["start_at", "end_at", "due_at"].filter((key) => key !== field)) {
      expect(updates).not.toHaveProperty(other)
    }
  })

  it("keeps explicit all-day edits as civil dates", async () => {
    const user = userEvent.setup()
    renderDrawer({ ...localItem, all_day: true, start_at: "2026-06-05", end_at: "2026-06-08" })
    await user.clear(screen.getByRole("textbox", { name: "End" }))
    await user.type(screen.getByRole("textbox", { name: "End" }), "2026-06-09")
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalled())
    expect(mocks.updateCalendarItem.mock.calls[0][1]).toMatchObject({ end_at: "2026-06-09" })
    expect(mocks.updateCalendarItem.mock.calls[0][1]).not.toHaveProperty("start_at")
  })

  it("still clears inactive times when explicitly changing an ordinary item's kind", async () => {
    const user = userEvent.setup()
    renderDrawer(localItem)
    await user.click(screen.getByRole("radio", { name: "Todo" }))
    await user.type(screen.getByRole("textbox", { name: "Due" }), "2026-06-05T17:00Z")
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => expect(mocks.updateCalendarItem).toHaveBeenCalledWith(
      7, expect.objectContaining({ kind: "todo", start_at: null, end_at: null, due_at: "2026-06-05T17:00Z" })
    ))
  })

  it("loads persisted links after refresh and removes one with confirmation", async () => {
    const user = userEvent.setup()
    mocks.listCalendarLinks.mockResolvedValue({ items: [{
      id: 11, calendar_item_id: 7, target_type: "url", target_id: "https://example.test/notes",
      label: "Saved notes", url: "https://example.test/notes", metadata: {},
      created_at: "2026-06-05T00:00:00Z", updated_at: "2026-06-05T00:00:00Z"
    }], total: 1 })
    mocks.deleteCalendarLink.mockResolvedValue({ removed: 1 })
    renderDrawer(localItem)
    const link = await screen.findByRole("link", { name: "Saved notes" })
    expect(link.getAttribute("href")).toBe("https://example.test/notes")
    expect(mocks.listCalendarLinks).toHaveBeenCalledWith(7)
    await user.click(screen.getByRole("button", { name: "Remove Saved notes" }))
    expect(mocks.deleteCalendarLink).not.toHaveBeenCalled()
    await user.click(screen.getByRole("button", { name: "Remove link" }))
    await waitFor(() => expect(mocks.deleteCalendarLink).toHaveBeenCalledWith(7, 11))
    await waitFor(() => expect(screen.queryByRole("link", { name: "Saved notes" })).toBeNull())
  })

  it.each([
    { kind: "event", recurrence_id: 3, occurrence_index: null },
    { kind: "todo", recurrence_id: 3, occurrence_index: null },
    { kind: "event", recurrence_id: null, occurrence_index: 0 },
    { kind: "todo", recurrence_id: null, occurrence_index: 0 }
  ])("hides deletion for $kind occurrences (recurrence_id=$recurrence_id, occurrence_index=$occurrence_index)", (occurrence) => {
    renderDrawer({
      ...localItem,
      ...occurrence,
      id: "calendar_item:7:occurrence:0:2026-06-05T09:00:00Z",
      end_at: occurrence.kind === "event" ? localItem.end_at : null,
      due_at: occurrence.kind === "todo" ? localItem.start_at : null
    })

    expect(screen.queryByRole("button", { name: "Delete item" })).toBeNull()
  })

  it.each(["event", "todo"])("requires confirmation before deleting a local %s item", async (kind) => {
    const user = userEvent.setup()
    const onSaved = vi.fn()
    const onClose = vi.fn()
    mocks.deleteCalendarItem.mockResolvedValue({ deleted: true })
    renderDrawer({
      ...localItem,
      kind,
      end_at: kind === "event" ? localItem.end_at : null,
      due_at: kind === "todo" ? localItem.start_at : null
    }, { onSaved, onClose })

    await user.click(screen.getByRole("button", { name: "Delete item" }))
    expect(mocks.deleteCalendarItem).not.toHaveBeenCalled()
    await user.click(screen.getByRole("button", { name: "Confirm delete" }))

    await waitFor(() => expect(mocks.deleteCalendarItem).toHaveBeenCalledWith(
      expect.objectContaining({ calendar_item_id: 7, source_owner: "tldw" })
    ))
    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1))
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
    expect(screen.queryByRole("button", { name: "Delete item" })).toBeNull()

    await user.click(screen.getByRole("button", { name: "Copy into tldw" }))

    await waitFor(() => {
      expect(mocks.copyCalendarItemIntoTldw).toHaveBeenCalledWith(42, {
        target_calendar_id: 1
      })
    })
  })

  it("lets provider-owned items save tldw-local tags, annotations, and links without editing provider fields", async () => {
    const user = userEvent.setup()
    mocks.updateCalendarLocalTags.mockResolvedValue({
      id: 3,
      calendar_item_id: 42,
      author_user_id: 1,
      body: "",
      tags: ["remote", "follow-up"],
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })
    mocks.createCalendarAnnotation.mockResolvedValue({
      id: 4,
      calendar_item_id: 42,
      author_user_id: 1,
      body: "Ask vendor",
      tags: [],
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })
    mocks.createCalendarLink.mockResolvedValue({
      id: 5,
      calendar_item_id: 42,
      target_type: "note",
      target_id: "/notes/vendor",
      label: "Vendor note",
      url: "/notes/vendor",
      metadata: {},
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })

    renderDrawer(providerItem)

    expect(await screen.findByDisplayValue("External sync review")).toHaveProperty("disabled", true)
    const tags = screen.getByRole("textbox", { name: "Tags" })
    expect(tags).toHaveProperty("disabled", false)
    await user.type(tags, ", follow-up")
    await user.type(screen.getByRole("textbox", { name: "Annotation" }), "Ask vendor")
    await user.type(screen.getByRole("textbox", { name: "Link label" }), "Vendor note")
    await user.type(screen.getByRole("textbox", { name: "Link URL" }), "/notes/vendor")
    await user.click(screen.getByRole("button", { name: "Save context" }))

    await waitFor(() => {
      expect(mocks.updateCalendarItem).not.toHaveBeenCalled()
      expect(mocks.updateCalendarLocalTags).toHaveBeenCalledWith(42, {
        tags: ["remote", "follow-up"]
      })
      expect(mocks.createCalendarAnnotation).toHaveBeenCalledWith(42, {
        body: "Ask vendor",
        tags: []
      })
      expect(mocks.createCalendarLink).toHaveBeenCalledWith(
        42,
        expect.objectContaining({
          label: "Vendor note",
          url: "/notes/vendor"
        })
      )
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

  it("does not clear local tags when an older view response omitted tag details", async () => {
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
      local_tags: ["unchanged"],
      metadata: {},
      created_at: "2026-06-05T00:00:00Z",
      updated_at: "2026-06-05T00:00:00Z"
    })

    renderDrawer({ ...localItem, local_tags: undefined, metadata: {} } as never)

    const title = await screen.findByRole("textbox", { name: "Title" })
    await user.clear(title)
    await user.type(title, "Draft notes updated")
    await user.click(screen.getByRole("button", { name: "Save item" }))

    await waitFor(() => {
      expect(mocks.updateCalendarItem).toHaveBeenCalledWith(
        7,
        expect.not.objectContaining({
          local_tags: expect.anything()
        })
      )
    })
  })
})
