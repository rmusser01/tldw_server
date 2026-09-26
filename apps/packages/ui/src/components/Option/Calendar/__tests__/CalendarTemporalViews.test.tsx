import React from "react"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest"
import type { CalendarViewItemResponse } from "@/services/calendar"
import { CalendarAgenda } from "../CalendarAgenda"
import { CalendarWeekView } from "../CalendarWeekView"

const allDayItem: CalendarViewItemResponse = {
  id: "calendar_item:7",
  calendar_item_id: 7,
  calendar_id: 1,
  kind: "event",
  title: "Research retreat",
  source_owner: "tldw",
  start_at: "2026-06-05",
  end_at: "2026-06-06",
  due_at: null,
  all_day: true,
  status: "confirmed",
  local_tags: [],
  metadata: {}
}

const dayLabel = (day: number, month = 5): string =>
  new Intl.DateTimeFormat(undefined, { weekday: "short", month: "short", day: "numeric" })
    .format(new Date(2026, month, day))
const dayGroup = (day: number, month = 5): HTMLElement => screen.getByText(dayLabel(day, month)).closest("div")!
const weekDay = (day: number): HTMLElement => dayGroup(day).parentElement!
const juneWindow = () => ({ windowStart: new Date(2026, 5, 1), windowEnd: new Date(2026, 6, 1) })

describe("Calendar civil-date rendering", () => {
  it("keeps an overnight timed event overlapping the start of the agenda window", () => {
    const item = { ...allDayItem, all_day: false,
      start_at: "2026-06-04T23:00:00-07:00", end_at: "2026-06-05T01:00:00-07:00"
    }
    render(<CalendarAgenda calendars={[]} items={[item]}
      windowStart={new Date(2026, 5, 5)} windowEnd={new Date(2026, 5, 19)} onSelectItem={vi.fn()} />)
    expect(within(dayGroup(5)).getByRole("button", { name: /Research retreat/ })).toBeTruthy()
    expect(screen.queryByText(dayLabel(4))).toBeNull()
    expect(screen.getAllByRole("button", { name: /Research retreat/ })).toHaveLength(1)
  })
  it("clamps a long all-day span to the requested agenda window", () => {
    render(<CalendarAgenda calendars={[]} items={[{
      ...allDayItem, start_at: "2020-01-01", end_at: "2027-01-01"
    }]} windowStart={new Date(2026, 5, 5)} windowEnd={new Date(2026, 5, 19)} onSelectItem={vi.fn()} />)
    expect(screen.getAllByRole("button", { name: /Research retreat/ })).toHaveLength(14)
    expect(screen.queryByText(dayLabel(4))).toBeNull()
    expect(screen.queryByText(dayLabel(19))).toBeNull()
  })
  beforeAll(() => {
    vi.stubEnv("TZ", "America/Los_Angeles")
    expect(new Date(2026, 5, 5).getTimezoneOffset()).toBe(420)
  })

  afterAll(() => vi.unstubAllEnvs())

  it("keeps date-only agenda entries on their civil day and selectable", async () => {
    const onSelectItem = vi.fn()
    render(<CalendarAgenda {...juneWindow()} calendars={[]} items={[allDayItem]} onSelectItem={onSelectItem} />)
    const group = dayGroup(5)
    await userEvent.setup().click(within(group).getByRole("button", { name: /Research retreat/ }))
    expect(onSelectItem).toHaveBeenCalledWith(allDayItem)
    expect(screen.queryByText(dayLabel(4))).toBeNull()
  })

  it("shows multi-day all-day agenda entries on every date before the exclusive end", () => {
    render(<CalendarAgenda {...juneWindow()} calendars={[]} items={[{ ...allDayItem, end_at: "2026-06-08" }]} onSelectItem={vi.fn()} />)
    for (const day of [5, 6, 7]) {
      expect(within(dayGroup(day)).getByRole("button", { name: /Research retreat/ })).toBeTruthy()
    }
    expect(screen.queryByText(dayLabel(8))).toBeNull()
    expect(screen.getAllByRole("button", { name: /Research retreat/ })).toHaveLength(3)
  })

  it("keeps a date-only all-day todo with no start on its due civil day", () => {
    render(<CalendarAgenda {...juneWindow()} calendars={[]} items={[{
      ...allDayItem, kind: "todo", start_at: null, due_at: "2026-06-05", end_at: null
    }]} onSelectItem={vi.fn()} />)
    expect(within(dayGroup(5)).getByRole("button", { name: /Research retreat/ })).toBeTruthy()
  })

  it.each([
    { start_at: "2026-06-05", end_at: "2026-06-06", expected: [5] },
    { start_at: "2026-06-05", end_at: "2026-06-08", expected: [5, 6, 7] },
    { start_at: "2026-05-30", end_at: "2026-06-03", expected: [1, 2] },
    { start_at: "2026-06-05", end_at: null, expected: [5] }
  ])("places all-day week spans $start_at to $end_at on overlapping civil dates only", ({ start_at, end_at, expected }) => {
    render(<CalendarWeekView calendars={[]} items={[{ ...allDayItem, start_at, end_at }]}
      weekStart={new Date(2026, 5, 1)} onSelectItem={vi.fn()} />)
    for (const day of [1, 2, 3, 4, 5, 6, 7]) {
      const buttons = within(weekDay(day)).queryAllByRole("button", { name: /Research retreat/ })
      expect(buttons).toHaveLength(expected.includes(day) ? 1 : 0)
    }
  })

  it("iterates civil dates across the fall daylight-saving transition", () => {
    render(<CalendarAgenda windowStart={new Date(2026, 9, 31)} windowEnd={new Date(2026, 10, 3)} calendars={[]} items={[{
      ...allDayItem, start_at: "2026-10-31", end_at: "2026-11-03"
    }]} onSelectItem={vi.fn()} />)
    for (const [day, month] of [[31, 9], [1, 10], [2, 10]]) {
      expect(within(dayGroup(day, month)).getByRole("button", { name: /Research retreat/ })).toBeTruthy()
    }
    expect(screen.queryByText(dayLabel(3, 10))).toBeNull()
  })

  it("continues placing timed offset-aware items on their browser-local day", () => {
    const item = { ...allDayItem, all_day: false, start_at: "2026-06-05T01:00:00Z", end_at: "2026-06-05T02:00:00Z" }
    const agenda = render(<CalendarAgenda {...juneWindow()} calendars={[]} items={[item]} onSelectItem={vi.fn()} />)
    expect(within(dayGroup(4)).getByRole("button", { name: /Research retreat/ })).toBeTruthy()
    expect(screen.queryByText(dayLabel(5))).toBeNull()
    agenda.unmount()
    render(<CalendarWeekView calendars={[]} items={[item]} weekStart={new Date(2026, 5, 1)} onSelectItem={vi.fn()} />)
    expect(within(weekDay(4)).getByRole("button", { name: /Research retreat/ })).toBeTruthy()
    expect(within(weekDay(5)).queryByRole("button", { name: /Research retreat/ })).toBeNull()
  })
})
