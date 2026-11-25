import type { CalendarViewItemResponse } from "@/services/calendar"

const parseCalendarDate = (value: string): Date => {
  if (/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [year, month, day] = value.split("-").map(Number)
    return new Date(year, month - 1, day)
  }
  return new Date(value)
}

export const calendarItemDayRange = (
  item: CalendarViewItemResponse
): { start: Date; end: Date } | null => {
  const value = item.start_at ?? item.due_at ?? item.end_at
  if (!value) return null
  const date = parseCalendarDate(value)
  const start = new Date(date.getFullYear(), date.getMonth(), date.getDate())
  const end = new Date(start)
  // Increment the civil day, not 24 hours: DST days need not be 24 hours long.
  end.setDate(end.getDate() + 1)
  if (item.all_day && item.kind === "event" && item.end_at) {
    const endDate = parseCalendarDate(item.end_at)
    return { start, end: new Date(endDate.getFullYear(), endDate.getMonth(), endDate.getDate()) }
  }
  if (!item.all_day && item.end_at) {
    const endDate = parseCalendarDate(item.end_at)
    if (endDate > date) {
      const lastDay = new Date(endDate.getFullYear(), endDate.getMonth(), endDate.getDate())
      if (endDate > lastDay) lastDay.setDate(lastDay.getDate() + 1)
      return { start, end: lastDay }
    }
  }
  return { start, end }
}
