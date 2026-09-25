import React from "react"
import { Button, Empty, Typography } from "antd"
import type {
  CalendarResponse,
  CalendarViewItemResponse
} from "@/services/calendar"
import { CalendarOwnershipBadge } from "./CalendarOwnershipBadge"

const startOfDay = (date: Date): Date =>
  new Date(date.getFullYear(), date.getMonth(), date.getDate())

const addDays = (date: Date, days: number): Date => {
  const next = new Date(date)
  next.setDate(next.getDate() + days)
  return next
}

const sameDay = (value: string | null | undefined, day: Date): boolean => {
  if (!value) return false
  return startOfDay(new Date(value)).getTime() === startOfDay(day).getTime()
}

const formatTime = (value: string | null | undefined): string => {
  if (!value) return ""
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit"
  }).format(new Date(value))
}

export interface CalendarWeekViewProps {
  calendars: CalendarResponse[]
  items: CalendarViewItemResponse[]
  weekStart: Date
  onSelectItem: (item: CalendarViewItemResponse) => void
}

export const CalendarWeekView: React.FC<CalendarWeekViewProps> = ({
  calendars,
  items,
  weekStart,
  onSelectItem
}) => {
  const calendarsById = React.useMemo(
    () => new Map(calendars.map((calendar) => [calendar.id, calendar])),
    [calendars]
  )
  const days = React.useMemo(
    () => Array.from({ length: 7 }, (_, index) => addDays(weekStart, index)),
    [weekStart]
  )

  return (
    <section aria-label="Week" className="min-w-0 flex-1">
      {items.length === 0 ? (
        <Empty description="No calendar items this week" />
      ) : null}
      <div className="grid min-h-[520px] grid-cols-7 overflow-hidden rounded border border-slate-200 bg-slate-50/40">
        {days.map((day) => {
          const dayItems = items.filter((item) =>
            sameDay(item.start_at ?? item.due_at, day)
          )
          const allDayItems = dayItems.filter((item) => item.all_day)
          const timedItems = dayItems.filter((item) => !item.all_day)
          return (
            <div key={day.toISOString()} className="min-w-0 border-r border-slate-200 last:border-r-0">
              <div className="border-b border-slate-200 bg-slate-100 px-2 py-2">
                <Typography.Text strong className="block truncate text-xs">
                  {new Intl.DateTimeFormat(undefined, {
                    weekday: "short",
                    month: "short",
                    day: "numeric"
                  }).format(day)}
                </Typography.Text>
              </div>
              <div className="border-b border-slate-200 px-2 py-2">
                <Typography.Text type="secondary" className="block text-xs">
                  All day
                </Typography.Text>
                <div className="mt-1 flex min-h-8 flex-col gap-1">
                  {allDayItems.map((item) => {
                    const calendar = item.calendar_id
                      ? calendarsById.get(item.calendar_id)
                      : null
                    return (
                      <Button
                        key={item.id}
                        type="text"
                        className="h-auto w-full border border-slate-200 bg-slate-100 px-2 py-1 text-left"
                        onClick={() => onSelectItem(item)}
                      >
                        <span className="block truncate">{item.title}</span>
                        <CalendarOwnershipBadge item={item} calendar={calendar} />
                      </Button>
                    )
                  })}
                </div>
              </div>
              <div className="flex min-h-[390px] flex-col gap-2 p-2">
                {timedItems.map((item) => {
                  const calendar = item.calendar_id
                    ? calendarsById.get(item.calendar_id)
                    : null
                  return (
                    <Button
                      key={item.id}
                      type="text"
                      className="h-auto w-full border border-slate-200 bg-slate-100 px-2 py-2 text-left"
                      onClick={() => onSelectItem(item)}
                    >
                      <span className="block truncate text-xs font-medium">
                        {formatTime(item.start_at ?? item.due_at)}
                      </span>
                      <span className="block truncate">{item.title}</span>
                      <span className="mt-1 flex items-center justify-between gap-2">
                        <span
                          aria-hidden="true"
                          className="h-2 w-2 shrink-0 rounded-full"
                          style={{
                            backgroundColor: calendar?.color ?? "rgb(100 116 139)"
                          }}
                        />
                        <CalendarOwnershipBadge item={item} calendar={calendar} />
                      </span>
                    </Button>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

export default CalendarWeekView
