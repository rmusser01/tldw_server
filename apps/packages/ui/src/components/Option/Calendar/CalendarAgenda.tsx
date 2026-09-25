import React from "react"
import { Button, Empty, Space, Typography } from "antd"
import type {
  CalendarResponse,
  CalendarViewItemResponse
} from "@/services/calendar"
import { CalendarOwnershipBadge } from "./CalendarOwnershipBadge"

const itemTimestamp = (item: CalendarViewItemResponse): string | null =>
  item.start_at ?? item.due_at ?? item.end_at ?? null

const formatDay = (value: string | null): string => {
  if (!value) return "Unscheduled"
  return new Intl.DateTimeFormat(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric"
  }).format(new Date(value))
}

const formatTime = (item: CalendarViewItemResponse): string => {
  if (item.all_day) return "All day"
  const start = item.start_at ? new Date(item.start_at) : null
  const end = item.end_at ? new Date(item.end_at) : null
  const due = item.due_at ? new Date(item.due_at) : null
  const formatter = new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit"
  })

  if (start && end) return `${formatter.format(start)} - ${formatter.format(end)}`
  if (start) return formatter.format(start)
  if (due) return `Due ${formatter.format(due)}`
  return "No time"
}

export interface CalendarAgendaProps {
  calendars: CalendarResponse[]
  items: CalendarViewItemResponse[]
  onSelectItem: (item: CalendarViewItemResponse) => void
}

export const CalendarAgenda: React.FC<CalendarAgendaProps> = ({
  calendars,
  items,
  onSelectItem
}) => {
  const calendarsById = React.useMemo(
    () => new Map(calendars.map((calendar) => [calendar.id, calendar])),
    [calendars]
  )
  const groupedItems = React.useMemo(() => {
    const groups = new Map<string, CalendarViewItemResponse[]>()
    for (const item of items) {
      const key = formatDay(itemTimestamp(item))
      groups.set(key, [...(groups.get(key) ?? []), item])
    }
    return Array.from(groups.entries())
  }, [items])

  return (
    <section aria-label="Agenda" className="min-w-0 flex-1">
      {items.length === 0 ? (
        <Empty description="No calendar items in this window" />
      ) : null}
      <div className="flex flex-col gap-4">
        {groupedItems.map(([day, dayItems]) => (
          <div key={day} className="flex flex-col gap-2">
            <Typography.Text strong>{day}</Typography.Text>
            <div className="flex flex-col divide-y rounded border border-slate-200 bg-slate-50/40">
              {dayItems.map((item) => {
                const calendar = item.calendar_id
                  ? calendarsById.get(item.calendar_id)
                  : null
                return (
                  <Button
                    key={item.id}
                    type="text"
                    className="h-auto w-full rounded-none px-3 py-2 text-left"
                    onClick={() => onSelectItem(item)}
                  >
                    <div className="grid w-full grid-cols-[10px_minmax(0,1fr)_auto] items-start gap-3">
                      <span
                        aria-hidden="true"
                        className="mt-1 h-8 rounded"
                        style={{
                          backgroundColor: calendar?.color ?? "rgb(100 116 139)"
                        }}
                      />
                      <span className="min-w-0">
                        <span className="block truncate font-medium text-slate-900">
                          {item.title}
                        </span>
                        <span className="block truncate text-xs text-slate-500">
                          {formatTime(item)}
                          {calendar?.name ? ` / ${calendar.name}` : ""}
                          {item.link?.label ? ` / ${item.link.label}` : ""}
                        </span>
                      </span>
                      <Space size={6} align="start">
                        {item.link?.label ? (
                          <Typography.Text type="secondary" className="text-xs">
                            {item.link.label}
                          </Typography.Text>
                        ) : null}
                        <CalendarOwnershipBadge item={item} calendar={calendar} />
                      </Space>
                    </div>
                  </Button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default CalendarAgenda
