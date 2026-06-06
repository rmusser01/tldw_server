import React from "react"
import { Checkbox, Divider, Typography } from "antd"
import type { CheckboxValueType } from "antd/es/checkbox/Group"
import type { CalendarResponse } from "@/services/calendar"

export type CalendarSourceFilter = "local" | "org" | "provider" | "linked"
export type CalendarKindFilter = "event" | "todo"

export interface CalendarFilterRailProps {
  calendars: CalendarResponse[]
  selectedCalendarIds: number[]
  selectedSources: CalendarSourceFilter[]
  selectedKinds: CalendarKindFilter[]
  onCalendarChange: (ids: number[]) => void
  onSourceChange: (sources: CalendarSourceFilter[]) => void
  onKindChange: (kinds: CalendarKindFilter[]) => void
}

export const CalendarFilterRail: React.FC<CalendarFilterRailProps> = ({
  calendars,
  selectedCalendarIds,
  selectedSources,
  selectedKinds,
  onCalendarChange,
  onSourceChange,
  onKindChange
}) => (
  <aside aria-label="Calendar filters" className="w-full shrink-0 md:w-60">
    <div className="rounded border border-slate-200 bg-slate-50/50 p-3">
      <Typography.Text strong>Calendars</Typography.Text>
      <Checkbox.Group
        className="mt-3 flex flex-col gap-2"
        value={selectedCalendarIds}
        onChange={(values: CheckboxValueType[]) =>
          onCalendarChange(values.map((value) => Number(value)))
        }
      >
        {calendars.map((calendar) => (
          <Checkbox key={calendar.id} value={calendar.id}>
            <span className="inline-flex min-w-0 items-center gap-2">
              <span
                aria-hidden="true"
                className="h-2.5 w-2.5 shrink-0 rounded-full"
                style={{ backgroundColor: calendar.color ?? "rgb(100 116 139)" }}
              />
              <span className="truncate">{calendar.name}</span>
            </span>
          </Checkbox>
        ))}
      </Checkbox.Group>

      <Divider className="my-3" />
      <Typography.Text strong>Source</Typography.Text>
      <Checkbox.Group
        className="mt-3 flex flex-col gap-2"
        value={selectedSources}
        onChange={(values: CheckboxValueType[]) =>
          onSourceChange(values as CalendarSourceFilter[])
        }
        options={[
          { label: "Local", value: "local" },
          { label: "Org", value: "org" },
          { label: "Provider", value: "provider" },
          { label: "Linked", value: "linked" }
        ]}
      />

      <Divider className="my-3" />
      <Typography.Text strong>Kind</Typography.Text>
      <Checkbox.Group
        className="mt-3 flex flex-col gap-2"
        value={selectedKinds}
        onChange={(values: CheckboxValueType[]) =>
          onKindChange(values as CalendarKindFilter[])
        }
        options={[
          { label: "Events", value: "event" },
          { label: "Todos", value: "todo" }
        ]}
      />
    </div>
  </aside>
)

export default CalendarFilterRail
