import React from "react"
import { Tag } from "antd"
import type {
  CalendarResponse,
  CalendarViewItemResponse
} from "@/services/calendar"

type OwnershipInput = Pick<
  CalendarViewItemResponse,
  "source_owner" | "read_only_reason" | "calendar_id"
> & {
  provider_owned?: boolean
}

export const getCalendarOwnershipLabel = (
  item: OwnershipInput,
  calendar?: CalendarResponse | null
): "Local" | "Org" | "Provider" | "Linked" => {
  if (item.source_owner === "linked_projection") return "Linked"
  if (item.source_owner === "provider" || item.provider_owned) return "Provider"
  if (calendar?.org_id) return "Org"
  return "Local"
}

const colorByLabel: Record<ReturnType<typeof getCalendarOwnershipLabel>, string> = {
  Local: "default",
  Org: "processing",
  Provider: "warning",
  Linked: "purple"
}

export interface CalendarOwnershipBadgeProps {
  item: OwnershipInput
  calendar?: CalendarResponse | null
}

export const CalendarOwnershipBadge: React.FC<CalendarOwnershipBadgeProps> = ({
  item,
  calendar
}) => {
  const label = getCalendarOwnershipLabel(item, calendar)

  return (
    <Tag color={colorByLabel[label]} style={{ marginInlineEnd: 0 }}>
      {label}
    </Tag>
  )
}

export default CalendarOwnershipBadge
