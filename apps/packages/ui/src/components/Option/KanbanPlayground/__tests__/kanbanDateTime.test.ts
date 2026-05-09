import { describe, expect, it } from "vitest"

import {
  formatKanbanDateTimeLocalValue,
  parseKanbanDateTimeLocalValue
} from "../kanbanDateTime"

const localDateTimeValue = (date: Date) => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  const hour = String(date.getHours()).padStart(2, "0")
  const minute = String(date.getMinutes()).padStart(2, "0")
  return `${year}-${month}-${day}T${hour}:${minute}`
}

describe("kanban date-time helpers", () => {
  it("formats ISO due dates for native datetime-local inputs", () => {
    const date = new Date(2026, 3, 3, 12, 34, 56)

    expect(formatKanbanDateTimeLocalValue(date.toISOString())).toBe(
      localDateTimeValue(date)
    )
  })

  it("returns an empty input value for missing or invalid due dates", () => {
    expect(formatKanbanDateTimeLocalValue(null)).toBe("")
    expect(formatKanbanDateTimeLocalValue("not-a-date")).toBe("")
  })

  it("converts native datetime-local input values to ISO timestamps", () => {
    expect(parseKanbanDateTimeLocalValue("2026-04-03T12:34")).toBe(
      new Date(2026, 3, 3, 12, 34).toISOString()
    )
  })

  it("returns null for cleared or invalid datetime-local input values", () => {
    expect(parseKanbanDateTimeLocalValue("")).toBeNull()
    expect(parseKanbanDateTimeLocalValue("not-a-date")).toBeNull()
  })
})
