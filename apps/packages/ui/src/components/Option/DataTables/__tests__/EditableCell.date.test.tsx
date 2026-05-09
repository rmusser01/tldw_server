import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { EditableCell } from "../EditableCell"

const renderDateCell = ({
  value,
  isEditing = false,
  onFinishEdit = vi.fn()
}: {
  value: unknown
  isEditing?: boolean
  onFinishEdit?: (value: unknown) => void
}) => {
  const props = {
    value,
    columnType: "date" as const,
    columnName: "Due date",
    rowIndex: 0,
    isEditing,
    onStartEdit: vi.fn(),
    onFinishEdit,
    onCancelEdit: vi.fn()
  }

  return render(<EditableCell {...props} />)
}

const toLocalDateLabel = (value: number) => {
  const date = new Date(value)
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

describe("EditableCell date handling", () => {
  it("renders valid date-like values as YYYY-MM-DD", () => {
    renderDateCell({ value: "2026-04-03T12:34:56" })

    expect(screen.getByText("2026-04-03")).toBeInTheDocument()
  })

  it("renders numeric timestamps as local YYYY-MM-DD dates", () => {
    const timestamp = new Date(2026, 3, 3).getTime()

    renderDateCell({ value: timestamp })

    expect(screen.getByText(toLocalDateLabel(timestamp))).toBeInTheDocument()
  })

  it("treats zero as a valid epoch timestamp", () => {
    renderDateCell({ value: 0 })

    expect(screen.getByText(toLocalDateLabel(0))).toBeInTheDocument()
  })

  it("preserves invalid date values instead of rendering Invalid Date", () => {
    renderDateCell({ value: "not-a-date" })

    expect(screen.getByText("not-a-date")).toBeInTheDocument()
    expect(screen.queryByText("Invalid Date")).not.toBeInTheDocument()
  })

  it("preserves malformed date-prefix values instead of truncating them", () => {
    renderDateCell({ value: "2026-99-99T12:34:56" })

    expect(screen.getByText("2026-99-99T12:34:56")).toBeInTheDocument()
  })

  it("uses a native date input and emits one YYYY-MM-DD change on blur", () => {
    const onFinishEdit = vi.fn()
    renderDateCell({
      value: "2026-04-03T12:34:56",
      isEditing: true,
      onFinishEdit
    })

    const input = screen.getByLabelText("Due date") as HTMLInputElement
    expect(input.type).toBe("date")
    expect(input).toHaveValue("2026-04-03")

    fireEvent.change(input, { target: { value: "2026-05-09" } })
    expect(onFinishEdit).not.toHaveBeenCalled()

    fireEvent.blur(input)

    expect(onFinishEdit).toHaveBeenCalledTimes(1)
    expect(onFinishEdit).toHaveBeenCalledWith("2026-05-09")
  })

  it("emits null when the native date input is cleared and blurred", () => {
    const onFinishEdit = vi.fn()
    renderDateCell({
      value: "2026-04-03",
      isEditing: true,
      onFinishEdit
    })

    fireEvent.change(screen.getByLabelText("Due date"), {
      target: { value: "" }
    })
    fireEvent.blur(screen.getByLabelText("Due date"))

    expect(onFinishEdit).toHaveBeenCalledWith(null)
  })
})
