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

describe("EditableCell date handling", () => {
  it("renders valid date-like values as YYYY-MM-DD", () => {
    renderDateCell({ value: "2026-04-03T12:34:56" })

    expect(screen.getByText("2026-04-03")).toBeInTheDocument()
  })

  it("preserves invalid date values instead of rendering Invalid Date", () => {
    renderDateCell({ value: "not-a-date" })

    expect(screen.getByText("not-a-date")).toBeInTheDocument()
    expect(screen.queryByText("Invalid Date")).not.toBeInTheDocument()
  })

  it("uses a native date input and emits YYYY-MM-DD changes", () => {
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

    expect(onFinishEdit).toHaveBeenCalledWith("2026-05-09")
  })

  it("emits null when the native date input is cleared", () => {
    const onFinishEdit = vi.fn()
    renderDateCell({
      value: "2026-04-03",
      isEditing: true,
      onFinishEdit
    })

    fireEvent.change(screen.getByLabelText("Due date"), {
      target: { value: "" }
    })

    expect(onFinishEdit).toHaveBeenCalledWith(null)
  })
})
