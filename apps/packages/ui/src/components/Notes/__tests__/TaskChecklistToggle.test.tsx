import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import TaskChecklistToggle from "../TaskChecklistToggle"
import type { NoteTask } from "@/services/notes-tasks"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key
  })
}))

const doneTask = {
  id: "task-2",
  note_id: "note-1",
  text: "Ship it",
  status: "done",
  metadata: {},
  projection_status: "live",
  version: 1,
  projection: { line_number: 2 }
} as unknown as NoteTask

describe("TaskChecklistToggle (NE-09)", () => {
  it("summarises tasks in one chip and keeps the duplicate list closed", () => {
    render(
      <TaskChecklistToggle
        content={"- [ ] Draft PRD\n- [ ] Ship it"}
        tasks={[doneTask]}
      />
    )

    const chip = screen.getByRole("button", { name: "Tasks 1/2" })
    expect(chip).toHaveAttribute("aria-expanded", "false")
    // The rendered preview already shows these items; no second list.
    expect(screen.queryAllByRole("checkbox")).toHaveLength(0)
  })

  it("opens the checkable list from the chip and toggles through it", () => {
    const onToggleLocal = vi.fn()
    render(
      <TaskChecklistToggle
        content={"- [ ] Draft PRD"}
        isDirty
        onToggleLocal={onToggleLocal}
      />
    )

    const chip = screen.getByRole("button", { name: "Tasks 0/1" })
    fireEvent.click(chip)
    expect(chip).toHaveAttribute("aria-expanded", "true")

    fireEvent.click(screen.getByRole("checkbox", { name: /Draft PRD/ }))
    expect(onToggleLocal).toHaveBeenCalledWith(
      expect.objectContaining({ lineNumber: 1, checked: false, nextStatus: "done" })
    )
  })

  it("renders nothing for a note without tasks", () => {
    const { container } = render(<TaskChecklistToggle content={"Just text"} />)
    expect(container).toBeEmptyDOMElement()
  })
})
