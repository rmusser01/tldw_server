import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import TaskChecklistToggle from "../TaskChecklistToggle"
import type { NoteTask } from "@/services/notes-tasks"

const translations = vi.hoisted(() => new Map<string, string>())

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string } & Record<string, unknown>) => {
      const template = translations.get(key) ?? options?.defaultValue ?? key
      return template.replace(/\{\{(\w+)\}\}/g, (match, name: string) =>
        options && name in options ? String(options[name]) : match
      )
    }
  })
}))

afterEach(() => {
  translations.clear()
})

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

  it("lets a translation put the counts anywhere in the chip label", () => {
    translations.set("option:notesSearch.taskChipLabel", "{{done}} von {{total}} Aufgaben")
    render(
      <TaskChecklistToggle
        content={"- [ ] Draft PRD\n- [ ] Ship it"}
        tasks={[doneTask]}
      />
    )

    expect(screen.getByRole("button", { name: "1 von 2 Aufgaben" })).toBeInTheDocument()
  })

  it("says the checkable list is behind the chip", async () => {
    render(<TaskChecklistToggle content={"- [ ] Draft PRD"} />)

    fireEvent.mouseEnter(screen.getByTestId("notes-task-chip"))
    expect(
      await screen.findByText(
        "Open the list to check off tasks. Changes update the note text and your task list."
      )
    ).toBeInTheDocument()
  })

  it("renders nothing for a note without tasks", () => {
    const { container } = render(<TaskChecklistToggle content={"Just text"} />)
    expect(container).toBeEmptyDOMElement()
  })
})
