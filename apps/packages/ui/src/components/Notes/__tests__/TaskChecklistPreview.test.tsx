import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import TaskChecklistPreview from "@/components/Notes/TaskChecklistPreview"
import type { NoteTask } from "@/services/notes-tasks"

const task = (
  id: string,
  lineNumber: number,
  text: string,
  status: "open" | "done",
  projectionStatus: NoteTask["projection_status"] = "live"
): NoteTask => ({
  id,
  note_id: "note-1",
  text,
  status,
  metadata: {},
  projection_status: projectionStatus,
  version: 1,
  projection: {
    note_id: "note-1",
    note_version: 2,
    line_number: lineNumber,
    start_offset: 0,
    end_offset: 0,
    raw_line: `- [${status === "done" ? "x" : " "}] ${text}`,
    has_child_content: false,
    projection_status: projectionStatus
  }
})

describe("TaskChecklistPreview", () => {
  it("renders accessible task checkboxes from markdown and task state", () => {
    render(
      <TaskChecklistPreview
        content={"- [ ] Draft PRD\n- [x] Review MCP tools"}
        tasks={[
          task("task-1", 1, "Draft PRD", "open"),
          task("task-2", 2, "Review MCP tools", "done")
        ]}
      />
    )

    expect(screen.getByRole("checkbox", { name: /Draft PRD/ })).not.toBeChecked()
    expect(screen.getByRole("checkbox", { name: /Review MCP tools/ })).toBeChecked()
  })

  it("routes dirty toggles to local markdown callbacks only", () => {
    const onToggleLocal = vi.fn()
    const onToggleTaskStatus = vi.fn()

    render(
      <TaskChecklistPreview
        content="- [ ] Draft PRD"
        tasks={[task("task-1", 1, "Draft PRD", "open")]}
        isDirty
        onToggleLocal={onToggleLocal}
        onToggleTaskStatus={onToggleTaskStatus}
      />
    )

    fireEvent.click(screen.getByRole("checkbox", { name: /Draft PRD/ }))

    expect(onToggleLocal).toHaveBeenCalledWith(
      expect.objectContaining({
        lineNumber: 1,
        checked: false,
        nextStatus: "done"
      })
    )
    expect(onToggleTaskStatus).not.toHaveBeenCalled()
  })

  it("routes clean toggles to backend task status callbacks", () => {
    const onToggleLocal = vi.fn()
    const onToggleTaskStatus = vi.fn()

    render(
      <TaskChecklistPreview
        content="- [x] Review MCP tools"
        tasks={[task("task-2", 1, "Review MCP tools", "done")]}
        onToggleLocal={onToggleLocal}
        onToggleTaskStatus={onToggleTaskStatus}
      />
    )

    fireEvent.click(screen.getByRole("checkbox", { name: /Review MCP tools/ }))

    expect(onToggleTaskStatus).toHaveBeenCalledWith(
      expect.objectContaining({
        task: expect.objectContaining({ id: "task-2" }),
        lineNumber: 1,
        nextStatus: "open"
      })
    )
    expect(onToggleLocal).not.toHaveBeenCalled()
  })

  it("renders non-blocking badges for unsafe projection states", () => {
    render(
      <TaskChecklistPreview
        content={"- [ ] Ambiguous task\n- [ ] Unlinked task"}
        tasks={[
          task("task-1", 1, "Ambiguous task", "open", "ambiguous"),
          task("task-2", 2, "Unlinked task", "open", "unlinked")
        ]}
      />
    )

    expect(screen.getByText("Ambiguous")).toBeInTheDocument()
    expect(screen.getByText("Unlinked")).toBeInTheDocument()
  })
})
