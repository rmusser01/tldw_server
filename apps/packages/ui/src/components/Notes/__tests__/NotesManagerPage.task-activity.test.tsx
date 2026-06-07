import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import TaskActivityNotice from "@/components/Notes/TaskActivityNotice"

const tMock = vi.hoisted(() => vi.fn())

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      options?: {
        defaultValue?: string
        [key: string]: unknown
      }
    ) => {
      tMock(_key, options)
      return options?.defaultValue || _key
    }
  })
}))

describe("NotesManagerPage task activity notice", () => {
  beforeEach(() => {
    tMock.mockClear()
  })

  it("shows actor, tool, affected note context, inspect, and dismiss actions", () => {
    const onInspect = vi.fn()
    const onDismiss = vi.fn()

    render(
      <TaskActivityNotice
        events={[
          {
            id: "event-1",
            task_id: "task-1",
            note_id: "note-1",
            event_type: "status_changed",
            actor_type: "agent",
            actor_id: "agent-1",
            tool_name: "notes.tasks.set_status",
            created_at: "2026-06-05T07:05:00Z"
          }
        ]}
        noteTitle="Task note"
        testId="notes-task-activity-notice"
        onInspect={onInspect}
        onDismiss={onDismiss}
      />
    )

    expect(screen.getByTestId("notes-task-activity-notice")).toHaveTextContent(
      "agent-1 via notes.tasks.set_status changed 1 task in Task note."
    )

    fireEvent.click(screen.getByRole("button", { name: "Inspect task activity" }))
    fireEvent.click(screen.getByRole("button", { name: "Dismiss task activity" }))

    expect(onInspect).toHaveBeenCalledTimes(1)
    expect(onDismiss).toHaveBeenCalledWith("event-1")
  })

  it("keeps i18next count numeric while interpolating the task count label", () => {
    render(
      <TaskActivityNotice
        events={[
          {
            id: "event-1",
            task_id: "task-1",
            note_id: "note-1",
            event_type: "status_changed",
            actor_type: "agent",
            actor_id: "agent-1",
            tool_name: "notes.tasks.set_status",
            created_at: "2026-06-05T07:05:00Z"
          },
          {
            id: "event-2",
            task_id: "task-2",
            note_id: "note-1",
            event_type: "status_changed",
            actor_type: "agent",
            actor_id: "agent-1",
            tool_name: "notes.tasks.set_status",
            created_at: "2026-06-05T07:06:00Z"
          }
        ]}
        noteTitle="Task note"
      />
    )

    expect(tMock).toHaveBeenCalledWith(
      "option:notesSearch_taskActivitySummary",
      expect.objectContaining({
        count: 2,
        countLabel: "2 tasks"
      })
    )
  })
})
