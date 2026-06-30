import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import TaskActivityNotice from "@/components/Notes/TaskActivityNotice"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      options?: {
        defaultValue?: string
        [key: string]: unknown
      }
    ) => options?.defaultValue || _key
  })
}))

describe("NotesDockPanel task activity notice", () => {
  it("summarizes multiple persisted agent events and exposes inspect/dismiss controls", () => {
    const onInspect = vi.fn()
    const onDismiss = vi.fn()

    render(
      <TaskActivityNotice
        events={[
          {
            id: "event-1",
            task_id: "task-1",
            note_id: "101",
            event_type: "status_changed",
            actor_type: "agent",
            actor_id: "agent-1",
            tool_name: "notes.tasks.set_status",
            created_at: "2026-06-05T07:05:00Z"
          },
          {
            id: "event-2",
            task_id: "task-2",
            note_id: "101",
            event_type: "updated",
            actor_type: "agent",
            actor_id: "agent-1",
            tool_name: "notes.tasks.update",
            created_at: "2026-06-05T07:06:00Z"
          }
        ]}
        noteTitle="Dock task note"
        testId="notes-dock-task-activity-notice"
        compact
        onInspect={onInspect}
        onDismiss={onDismiss}
      />
    )

    expect(screen.getByTestId("notes-dock-task-activity-notice")).toHaveTextContent(
      "agent-1 via notes.tasks.set_status changed 2 tasks in Dock task note."
    )

    fireEvent.click(screen.getByRole("button", { name: "Inspect task activity" }))
    fireEvent.click(screen.getByRole("button", { name: "Dismiss task activity" }))

    expect(onInspect).toHaveBeenCalledTimes(1)
    expect(onDismiss).toHaveBeenCalledWith("event-1")
  })
})
