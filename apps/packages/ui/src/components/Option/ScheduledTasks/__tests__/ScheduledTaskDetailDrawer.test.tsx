// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"
import { ScheduledTaskDetailDrawer } from "../ScheduledTaskDetailDrawer"

const reminderTask: ScheduledTask = {
  id: "reminder_task:1",
  primitive: "reminder_task",
  title: "Review notes",
  description: "Check the backlog",
  status: "scheduled",
  enabled: true,
  schedule_summary: "2026-03-21T09:00:00+00:00",
  timezone: "UTC",
  next_run_at: "2030-04-05T12:30:00+00:00",
  last_run_at: null,
  edit_mode: "native",
  manage_url: null,
  source_ref: {
    task_id: "1",
    link_type: "note",
    link_id: "note-7",
    link_url: "/notes/note-7"
  }
}

const watchlistTask: ScheduledTask = {
  id: "watchlist_job:42",
  primitive: "watchlist_job",
  title: "Morning digest",
  description: "Watchlist run",
  status: "found results",
  enabled: true,
  schedule_summary: "0 9 * * *",
  timezone: "UTC",
  next_run_at: "2030-04-06T09:00:00+00:00",
  last_run_at: "2030-04-05T09:00:00+00:00",
  edit_mode: "external",
  manage_url: "/watchlists?tab=jobs",
  source_ref: {
    job_id: 42,
    scope: "source tuning",
    latest_run_id: 101,
    latest_output_id: 202
  }
}

describe("ScheduledTaskDetailDrawer", () => {
  it("shows reminder task details and native reminder actions", () => {
    render(
      <ScheduledTaskDetailDrawer
        open
        task={reminderTask}
        onClose={vi.fn()}
        onEditReminder={vi.fn()}
        onDeleteReminder={vi.fn()}
      />
    )

    expect(screen.getByRole("dialog", { name: /Review notes/i })).toBeInTheDocument()
    expect(screen.getByText("Reminder")).toBeInTheDocument()
    expect(screen.getByText("Managed here")).toBeInTheDocument()
    expect(screen.getByText("2026-03-21T09:00:00+00:00")).toBeInTheDocument()
    expect(screen.getByText("Reminder task id")).toBeInTheDocument()
    expect(screen.getByText("1")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Edit reminder" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete reminder" })).toBeInTheDocument()
  })

  it("shows Watchlists task links without moving workspace configuration into the drawer", () => {
    render(
      <ScheduledTaskDetailDrawer
        open
        task={watchlistTask}
        onClose={vi.fn()}
        onEditReminder={vi.fn()}
        onDeleteReminder={vi.fn()}
      />
    )

    expect(screen.getByRole("dialog", { name: /Morning digest/i })).toBeInTheDocument()
    expect(screen.getByText("Watchlist monitor")).toBeInTheDocument()
    expect(screen.getByText("Managed in Watchlists")).toBeInTheDocument()
    expect(screen.getByText("Watchlists job id")).toBeInTheDocument()
    expect(screen.getByText("42")).toBeInTheDocument()
    expect(screen.getByText("source tuning")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open monitor settings" })).toHaveAttribute(
      "href",
      "/watchlists?tab=jobs"
    )
    expect(screen.getByRole("link", { name: "Open activity" })).toHaveAttribute(
      "href",
      "/watchlists?tab=runs&job_id=42"
    )
    expect(screen.getByRole("link", { name: "Open reports" })).toHaveAttribute(
      "href",
      "/watchlists?tab=outputs&job_id=42"
    )
    expect(screen.getByRole("link", { name: "Open latest run" })).toHaveAttribute(
      "href",
      "/watchlists?tab=runs&run_id=101&open_run=1"
    )
    expect(screen.getByRole("link", { name: "Open latest report" })).toHaveAttribute(
      "href",
      "/watchlists?tab=outputs&output_id=202&open_output=1"
    )
    expect(screen.getByText(/Watchlists remains the full workspace/i)).toBeInTheDocument()
  })
})
