// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"

import { ScheduledTaskResultsPanel } from "../ScheduledTaskResultsPanel"
import { projectScheduledTaskResults } from "../scheduled-task-results"

const buildTask = (overrides: Partial<ScheduledTask>): ScheduledTask => ({
  id: "watchlist_job:1",
  primitive: "watchlist_job",
  title: "Morning monitor",
  description: "Track a source",
  status: "scheduled",
  enabled: true,
  schedule_summary: "Every morning",
  timezone: "UTC",
  next_run_at: "2030-01-02T09:00:00Z",
  last_run_at: "2030-01-01T09:00:00Z",
  edit_mode: "external",
  manage_url: "/watchlists?tab=jobs",
  source_ref: { job_id: 1 },
  ...overrides
})

const buildResults = () =>
  projectScheduledTaskResults(
    [
      buildTask({
        id: "watchlist_job:result",
        title: "Release monitor",
        source_ref: {
          job_id: 42,
          latest_run_id: 101,
          latest_output_id: 202,
          result_count: 3,
          source_label: "Release feed"
        }
      }),
      buildTask({
        id: "reminder_task:failed",
        primitive: "reminder_task",
        title: "Follow up",
        status: "failed",
        edit_mode: "native",
        manage_url: null,
        source_ref: { task_id: "failed", run_id: 77 }
      }),
      buildTask({
        id: "watchlist_job:running",
        title: "Running monitor",
        status: "running",
        source_ref: { job_id: 43, latest_run_id: 103 }
      }),
      buildTask({
        id: "watchlist_job:completed",
        title: "Quiet monitor",
        status: "completed",
        source_ref: { job_id: 44, latest_run_id: 104 }
      })
    ],
    { includeCompletedNoResults: true }
  )

describe("ScheduledTaskResultsPanel", () => {
  it("renders projected success, failure, running, and completed-no-results signals", () => {
    render(
      <ScheduledTaskResultsPanel
        results={buildResults()}
        taskCount={4}
        capabilityMode="projected_signals"
        onCreateTask={vi.fn()}
        onOpenResult={vi.fn()}
      />
    )

    expect(screen.getByRole("heading", { name: "Scheduled task results" })).toBeInTheDocument()
    expect(screen.getByText("Latest automation signals")).toBeInTheDocument()
    expect(screen.getByText("Release monitor")).toBeInTheDocument()
    expect(screen.getByText("Found 3 results from Release feed.")).toBeInTheDocument()
    expect(screen.getByText("Follow up")).toBeInTheDocument()
    expect(screen.getByText("Needs attention")).toBeInTheDocument()
    expect(screen.getByText("Running monitor")).toBeInTheDocument()
    expect(screen.getByText("Running now")).toBeInTheDocument()
    expect(screen.getByText("Quiet monitor")).toBeInTheDocument()
    expect(screen.getByText("Completed/no results")).toBeInTheDocument()
    expect(screen.getAllByText("Watchlists").length).toBeGreaterThan(0)
    expect(screen.getByText("Reminders")).toBeInTheDocument()
  })

  it("filters by result state, task type, and owner in projected mode", async () => {
    const user = userEvent.setup()
    render(
      <ScheduledTaskResultsPanel
        results={buildResults()}
        taskCount={4}
        capabilityMode="projected_signals"
        onCreateTask={vi.fn()}
        onOpenResult={vi.fn()}
      />
    )

    await user.click(screen.getByRole("combobox", { name: "Result state filter" }))
    await user.click(await screen.findByTitle("Needs attention"))

    expect(screen.getByText("Follow up")).toBeInTheDocument()
    expect(screen.queryByText("Release monitor")).not.toBeInTheDocument()

    await user.click(screen.getByRole("combobox", { name: "Task type filter" }))
    await user.click(await screen.findByTitle("Watchlist monitor"))

    expect(screen.getByText("No results match these filters")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Clear filters" }))
    await user.click(screen.getByRole("combobox", { name: "Owner filter" }))
    await user.click(await screen.findByTitle("Watchlists"))

    expect(screen.getByText("Release monitor")).toBeInTheDocument()
    expect(screen.queryByText("Follow up")).not.toBeInTheDocument()
  })

  it("hides review-state filtering in projected mode and shows it in normalized modes", () => {
    const { rerender } = render(
      <ScheduledTaskResultsPanel
        results={buildResults()}
        taskCount={4}
        capabilityMode="projected_signals"
        onCreateTask={vi.fn()}
        onOpenResult={vi.fn()}
      />
    )

    expect(screen.queryByRole("combobox", { name: "Review state filter" })).not.toBeInTheDocument()

    rerender(
      <ScheduledTaskResultsPanel
        results={buildResults()}
        taskCount={4}
        capabilityMode="normalized_results_read"
        onCreateTask={vi.fn()}
        onOpenResult={vi.fn()}
      />
    )

    expect(screen.getByRole("combobox", { name: "Review state filter" })).toBeInTheDocument()
  })

  it("distinguishes no tasks, no results, and no filter matches", async () => {
    const user = userEvent.setup()
    const onCreateTask = vi.fn()
    const { rerender } = render(
      <ScheduledTaskResultsPanel
        results={[]}
        taskCount={0}
        capabilityMode="projected_signals"
        onCreateTask={onCreateTask}
        onOpenResult={vi.fn()}
      />
    )

    expect(screen.getByText("No scheduled tasks yet")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Create scheduled task" }))
    expect(onCreateTask).toHaveBeenCalledTimes(1)

    rerender(
      <ScheduledTaskResultsPanel
        results={[]}
        taskCount={2}
        capabilityMode="projected_signals"
        onCreateTask={onCreateTask}
        onOpenResult={vi.fn()}
      />
    )

    expect(screen.getByText("No automation signals yet")).toBeInTheDocument()
  })

  it("opens a result with an accessible action name", async () => {
    const user = userEvent.setup()
    const onOpenResult = vi.fn()
    render(
      <ScheduledTaskResultsPanel
        results={buildResults()}
        taskCount={4}
        capabilityMode="projected_signals"
        onCreateTask={vi.fn()}
        onOpenResult={onOpenResult}
      />
    )

    await user.click(screen.getByRole("button", { name: "Open signal for Release monitor" }))

    expect(onOpenResult).toHaveBeenCalledWith(
      expect.objectContaining({
        taskTitle: "Release monitor",
        resultId: "202"
      })
    )
  })
})
