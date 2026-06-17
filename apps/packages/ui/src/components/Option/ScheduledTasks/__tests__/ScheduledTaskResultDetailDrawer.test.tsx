// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"
import { expectInsideDesignSystemAlert } from "@/test-utils/designSystemAlert"

import { ScheduledTaskResultDetailDrawer } from "../ScheduledTaskResultDetailDrawer"
import { projectScheduledTaskResults } from "../scheduled-task-results"

const buildTask = (overrides: Partial<ScheduledTask> = {}): ScheduledTask => ({
  id: "watchlist_job:release",
  primitive: "watchlist_job",
  title: "Release monitor",
  description: "Track release notes",
  status: "scheduled",
  enabled: true,
  schedule_summary: "Every morning",
  timezone: "UTC",
  next_run_at: "2030-01-02T09:00:00Z",
  last_run_at: "2030-01-01T09:00:00Z",
  edit_mode: "external",
  manage_url: "/watchlists?tab=jobs",
  source_ref: {
    job_id: 42,
    latest_run_id: 101,
    latest_output_id: 202,
    result_count: 3,
    source_label: "Release feed",
    matched_rule_label: "New release"
  },
  ...overrides
})

const buildResult = () =>
  projectScheduledTaskResults([buildTask()], {
    capabilityMode: "projected_signals"
  })[0]

const expectInsideDesignSystemBadge = (text: string | RegExp): HTMLElement => {
  const match = screen
    .getAllByText(text)
    .map((node) => node.closest('[data-ds-component="Badge"]'))
    .find((node): node is HTMLElement => node instanceof HTMLElement)

  expect(match).toBeTruthy()
  return match
}

describe("ScheduledTaskResultDetailDrawer", () => {
  it("shows result provenance, owner, ids, and Watchlists deep links", () => {
    const result = buildResult()

    render(
      <ScheduledTaskResultDetailDrawer
        open
        result={result}
        onClose={vi.fn()}
        onReviewResult={vi.fn()}
        onRetryRun={vi.fn()}
      />
    )

    expect(screen.getByRole("dialog", { name: /Release monitor/i })).toBeInTheDocument()
    expect(screen.getByText("Why this is here")).toBeInTheDocument()
    expect(screen.getByText("Found 3 results from Release feed.")).toBeInTheDocument()
    expectInsideDesignSystemBadge("Found results")
    expect(screen.getByText("Watchlists")).toBeInTheDocument()
    expect(screen.getByText("Release feed")).toBeInTheDocument()
    expect(screen.getByText("New release")).toBeInTheDocument()
    expect(screen.getByText("Result id")).toBeInTheDocument()
    expect(screen.getByText("202")).toBeInTheDocument()
    expect(screen.getByText("Run id")).toBeInTheDocument()
    expect(screen.getByText("101")).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Open result" })).toHaveAttribute(
      "href",
      "/watchlists?tab=outputs&output_id=202&open_output=1"
    )
    expect(screen.getByRole("link", { name: "Open run" })).toHaveAttribute(
      "href",
      "/watchlists?tab=runs&run_id=101&open_run=1"
    )
    expect(screen.getByRole("link", { name: "Open owner workspace" })).toHaveAttribute(
      "href",
      "/watchlists?tab=jobs"
    )
  })

  it("hides unsupported review and retry buttons in projected mode", () => {
    render(
      <ScheduledTaskResultDetailDrawer
        open
        result={buildResult()}
        onClose={vi.fn()}
        onReviewResult={vi.fn()}
        onRetryRun={vi.fn()}
      />
    )

    expect(screen.queryByRole("button", { name: "Mark reviewed" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Retry run" })).not.toBeInTheDocument()
    expect(
      screen.getByText("Review and retry actions appear when this server supports them for the selected result.")
    ).toBeInTheDocument()
    expectInsideDesignSystemAlert(
      "Review and retry actions appear when this server supports them for the selected result."
    )
  })

  it("shows mutation actions only when item capabilities allow them", () => {
    const reviewableResult = projectScheduledTaskResults([buildTask()], {
      capabilityMode: "normalized_results_mutation"
    })[0]
    const retryableFailure = projectScheduledTaskResults(
      [
        buildTask({
          id: "watchlist_job:failure",
          title: "Failure monitor",
          status: "failed",
          source_ref: { job_id: 42, latest_run_id: 101 }
        })
      ],
      { capabilityMode: "normalized_results_mutation" }
    )[0]

    const { rerender } = render(
      <ScheduledTaskResultDetailDrawer
        open
        result={reviewableResult}
        onClose={vi.fn()}
        onReviewResult={vi.fn()}
        onRetryRun={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: "Mark reviewed" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Retry run" })).not.toBeInTheDocument()

    rerender(
      <ScheduledTaskResultDetailDrawer
        open
        result={retryableFailure}
        onClose={vi.fn()}
        onReviewResult={vi.fn()}
        onRetryRun={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: "Retry run" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Mark reviewed" })).not.toBeInTheDocument()
  })
})
