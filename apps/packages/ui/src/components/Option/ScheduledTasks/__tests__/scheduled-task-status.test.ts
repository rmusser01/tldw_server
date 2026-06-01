import { describe, expect, it } from "vitest"

import {
  buildWatchlistTaskLinks,
  getScheduledTaskProductStatus,
  getScheduledTaskTypeLabel
} from "../scheduled-task-status"

describe("scheduled task status helpers", () => {
  it("maps disabled tasks to Disabled before backend status text", () => {
    expect(
      getScheduledTaskProductStatus({
        id: "watchlist_job:1",
        primitive: "watchlist_job",
        title: "Monitor",
        status: "scheduled",
        enabled: false,
        edit_mode: "external",
        source_ref: { job_id: 1 }
      })
    ).toMatchObject({
      label: "Disabled",
      tone: "default"
    })
  })

  it("maps failed-like statuses to Needs attention", () => {
    expect(
      getScheduledTaskProductStatus({
        id: "reminder_task:1",
        primitive: "reminder_task",
        title: "Reminder",
        status: "failed",
        enabled: true,
        edit_mode: "native",
        source_ref: {}
      })
    ).toMatchObject({
      label: "Needs attention",
      tone: "error"
    })
  })

  it("distinguishes blocked, found-results, and draft states", () => {
    expect(
      getScheduledTaskProductStatus({
        id: "watchlist_job:2",
        primitive: "watchlist_job",
        title: "Blocked monitor",
        status: "blocked",
        enabled: true,
        edit_mode: "external",
        source_ref: {}
      }).label
    ).toBe("Blocked")

    expect(
      getScheduledTaskProductStatus({
        id: "watchlist_job:3",
        primitive: "watchlist_job",
        title: "Monitor with outputs",
        status: "scheduled",
        enabled: true,
        edit_mode: "external",
        source_ref: { result_count: 2 }
      }).label
    ).toBe("Found results")

    expect(
      getScheduledTaskProductStatus({
        id: "reminder_task:4",
        primitive: "reminder_task",
        title: "Draft reminder",
        status: "draft",
        enabled: true,
        edit_mode: "native",
        source_ref: {}
      }).label
    ).toBe("Draft")
  })

  it("builds Watchlists deep links from source_ref.job_id", () => {
    expect(
      buildWatchlistTaskLinks({
        id: "watchlist_job:42",
        primitive: "watchlist_job",
        title: "Morning brief",
        status: "scheduled",
        enabled: true,
        edit_mode: "external",
        manage_url: "/watchlists?tab=jobs",
        source_ref: { job_id: 42 }
      })
    ).toMatchObject({
      settingsUrl: "/watchlists?tab=jobs",
      activityUrl: "/watchlists?tab=runs&job_id=42",
      reportsUrl: "/watchlists?tab=outputs&job_id=42"
    })
  })

  it("builds exact Watchlists run and output links when ids are available", () => {
    expect(
      buildWatchlistTaskLinks({
        id: "watchlist_job:42",
        primitive: "watchlist_job",
        title: "Morning brief",
        status: "scheduled",
        enabled: true,
        edit_mode: "external",
        source_ref: { job_id: 42, latest_run_id: 101, latest_output_id: 202 }
      })
    ).toMatchObject({
      latestRunUrl: "/watchlists?tab=runs&run_id=101&open_run=1",
      latestOutputUrl: "/watchlists?tab=outputs&output_id=202&open_output=1"
    })
  })

  it("treats accepted output id aliases as result signals", () => {
    const task = {
      id: "watchlist_job:43",
      primitive: "watchlist_job" as const,
      title: "Alias output monitor",
      status: "scheduled",
      enabled: true,
      edit_mode: "external" as const,
      source_ref: { latestOutputId: 202 }
    }

    expect(getScheduledTaskProductStatus(task).label).toBe("Found results")
    expect(buildWatchlistTaskLinks(task).latestOutputUrl).toBe(
      "/watchlists?tab=outputs&output_id=202&open_output=1"
    )
  })

  it("uses token-aware status matching to avoid misleading substring matches", () => {
    const baseTask = {
      id: "watchlist_job:44",
      primitive: "watchlist_job" as const,
      title: "Boundary monitor",
      enabled: true,
      edit_mode: "external" as const,
      source_ref: {}
    }

    expect(getScheduledTaskProductStatus({ ...baseTask, status: "inactive" }).label).toBe(
      "Waiting for next run"
    )
    expect(getScheduledTaskProductStatus({ ...baseTask, status: "unmatched" }).label).toBe(
      "Waiting for next run"
    )
    expect(getScheduledTaskProductStatus({ ...baseTask, status: "output_error" }).label).toBe(
      "Needs attention"
    )
    expect(getScheduledTaskProductStatus({ ...baseTask, status: "in_progress" }).label).toBe(
      "Running now"
    )
  })

  it("rejects unsafe numeric ids when building Watchlists links", () => {
    expect(
      buildWatchlistTaskLinks({
        id: "watchlist_job:45",
        primitive: "watchlist_job",
        title: "Unsafe id monitor",
        status: "scheduled",
        enabled: true,
        edit_mode: "external",
        source_ref: {
          job_id: Number.MAX_SAFE_INTEGER + 1,
          latest_output_id: Number.MAX_SAFE_INTEGER + 1
        }
      })
    ).toMatchObject({
      activityUrl: null,
      reportsUrl: null,
      latestOutputUrl: null
    })
  })

  it("maps scheduled task type labels", () => {
    expect(getScheduledTaskTypeLabel({ primitive: "reminder_task" })).toBe("Reminder")
    expect(getScheduledTaskTypeLabel({ primitive: "watchlist_job" })).toBe(
      "Watchlist monitor"
    )
    expect(getScheduledTaskTypeLabel({ primitive: "future_task" })).toBe("Scheduled task")
  })
})
