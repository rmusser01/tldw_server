import { describe, expect, it } from "vitest"

import type { ScheduledTask } from "@/services/scheduled-tasks-control-plane"

import {
  buildScheduledTaskResultDedupeKey,
  buildScheduledTaskResultHref,
  mergeScheduledTaskNotificationTargets,
  normalizeScheduledTaskNotificationTarget
} from "../scheduled-task-result-links"
import {
  buildScheduledTaskAutomationHomeItems,
  buildScheduledTaskAutomationHomeItemsFromNotifications,
  buildScheduledTaskCompanionHomeItems,
  findScheduledTaskResultByRouteState,
  filterScheduledTaskResults,
  mergeScheduledTaskAutomationHomeItems,
  projectScheduledTaskResults,
  resolveScheduledTaskResultsCapabilityMode
} from "../scheduled-task-results"

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

describe("scheduled task result helpers", () => {
  it("defaults to projected signals when normalized result endpoints are unavailable", () => {
    expect(resolveScheduledTaskResultsCapabilityMode({})).toBe("projected_signals")
    expect(
      resolveScheduledTaskResultsCapabilityMode({
        "/api/v1/scheduled-tasks/results": {
          get: {}
        }
      })
    ).toBe("normalized_results_read")
    expect(
      resolveScheduledTaskResultsCapabilityMode({
        "/api/v1/scheduled-tasks/results": { get: {} },
        "/api/v1/scheduled-tasks/results/{result_id}/review": { post: {} }
      })
    ).toBe("normalized_results_mutation")
  })

  it("projects found-result tasks into non-reviewable success signals in projected mode", () => {
    const [result] = projectScheduledTaskResults([
      buildTask({
        title: "Release monitor",
        status: "scheduled",
        source_ref: {
          job_id: 42,
          latest_run_id: 101,
          latest_output_id: 202,
          result_count: 3,
          source_label: "Release feed",
          matched_rule_label: "New items"
        }
      })
    ])

    expect(result).toMatchObject({
      capabilityMode: "projected_signals",
      signalKind: "result",
      state: "new",
      severity: "success",
      reviewed: false,
      reviewAvailable: false,
      retryAvailable: false,
      taskTitle: "Release monitor",
      resultId: "202",
      runId: "101",
      sourceLabel: "Release feed",
      matchedRuleLabel: "New items",
      owner: "watchlists",
      ownerLabel: "Watchlists"
    })
    expect(result?.primaryHref).toBe("/scheduled-tasks?tab=results&result_id=202")
    expect(result?.resultHref).toBe("/watchlists?tab=outputs&output_id=202&open_output=1")
    expect(result?.runHref).toBe("/watchlists?tab=runs&run_id=101&open_run=1")
  })

  it("projects failed tasks into attention signals with recovery copy", () => {
    const [result] = projectScheduledTaskResults([
      buildTask({
        id: "reminder_task:failed",
        primitive: "reminder_task",
        title: "Follow up",
        status: "failed",
        edit_mode: "native",
        manage_url: null,
        source_ref: {
          task_id: "failed",
          run_id: 77,
          error_msg: "Request timed out"
        }
      })
    ])

    expect(result).toMatchObject({
      signalKind: "failure",
      state: "failed",
      severity: "error",
      taskId: "reminder_task:failed",
      runId: "77",
      retryAvailable: false
    })
    expect(result?.summary).toContain("needs attention")
    expect(result?.primaryHref).toBe("/scheduled-tasks?tab=results&run_id=77")
  })

  it("keeps result and failure signals separate for failed tasks that produced output", () => {
    const results = projectScheduledTaskResults([
      buildTask({
        title: "Mixed state monitor",
        status: "failed with results",
        source_ref: {
          job_id: 42,
          latest_run_id: 101,
          latest_output_id: 202,
          result_count: 1
        }
      })
    ])

    expect(results.map((result) => result.signalKind).sort()).toEqual([
      "failure",
      "result"
    ])
    expect(new Set(results.map((result) => result.dedupeKey)).size).toBe(2)
  })

  it("does not leak private-looking source reference values into provenance", () => {
    const [result] = projectScheduledTaskResults([
      buildTask({
        title: "Private source monitor",
        source_ref: {
          latest_output_id: 202,
          result_count: 1,
          source_label: "https://example.com/feed?token=secret",
          link_url: "https://example.com/feed?api_key=secret",
          matched_rule_label: "Authorization: Bearer secret"
        }
      })
    ])

    const serialized = JSON.stringify(result)
    expect(serialized).not.toContain("token=secret")
    expect(serialized).not.toContain("api_key=secret")
    expect(serialized).not.toContain("Bearer secret")
    expect(result?.sourceLabel).toBeNull()
    expect(result?.matchedRuleLabel).toBeNull()
  })

  it("excludes waiting tasks and completed-no-result signals from default Home items", () => {
    const results = projectScheduledTaskResults(
      [
        buildTask({ id: "watchlist_job:waiting", title: "Waiting", source_ref: { job_id: 1 } }),
        buildTask({
          id: "watchlist_job:completed",
          title: "Completed",
          status: "completed",
          source_ref: { job_id: 2 }
        }),
        buildTask({
          id: "watchlist_job:result",
          title: "Found",
          source_ref: { job_id: 3, latest_output_id: 303, result_count: 1 }
        })
      ],
      { includeCompletedNoResults: true }
    )

    expect(results.some((result) => result.state === "completed_no_results")).toBe(true)
    expect(filterScheduledTaskResults(results, { states: ["completed_no_results"] })).toHaveLength(1)
    expect(buildScheduledTaskAutomationHomeItems(results).map((item) => item.title)).toEqual([
      "Found"
    ])
  })

  it("maps scheduled-task automation items to Companion Home item shape", () => {
    const results = projectScheduledTaskResults([
      buildTask({
        id: "watchlist_job:result",
        title: "Found",
        source_ref: { job_id: 3, latest_output_id: 303, result_count: 1 }
      })
    ])

    expect(buildScheduledTaskCompanionHomeItems(buildScheduledTaskAutomationHomeItems(results))).toEqual([
      expect.objectContaining({
        id: "automation:result:303",
        entityId: "result:303",
        entityType: "scheduled_task_result",
        source: "scheduled_task",
        title: "Found",
        href: "/scheduled-tasks?tab=results&result_id=303"
      })
    ])
  })

  it("builds safe result hrefs and dedupe keys", () => {
    expect(buildScheduledTaskResultHref({ resultId: "202", runId: "101", taskId: "x" })).toBe(
      "/scheduled-tasks?tab=results&result_id=202"
    )
    expect(buildScheduledTaskResultHref({ runId: "101", taskId: "x" })).toBe(
      "/scheduled-tasks?tab=results&run_id=101"
    )
    expect(buildScheduledTaskResultHref({ taskId: "watchlist_job:42" })).toBe(
      "/scheduled-tasks?tab=results&task_id=watchlist_job%3A42"
    )
    expect(
      buildScheduledTaskResultDedupeKey({
        signalKind: "failure",
        taskId: "watchlist_job:42",
        runId: null,
        resultId: null,
        state: "failed",
        occurredAt: "2030-01-01T09:00:00Z"
      })
    ).toBe("task:watchlist_job:42:state:failure:time:2030-01-01T09:00:00Z")
  })

  it("finds projected results by result, run, then task route state", () => {
    const results = projectScheduledTaskResults([
      buildTask({
        id: "watchlist_job:release",
        title: "Release monitor",
        source_ref: {
          job_id: 42,
          latest_run_id: 101,
          latest_output_id: 202,
          result_count: 1
        }
      })
    ])

    expect(findScheduledTaskResultByRouteState(results, { resultId: "202" })?.taskTitle).toBe(
      "Release monitor"
    )
    expect(findScheduledTaskResultByRouteState(results, { runId: "101" })?.taskTitle).toBe(
      "Release monitor"
    )
    expect(
      findScheduledTaskResultByRouteState(results, { taskId: "watchlist_job:release" })
        ?.taskTitle
    ).toBe("Release monitor")
    expect(findScheduledTaskResultByRouteState(results, { resultId: "missing" })).toBeNull()
  })

  it("normalizes notification targets without replacing notification behavior", () => {
    expect(
      normalizeScheduledTaskNotificationTarget({
        id: 9,
        kind: "job_completed",
        title: "Done",
        message: "Output ready",
        severity: "info",
        created_at: "2030-01-01T09:00:00Z",
        link_type: "scheduled_task_result",
        link_id: "202",
        source_task_id: "watchlist_job:42",
        source_task_run_id: "101"
      })
    ).toMatchObject({
      notificationId: 9,
      resultId: "202",
      runId: "101",
      taskId: "watchlist_job:42",
      href: "/scheduled-tasks?tab=results&result_id=202"
    })
  })

  it("normalizes notification targets by exact result, run, then task priority across link shapes", () => {
    expect(
      normalizeScheduledTaskNotificationTarget({
        id: 10,
        kind: "job_completed",
        title: "Output ready",
        message: "Output ready",
        severity: "info",
        created_at: "2030-01-01T09:00:00Z",
        link_type: "scheduled_task_run",
        link_id: "101",
        link_url: "/scheduled-tasks?tab=results&result_id=202&run_id=303&task_id=watchlist_job%3A42",
        source_task_id: "watchlist_job:42",
        source_task_run_id: "404"
      })
    ).toMatchObject({
      resultId: "202",
      runId: "404",
      taskId: "watchlist_job:42",
      href: "/scheduled-tasks?tab=results&result_id=202",
      dedupeKey: "result:202"
    })

    expect(
      normalizeScheduledTaskNotificationTarget({
        id: 11,
        kind: "job_failed",
        title: "Run failed",
        message: "Run failed",
        severity: "error",
        created_at: "2030-01-01T09:05:00Z",
        link_type: "scheduled_task_run",
        link_id: "101",
        source_job_id: 42
      })
    ).toMatchObject({
      resultId: null,
      runId: "101",
      taskId: "watchlist_job:42",
      href: "/scheduled-tasks?tab=results&run_id=101",
      dedupeKey: "run:101:state:failure"
    })

    expect(
      normalizeScheduledTaskNotificationTarget({
        id: 12,
        kind: "job_completed",
        title: "Task completed",
        message: "Task completed",
        severity: "info",
        created_at: "2030-01-01T09:10:00Z",
        link_type: "scheduled_task",
        link_id: "watchlist_job:42"
      })
    ).toMatchObject({
      resultId: null,
      runId: null,
      taskId: "watchlist_job:42",
      href: "/scheduled-tasks?tab=results&task_id=watchlist_job%3A42"
    })
  })

  it("shares dedupe keys between projected task signals and notification-derived targets", () => {
    const [failureResult] = projectScheduledTaskResults([
      buildTask({
        id: "watchlist_job:failure",
        title: "Failure monitor",
        status: "failed",
        source_ref: {
          job_id: 42,
          latest_run_id: 101
        }
      })
    ])
    const notificationTarget = normalizeScheduledTaskNotificationTarget({
      id: 13,
      kind: "job_failed",
      title: "Run failed",
      message: "Run failed",
      severity: "error",
      created_at: "2030-01-01T09:00:00Z",
      source_task_id: "watchlist_job:failure",
      source_task_run_id: "101"
    })

    expect(notificationTarget?.dedupeKey).toBe(failureResult?.dedupeKey)

    expect(
      buildScheduledTaskResultDedupeKey({
        signalKind: "failure",
        taskId: "watchlist_job:42",
        runId: "101",
        resultId: null
      })
    ).not.toBe(
      buildScheduledTaskResultDedupeKey({
        signalKind: "result",
        taskId: "watchlist_job:42",
        runId: "101",
        resultId: null
      })
    )
  })

  it("merges notification targets by dedupe key while preserving notification ids", () => {
    const first = normalizeScheduledTaskNotificationTarget({
      id: 14,
      kind: "job_completed",
      title: "Output ready",
      message: "Output ready",
      severity: "info",
      created_at: "2030-01-01T09:00:00Z",
      link_type: "scheduled_task_result",
      link_id: "202",
      source_task_id: "watchlist_job:42",
      source_task_run_id: "101"
    })
    const duplicate = normalizeScheduledTaskNotificationTarget({
      id: 15,
      kind: "job_completed",
      title: "Output ready again",
      message: "Output ready again",
      severity: "info",
      created_at: "2030-01-01T09:05:00Z",
      link_url: "/scheduled-tasks?tab=results&result_id=202",
      source_task_id: "watchlist_job:42",
      source_task_run_id: "101"
    })
    const distinctRun = normalizeScheduledTaskNotificationTarget({
      id: 16,
      kind: "job_failed",
      title: "Next run failed",
      message: "Next run failed",
      severity: "error",
      created_at: "2030-01-01T09:10:00Z",
      source_task_id: "watchlist_job:42",
      source_task_run_id: "102"
    })

    const merged = mergeScheduledTaskNotificationTargets([
      first,
      duplicate,
      distinctRun,
      null
    ])

    expect(merged).toHaveLength(2)
    expect(merged[0]).toMatchObject({
      dedupeKey: "result:202",
      notificationIds: [14, 15]
    })
    expect(merged[1]).toMatchObject({
      dedupeKey: "run:102:state:failure",
      notificationIds: [16]
    })
  })

  it("dedupes Home automation items from projected tasks and notifications", () => {
    const projectedItems = buildScheduledTaskAutomationHomeItems(
      projectScheduledTaskResults([
        buildTask({
          id: "watchlist_job:release",
          title: "Release monitor",
          source_ref: {
            job_id: 42,
            latest_run_id: 101,
            latest_output_id: 202,
            result_count: 1
          }
        })
      ])
    )
    const notificationItems = buildScheduledTaskAutomationHomeItemsFromNotifications([
      {
        id: 17,
        kind: "job_completed",
        title: "Release monitor notification",
        message: "Output ready",
        severity: "info",
        created_at: "2030-01-01T09:05:00Z",
        link_type: "scheduled_task_result",
        link_id: "202",
        source_task_id: "watchlist_job:release",
        source_task_run_id: "101"
      },
      {
        id: 18,
        kind: "job_failed",
        title: "Different run failed",
        message: "The latest run failed.",
        severity: "error",
        created_at: "2030-01-01T09:10:00Z",
        source_task_id: "watchlist_job:release",
        source_task_run_id: "102"
      }
    ])

    const merged = mergeScheduledTaskAutomationHomeItems([
      projectedItems,
      notificationItems
    ])

    expect(merged.map((item) => item.dedupeKey)).toEqual([
      "run:102:state:failure",
      "result:202"
    ])
    expect(merged.filter((item) => item.dedupeKey === "result:202")).toHaveLength(1)
    expect(merged.find((item) => item.dedupeKey === "result:202")?.title).toBe(
      "Release monitor notification"
    )
  })
})
