import { describe, expect, it } from "vitest"

import {
  SCHEDULED_TASK_TABS,
  buildScheduledTaskSearch,
  parseScheduledTaskRouteState
} from "../scheduled-task-route-state"

describe("scheduled task route state", () => {
  it("defaults to overview when no tab is provided", () => {
    expect(parseScheduledTaskRouteState(new URLSearchParams())).toMatchObject({
      tab: "overview",
      invalidTab: null
    })
  })

  it("accepts results, tasks, and create tabs", () => {
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=results")).tab).toBe("results")
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=tasks")).tab).toBe("tasks")
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=create")).tab).toBe("create")
  })

  it("falls back to overview for invalid tabs", () => {
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=runs"))).toMatchObject({
      tab: "overview",
      invalidTab: "runs"
    })
  })

  it("keeps valid template, task, run, and result ids", () => {
    expect(
      parseScheduledTaskRouteState(new URLSearchParams("tab=create&template=watch"))
    ).toMatchObject({ tab: "create", templateId: "watch" })
    expect(
      parseScheduledTaskRouteState(new URLSearchParams("tab=tasks&task_id=reminder_task%3A2"))
    ).toMatchObject({ tab: "tasks", taskId: "reminder_task:2" })
    expect(
      parseScheduledTaskRouteState(
        new URLSearchParams(
          "tab=results&result_id=202&run_id=101&task_id=watchlist_job%3A2"
        )
      )
    ).toMatchObject({
      tab: "results",
      resultId: "202",
      runId: "101",
      taskId: "watchlist_job:2"
    })
  })

  it("normalizes whitespace-only parsed params to null", () => {
    expect(
      parseScheduledTaskRouteState(
        new URLSearchParams(
          "tab=%20%20&template=%20%20&task_id=%20%20&run_id=%20%20&result_id=%20%20"
        )
      )
    ).toMatchObject({
      tab: "overview",
      invalidTab: null,
      templateId: null,
      taskId: null,
      runId: null,
      resultId: null
    })
  })

  it("rejects newline-bearing parsed ids instead of serializing unsafe route state", () => {
    expect(
      parseScheduledTaskRouteState(
        new URLSearchParams("tab=results&result_id=202%0Asecret&run_id=101%0Dsecret")
      )
    ).toMatchObject({
      tab: "results",
      resultId: null,
      runId: null
    })
  })

  it("builds search strings without dropping existing valid state", () => {
    expect(buildScheduledTaskSearch({ tab: "tasks", taskId: "reminder_task:2" })).toBe(
      "?tab=tasks&task_id=reminder_task%3A2"
    )
    expect(
      buildScheduledTaskSearch({
        tab: "results",
        resultId: "202",
        runId: "101",
        taskId: "watchlist_job:2"
      })
    ).toBe("?tab=results&result_id=202&run_id=101&task_id=watchlist_job%3A2")
  })

  it("omits whitespace-only template and task ids when building search strings", () => {
    expect(buildScheduledTaskSearch({ tab: "create", templateId: "   " })).toBe("?tab=create")
    expect(buildScheduledTaskSearch({ tab: "tasks", taskId: "   " })).toBe("?tab=tasks")
    expect(buildScheduledTaskSearch({ tab: "results", resultId: "   ", runId: "   " })).toBe(
      "?tab=results"
    )
  })

  it("uses the Phase 3 scheduled task information architecture tabs", () => {
    expect(SCHEDULED_TASK_TABS.map((tab) => tab.id)).toEqual([
      "overview",
      "results",
      "tasks",
      "create"
    ])
  })

  it("can default alias routes into the Results tab without a tab query param", () => {
    expect(
      parseScheduledTaskRouteState(new URLSearchParams("result_id=202"), {
        defaultTab: "results"
      })
    ).toMatchObject({
      tab: "results",
      resultId: "202"
    })
  })
})
