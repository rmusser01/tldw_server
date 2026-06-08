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

  it("accepts tasks and create tabs", () => {
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=tasks")).tab).toBe("tasks")
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=create")).tab).toBe("create")
  })

  it("falls back to overview for invalid tabs", () => {
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=runs"))).toMatchObject({
      tab: "overview",
      invalidTab: "runs"
    })
  })

  it("keeps valid template and task ids", () => {
    expect(
      parseScheduledTaskRouteState(new URLSearchParams("tab=create&template=watch"))
    ).toMatchObject({ tab: "create", templateId: "watch" })
    expect(
      parseScheduledTaskRouteState(new URLSearchParams("tab=tasks&task_id=reminder_task%3A2"))
    ).toMatchObject({ tab: "tasks", taskId: "reminder_task:2" })
  })

  it("normalizes whitespace-only parsed params to null", () => {
    expect(
      parseScheduledTaskRouteState(
        new URLSearchParams("tab=%20%20&template=%20%20&task_id=%20%20")
      )
    ).toMatchObject({
      tab: "overview",
      invalidTab: null,
      templateId: null,
      taskId: null
    })
  })

  it("builds search strings without dropping existing valid state", () => {
    expect(buildScheduledTaskSearch({ tab: "tasks", taskId: "reminder_task:2" })).toBe(
      "?tab=tasks&task_id=reminder_task%3A2"
    )
  })

  it("omits whitespace-only template and task ids when building search strings", () => {
    expect(buildScheduledTaskSearch({ tab: "create", templateId: "   " })).toBe("?tab=create")
    expect(buildScheduledTaskSearch({ tab: "tasks", taskId: "   " })).toBe("?tab=tasks")
  })

  it("exposes exactly the Phase 2A tabs", () => {
    expect(SCHEDULED_TASK_TABS.map((tab) => tab.id)).toEqual([
      "overview",
      "tasks",
      "create"
    ])
  })
})
