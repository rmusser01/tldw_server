import { describe, expect, it } from "vitest"

import {
  getAutomationDefinitionFamilyLabel,
  getAutomationDefinitionProductStatus,
  isAutomationDefinitionTask
} from "../scheduled-task-automation-status"

describe("scheduled task automation status", () => {
  it("labels configured non-executable definitions without waiting-for-run copy", () => {
    const task = {
      id: "automation_definition:def_1",
      primitive: "automation_definition",
      title: "Track answer",
      status: "configured_execution_unavailable",
      enabled: true,
      edit_mode: "native",
      source_ref: {
        family: "recurring_question",
        lifecycle: "configured",
        health: "execution_unavailable",
        execution_available: false
      }
    } as const

    expect(isAutomationDefinitionTask(task)).toBe(true)
    expect(getAutomationDefinitionFamilyLabel(task)).toBe("Recurring question")
    expect(getAutomationDefinitionProductStatus(task).label).toBe(
      "Configured, execution unavailable"
    )
  })

  it("does not collapse paused automation definitions into disabled", () => {
    const status = getAutomationDefinitionProductStatus({
      id: "automation_definition:def_2",
      primitive: "automation_definition",
      title: "Agent task",
      status: "paused",
      enabled: false,
      edit_mode: "native",
      source_ref: {
        family: "agent_task",
        lifecycle: "paused",
        health: "execution_unavailable"
      }
    } as const)

    expect(status.label).toBe("Paused")
  })
})
