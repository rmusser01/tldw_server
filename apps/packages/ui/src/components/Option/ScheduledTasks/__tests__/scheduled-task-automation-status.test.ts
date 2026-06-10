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

  it("maps ready backend-projected definitions to configured", () => {
    const status = getAutomationDefinitionProductStatus({
      id: "automation_definition:def_ready",
      primitive: "automation_definition",
      title: "Ready task",
      status: "ready",
      enabled: true,
      edit_mode: "native",
      source_ref: {
        family: "agent_task",
        lifecycle: "configured",
        health: "ready"
      }
    } as const)

    expect(status).toMatchObject({
      key: "waiting",
      label: "Configured",
      tone: "processing"
    })
  })

  it("maps capability-blocked backend projection to blocked warning copy", () => {
    const status = getAutomationDefinitionProductStatus({
      id: "automation_definition:def_capability",
      primitive: "automation_definition",
      title: "Capability blocked task",
      status: "blocked_capability_unavailable",
      enabled: true,
      edit_mode: "native",
      source_ref: {
        family: "agent_task",
        lifecycle: "configured",
        health: "capability_unavailable"
      }
    } as const)

    expect(status).toMatchObject({
      key: "blocked",
      label: "Execution unavailable",
      tone: "warning"
    })
  })

  it("maps permission-blocked backend projection to blocked warning copy", () => {
    const status = getAutomationDefinitionProductStatus({
      id: "automation_definition:def_permission",
      primitive: "automation_definition",
      title: "Permission blocked task",
      status: "blocked_permission_required",
      enabled: true,
      edit_mode: "native",
      source_ref: {
        family: "recurring_question",
        lifecycle: "configured",
        health: "permission_required"
      }
    } as const)

    expect(status).toMatchObject({
      key: "blocked",
      label: "Execution unavailable",
      tone: "warning"
    })
  })

  it("maps needs-attention backend projection intentionally", () => {
    const status = getAutomationDefinitionProductStatus({
      id: "automation_definition:def_attention",
      primitive: "automation_definition",
      title: "Attention task",
      status: "needs_attention",
      enabled: true,
      edit_mode: "native",
      source_ref: {
        family: "agent_task",
        lifecycle: "configured",
        health: "needs_attention"
      }
    } as const)

    expect(status).toMatchObject({
      key: "needs_attention",
      label: "Needs attention",
      tone: "error"
    })
  })

  it("treats unknown automation lifecycle and status as needing attention", () => {
    const status = getAutomationDefinitionProductStatus({
      id: "automation_definition:def_3",
      primitive: "automation_definition",
      title: "Future task",
      status: "warming_up",
      enabled: true,
      edit_mode: "native",
      source_ref: {
        family: "agent_task",
        lifecycle: "warming_up",
        health: "initializing"
      }
    } as const)

    expect(status).toMatchObject({
      key: "needs_attention",
      label: "Needs attention"
    })
    expect(status.description).toContain("unrecognized")
  })

  it("does not treat unknown automation status as ready when lifecycle is configured", () => {
    const status = getAutomationDefinitionProductStatus({
      id: "automation_definition:def_5",
      primitive: "automation_definition",
      title: "Future status",
      status: "awaiting_worker",
      enabled: true,
      edit_mode: "native",
      source_ref: {
        family: "agent_task",
        lifecycle: "configured",
        health: "ready"
      }
    } as const)

    expect(status).toMatchObject({
      key: "needs_attention",
      label: "Needs attention"
    })
  })

  it("does not mark automation definitions with missing source metadata as configured", () => {
    const status = getAutomationDefinitionProductStatus({
      id: "automation_definition:def_4",
      primitive: "automation_definition",
      title: "Missing metadata",
      status: "configured",
      enabled: true,
      edit_mode: "native",
      source_ref: {}
    } as const)

    expect(status).toMatchObject({
      key: "needs_attention",
      label: "Needs attention"
    })
  })
})
