import { describe, expect, it } from "vitest"

import {
  buildPlannedScheduledTaskPanelModel,
  isPlannedAutomationTemplate,
  type PlannedScheduledTaskLink,
  type PlannedScheduledTaskRequirement
} from "../scheduled-task-planned-template-copy"

interface MutablePlannedModel {
  requirements: PlannedScheduledTaskRequirement[]
  resultDestinations: string[]
  safetyLines: string[]
  links: PlannedScheduledTaskLink[]
}

describe("scheduled task planned template copy", () => {
  it("builds API-first Recurring Question copy without executable support", () => {
    const model = buildPlannedScheduledTaskPanelModel("recurring_question")

    expect(model).not.toBeNull()
    expect(model?.templateId).toBe("recurring_question")
    expect(model?.statusLabel).toBe("Planned automation type")
    expect(model?.jobStatement).toBe(
      "Run this question on a schedule across selected searchable content."
    )
    expect(model?.availabilityReason).toBe(
      "Recurring Question scheduling is planned for the API contract and is not executable in this client yet."
    )
    expect(model?.requirements).toContainEqual(
      expect.objectContaining({ label: "Scheduled RAG query support" })
    )
    expect(model?.requirements).toContainEqual(
      expect.objectContaining({ label: "Searchable scope selection" })
    )
    expect(model?.requirements).toContainEqual(
      expect.objectContaining({ label: "Normalized run history" })
    )
    expect(model?.requirements).toContainEqual(
      expect.objectContaining({ label: "Task visibility policy" })
    )
    expect(model?.resultDestinations).toContain(
      "Every run is recorded in task history."
    )
    expect(model?.resultDestinations).toContain(
      "Home and Results receive summaries only when selected by the task visibility policy."
    )
    expect(model?.links).toEqual([
      { label: "Open Research", href: "/research" },
      { label: "Open Results", href: "/scheduled-tasks/results" }
    ])
    expect(model?.createEnabled).toBe(false)
  })

  it("builds API-first Agent Task copy with preview and approval expectations", () => {
    const model = buildPlannedScheduledTaskPanelModel("agent_task")

    expect(model).not.toBeNull()
    expect(model?.templateId).toBe("agent_task")
    expect(model?.statusLabel).toBe("Planned automation type")
    expect(model?.jobStatement).toBe(
      "Send this message to the selected agent at the scheduled time."
    )
    expect(model?.availabilityReason).toBe(
      "Agent Task scheduling is planned for the API contract and is not executable in this client yet."
    )
    expect(model?.requirements).toContainEqual(
      expect.objectContaining({ label: "Schedulable ACP/API agents" })
    )
    expect(model?.requirements).toContainEqual(
      expect.objectContaining({ label: "Preview and risk classification" })
    )
    expect(model?.requirements).toContainEqual(
      expect.objectContaining({ label: "Approval policy" })
    )
    expect(model?.requirements).toContainEqual(
      expect.objectContaining({ label: "Normalized agent run outputs" })
    )
    expect(model?.safetyLines).toContain(
      "Preview is required before scheduling an agent task."
    )
    expect(model?.safetyLines).toContain(
      "Some permission classes may require approval before each run."
    )
    expect(model?.links).toEqual([
      { label: "Open Agent Tasks", href: "/agent-tasks" },
      { label: "Open ACP Playground", href: "/acp-playground" },
      { label: "Open Results", href: "/scheduled-tasks/results" }
    ])
    expect(model?.createEnabled).toBe(false)
  })

  it("identifies only planned automation templates", () => {
    expect(isPlannedAutomationTemplate("recurring_question")).toBe(true)
    expect(isPlannedAutomationTemplate("agent_task")).toBe(true)
    expect(isPlannedAutomationTemplate("watch")).toBe(false)
    expect(isPlannedAutomationTemplate("reminder")).toBe(false)
    expect(isPlannedAutomationTemplate("ingest")).toBe(false)
    expect(isPlannedAutomationTemplate("advanced")).toBe(false)
  })

  it("treats non-planned families as unsupported by this helper", () => {
    expect(buildPlannedScheduledTaskPanelModel("watch")).toBeNull()
  })

  it("returns defensive copies so caller mutations cannot leak to later callers", () => {
    const first = buildPlannedScheduledTaskPanelModel("recurring_question")
    const second = buildPlannedScheduledTaskPanelModel("recurring_question")

    expect(first).not.toBeNull()
    expect(second).not.toBeNull()
    expect(first).not.toBe(second)
    expect(first?.requirements).not.toBe(second?.requirements)
    expect(first?.resultDestinations).not.toBe(second?.resultDestinations)
    expect(first?.safetyLines).not.toBe(second?.safetyLines)
    expect(first?.links).not.toBe(second?.links)

    const mutableFirst = first as MutablePlannedModel
    mutableFirst.requirements.push({
      label: "Leaked mutation",
      detail: "This should not persist.",
      status: "missing"
    })
    mutableFirst.resultDestinations.push("Leaked destination")
    mutableFirst.safetyLines.push("Leaked safety line")
    mutableFirst.links[0] = { label: "Leaked link", href: "/leaked" }

    const later = buildPlannedScheduledTaskPanelModel("recurring_question")

    expect(later?.requirements).not.toContainEqual(
      expect.objectContaining({ label: "Leaked mutation" })
    )
    expect(later?.resultDestinations).not.toContain("Leaked destination")
    expect(later?.safetyLines).not.toContain("Leaked safety line")
    expect(later?.links[0]).toEqual({ label: "Open Research", href: "/research" })
  })
})
