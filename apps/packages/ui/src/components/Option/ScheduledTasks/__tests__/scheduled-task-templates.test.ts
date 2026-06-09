import { describe, expect, it } from "vitest"

import { UNAVAILABLE_STATE_LABEL } from "@/design-system"

import {
  SCHEDULED_TASK_TEMPLATE_FILTERS,
  SCHEDULED_TASK_TEMPLATES,
  filterScheduledTaskTemplates,
  findScheduledTaskTemplates,
  getScheduledTaskTemplate,
  getScheduledTaskTemplateStateLabel,
  toSafeHandoffSourceText
} from "../scheduled-task-templates"

describe("scheduled task templates", () => {
  it("keeps Reminder as the only available Phase 2A creation template", () => {
    expect(
      SCHEDULED_TASK_TEMPLATES.filter((template) => template.state === "available").map(
        (template) => template.id
      )
    ).toEqual(["reminder"])
  })

  it("marks Watch, Ingest, and Advanced as handoff-only", () => {
    expect(getScheduledTaskTemplate("watch")?.state).toBe("handoff_only")
    expect(getScheduledTaskTemplate("ingest")?.state).toBe("handoff_only")
    expect(getScheduledTaskTemplate("advanced")?.state).toBe("handoff_only")
  })

  it("marks Recurring Question and Agent Task as planned", () => {
    expect(getScheduledTaskTemplate("recurring_question")?.state).toBe("planned")
    expect(getScheduledTaskTemplate("agent_task")?.state).toBe("planned")
  })

  it("matches prompt text deterministically without inferring config", () => {
    expect(
      findScheduledTaskTemplates("keep this channel searchable").map(
        (template) => template.id
      )
    ).toContain("ingest")
    expect(
      findScheduledTaskTemplates("send this prompt to an agent tomorrow").map(
        (template) => template.id
      )
    ).toContain("agent_task")
    expect(
      findScheduledTaskTemplates("watch new issues").map((template) => template.id)
    ).toContain("watch")
  })

  it("matches prompt text case-insensitively and returns no matches for empty text", () => {
    expect(
      findScheduledTaskTemplates("REMIND me weekly").map((template) => template.id)
    ).toContain("reminder")
    expect(findScheduledTaskTemplates("   ")).toEqual([])
  })

  it("matches keywords by word or phrase boundaries instead of substrings", () => {
    expect(
      findScheduledTaskTemplates("renew credentials").map((template) => template.id)
    ).not.toContain("watch")
    expect(
      findScheduledTaskTemplates("paragraph summary").map((template) => template.id)
    ).not.toContain("recurring_question")
  })

  it("keeps registry ordering when keyword match scores tie", () => {
    expect(findScheduledTaskTemplates("new index").map((template) => template.id)).toEqual([
      "watch",
      "ingest"
    ])
  })

  it("filters templates by availability and category", () => {
    expect(
      filterScheduledTaskTemplates("available_now").map((template) => template.id)
    ).toEqual(["reminder"])
    expect(filterScheduledTaskTemplates("agent").map((template) => template.id)).toEqual([
      "agent_task"
    ])
  })

  it("labels Limited availability", () => {
    expect(getScheduledTaskTemplateStateLabel("limited_availability")).toBe(
      "Limited availability"
    )
  })

  it("does not include Limited availability in Available now", () => {
    const templates = [
      { ...getScheduledTaskTemplate("watch")!, state: "limited_availability" as const }
    ]

    expect(filterScheduledTaskTemplates("available_now", templates)).toEqual([])
  })

  it("can look up templates from an effective template list", () => {
    const templates = [
      { ...getScheduledTaskTemplate("watch")!, state: "limited_availability" as const }
    ]

    expect(getScheduledTaskTemplate("watch", templates)?.state).toBe("limited_availability")
  })

  it("sanitizes unsafe handoff source text", () => {
    expect(toSafeHandoffSourceText("https://example.com/feed?token=secret")).toBe(null)
    expect(toSafeHandoffSourceText("https://example.com/feed")).toBe("https://example.com/feed")
    expect(toSafeHandoffSourceText("  ")).toBe(null)
  })

  it("rejects URL fragments and private URL params in handoff source text", () => {
    expect(toSafeHandoffSourceText("https://example.com/feed#private")).toBe(null)
    expect(toSafeHandoffSourceText("example.com/feed#private")).toBe(null)
    expect(toSafeHandoffSourceText("https://example.com/feed?api_key=secret")).toBe(null)
    expect(toSafeHandoffSourceText("repository issues URL")).toBe("repository issues URL")
  })

  it("rejects compound sensitive URL params in handoff source text", () => {
    expect(toSafeHandoffSourceText("https://example.com/feed?access_token=secret")).toBe(
      null
    )
    expect(toSafeHandoffSourceText("https://example.com/feed?refresh_token=secret")).toBe(
      null
    )
    expect(toSafeHandoffSourceText("https://example.com/feed?id_token=secret")).toBe(null)
    expect(toSafeHandoffSourceText("https://example.com/feed?client_secret=secret")).toBe(
      null
    )
    expect(toSafeHandoffSourceText("https://example.com/feed")).toBe("https://example.com/feed")
  })

  it("rejects non-string handoff source text", () => {
    expect(toSafeHandoffSourceText(null)).toBe(null)
    expect(toSafeHandoffSourceText(123)).toBe(null)
    expect(toSafeHandoffSourceText({})).toBe(null)
  })

  it("exposes the expected state labels", () => {
    expect(getScheduledTaskTemplateStateLabel("available")).toBe("Available")
    expect(getScheduledTaskTemplateStateLabel("handoff_only")).toBe("Handoff only")
    expect(getScheduledTaskTemplateStateLabel("needs_setup")).toBe("Needs setup")
    expect(getScheduledTaskTemplateStateLabel("managed_in_watchlists")).toBe(
      "Managed in Watchlists"
    )
    expect(getScheduledTaskTemplateStateLabel("planned")).toBe("Planned capability")
    expect(getScheduledTaskTemplateStateLabel("unavailable")).toBe(UNAVAILABLE_STATE_LABEL)
  })

  it("exposes the expected filter list", () => {
    expect(SCHEDULED_TASK_TEMPLATE_FILTERS.map((filter) => filter.id)).toEqual([
      "all",
      "available_now",
      "watch",
      "ingest",
      "research",
      "agent",
      "advanced"
    ])
  })
})
