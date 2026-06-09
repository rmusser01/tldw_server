import { describe, expect, it } from "vitest"

import {
  SCHEDULED_TASK_TEMPLATES,
  type ScheduledTaskTemplateId
} from "../scheduled-task-templates"
import {
  DEFAULT_SCHEDULED_TASK_TEMPLATE_CAPABILITIES,
  REQUIRED_INGEST_AVAILABILITY_GATES,
  REQUIRED_WATCH_AVAILABILITY_GATES,
  applyScheduledTaskTemplateCapabilities,
  buildNotificationPolicyCopy,
  buildResultDestinationCopy,
  buildScheduledTaskTemplateCapability,
  buildSourceIntentCopy,
  getMissingAvailabilityGates,
  redactCapabilityPreviewText,
  resolveTemplateCapabilityState
} from "../scheduled-task-template-capabilities"

describe("scheduled task template capabilities", () => {
  const plannedTemplateIds = [
    "recurring_question",
    "agent_task"
  ] as const satisfies readonly ScheduledTaskTemplateId[]

  it("exports an immutable default capability map", () => {
    expect(Object.isFrozen(DEFAULT_SCHEDULED_TASK_TEMPLATE_CAPABILITIES)).toBe(true)
  })

  it("requires preview before Watch can be available", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES.filter(
        (gate) => gate !== "source_preview"
      )
    })

    expect(resolveTemplateCapabilityState("watch", capability)).toBe("limited_availability")
    expect(getMissingAvailabilityGates("watch", capability)).toContain("source_preview")
  })

  it("keeps Watch limited when gates pass but no creation adapter is supported", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
    })

    expect(resolveTemplateCapabilityState("watch", capability)).toBe("limited_availability")
  })

  it("allows Watch only when every Watch gate and the creation adapter guard pass", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      creationAdapterSupported: true,
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
    })

    expect(resolveTemplateCapabilityState("watch", capability)).toBe("available")
  })

  it("does not require notification gate for Ingest availability once creation is supported", () => {
    const capability = buildScheduledTaskTemplateCapability("ingest", {
      creationAdapterSupported: true,
      passedGates: REQUIRED_INGEST_AVAILABILITY_GATES
    })

    expect(resolveTemplateCapabilityState("ingest", capability)).toBe("available")
  })

  it.each(plannedTemplateIds)(
    "ignores Watch/Ingest fallback gates for planned template %s",
    (templateId) => {
      const capability = buildScheduledTaskTemplateCapability(templateId, {
        creationAdapterSupported: true,
        passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
      })

      expect(resolveTemplateCapabilityState(templateId, capability)).toBeNull()
    }
  )

  it("preserves planned template states when capability input would make Watch available", () => {
    const templates = applyScheduledTaskTemplateCapabilities(SCHEDULED_TASK_TEMPLATES, {
      recurring_question: buildScheduledTaskTemplateCapability("recurring_question", {
        creationAdapterSupported: true,
        passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
      }),
      agent_task: buildScheduledTaskTemplateCapability("agent_task", {
        creationAdapterSupported: true,
        passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
      })
    })

    expect(templates.find((template) => template.id === "recurring_question")?.state).toBe(
      "planned"
    )
    expect(templates.find((template) => template.id === "agent_task")?.state).toBe(
      "planned"
    )
  })

  it("generates source-intent copy from source support metadata", () => {
    expect(
      buildSourceIntentCopy({
        sourceFamily: "feed",
        can_watch: true,
        can_ingest: false,
        can_preview: true,
        can_notify: false,
        can_index_search: false,
        can_index_rag: false,
        can_create: false,
        reason: "Ingest setup continues in Watchlists."
      })
    ).toEqual([
      "Detected source: feed.",
      "Watch: supported.",
      "Ingest: not supported for this source yet.",
      "Ingest setup continues in Watchlists."
    ])
  })

  it("falls back to unknown source family when source metadata is partial", () => {
    expect(
      buildSourceIntentCopy({
        sourceFamily: undefined as never,
        can_watch: false,
        can_ingest: false,
        can_preview: false,
        can_notify: false,
        can_index_search: false,
        can_index_rag: false,
        can_create: false
      })[0]
    ).toBe("Detected source: unknown.")
  })

  it("generates destination copy from metadata", () => {
    expect(
      buildResultDestinationCopy({
        home_supported: false,
        notifications_supported: false,
        search_indexed: false,
        rag_scope_included: false
      })
    ).toEqual([
      "Home: not yet shown.",
      "Notifications: not available for this source yet.",
      "Search: content may be saved but not searchable.",
      "RAG: not included in the selected knowledge scope."
    ])
  })

  it("generates notification copy from support state", () => {
    expect(buildNotificationPolicyCopy(null)).toBe("Notifications: configured in Watchlists.")
    expect(buildNotificationPolicyCopy({ notifications_supported: false })).toBe(
      "Notifications are not available for this source yet."
    )
    expect(buildNotificationPolicyCopy({ notifications_supported: true })).toBe(
      "Notifications can open exact task, run, or result detail when supported."
    )
  })

  it("redacts private-looking preview text", () => {
    expect(redactCapabilityPreviewText("https://example.com/feed?token=secret")).toBe(
      "[redacted private source]"
    )
    expect(redactCapabilityPreviewText("https://example.com/feed#private")).toBe(
      "[redacted private source]"
    )
    expect(redactCapabilityPreviewText("https://example.com/feed?api_key=secret")).toBe(
      "[redacted private source]"
    )
    expect(redactCapabilityPreviewText("https://example.com/feed?access_token=secret")).toBe(
      "[redacted private source]"
    )
    expect(redactCapabilityPreviewText("https://example.com/feed?client_secret=secret")).toBe(
      "[redacted private source]"
    )
    expect(redactCapabilityPreviewText("Authorization: Bearer sample-token")).toBe(
      "[redacted private source]"
    )
    expect(redactCapabilityPreviewText("api key: sample credential")).toBe(
      "[redacted private source]"
    )
    expect(redactCapabilityPreviewText("Provider response: token=private-value")).toBe(
      "[redacted private source]"
    )
    expect(redactCapabilityPreviewText("Public release feed")).toBe("Public release feed")
  })
})
