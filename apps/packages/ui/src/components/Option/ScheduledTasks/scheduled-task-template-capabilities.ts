import type {
  ScheduledTaskTemplate,
  ScheduledTaskTemplateId,
  ScheduledTaskTemplateState
} from "./scheduled-task-templates"

export type ScheduledTaskAvailabilityGate =
  | "capability_health"
  | "source_preview"
  | "duplicate_detection"
  | "created_entity_response"
  | "task_visibility"
  | "run_result_links"
  | "failure_contract"
  | "result_destination"
  | "notification_contract"
  | "safe_source_handling"
  | "watchlists_preservation"

export type ScheduledTaskSourceFamily =
  | "unknown"
  | "feed"
  | "website"
  | "repository_issues"
  | "video_channel"
  | "publication"
  | "advisory"

export interface ScheduledTaskSourceIntentCapability {
  sourceFamily: ScheduledTaskSourceFamily
  can_watch: boolean
  can_ingest: boolean
  can_preview: boolean
  can_notify: boolean
  can_index_search: boolean
  can_index_rag: boolean
  can_create: boolean
  reason?: string | null
}

export interface ScheduledTaskResultDestinationMetadata {
  home_supported: boolean
  notifications_supported: boolean
  search_indexed: boolean
  rag_scope_included: boolean
}

export interface ScheduledTaskTemplateCapability {
  templateId: ScheduledTaskTemplateId
  passedGates: readonly ScheduledTaskAvailabilityGate[]
  creationAdapterSupported?: boolean
  sourceIntent?: ScheduledTaskSourceIntentCapability | null
  resultDestinations?: ScheduledTaskResultDestinationMetadata | null
  reason?: string | null
}

export type ScheduledTaskTemplateCapabilityMap = Partial<
  Record<ScheduledTaskTemplateId, ScheduledTaskTemplateCapability>
>

export const REQUIRED_WATCH_AVAILABILITY_GATES = [
  "capability_health",
  "source_preview",
  "duplicate_detection",
  "created_entity_response",
  "task_visibility",
  "run_result_links",
  "failure_contract",
  "result_destination",
  "notification_contract",
  "safe_source_handling",
  "watchlists_preservation"
] as const satisfies readonly ScheduledTaskAvailabilityGate[]

export const REQUIRED_INGEST_AVAILABILITY_GATES = [
  "capability_health",
  "source_preview",
  "duplicate_detection",
  "created_entity_response",
  "task_visibility",
  "run_result_links",
  "failure_contract",
  "result_destination",
  "safe_source_handling",
  "watchlists_preservation"
] as const satisfies readonly ScheduledTaskAvailabilityGate[]

export const getRequiredAvailabilityGates = (
  templateId: ScheduledTaskTemplateId
): readonly ScheduledTaskAvailabilityGate[] =>
  templateId === "watch"
    ? REQUIRED_WATCH_AVAILABILITY_GATES
    : templateId === "ingest"
      ? REQUIRED_INGEST_AVAILABILITY_GATES
      : []

export const getMissingAvailabilityGates = (
  templateId: ScheduledTaskTemplateId,
  capability: ScheduledTaskTemplateCapability | null | undefined
): ScheduledTaskAvailabilityGate[] => {
  const required = getRequiredAvailabilityGates(templateId)
  const passed = new Set(capability?.passedGates ?? [])
  return required.filter((gate) => !passed.has(gate))
}

export const resolveTemplateCapabilityState = (
  templateId: ScheduledTaskTemplateId,
  capability: ScheduledTaskTemplateCapability | null | undefined
): ScheduledTaskTemplateState | null => {
  if (templateId !== "watch" && templateId !== "ingest") {
    return null
  }

  if (!capability) {
    return null
  }

  if (getMissingAvailabilityGates(templateId, capability).length > 0) {
    return "limited_availability"
  }

  // Keep this separate from sourceIntent.can_create. That flag describes source-level
  // support, not whether this /scheduled-tasks shell has a real creation adapter.
  return capability.creationAdapterSupported === true
    ? "available"
    : "limited_availability"
}

export const applyScheduledTaskTemplateCapabilities = (
  templates: readonly ScheduledTaskTemplate[],
  capabilities: ScheduledTaskTemplateCapabilityMap | null | undefined
): ScheduledTaskTemplate[] =>
  templates.map((template) => {
    const resolvedState = resolveTemplateCapabilityState(
      template.id,
      capabilities?.[template.id]
    )
    return resolvedState ? { ...template, state: resolvedState } : template
  })

export const buildScheduledTaskTemplateCapability = (
  templateId: ScheduledTaskTemplateId,
  overrides: Partial<ScheduledTaskTemplateCapability> = {}
): ScheduledTaskTemplateCapability => ({
  templateId,
  passedGates: [],
  creationAdapterSupported: false,
  sourceIntent: null,
  resultDestinations: null,
  reason: null,
  ...overrides
})
