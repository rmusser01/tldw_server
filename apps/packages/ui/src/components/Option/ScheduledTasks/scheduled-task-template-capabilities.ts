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

const REDACTED_CAPABILITY_PREVIEW = "[redacted private source]"

const SENSITIVE_CAPABILITY_URL_PARAM_PATTERN =
  /(^|[?&#])([a-z0-9]+[_-])*(token|api[_-]?key|key|secret|session|sid|auth|code|invite|password)([_-][a-z0-9]+)*=/i

const PRIVATE_CAPABILITY_PROSE_PATTERN =
  /\b(api[_ -]?key|password|passphrase|secret|bearer\s+[A-Za-z0-9._-]+|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret)\b|sk-[A-Za-z0-9_-]+/i

const PROVIDER_SECRET_SNIPPET_PATTERN =
  /\b(provider response|authorization|credential|header)\b.*\b(token|secret|api[_ -]?key|password|auth)\s*[:=]/i

const appearsToBeUrl = (value: string): boolean =>
  /^[a-z][a-z0-9+.-]*:\/\//i.test(value) ||
  /^www\./i.test(value) ||
  /^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}([/:?#]|$)/i.test(value)

export const redactCapabilityPreviewText = (value: string): string => {
  const trimmed = value.trim()

  if (
    SENSITIVE_CAPABILITY_URL_PARAM_PATTERN.test(trimmed) ||
    (appearsToBeUrl(trimmed) && trimmed.includes("#")) ||
    PRIVATE_CAPABILITY_PROSE_PATTERN.test(trimmed) ||
    PROVIDER_SECRET_SNIPPET_PATTERN.test(trimmed)
  ) {
    return REDACTED_CAPABILITY_PREVIEW
  }

  return trimmed
}

export const buildSourceIntentCopy = (
  intent: ScheduledTaskSourceIntentCapability | null | undefined
): string[] => {
  if (!intent) {
    return ["Source support: configured in Watchlists."]
  }

  return [
    `Detected source: ${intent.sourceFamily.replace(/_/g, " ")}.`,
    intent.can_watch ? "Watch: supported." : "Watch: not supported for this source yet.",
    intent.can_ingest ? "Ingest: supported." : "Ingest: not supported for this source yet.",
    ...(intent.reason ? [redactCapabilityPreviewText(intent.reason)] : [])
  ]
}

export const buildResultDestinationCopy = (
  metadata: ScheduledTaskResultDestinationMetadata | null | undefined
): string[] => {
  if (!metadata) {
    return ["Results destination: configured in Watchlists."]
  }

  return [
    metadata.home_supported ? "Home: latest results will appear." : "Home: not yet shown.",
    metadata.notifications_supported
      ? "Notifications: available when the task policy triggers."
      : "Notifications: not available for this source yet.",
    metadata.search_indexed
      ? "Search: indexed when ingest completes."
      : "Search: content may be saved but not searchable.",
    metadata.rag_scope_included
      ? "RAG: included in the selected knowledge scope."
      : "RAG: not included in the selected knowledge scope."
  ]
}

export const buildNotificationPolicyCopy = (
  metadata:
    | Pick<ScheduledTaskResultDestinationMetadata, "notifications_supported">
    | null
    | undefined
): string =>
  metadata?.notifications_supported
    ? "Notifications can open exact task, run, or result detail when supported."
    : "Notifications are not available for this source yet."
