import type {
  EffectiveWorkspaceAssistantDefault,
  WorkspaceAssistantDefaultDegradedReason,
  WorkspaceAssistantDefaults
} from "@/types/workspace"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)

export const WORKSPACE_ASSISTANT_DEFAULT_DEGRADED_REASONS = new Set<string>([
  "persona_deleted",
  "persona_unavailable",
  "persona_feature_disabled",
  "permission_denied",
  "invalid_default",
  "unsupported_assistant_kind"
])

export const normalizeWorkspaceAssistantDefaults = (
  value: unknown
): WorkspaceAssistantDefaults | null => {
  if (!isRecord(value)) return null
  if (value.assistant_kind !== "persona" && value.assistantKind !== "persona") {
    return null
  }

  const assistantId =
    typeof value.assistant_id === "string"
      ? value.assistant_id.trim()
      : typeof value.assistantId === "string"
        ? value.assistantId.trim()
        : ""
  if (!assistantId) return null

  const personaMemoryMode =
    value.persona_memory_mode === "read_write" ||
    value.personaMemoryMode === "read_write"
      ? "read_write"
      : "read_only"

  return {
    assistantKind: "persona",
    assistantId,
    personaMemoryMode,
    voice: null,
    style: null,
    toolPolicyProfileId: null
  }
}

export const normalizeEffectiveWorkspaceAssistantDefault = (
  value: unknown
): EffectiveWorkspaceAssistantDefault => {
  if (!isRecord(value)) {
    return {
      status: "none",
      source: "none",
      assistantKind: null,
      assistantId: null,
      label: null,
      personaMemoryMode: null,
      degradedReason: null
    }
  }

  const status =
    value.status === "available" || value.status === "unavailable"
      ? value.status
      : "none"
  const source = value.source === "workspace" ? "workspace" : "none"
  const assistantKind =
    value.assistant_kind === "persona" || value.assistantKind === "persona"
      ? "persona"
      : null
  const assistantId =
    typeof value.assistant_id === "string"
      ? value.assistant_id
      : typeof value.assistantId === "string"
        ? value.assistantId
        : null
  const personaMemoryMode =
    value.persona_memory_mode === "read_write" ||
    value.personaMemoryMode === "read_write"
      ? "read_write"
      : value.persona_memory_mode === "read_only" ||
          value.personaMemoryMode === "read_only"
        ? "read_only"
        : null
  const degradedReason =
    typeof value.degraded_reason === "string"
      ? value.degraded_reason
      : typeof value.degradedReason === "string"
        ? value.degradedReason
        : null
  const normalizedDegradedReason =
    degradedReason &&
    WORKSPACE_ASSISTANT_DEFAULT_DEGRADED_REASONS.has(degradedReason)
      ? degradedReason
      : null

  return {
    status,
    source,
    assistantKind,
    assistantId,
    label: typeof value.label === "string" ? value.label : null,
    personaMemoryMode,
    degradedReason:
      normalizedDegradedReason as WorkspaceAssistantDefaultDegradedReason | null
  }
}
