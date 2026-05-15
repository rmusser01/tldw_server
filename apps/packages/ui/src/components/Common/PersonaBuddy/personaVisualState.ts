import type {
  PersonaVisualAuthoredTrigger,
  PersonaVisualStateId
} from "@/types/persona-visuals"

export type PersonaVisualRuntimeOverride = {
  state: PersonaVisualStateId
  reason?: string | null
  expiresAt: number
}

export type ResolvePersonaVisualStateInput = {
  liveVoiceState?: string | null
  hasError?: boolean
  recovering?: boolean
  approvalNeeded?: boolean
  runtimeOverride?: PersonaVisualRuntimeOverride | null
  authoredTriggers?: PersonaVisualAuthoredTrigger[] | null
  activeToolName?: string | null
  activeToolStatus?: string | null
  wakeArmed?: boolean
  isOffline?: boolean
  mcpRuntimeReason?: string | null
  now?: number
}

const BUILTIN_VISUAL_STATES = new Set<PersonaVisualStateId>([
  "idle",
  "wake_armed",
  "listening",
  "thinking",
  "speaking",
  "tool_running",
  "approval_needed",
  "error",
  "offline"
])

const toNormalizedToken = (value: string | null | undefined): string =>
  String(value || "")
    .trim()
    .toLowerCase()

const normalizeLiveVoiceState = (
  value: string | null | undefined
): PersonaVisualStateId | null => {
  const normalized = toNormalizedToken(value).replace(/[\s-]+/g, "_")
  if (!normalized) return null
  if (BUILTIN_VISUAL_STATES.has(normalized as PersonaVisualStateId)) {
    return normalized as PersonaVisualStateId
  }
  if (
    normalized === "recording" ||
    normalized === "transcribing" ||
    normalized === "wake_detected"
  ) {
    return "listening"
  }
  if (normalized === "processing" || normalized === "responding") {
    return "thinking"
  }
  if (normalized === "playing" || normalized === "talking") {
    return "speaking"
  }
  return null
}

const parseToolCategory = (activeToolStatus: string | null | undefined): string => {
  const normalized = toNormalizedToken(activeToolStatus)
  if (!normalized) return ""
  const withoutVerb = normalized
    .replace(/^(running|calling|using|executing)\s+/, "")
    .trim()
  const token = withoutVerb.split(/\s+/)[0] || ""
  return token.split(/[.:/]/)[0] || token
}

const triggerMatches = (
  trigger: PersonaVisualAuthoredTrigger,
  input: ResolvePersonaVisualStateInput,
  liveState: PersonaVisualStateId | null
): boolean => {
  const match = toNormalizedToken(trigger.match)
  if (!match) return false
  if (trigger.source === "live_state") {
    return match === liveState || match === toNormalizedToken(input.liveVoiceState)
  }
  if (trigger.source === "tool_category") {
    const toolStatus = toNormalizedToken(input.activeToolStatus)
    const category = parseToolCategory(input.activeToolStatus)
    return match === category || toolStatus.startsWith(match)
  }
  if (trigger.source === "tool_name") {
    return match === toNormalizedToken(input.activeToolName)
  }
  if (trigger.source === "mcp_runtime") {
    const runtimeReason =
      toNormalizedToken(input.mcpRuntimeReason) ||
      toNormalizedToken(input.runtimeOverride?.reason)
    return match === runtimeReason
  }
  return false
}

const resolveAuthoredTriggerState = (
  input: ResolvePersonaVisualStateInput,
  liveState: PersonaVisualStateId | null
): PersonaVisualStateId | null => {
  const triggers = [...(input.authoredTriggers || [])]
    .filter((trigger) => triggerMatches(trigger, input, liveState))
    .sort((a, b) => b.priority - a.priority)
  return triggers[0]?.state ?? null
}

export const resolvePersonaVisualState = (
  input: ResolvePersonaVisualStateInput = {}
): PersonaVisualStateId => {
  const liveState = normalizeLiveVoiceState(input.liveVoiceState)
  if (input.hasError || input.recovering) return "error"
  if (input.approvalNeeded) return "approval_needed"

  const now = input.now ?? Date.now()
  if (
    input.runtimeOverride &&
    input.runtimeOverride.expiresAt > now &&
    BUILTIN_VISUAL_STATES.has(input.runtimeOverride.state)
  ) {
    return input.runtimeOverride.state
  }

  const triggerState = resolveAuthoredTriggerState(input, liveState)
  if (triggerState) return triggerState

  if (toNormalizedToken(input.activeToolStatus)) return "tool_running"
  if (input.wakeArmed && (!liveState || liveState === "idle")) return "wake_armed"
  if (liveState) return liveState
  if (input.isOffline) return "offline"
  return "idle"
}
