import type { PersonaVoiceDefaults } from "@/hooks/useResolvedPersonaVoiceDefaults"
import type { PersonaTurnDetectionValues } from "@/components/PersonaGarden/PersonaTurnDetectionControls"
import type {
  SetupReviewSummary,
  SetupHandoffRecommendedAction
} from "@/components/PersonaGarden/PersonaSetupHandoffCard"
import type { PersonaBuddySummary } from "@/types/persona-buddy"
import type { PersonaGardenTabKey } from "@/utils/persona-garden-route"
import type { PersonaSetupState, PersonaSetupStep } from "@/hooks/usePersonaSetupWizard"
import {
  PERSONA_STARTER_COMMAND_TEMPLATES
} from "@/components/PersonaGarden/personaStarterCommandTemplates"

export type PersonaInfo = {
  id: string
  name: string
  mode?: "session_scoped" | "persistent_scoped"
  description?: string | null
  voice?: string | null
  avatar_url?: string | null
  system_prompt?: string | null
  greeting?: string | null
  capabilities?: string[]
  default_tools?: string[]
  extensions?: Record<string, unknown> | null
  buddy_summary?: PersonaBuddySummary | null
  metadata?: Record<string, unknown> | null
}

export type PersonaPlanStep = {
  idx: number
  tool: string
  args?: Record<string, unknown>
  description?: string
  why?: string
  policy?: PersonaToolPolicy
}

export type PendingPlan = {
  planId: string
  steps: PersonaPlanStep[]
  memory?: PersonaMemoryUsage
  companion?: PersonaCompanionUsage
}

export type PersonaLogEntry = {
  id: string
  kind: "user" | "assistant" | "tool" | "notice"
  text: string
}

export type PersonaToolPolicy = {
  allow?: boolean
  requires_confirmation?: boolean
  required_scope?: string | null
  reason_code?: string | null
  reason?: string | null
  action?: string | null
}

export type PersonaMemoryUsage = {
  enabled?: boolean
  requested_top_k?: number
  applied_count?: number
}

export type PersonaCompanionUsage = {
  enabled?: boolean
  requested_enabled?: boolean
  applied_card_count?: number
  applied_activity_count?: number
}

export type PersonaProfileResponse = {
  id?: string
  version?: number
  mode?: "session_scoped" | "persistent_scoped"
  buddy_summary?: PersonaBuddySummary | null
  use_persona_state_context_default?: boolean
  voice_defaults?: PersonaVoiceDefaults | null
  setup?: PersonaSetupState | null
}

export type SetupStepErrors = {
  persona?: string | null
  voice?: string | null
  commands?: string | null
  safety?: string | null
  test?: string | null
}

export type SetupHandoffState = {
  runId: string
  targetTab: PersonaGardenTabKey
  completionType: "dry_run" | "live_session"
  reviewSummary: SetupReviewSummary
  recommendedAction: SetupHandoffRecommendedAction
  consumedAction: SetupHandoffConsumedAction | null
  compact: boolean
}

export type SetupHandoffSectionTarget =
  | { tab: "commands"; section: "command_form" | "command_list" }
  | {
      tab: "connections"
      section: "connection_form" | "saved_connections"
      connectionId?: string | null
      connectionName?: string | null
    }
  | { tab: "profiles"; section: "assistant_defaults" | "confirmation_mode" }
  | { tab: "test-lab"; section: "dry_run_form" }

export type SetupHandoffFocusRequest = {
  tab: SetupHandoffSectionTarget["tab"]
  section: SetupHandoffSectionTarget["section"]
  token: number
  connectionId?: string | null
  connectionName?: string | null
}

export type SetupHandoffConsumedAction =
  | "command_saved"
  | "connection_saved"
  | "connection_test_succeeded"
  | "voice_defaults_saved"
  | "dry_run_match"
  | "live_response_received"

export type SetupCommandDetourState = {
  phrase: string
  returnStep: "test"
}

export type SetupLiveDetourState = {
  source: "live_unavailable" | "live_failure"
  lastText: string
}

export type SetupVisualDetourState = {
  source: "wizard_optional_card"
  returnStep: PersonaSetupStep
}

export const DEFAULT_SETUP_REVIEW_SUMMARY: SetupReviewSummary = {
  starterCommands: { mode: "skipped" },
  confirmationMode: null,
  connection: { mode: "skipped" }
}

export const SETUP_STARTER_COMMAND_DESCRIPTIONS = new Set(
  PERSONA_STARTER_COMMAND_TEMPLATES.map((template) => template.commandDescription)
)

export const isSetupCreatedStarterCommand = (value: unknown): boolean => {
  if (!value || typeof value !== "object") return false
  const record = value as Record<string, unknown>
  const description = String(record.description || "").trim()
  if (!description) return false
  return (
    SETUP_STARTER_COMMAND_DESCRIPTIONS.has(description) ||
    / from assistant setup$/i.test(description)
  )
}

export const pickAvailableConnectionName = (value: unknown): string | null => {
  if (!Array.isArray(value)) return null
  const names = value
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null
      const record = entry as Record<string, unknown>
      const name = String(record.name || "").trim()
      const createdAt = String(record.created_at || record.last_modified || "").trim()
      if (!name) return null
      return {
        name,
        createdAt
      }
    })
    .filter((entry): entry is { name: string; createdAt: string } => Boolean(entry))

  if (names.length === 0) return null

  return names
    .slice()
    .sort((left, right) => right.createdAt.localeCompare(left.createdAt))[0]?.name || null
}

export const summarizeFallbackStarterCommands = (value: unknown): SetupReviewSummary["starterCommands"] => {
  const commands = Array.isArray((value as { commands?: unknown[] } | null | undefined)?.commands)
    ? ((value as { commands: unknown[] }).commands)
    : []
  const setupCreatedCount = commands.filter(isSetupCreatedStarterCommand).length

  if (setupCreatedCount > 0) {
    return {
      mode: "configured",
      count: setupCreatedCount
    }
  }

  return { mode: "skipped" }
}

export const deriveSetupHandoffRecommendedAction = ({
  completionType,
  reviewSummary
}: {
  completionType: "dry_run" | "live_session"
  reviewSummary: SetupReviewSummary
}): SetupHandoffRecommendedAction => {
  if (reviewSummary.starterCommands.mode === "skipped") {
    return "add_command"
  }
  if (reviewSummary.connection.mode === "skipped") {
    return "add_connection"
  }
  if (completionType === "dry_run") {
    return "try_live"
  }
  return "review_commands"
}

export const toSetupHandoffActionTarget = (
  action: SetupHandoffConsumedAction
): PersonaGardenTabKey => {
  if (action === "command_saved") return "commands"
  if (action === "connection_saved" || action === "connection_test_succeeded") {
    return "connections"
  }
  if (action === "voice_defaults_saved") return "profiles"
  if (action === "live_response_received") return "live"
  return "test-lab"
}

export type PersonaStateDocsResponse = {
  persona_id?: string
  soul_md?: string | null
  identity_md?: string | null
  heartbeat_md?: string | null
  last_modified?: string | null
}

export type PersonaStateHistoryEntry = {
  entry_id: string
  field: "soul_md" | "identity_md" | "heartbeat_md"
  content: string
  is_active?: boolean
  created_at?: string | null
  last_modified?: string | null
  version?: number
}

export type PersonaStateHistoryResponse = {
  persona_id?: string
  entries?: PersonaStateHistoryEntry[]
}

export type UnsavedStateDiscardReason =
  | "generic"
  | "connect"
  | "disconnect"
  | "reload_state"
  | "persona_switch"
  | "session_switch"
  | "restore_state"
  | "route_transition"
  | "before_unload"

export const _historyEntrySortEpoch = (entry: PersonaStateHistoryEntry): number => {
  const candidate = String(entry.created_at || entry.last_modified || "").trim()
  if (!candidate) return 0
  const parsed = Date.parse(candidate)
  return Number.isFinite(parsed) ? parsed : 0
}

export const PERSONA_STATE_EDITOR_EXPANDED_PREF_KEY =
  "sidepanel:persona:state-editor-expanded"
export const PERSONA_STATE_HISTORY_ORDER_PREF_KEY = "sidepanel:persona:state-history-order"

export const hasExplicitTurnDetectionDefaults = (
  voiceDefaults?: PersonaVoiceDefaults | null
): boolean =>
  typeof voiceDefaults?.auto_commit_enabled === "boolean" &&
  typeof voiceDefaults?.vad_threshold === "number" &&
  typeof voiceDefaults?.min_silence_ms === "number" &&
  typeof voiceDefaults?.turn_stop_secs === "number" &&
  typeof voiceDefaults?.min_utterance_secs === "number"

export const buildTurnDetectionValuesFromSavedDefaults = (
  voiceDefaults?: PersonaVoiceDefaults | null
): PersonaTurnDetectionValues | null => {
  if (!hasExplicitTurnDetectionDefaults(voiceDefaults)) return null
  return {
    autoCommitEnabled: Boolean(voiceDefaults?.auto_commit_enabled),
    vadThreshold: Number(voiceDefaults?.vad_threshold),
    minSilenceMs: Number(voiceDefaults?.min_silence_ms),
    turnStopSecs: Number(voiceDefaults?.turn_stop_secs),
    minUtteranceSecs: Number(voiceDefaults?.min_utterance_secs)
  }
}

export const areTurnDetectionValuesEqual = (
  left: PersonaTurnDetectionValues,
  right: PersonaTurnDetectionValues
): boolean =>
  left.autoCommitEnabled === right.autoCommitEnabled &&
  left.vadThreshold === right.vadThreshold &&
  left.minSilenceMs === right.minSilenceMs &&
  left.turnStopSecs === right.turnStopSecs &&
  left.minUtteranceSecs === right.minUtteranceSecs

export const _readBoolPreference = (key: string, fallback: boolean): boolean => {
  if (typeof window === "undefined") return fallback
  try {
    const raw = window.localStorage.getItem(key)
    if (raw == null) return fallback
    const normalized = raw.trim().toLowerCase()
    if (normalized === "true") return true
    if (normalized === "false") return false
  } catch {
    // ignore storage access errors
  }
  return fallback
}

export const _readHistoryOrderPreference = (): "newest" | "oldest" => {
  if (typeof window === "undefined") return "newest"
  try {
    const raw = window.localStorage
      .getItem(PERSONA_STATE_HISTORY_ORDER_PREF_KEY)
      ?.trim()
      .toLowerCase()
    if (raw === "oldest") return "oldest"
  } catch {
    // ignore storage access errors
  }
  return "newest"
}

export const _confirmWithBrowserPrompt = (message: string): boolean => {
  if (typeof window === "undefined" || typeof window.confirm !== "function") return true
  try {
    return window.confirm(message)
  } catch {
      return true
  }
}

export type PersonaRouteMode = "persona" | "companion"
export type PersonaRouteShell = "sidepanel" | "options"

export type SidepanelPersonaProps = {
  mode?: PersonaRouteMode
  shell?: PersonaRouteShell
}

export const DEFAULT_PERSONA_ID = "research_assistant"
