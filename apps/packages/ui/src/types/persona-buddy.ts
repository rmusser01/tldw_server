import type {
  PersonaVisualPack,
  PersonaVisualStateId
} from "@/types/persona-visuals"

export type PersonaBuddyPositionBucket =
  | "web-desktop"
  | "sidepanel-desktop"

export interface PersonaBuddyVisualSummary {
  species_id: string
  silhouette_id: string
  palette_id: string
}

export interface PersonaBuddySummary {
  has_buddy: boolean
  persona_name: string
  role_summary: string | null
  visual: PersonaBuddyVisualSummary | null
  active_visual_pack?: PersonaVisualPack | null
}

export interface PersonaBuddyLiveSessionSummary {
  sessionId: string
  personaId: string
  personaName: string
  lifecycle: string
  pendingApprovalCount: number
  capabilities?: {
    text?: boolean
    voice?: boolean
    browserMicrophoneRequired?: boolean
  } | null
  suggestedVisualState?: string | null
}

export interface PersonaBuddyLiveControlView {
  sessions: PersonaBuddyLiveSessionSummary[]
  focusedSessionId: string | null
  focusedSession: PersonaBuddyLiveSessionSummary | null
  streamState: string
  canSendText: boolean
  voiceAvailable?: boolean
  voiceState?: string | null
  pendingFocusSessionId: string | null
  startTextSession: (personaId?: string | null) => Promise<unknown>
  stopSession: (sessionId?: string | null) => Promise<unknown>
  focusSession: (sessionId: string) => Promise<unknown>
  sendText: (
    text: string,
    options?: { clientMessageId?: string | null }
  ) => Promise<{ ok: boolean; clientMessageId: string; error?: string }>
}

export interface PersonaBuddyRenderContext {
  surface_id: string
  surface_active: boolean
  active_persona_id: string | null
  position_bucket: PersonaBuddyPositionBucket
  buddy_summary?: PersonaBuddySummary | null
  live_session_id?: string | null
  live_voice_state?: string | null
  active_tool_name?: string | null
  active_tool_status?: string | null
  wake_armed?: boolean
  recovery_mode?: string | null
  visual_state?: PersonaVisualStateId | null
  persona_source:
    | "route-local"
    | "route-bootstrap"
    | "catalog"
    | "selected-assistant-fallback"
    | null
}

const normalizeText = (value: unknown): string | null => {
  if (value == null) {
    return null
  }
  const text = String(value).trim()
  return text.length > 0 ? text : null
}

const normalizeBoolean = (value: unknown): boolean => {
  if (typeof value === "boolean") {
    return value
  }
  if (typeof value === "number") {
    return value !== 0
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    if (normalized === "true" || normalized === "1" || normalized === "yes") {
      return true
    }
    if (normalized === "false" || normalized === "0" || normalized === "no") {
      return false
    }
  }
  return Boolean(value)
}

export const normalizePersonaBuddySummary = (
  value: unknown
): PersonaBuddySummary | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null
  }

  const candidate = value as Record<string, unknown>
  const personaName = normalizeText(
    candidate.persona_name ?? candidate.personaName ?? candidate.name
  )

  if (!personaName) {
    return null
  }

  const visualValue =
    candidate.visual && typeof candidate.visual === "object" && !Array.isArray(candidate.visual)
      ? (candidate.visual as Record<string, unknown>)
      : null
  const speciesId = normalizeText(visualValue?.species_id ?? visualValue?.speciesId)
  const silhouetteId = normalizeText(
    visualValue?.silhouette_id ?? visualValue?.silhouetteId
  )
  const paletteId = normalizeText(visualValue?.palette_id ?? visualValue?.paletteId)
  const rawHasBuddy =
    candidate.has_buddy !== undefined
      ? candidate.has_buddy
      : candidate.hasBuddy

  return {
    has_buddy:
      rawHasBuddy === undefined ? true : normalizeBoolean(rawHasBuddy),
    persona_name: personaName,
    role_summary: normalizeText(
      candidate.role_summary ?? candidate.roleSummary
    ),
    visual:
      speciesId && silhouetteId && paletteId
        ? {
            species_id: speciesId,
            silhouette_id: silhouetteId,
            palette_id: paletteId
          }
        : null
  }
}
