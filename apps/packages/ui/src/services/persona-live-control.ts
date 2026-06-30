import { tldwClient } from "@/services/tldw/TldwApiClient"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"

export type PersonaLiveLifecycle =
  | "idle"
  | "connecting"
  | "connected"
  | "recovering"
  | "stopping"
  | "stopped"
  | "error"

export type PersonaLiveSessionSummary = {
  sessionId: string
  personaId: string
  personaName: string
  lifecycle: PersonaLiveLifecycle
  status: string | null
  isFocused: boolean
  focusedAt: string | null
  focusGeneration: number | null
  lastActivityAt: string | null
  pendingApprovalCount: number
  activeToolName: string | null
  errorState: string | null
  recoveryHint: string | null
  suggestedVisualState: string | null
  allowedActions: string[]
  capabilities: {
    text: boolean
    voice: boolean
    browserMicrophoneRequired: boolean
  }
}

export type PersonaLiveSessionList = {
  sessions: PersonaLiveSessionSummary[]
  focusedSessionId: string | null
}

export type CreatePersonaLiveSessionInput = {
  personaId: string
  reusePolicy?: "resume_compatible" | "create_new"
  idempotencyKey?: string | null
  surface?: string | null
}

export type ListPersonaLiveSessionsInput = {
  personaId?: string | null
  surface?: string | null
  limit?: number | null
}

export class PersonaLiveControlApiError extends Error {
  status?: number
  detail?: unknown

  constructor(
    message: string,
    options: { status?: number; detail?: unknown } = {}
  ) {
    super(message)
    this.name = "PersonaLiveControlApiError"
    this.status = options.status
    this.detail = options.detail
  }
}

const allowedLifecycles = new Set<PersonaLiveLifecycle>([
  "idle",
  "connecting",
  "connected",
  "recovering",
  "stopping",
  "stopped",
  "error"
])

const toRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}

const toStringOrNull = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed || null
}

const toRequiredString = (value: unknown, fallback: string): string =>
  toStringOrNull(value) ?? fallback

const toFiniteNumberOrNull = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

const toNonNegativeInteger = (value: unknown): number => {
  const parsed = toFiniteNumberOrNull(value)
  if (parsed === null) return 0
  return Math.max(0, Math.trunc(parsed))
}

const toStringList = (value: unknown): string[] =>
  Array.isArray(value)
    ? value
        .map((item) => (typeof item === "string" ? item.trim() : ""))
        .filter(Boolean)
    : []

const normalizeLifecycle = (value: unknown): PersonaLiveLifecycle => {
  const candidate = typeof value === "string" ? value.trim() : ""
  return allowedLifecycles.has(candidate as PersonaLiveLifecycle)
    ? (candidate as PersonaLiveLifecycle)
    : "idle"
}

export const normalizePersonaLiveSessionSummary = (
  value: unknown
): PersonaLiveSessionSummary => {
  const item = toRecord(value)
  const capabilities = toRecord(item.capabilities)
  return {
    sessionId: toRequiredString(item.session_id, ""),
    personaId: toRequiredString(item.persona_id, ""),
    personaName: toRequiredString(item.persona_name, "Persona Buddy"),
    lifecycle: normalizeLifecycle(item.lifecycle),
    status: toStringOrNull(item.status),
    isFocused: item.is_focused === true,
    focusedAt: toStringOrNull(item.focused_at),
    focusGeneration: toFiniteNumberOrNull(item.focus_generation),
    lastActivityAt: toStringOrNull(item.last_activity_at),
    pendingApprovalCount: toNonNegativeInteger(item.pending_approval_count),
    activeToolName: toStringOrNull(item.active_tool_name),
    errorState: toStringOrNull(item.error_state),
    recoveryHint: toStringOrNull(item.recovery_hint),
    suggestedVisualState: toStringOrNull(item.suggested_visual_state),
    allowedActions: toStringList(item.allowed_actions),
    capabilities: {
      text: capabilities.text === true,
      voice: capabilities.voice === true,
      browserMicrophoneRequired:
        capabilities.browser_microphone_required === true
    }
  }
}

export const normalizePersonaLiveSessionList = (
  value: unknown
): PersonaLiveSessionList => {
  const payload = toRecord(value)
  return {
    sessions: Array.isArray(payload.sessions)
      ? payload.sessions.map(normalizePersonaLiveSessionSummary)
      : [],
    focusedSessionId: toStringOrNull(payload.focused_session_id)
  }
}

const normalizeSessionEnvelope = (value: unknown): PersonaLiveSessionSummary => {
  const payload = toRecord(value)
  return normalizePersonaLiveSessionSummary(payload.session)
}

const normalizeJsonBody = (
  body: Record<string, unknown>
): Record<string, unknown> => {
  const normalized: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(body)) {
    if (value === undefined || value === null) continue
    normalized[key] = value
  }
  return normalized
}

const readJsonOrNull = async (response: Awaited<ReturnType<typeof tldwClient.fetchWithAuth>>) =>
  response.json().catch(() => null)

const fetchPersonaLiveJson = async (
  path: string,
  init: {
    method?: "GET" | "POST"
    body?: Record<string, unknown>
  } = {}
): Promise<unknown> => {
  const hasBody = init.body != null
  const response = await tldwClient.fetchWithAuth(toAllowedPath(path), {
    method: init.method || "GET",
    headers: hasBody ? { "Content-Type": "application/json" } : undefined,
    body: hasBody ? JSON.stringify(init.body) : undefined
  })
  const payload = await readJsonOrNull(response)
  if (!response.ok) {
    const detail = toRecord(payload).detail ?? payload
    throw new PersonaLiveControlApiError(
      response.error || "Persona live-control request failed",
      {
        status: response.status,
        detail
      }
    )
  }
  return payload
}

export async function listPersonaLiveSessions(
  input: ListPersonaLiveSessionsInput = {}
): Promise<PersonaLiveSessionList> {
  const params = new URLSearchParams()
  const personaId = toStringOrNull(input.personaId)
  const surface = toStringOrNull(input.surface)
  if (personaId) params.set("persona_id", personaId)
  if (surface) params.set("surface", surface)
  if (typeof input.limit === "number" && Number.isFinite(input.limit)) {
    params.set("limit", String(Math.max(1, Math.trunc(input.limit))))
  }
  const query = params.toString()
  const payload = await fetchPersonaLiveJson(
    appendPathQuery(toAllowedPath("/api/v1/persona/live/sessions"), query ? `?${query}` : "")
  )
  return normalizePersonaLiveSessionList(payload)
}

export async function createPersonaLiveSession(
  input: CreatePersonaLiveSessionInput
): Promise<PersonaLiveSessionSummary> {
  const personaId = toStringOrNull(input.personaId)
  if (!personaId) {
    throw new PersonaLiveControlApiError("personaId is required")
  }
  const payload = await fetchPersonaLiveJson("/api/v1/persona/live/sessions", {
    method: "POST",
    body: normalizeJsonBody({
      persona_id: personaId,
      reuse_policy: input.reusePolicy || "resume_compatible",
      idempotency_key: toStringOrNull(input.idempotencyKey),
      surface: toStringOrNull(input.surface)
    })
  })
  return normalizeSessionEnvelope(payload)
}

export async function focusPersonaLiveSession(
  sessionId: string
): Promise<PersonaLiveSessionSummary> {
  const normalizedSessionId = toStringOrNull(sessionId)
  if (!normalizedSessionId) {
    throw new PersonaLiveControlApiError("sessionId is required")
  }
  const payload = await fetchPersonaLiveJson(
    `/api/v1/persona/live/sessions/${encodeURIComponent(normalizedSessionId)}/focus`,
    { method: "POST" }
  )
  return normalizeSessionEnvelope(payload)
}

export async function stopPersonaLiveSession(
  sessionId: string
): Promise<PersonaLiveSessionSummary> {
  const normalizedSessionId = toStringOrNull(sessionId)
  if (!normalizedSessionId) {
    throw new PersonaLiveControlApiError("sessionId is required")
  }
  const payload = await fetchPersonaLiveJson(
    `/api/v1/persona/live/sessions/${encodeURIComponent(normalizedSessionId)}/stop`,
    { method: "POST" }
  )
  return normalizeSessionEnvelope(payload)
}
