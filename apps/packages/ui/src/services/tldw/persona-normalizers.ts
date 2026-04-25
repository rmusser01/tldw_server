import {
  normalizePersonaBuddySummary
} from "@/types/persona-buddy"
import type {
  PersonaExemplar,
  PersonaProfile
} from "./TldwApiClient"

const isObjectRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

export const normalizePersonaProfile = <T extends Record<string, unknown>>(
  input: T | null | undefined
): PersonaProfile => {
  const candidate = isObjectRecord(input) ? input : ({} as T)
  const rawBuddySummary = Object.prototype.hasOwnProperty.call(
    candidate,
    "buddy_summary"
  )
    ? candidate.buddy_summary
    : candidate?.buddySummary
  return {
    ...candidate,
    id: String(candidate?.id ?? candidate?.persona_id ?? ""),
    buddy_summary: normalizePersonaBuddySummary(rawBuddySummary)
  }
}

const normalizeStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return []
  }
  return value
    .map((item) => String(item ?? "").trim())
    .filter((item) => item.length > 0)
}

export const normalizePersonaExemplar = (
  input: Record<string, unknown> | null | undefined
): PersonaExemplar => {
  const candidate = isObjectRecord(input) ? input : {}
  const priorityValue = Number(candidate?.priority)
  return {
    id: String(candidate?.id ?? ""),
    persona_id: String(candidate?.persona_id ?? candidate?.personaId ?? ""),
    kind: String(candidate?.kind ?? "style"),
    content: String(candidate?.content ?? ""),
    tone:
      candidate?.tone == null || String(candidate.tone).trim() === ""
        ? null
        : String(candidate.tone),
    scenario_tags: normalizeStringArray(
      candidate?.scenario_tags ?? candidate?.scenarioTags
    ),
    capability_tags: normalizeStringArray(
      candidate?.capability_tags ?? candidate?.capabilityTags
    ),
    priority: Number.isFinite(priorityValue) ? priorityValue : 0,
    enabled: candidate?.enabled !== false,
    source_type:
      candidate?.source_type == null || String(candidate.source_type).trim() === ""
        ? null
        : String(candidate.source_type),
    source_ref:
      candidate?.source_ref == null || String(candidate.source_ref).trim() === ""
        ? null
        : String(candidate.source_ref),
    notes:
      candidate?.notes == null || String(candidate.notes).trim() === ""
        ? null
        : String(candidate.notes),
    created_at:
      candidate?.created_at == null ? null : String(candidate.created_at),
    last_modified:
      candidate?.last_modified == null ? null : String(candidate.last_modified)
  }
}
