import type { Character } from "@/types/character"
import {
  normalizePersonaBuddySummary,
  type PersonaBuddySummary
} from "@/types/persona-buddy"

export type AssistantKind = "character" | "persona"

export type AssistantSelection = {
  kind: AssistantKind
  id: string
  slug?: string | null
  name: string
  avatar_url?: string | null
  greeting?: string | null
  system_prompt?: string | null
  extensions?: Record<string, unknown> | null
  buddy_summary?: PersonaBuddySummary | null
  metadata?: Record<string, unknown> | null
}

type StoredSelectionRecord = Record<string, unknown>

const normalizeSelectionId = (value: unknown): string | null => {
  if (typeof value === "string") {
    const trimmed = value.trim()
    return trimmed.length > 0 ? trimmed : null
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value)
  }
  return null
}

const resolveAssistantName = (value: unknown, fallback: AssistantKind): string => {
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (trimmed.length > 0) return trimmed
  }
  return fallback === "persona" ? "Persona" : "Assistant"
}

const normalizeOptionalText = (value: unknown): string | null | undefined => {
  if (value == null) return value as null | undefined
  if (typeof value !== "string") return null
  return value
}

export const isAssistantKind = (value: unknown): value is AssistantKind =>
  value === "character" || value === "persona"

export const normalizeAssistantSelection = (
  value: unknown
): AssistantSelection | null => {
  if (!value || typeof value !== "object") return null

  const candidate = value as StoredSelectionRecord
  const { buddySummary: _buddySummary, ...rest } = candidate
  const kind = candidate.kind
  if (!isAssistantKind(kind)) return null

  const id = normalizeSelectionId(candidate.id)
  if (!id) return null

  const name = resolveAssistantName(candidate.name ?? candidate.title, kind)

  const rawBuddySummary = Object.prototype.hasOwnProperty.call(
    candidate,
    "buddy_summary"
  )
    ? candidate.buddy_summary
    : candidate.buddySummary

  return {
    ...rest,
    kind,
    id,
    slug: normalizeOptionalText(candidate.slug),
    name,
    avatar_url: normalizeOptionalText(candidate.avatar_url),
    greeting: normalizeOptionalText(candidate.greeting),
    system_prompt: normalizeOptionalText(candidate.system_prompt),
    buddy_summary: normalizePersonaBuddySummary(rawBuddySummary),
    extensions:
      candidate.extensions &&
      typeof candidate.extensions === "object" &&
      !Array.isArray(candidate.extensions)
        ? (candidate.extensions as Record<string, unknown>)
        : null
  }
}

export const isCharacterAssistantSelection = (
  value: unknown
): value is AssistantSelection & { kind: "character" } => {
  const normalized = normalizeAssistantSelection(value)
  return normalized?.kind === "character"
}

export const isPersonaAssistantSelection = (
  value: unknown
): value is AssistantSelection & { kind: "persona" } => {
  const normalized = normalizeAssistantSelection(value)
  return normalized?.kind === "persona"
}

export const characterToAssistantSelection = <
  T extends object = Character
>(
  character: T | null | undefined
): AssistantSelection | null => {
  if (!character || typeof character !== "object") return null
  return normalizeAssistantSelection({
    ...character,
    kind: "character"
  })
}

export const personaToAssistantSelection = <
  T extends object = Record<string, unknown>
>(
  persona: T | null | undefined
): AssistantSelection | null => {
  if (!persona || typeof persona !== "object") return null
  return normalizeAssistantSelection({
    ...persona,
    kind: "persona"
  })
}

export const assistantSelectionToCharacter = <
  T = Character & Record<string, unknown>
>(
  selection: AssistantSelection | null | undefined
): T | null => {
  if (!selection || selection.kind !== "character") return null
  const { kind: _kind, ...character } = selection
  return character as unknown as T
}
