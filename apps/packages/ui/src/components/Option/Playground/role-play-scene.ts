import type { ActorSettings } from "@/types/actor"
import { createDefaultActorSettings } from "@/types/actor"
import { buildActorPrompt, estimateActorTokens } from "@/utils/actor"

export type RolePlayScenePreview = {
  active: boolean
  summary: string
  prompt: string
  tokenCount: number
}

const populatedAspectCount = (settings: ActorSettings | null): number =>
  (settings?.aspects || []).filter((aspect) => aspect.value?.trim()).length

const hasNotes = (settings: ActorSettings | null): boolean =>
  Boolean(settings?.notes?.trim())

const summarizeSceneParts = (settings: ActorSettings): string => {
  const details = populatedAspectCount(settings)
  const notes = hasNotes(settings)
  const parts: string[] = []

  if (details > 0) {
    parts.push(`${details} ${details === 1 ? "detail" : "details"}`)
  }
  if (notes) {
    parts.push(settings.notesGmOnly ? "GM-only notes" : "notes")
  }

  return parts.length > 0 ? parts.join(" + ") : "Actor enabled"
}

export function summarizeRolePlayScene(
  settings: ActorSettings | null
): RolePlayScenePreview {
  const prompt = buildActorPrompt(settings)
  const tokenCount = estimateActorTokens(prompt)
  const active = Boolean(
    settings?.isEnabled &&
      (prompt.trim().length > 0 ||
        populatedAspectCount(settings) > 0 ||
        hasNotes(settings))
  )

  return {
    active,
    summary: active && settings ? summarizeSceneParts(settings) : "No scene",
    prompt,
    tokenCount
  }
}

export function clearRolePlayScene(
  settings: ActorSettings | null
): ActorSettings {
  const base = settings ?? createDefaultActorSettings()
  return {
    ...base,
    isEnabled: false,
    notes: "",
    notesGmOnly: false,
    aspects: (base.aspects || []).map((aspect) => ({
      ...aspect,
      value: ""
    }))
  }
}

export function resetRolePlayScene(): ActorSettings {
  return createDefaultActorSettings()
}
