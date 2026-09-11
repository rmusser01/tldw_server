import { getPromptById } from "@/db/dexie/helpers"
import type { Prompt } from "@/db/dexie/types"

export type GetPromptByIdFn = (id: string) => Promise<Prompt | undefined>

export type EffectiveSystemPromptState = {
  templateContent: string
  effectiveContent: string
  overrideActive: boolean
}

export type SystemPromptOverrideSnapshot = {
  rawOverride: string | undefined
}

const getPromptContent = (prompt?: Prompt | null): string =>
  prompt?.system_prompt ?? prompt?.content ?? ""

export const resolveSelectedSystemPromptContent = async (
  selectedSystemPrompt?: string | null,
  getPromptByIdFn: GetPromptByIdFn = getPromptById
): Promise<string> => {
  if (!selectedSystemPrompt?.trim()) {
    return ""
  }

  try {
    const prompt = await getPromptByIdFn(selectedSystemPrompt)
    return getPromptContent(prompt)
  } catch {
    return ""
  }
}

export const resolveEffectiveSystemPromptState = async ({
  selectedSystemPrompt,
  systemPrompt,
  getPromptByIdFn = getPromptById
}: {
  selectedSystemPrompt?: string | null
  systemPrompt?: string | null
  getPromptByIdFn?: GetPromptByIdFn
}): Promise<EffectiveSystemPromptState> => {
  const templateContent = await resolveSelectedSystemPromptContent(
    selectedSystemPrompt,
    getPromptByIdFn
  )
  const overrideValue = typeof systemPrompt === "string" ? systemPrompt : ""
  const overrideActive =
    overrideValue.trim().length > 0 && overrideValue !== templateContent

  return {
    templateContent,
    effectiveContent: overrideActive ? overrideValue : templateContent,
    overrideActive
  }
}

export const normalizeSystemPromptOverrideValue = ({
  draft,
  templateContent
}: {
  draft?: string | null
  templateContent?: string | null
}): string => {
  const normalizedDraft = typeof draft === "string" ? draft : ""
  if (
    typeof templateContent === "string" &&
    normalizedDraft === templateContent
  ) {
    return ""
  }
  return normalizedDraft
}

export const captureSystemPromptOverrideSnapshot = (
  rawOverride: string | undefined
): SystemPromptOverrideSnapshot => ({ rawOverride })

export const restoreSystemPromptOverrideSnapshot = (
  snapshot: SystemPromptOverrideSnapshot
): string | undefined => snapshot.rawOverride
