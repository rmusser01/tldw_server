import type { Prompt } from "@/db/dexie/types"
import type { ServerPromptBrief } from "@/services/server-prompt-library"

/**
 * Where a Prompt picker entry lives. "both" marks a local prompt that is a
 * copy of a server library prompt (linked by uuid or sharing its unique name).
 */
export type PromptPickerSource = "local" | "server" | "both"

export type LocalPromptPickerEntry = {
  prompt: Prompt
  source: Extract<PromptPickerSource, "local" | "both">
}

export type MergedPromptLibrary = {
  localEntries: LocalPromptPickerEntry[]
  /** Server prompts with no local copy, sorted by name. */
  serverOnly: ServerPromptBrief[]
}

/** Server prompt names are unique, so a case-insensitive title match is a copy. */
export const promptTitleKey = (title: string | null | undefined): string =>
  (title ?? "").trim().toLocaleLowerCase()

/**
 * Merge the device's prompts with the server library, collapsing synced copies
 * into the local entry so each prompt is offered once.
 */
export const mergePromptLibraries = (
  localPrompts: readonly Prompt[],
  serverPrompts: readonly ServerPromptBrief[]
): MergedPromptLibrary => {
  const serverByUuid = new Map<string, ServerPromptBrief>()
  const serverByTitle = new Map<string, ServerPromptBrief>()
  for (const serverPrompt of serverPrompts) {
    if (serverPrompt.uuid) serverByUuid.set(serverPrompt.uuid, serverPrompt)
    const key = promptTitleKey(serverPrompt.name)
    if (key && !serverByTitle.has(key)) serverByTitle.set(key, serverPrompt)
  }

  const copiedServerPrompts = new Set<ServerPromptBrief>()
  const localEntries = localPrompts.map<LocalPromptPickerEntry>((prompt) => {
    const linked = prompt.serverLibraryUuid
      ? serverByUuid.get(prompt.serverLibraryUuid)
      : undefined
    const match = linked ?? serverByTitle.get(promptTitleKey(prompt.title))
    if (!match) return { prompt, source: "local" }
    copiedServerPrompts.add(match)
    return { prompt, source: "both" }
  })

  const serverOnly = serverPrompts
    .filter((serverPrompt) => !copiedServerPrompts.has(serverPrompt))
    .sort((left, right) =>
      left.name.localeCompare(right.name, undefined, { sensitivity: "base" })
    )

  return { localEntries, serverOnly }
}

export const matchesLocalPromptSearch = (prompt: Prompt, query: string) => {
  const q = query.trim().toLowerCase()
  if (!q) return true
  return Boolean(
    prompt.title?.toLowerCase().includes(q) ||
      prompt.content?.toLowerCase().includes(q)
  )
}

export const matchesServerPromptSearch = (
  prompt: ServerPromptBrief,
  query: string
) => {
  const q = query.trim().toLowerCase()
  if (!q) return true
  return Boolean(
    prompt.name.toLowerCase().includes(q) ||
      prompt.author?.toLowerCase().includes(q)
  )
}

export type ServerPromptApplication =
  | { kind: "system"; content: string }
  | { kind: "quick"; content: string }
  | { kind: "empty" }

/**
 * Decide how a server prompt applies, using the same rule as a pulled local
 * copy: a prompt with system text is a system prompt, otherwise a quick prompt.
 */
export const resolveServerPromptApplication = (detail: {
  system_prompt?: string | null
  user_prompt?: string | null
}): ServerPromptApplication => {
  const systemText = detail.system_prompt ?? ""
  if (systemText.trim()) return { kind: "system", content: systemText }
  const userText = detail.user_prompt ?? ""
  if (userText.trim()) return { kind: "quick", content: userText }
  return { kind: "empty" }
}
