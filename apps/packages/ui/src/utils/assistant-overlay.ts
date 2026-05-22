import type { AssistantSelection } from "@/types/assistant-selection"
import type { ChatAssistantOverlay } from "@/types/chat-session-settings"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const normalizeText = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const buildOverlayPayload = ({
  kind,
  id,
  name,
  avatarUrl,
  systemPrompt
}: {
  kind: ChatAssistantOverlay["kind"]
  id: string
  name: string
  avatarUrl?: string | null
  systemPrompt?: string | null
}): ChatAssistantOverlay => ({
  kind,
  id,
  name,
  avatar_url: avatarUrl ?? null,
  system_prompt_snapshot: systemPrompt ?? null,
  updatedAt: new Date().toISOString()
})

export const buildAssistantOverlaySnapshotFromSelection = (
  selection: AssistantSelection
): ChatAssistantOverlay =>
  buildOverlayPayload({
    kind: selection.kind,
    id: selection.id,
    name:
      normalizeText(selection.name) ??
      (selection.kind === "persona" ? "Persona" : "Assistant"),
    avatarUrl: normalizeText(selection.avatar_url) ?? null,
    systemPrompt: normalizeText(selection.system_prompt) ?? null
  })

export const resolveAssistantOverlaySnapshot = async (
  selection: AssistantSelection
): Promise<ChatAssistantOverlay> => {
  if (selection.kind === "persona") {
    let profile = null
    try {
      profile = await tldwClient.getPersonaProfile(selection.id)
    } catch (error) {
      console.warn(
        "[assistant-overlay] Failed to load persona detail; using summary snapshot",
        error
      )
    }

    return buildOverlayPayload({
      kind: "persona",
      id: selection.id,
      name:
        normalizeText(profile?.name) ??
        normalizeText(selection.name) ??
        "Persona",
      avatarUrl:
        normalizeText(profile?.avatar_url) ??
        normalizeText(selection.avatar_url) ??
        null,
      systemPrompt:
        normalizeText(profile?.system_prompt) ??
        normalizeText(selection.system_prompt) ??
        null
    })
  }

  let detail = null
  try {
    detail = await tldwClient.getCharacter(selection.id, { forceRefresh: true })
  } catch (error) {
    console.warn(
      "[assistant-overlay] Failed to refresh character detail; using summary snapshot",
      error
    )
  }

  return buildOverlayPayload({
    kind: "character",
    id: selection.id,
    name:
      normalizeText(detail?.name) ??
      normalizeText(detail?.title) ??
      normalizeText(selection.name) ??
      "Assistant",
    avatarUrl:
      normalizeText(detail?.avatar_url) ??
      normalizeText(selection.avatar_url) ??
      null,
    systemPrompt:
      normalizeText(detail?.system_prompt) ??
      normalizeText(selection.system_prompt) ??
      null
  })
}
