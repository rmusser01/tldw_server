import { buildCharacterChatPath, CHAT_PATH } from "@/routes/route-paths"
import type { AssistantSelection } from "@/types/assistant-selection"

type BuildSidepanelFullAppChatPathOptions = {
  selectedAssistant?: AssistantSelection | null
  selectedCharacterId?: string | number | null
}

const normalizeId = (value: string | number | null | undefined): string | null => {
  const trimmed = value == null ? "" : String(value).trim()
  return trimmed.length > 0 ? trimmed : null
}

export const buildSidepanelFullAppChatPath = (
  options: BuildSidepanelFullAppChatPathOptions = {}
): string => {
  const { selectedAssistant, selectedCharacterId } = options

  if (selectedAssistant?.kind === "persona") {
    return buildCharacterChatPath()
  }

  const characterId =
    selectedAssistant?.kind === "character"
      ? normalizeId(selectedAssistant.id)
      : normalizeId(selectedCharacterId)

  if (characterId) {
    return buildCharacterChatPath({ characterId })
  }

  return CHAT_PATH
}
