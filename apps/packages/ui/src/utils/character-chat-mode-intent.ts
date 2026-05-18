export const CHARACTER_CHAT_MODE_INTENT_EVENT =
  "tldw:character-chat-mode-intent"

export type CharacterChatModeIntentDetail = {
  source?: string
  characterId?: string | number | null
}

export type CharacterChatRouteIntent = {
  mode: "character"
  characterId: string | null
}

export const getCharacterChatRouteIntent = (
  search: string
): CharacterChatRouteIntent | null => {
  const params = new URLSearchParams(search)
  const mode = params.get("mode")?.trim().toLowerCase()
  if (mode !== "character") return null
  return {
    mode: "character",
    characterId:
      params.get("characterId")?.trim() ||
      params.get("character_id")?.trim() ||
      null
  }
}

export const dispatchCharacterChatModeIntent = (
  detail: CharacterChatModeIntentDetail = {}
) => {
  if (typeof window === "undefined") return
  window.dispatchEvent(
    new CustomEvent<CharacterChatModeIntentDetail>(
      CHARACTER_CHAT_MODE_INTENT_EVENT,
      { detail }
    )
  )
}
