export const CHARACTER_CHAT_MODE_INTENT_EVENT =
  "tldw:character-chat-mode-intent"

export type CharacterChatModeIntentDetail = {
  source?: string
  characterId?: string | number | null
}

export type CharacterChatRouteIntent = {
  mode: "character"
  chatId: string | null
  characterId: string | null
}

const CHARACTER_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/

export const normalizeCharacterChatCharacterId = (
  value: string | null
): string | null => {
  const trimmed = value?.trim() ?? ""
  if (!trimmed) return null
  return CHARACTER_ID_PATTERN.test(trimmed) ? trimmed : null
}

export const normalizeCharacterChatSessionId = (
  value: string | null
): string | null => {
  const trimmed = value?.trim() ?? ""
  if (!trimmed) return null
  return CHARACTER_ID_PATTERN.test(trimmed) ? trimmed : null
}

export const getCharacterChatRouteIntent = (
  search: string
): CharacterChatRouteIntent | null => {
  const params = new URLSearchParams(search)
  const mode = params.get("mode")?.trim().toLowerCase()
  if (mode !== "character") return null
  const chatId =
    [
      params.get("chatId"),
      params.get("chat_id"),
      params.get("serverChatId"),
      params.get("server_chat_id")
    ]
      .map((value) => normalizeCharacterChatSessionId(value))
      .find((value): value is string => value !== null) ?? null

  return {
    mode: "character",
    chatId,
    characterId: normalizeCharacterChatCharacterId(
      params.get("characterId") ?? params.get("character_id")
    )
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

// Accepted local actions retire their captured route before asynchronous routing.
export const CHAT_ROUTE_REPLACEMENT_EVENT = "tldw:chat-route-replacement"
export type ChatRouteReplacementDetail = {
  href: string
  serverChatId: string | null
  historyId: string | null
  restoreRevision: number
  // Undefined means the clear action owns navigation; null is a neutral assistant.
  characterId?: string | null
}
export const dispatchChatRouteReplacement = (
  owner: Omit<ChatRouteReplacementDetail, "href">
) => {
  if (typeof window === "undefined") return
  window.dispatchEvent(new CustomEvent<ChatRouteReplacementDetail>(
    CHAT_ROUTE_REPLACEMENT_EVENT, { detail: { ...owner, href: window.location.href } }
  ))
}
