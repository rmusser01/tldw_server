export const WEBUI_CHARACTER_CHAT_SOURCE = "webui-character-chat"
export const WEBUI_CHAT_SOURCE = "webui-chat"

const DEFAULT_TITLE_MAX_LENGTH = 80

const normalizeText = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const normalized = value.replace(/\s+/g, " ").trim()
  return normalized.length > 0 ? normalized : null
}

export const truncateCharacterChatTitle = (
  value: string,
  maxLength = DEFAULT_TITLE_MAX_LENGTH
): string => {
  if (value.length <= maxLength) return value
  return `${value.slice(0, Math.max(0, maxLength - 1))}…`
}

export const buildCharacterChatSessionTitle = ({
  characterName,
  firstUserMessage,
  fallbackTitle,
  maxLength = DEFAULT_TITLE_MAX_LENGTH
}: {
  characterName: unknown
  firstUserMessage?: unknown
  fallbackTitle: string
  maxLength?: number
}): string => {
  const name = normalizeText(characterName)
  const message = normalizeText(firstUserMessage)
  const title = name && message ? `${name}: ${message}` : fallbackTitle
  return truncateCharacterChatTitle(title, maxLength)
}
