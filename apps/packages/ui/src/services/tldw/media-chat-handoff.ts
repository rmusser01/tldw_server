export type MediaChatHandoffMode = "normal" | "rag_media"

export type MediaChatHandoffPayload = {
  ownerScope?: string
  mediaId?: string
  url?: string
  title?: string
  content?: string
  mode?: MediaChatHandoffMode
}

const toNonEmptyString = (value: unknown): string | undefined => {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

export const normalizeMediaChatHandoffPayload = (
  value: unknown
): MediaChatHandoffPayload | undefined => {
  if (!value || typeof value !== "object" || Array.isArray(value))
    return undefined
  const payload = value as Record<string, unknown>
  const result: MediaChatHandoffPayload = {}
  const ownerScope = toNonEmptyString(payload.ownerScope)
  if (ownerScope) result.ownerScope = ownerScope
  const mediaId = toNonEmptyString(payload.mediaId)
  if (mediaId) result.mediaId = mediaId
  const url = toNonEmptyString(payload.url)
  if (url) result.url = url
  const title = toNonEmptyString(payload.title)
  if (title) result.title = title
  const content = toNonEmptyString(payload.content)
  if (content) result.content = content
  if (payload.mode === "normal" || payload.mode === "rag_media") {
    result.mode = payload.mode
  }
  return Object.keys(result).length > 0 ? result : undefined
}

export const getMediaChatHandoffMode = (
  payload: MediaChatHandoffPayload
): MediaChatHandoffMode => {
  if (payload.mode === "rag_media") return "rag_media"
  return "normal"
}

export const parseMediaIdAsNumber = (
  payload: MediaChatHandoffPayload
): number | null => {
  const parsed = Number(payload.mediaId)
  if (!Number.isFinite(parsed)) return null
  if (parsed <= 0) return null
  return Math.trunc(parsed)
}

export const buildDiscussMediaHint = (
  payload: MediaChatHandoffPayload
): string => {
  if (payload.content && (payload.title || payload.mediaId)) {
    const header = `Chat with this media: ${
      payload.title || payload.mediaId
    }`.trim()
    return `${header}\n\n${payload.content}`.trim()
  }
  if (payload.url) {
    return `Let's talk about the media I just ingested: ${payload.url}`
  }
  if (payload.mediaId) {
    return `Let's talk about media ${payload.mediaId}.`
  }
  return ""
}
