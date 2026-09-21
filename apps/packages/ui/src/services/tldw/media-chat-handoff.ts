import { createSafeStorage } from "@/utils/safe-storage"
import { isExtensionRuntime } from "@/utils/browser-runtime"

export type MediaChatHandoffMode = "normal" | "rag_media"

export type MediaChatHandoffPayload = {
  ownerScope?: string
  mediaId?: string
  mediaIds?: number[]
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
  value: unknown,
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
  const content =
    typeof payload.content === "string" && payload.content.trim()
      ? payload.content
      : undefined
  if (content) result.content = content
  if (Array.isArray(payload.mediaIds)) {
    const ids = payload.mediaIds.filter(
      (id): id is number =>
        typeof id === "number" && Number.isInteger(id) && id > 0,
    )
    if (ids.length) result.mediaIds = [...new Set(ids)]
  }
  if (payload.mode === "normal" || payload.mode === "rag_media") {
    result.mode = payload.mode
  }
  return Object.keys(result).length > 0 ? result : undefined
}

export const getMediaChatHandoffMode = (
  payload: MediaChatHandoffPayload,
): MediaChatHandoffMode => {
  if (payload.mode === "rag_media") return "rag_media"
  return "normal"
}

export const parseMediaIdAsNumber = (
  payload: MediaChatHandoffPayload,
): number | null => {
  const parsed = Number(payload.mediaId)
  if (!Number.isFinite(parsed)) return null
  if (parsed <= 0) return null
  return Math.trunc(parsed)
}

export const buildDiscussMediaHint = (
  payload: MediaChatHandoffPayload,
): string => {
  if (payload.content && (payload.title || payload.mediaId)) {
    const header = `Chat with this media: ${
      payload.title || payload.mediaId
    }`.trim()
    return `${header}\n\n${payload.content}`
  }
  if (payload.url) {
    return `Let's talk about the media I just ingested: ${payload.url}`
  }
  if (payload.mediaId) {
    return `Let's talk about media ${payload.mediaId}.`
  }
  return ""
}

export const MEDIA_CHAT_HANDOFF_PARAM = "media_handoff"
export const MEDIA_CHAT_HANDOFF_TTL_MS = 10 * 60 * 1000
const HANDOFF_PREFIX = "tldw:media-chat-handoff:"
const TOKEN_PATTERN = /^(tab|extension)-[0-9a-f-]{36}$/i
type HandoffRecord = { createdAt: number; payload: MediaChatHandoffPayload }
type HandoffStorage = {
  get: () => Promise<unknown>
  set: (record: HandoffRecord) => Promise<void>
  remove: () => Promise<void>
}
const tabOperations = new Map<string, Promise<unknown>>()

/** Native sessionStorage is tab-owned, including extension options documents.
 * Only an explicit extension new-tab transfer uses shared, token-addressed storage.
 */
const withHandoffStorage = async <T>(
  token: string,
  work: (storage: HandoffStorage) => Promise<T>,
): Promise<T> => {
  if (!TOKEN_PATTERN.test(token))
    throw new Error("The media transfer is invalid.")
  const key = HANDOFF_PREFIX + token
  if (token.startsWith("extension-")) {
    if (!isExtensionRuntime() || !navigator.locks?.request) {
      throw new Error("Private media transfer storage is unavailable.")
    }
    return navigator.locks.request(key, async () => {
      const storage = createSafeStorage({ area: "local" })
      return work({
        get: () => storage.get(key),
        set: (record) => storage.set(key, record),
        remove: () => storage.remove(key),
      })
    })
  }
  let storage: Storage
  try {
    storage = window.sessionStorage
  } catch {
    throw new Error("Private media transfer storage is unavailable.")
  }
  const operation = () =>
    work({
      get: async () => JSON.parse(storage.getItem(key) || "null"),
      set: async (record) => {
        storage.setItem(key, JSON.stringify(record))
      },
      remove: async () => {
        storage.removeItem(key)
      },
    })
  // Serialize reads/claims within this tab even when Web Locks is unavailable.
  const pending = (tabOperations.get(token) ?? Promise.resolve()).then(
    operation,
  )
  const tail = pending.catch(() => undefined)
  tabOperations.set(token, tail)
  try {
    return await pending
  } finally {
    if (tabOperations.get(token) === tail) tabOperations.delete(token)
  }
}

/** Save full source text privately before navigating. Never fall back to a broadcast. */
export const createMediaChatHandoff = async (
  payload: MediaChatHandoffPayload,
  options: { newTab?: boolean } = {},
): Promise<string> => {
  const normalized = normalizeMediaChatHandoffPayload(payload)
  if (!normalized?.ownerScope)
    throw new Error("Wait for your account before preparing this source.")
  if (options.newTab && !isExtensionRuntime())
    throw new Error("Open this media transfer in the current tab.")
  const token = `${options.newTab ? "extension" : "tab"}-${crypto.randomUUID()}`
  await withHandoffStorage(token, async (storage) => {
    const record = { createdAt: Date.now(), payload: normalized }
    await storage.set(record)
    if (JSON.stringify(await storage.get()) !== JSON.stringify(record)) {
      await storage.remove()
      throw new Error(
        "The media transfer could not be saved. Check browser storage and try again.",
      )
    }
  })
  return token
}

const loadMediaChatHandoff = async (
  token: string,
  ownerScope: string | null,
  consume: boolean,
  accept?: (payload: MediaChatHandoffPayload) => boolean,
): Promise<MediaChatHandoffPayload | null> => {
  if (!ownerScope || !TOKEN_PATTERN.test(token)) return null
  try {
    return await withHandoffStorage(token, async (storage) => {
      const raw = (await storage.get()) as HandoffRecord | null
      if (!raw) return null
      const payload = normalizeMediaChatHandoffPayload(raw.payload)
      if (
        !Number.isFinite(raw.createdAt) ||
        Date.now() - raw.createdAt > MEDIA_CHAT_HANDOFF_TTL_MS ||
        !payload?.ownerScope
      ) {
        await storage.remove()
        return null
      }
      if (payload.ownerScope !== ownerScope) return null
      if (consume) {
        // Recheck the destination after the asynchronous read, before any text
        // is applied or removed. A draft conflict leaves the source available.
        if (accept && !accept(payload)) return null
        await storage.remove()
      }
      return payload
    })
  } catch {
    return null
  }
}

export const readMediaChatHandoff = (
  token: string,
  ownerScope: string | null,
) => loadMediaChatHandoff(token, ownerScope, false)
export const consumeMediaChatHandoff = (
  token: string,
  ownerScope: string | null,
  accept?: (payload: MediaChatHandoffPayload) => boolean,
) => loadMediaChatHandoff(token, ownerScope, true, accept)
export const removeMediaChatHandoff = (token: string): Promise<void> =>
  withHandoffStorage(token, (storage) => storage.remove())

/** The URL carries no source text, title, media identifiers, or credentials. */
export const buildMediaChatHandoffRoute = (token: string): string => {
  if (!TOKEN_PATTERN.test(token))
    throw new Error("The media transfer is invalid.")
  return `/chat?${MEDIA_CHAT_HANDOFF_PARAM}=${token}`
}
