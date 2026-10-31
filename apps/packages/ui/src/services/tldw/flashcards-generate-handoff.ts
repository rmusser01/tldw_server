import { createSafeStorage } from "@/utils/safe-storage"
import { isExtensionRuntime } from "@/utils/browser-runtime"
import { normalizeStudyPackIntent, type StudyPackIntent } from "./study-pack-intent"

export type FlashcardsGenerateSourceType = "media" | "note" | "message" | "manual"
export type FlashcardsGenerateIntent = {
  text: string
  sourceType?: FlashcardsGenerateSourceType
  sourceId?: string
  sourceTitle?: string
  conversationId?: string
  messageId?: string
  truncated?: boolean
}

export const FLASHCARDS_GENERATE_HANDOFF_PREFIX = "tldw:flashcards-generate-handoff:"
export const FLASHCARDS_GENERATE_HANDOFF_TTL_MS = 5 * 60 * 1000
export const MAX_GENERATE_PREFILL_CHARS = 12_000
const HANDOFF_LOCK = "tldw:flashcards-generate-handoffs"
const TOKEN_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
const PRIVATE_PARAMS = [
  "generate_text", "generate_source_type", "generate_source_id",
  "generate_source_title", "generate_conversation_id", "generate_message_id"
]
const STUDY_PACK_PRIVATE_PARAMS = ["study_pack", "study_pack_title", "study_pack_payload"]
const SOURCE_TYPES = new Set(["media", "note", "message", "manual"])

export type FlashcardsHandoffKind = "generate" | "study-pack"
type PrivateIntent = FlashcardsGenerateIntent | StudyPackIntent

type HandoffRecord = {
  version: 1
  id: string
  authority: string
  expiresAt: number
  kind?: FlashcardsHandoffKind
  intent: PrivateIntent
}

const handoffError = (message: string) => Object.assign(new Error(message), {
  code: "flashcards_handoff_unavailable"
})
const storageError = () => handoffError(
  "Private transfer storage is unavailable. Keep the source open, enable browser storage, and try again."
)

const withHandoffStorage = async <T>(
  work: (storage: ReturnType<typeof createSafeStorage>, webStorage: Storage | null) => Promise<T>,
  signal?: AbortSignal
): Promise<T> => {
  if (typeof navigator === "undefined" || !navigator.locks?.request) {
    throw handoffError("This browser cannot securely transfer source text. Keep the source open and use a browser with Web Locks support.")
  }
  let webStorage: Storage | null = null
  if (!isExtensionRuntime()) {
    try {
      webStorage = window.localStorage
      if (!webStorage) throw storageError()
    } catch {
      throw storageError()
    }
  }
  try {
    return await navigator.locks.request(HANDOFF_LOCK, async () => {
      signal?.throwIfAborted()
      return work(createSafeStorage({ area: "local" }), webStorage)
    })
  } catch (error) {
    if (error instanceof Error && (error.name === "AbortError" || "code" in error && error.code === "flashcards_handoff_unavailable")) throw error
    throw storageError()
  }
}

const normalizeIntent = (intent: FlashcardsGenerateIntent): FlashcardsGenerateIntent => {
  if (typeof intent.text !== "string" || !intent.text.trim()) {
    throw handoffError("Add source content before generating flashcards.")
  }
  const result: FlashcardsGenerateIntent = {
    text: intent.text.slice(0, MAX_GENERATE_PREFILL_CHARS)
  }
  if (intent.text.length > MAX_GENERATE_PREFILL_CHARS || intent.truncated) result.truncated = true
  if (intent.sourceType && SOURCE_TYPES.has(intent.sourceType)) result.sourceType = intent.sourceType
  for (const key of ["sourceId", "sourceTitle", "conversationId", "messageId"] as const) {
    const value = intent[key]
    if (value == null || value === "") continue
    if (typeof value !== "string" || value.length > 2_048) {
      throw handoffError("Source details are too long to transfer. Keep the source open and shorten its title.")
    }
    result[key] = value
  }
  return result
}

const isRecord = (value: unknown, id: string): value is HandoffRecord => {
  if (!value || typeof value !== "object") return false
  const record = value as Partial<HandoffRecord>
  return record.version === 1 && record.id === id &&
    typeof record.authority === "string" && Boolean(record.authority) &&
    typeof record.expiresAt === "number" && Number.isFinite(record.expiresAt) &&
    (record.kind === "study-pack"
      ? normalizeStudyPackIntent(record.intent) !== null
      : (record.kind == null || record.kind === "generate") &&
        typeof (record.intent as FlashcardsGenerateIntent)?.text === "string" &&
        (record.intent as FlashcardsGenerateIntent).text.length <= MAX_GENERATE_PREFILL_CHARS)
}

const pruneExpired = async (storage: ReturnType<typeof createSafeStorage>) => {
  const records = await storage.getAll()
  const expired = Object.entries(records).filter(([key, record]) =>
    key.startsWith(FLASHCARDS_GENERATE_HANDOFF_PREFIX) &&
    (!isRecord(record, key.slice(FLASHCARDS_GENERATE_HANDOFF_PREFIX.length)) || record.expiresAt <= Date.now())
  ).map(([key]) => key)
  await storage.removeMany(expired)
}

/** The authority is the verified request target/principal tuple; never credentials. */
const createPrivateHandoff = async (
  intent: PrivateIntent,
  authority: string,
  signal: AbortSignal | undefined,
  kind: FlashcardsHandoffKind
): Promise<string> => {
  if (!authority) throw handoffError("Sign in before transferring source content.")
  const normalized = kind === "study-pack"
    ? normalizeStudyPackIntent(intent)
    : normalizeIntent(intent as FlashcardsGenerateIntent)
  if (!normalized) throw handoffError("Choose supported sources and a title before creating a study pack.")
  return withHandoffStorage(async (storage, webStorage) => {
    await pruneExpired(storage)
    signal?.throwIfAborted()
    const id = crypto.randomUUID()
    const key = `${FLASHCARDS_GENERATE_HANDOFF_PREFIX}${id}`
    const record: HandoffRecord = {
      version: 1, id, authority,
      expiresAt: Date.now() + FLASHCARDS_GENERATE_HANDOFF_TTL_MS,
      kind, intent: normalized
    }
    try {
      await storage.set(key, record)
      signal?.throwIfAborted()
      // The WebUI shim may silently use per-instance memory. Verify the real
      // shared backend, not only the same Storage object's successful read.
      const saved = webStorage
        ? JSON.parse(webStorage.getItem(key) || "null")
        : await createSafeStorage({ area: "local" }).get(key)
      if (JSON.stringify(saved) !== JSON.stringify(record)) throw storageError()
      signal?.throwIfAborted()
      return id
    } catch (error) {
      await storage.remove(key)
      throw error
    }
  }, signal)
}

const consumePrivateHandoff = async (
  token: string,
  authority: string,
  signal: AbortSignal | undefined,
  kind: FlashcardsHandoffKind
): Promise<PrivateIntent> => {
  if (!authority) throw handoffError("Sign in to resolve the transfer authority before opening source content.")
  return withHandoffStorage(async (storage) => {
    const key = `${FLASHCARDS_GENERATE_HANDOFF_PREFIX}${token}`
    const record = TOKEN_PATTERN.test(token) ? await storage.get(key) : null
    signal?.throwIfAborted()
    if (!isRecord(record, token) || record.expiresAt <= Date.now()) {
      await storage.remove(key)
      throw handoffError("This transfer is missing, expired, or already consumed. Reopen it from the source.")
    }
    if ((record.kind ?? "generate") !== kind) {
      await storage.remove(key)
      throw handoffError("This transfer is for a different Flashcards action. Reopen it from its source.")
    }
    if (record.authority !== authority) {
      await storage.remove(key)
      throw handoffError("This transfer belongs to another account or server. Reopen it from the source.")
    }
    const intent = kind === "study-pack"
      ? normalizeStudyPackIntent(record.intent)
      : normalizeIntent(record.intent as FlashcardsGenerateIntent)
    if (!intent) throw handoffError("The transfer source is invalid. Reopen it from its source.")
    signal?.throwIfAborted()
    await storage.remove(key)
    signal?.throwIfAborted()
    return intent
  }, signal)
}

export const createFlashcardsGenerateHandoff = (intent: FlashcardsGenerateIntent, authority: string, signal?: AbortSignal) =>
  createPrivateHandoff(intent, authority, signal, "generate")

export const createStudyPackHandoff = (intent: StudyPackIntent, authority: string, signal?: AbortSignal) =>
  createPrivateHandoff(intent, authority, signal, "study-pack")

export const consumeFlashcardsGenerateHandoff = (token: string, authority: string, signal?: AbortSignal): Promise<FlashcardsGenerateIntent> =>
  consumePrivateHandoff(token, authority, signal, "generate") as Promise<FlashcardsGenerateIntent>

export const consumeStudyPackHandoff = (token: string, authority: string, signal?: AbortSignal): Promise<StudyPackIntent> =>
  consumePrivateHandoff(token, authority, signal, "study-pack") as Promise<StudyPackIntent>

export const removeFlashcardsGenerateHandoff = async (token: string): Promise<void> => {
  await withHandoffStorage(storage => storage.remove(`${FLASHCARDS_GENERATE_HANDOFF_PREFIX}${token}`))
}

/** Called by the existing logout/config boundary; unrelated handoffs remain intact. */
export const clearFlashcardsGenerateHandoffs = async (): Promise<void> => {
  await withHandoffStorage(async (storage) => {
    const entries = await storage.getAll()
    await storage.removeMany(Object.keys(entries).filter(key => key.startsWith(FLASHCARDS_GENERATE_HANDOFF_PREFIX)))
  })
}

export const buildFlashcardsGenerateRoute = (token?: string | null): string => {
  if (!token) return "/flashcards?tab=importExport"
  if (!TOKEN_PATTERN.test(token)) throw handoffError("The source transfer could not be opened. Try again from the source.")
  return `/flashcards?tab=importExport&generate_handoff=${token}`
}

// Old URLs have no verified owner. Never hydrate their text or provenance.
export const parseFlashcardsGenerateIntentFromSearch = (_search: string): null => null
export const parseFlashcardsGenerateIntentFromLocation = (_location: { search?: string; hash?: string }): null => null

export const readFlashcardsGenerateRoute = (location: {
  pathname?: string; search?: string; hash?: string
}, kind: FlashcardsHandoffKind = "generate"): { token: string | null; legacy: boolean; cleanRoute: string } => {
  const privateParams = kind === "study-pack"
    ? STUDY_PACK_PRIVATE_PARAMS
    : PRIVATE_PARAMS
  const tokenParam = kind === "study-pack" ? "study_pack_handoff" : "generate_handoff"
  const search = new URLSearchParams(location.search || "")
  const hash = location.hash || ""
  const hashQueryIndex = hash.indexOf("?")
  const hashPath = hashQueryIndex < 0 ? hash : hash.slice(0, hashQueryIndex)
  const hashSearch = new URLSearchParams(hashQueryIndex < 0 ? "" : hash.slice(hashQueryIndex + 1))
  const legacy = privateParams.some(key => search.has(key) || hashSearch.has(key))
  const token = search.get(tokenParam) || hashSearch.get(tokenParam)
  for (const params of [search, hashSearch]) {
    // Both consumers can replace the same history entry. Give each the same
    // private-data-free destination so neither restores the other's fields.
    for (const key of [...PRIVATE_PARAMS, ...STUDY_PACK_PRIVATE_PARAMS, "generate_handoff", "study_pack_handoff"]) params.delete(key)
  }
  const cleanSearch = search.toString()
  const cleanHashSearch = hashSearch.toString()
  return {
    token: legacy ? null : token,
    legacy,
    cleanRoute: `${location.pathname || "/flashcards"}${cleanSearch ? `?${cleanSearch}` : ""}${hashPath}${cleanHashSearch ? `?${cleanHashSearch}` : ""}`
  }
}
