import { db } from '@/db/dexie/schema'
import type { MediaReadAlongAudioCacheEntry } from '@/db/dexie/types'

export const MEDIA_READ_ALONG_CACHE_MAX_ENTRIES = 200
export const MEDIA_READ_ALONG_CACHE_MAX_BYTES = 250 * 1024 * 1024

type SaveOptions = {
  maxBytes?: number
  maxEntries?: number
  signal?: AbortSignal
  shouldContinue?: () => boolean
}

type CacheEntryMetadata = Pick<
  MediaReadAlongAudioCacheEntry,
  'id' | 'lastUsedAt' | 'sizeBytes'
>

const CACHE_METADATA_INDEX = '[lastUsedAt+sizeBytes+id]'

let cacheDisabledForSession = false

const isQuotaExceededError = (error: unknown): boolean => {
  if (!error || typeof error !== 'object') return false
  const name = 'name' in error ? String(error.name) : ''
  return name === 'QuotaExceededError'
}

const table = () => db.mediaReadAlongAudioCache

const canContinueSave = (options: SaveOptions): boolean => {
  if (options.signal?.aborted) return false
  if (options.shouldContinue && !options.shouldContinue()) return false
  return true
}

const toCacheEntryMetadata = (key: unknown): CacheEntryMetadata | null => {
  if (!Array.isArray(key) || key.length < 3) return null
  const [lastUsedAt, sizeBytes, id] = key
  if (typeof id !== 'string') return null

  return {
    id,
    lastUsedAt: Number(lastUsedAt) || 0,
    sizeBytes: Number(sizeBytes) || 0
  }
}

const listEntryMetadata = async (): Promise<CacheEntryMetadata[]> => {
  const keys = await table().orderBy(CACHE_METADATA_INDEX).keys()
  return keys
    .map(toCacheEntryMetadata)
    .filter((entry): entry is CacheEntryMetadata => entry !== null)
}

const resolveMaxEntries = (maxEntries: number): number => {
  if (!Number.isFinite(maxEntries)) return MEDIA_READ_ALONG_CACHE_MAX_ENTRIES
  return Math.max(0, Math.floor(maxEntries))
}

const evictLeastRecentlyUsed = async (
  incomingEntry: MediaReadAlongAudioCacheEntry,
  maxBytes: number,
  maxEntries: number,
  options: SaveOptions,
  forceOne = false
): Promise<boolean> => {
  const sorted = await listEntryMetadata()
  const idsToDelete: string[] = []
  const existingEntry = sorted.find((entry) => entry.id === incomingEntry.id)
  let totalBytesAfterWrite =
    sorted.reduce((sum, entry) => sum + (entry.sizeBytes || 0), 0) -
    (existingEntry?.sizeBytes || 0) +
    incomingEntry.sizeBytes
  let entryCountAfterWrite = sorted.length + (existingEntry ? 0 : 1)

  for (const entry of sorted) {
    if (
      !forceOne &&
      totalBytesAfterWrite <= maxBytes &&
      entryCountAfterWrite <= maxEntries
    ) {
      break
    }
    if (entry.id === incomingEntry.id && !forceOne) {
      continue
    }
    idsToDelete.push(entry.id)
    totalBytesAfterWrite -= entry.sizeBytes || 0
    entryCountAfterWrite -= 1
    if (forceOne) break
  }

  if (idsToDelete.length > 0) {
    if (!canContinueSave(options)) return false
    await table().bulkDelete(idsToDelete)
  }
  return canContinueSave(options)
}

const matchesAttemptedWrite = (
  current: MediaReadAlongAudioCacheEntry | undefined,
  entry: MediaReadAlongAudioCacheEntry
): boolean => {
  return Boolean(
    current &&
      current.createdAt === entry.createdAt &&
      current.lastUsedAt === entry.lastUsedAt &&
      current.textHash === entry.textHash &&
      current.settingsSignature === entry.settingsSignature &&
      current.segmentId === entry.segmentId &&
      current.sizeBytes === entry.sizeBytes
  )
}

const removeAttemptedWriteIfCurrent = async (
  entry: MediaReadAlongAudioCacheEntry
): Promise<void> => {
  const current = await table().get(entry.id)
  if (matchesAttemptedWrite(current, entry)) {
    await table().delete(entry.id)
  }
}

const putEntryIfCurrent = async (
  entry: MediaReadAlongAudioCacheEntry,
  options: SaveOptions
): Promise<boolean> => {
  if (!canContinueSave(options)) return false
  await table().put(entry)
  if (canContinueSave(options)) return true

  await removeAttemptedWriteIfCurrent(entry)
  return false
}

export const resetMediaReadAlongAudioCacheSessionForTests = (): void => {
  cacheDisabledForSession = false
}

export const getMediaReadAlongAudioCacheEntry = async (
  id: string
): Promise<MediaReadAlongAudioCacheEntry | undefined> => {
  if (cacheDisabledForSession) return undefined

  let entry: MediaReadAlongAudioCacheEntry | undefined
  try {
    entry = await table().get(id)
  } catch {
    return undefined
  }
  if (!entry) return undefined

  try {
    await table().update(id, { lastUsedAt: Date.now() })
  } catch {
    // Metadata updates are best effort; do not discard a valid cached blob.
  }
  return entry
}

export const saveMediaReadAlongAudioCacheEntry = async (
  entry: MediaReadAlongAudioCacheEntry,
  options: SaveOptions = {}
): Promise<boolean> => {
  if (cacheDisabledForSession) return false

  const maxBytes = options.maxBytes ?? MEDIA_READ_ALONG_CACHE_MAX_BYTES
  const maxEntries = resolveMaxEntries(
    options.maxEntries ?? MEDIA_READ_ALONG_CACHE_MAX_ENTRIES
  )
  if (entry.sizeBytes > maxBytes) return false
  if (maxEntries < 1) return false
  if (!canContinueSave(options)) return false

  try {
    if (!(await evictLeastRecentlyUsed(entry, maxBytes, maxEntries, options))) {
      return false
    }
    return await putEntryIfCurrent(entry, options)
  } catch (error) {
    if (!isQuotaExceededError(error)) return false

    try {
      if (!(await evictLeastRecentlyUsed(entry, maxBytes, maxEntries, options, true))) {
        return false
      }
      return await putEntryIfCurrent(entry, options)
    } catch (retryError) {
      if (isQuotaExceededError(retryError)) {
        cacheDisabledForSession = true
      }
      return false
    }
  }
}
