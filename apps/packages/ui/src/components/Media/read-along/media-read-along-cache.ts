import { db } from '@/db/dexie/schema'
import type { MediaReadAlongAudioCacheEntry } from '@/db/dexie/types'

export const MEDIA_READ_ALONG_CACHE_MAX_BYTES = 50 * 1024 * 1024

type SaveOptions = {
  maxBytes?: number
}

let cacheDisabledForSession = false

const isQuotaExceededError = (error: unknown): boolean => {
  if (!error || typeof error !== 'object') return false
  const name = 'name' in error ? String(error.name) : ''
  return name === 'QuotaExceededError'
}

const table = () => db.mediaReadAlongAudioCache

const listEntries = async (): Promise<MediaReadAlongAudioCacheEntry[]> => {
  if (typeof table().toArray === 'function') {
    return await table().toArray()
  }
  return await table().orderBy('lastUsedAt').toArray()
}

const evictLeastRecentlyUsed = async (
  incomingSizeBytes: number,
  maxBytes: number,
  forceOne = false
): Promise<void> => {
  const entries = await listEntries()
  const sorted = [...entries].sort((a, b) => a.lastUsedAt - b.lastUsedAt)
  const idsToDelete: string[] = []
  let totalBytes = sorted.reduce((sum, entry) => sum + (entry.sizeBytes || 0), 0)

  for (const entry of sorted) {
    if (!forceOne && totalBytes + incomingSizeBytes <= maxBytes) break
    idsToDelete.push(entry.id)
    totalBytes -= entry.sizeBytes || 0
    if (forceOne) break
  }

  if (idsToDelete.length > 0) {
    await table().bulkDelete(idsToDelete)
  }
}

export const resetMediaReadAlongAudioCacheSessionForTests = (): void => {
  cacheDisabledForSession = false
}

export const getMediaReadAlongAudioCacheEntry = async (
  id: string
): Promise<MediaReadAlongAudioCacheEntry | undefined> => {
  if (cacheDisabledForSession) return undefined

  try {
    const entry = await table().get(id)
    if (!entry) return undefined

    await table().update(id, { lastUsedAt: Date.now() })
    return entry
  } catch {
    return undefined
  }
}

export const saveMediaReadAlongAudioCacheEntry = async (
  entry: MediaReadAlongAudioCacheEntry,
  options: SaveOptions = {}
): Promise<boolean> => {
  if (cacheDisabledForSession) return false

  const maxBytes = options.maxBytes ?? MEDIA_READ_ALONG_CACHE_MAX_BYTES
  if (entry.sizeBytes > maxBytes) return false

  try {
    await evictLeastRecentlyUsed(entry.sizeBytes, maxBytes)
    await table().put(entry)
    return true
  } catch (error) {
    if (!isQuotaExceededError(error)) return false

    try {
      await evictLeastRecentlyUsed(entry.sizeBytes, maxBytes, true)
      await table().put(entry)
      return true
    } catch (retryError) {
      if (isQuotaExceededError(retryError)) {
        cacheDisabledForSession = true
      }
      return false
    }
  }
}
