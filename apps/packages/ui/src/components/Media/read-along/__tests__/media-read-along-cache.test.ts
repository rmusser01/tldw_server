import { beforeEach, describe, expect, it, vi } from 'vitest'

const { rows, mockTable, putState } = vi.hoisted(() => {
  const rows = new Map<string, any>()
  const putState = {
    throwQuotaTimes: 0
  }

  const mockTable = {
    put: vi.fn(async (row: any) => {
      if (putState.throwQuotaTimes > 0) {
        putState.throwQuotaTimes -= 1
        const error = new Error('Quota exceeded')
        error.name = 'QuotaExceededError'
        throw error
      }
      rows.set(row.id, row)
      return row.id
    }),
    get: vi.fn(async (id: string) => rows.get(id)),
    update: vi.fn(async (id: string, changes: Record<string, unknown>) => {
      const current = rows.get(id)
      if (!current) return 0
      rows.set(id, { ...current, ...changes })
      return 1
    }),
    delete: vi.fn(async (id: string) => {
      rows.delete(id)
    }),
    bulkDelete: vi.fn(async (ids: string[]) => {
      for (const id of ids) rows.delete(id)
    }),
    toArray: vi.fn(async () => Array.from(rows.values())),
    orderBy: vi.fn((field: string) => ({
      toArray: vi.fn(async () => {
        const all = Array.from(rows.values())
        all.sort((a, b) => a[field] - b[field])
        return all
      })
    }))
  }

  return { rows, mockTable, putState }
})

vi.mock('@/db/dexie/schema', () => ({
  db: {
    mediaReadAlongAudioCache: mockTable
  }
}))

import {
  getMediaReadAlongAudioCacheEntry,
  resetMediaReadAlongAudioCacheSessionForTests,
  saveMediaReadAlongAudioCacheEntry
} from '../media-read-along-cache'

function fakeBlob(size = 1024): Blob {
  return new Blob([new Uint8Array(size)], { type: 'audio/mpeg' })
}

function cacheEntry(overrides: Record<string, unknown> = {}) {
  return {
    id: 'cache-1',
    createdAt: 1000,
    lastUsedAt: 1000,
    mediaId: 'media-1',
    mediaKind: 'media',
    segmentId: 'segment-1',
    settingsSignature: 'provider:tldw|voice:af',
    textHash: 'a'.repeat(64),
    mimeType: 'audio/mpeg',
    format: 'mp3',
    blob: fakeBlob(),
    sizeBytes: 1024,
    ...overrides
  }
}

describe('media read-along audio cache', () => {
  beforeEach(() => {
    rows.clear()
    putState.throwQuotaTimes = 0
    vi.clearAllMocks()
    resetMediaReadAlongAudioCacheSessionForTests()
  })

  it('saves and retrieves cache entries by id', async () => {
    const entry = cacheEntry()

    const saved = await saveMediaReadAlongAudioCacheEntry(entry)
    const result = await getMediaReadAlongAudioCacheEntry(entry.id)

    expect(saved).toBe(true)
    expect(mockTable.put).toHaveBeenCalledWith(entry)
    expect(result).toMatchObject({
      id: entry.id,
      mediaId: entry.mediaId,
      segmentId: entry.segmentId,
      textHash: entry.textHash
    })
    expect(mockTable.update).toHaveBeenCalledWith(
      entry.id,
      expect.objectContaining({ lastUsedAt: expect.any(Number) })
    )
  })

  it('evicts least recently used entries before writes that exceed the byte cap', async () => {
    rows.set('oldest', cacheEntry({ id: 'oldest', lastUsedAt: 10, sizeBytes: 700 }))
    rows.set('middle', cacheEntry({ id: 'middle', lastUsedAt: 20, sizeBytes: 200 }))
    rows.set('newest', cacheEntry({ id: 'newest', lastUsedAt: 30, sizeBytes: 100 }))

    await saveMediaReadAlongAudioCacheEntry(
      cacheEntry({ id: 'incoming', sizeBytes: 250 }),
      { maxBytes: 1000 }
    )

    expect(mockTable.bulkDelete).toHaveBeenCalledWith(['oldest'])
    expect(rows.has('oldest')).toBe(false)
    expect(rows.has('incoming')).toBe(true)
  })

  it('skips oversized entries without evicting or writing', async () => {
    rows.set('oldest', cacheEntry({ id: 'oldest', lastUsedAt: 10, sizeBytes: 700 }))

    const saved = await saveMediaReadAlongAudioCacheEntry(
      cacheEntry({ id: 'oversized', sizeBytes: 1200 }),
      { maxBytes: 1000 }
    )

    expect(saved).toBe(false)
    expect(mockTable.put).not.toHaveBeenCalled()
    expect(mockTable.bulkDelete).not.toHaveBeenCalled()
    expect(rows.has('oldest')).toBe(true)
    expect(rows.has('oversized')).toBe(false)
  })

  it('retries once after QuotaExceededError by evicting LRU entries', async () => {
    rows.set('oldest', cacheEntry({ id: 'oldest', lastUsedAt: 10, sizeBytes: 500 }))
    rows.set('newest', cacheEntry({ id: 'newest', lastUsedAt: 20, sizeBytes: 100 }))
    putState.throwQuotaTimes = 1

    const saved = await saveMediaReadAlongAudioCacheEntry(
      cacheEntry({ id: 'incoming', sizeBytes: 100 }),
      { maxBytes: 1000 }
    )

    expect(saved).toBe(true)
    expect(mockTable.put).toHaveBeenCalledTimes(2)
    expect(mockTable.bulkDelete).toHaveBeenCalledWith(['oldest'])
    expect(rows.has('incoming')).toBe(true)
  })

  it('disables cache for the session after repeated quota failures', async () => {
    rows.set('oldest', cacheEntry({ id: 'oldest', lastUsedAt: 10, sizeBytes: 500 }))
    putState.throwQuotaTimes = 2

    const saved = await saveMediaReadAlongAudioCacheEntry(
      cacheEntry({ id: 'incoming', sizeBytes: 100 }),
      { maxBytes: 1000 }
    )
    const result = await getMediaReadAlongAudioCacheEntry('oldest')
    const skipped = await saveMediaReadAlongAudioCacheEntry(cacheEntry({ id: 'skipped' }))

    expect(saved).toBe(false)
    expect(result).toBeUndefined()
    expect(skipped).toBe(false)
    expect(mockTable.put).toHaveBeenCalledTimes(2)
  })
})
