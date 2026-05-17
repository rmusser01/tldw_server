import { beforeEach, describe, expect, it, vi } from 'vitest'

const { rows, mockTable, putState, tableState } = vi.hoisted(() => {
  const rows = new Map<string, any>()
  const putState = {
    throwQuotaTimes: 0,
    beforeCommit: undefined as Promise<void> | undefined,
    afterCommit: undefined as ((row: any) => void) | undefined
  }
  const tableState = {
    afterToArray: undefined as (() => void) | undefined
  }

  const mockTable = {
    put: vi.fn(async (row: any) => {
      if (putState.throwQuotaTimes > 0) {
        putState.throwQuotaTimes -= 1
        const error = new Error('Quota exceeded')
        error.name = 'QuotaExceededError'
        throw error
      }
      await putState.beforeCommit
      rows.set(row.id, row)
      putState.afterCommit?.(row)
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
    toArray: vi.fn(async () => {
      const values = Array.from(rows.values())
      tableState.afterToArray?.()
      return values
    }),
    orderBy: vi.fn((field: string) => ({
      toArray: vi.fn(async () => {
        const all = Array.from(rows.values())
        all.sort((a, b) => a[field] - b[field])
        tableState.afterToArray?.()
        return all
      })
    }))
  }

  return { rows, mockTable, putState, tableState }
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
    putState.beforeCommit = undefined
    putState.afterCommit = undefined
    tableState.afterToArray = undefined
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

  it('does not evict or write when the save guard fails after preflight', async () => {
    rows.set('oldest', cacheEntry({ id: 'oldest', lastUsedAt: 10, sizeBytes: 900 }))
    const entry = cacheEntry({ id: 'guarded', sizeBytes: 200 })
    let checks = 0

    const saved = await saveMediaReadAlongAudioCacheEntry(entry, {
      maxBytes: 1000,
      shouldContinue: () => checks++ === 0
    })

    expect(saved).toBe(false)
    expect(mockTable.put).not.toHaveBeenCalled()
    expect(mockTable.bulkDelete).not.toHaveBeenCalled()
    expect(rows.has('oldest')).toBe(true)
    expect(rows.has('guarded')).toBe(false)
  })

  it('does not evict entries when the save guard fails immediately before bulk delete', async () => {
    rows.set('oldest', cacheEntry({ id: 'oldest', lastUsedAt: 10, sizeBytes: 900 }))
    const entry = cacheEntry({ id: 'guarded-before-delete', sizeBytes: 200 })
    let allowSave = true
    tableState.afterToArray = () => {
      allowSave = false
    }

    const saved = await saveMediaReadAlongAudioCacheEntry(entry, {
      maxBytes: 1000,
      shouldContinue: () => allowSave
    })

    expect(saved).toBe(false)
    expect(mockTable.bulkDelete).not.toHaveBeenCalled()
    expect(mockTable.put).not.toHaveBeenCalled()
    expect(rows.has('oldest')).toBe(true)
    expect(rows.has(entry.id)).toBe(false)
  })

  it('removes a row when the save guard fails after a delayed put commits', async () => {
    const entry = cacheEntry({ id: 'stale-after-put' })
    let allowSave = true
    let releasePut!: () => void
    putState.beforeCommit = new Promise<void>((resolve) => {
      releasePut = resolve
    })

    const savePromise = saveMediaReadAlongAudioCacheEntry(entry, {
      shouldContinue: () => allowSave
    })

    await vi.waitFor(() => expect(mockTable.put).toHaveBeenCalledWith(entry))
    allowSave = false
    releasePut()

    await expect(savePromise).resolves.toBe(false)
    expect(mockTable.delete).toHaveBeenCalledWith(entry.id)
    expect(rows.has(entry.id)).toBe(false)
  })

  it('does not remove a newer matching-key write when cleaning up a stale delayed put', async () => {
    const entry = cacheEntry({ id: 'shared-key' })
    const newerEntry = cacheEntry({ id: entry.id, createdAt: 2000, lastUsedAt: 2000 })
    let allowSave = true
    let releasePut!: () => void
    putState.beforeCommit = new Promise<void>((resolve) => {
      releasePut = resolve
    })
    putState.afterCommit = () => {
      rows.set(entry.id, newerEntry)
    }

    const savePromise = saveMediaReadAlongAudioCacheEntry(entry, {
      shouldContinue: () => allowSave
    })

    await vi.waitFor(() => expect(mockTable.put).toHaveBeenCalledWith(entry))
    allowSave = false
    releasePut()

    await expect(savePromise).resolves.toBe(false)
    expect(mockTable.delete).not.toHaveBeenCalled()
    expect(rows.get(entry.id)).toBe(newerEntry)
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

  it('does not retry-evict entries when the save guard fails before quota bulk delete', async () => {
    rows.set('oldest', cacheEntry({ id: 'oldest', lastUsedAt: 10, sizeBytes: 100 }))
    const entry = cacheEntry({ id: 'stale-before-retry-delete', sizeBytes: 100 })
    putState.throwQuotaTimes = 1
    let allowSave = true
    let reads = 0
    tableState.afterToArray = () => {
      reads += 1
      if (reads === 2) {
        allowSave = false
      }
    }

    const saved = await saveMediaReadAlongAudioCacheEntry(entry, {
      maxBytes: 1000,
      shouldContinue: () => allowSave
    })

    expect(saved).toBe(false)
    expect(mockTable.put).toHaveBeenCalledTimes(1)
    expect(mockTable.bulkDelete).not.toHaveBeenCalled()
    expect(rows.has('oldest')).toBe(true)
    expect(rows.has(entry.id)).toBe(false)
  })

  it('removes a row when the save guard fails after a delayed quota retry put commits', async () => {
    const entry = cacheEntry({ id: 'stale-retry-after-put', sizeBytes: 100 })
    rows.set('oldest', cacheEntry({ id: 'oldest', lastUsedAt: 10, sizeBytes: 500 }))
    putState.throwQuotaTimes = 1
    let allowSave = true
    let releasePut!: () => void
    putState.beforeCommit = new Promise<void>((resolve) => {
      releasePut = resolve
    })

    const savePromise = saveMediaReadAlongAudioCacheEntry(entry, {
      maxBytes: 1000,
      shouldContinue: () => allowSave
    })

    await vi.waitFor(() => expect(mockTable.put).toHaveBeenCalledTimes(2))
    allowSave = false
    releasePut()

    await expect(savePromise).resolves.toBe(false)
    expect(mockTable.bulkDelete).toHaveBeenCalledWith(['oldest'])
    expect(mockTable.delete).toHaveBeenCalledWith(entry.id)
    expect(rows.has(entry.id)).toBe(false)
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
