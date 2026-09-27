import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStorage } from '@plasmohq/storage/hook'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { clearSetting, setSetting } from '@/services/settings/registry'
import { bgRequest } from '@/services/background-proxy'
import {
  LAST_MEDIA_ID_SETTING,
  MEDIA_REVIEW_SELECTION_SETTING
} from '@/services/settings/ui-settings'
import { downloadBlob } from '@/utils/download-blob'
import { useUndoNotification } from '@/hooks/useUndoNotification'
import type { MediaResultItem } from '@/components/Media/types'
import type { MediaLibraryStorageUsage } from '@/components/Media/MediaLibraryStatsPanel'
import { getErrorStatusCode, useMediaRequestLifetime } from './useMediaSearch'

const MEDIA_COLLECTIONS_STORAGE_KEY = 'media:collections:v1'
const READING_PROGRESS_MEDIA_ID_SEPARATOR = '\u001f'

const toNonNegativeFiniteNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0) return value
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value)
    if (Number.isFinite(parsed) && parsed >= 0) return parsed
  }
  return null
}

const DEFAULT_MEDIA_LIBRARY_STORAGE_USAGE: MediaLibraryStorageUsage = {
  loading: true,
  error: null,
  totalMb: null,
  quotaMb: null,
  usagePercentage: null,
  warning: null
}

const READING_PROGRESS_SUPPORTED_MEDIA_TYPES = new Set([
  'document',
  'docx',
  'epub',
  'html',
  'markdown',
  'md',
  'pdf',
  'text'
])

const isReadingProgressEndpointUnavailableError = (error: unknown): boolean => {
  const statusCode = getErrorStatusCode(error)
  if (statusCode == null) return false
  return statusCode >= 500 || statusCode === 404 || statusCode === 405 || statusCode === 410
}

const supportsReadingProgressForResult = (item: MediaResultItem): boolean => {
  if (item.kind !== 'media') return false
  const rawType =
    item.meta?.type ??
    item.raw?.type ??
    item.raw?.media_type ??
    item.raw?.metadata?.type ??
    item.raw?.safe_metadata?.type
  if (typeof rawType !== 'string') return false
  return READING_PROGRESS_SUPPORTED_MEDIA_TYPES.has(rawType.trim().toLowerCase())
}

const areReadingProgressMapsEqual = (
  left: Map<string, number>,
  right: Map<string, number>
): boolean => {
  if (left.size !== right.size) return false
  for (const [key, value] of right) {
    if (left.get(key) !== value) return false
  }
  return true
}

type MediaCollectionRecord = {
  id: string
  name: string
  itemIds: string[]
  createdAt: string
  updatedAt: string
}

type OwnedMediaValues<T> = { ownerScope: string; values: T[] }

/** Never adopt unowned legacy values, or render the previous key while a new key hydrates. */
function useOwnedMediaValues<T>(key: string, ownerScope: string | null, isCurrent: () => boolean) {
  const [record, setRecord, meta] = useStorage<OwnedMediaValues<T> | null>(
    `${key}:owner:${ownerScope ?? 'unresolved'}`, null
  )
  const current = isCurrent()
  const values = useMemo(() => current && record?.ownerScope === ownerScope && Array.isArray(record.values) ? record.values : [],
    [current, record, ownerScope])
  const setValues = useCallback((next: React.SetStateAction<T[]>) => {
    if (!ownerScope || !isCurrent() || meta?.isLoading) return
    return setRecord(previous => {
      const values = previous?.ownerScope === ownerScope && Array.isArray(previous.values) ? previous.values : []
      return { ownerScope, values: typeof next === 'function' ? next(values) : next }
    })
  }, [ownerScope, isCurrent, meta?.isLoading, setRecord])
  return [values, setValues] as const
}

export interface UseMediaSelectionDeps {
  ownerScope: string | null
  t: (key: string, opts?: Record<string, any>) => string
  message: {
    error: (msg: string) => void
    warning: (msg: string) => void
    success: (msg: string) => void
  }
  displayResults: MediaResultItem[]
  selected: MediaResultItem | null
  setSelected: React.Dispatch<React.SetStateAction<MediaResultItem | null>>
  setSelectedContent: React.Dispatch<React.SetStateAction<string>>
  setSelectedDetail: React.Dispatch<React.SetStateAction<any>>
  setLastFetchedId: React.Dispatch<React.SetStateAction<string | number | null>>
  refetch: () => Promise<any>
}

export function useMediaSelection(deps: UseMediaSelectionDeps) {
  const {
    t, message, displayResults, ownerScope,
    selected, setSelected, setSelectedContent, setSelectedDetail, setLastFetchedId,
    refetch
  } = deps
  const navigate = useNavigate()
  const { showUndoNotification } = useUndoNotification()

  const ownerRef = useRef(ownerScope)
  ownerRef.current = ownerScope
  const lifetime = useMediaRequestLifetime(() => {
    setBulkSelectedIds([])
    setBulkKeywordsDraft('')
    setCollectionDraftName('')
    setActiveCollectionId(null)
    setShowFavoritesOnly(false)
    setReadingProgressMap(new Map())
    setLibraryStorageUsage({ ...DEFAULT_MEDIA_LIBRARY_STORAGE_USAGE, loading: false })
  })
  const captureOperation = useCallback(() => {
    const signal = lifetime.current.signal
    const isCurrent = () => Boolean(ownerScope && ownerRef.current === ownerScope && !signal.aborted)
    const request = async <T,>(init: Parameters<typeof bgRequest<T>>[0]): Promise<T> => {
      if (!isCurrent()) throw new DOMException('Media account changed', 'AbortError')
      const result = await bgRequest<T>({ ...init, abortSignal: signal })
      if (!isCurrent()) throw new DOMException('Media account changed', 'AbortError')
      return result
    }
    return { signal, isCurrent, request }
  }, [lifetime, ownerScope])
  const isOwnerCurrent = useCallback(() => captureOperation().isCurrent(), [captureOperation])

  // Favorites
  const [favorites, setFavorites] = useOwnedMediaValues<string>('media:favorites', ownerScope, isOwnerCurrent)
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false)
  const favoritesSet = useMemo(() => new Set(favorites || []), [favorites])

  // Bulk selection
  const [bulkSelectionMode, setBulkSelectionMode] = useState(false)
  const [bulkSelectedIds, setBulkSelectedIds] = useState<string[]>([])
  const [bulkKeywordsDraft, setBulkKeywordsDraft] = useState('')
  const [bulkExportFormat, setBulkExportFormat] = useState<'json' | 'markdown' | 'text'>(
    'json'
  )

  // Collections
  const [mediaCollections, setMediaCollections] = useOwnedMediaValues<MediaCollectionRecord>(
    MEDIA_COLLECTIONS_STORAGE_KEY, ownerScope, isOwnerCurrent
  )
  const [activeCollectionId, setActiveCollectionId] = useState<string | null>(null)
  const [collectionDraftName, setCollectionDraftName] = useState('')

  // Storage usage
  const [libraryStorageUsage, setLibraryStorageUsage] = useState<MediaLibraryStorageUsage>(
    DEFAULT_MEDIA_LIBRARY_STORAGE_USAGE
  )

  // Reading progress
  const [readingProgressMap, setReadingProgressMap] = useState<Map<string, number>>(new Map())
  const readingProgressUnavailableRef = useRef(false)
  const readingProgressMediaIdsKey = useMemo(() => {
    const mediaIds = displayResults
      .filter((r) => supportsReadingProgressForResult(r))
      .map((r) => String(r.id))
    return mediaIds.join(READING_PROGRESS_MEDIA_ID_SEPARATOR)
  }, [displayResults])
  const updateReadingProgressMap = useCallback((entries: Array<[string, number]>) => {
    const next = new Map(entries)
    setReadingProgressMap((prev) =>
      areReadingProgressMapsEqual(prev, next) ? prev : next
    )
  }, [])

  const toggleFavorite = useCallback((id: string) => {
    const idStr = String(id)
    setFavorites((prev: string[] | undefined) => {
      const currentFavorites = prev || []
      const set = new Set(currentFavorites)
      if (set.has(idStr)) {
        set.delete(idStr)
      } else {
        set.add(idStr)
      }
      return Array.from(set)
    })
  }, [setFavorites])

  const isFavorite = useCallback((id: string) => {
    return favoritesSet.has(String(id))
  }, [favoritesSet])

  const bulkSelectedIdSet = useMemo(
    () => new Set(bulkSelectedIds),
    [bulkSelectedIds]
  )
  const bulkSelectedItems = useMemo(
    () => displayResults.filter((item) => bulkSelectedIdSet.has(String(item.id))),
    [bulkSelectedIdSet, displayResults]
  )
  const bulkSelectedMediaItems = useMemo(
    () => bulkSelectedItems.filter((item) => item.kind === 'media'),
    [bulkSelectedItems]
  )
  const bulkSelectedNoteCount = bulkSelectedItems.length - bulkSelectedMediaItems.length

  const activeCollection = useMemo(
    () => mediaCollections.find((entry) => entry.id === activeCollectionId) || null,
    [activeCollectionId, mediaCollections]
  )

  // Sync bulk selection with visible results
  useEffect(() => {
    if (bulkSelectedIds.length === 0) return
    const visibleIdSet = new Set(displayResults.map((item) => String(item.id)))
    setBulkSelectedIds((prev) => {
      const next = prev.filter((id) => visibleIdSet.has(id))
      return next.length === prev.length ? prev : next
    })
  }, [bulkSelectedIds.length, displayResults])

  useEffect(() => {
    if (bulkSelectionMode || bulkSelectedIds.length === 0) return
    setBulkSelectedIds([])
  }, [bulkSelectedIds.length, bulkSelectionMode])

  useEffect(() => {
    if (!activeCollectionId) return
    const exists = mediaCollections.some((entry) => entry.id === activeCollectionId)
    if (!exists) {
      setActiveCollectionId(null)
    }
  }, [activeCollectionId, mediaCollections])

  // Reading progress
  useEffect(() => {
    const { signal, isCurrent } = captureOperation()
    if (!isCurrent()) return
    if (readingProgressUnavailableRef.current) {
      updateReadingProgressMap([])
      return
    }
    const mediaIds = readingProgressMediaIdsKey === ''
      ? []
      : readingProgressMediaIdsKey.split(READING_PROGRESS_MEDIA_ID_SEPARATOR)
    if (mediaIds.length === 0) {
      updateReadingProgressMap([])
      return
    }
    const getReadingProgress = (tldwClient as any).getReadingProgress
    if (typeof getReadingProgress !== 'function') {
      updateReadingProgressMap([])
      return
    }
    let cancelled = false
    const fetchProgress = async () => {
      const entries: Array<[string, number]> = []
      for (const mediaId of mediaIds) {
        if (cancelled || !isCurrent()) return
        try {
          const result = await getReadingProgress.call(tldwClient, mediaId, { signal })
          if (cancelled || !isCurrent()) return
          if (result?.has_progress !== false) {
            const pct = result?.percent_complete
            if (typeof pct === 'number' && pct > 0) {
              entries.push([mediaId, pct])
            }
          }
        } catch (error) {
          if (cancelled || !isCurrent()) return
          if (isReadingProgressEndpointUnavailableError(error)) {
            readingProgressUnavailableRef.current = true
            updateReadingProgressMap(entries)
            return
          }
        }
      }
      if (!cancelled && isCurrent()) {
        updateReadingProgressMap(entries)
      }
    }
    void fetchProgress()
    return () => { cancelled = true }
  }, [readingProgressMediaIdsKey, updateReadingProgressMap, captureOperation])

  // Storage usage
  const refreshLibraryStorageUsage = useCallback(async () => {
    const { request, isCurrent } = captureOperation()
    if (!isCurrent()) return
    setLibraryStorageUsage((prev) => ({
      ...prev,
      loading: true,
      error: null
    }))

    try {
      let response: any = null
      if (typeof (tldwClient as any).getCurrentUserStorageQuota === 'function') {
        try {
          response = await request({ path: '/api/v1/users/storage', method: 'GET' })
        } catch {
          if (!isCurrent()) return
          response = null
        }
      }
      if (
        response == null &&
        typeof (tldwClient as any).getCurrentUserProfile === 'function'
      ) {
        const profile = await request<{ quotas?: Record<string, unknown> }>({
          path: '/api/v1/users/me/profile?sections=quotas', method: 'GET'
        })
        const quotas = profile?.quotas ?? {}
        response = {
          storage_used_mb: quotas?.storage_used_mb,
          storage_quota_mb: quotas?.storage_quota_mb,
          usage_percentage: quotas?.usage_percentage
        }
      }
      if (!isCurrent()) return
      const totalMb = toNonNegativeFiniteNumber(
        response?.storage_used_mb ??
          response?.storageUsedMb ??
          response?.usage?.total_mb ??
          response?.usage?.totalMb
      )
      const quotaMb = toNonNegativeFiniteNumber(
        response?.storage_quota_mb ??
          response?.storageQuotaMb ??
          response?.quota_mb ??
          response?.quotaMb
      )
      const usagePercentage = toNonNegativeFiniteNumber(
        response?.usage_percentage ?? response?.usagePercentage
      )
      const warning =
        typeof response?.warning === 'string' && response.warning.trim().length > 0
          ? response.warning.trim()
          : null

      setLibraryStorageUsage({
        loading: false,
        error: null,
        totalMb,
        quotaMb,
        usagePercentage,
        warning
      })
    } catch {
      if (!isCurrent()) return
      setLibraryStorageUsage({
        loading: false,
        error: 'Unable to load storage usage.',
        totalMb: null,
        quotaMb: null,
        usagePercentage: null,
        warning: null
      })
    }
  }, [captureOperation])

  useEffect(() => {
    void refreshLibraryStorageUsage()
  }, [refreshLibraryStorageUsage])

  // Bulk action handlers
  const handleToggleBulkSelectionMode = useCallback(() => {
    setBulkSelectionMode((prev) => !prev)
    setBulkKeywordsDraft('')
  }, [])

  const toggleBulkItemSelection = useCallback((id: string | number) => {
    const idStr = String(id)
    setBulkSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(idStr)) {
        next.delete(idStr)
      } else {
        next.add(idStr)
      }
      return Array.from(next)
    })
  }, [])

  const handleSelectAllVisibleItems = useCallback(() => {
    setBulkSelectedIds(displayResults.map((item) => String(item.id)))
  }, [displayResults])

  const handleClearBulkSelection = useCallback(() => {
    setBulkSelectedIds([])
  }, [])

  const handleBulkAddKeywords = useCallback(async () => {
    const { request, isCurrent } = captureOperation()
    if (!isCurrent()) return
    const keywordsToAdd = bulkKeywordsDraft
      .split(',')
      .map((keyword) => keyword.trim())
      .filter((keyword, index, all) => keyword.length > 0 && all.indexOf(keyword) === index)

    if (keywordsToAdd.length === 0) {
      message.warning(
        t('review:mediaPage.bulkAddKeywordsMissing', {
          defaultValue: 'Enter at least one keyword.'
        })
      )
      return
    }

    if (bulkSelectedMediaItems.length === 0) {
      message.warning(
        t('review:mediaPage.bulkAddKeywordsNoMedia', {
          defaultValue: 'Select at least one media item to tag.'
        })
      )
      return
    }

    let updatedCount = 0
    let failedCount = 0
    const updatedKeywordMap = new Map<string, string[]>()

    for (const item of bulkSelectedMediaItems) {
      if (!isCurrent()) return
      const currentKeywords = Array.isArray(item.keywords) ? item.keywords : []
      const mergedKeywords = Array.from(new Set([...currentKeywords, ...keywordsToAdd]))
      try {
        await request({
          path: `/api/v1/media/${item.id}` as any,
          method: 'PUT' as any,
          headers: { 'Content-Type': 'application/json' },
          body: { keywords: mergedKeywords }
        })
        updatedKeywordMap.set(String(item.id), mergedKeywords)
        updatedCount += 1
      } catch {
        if (!isCurrent()) return
        failedCount += 1
      }
    }

    if (updatedKeywordMap.size > 0) {
      setSelected((prev) => {
        if (!prev) return prev
        const nextKeywords = updatedKeywordMap.get(String(prev.id))
        if (!nextKeywords) return prev
        return { ...prev, keywords: nextKeywords }
      })
      await refetch()
      if (!isCurrent()) return
    }

    if (failedCount > 0) {
      message.warning(
        t('review:mediaPage.bulkAddKeywordsPartial', {
          defaultValue: 'Updated {{updated}} item(s), {{failed}} failed.',
          updated: updatedCount,
          failed: failedCount
        })
      )
    } else {
      message.success(
        t('review:mediaPage.bulkAddKeywordsSuccess', {
          defaultValue: 'Updated keywords for {{count}} item(s).',
          count: updatedCount
        })
      )
    }

    if (bulkSelectedNoteCount > 0) {
      message.warning(
        t('review:mediaPage.bulkAddKeywordsSkippedNotes', {
          defaultValue: 'Skipped {{count}} note item(s).',
          count: bulkSelectedNoteCount
        })
      )
    }
  }, [
    captureOperation,
    bulkKeywordsDraft,
    bulkSelectedMediaItems,
    bulkSelectedNoteCount,
    message,
    refetch,
    setSelected,
    t
  ])

  const handleBulkDelete = useCallback(async () => {
    const { request, isCurrent } = captureOperation()
    if (!isCurrent()) return
    if (bulkSelectedItems.length === 0) {
      message.warning(
        t('review:mediaPage.bulkDeleteNothingSelected', {
          defaultValue: 'Select items to delete.'
        })
      )
      return
    }

    const parseVersion = (value: unknown): number | null => {
      if (typeof value === 'number' && Number.isFinite(value)) return value
      if (typeof value === 'string') {
        const trimmed = value.trim()
        if (/^\d+$/.test(trimmed)) return Number.parseInt(trimmed, 10)
      }
      return null
    }

    let deletedCount = 0
    let failedCount = 0
    const deletedIdSet = new Set<string>()

    for (const item of bulkSelectedItems) {
      if (!isCurrent()) return
      try {
        if (item.kind === 'note') {
          const latest = await request<any>({
            path: `/api/v1/notes/${item.id}` as any,
            method: 'GET' as any
          })
          const expectedVersion =
            parseVersion(latest?.version) ?? parseVersion(latest?.metadata?.version)
          if (expectedVersion == null) {
            throw new Error('Missing expected version')
          }
          await request({
            path: `/api/v1/notes/${item.id}` as any,
            method: 'DELETE' as any,
            headers: { 'expected-version': String(expectedVersion) }
          })
        } else {
          await request({
            path: `/api/v1/media/${item.id}` as any,
            method: 'DELETE' as any
          })
        }
        deletedIdSet.add(String(item.id))
        deletedCount += 1
      } catch {
        if (!isCurrent()) return
        failedCount += 1
      }
    }

    if (deletedIdSet.size > 0) {
      setFavorites((prev: string[] | undefined) =>
        (prev || []).filter((favoriteId) => !deletedIdSet.has(String(favoriteId)))
      )
      setSelected((prev) => {
        if (!prev) return prev
        if (!deletedIdSet.has(String(prev.id))) return prev
        return null
      })
      setBulkSelectedIds((prev) => prev.filter((id) => !deletedIdSet.has(id)))
      setSelectedContent('')
      setSelectedDetail(null)
      setLastFetchedId(null)
      await refetch()
      if (!isCurrent()) return
      void refreshLibraryStorageUsage()
    }

    if (failedCount > 0) {
      message.warning(
        t('review:mediaPage.bulkDeletePartial', {
          defaultValue: 'Deleted {{deleted}} item(s), {{failed}} failed.',
          deleted: deletedCount,
          failed: failedCount
        })
      )
      return
    }

    message.success(
      t('review:mediaPage.bulkDeleteSuccess', {
        defaultValue: 'Deleted {{count}} item(s).',
        count: deletedCount
      })
    )
  }, [
    captureOperation,
    bulkSelectedItems,
    message,
    refetch,
    refreshLibraryStorageUsage,
    setFavorites,
    setLastFetchedId,
    setSelected,
    setSelectedContent,
    setSelectedDetail,
    t
  ])

  const handleBulkExport = useCallback(() => {
    if (!isOwnerCurrent()) return
    if (bulkSelectedItems.length === 0) {
      message.warning(
        t('review:mediaPage.bulkExportNothingSelected', {
          defaultValue: 'Select items to export.'
        })
      )
      return
    }

    const exportPayload = {
      exported_at: new Date().toISOString(),
      items: bulkSelectedItems.map((item) => ({
        id: item.id,
        kind: item.kind,
        title: item.title || `${item.kind} ${item.id}`,
        snippet: item.snippet || '',
        keywords: Array.isArray(item.keywords) ? item.keywords : [],
        type: item.meta?.type || null,
        source: item.meta?.source || null
      }))
    }

    let fileContent = ''
    let extension = 'json'
    let mimeType = 'application/json'

    if (bulkExportFormat === 'markdown') {
      extension = 'md'
      mimeType = 'text/markdown'
      const lines: string[] = ['# Media Bulk Export', '']
      for (const item of exportPayload.items) {
        lines.push(`## ${item.title}`)
        lines.push(`- ID: ${item.id}`)
        lines.push(`- Kind: ${item.kind}`)
        if (item.type) lines.push(`- Type: ${item.type}`)
        if (item.source) lines.push(`- Source: ${item.source}`)
        if (item.keywords.length > 0) {
          lines.push(`- Keywords: ${item.keywords.join(', ')}`)
        }
        if (item.snippet) {
          lines.push('', item.snippet)
        }
        lines.push('')
      }
      fileContent = lines.join('\n')
    } else if (bulkExportFormat === 'text') {
      extension = 'txt'
      mimeType = 'text/plain'
      const lines: string[] = []
      for (const item of exportPayload.items) {
        lines.push(`${item.title} [${item.kind} #${item.id}]`)
        if (item.type) lines.push(`Type: ${item.type}`)
        if (item.source) lines.push(`Source: ${item.source}`)
        if (item.keywords.length > 0) lines.push(`Keywords: ${item.keywords.join(', ')}`)
        if (item.snippet) lines.push(`Snippet: ${item.snippet}`)
        lines.push('')
      }
      fileContent = lines.join('\n')
    } else {
      fileContent = JSON.stringify(exportPayload, null, 2)
    }

    const blob = new Blob([fileContent], { type: mimeType })
    downloadBlob(blob, `media-bulk-export-${Date.now()}.${extension}`)
    message.success(
      t('review:mediaPage.bulkExportReady', {
        defaultValue: 'Bulk export ready.'
      })
    )
  }, [bulkExportFormat, bulkSelectedItems, isOwnerCurrent, message, t])

  const handleAddSelectionToCollection = useCallback(() => {
    if (!isOwnerCurrent()) return
    if (bulkSelectedItems.length === 0) {
      message.warning(
        t('review:mediaPage.collectionRequiresSelection', {
          defaultValue: 'Select at least one item first.'
        })
      )
      return
    }

    const targetCollectionName =
      collectionDraftName.trim() || activeCollection?.name?.trim() || ''
    if (!targetCollectionName) {
      message.warning(
        t('review:mediaPage.collectionNameRequired', {
          defaultValue: 'Enter a collection name.'
        })
      )
      return
    }

    const selectedItemIds = bulkSelectedItems.map((item) => String(item.id))
    const now = new Date().toISOString()
    const normalizedName = targetCollectionName.toLowerCase()
    const existing = mediaCollections.find(
      (entry) => entry.name.trim().toLowerCase() === normalizedName
    )
    if (existing) {
      setMediaCollections((prevCollections) => {
        const collections = Array.isArray(prevCollections) ? prevCollections : []
        return collections.map((entry) => {
          if (entry.id !== existing.id) return entry
          const mergedIds = Array.from(
            new Set([...entry.itemIds.map((id) => String(id)), ...selectedItemIds])
          )
          return {
            ...entry,
            itemIds: mergedIds,
            updatedAt: now
          }
        })
      })
      setActiveCollectionId(existing.id)
    } else {
      const slug =
        targetCollectionName
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, '-')
          .replace(/(^-|-$)/g, '') || 'collection'
      const created: MediaCollectionRecord = {
        id: `${slug}-${Date.now()}`,
        name: targetCollectionName,
        itemIds: Array.from(new Set(selectedItemIds)),
        createdAt: now,
        updatedAt: now
      }
      setMediaCollections((prevCollections) => {
        const collections = Array.isArray(prevCollections) ? prevCollections : []
        return [...collections, created]
      })
      setActiveCollectionId(created.id)
    }
    setCollectionDraftName('')
    message.success(
      t('review:mediaPage.collectionSaved', {
        defaultValue: 'Saved selection to collection.'
      })
    )
  }, [
    isOwnerCurrent,
    activeCollection?.name,
    bulkSelectedItems,
    collectionDraftName,
    mediaCollections,
    message,
    setMediaCollections,
    t
  ])

  const handleOpenSelectionInMultiReview = useCallback(async () => {
    const { isCurrent } = captureOperation()
    if (!isCurrent()) return
    if (bulkSelectedIds.length === 0) {
      message.warning(
        t('review:mediaPage.bulkOpenInMultiReviewNone', {
          defaultValue: 'Select items to open in multi-review.'
        })
      )
      return
    }
    await setSetting(MEDIA_REVIEW_SELECTION_SETTING, bulkSelectedIds)
    if (!isCurrent()) return
    await setSetting(LAST_MEDIA_ID_SETTING, String(bulkSelectedIds[0]))
    if (!isCurrent()) return
    navigate('/media-multi')
  }, [bulkSelectedIds, captureOperation, message, navigate, t])

  const handleOpenCollectionInMultiReview = useCallback(async () => {
    const { isCurrent } = captureOperation()
    if (!isCurrent()) return
    if (!activeCollection || activeCollection.itemIds.length === 0) {
      message.warning(
        t('review:mediaPage.collectionEmpty', {
          defaultValue: 'No items in this collection.'
        })
      )
      return
    }
    const collectionIds = activeCollection.itemIds.map((id) => String(id))
    await setSetting(MEDIA_REVIEW_SELECTION_SETTING, collectionIds)
    if (!isCurrent()) return
    await setSetting(LAST_MEDIA_ID_SETTING, String(collectionIds[0]))
    if (!isCurrent()) return
    navigate('/media-multi')
  }, [activeCollection, captureOperation, message, navigate, t])

  const handleDeleteItem = useCallback(
    async (item: MediaResultItem, detail: any | null) => {
      const { request, isCurrent } = captureOperation()
      if (!isCurrent()) return
      const id = item.id
      const idStr = String(id)
      const wasFavorite = favoritesSet.has(idStr)
      const itemTitle =
        item.title ||
        `${t('review:mediaPage.media', { defaultValue: 'Media' })} ${idStr}`

      const parseVersionCandidate = (value: unknown): number | null => {
        if (typeof value === 'number' && Number.isFinite(value)) return value
        if (typeof value === 'string') {
          const trimmed = value.trim()
          if (/^\d+$/.test(trimmed)) return Number.parseInt(trimmed, 10)
        }
        return null
      }

      let deletedAtVersion: number | null = null

      try {
        if (item.kind === 'note') {
          let expectedVersion: number | null = null
          const versionCandidates = [
            detail?.version,
            detail?.metadata?.version,
            item.raw?.version,
            item.raw?.metadata?.version
          ]
          for (const candidate of versionCandidates) {
            const parsed = parseVersionCandidate(candidate)
            if (parsed != null) {
              expectedVersion = parsed
              break
            }
          }
          if (expectedVersion == null) {
            try {
              const latest = await request<any>({
                path: `/api/v1/notes/${id}` as any,
                method: 'GET' as any
              })
              expectedVersion =
                parseVersionCandidate(latest?.version) ??
                parseVersionCandidate(latest?.metadata?.version)
            } catch {
              throw new Error(
                t('review:mediaPage.noteDeleteNeedsReload', {
                  defaultValue: 'Unable to delete note. Reload and try again.'
                })
              )
            }
          }
          if (expectedVersion == null) {
            throw new Error(
              t('review:mediaPage.noteDeleteNeedsReload', {
                defaultValue: 'Unable to delete note. Reload and try again.'
              })
            )
          }
          await request({
            path: `/api/v1/notes/${id}` as any,
            method: 'DELETE' as any,
            headers: { 'expected-version': String(expectedVersion) }
          })
          deletedAtVersion = expectedVersion + 1
        } else {
          await request({
            path: `/api/v1/media/${id}` as any,
            method: 'DELETE' as any
          })
        }
      } catch (err) {
        if (!isCurrent()) return
        const status = err && typeof err === 'object' && 'status' in err
          ? (err as { status?: number }).status
          : undefined
        const msg = err && typeof err === 'object' && 'message' in err
          ? String((err as { message?: unknown }).message || '')
          : ''
        if (
          item.kind === 'note' &&
          (status === 409 ||
            msg.toLowerCase().includes('expected-version') ||
            msg.toLowerCase().includes('version'))
        ) {
          throw new Error(
            t('review:mediaPage.noteDeleteNeedsReload', {
              defaultValue: 'Unable to delete note. Reload and try again.'
            })
          )
        }
        throw err
      }

      setFavorites((prev: string[] | undefined) =>
        (prev || []).filter((fav) => fav !== idStr)
      )

      await clearSetting(LAST_MEDIA_ID_SETTING)
      if (!isCurrent()) return

      const remainingResults = displayResults.filter(
        (r) => String(r.id) !== idStr
      )
      if (remainingResults.length > 0) {
        const currentIndex = displayResults.findIndex(
          (r) => String(r.id) === idStr
        )
        const nextIndex =
          currentIndex >= 0
            ? Math.min(currentIndex, remainingResults.length - 1)
            : 0
        setSelected(remainingResults[nextIndex])
      } else {
        setSelected(null)
        setSelectedContent('')
        setSelectedDetail(null)
        setLastFetchedId(null)
      }

      void refetch()
      void refreshLibraryStorageUsage()

      showUndoNotification({
        title: t('review:mediaPage.itemMovedToTrash', {
          defaultValue: 'Moved to trash'
        }),
        description: t('review:mediaPage.itemMovedToTrashDesc', {
          defaultValue: '"{{title}}" moved to trash.',
          title: itemTitle
        }),
        onUndo: async () => {
          if (!isCurrent()) return
          if (item.kind === 'note') {
            if (deletedAtVersion != null) {
              await request({
                path: `/api/v1/notes/${id}/restore?expected_version=${deletedAtVersion}` as any,
                method: 'POST' as any
              })
            }
          } else {
            await request({
              path: `/api/v1/media/${id}/restore` as any,
              method: 'POST' as any
            })
          }
          if (wasFavorite) {
            setFavorites((prev: string[] | undefined) => {
              const next = new Set(prev || [])
              next.add(idStr)
              return Array.from(next)
            })
          }
          const refreshed = await refetch()
          if (!isCurrent()) return
          void refreshLibraryStorageUsage()
          const restoredItem = refreshed.data?.find(
            (r: MediaResultItem) => String(r.id) === idStr
          )
          if (restoredItem) {
            setSelected((prev) => {
              const prevId = prev?.id != null ? String(prev.id) : null
              if (!prevId || prevId === idStr) return restoredItem
              return prev
            })
          }
        }
      })
    },
    [
      captureOperation,
      displayResults,
      favoritesSet,
      refetch,
      refreshLibraryStorageUsage,
      setFavorites,
      setLastFetchedId,
      setSelected,
      setSelectedContent,
      setSelectedDetail,
      showUndoNotification,
      t
    ]
  )

  return {
    // Favorites
    favorites, setFavorites,
    favoritesSet,
    showFavoritesOnly, setShowFavoritesOnly,
    toggleFavorite,
    isFavorite,
    // Bulk selection
    bulkSelectionMode, setBulkSelectionMode,
    bulkSelectedIds, setBulkSelectedIds,
    bulkKeywordsDraft, setBulkKeywordsDraft,
    bulkExportFormat, setBulkExportFormat,
    bulkSelectedIdSet,
    bulkSelectedItems,
    bulkSelectedMediaItems,
    bulkSelectedNoteCount,
    // Collections
    mediaCollections, setMediaCollections,
    activeCollectionId, setActiveCollectionId,
    collectionDraftName, setCollectionDraftName,
    activeCollection,
    // Storage
    libraryStorageUsage,
    refreshLibraryStorageUsage,
    // Reading progress
    readingProgressMap,
    // Actions
    handleToggleBulkSelectionMode,
    toggleBulkItemSelection,
    handleSelectAllVisibleItems,
    handleClearBulkSelection,
    handleBulkAddKeywords,
    handleBulkDelete,
    handleBulkExport,
    handleAddSelectionToCollection,
    handleOpenSelectionInMultiReview,
    handleOpenCollectionInMultiReview,
    handleDeleteItem,
  }
}
