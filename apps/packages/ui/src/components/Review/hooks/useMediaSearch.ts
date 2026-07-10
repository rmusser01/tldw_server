import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Storage } from '@plasmohq/storage'
import { safeStorageSerde } from '@/utils/safe-storage'
import { bgRequest } from '@/services/background-proxy'
import { useDebounce } from '@/hooks/useDebounce'
import {
  buildMediaSearchPayload,
  DEFAULT_MEDIA_SEARCH_FIELDS,
  hasDefaultMediaSearchFields,
  hasMediaSearchFilters,
  type MediaBoostFields,
  type MediaDateRange,
  type MediaSearchField,
  type MediaSearchMode,
  type MediaSortBy
} from '@/components/Review/mediaSearchRequest'
import {
  buildMetadataSearchPath,
  createMetadataSearchFilter,
  normalizeMetadataSearchFilters,
  type MetadataMatchMode,
  type MetadataSearchFilter,
  validateMetadataSearchFilters
} from '@/components/Review/mediaMetadataSearchRequest'
import {
  resolveKindsForTab
} from '@/components/Review/mediaKinds'
import type { MediaResultItem } from '@/components/Media/types'
import {
  getImmediateCachedMediaTypes,
  isMediaTypesCacheFresh,
  MEDIA_TYPES_CACHE_KEY,
  MEDIA_TYPES_CACHE_TTL_MS,
  normalizeMediaTypesCacheRecord,
  seedMediaTypesCache
} from '@/components/Review/mediaTypeCache'

const MEDIA_KEYWORD_ENDPOINT_RETRY_COOLDOWN_MS = 30_000
const EMPTY_MEDIA_RESULTS: MediaResultItem[] = []

export const deriveMediaMeta = (m: any): {
  type: string
  created_at?: string
  status?: any
  source?: string | null
  duration?: number | null
  author?: string | null
  published_at?: string | null
  transcription_model?: string | null
  word_count?: number | null
  page_count?: number | null
} => {
  const rawType = m?.type ?? m?.media_type ?? ''
  const type = typeof rawType === 'string' ? rawType.toLowerCase().trim() : ''
  const status =
    m?.status ??
    m?.ingest_status ??
    m?.ingestStatus ??
    m?.processing_state ??
    m?.processingStatus

  let source: string | null = null
  const rawSource =
    (m?.source as string | null | undefined) ??
    (m?.origin as string | null | undefined) ??
    (m?.provider as string | null | undefined)
  if (typeof rawSource === 'string' && rawSource.trim().length > 0) {
    source = rawSource.trim()
  } else if (m?.url) {
    try {
      const u = new URL(String(m.url))
      const host = u.hostname.replace(/^www\./i, '')
      if (/youtube\.com|youtu\.be/i.test(host)) {
        source = 'YouTube'
      } else if (/vimeo\.com/i.test(host)) {
        source = 'Vimeo'
      } else if (/soundcloud\.com/i.test(host)) {
        source = 'SoundCloud'
      } else {
        source = host
      }
    } catch {
      // ignore URL parse errors
    }
  }

  let duration: number | null = null
  const rawDuration =
    (m?.duration as number | string | null | undefined) ??
    (m?.media_duration as number | string | null | undefined) ??
    (m?.length_seconds as number | string | null | undefined) ??
    (m?.duration_seconds as number | string | null | undefined)
  if (typeof rawDuration === 'number') {
    duration = rawDuration
  } else if (typeof rawDuration === 'string') {
    const n = Number(rawDuration)
    if (!Number.isNaN(n)) {
      duration = n
    }
  }

  const rawAuthor =
    m?.author ??
    m?.authors ??
    m?.metadata?.author ??
    m?.metadata?.authors ??
    m?.safe_metadata?.author ??
    m?.safe_metadata?.authors ??
    m?.metadata?.creator ??
    m?.safe_metadata?.creator
  const author = typeof rawAuthor === 'string' && rawAuthor.trim().length > 0
    ? rawAuthor.trim()
    : Array.isArray(rawAuthor) && rawAuthor.length > 0
      ? rawAuthor.filter((a: any) => typeof a === 'string' && a.trim()).join(', ')
      : null

  const rawPublished =
    m?.published_at ??
    m?.publication_date ??
    m?.metadata?.publication_date ??
    m?.metadata?.published_at ??
    m?.metadata?.date ??
    m?.safe_metadata?.publication_date ??
    m?.safe_metadata?.published_at ??
    m?.safe_metadata?.date ??
    m?.metadata?.publish_date ??
    m?.safe_metadata?.publish_date
  const published_at = typeof rawPublished === 'string' && rawPublished.trim().length > 0
    ? rawPublished.trim()
    : null

  const rawTranscriptionModel =
    m?.transcription_model ??
    m?.metadata?.transcription_model ??
    m?.safe_metadata?.transcription_model ??
    m?.processing?.transcription_model
  const transcription_model = typeof rawTranscriptionModel === 'string' && rawTranscriptionModel.trim().length > 0
    ? rawTranscriptionModel.trim()
    : null

  const rawWordCount =
    m?.word_count ??
    m?.metadata?.word_count ??
    m?.safe_metadata?.word_count ??
    m?.content_length
  const word_count = typeof rawWordCount === 'number' && rawWordCount > 0 ? rawWordCount : null

  const rawPageCount =
    m?.metadata?.page_count ??
    m?.metadata?.num_pages ??
    m?.safe_metadata?.page_count ??
    m?.safe_metadata?.num_pages ??
    m?.page_count
  const page_count = typeof rawPageCount === 'number' && rawPageCount > 0 ? Math.trunc(rawPageCount) : null

  return {
    type,
    created_at: m?.created_at,
    status,
    source,
    duration,
    author,
    published_at,
    transcription_model,
    word_count,
    page_count
  }
}

export const extractKeywordsFromMedia = (m: any): string[] => {
  const possibleKeywordFields = [
    m?.metadata?.keywords,
    m?.keywords,
    m?.tags,
    m?.metadata?.tags,
    m?.processing?.keywords
  ]

  for (const field of possibleKeywordFields) {
    if (field && Array.isArray(field) && field.length > 0) {
      const keywords = field
        .map((k: any) => {
          if (typeof k === 'string') return k
          if (k && typeof k === 'object' && k.keyword) return k.keyword
          if (k && typeof k === 'object' && k.text) return k.text
          if (k && typeof k === 'object' && k.tag) return k.tag
          if (k && typeof k === 'object' && k.name) return k.name
          return null
        })
        .filter((k): k is string => k !== null && k.trim().length > 0)

      if (keywords.length > 0) return keywords
    }
  }
  return []
}

export const isMediaEndpointMissingError = (error: unknown): boolean => {
  const statusCode = getErrorStatusCode(error)
  if (statusCode !== 404 && statusCode !== 405 && statusCode !== 410) {
    return false
  }
  const message = error instanceof Error ? error.message : String(error || '')
  return /\/api\/v1\/media(?:\/|\?|$)/i.test(message)
}

export const getErrorStatusCode = (error: unknown): number | null => {
  if (!error || typeof error !== 'object') return null
  const candidate = error as Record<string, unknown>
  const rawStatus =
    candidate.status ??
    (candidate.response &&
    typeof candidate.response === 'object' &&
    (candidate.response as Record<string, unknown>).status != null
      ? (candidate.response as Record<string, unknown>).status
      : null) ??
    candidate.statusCode
  const parsed = Number(rawStatus)
  return Number.isFinite(parsed) ? parsed : null
}

export interface UseMediaSearchDeps {
  t: (key: string, opts?: Record<string, any>) => string
  message: { error: (msg: string) => void; warning: (msg: string) => void }
}

export function useMediaSearch(deps: UseMediaSearchDeps) {
  const { t, message } = deps

  const [searchMode, setSearchMode] = useState<MediaSearchMode>('full_text')
  const [query, setQuery] = useState<string>('')
  const debouncedQuery = useDebounce(query, 300)
  const [kinds, setKinds] = useState<{ media: boolean; notes: boolean }>({
    media: true,
    notes: false
  })
  const [page, setPage] = useState<number>(1)
  const [pageSize, setPageSize] = useState<number>(20)
  const [mediaTotal, setMediaTotal] = useState<number>(0)
  const [notesTotal, setNotesTotal] = useState<number>(0)
  const [combinedTotal, setCombinedTotal] = useState<number>(0)
  const [mediaTypes, setMediaTypes] = useState<string[]>([])
  const [availableMediaTypes, setAvailableMediaTypes] = useState<string[]>([])
  const [keywordTokens, setKeywordTokens] = useState<string[]>([])
  const [excludeKeywordTokens, setExcludeKeywordTokens] = useState<string[]>([])
  const [sortBy, setSortBy] = useState<MediaSortBy>('relevance')
  const [dateRange, setDateRange] = useState<MediaDateRange>({
    startDate: null,
    endDate: null
  })
  const [exactPhrase, setExactPhrase] = useState<string>('')
  const [searchFields, setSearchFields] = useState<MediaSearchField[]>([
    ...DEFAULT_MEDIA_SEARCH_FIELDS
  ])
  const [enableBoostFields, setEnableBoostFields] = useState(false)
  const [boostFields, setBoostFields] = useState<MediaBoostFields>({
    title: 2,
    content: 1
  })
  const [metadataFilters, setMetadataFilters] = useState<MetadataSearchFilter[]>([
    createMetadataSearchFilter()
  ])
  const [metadataMatchMode, setMetadataMatchMode] =
    useState<MetadataMatchMode>('all')
  const [metadataValidationError, setMetadataValidationError] = useState<string | null>(
    null
  )
  const [keywordOptions, setKeywordOptions] = useState<string[]>([])
  const [keywordSourceMode, setKeywordSourceMode] = useState<'endpoint' | 'results'>('results')
  const [mediaApiUnavailable, setMediaApiUnavailable] = useState(false)
  const [searchCollapsed, setSearchCollapsed] = useState(false)

  const searchInputRef = useRef<HTMLInputElement | null>(null)
  const hasRunInitialSearch = useRef(false)
  const previousSearchCriteriaKeyRef = useRef<string | null>(null)
  const previousPageSizeRef = useRef<number>(pageSize)
  const keywordEndpointUnavailableRef = useRef(false)
  const keywordEndpointRetryAtRef = useRef(0)
  const keywordEndpointUnsupportedRef = useRef(false)
  const keywordEndpointInFlightRef = useRef<Promise<any[] | null> | null>(null)
  const mediaApiUnavailableNotifiedRef = useRef(false)

  const markMediaApiUnavailable = useCallback((error?: unknown) => {
    if (mediaApiUnavailableNotifiedRef.current) return
    if (error && !isMediaEndpointMissingError(error)) return
    mediaApiUnavailableNotifiedRef.current = true
    setMediaApiUnavailable(true)
    setMediaTotal(0)
    message.warning(
      t('review:mediaPage.mediaApiUnavailable', {
        defaultValue:
          'Media list/search endpoints are unavailable on this server. Media loading has been paused.'
      })
    )
  }, [message, t])

  const runSearch = useCallback(async (): Promise<MediaResultItem[]> => {
    const results: MediaResultItem[] = []
    const hasTextQuery = query.trim().length > 0
    const hasQuery =
      hasTextQuery ||
      (searchMode === 'full_text' && exactPhrase.trim().length > 0)
    const hasMediaFilters = hasMediaSearchFilters({
      mediaTypes,
      includeKeywords: keywordTokens,
      excludeKeywords: excludeKeywordTokens,
      sortBy,
      dateRange,
      exactPhrase,
      fields: searchFields,
      boostFields: enableBoostFields ? boostFields : undefined
    })
    let actualMediaCount = 0
    let actualNotesCount = 0

    if (kinds.media && !mediaApiUnavailable) {
      try {
        if (searchMode === 'metadata') {
          const normalizedFilters = normalizeMetadataSearchFilters(metadataFilters)
          const validationError = validateMetadataSearchFilters(normalizedFilters)
          if (validationError) {
            setMetadataValidationError(validationError)
            setMediaTotal(0)
            actualMediaCount = 0
          } else {
            setMetadataValidationError(null)
            const path = buildMetadataSearchPath({
              filters: normalizedFilters,
              matchMode: metadataMatchMode,
              page,
              perPage: pageSize,
              textQuery: query,
              mediaTypes,
              includeKeywords: keywordTokens,
              excludeKeywords: excludeKeywordTokens,
              dateRange,
              sortBy
            })
            const metadataResp = await bgRequest<any>({
              path: path as any,
              method: 'GET' as any
            })
            const rows = Array.isArray(metadataResp?.results)
              ? metadataResp.results
              : []

            const metadataKeys = [
              'doi',
              'pmid',
              'pmcid',
              'arxiv_id',
              's2_paper_id',
              'journal',
              'license'
            ]

            for (const row of rows) {
              const id = row?.media_id ?? row?.id ?? row?.pk ?? row?.uuid
              const type =
                typeof row?.type === 'string'
                  ? row.type.toLowerCase().trim()
                  : 'document'
              if (type && !availableMediaTypes.includes(type)) {
                setAvailableMediaTypes((prev) =>
                  prev.includes(type) ? prev : [...prev, type]
                )
              }

              const safeMetadata =
                row?.safe_metadata && typeof row.safe_metadata === 'object'
                  ? row.safe_metadata
                  : {}
              const snippet = metadataKeys
                .map((key) => {
                  const value = safeMetadata?.[key]
                  if (value == null || String(value).trim().length === 0) {
                    return null
                  }
                  return `${key}: ${String(value)}`
                })
                .filter((value): value is string => Boolean(value))
                .join(' • ')

              results.push({
                kind: 'media',
                id,
                title: row?.title || `Media ${id}`,
                snippet,
                keywords: [],
                meta: {
                  type,
                  created_at: row?.created_at
                },
                raw: row
              })
            }

            const serverTotal = Number(metadataResp?.pagination?.total || rows.length || 0)
            setMediaTotal(serverTotal)
            actualMediaCount = serverTotal
          }
        } else if (!hasQuery && !hasMediaFilters) {
          const listing = await bgRequest<any>({
            path: `/api/v1/media/?page=${page}&results_per_page=${pageSize}&include_keywords=true` as any,
            method: 'GET' as any
          })
          const items = Array.isArray(listing?.items) ? listing.items : []
          const pagination = listing?.pagination
          const mediaServerTotal = Number(pagination?.total_items || items.length || 0)
          setMediaTotal(mediaServerTotal)
          actualMediaCount = mediaServerTotal
          for (const m of items) {
            const id = m?.id ?? m?.media_id ?? m?.pk ?? m?.uuid
            const meta = deriveMediaMeta(m)
            const type = meta.type
            if (type && !availableMediaTypes.includes(type)) {
              setAvailableMediaTypes((prev) =>
                prev.includes(type) ? prev : [...prev, type]
              )
            }
            const keywords = extractKeywordsFromMedia(m)

            results.push({
              kind: 'media',
              id,
              title: m?.title || m?.filename || `Media ${id}`,
              snippet: m?.snippet || m?.summary || '',
              keywords,
              meta: meta,
              raw: m
            })
          }
        } else {
          const body = buildMediaSearchPayload({
            query,
            mediaTypes,
            includeKeywords: keywordTokens,
            excludeKeywords: excludeKeywordTokens,
            sortBy,
            dateRange,
            exactPhrase,
            fields: searchFields,
            boostFields: enableBoostFields ? boostFields : undefined
          })
          const mediaResp = await bgRequest<any>({
            path: `/api/v1/media/search?page=${page}&results_per_page=${pageSize}&include_keywords=true` as any,
            method: 'POST' as any,
            headers: { 'Content-Type': 'application/json' },
            body
          })
          const items = Array.isArray(mediaResp?.items)
            ? mediaResp.items
            : Array.isArray(mediaResp?.results)
              ? mediaResp.results
              : []
          const pagination = mediaResp?.pagination
          const mediaServerTotal = Number(pagination?.total_items || items.length || 0)
          setMediaTotal(mediaServerTotal)
          actualMediaCount = mediaServerTotal
          for (const m of items) {
            const id = m?.id ?? m?.media_id ?? m?.pk ?? m?.uuid
            const meta = deriveMediaMeta(m)
            const type = meta.type
            if (type && !availableMediaTypes.includes(type)) {
              setAvailableMediaTypes((prev) =>
                prev.includes(type) ? prev : [...prev, type]
              )
            }
            const keywords = extractKeywordsFromMedia(m)

            results.push({
              kind: 'media',
              id,
              title: m?.title || m?.filename || `Media ${id}`,
              snippet: m?.snippet || m?.summary || '',
              keywords,
              meta: meta,
              raw: m
            })
          }
        }
      } catch (err) {
        if (isMediaEndpointMissingError(err)) {
          markMediaApiUnavailable(err)
          actualMediaCount = 0
          setMediaTotal(0)
        } else {
          console.error('Media search error:', err)
          message.error(t('review:mediaPage.searchError', { defaultValue: 'Failed to search media' }))
        }
      }
    } else if (kinds.media) {
      setMediaTotal(0)
      actualMediaCount = 0
    }

    // Fetch notes if enabled
    if (kinds.notes && searchMode !== 'metadata') {
      try {
        const extractNoteKeywords = (note: any): string[] => {
          const possibleFields = [
            note?.metadata?.keywords,
            note?.keywords,
            note?.tags
          ]
          for (const field of possibleFields) {
            if (field && Array.isArray(field) && field.length > 0) {
              return field
                .map((k: any) => {
                  if (typeof k === 'string') return k
                  if (k && typeof k === 'object' && k.keyword) return k.keyword
                  if (k && typeof k === 'object' && k.text) return k.text
                  return null
                })
                .filter((k): k is string => k !== null && k.trim().length > 0)
            }
          }
          return []
        }

        if (hasTextQuery) {
          const keywordFilterActive = keywordTokens.length > 0
          let notesResp: any
          let usedKeywordServerFilter = false

          try {
            const body: any = { query }
            if (keywordFilterActive) {
              body.must_have = keywordTokens
              usedKeywordServerFilter = true
            }
            notesResp = await bgRequest<any>({
              path: `/api/v1/notes/search/?page=${page}&results_per_page=${pageSize}&include_keywords=true` as any,
              method: 'POST' as any,
              headers: { 'Content-Type': 'application/json' },
              body
            })
          } catch {
            usedKeywordServerFilter = false
            notesResp = await bgRequest<any>({
              path: `/api/v1/notes/search/?query=${encodeURIComponent(
                query
              )}&page=${page}&results_per_page=${pageSize}&include_keywords=true` as any,
              method: 'GET' as any
            })
          }

          const items = Array.isArray(notesResp) ? notesResp : (notesResp?.items || [])
          const pagination = notesResp?.pagination

          let filteredItems = items
          if (keywordFilterActive && !usedKeywordServerFilter) {
            filteredItems = items.filter((n: any) => {
              const noteKws = extractNoteKeywords(n)
              return keywordTokens.some((kw) =>
                noteKws.some((nkw) => nkw.toLowerCase().includes(kw.toLowerCase()))
              )
            })
          }

          if (keywordFilterActive && !usedKeywordServerFilter) {
            const notesClientTotal = filteredItems.length
            setNotesTotal(notesClientTotal)
            actualNotesCount = notesClientTotal
          } else {
            const notesServerTotal = Number(pagination?.total_items || items.length || 0)
            setNotesTotal(notesServerTotal)
            actualNotesCount = notesServerTotal
          }

          for (const n of filteredItems) {
            const id = n?.id ?? n?.note_id ?? n?.pk ?? n?.uuid
            results.push({
              kind: 'note',
              id,
              title: n?.title || `Note ${id}`,
              snippet: n?.content?.substring(0, 200) || '',
              keywords: extractNoteKeywords(n),
              meta: {
                type: 'note',
                source: n?.metadata?.conversation_id ? 'conversation' : null
              },
              raw: n
            })
          }
        } else {
          const notesResp = await bgRequest<any>({
            path: `/api/v1/notes/?page=${page}&results_per_page=${pageSize}` as any,
            method: 'GET' as any
          })
          const items = Array.isArray(notesResp?.items) ? notesResp.items : []
          const pagination = notesResp?.pagination
          const notesServerTotal = Number(pagination?.total_items || items.length || 0)
          setNotesTotal(notesServerTotal)
          actualNotesCount = notesServerTotal

          for (const n of items) {
            const id = n?.id ?? n?.note_id ?? n?.pk ?? n?.uuid
            results.push({
              kind: 'note',
              id,
              title: n?.title || `Note ${id}`,
              snippet: n?.content?.substring(0, 200) || '',
              keywords: extractNoteKeywords(n),
              meta: {
                type: 'note',
                source: n?.metadata?.conversation_id ? 'conversation' : null
              },
              raw: n
            })
          }
        }
      } catch (err) {
        console.error('Notes search error:', err)
        message.error(t('review:mediaPage.notesSearchError', { defaultValue: 'Failed to search notes' }))
      }
    }

    const finalCombinedTotal = actualMediaCount + actualNotesCount
    setCombinedTotal(finalCombinedTotal)

    return results
  }, [
    searchMode,
    query,
    kinds,
    mediaTypes,
    keywordTokens,
    excludeKeywordTokens,
    sortBy,
    dateRange.startDate,
    dateRange.endDate,
    exactPhrase,
    searchFields,
    enableBoostFields,
    boostFields.title,
    boostFields.content,
    metadataMatchMode,
    metadataFilters,
    mediaApiUnavailable,
    markMediaApiUnavailable,
    message,
    page,
    pageSize,
    availableMediaTypes,
    t
  ])

  const { data: queryResults, refetch, isLoading, isFetching } = useQuery({
    queryKey: [
      'media-search',
      query,
      kinds,
      mediaTypes,
      keywordTokens.join('|'),
      excludeKeywordTokens.join('|'),
      sortBy,
      dateRange.startDate,
      dateRange.endDate,
      exactPhrase,
      searchFields.join('|'),
      enableBoostFields,
      boostFields.title,
      boostFields.content,
      searchMode,
      metadataMatchMode,
      JSON.stringify(metadataFilters),
      page,
      pageSize
    ],
    queryFn: runSearch,
    enabled: false
  })
  const results = queryResults ?? EMPTY_MEDIA_RESULTS

  const normalizedMetadataFilters = useMemo(
    () => normalizeMetadataSearchFilters(metadataFilters),
    [metadataFilters]
  )
  const searchCriteriaKey = useMemo(
    () =>
      JSON.stringify({
        debouncedQuery,
        kinds,
        searchMode,
        mediaTypes,
        keywordTokens,
        excludeKeywordTokens,
        sortBy,
        dateStart: dateRange.startDate,
        dateEnd: dateRange.endDate,
        exactPhrase: exactPhrase.trim(),
        searchFields,
        enableBoostFields,
        boostTitle: boostFields.title,
        boostContent: boostFields.content,
        metadataMatchMode,
        metadataFilters: normalizedMetadataFilters.map((filter) => ({
          field: filter.field,
          op: filter.op,
          value: filter.value
        }))
      }),
    [
      boostFields.content,
      boostFields.title,
      dateRange.endDate,
      dateRange.startDate,
      debouncedQuery,
      enableBoostFields,
      exactPhrase,
      excludeKeywordTokens,
      keywordTokens,
      kinds,
      mediaTypes,
      metadataMatchMode,
      normalizedMetadataFilters,
      searchFields,
      searchMode,
      sortBy
    ]
  )

  const hasActiveFilters =
    mediaTypes.length > 0 ||
    keywordTokens.length > 0 ||
    excludeKeywordTokens.length > 0 ||
    Boolean(dateRange.startDate || dateRange.endDate) ||
    sortBy !== 'relevance' ||
    Boolean(exactPhrase.trim()) ||
    !hasDefaultMediaSearchFields(searchFields) ||
    enableBoostFields ||
    searchMode === 'metadata' ||
    normalizedMetadataFilters.length > 0

  const activeFilterCount = useMemo(() => {
    return (
      mediaTypes.length +
      keywordTokens.length +
      excludeKeywordTokens.length +
      Number(Boolean(dateRange.startDate || dateRange.endDate)) +
      Number(sortBy !== 'relevance') +
      Number(Boolean(exactPhrase.trim())) +
      Number(!hasDefaultMediaSearchFields(searchFields)) +
      Number(enableBoostFields) +
      Number(searchMode === 'metadata') +
      Number(normalizedMetadataFilters.length > 0)
    )
  }, [
    dateRange.endDate,
    dateRange.startDate,
    enableBoostFields,
    exactPhrase,
    excludeKeywordTokens.length,
    keywordTokens.length,
    mediaTypes.length,
    normalizedMetadataFilters.length,
    searchFields,
    searchMode,
    sortBy
  ])

  const resetAllFilters = useCallback(() => {
    setSearchMode('full_text')
    setMediaTypes([])
    setKeywordTokens([])
    setExcludeKeywordTokens([])
    setDateRange({ startDate: null, endDate: null })
    setSortBy('relevance')
    setExactPhrase('')
    setSearchFields([...DEFAULT_MEDIA_SEARCH_FIELDS])
    setEnableBoostFields(false)
    setBoostFields({ title: 2, content: 1 })
    setMetadataFilters([createMetadataSearchFilter()])
    setMetadataMatchMode('all')
    setMetadataValidationError(null)
    setPage(1)
  }, [])

  useEffect(() => {
    if (searchMode !== 'metadata') {
      setMetadataValidationError(null)
      return
    }
    setKinds((prev) => (prev.media && !prev.notes ? prev : { media: true, notes: false }))
    setNotesTotal(0)
  }, [searchMode])

  const activeTotalCount =
    kinds.media && kinds.notes
      ? combinedTotal
      : kinds.notes
        ? notesTotal
        : mediaTotal
  const totalPages = Math.ceil(activeTotalCount / pageSize)

  // Coordinate all search refetches
  useEffect(() => {
    const previousCriteriaKey = previousSearchCriteriaKeyRef.current
    const isInitialRun = previousCriteriaKey === null
    const criteriaChanged = !isInitialRun && previousCriteriaKey !== searchCriteriaKey
    const pageSizeChanged = previousPageSizeRef.current !== pageSize

    if (isInitialRun) {
      hasRunInitialSearch.current = true
      previousSearchCriteriaKeyRef.current = searchCriteriaKey
      previousPageSizeRef.current = pageSize
      refetch()
      return
    }

    if (criteriaChanged || pageSizeChanged) {
      previousSearchCriteriaKeyRef.current = searchCriteriaKey
      previousPageSizeRef.current = pageSize
      if (page !== 1) {
        setPage(1)
        return
      }
      refetch()
      return
    }

    refetch()
  }, [
    searchCriteriaKey,
    page,
    pageSize,
    refetch
  ])

  // Initial load: populate media types
  useEffect(() => {
    const immediateCachedTypes = getImmediateCachedMediaTypes()
    if (immediateCachedTypes.length > 0) {
      setAvailableMediaTypes((prev) =>
        Array.from(new Set<string>([...prev, ...immediateCachedTypes])) as string[]
      )
    }

    ;(async () => {
      try {
        const storage = new Storage({ area: 'local', serde: safeStorageSerde } as any)
        const cached = normalizeMediaTypesCacheRecord(
          await storage.get(MEDIA_TYPES_CACHE_KEY).catch(() => null)
        )
        const now = Date.now()
        if (cached && isMediaTypesCacheFresh(cached.cachedAt, now, MEDIA_TYPES_CACHE_TTL_MS)) {
          setAvailableMediaTypes(
            Array.from(new Set<string>(cached.types)) as string[]
          )
          seedMediaTypesCache(cached.types, { cachedAt: cached.cachedAt })
        }

        const first = await bgRequest<any>({
          path: `/api/v1/media/?page=1&results_per_page=50` as any,
          method: 'GET' as any
        })
        const totalPgs = Math.max(
          1,
          Number(first?.pagination?.total_pages || 1)
        )
        const pagesToFetch = [1, 2, 3].filter((p) => p <= totalPgs)
        const listings = await Promise.all(
          pagesToFetch.map((p) =>
            p === 1
              ? Promise.resolve(first)
              : bgRequest<any>({
                  path: `/api/v1/media/?page=${p}&results_per_page=50` as any,
                  method: 'GET' as any
                })
          )
        )
        const typeSet = new Set<string>()
        for (const listing of listings) {
          const items = Array.isArray(listing?.items) ? listing.items : []
          for (const m of items) {
            const tp = deriveMediaMeta(m).type
            if (tp) typeSet.add(tp)
          }
        }
        const newTypes = Array.from(typeSet)
        if (newTypes.length) {
          setAvailableMediaTypes((prev) =>
            Array.from(new Set<string>([...prev, ...newTypes])) as string[]
          )
          const cacheRecord = seedMediaTypesCache(newTypes, { cachedAt: now })
          if (cacheRecord) {
            await storage.set(MEDIA_TYPES_CACHE_KEY, cacheRecord)
          }
        }
      } catch (error) {
        if (isMediaEndpointMissingError(error)) {
          mediaApiUnavailableNotifiedRef.current = true
          setMediaApiUnavailable(true)
          setMediaTotal(0)
        }
      }

      try {
        await refetch()
      } catch {}
    })()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Load keyword suggestions
  const loadKeywordSuggestions = useCallback(async (searchText?: string) => {
    const normalizeKeywords = (items: any[]): string[] => {
      const out = new Set<string>()
      for (const item of items) {
        const raw =
          typeof item === 'string'
            ? item
            : item?.keyword ?? item?.text ?? item?.tag ?? item?.name

        if (typeof raw !== 'string') continue
        const trimmed = raw.trim()
        if (!trimmed) continue
        if (searchText && !trimmed.toLowerCase().includes(searchText.toLowerCase())) {
          continue
        }
        out.add(trimmed)
      }
      return Array.from(out)
    }

    if (mediaApiUnavailable) {
      const keywordsFromResults = new Set<string>()
      for (const result of results) {
        if (!result.keywords) continue
        for (const kw of result.keywords) {
          if (!searchText || kw.toLowerCase().includes(searchText.toLowerCase())) {
            keywordsFromResults.add(kw)
          }
        }
      }
      setKeywordOptions(Array.from(keywordsFromResults))
      setKeywordSourceMode('results')
      return
    }

    const trimmedSearch = searchText?.trim()

    const applyKeywordResultsFallback = () => {
      const keywordsFromResults = new Set<string>()
      for (const result of results) {
        if (!result.keywords) continue
        for (const kw of result.keywords) {
          if (!searchText || kw.toLowerCase().includes(searchText.toLowerCase())) {
            keywordsFromResults.add(kw)
          }
        }
      }
      setKeywordOptions(Array.from(keywordsFromResults))
      setKeywordSourceMode('results')
    }

    if (!trimmedSearch) {
      applyKeywordResultsFallback()
      return
    }

    if (keywordEndpointUnsupportedRef.current) {
      applyKeywordResultsFallback()
      return
    }

    if (keywordEndpointInFlightRef.current) {
      try {
        const pendingItems = await keywordEndpointInFlightRef.current
        if (pendingItems) {
          setKeywordOptions(normalizeKeywords(pendingItems))
          setKeywordSourceMode('endpoint')
          return
        }
      } catch {
        applyKeywordResultsFallback()
        return
      }
    }

    const now = Date.now()
    if (
      !keywordEndpointUnavailableRef.current ||
      now >= keywordEndpointRetryAtRef.current
    ) {
      try {
        const endpointRequest = (async () => {
          const endpointPath = `/api/v1/media/keywords?query=${encodeURIComponent(trimmedSearch)}`
          const keywordResp = await bgRequest<any>({
            path: endpointPath as any,
            method: 'GET' as any
          })
          const endpointItems = Array.isArray(keywordResp)
            ? keywordResp
            : Array.isArray(keywordResp?.keywords)
              ? keywordResp.keywords
              : Array.isArray(keywordResp?.items)
                ? keywordResp.items
                : null

          if (!endpointItems) {
            throw new Error('Unexpected keyword endpoint response')
          }

          return endpointItems
        })()
        keywordEndpointInFlightRef.current = endpointRequest
        const endpointItems = await endpointRequest

        setKeywordOptions(normalizeKeywords(endpointItems))
        setKeywordSourceMode('endpoint')
        keywordEndpointUnsupportedRef.current = false
        keywordEndpointUnavailableRef.current = false
        keywordEndpointRetryAtRef.current = 0
        return
      } catch (error) {
        if (isMediaEndpointMissingError(error)) {
          keywordEndpointUnsupportedRef.current = true
          keywordEndpointUnavailableRef.current = true
          keywordEndpointRetryAtRef.current = Number.POSITIVE_INFINITY
        } else {
          keywordEndpointUnavailableRef.current = true
          keywordEndpointRetryAtRef.current =
            Date.now() + MEDIA_KEYWORD_ENDPOINT_RETRY_COOLDOWN_MS
        }
      } finally {
        keywordEndpointInFlightRef.current = null
      }
    }

    applyKeywordResultsFallback()
  }, [mediaApiUnavailable, results])

  // Keep keyword suggestions in sync with results
  useEffect(() => {
    loadKeywordSuggestions()
  }, [loadKeywordSuggestions, results])

  const handleKindChange = useCallback((nextKind: 'media' | 'notes') => {
    if (searchMode === 'metadata' && nextKind === 'notes') {
      return
    }
    setKinds((prev) => resolveKindsForTab(prev, nextKind))
    setPage(1)
  }, [searchMode])

  const handleSearch = useCallback(() => {
    setPage(1)
    refetch()
  }, [refetch])

  return {
    // State
    searchMode, setSearchMode,
    query, setQuery,
    debouncedQuery,
    kinds, setKinds,
    page, setPage,
    pageSize, setPageSize,
    mediaTotal,
    notesTotal,
    combinedTotal,
    mediaTypes, setMediaTypes,
    availableMediaTypes,
    keywordTokens, setKeywordTokens,
    excludeKeywordTokens, setExcludeKeywordTokens,
    sortBy, setSortBy,
    dateRange, setDateRange,
    exactPhrase, setExactPhrase,
    searchFields, setSearchFields,
    enableBoostFields, setEnableBoostFields,
    boostFields, setBoostFields,
    metadataFilters, setMetadataFilters,
    metadataMatchMode, setMetadataMatchMode,
    metadataValidationError,
    keywordOptions,
    keywordSourceMode,
    mediaApiUnavailable,
    searchCollapsed, setSearchCollapsed,
    searchInputRef,
    hasRunInitialSearch,
    // Computed
    normalizedMetadataFilters,
    searchCriteriaKey,
    hasActiveFilters,
    activeFilterCount,
    activeTotalCount,
    totalPages,
    // Query results
    results,
    refetch,
    isLoading,
    isFetching,
    // Callbacks
    markMediaApiUnavailable,
    resetAllFilters,
    loadKeywordSuggestions,
    handleKindChange,
    handleSearch,
  }
}
