import React from 'react'
import type { MessageInstance } from 'antd/es/message/interface'
import type { QueryClient } from '@tanstack/react-query'
import { useQuery } from '@tanstack/react-query'
import { bgRequest } from '@/services/background-proxy'
import { CAPTURED_NOTE_KEYWORD } from '@/services/note-capture'
import { getSetting, setSetting } from '@/services/settings/registry'
import {
  NOTES_PAGE_SIZE_SETTING,
  NOTES_NOTEBOOKS_SETTING,
} from '@/services/settings/ui-settings'
import type { NoteListItem } from '@/components/Notes/notes-manager-types'
import type {
  NotesSortOption,
  NotesListViewMode,
  MoodboardSummary,
  NotebookFilterOption,
} from '../notes-manager-types'
import {
  extractBacklink,
  extractKeywords,
  toNoteVersion,
  NOTE_SORT_API_PARAMS,
  sortNoteRows,
  normalizeNotebookKeywords,
  normalizeNotebookName,
  normalizeNotebookOptions,
  NOTEBOOK_COLLECTION_PAGE_SIZE,
  NOTEBOOK_COLLECTION_MAX_PAGES,
  normalizeNotebookCollectionFromServer,
  normalizeNotebookCollectionsResponse,
  buildNotebookDefaultName,
  NOTE_SEARCH_DEBOUNCE_MS,
  promptModal,
} from '../notes-manager-utils'
import type { ConfirmDangerOptions } from '@/components/Common/confirm-danger'

type ConfirmDanger = (options: ConfirmDangerOptions) => Promise<boolean>
type NotesAuthorityScope = string | null | undefined
type NotesAuthorityOwner = {
  scope: NotesAuthorityScope
  generation: number
}

const applyStateAction = <T,>(
  action: React.SetStateAction<T>,
  current: T
): T =>
  typeof action === 'function'
    ? (action as (value: T) => T)(current)
    : action

const useAuthorityOwnedState = <T,>(
  authorityOwner: NotesAuthorityOwner,
  isCurrentAuthority: (owner: NotesAuthorityOwner) => boolean,
  emptyValue: T
): [T, React.Dispatch<React.SetStateAction<T>>] => {
  const emptyValueRef = React.useRef(emptyValue)
  const [state, setState] = React.useState({
    authorityOwner,
    value: emptyValue
  })
  const value =
    authorityOwner.scope !== null && state.authorityOwner === authorityOwner
      ? state.value
      : emptyValueRef.current
  const setValue = React.useCallback<
    React.Dispatch<React.SetStateAction<T>>
  >(
    (nextValue) => {
      if (!isCurrentAuthority(authorityOwner)) return
      setState((current) => {
        if (!isCurrentAuthority(authorityOwner)) return current
        return {
          authorityOwner,
          value: applyStateAction(
            nextValue,
            current.authorityOwner === authorityOwner
              ? current.value
              : emptyValueRef.current
          )
        }
      })
    },
    [authorityOwner, isCurrentAuthority]
  )
  return [value, setValue]
}

export interface UseNotesListManagementDeps {
  authorityScope?: string | null
  isOnline: boolean
  message: MessageInstance
  confirmDanger: ConfirmDanger
  queryClient: QueryClient
  t: (key: string, opts?: Record<string, any>) => string
  /** Keyword tokens from the keyword hook */
  keywordTokens: string[]
  setKeywordTokens: React.Dispatch<React.SetStateAction<string[]>>
  /** Notebook keyword tokens computed from selected notebook */
  notebookKeywordTokens: string[]
}

export function useNotesListManagement(deps: UseNotesListManagementDeps) {
  const {
    authorityScope,
    isOnline,
    message,
    confirmDanger,
    queryClient,
    t,
    keywordTokens,
    setKeywordTokens,
    notebookKeywordTokens,
  } = deps

  // ---- list state ----
  const [query, setQuery] = React.useState('')
  const [queryInput, setQueryInput] = React.useState('')
  const [searchTipsQuery, setSearchTipsQuery] = React.useState('')
  const [page, setPage] = React.useState(1)
  const [pageSize, setPageSize] = React.useState(20)
  const [sortOption, setSortOption] = React.useState<NotesSortOption>('modified_desc')
  const [listMode, setListMode] = React.useState<'active' | 'trash'>('active')
  const [listViewMode, setListViewMode] = React.useState<NotesListViewMode>('list')
  const listQueryViewMode = listViewMode === 'graph' ? 'list' : listViewMode
  const authorityOwnerRef = React.useRef<NotesAuthorityOwner>({
    scope: authorityScope,
    generation: 0
  })
  if (authorityOwnerRef.current.scope !== authorityScope) {
    authorityOwnerRef.current = {
      scope: authorityScope,
      generation: authorityOwnerRef.current.generation + 1
    }
  }
  const authorityOwner = authorityOwnerRef.current
  const isCurrentAuthority = React.useCallback(
    (owner: NotesAuthorityOwner) =>
      owner.scope !== null && authorityOwnerRef.current === owner,
    []
  )
  const [total, setTotal] = useAuthorityOwnedState(
    authorityOwner,
    isCurrentAuthority,
    0
  )
  const [bulkSelectedIds, setBulkSelectedIds] = useAuthorityOwnedState(
    authorityOwner,
    isCurrentAuthority,
    [] as string[]
  )
  const bulkSelectionAnchorRef = React.useRef<{
    authorityOwner: NotesAuthorityOwner
    value: string | null
  }>({ authorityOwner, value: null })
  if (bulkSelectionAnchorRef.current.authorityOwner !== authorityOwner) {
    bulkSelectionAnchorRef.current = { authorityOwner, value: null }
  }

  // ---- moodboard state ----
  const [moodboards, setMoodboards] = useAuthorityOwnedState(
    authorityOwner,
    isCurrentAuthority,
    [] as MoodboardSummary[]
  )
  const [selectedMoodboardId, setSelectedMoodboardId] = useAuthorityOwnedState(
    authorityOwner,
    isCurrentAuthority,
    null as number | null
  )

  // ---- notebook state ----
  const [notebookOptions, setNotebookOptions] = useAuthorityOwnedState(
    authorityOwner,
    isCurrentAuthority,
    [] as NotebookFilterOption[]
  )
  const [selectedNotebookId, setSelectedNotebookId] = useAuthorityOwnedState(
    authorityOwner,
    isCurrentAuthority,
    null as number | null
  )

  const searchQueryTimeoutRef = React.useRef<number | null>(null)
  const pageSizeSettingHydratedRef = React.useRef(false)
  const notebookSettingsHydratedRef = React.useRef<{
    authorityOwner: NotesAuthorityOwner
  } | null>(null)

  // ---- derived ----
  const selectedNotebook = React.useMemo(
    () =>
      selectedNotebookId == null
        ? null
        : notebookOptions.find((option) => option.id === selectedNotebookId) || null,
    [notebookOptions, selectedNotebookId]
  )

  const effectiveKeywordTokens = React.useMemo(() => {
    const effectiveNotebookKeywordTokens =
      authorityScope === undefined
        ? notebookKeywordTokens
        : selectedNotebook?.keywords ?? []
    const merged = [
      ...(listViewMode === 'inbox' ? [CAPTURED_NOTE_KEYWORD] : []),
      ...keywordTokens,
      ...effectiveNotebookKeywordTokens
    ]
    const deduped: string[] = []
    for (const token of merged) {
      const normalized = String(token || '').trim().toLowerCase()
      if (!normalized) continue
      if (deduped.includes(normalized)) continue
      deduped.push(normalized)
    }
    return deduped
  }, [authorityScope, keywordTokens, listViewMode, notebookKeywordTokens, selectedNotebook])

  const clearSearchQueryTimeout = React.useCallback(() => {
    if (searchQueryTimeoutRef.current != null) {
      window.clearTimeout(searchQueryTimeoutRef.current)
      searchQueryTimeoutRef.current = null
    }
  }, [])

  // ---- fetch notes helpers ----
  const fetchFilteredNotesRaw = React.useCallback(async (
    q: string,
    toks: string[],
    fetchPage: number,
    fetchPageSize: number
  ): Promise<{ items: any[]; total: number }> => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) {
      return { items: [], total: 0 }
    }
    const qstr = q.trim()
    if (!qstr && toks.length === 0) {
      return { items: [], total: 0 }
    }

    const params = new URLSearchParams()
    if (qstr) params.set('query', qstr)
    params.set('limit', String(fetchPageSize))
    params.set('offset', String((fetchPage - 1) * fetchPageSize))
    params.set('include_keywords', 'true')
    params.set('sort_by', NOTE_SORT_API_PARAMS[sortOption].sortBy)
    params.set('sort_order', NOTE_SORT_API_PARAMS[sortOption].sortOrder)
    toks.forEach((tok) => {
      const v = tok.trim()
      if (v.length > 0) {
        params.append('tokens', v)
      }
    })

    const abs = await bgRequest<any>({
      path: `/api/v1/notes/search/?${params.toString()}` as any,
      method: 'GET' as any
    })
    if (!isCurrentAuthority(requestOwner)) {
      return { items: [], total: 0 }
    }

    let items: any[] = []
    let totalVal = 0

    if (Array.isArray(abs)) {
      items = abs
      totalVal = abs.length
    } else if (abs && typeof abs === 'object') {
      if (Array.isArray((abs as any).items)) {
        items = (abs as any).items
      } else if (Array.isArray((abs as any).notes)) {
        items = (abs as any).notes
      } else if (Array.isArray((abs as any).results)) {
        items = (abs as any).results
      }
      const pagination = (abs as any).pagination
      const totalCandidate =
        (abs as any).total ??
        pagination?.total ??
        pagination?.total_items ??
        (abs as any).count ??
        items.length
      totalVal = Number(totalCandidate) || 0
    }

    return { items: sortNoteRows(items, sortOption), total: totalVal }
  }, [authorityOwner, isCurrentAuthority, sortOption])

  const fetchNotes = React.useCallback(async (): Promise<NoteListItem[]> => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) return []
    const mapNoteListItem = (n: any): NoteListItem => {
      const links = extractBacklink(n)
      const keywords = extractKeywords(n)
      return {
        id: n?.id,
        title: n?.title,
        content: n?.content ?? n?.content_preview,
        content_preview: n?.content_preview ?? null,
        updated_at: n?.updated_at ?? n?.last_modified ?? n?.lastModified,
        deleted: Boolean(n?.deleted),
        conversation_id: links.conversation_id,
        message_id: links.message_id,
        keywords,
        cover_image_url:
          typeof n?.cover_image_url === 'string' && n.cover_image_url.trim().length > 0
            ? n.cover_image_url
            : null,
        membership_source:
          n?.membership_source === 'manual' ||
          n?.membership_source === 'smart' ||
          n?.membership_source === 'both'
            ? n.membership_source
            : undefined,
        version: toNoteVersion(n) ?? undefined
      }
    }

    if (listMode === 'trash') {
      const params = new URLSearchParams()
      params.set('limit', String(pageSize))
      params.set('offset', String((page - 1) * pageSize))
      params.set('include_keywords', 'true')
      params.set('sort_by', NOTE_SORT_API_PARAMS[sortOption].sortBy)
      params.set('sort_order', NOTE_SORT_API_PARAMS[sortOption].sortOrder)
      const res = await bgRequest<any>({
        path: `/api/v1/notes/trash?${params.toString()}` as any,
        method: 'GET' as any
      })
      if (!isCurrentAuthority(requestOwner)) return []
      const items = Array.isArray(res?.items) ? res.items : (Array.isArray(res) ? res : [])
      const totalItems =
        Number(
          res?.total ??
            res?.pagination?.total_items ??
            res?.count ??
            items.length ??
            0
        ) || 0
      setTotal(totalItems)
      return sortNoteRows(items, sortOption).map(mapNoteListItem)
    }

    if (listViewMode === 'moodboard') {
      if (selectedMoodboardId == null) {
        setTotal(0)
        return []
      }
      const params = new URLSearchParams()
      params.set('limit', String(pageSize))
      params.set('offset', String((page - 1) * pageSize))
      const res = await bgRequest<any>({
        path: `/api/v1/notes/moodboards/${selectedMoodboardId}/notes?${params.toString()}` as any,
        method: 'GET' as any
      })
      if (!isCurrentAuthority(requestOwner)) return []
      const items = Array.isArray(res?.notes)
        ? res.notes
        : Array.isArray(res?.items)
          ? res.items
          : Array.isArray(res)
            ? res
            : []
      const totalItems =
        Number(
          res?.total ??
            res?.count ??
            res?.pagination?.total_items ??
            items.length ??
            0
        ) || 0
      setTotal(totalItems)
      return items.map(mapNoteListItem)
    }

    const q = query.trim()
    const toks = effectiveKeywordTokens.map((k) => k.toLowerCase())
    if (q || toks.length > 0) {
      const { items, total: totalVal } = await fetchFilteredNotesRaw(q, toks, page, pageSize)
      setTotal(totalVal)
      return items.map(mapNoteListItem)
    }
    const browsePath =
      (`/api/v1/notes/?page=${page}&results_per_page=${pageSize}` +
        `&sort_by=${NOTE_SORT_API_PARAMS[sortOption].sortBy}` +
        `&sort_order=${NOTE_SORT_API_PARAMS[sortOption].sortOrder}`) as `/${string}`
    const res = await bgRequest<any>({
      path: browsePath,
      method: 'GET' as any
    })
    if (!isCurrentAuthority(requestOwner)) return []
    const items = Array.isArray(res?.items) ? res.items : (Array.isArray(res) ? res : [])
    const pagination = res?.pagination
    setTotal(Number(pagination?.total_items || items.length || 0))
    return sortNoteRows(items, sortOption).map(mapNoteListItem)
  }, [authorityOwner, effectiveKeywordTokens, fetchFilteredNotesRaw, isCurrentAuthority, listMode, listViewMode, page, pageSize, query, selectedMoodboardId, setTotal, sortOption])

  const listQueryKey = [
    'notes',
    listMode,
    listQueryViewMode,
    selectedMoodboardId ?? 'none',
    query,
    page,
    pageSize,
    sortOption,
    selectedNotebookId ?? 'none',
    effectiveKeywordTokens.join('|'),
    ...(authorityScope === undefined
      ? []
      : ['authority', authorityScope, 'generation', authorityOwner.generation])
  ]
  const {
    data: queryData,
    error,
    isError,
    isFetching,
    isPlaceholderData,
    refetch: refetchQuery
  } = useQuery({
    queryKey: listQueryKey,
    queryFn: fetchNotes,
    placeholderData: (previousData, previousQuery) => {
      if (authorityScope === undefined) return previousData
      if (authorityScope === null) return undefined
      const previousKey = previousQuery?.queryKey ?? []
      return previousKey.at(-4) === 'authority' &&
        previousKey.at(-3) === authorityScope &&
        previousKey.at(-2) === 'generation' &&
        previousKey.at(-1) === authorityOwner.generation
        ? previousData
        : undefined
    },
    enabled: isOnline && authorityScope !== null
  })
  const refetch = React.useCallback(
    (options?: Parameters<typeof refetchQuery>[0]) => {
      const requestOwner = authorityOwner
      if (!isCurrentAuthority(requestOwner)) {
        return Promise.resolve(undefined)
      }
      return refetchQuery(options)
    },
    [authorityOwner, isCurrentAuthority, refetchQuery]
  )
  const data = authorityScope === null ? undefined : queryData
  const listErrorMessage = React.useMemo(() => {
    if (!isError) return null
    const messageText = String((error as any)?.message || error || '').trim()
    return messageText || t('option:notesSearch.loadErrorDescription', {
      defaultValue: 'The notes list failed to load. Retry the request or check server health.'
    })
  }, [error, isError, t])

  /** Raw notes from the query - pinning is applied at the component level */
  const rawNotes = React.useMemo(() => {
    if (!Array.isArray(data)) return []
    return data
  }, [data])

  const filteredCount = rawNotes.length
  const orderedVisibleNoteIds = React.useMemo(
    () => rawNotes.map((note) => String(note.id)),
    [rawNotes]
  )
  const bulkSelectedIdSet = React.useMemo(
    () => new Set(bulkSelectedIds),
    [bulkSelectedIds]
  )
  const selectedBulkNotes = React.useMemo(
    () => rawNotes.filter((note) => bulkSelectedIdSet.has(String(note.id))),
    [bulkSelectedIdSet, rawNotes]
  )

  // ---- moodboard pagination ----
  const moodboardTotalPages = React.useMemo(() => {
    const normalizedPageSize = Math.max(1, Number(pageSize) || 1)
    return Math.max(1, Math.ceil(Math.max(0, Number(total) || 0) / normalizedPageSize))
  }, [pageSize, total])
  const moodboardCanGoPrev = page > 1
  const moodboardCanGoNext = page < moodboardTotalPages
  const moodboardRangeStart = total <= 0 ? 0 : (page - 1) * pageSize + 1
  const moodboardRangeEnd = total <= 0 ? 0 : Math.min(total, page * pageSize)

  React.useEffect(() => {
    if (listMode !== 'active' || listViewMode !== 'moodboard') return
    if (page <= moodboardTotalPages) return
    setPage(moodboardTotalPages)
  }, [listMode, listViewMode, moodboardTotalPages, page])

  // ---- moodboard fetch ----
  const fetchMoodboards = React.useCallback(async (): Promise<MoodboardSummary[]> => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) return []
    const pageLimit = 200
    const maxPages = 50
    const collected: any[] = []
    let offset = 0

    for (let pageIndex = 0; pageIndex < maxPages; pageIndex += 1) {
      if (!isCurrentAuthority(requestOwner)) return []
      const res = await bgRequest<any>({
        path: `/api/v1/notes/moodboards?limit=${pageLimit}&offset=${offset}` as any,
        method: "GET" as any
      })
      if (!isCurrentAuthority(requestOwner)) return []
      const rows = Array.isArray(res?.moodboards)
        ? res.moodboards
        : Array.isArray(res?.items)
          ? res.items
          : Array.isArray(res)
            ? res
            : []

      if (!Array.isArray(rows) || rows.length === 0) break
      collected.push(...rows)

      const paginationTotalRaw = Number(
        res?.total ??
          res?.pagination?.total_items ??
          NaN
      )
      if (Number.isFinite(paginationTotalRaw) && collected.length >= paginationTotalRaw) break
      if (rows.length < pageLimit) break

      offset += pageLimit
    }

    const deduped = new Map<number, any>()
    for (const row of collected) {
      const id = Number(row?.id)
      if (!Number.isFinite(id)) continue
      deduped.set(Math.floor(id), row)
    }

    return Array.from(deduped.values())
      .map((item: any) => {
        const id = Number(item?.id)
        if (!Number.isFinite(id)) return null
        return {
          id: Math.floor(id),
          name: String(item?.name || "").trim() || `Collection ${id}`,
          description: item?.description ?? null,
          smart_rule: item?.smart_rule ?? null,
          version:
            typeof item?.version === "number"
              ? item.version
              : Number.isFinite(Number(item?.version))
                ? Number(item.version)
                : undefined,
          last_modified:
            typeof item?.last_modified === "string" ? item.last_modified : undefined
        } as MoodboardSummary
      })
      .filter((item): item is MoodboardSummary => item != null)
  }, [authorityOwner, isCurrentAuthority])

  const {
    data: moodboardQueryData,
    isFetching: isMoodboardsFetching,
    refetch: refetchMoodboardsQuery
  } = useQuery({
    queryKey: [
      "notes-moodboards",
      listMode,
      listViewMode,
      ...(authorityScope === undefined
        ? []
        : [
            "authority",
            authorityScope,
            "generation",
            authorityOwner.generation
          ])
    ],
    queryFn: fetchMoodboards,
    enabled:
      isOnline &&
      authorityScope !== null &&
      listMode === "active" &&
      listViewMode === "moodboard"
  })
  const moodboardData =
    authorityScope === null ? undefined : moodboardQueryData
  const refetchMoodboards = React.useCallback(() => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) {
      return Promise.resolve(undefined)
    }
    return refetchMoodboardsQuery()
  }, [authorityOwner, isCurrentAuthority, refetchMoodboardsQuery])

  React.useEffect(() => {
    const nextMoodboards = Array.isArray(moodboardData) ? moodboardData : []
    setMoodboards(nextMoodboards)
    if (nextMoodboards.length === 0) {
      setSelectedMoodboardId(null)
      return
    }
    setSelectedMoodboardId((current) => {
      if (current != null && nextMoodboards.some((item) => item.id === current)) return current
      return nextMoodboards[0].id
    })
  }, [moodboardData, setMoodboards, setSelectedMoodboardId])

  const selectedMoodboard = React.useMemo(() => {
    if (selectedMoodboardId == null) return null
    return moodboards.find((item) => item.id === selectedMoodboardId) || null
  }, [moodboards, selectedMoodboardId])

  const createMoodboard = React.useCallback(async () => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) return
    const name = await promptModal({
      title: 'Create collection',
      placeholder: 'Collection name',
      okText: 'Create',
    })
    if (!name || !isCurrentAuthority(requestOwner)) return
    try {
      const created = await bgRequest<any>({
        path: "/api/v1/notes/moodboards" as any,
        method: "POST" as any,
        body: { name }
      })
      if (!isCurrentAuthority(requestOwner)) return
      const createdId = Number(created?.id)
      await refetchMoodboards()
      if (!isCurrentAuthority(requestOwner)) return
      if (Number.isFinite(createdId)) {
        setSelectedMoodboardId(Math.floor(createdId))
      }
      setListViewMode("moodboard")
      setPage(1)
      message.success(`Created collection "${name}"`)
    } catch {
      if (isCurrentAuthority(requestOwner)) {
        message.error("Could not create collection")
      }
    }
  }, [authorityOwner, isCurrentAuthority, message, refetchMoodboards, setSelectedMoodboardId])

  const renameMoodboard = React.useCallback(async () => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) return
    if (!selectedMoodboard) {
      message.warning("Select a collection first")
      return
    }
    const nextName = await promptModal({
      title: 'Rename collection',
      defaultValue: selectedMoodboard.name,
      placeholder: 'Collection name',
      okText: 'Rename',
    })
    if (
      !nextName ||
      nextName === selectedMoodboard.name ||
      !isCurrentAuthority(requestOwner)
    ) return
    const expectedVersion = selectedMoodboard.version ?? 1
    try {
      await bgRequest({
        path: `/api/v1/notes/moodboards/${selectedMoodboard.id}` as any,
        method: "PATCH" as any,
        headers: { "expected-version": String(expectedVersion) } as any,
        body: { name: nextName }
      })
      if (!isCurrentAuthority(requestOwner)) return
      await refetchMoodboards()
      if (!isCurrentAuthority(requestOwner)) return
      message.success(`Renamed collection to "${nextName}"`)
    } catch {
      if (isCurrentAuthority(requestOwner)) {
        message.error("Could not rename collection")
      }
    }
  }, [authorityOwner, isCurrentAuthority, message, refetchMoodboards, selectedMoodboard])

  const deleteMoodboard = React.useCallback(async () => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) return
    if (!selectedMoodboard) {
      message.warning("Select a collection first")
      return
    }
    const ok = await confirmDanger({
      title: "Delete collection?",
      content: `Delete "${selectedMoodboard.name}"?`,
      okText: "Delete",
      cancelText: "Cancel"
    })
    if (!ok || !isCurrentAuthority(requestOwner)) return
    const expectedVersion = selectedMoodboard.version ?? 1
    try {
      await bgRequest({
        path: `/api/v1/notes/moodboards/${selectedMoodboard.id}` as any,
        method: "DELETE" as any,
        headers: { "expected-version": String(expectedVersion) } as any
      })
      if (!isCurrentAuthority(requestOwner)) return
      await refetchMoodboards()
      if (!isCurrentAuthority(requestOwner)) return
      setPage(1)
      message.success("Collection deleted")
    } catch {
      if (isCurrentAuthority(requestOwner)) {
        message.error("Could not delete collection")
      }
    }
  }, [authorityOwner, confirmDanger, isCurrentAuthority, message, refetchMoodboards, selectedMoodboard])

  // ---- notebook server operations ----
  const fetchServerNotebooks = React.useCallback(async (): Promise<NotebookFilterOption[]> => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) return []
    const merged: NotebookFilterOption[] = []
    const seenIds = new Set<number>()
    let offset = 0
    for (let pageIndex = 0; pageIndex < NOTEBOOK_COLLECTION_MAX_PAGES; pageIndex += 1) {
      if (!isCurrentAuthority(requestOwner)) return []
      const params = new URLSearchParams()
      params.set('limit', String(NOTEBOOK_COLLECTION_PAGE_SIZE))
      params.set('offset', String(offset))
      params.set('include_keywords', 'true')
      const response = await bgRequest<any>({
        path: `/api/v1/notes/collections?${params.toString()}` as any,
        method: 'GET' as any
      })
      if (!isCurrentAuthority(requestOwner)) return []
      const pageItems = normalizeNotebookCollectionsResponse(response)
      for (const notebook of pageItems) {
        if (seenIds.has(notebook.id)) continue
        seenIds.add(notebook.id)
        merged.push(notebook)
      }
      const totalHint = Number(
        (response as any)?.total ??
          (response as any)?.pagination?.total_items ??
          NaN
      )
      if (pageItems.length < NOTEBOOK_COLLECTION_PAGE_SIZE) break
      if (Number.isFinite(totalHint) && merged.length >= totalHint) break
      offset += NOTEBOOK_COLLECTION_PAGE_SIZE
    }
    return normalizeNotebookOptions(merged)
  }, [authorityOwner, isCurrentAuthority])

  const upsertNotebookOnServer = React.useCallback(
    async ({
      notebookName,
      keywords,
      existingNotebookId
    }: {
      notebookName: string
      keywords: string[]
      existingNotebookId?: number | null
    }): Promise<NotebookFilterOption | null> => {
      const requestOwner = authorityOwner
      if (!isCurrentAuthority(requestOwner)) return null
      const payload = {
        name: notebookName,
        parent_id: null,
        keywords: normalizeNotebookKeywords(keywords)
      }

      if (existingNotebookId != null) {
        try {
          if (!isCurrentAuthority(requestOwner)) return null
          const updated = await bgRequest<any>({
            path: `/api/v1/notes/collections/${existingNotebookId}` as any,
            method: 'PATCH' as any,
            body: payload as any
          })
          if (!isCurrentAuthority(requestOwner)) return null
          const normalizedUpdated = normalizeNotebookCollectionFromServer(updated)
          if (normalizedUpdated) return normalizedUpdated
        } catch {
          if (!isCurrentAuthority(requestOwner)) return null
          // Fall back to create for local IDs that do not exist server-side.
        }
      }

      if (!isCurrentAuthority(requestOwner)) return null
      const created = await bgRequest<any>({
        path: '/api/v1/notes/collections' as any,
        method: 'POST' as any,
        body: payload as any
      })
      if (!isCurrentAuthority(requestOwner)) return null
      return normalizeNotebookCollectionFromServer(created)
    },
    [authorityOwner, isCurrentAuthority]
  )

  const deleteNotebookOnServer = React.useCallback(async (notebookId: number) => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) return
    await bgRequest<any>({
      path: `/api/v1/notes/collections/${notebookId}` as any,
      method: 'DELETE' as any
    })
  }, [authorityOwner, isCurrentAuthority])

  const migrateLocalNotebooksToServer = React.useCallback(
    async (localNotebooks: NotebookFilterOption[]): Promise<NotebookFilterOption[]> => {
      const requestOwner = authorityOwner
      if (!isCurrentAuthority(requestOwner)) return []
      const normalizedLocal = normalizeNotebookOptions(localNotebooks)
      if (normalizedLocal.length === 0) return []
      for (const notebook of normalizedLocal) {
        if (!isCurrentAuthority(requestOwner)) return []
        try {
          await upsertNotebookOnServer({
            notebookName: notebook.name,
            keywords: notebook.keywords,
            existingNotebookId: notebook.id
          })
        } catch {
          // Continue best-effort migration for remaining notebooks.
        }
      }
      if (!isCurrentAuthority(requestOwner)) return []
      const fetched = await fetchServerNotebooks()
      if (!isCurrentAuthority(requestOwner)) return []
      return fetched.length > 0 ? fetched : normalizedLocal
    },
    [authorityOwner, fetchServerNotebooks, isCurrentAuthority, upsertNotebookOnServer]
  )

  const createNotebookFromCurrentKeywords = React.useCallback(async () => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) return
    const normalizedKeywords = normalizeNotebookKeywords(keywordTokens)
    if (normalizedKeywords.length === 0) {
      message.info('Select at least one tag before saving a filter.')
      return
    }
    if (typeof window === 'undefined') return

    const defaultName = buildNotebookDefaultName(normalizedKeywords)
    const rawName = await promptModal({
      title: 'Save filter',
      label: 'Save the current tag filters as a reusable preset.',
      defaultValue: defaultName,
      placeholder: 'Filter name',
      okText: 'Save',
    })
    if (rawName == null || !isCurrentAuthority(requestOwner)) return

    const notebookName = normalizeNotebookName(rawName)
    if (!notebookName) {
      message.warning('Saved filter name cannot be empty.')
      return
    }

    const normalizedCurrent = normalizeNotebookOptions(notebookOptions)
    const existing = normalizedCurrent.find(
      (entry) => entry.name.toLowerCase() === notebookName.toLowerCase()
    )
    let selectedNotebookAfterSave: NotebookFilterOption | null = null
    if (existing) {
      const updatedOptions = normalizeNotebookOptions(
        normalizedCurrent.map((entry) =>
          entry.id === existing.id
            ? {
                ...entry,
                name: notebookName,
                keywords: normalizedKeywords
              }
            : entry
        )
      )
      setNotebookOptions(updatedOptions)
      selectedNotebookAfterSave = updatedOptions.find((entry) => entry.id === existing.id) || null
      setSelectedNotebookId(existing.id)
    } else {
      const nextId =
        normalizedCurrent.reduce((maxId, entry) => Math.max(maxId, entry.id), 0) + 1
      const createdLocal = { id: nextId, name: notebookName, keywords: normalizedKeywords }
      const nextOptions = normalizeNotebookOptions([
        ...normalizedCurrent,
        createdLocal
      ])
      setNotebookOptions(nextOptions)
      selectedNotebookAfterSave = nextOptions.find((entry) => entry.id === nextId) || null
      setSelectedNotebookId(nextId)
    }
    setKeywordTokens([])
    setPage(1)
    message.success(`Saved filter "${notebookName}"`)

    if (
      isOnline &&
      selectedNotebookAfterSave &&
      isCurrentAuthority(requestOwner)
    ) {
      try {
        const persisted = await upsertNotebookOnServer({
          notebookName: selectedNotebookAfterSave.name,
          keywords: selectedNotebookAfterSave.keywords,
          existingNotebookId: selectedNotebookAfterSave.id
        })
        if (!isCurrentAuthority(requestOwner)) return
        if (persisted) {
          setNotebookOptions((current) =>
            normalizeNotebookOptions(
              current.map((entry) =>
                entry.id === selectedNotebookAfterSave?.id
                  ? persisted
                  : entry
              )
            )
          )
          setSelectedNotebookId(persisted.id)
        }
      } catch {
        if (isCurrentAuthority(requestOwner)) {
          message.warning('Saved locally, but failed to sync saved filter to server.')
        }
      }
    }
  }, [authorityOwner, isCurrentAuthority, isOnline, keywordTokens, message, notebookOptions, setKeywordTokens, setNotebookOptions, setSelectedNotebookId, upsertNotebookOnServer])

  const removeSelectedNotebook = React.useCallback(async () => {
    const requestOwner = authorityOwner
    if (!isCurrentAuthority(requestOwner)) return
    if (selectedNotebookId == null) return
    const notebookToRemove =
      notebookOptions.find((entry) => entry.id === selectedNotebookId) || null
    if (!notebookToRemove) {
      setSelectedNotebookId(null)
      return
    }
    const ok = await confirmDanger({
      title: 'Remove saved filter?',
      content: `Remove "${notebookToRemove.name}" from saved filters? This does not delete any notes.`,
      okText: 'Remove',
      cancelText: 'Cancel'
    })
    if (!ok || !isCurrentAuthority(requestOwner)) return
    setNotebookOptions((current) =>
      current.filter((entry) => entry.id !== notebookToRemove.id)
    )
    setSelectedNotebookId(null)
    setPage(1)
    message.success(`Removed saved filter "${notebookToRemove.name}"`)
    if (isOnline) {
      try {
        await deleteNotebookOnServer(notebookToRemove.id)
        if (!isCurrentAuthority(requestOwner)) return
      } catch {
        if (isCurrentAuthority(requestOwner)) {
          message.warning('Removed locally, but failed to remove saved filter on server.')
        }
      }
    }
  }, [authorityOwner, confirmDanger, deleteNotebookOnServer, isCurrentAuthority, isOnline, message, notebookOptions, selectedNotebookId, setNotebookOptions, setSelectedNotebookId])

  const handleClearFilters = React.useCallback(() => {
    setQuery('')
    setQueryInput('')
    setKeywordTokens([])
    setSelectedNotebookId(null)
    setListViewMode('list')
    setPage(1)
  }, [setKeywordTokens, setListViewMode, setSelectedNotebookId])

  const clearBulkSelection = React.useCallback(() => {
    setBulkSelectedIds([])
    bulkSelectionAnchorRef.current.value = null
  }, [setBulkSelectedIds])

  const handleToggleBulkSelection = React.useCallback(
    (id: string | number, checked: boolean, shiftKey: boolean) => {
      if (listMode !== 'active') return
      const targetId = String(id)
      setBulkSelectedIds((current) => {
        const next = new Set(current)
        const anchorId = bulkSelectionAnchorRef.current.value
        const canApplyRange =
          shiftKey &&
          !!anchorId &&
          orderedVisibleNoteIds.includes(anchorId) &&
          orderedVisibleNoteIds.includes(targetId)

        if (canApplyRange && anchorId) {
          const start = orderedVisibleNoteIds.indexOf(anchorId)
          const end = orderedVisibleNoteIds.indexOf(targetId)
          const [minIndex, maxIndex] = start <= end ? [start, end] : [end, start]
          const rangeIds = orderedVisibleNoteIds.slice(minIndex, maxIndex + 1)
          for (const rangeId of rangeIds) {
            if (checked) next.add(rangeId)
            else next.delete(rangeId)
          }
        } else {
          if (checked) next.add(targetId)
          else next.delete(targetId)
        }

        bulkSelectionAnchorRef.current.value = targetId
        return orderedVisibleNoteIds.filter((visibleId) => next.has(visibleId))
      })
    },
    [listMode, orderedVisibleNoteIds, setBulkSelectedIds]
  )

  // ---- search debounce ----
  React.useEffect(() => {
    if (queryInput === query) return
    if (typeof window === 'undefined') {
      setQuery(queryInput)
      setPage(1)
      return
    }
    clearSearchQueryTimeout()
    searchQueryTimeoutRef.current = window.setTimeout(() => {
      setQuery(queryInput)
      setPage(1)
      searchQueryTimeoutRef.current = null
    }, NOTE_SEARCH_DEBOUNCE_MS)
    return () => {
      clearSearchQueryTimeout()
    }
  }, [clearSearchQueryTimeout, query, queryInput])

  // ---- bulk selection sync with visible ids ----
  React.useEffect(() => {
    if (listMode !== 'active') {
      setBulkSelectedIds([])
      bulkSelectionAnchorRef.current.value = null
      return
    }
    if (orderedVisibleNoteIds.length === 0) {
      setBulkSelectedIds([])
      bulkSelectionAnchorRef.current.value = null
      return
    }
    setBulkSelectedIds((current) => {
      const visibleSet = new Set(orderedVisibleNoteIds)
      const filtered = current.filter((id) => visibleSet.has(id))
      const unchanged =
        filtered.length === current.length &&
        filtered.every((id, index) => id === current[index])
      return unchanged ? current : filtered
    })
    if (
      bulkSelectionAnchorRef.current.value &&
      !orderedVisibleNoteIds.includes(bulkSelectionAnchorRef.current.value)
    ) {
      bulkSelectionAnchorRef.current.value = null
    }
  }, [listMode, orderedVisibleNoteIds, setBulkSelectedIds])

  // ---- page size persistence ----
  React.useEffect(() => {
    let cancelled = false
    void (async () => {
      const savedPageSize = await getSetting(NOTES_PAGE_SIZE_SETTING)
      if (cancelled) return
      if (typeof savedPageSize === 'number' && [20, 50, 100].includes(savedPageSize)) {
        setPageSize(savedPageSize)
      }
      pageSizeSettingHydratedRef.current = true
    })()
    return () => {
      cancelled = true
    }
  }, [])

  React.useEffect(() => {
    if (!pageSizeSettingHydratedRef.current) return
    void setSetting(NOTES_PAGE_SIZE_SETTING, pageSize)
  }, [pageSize])

  // ---- notebook persistence ----
  React.useEffect(() => {
    let cancelled = false
    const requestOwner = authorityOwner
    notebookSettingsHydratedRef.current = null
    if (!isCurrentAuthority(requestOwner)) {
      return () => {
        cancelled = true
      }
    }
    void (async () => {
      try {
        if (requestOwner.scope !== undefined) {
          if (!isOnline) {
            setNotebookOptions([])
            return
          }
          try {
            const serverNotebooks = await fetchServerNotebooks()
            if (cancelled || !isCurrentAuthority(requestOwner)) return
            setNotebookOptions(serverNotebooks)
          } catch {
            if (cancelled || !isCurrentAuthority(requestOwner)) return
            setNotebookOptions([])
          }
          return
        }
        const savedNotebooks = await getSetting(NOTES_NOTEBOOKS_SETTING)
        if (cancelled || !isCurrentAuthority(requestOwner)) return
        const localNotebooks = normalizeNotebookOptions(savedNotebooks)
        if (!isOnline) {
          setNotebookOptions(localNotebooks)
          return
        }
        try {
          const serverNotebooks = await fetchServerNotebooks()
          if (cancelled || !isCurrentAuthority(requestOwner)) return
          if (serverNotebooks.length > 0) {
            setNotebookOptions(serverNotebooks)
            return
          }
          if (localNotebooks.length > 0) {
            const migrated = await migrateLocalNotebooksToServer(localNotebooks)
            if (cancelled || !isCurrentAuthority(requestOwner)) return
            setNotebookOptions(migrated)
            return
          }
          setNotebookOptions([])
        } catch {
          if (cancelled || !isCurrentAuthority(requestOwner)) return
          setNotebookOptions(localNotebooks)
        }
      } finally {
        if (!cancelled && isCurrentAuthority(requestOwner)) {
          notebookSettingsHydratedRef.current = {
            authorityOwner: requestOwner
          }
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [authorityOwner, fetchServerNotebooks, isCurrentAuthority, isOnline, migrateLocalNotebooksToServer, setNotebookOptions])

  React.useEffect(() => {
    if (
      authorityOwner.scope !== undefined ||
      !isCurrentAuthority(authorityOwner) ||
      notebookSettingsHydratedRef.current?.authorityOwner !== authorityOwner
    ) return
    void setSetting(NOTES_NOTEBOOKS_SETTING, normalizeNotebookOptions(notebookOptions))
  }, [authorityOwner, isCurrentAuthority, notebookOptions])

  React.useEffect(() => {
    if (selectedNotebookId == null) return
    if (notebookOptions.some((entry) => entry.id === selectedNotebookId)) return
    setSelectedNotebookId(null)
  }, [notebookOptions, selectedNotebookId, setSelectedNotebookId])

  // ---- cleanup ----
  React.useEffect(() => {
    return () => {
      clearSearchQueryTimeout()
    }
  }, [clearSearchQueryTimeout])

  // ---- computed filters ----
  const hasActiveFilters =
    listMode === 'active' &&
    listViewMode !== 'moodboard' &&
    (queryInput.trim().length > 0 || effectiveKeywordTokens.length > 0 || selectedNotebookId != null)

  return {
    // state
    query, setQuery,
    queryInput, setQueryInput,
    searchTipsQuery, setSearchTipsQuery,
    page, setPage,
    pageSize, setPageSize,
    sortOption, setSortOption,
    listMode, setListMode,
    listViewMode, setListViewMode,
    total, setTotal,
    bulkSelectedIds, setBulkSelectedIds,
    moodboards, selectedMoodboardId, setSelectedMoodboardId,
    selectedMoodboard,
    isMoodboardsFetching,
    moodboardTotalPages, moodboardCanGoPrev, moodboardCanGoNext,
    moodboardRangeStart, moodboardRangeEnd,
    notebookOptions, setNotebookOptions,
    selectedNotebookId, setSelectedNotebookId,
    selectedNotebook,
    // derived
    effectiveKeywordTokens,
    rawNotes, filteredCount,
    orderedVisibleNoteIds,
    bulkSelectedIdSet, selectedBulkNotes,
    hasActiveFilters,
    // query data
    data, error, isError, isFetching, isPlaceholderData, refetch, listErrorMessage,
    // helpers
    fetchFilteredNotesRaw,
    clearSearchQueryTimeout,
    searchQueryTimeoutRef,
    // callbacks
    createMoodboard, renameMoodboard, deleteMoodboard,
    createNotebookFromCurrentKeywords, removeSelectedNotebook,
    handleClearFilters,
    clearBulkSelection, handleToggleBulkSelection,
  }
}
