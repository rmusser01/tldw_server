import React from 'react'
import type { MessageInstance } from 'antd/es/message/interface'
import type { QueryClient } from '@tanstack/react-query'
import { useQuery, keepPreviousData } from '@tanstack/react-query'
import { bgRequest } from '@/services/background-proxy'
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
} from '../notes-manager-utils'
import type { ConfirmDangerOptions } from '@/components/Common/confirm-danger'

type ConfirmDanger = (options: ConfirmDangerOptions) => Promise<boolean>

export interface UseNotesListManagementDeps {
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
  const [total, setTotal] = React.useState(0)
  const [bulkSelectedIds, setBulkSelectedIds] = React.useState<string[]>([])
  const bulkSelectionAnchorRef = React.useRef<string | null>(null)

  // ---- moodboard state ----
  const [moodboards, setMoodboards] = React.useState<MoodboardSummary[]>([])
  const [selectedMoodboardId, setSelectedMoodboardId] = React.useState<number | null>(null)

  // ---- notebook state ----
  const [notebookOptions, setNotebookOptions] = React.useState<NotebookFilterOption[]>([])
  const [selectedNotebookId, setSelectedNotebookId] = React.useState<number | null>(null)

  const searchQueryTimeoutRef = React.useRef<number | null>(null)
  const pageSizeSettingHydratedRef = React.useRef(false)
  const notebookSettingsHydratedRef = React.useRef(false)

  // ---- derived ----
  const effectiveKeywordTokens = React.useMemo(() => {
    const merged = [...keywordTokens, ...notebookKeywordTokens]
    const deduped: string[] = []
    for (const token of merged) {
      const normalized = String(token || '').trim().toLowerCase()
      if (!normalized) continue
      if (deduped.includes(normalized)) continue
      deduped.push(normalized)
    }
    return deduped
  }, [keywordTokens, notebookKeywordTokens])

  const selectedNotebook = React.useMemo(
    () =>
      selectedNotebookId == null
        ? null
        : notebookOptions.find((option) => option.id === selectedNotebookId) || null,
    [notebookOptions, selectedNotebookId]
  )

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

    let items: any[] = []
    let totalVal = 0

    if (Array.isArray(abs)) {
      items = abs
      totalVal = abs.length
    } else if (abs && typeof abs === 'object') {
      if (Array.isArray((abs as any).items)) {
        items = (abs as any).items
      }
      const pagination = (abs as any).pagination
      if (pagination && typeof pagination.total_items === 'number') {
        totalVal = Number(pagination.total_items)
      } else if (Array.isArray((abs as any).items)) {
        totalVal = (abs as any).items.length
      }
    }

    return { items: sortNoteRows(items, sortOption), total: totalVal }
  }, [sortOption])

  const fetchNotes = React.useCallback(async (): Promise<NoteListItem[]> => {
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
    const items = Array.isArray(res?.items) ? res.items : (Array.isArray(res) ? res : [])
    const pagination = res?.pagination
    setTotal(Number(pagination?.total_items || items.length || 0))
    return sortNoteRows(items, sortOption).map(mapNoteListItem)
  }, [effectiveKeywordTokens, fetchFilteredNotesRaw, listMode, listViewMode, page, pageSize, query, selectedMoodboardId, sortOption])

  const { data, isFetching, refetch } = useQuery({
    queryKey: [
      'notes',
      listMode,
      listViewMode,
      selectedMoodboardId ?? 'none',
      query,
      page,
      pageSize,
      sortOption,
      selectedNotebookId ?? 'none',
      effectiveKeywordTokens.join('|')
    ],
    queryFn: fetchNotes,
    placeholderData: keepPreviousData,
    enabled: isOnline
  })

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
    const pageLimit = 200
    const maxPages = 50
    const collected: any[] = []
    let offset = 0

    for (let pageIndex = 0; pageIndex < maxPages; pageIndex += 1) {
      const res = await bgRequest<any>({
        path: `/api/v1/notes/moodboards?limit=${pageLimit}&offset=${offset}` as any,
        method: "GET" as any
      })
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
          name: String(item?.name || "").trim() || `Moodboard ${id}`,
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
  }, [])

  const {
    data: moodboardData,
    isFetching: isMoodboardsFetching,
    refetch: refetchMoodboards
  } = useQuery({
    queryKey: ["notes-moodboards", listMode, listViewMode],
    queryFn: fetchMoodboards,
    enabled: isOnline && listMode === "active" && listViewMode === "moodboard"
  })

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
  }, [moodboardData])

  const selectedMoodboard = React.useMemo(() => {
    if (selectedMoodboardId == null) return null
    return moodboards.find((item) => item.id === selectedMoodboardId) || null
  }, [moodboards, selectedMoodboardId])

  const createMoodboard = React.useCallback(async () => {
    const name = String(window.prompt("Moodboard name") || "").trim()
    if (!name) return
    try {
      const created = await bgRequest<any>({
        path: "/api/v1/notes/moodboards" as any,
        method: "POST" as any,
        body: { name }
      })
      const createdId = Number(created?.id)
      await refetchMoodboards()
      if (Number.isFinite(createdId)) {
        setSelectedMoodboardId(Math.floor(createdId))
      }
      setListViewMode("moodboard")
      setPage(1)
      message.success(`Created moodboard "${name}"`)
    } catch {
      message.error("Could not create moodboard")
    }
  }, [message, refetchMoodboards])

  const renameMoodboard = React.useCallback(async () => {
    if (!selectedMoodboard) {
      message.warning("Select a moodboard first")
      return
    }
    const nextName = String(window.prompt("Rename moodboard", selectedMoodboard.name) || "").trim()
    if (!nextName || nextName === selectedMoodboard.name) return
    const expectedVersion = selectedMoodboard.version ?? 1
    try {
      await bgRequest({
        path: `/api/v1/notes/moodboards/${selectedMoodboard.id}` as any,
        method: "PATCH" as any,
        headers: { "expected-version": String(expectedVersion) } as any,
        body: { name: nextName }
      })
      await refetchMoodboards()
      message.success(`Renamed moodboard to "${nextName}"`)
    } catch {
      message.error("Could not rename moodboard")
    }
  }, [message, refetchMoodboards, selectedMoodboard])

  const deleteMoodboard = React.useCallback(async () => {
    if (!selectedMoodboard) {
      message.warning("Select a moodboard first")
      return
    }
    const ok = await confirmDanger({
      title: "Delete moodboard?",
      content: `Delete "${selectedMoodboard.name}"?`,
      okText: "Delete",
      cancelText: "Cancel"
    })
    if (!ok) return
    const expectedVersion = selectedMoodboard.version ?? 1
    try {
      await bgRequest({
        path: `/api/v1/notes/moodboards/${selectedMoodboard.id}` as any,
        method: "DELETE" as any,
        headers: { "expected-version": String(expectedVersion) } as any
      })
      await refetchMoodboards()
      setPage(1)
      message.success("Moodboard deleted")
    } catch {
      message.error("Could not delete moodboard")
    }
  }, [confirmDanger, message, refetchMoodboards, selectedMoodboard])

  // ---- notebook server operations ----
  const fetchServerNotebooks = React.useCallback(async (): Promise<NotebookFilterOption[]> => {
    const merged: NotebookFilterOption[] = []
    const seenIds = new Set<number>()
    let offset = 0
    for (let pageIndex = 0; pageIndex < NOTEBOOK_COLLECTION_MAX_PAGES; pageIndex += 1) {
      const params = new URLSearchParams()
      params.set('limit', String(NOTEBOOK_COLLECTION_PAGE_SIZE))
      params.set('offset', String(offset))
      params.set('include_keywords', 'true')
      const response = await bgRequest<any>({
        path: `/api/v1/notes/collections?${params.toString()}` as any,
        method: 'GET' as any
      })
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
  }, [])

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
      const payload = {
        name: notebookName,
        parent_id: null,
        keywords: normalizeNotebookKeywords(keywords)
      }

      if (existingNotebookId != null) {
        try {
          const updated = await bgRequest<any>({
            path: `/api/v1/notes/collections/${existingNotebookId}` as any,
            method: 'PATCH' as any,
            body: payload as any
          })
          const normalizedUpdated = normalizeNotebookCollectionFromServer(updated)
          if (normalizedUpdated) return normalizedUpdated
        } catch {
          // Fall back to create for local IDs that do not exist server-side.
        }
      }

      const created = await bgRequest<any>({
        path: '/api/v1/notes/collections' as any,
        method: 'POST' as any,
        body: payload as any
      })
      return normalizeNotebookCollectionFromServer(created)
    },
    []
  )

  const deleteNotebookOnServer = React.useCallback(async (notebookId: number) => {
    await bgRequest<any>({
      path: `/api/v1/notes/collections/${notebookId}` as any,
      method: 'DELETE' as any
    })
  }, [])

  const migrateLocalNotebooksToServer = React.useCallback(
    async (localNotebooks: NotebookFilterOption[]): Promise<NotebookFilterOption[]> => {
      const normalizedLocal = normalizeNotebookOptions(localNotebooks)
      if (normalizedLocal.length === 0) return []
      for (const notebook of normalizedLocal) {
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
      const fetched = await fetchServerNotebooks()
      return fetched.length > 0 ? fetched : normalizedLocal
    },
    [fetchServerNotebooks, upsertNotebookOnServer]
  )

  const createNotebookFromCurrentKeywords = React.useCallback(async () => {
    const normalizedKeywords = normalizeNotebookKeywords(keywordTokens)
    if (normalizedKeywords.length === 0) {
      message.info('Select at least one keyword before saving a notebook.')
      return
    }
    if (typeof window === 'undefined') return

    const defaultName = buildNotebookDefaultName(normalizedKeywords)
    const rawName = window.prompt('Smart collection name', defaultName)
    if (rawName == null) return

    const notebookName = normalizeNotebookName(rawName)
    if (!notebookName) {
      message.warning('Smart collection name cannot be empty.')
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
    message.success(`Saved smart collection "${notebookName}"`)

    if (isOnline && selectedNotebookAfterSave) {
      try {
        const persisted = await upsertNotebookOnServer({
          notebookName: selectedNotebookAfterSave.name,
          keywords: selectedNotebookAfterSave.keywords,
          existingNotebookId: selectedNotebookAfterSave.id
        })
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
        message.warning('Saved locally, but failed to sync notebook to server.')
      }
    }
  }, [isOnline, keywordTokens, message, notebookOptions, setKeywordTokens, upsertNotebookOnServer])

  const removeSelectedNotebook = React.useCallback(async () => {
    if (selectedNotebookId == null) return
    const notebookToRemove =
      notebookOptions.find((entry) => entry.id === selectedNotebookId) || null
    if (!notebookToRemove) {
      setSelectedNotebookId(null)
      return
    }
    const ok = await confirmDanger({
      title: 'Remove notebook?',
      content: `Remove "${notebookToRemove.name}" from notebook filters? This does not delete any notes.`,
      okText: 'Remove',
      cancelText: 'Cancel'
    })
    if (!ok) return
    setNotebookOptions((current) =>
      current.filter((entry) => entry.id !== notebookToRemove.id)
    )
    setSelectedNotebookId(null)
    setPage(1)
    message.success(`Removed notebook "${notebookToRemove.name}"`)
    if (isOnline) {
      try {
        await deleteNotebookOnServer(notebookToRemove.id)
      } catch {
        message.warning('Removed locally, but failed to remove notebook on server.')
      }
    }
  }, [confirmDanger, deleteNotebookOnServer, isOnline, message, notebookOptions, selectedNotebookId])

  const handleClearFilters = React.useCallback(() => {
    setQuery('')
    setQueryInput('')
    setKeywordTokens([])
    setSelectedNotebookId(null)
    setPage(1)
  }, [setKeywordTokens])

  const clearBulkSelection = React.useCallback(() => {
    setBulkSelectedIds([])
    bulkSelectionAnchorRef.current = null
  }, [])

  const handleToggleBulkSelection = React.useCallback(
    (id: string | number, checked: boolean, shiftKey: boolean) => {
      if (listMode !== 'active') return
      const targetId = String(id)
      setBulkSelectedIds((current) => {
        const next = new Set(current)
        const anchorId = bulkSelectionAnchorRef.current
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

        bulkSelectionAnchorRef.current = targetId
        return orderedVisibleNoteIds.filter((visibleId) => next.has(visibleId))
      })
    },
    [listMode, orderedVisibleNoteIds]
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
      bulkSelectionAnchorRef.current = null
      return
    }
    if (orderedVisibleNoteIds.length === 0) {
      setBulkSelectedIds([])
      bulkSelectionAnchorRef.current = null
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
      bulkSelectionAnchorRef.current &&
      !orderedVisibleNoteIds.includes(bulkSelectionAnchorRef.current)
    ) {
      bulkSelectionAnchorRef.current = null
    }
  }, [listMode, orderedVisibleNoteIds])

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
    void (async () => {
      try {
        const savedNotebooks = await getSetting(NOTES_NOTEBOOKS_SETTING)
        if (cancelled) return
        const localNotebooks = normalizeNotebookOptions(savedNotebooks)
        if (!isOnline) {
          setNotebookOptions(localNotebooks)
          return
        }
        try {
          const serverNotebooks = await fetchServerNotebooks()
          if (cancelled) return
          if (serverNotebooks.length > 0) {
            setNotebookOptions(serverNotebooks)
            return
          }
          if (localNotebooks.length > 0) {
            const migrated = await migrateLocalNotebooksToServer(localNotebooks)
            if (cancelled) return
            setNotebookOptions(migrated)
            return
          }
          setNotebookOptions([])
        } catch {
          if (cancelled) return
          setNotebookOptions(localNotebooks)
        }
      } finally {
        if (!cancelled) {
          notebookSettingsHydratedRef.current = true
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [fetchServerNotebooks, isOnline, migrateLocalNotebooksToServer])

  React.useEffect(() => {
    if (!notebookSettingsHydratedRef.current) return
    void setSetting(NOTES_NOTEBOOKS_SETTING, normalizeNotebookOptions(notebookOptions))
  }, [notebookOptions])

  React.useEffect(() => {
    if (selectedNotebookId == null) return
    if (notebookOptions.some((entry) => entry.id === selectedNotebookId)) return
    setSelectedNotebookId(null)
  }, [notebookOptions, selectedNotebookId])

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
    data, isFetching, refetch,
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
