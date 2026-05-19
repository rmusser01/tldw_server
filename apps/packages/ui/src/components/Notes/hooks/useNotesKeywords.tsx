import React from 'react'
import type { MessageInstance } from 'antd/es/message/interface'
import type { QueryClient } from '@tanstack/react-query'
import { bgRequest } from '@/services/background-proxy'
import { getAllNoteKeywordStats, searchNoteKeywords } from '@/services/note-keywords'
import type {
  KeywordPickerSortMode,
  KeywordFrequencyTone,
  KeywordManagementItem,
  KeywordRenameDraft,
  KeywordMergeDraft,
  NotesAssistAction,
} from '../notes-manager-types'
import {
  KEYWORD_FREQUENCY_DOT_CLASS,
  toKeywordTestIdSegment,
  suggestKeywordsDraft,
} from '../notes-manager-utils'
import type { ConfirmDangerOptions } from '@/components/Common/confirm-danger'

type ConfirmDanger = (options: ConfirmDangerOptions) => Promise<boolean>

export interface UseNotesKeywordsDeps {
  isOnline: boolean
  /** Current list mode (active|trash) - needed for auto-loading keywords */
  listMode: 'active' | 'trash'
  message: MessageInstance
  confirmDanger: ConfirmDanger
  restoreFocusAfterOverlayClose: (target: HTMLElement | null) => void
  /** Notebook keyword tokens computed by the orchestrator */
  notebookKeywordTokens: string[]
  /** React Query client for cache invalidation */
  queryClient: QueryClient
  /** i18n translator */
  t: (key: string, opts?: Record<string, any>) => string
  /** Cross-cutting state setters the keyword layer needs */
  setPage: React.Dispatch<React.SetStateAction<number>>
  setIsDirty: React.Dispatch<React.SetStateAction<boolean>>
  setSaveIndicator: (state: 'dirty' | 'saving' | 'saved' | 'error' | 'idle') => void
  setMonitoringNotice: (notice: any) => void
  markGeneratedEdit: (action: NotesAssistAction) => void
}

export function useNotesKeywords(deps: UseNotesKeywordsDeps) {
  const {
    isOnline,
    listMode,
    message,
    confirmDanger,
    restoreFocusAfterOverlayClose,
    notebookKeywordTokens,
    queryClient,
    t,
    setPage,
    setIsDirty,
    setSaveIndicator,
    setMonitoringNotice,
    markGeneratedEdit,
  } = deps

  const [keywordTokens, setKeywordTokens] = React.useState<string[]>([])
  const [keywordOptions, setKeywordOptions] = React.useState<string[]>([])
  const [allKeywords, setAllKeywords] = React.useState<string[]>([])
  const [keywordNoteCountByKey, setKeywordNoteCountByKey] = React.useState<Record<string, number>>({})
  const allKeywordsRef = React.useRef<string[]>([])
  allKeywordsRef.current = allKeywords
  const [keywordPickerOpen, setKeywordPickerOpen] = React.useState(false)
  const [keywordPickerQuery, setKeywordPickerQuery] = React.useState('')
  const [keywordPickerSelection, setKeywordPickerSelection] = React.useState<string[]>([])
  const [keywordPickerSortMode, setKeywordPickerSortMode] =
    React.useState<KeywordPickerSortMode>('frequency_desc')
  const [recentKeywordHistory, setRecentKeywordHistory] = React.useState<string[]>([])
  const [keywordManagerOpen, setKeywordManagerOpen] = React.useState(false)
  const [keywordManagerLoading, setKeywordManagerLoading] = React.useState(false)
  const [keywordManagerQuery, setKeywordManagerQuery] = React.useState('')
  const [keywordManagerItems, setKeywordManagerItems] = React.useState<KeywordManagementItem[]>([])
  const [keywordRenameDraft, setKeywordRenameDraft] = React.useState<KeywordRenameDraft | null>(
    null
  )
  const [keywordMergeDraft, setKeywordMergeDraft] = React.useState<KeywordMergeDraft | null>(null)
  const [keywordManagerActionLoading, setKeywordManagerActionLoading] = React.useState(false)
  const [keywordSuggestionOptions, setKeywordSuggestionOptions] = React.useState<string[]>([])
  const [keywordSuggestionSelection, setKeywordSuggestionSelection] = React.useState<string[]>([])
  const [editorKeywords, setEditorKeywords] = React.useState<string[]>([])

  const keywordPickerReturnFocusRef = React.useRef<HTMLElement | null>(null)
  const keywordManagerReturnFocusRef = React.useRef<HTMLElement | null>(null)
  const keywordSuggestionReturnFocusRef = React.useRef<HTMLElement | null>(null)
  const keywordSearchTimeoutRef = React.useRef<number | null>(null)

  const availableKeywords = React.useMemo(() => {
    const base = allKeywords.length ? allKeywords : keywordOptions
    const seen = new Set<string>()
    const out: string[] = []
    const add = (value: string) => {
      const text = String(value || '').trim()
      if (!text) return
      const key = text.toLowerCase()
      if (seen.has(key)) return
      seen.add(key)
      out.push(text)
    }
    base.forEach(add)
    keywordTokens.forEach(add)
    notebookKeywordTokens.forEach(add)
    return out
  }, [allKeywords, keywordOptions, keywordTokens, notebookKeywordTokens])

  const rememberRecentKeywords = React.useCallback((keywords: string[]) => {
    const nextKeywords = keywords
      .map((keyword) => String(keyword || '').trim())
      .filter(Boolean)
    if (nextKeywords.length === 0) return
    setRecentKeywordHistory((current) => {
      const seen = new Set<string>()
      const ordered: string[] = []
      for (const keyword of nextKeywords) {
        const key = keyword.toLowerCase()
        if (seen.has(key)) continue
        seen.add(key)
        ordered.push(keyword)
      }
      for (const keyword of current) {
        const key = keyword.toLowerCase()
        if (seen.has(key)) continue
        seen.add(key)
        ordered.push(keyword)
      }
      return ordered.slice(0, 20)
    })
  }, [])

  const filteredKeywordPickerOptions = React.useMemo(() => {
    const q = keywordPickerQuery.trim().toLowerCase()
    if (!q) return availableKeywords
    return availableKeywords.filter((kw) => kw.toLowerCase().includes(q))
  }, [availableKeywords, keywordPickerQuery])

  const sortedKeywordPickerOptions = React.useMemo(() => {
    const options = [...filteredKeywordPickerOptions]
    options.sort((a, b) => {
      if (keywordPickerSortMode === 'alpha_asc') {
        return a.localeCompare(b)
      }
      if (keywordPickerSortMode === 'alpha_desc') {
        return b.localeCompare(a)
      }
      const countA = keywordNoteCountByKey[a.toLowerCase()] ?? 0
      const countB = keywordNoteCountByKey[b.toLowerCase()] ?? 0
      if (countA !== countB) {
        return countB - countA
      }
      return a.localeCompare(b)
    })
    return options
  }, [filteredKeywordPickerOptions, keywordNoteCountByKey, keywordPickerSortMode])

  const recentKeywordPickerOptions = React.useMemo(() => {
    const availableByKey = new Map<string, string>()
    for (const keyword of availableKeywords) {
      availableByKey.set(keyword.toLowerCase(), keyword)
    }
    const q = keywordPickerQuery.trim().toLowerCase()
    const recent: string[] = []
    for (const keyword of recentKeywordHistory) {
      const resolved = availableByKey.get(keyword.toLowerCase())
      if (!resolved) continue
      if (q && !resolved.toLowerCase().includes(q)) continue
      recent.push(resolved)
      if (recent.length >= 8) break
    }
    return recent
  }, [availableKeywords, keywordPickerQuery, recentKeywordHistory])

  const maxKeywordNoteCount = React.useMemo(() => {
    let maxCount = 0
    for (const rawCount of Object.values(keywordNoteCountByKey)) {
      const count = Number(rawCount)
      if (!Number.isFinite(count)) continue
      if (count > maxCount) maxCount = count
    }
    return maxCount
  }, [keywordNoteCountByKey])

  const getKeywordFrequencyTone = React.useCallback(
    (keyword: string): KeywordFrequencyTone => {
      const count = keywordNoteCountByKey[keyword.toLowerCase()]
      if (typeof count !== 'number' || count <= 0 || maxKeywordNoteCount <= 0) {
        return 'none'
      }
      const ratio = count / maxKeywordNoteCount
      if (ratio >= 0.67) return 'high'
      if (ratio >= 0.34) return 'medium'
      return 'low'
    },
    [keywordNoteCountByKey, maxKeywordNoteCount]
  )

  const renderKeywordLabelWithFrequency = React.useCallback(
    (
      keyword: string,
      options?: {
        includeCount?: boolean
        testIdPrefix?: string
      }
    ) => {
      const includeCount = options?.includeCount ?? true
      const noteCount = keywordNoteCountByKey[keyword.toLowerCase()]
      const displayLabel =
        includeCount && typeof noteCount === 'number' ? `${keyword} (${noteCount})` : keyword
      const tone = getKeywordFrequencyTone(keyword)
      const dotClass = KEYWORD_FREQUENCY_DOT_CLASS[tone]
      const testId = options?.testIdPrefix
        ? `${options.testIdPrefix}-${toKeywordTestIdSegment(keyword)}`
        : undefined
      return (
        <span
          className="inline-flex items-center gap-1.5"
          data-frequency-tone={tone}
          data-testid={testId}
        >
          <span className={`inline-block h-2 w-2 rounded-full ${dotClass}`} aria-hidden="true" />
          <span>{displayLabel}</span>
        </span>
      )
    },
    [getKeywordFrequencyTone, keywordNoteCountByKey]
  )

  const loadAllKeywords = React.useCallback(async (force = false) => {
    // Cached for session; add a refresh/TTL if keyword updates become frequent.
    if (!force && allKeywordsRef.current.length > 0) return
    try {
      const keywordStats = await getAllNoteKeywordStats()
      const arr = keywordStats.map((entry) => entry.keyword)
      const nextCounts: Record<string, number> = {}
      for (const entry of keywordStats) {
        const key = entry.keyword.toLowerCase()
        nextCounts[key] = Math.max(0, Number(entry.noteCount) || 0)
      }
      setAllKeywords(arr)
      setKeywordOptions(arr)
      setKeywordNoteCountByKey(nextCounts)
    } catch {
      console.debug('[NotesManagerPage] Keyword suggestions load failed')
    }
  }, [])

  // Auto-load keywords when entering active list mode while online
  React.useEffect(() => {
    if (!isOnline || listMode !== 'active') return
    void loadAllKeywords()
  }, [isOnline, listMode, loadAllKeywords])

  const loadKeywordManagementItems = React.useCallback(async () => {
    if (!isOnline) return [] as KeywordManagementItem[]
    setKeywordManagerLoading(true)
    try {
      const pageSize = 250
      const maxPages = 40
      const collected = new Map<string, KeywordManagementItem>()
      let offset = 0

      for (let pageIndex = 0; pageIndex < maxPages; pageIndex += 1) {
        const result = await bgRequest<any>({
          path:
            `/api/v1/notes/keywords/?limit=${pageSize}&offset=${offset}&include_note_counts=true` as any,
          method: 'GET' as any
        })
        const rows = Array.isArray(result) ? result : []
        if (rows.length === 0) break

        for (const row of rows) {
          const keyword = String(row?.keyword || row?.keyword_text || '').trim()
          const id = Number(row?.id)
          if (!keyword || !Number.isFinite(id)) continue
          const key = keyword.toLowerCase()
          const versionRaw = Number(row?.version)
          const version = Number.isFinite(versionRaw) && versionRaw > 0 ? Math.floor(versionRaw) : 1
          const noteCountRaw = Number(row?.note_count ?? row?.count ?? 0)
          const noteCount =
            Number.isFinite(noteCountRaw) && noteCountRaw >= 0 ? Math.floor(noteCountRaw) : 0

          const existing = collected.get(key)
          if (!existing || version >= existing.version) {
            collected.set(key, {
              id: Math.floor(id),
              keyword,
              version,
              noteCount
            })
          }
        }

        if (rows.length < pageSize) break
        offset += pageSize
      }

      const nextItems = Array.from(collected.values()).sort((a, b) =>
        a.keyword.localeCompare(b.keyword)
      )
      const nextKeywords = nextItems.map((item) => item.keyword)
      const nextCounts: Record<string, number> = {}
      for (const item of nextItems) {
        nextCounts[item.keyword.toLowerCase()] = item.noteCount
      }

      setKeywordManagerItems(nextItems)
      setAllKeywords(nextKeywords)
      setKeywordOptions(nextKeywords)
      setKeywordNoteCountByKey(nextCounts)
      return nextItems
    } catch (error: any) {
      message.error(String(error?.message || 'Could not load tag management data'))
      return [] as KeywordManagementItem[]
    } finally {
      setKeywordManagerLoading(false)
    }
  }, [isOnline, message])

  const refreshKeywordDataAfterManagement = React.useCallback(async () => {
    await loadKeywordManagementItems()
    await queryClient.invalidateQueries({ queryKey: ['notes'] })
  }, [loadKeywordManagementItems, queryClient])

  const openKeywordManager = React.useCallback(() => {
    keywordManagerReturnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    setKeywordManagerQuery('')
    setKeywordManagerOpen(true)
    setKeywordRenameDraft(null)
    setKeywordMergeDraft(null)
    void loadKeywordManagementItems()
  }, [loadKeywordManagementItems])

  const closeKeywordManager = React.useCallback(() => {
    setKeywordManagerOpen(false)
    setKeywordRenameDraft(null)
    setKeywordMergeDraft(null)
    restoreFocusAfterOverlayClose(keywordManagerReturnFocusRef.current)
  }, [restoreFocusAfterOverlayClose])

  const openKeywordManagerFromPicker = React.useCallback(() => {
    setKeywordPickerOpen(false)
    openKeywordManager()
  }, [openKeywordManager])

  const keywordManagerVisibleItems = React.useMemo(() => {
    const q = keywordManagerQuery.trim().toLowerCase()
    if (!q) return keywordManagerItems
    return keywordManagerItems.filter((item) => item.keyword.toLowerCase().includes(q))
  }, [keywordManagerItems, keywordManagerQuery])

  const keywordMergeTargetOptions = React.useMemo(() => {
    if (!keywordMergeDraft) return [] as KeywordManagementItem[]
    return keywordManagerItems
      .filter((item) => item.id !== keywordMergeDraft.source.id)
      .sort((a, b) => a.keyword.localeCompare(b.keyword))
  }, [keywordManagerItems, keywordMergeDraft])

  const getRequestStatusCode = React.useCallback((error: any): number => {
    const raw = Number(error?.status ?? error?.response?.status)
    return Number.isFinite(raw) ? Math.floor(raw) : 0
  }, [])

  const handleKeywordManagerDelete = React.useCallback(
    async (item: KeywordManagementItem) => {
      if (keywordManagerActionLoading) return
      const ok = await confirmDanger({
        title: 'Delete tag?',
        content: `Delete tag "${item.keyword}"?`,
        okText: 'Delete',
        cancelText: 'Cancel'
      })
      if (!ok) return

      setKeywordManagerActionLoading(true)
      try {
        await bgRequest({
          path: `/api/v1/notes/keywords/${item.id}` as any,
          method: 'DELETE' as any,
          headers: {
            'expected-version': String(item.version)
          }
        })
        message.success(`Deleted tag "${item.keyword}"`)
        await refreshKeywordDataAfterManagement()
      } catch (error: any) {
        const statusCode = getRequestStatusCode(error)
        if (statusCode === 404 || statusCode === 409) {
          message.warning('Tag changed on the server. Reloaded tag list.')
          await loadKeywordManagementItems()
          return
        }
        message.error(String(error?.message || 'Could not delete tag'))
      } finally {
        setKeywordManagerActionLoading(false)
      }
    },
    [
      confirmDanger,
      getRequestStatusCode,
      keywordManagerActionLoading,
      loadKeywordManagementItems,
      message,
      refreshKeywordDataAfterManagement
    ]
  )

  const submitKeywordRename = React.useCallback(async () => {
    if (!keywordRenameDraft || keywordManagerActionLoading) return
    const nextKeyword = keywordRenameDraft.nextKeyword.trim()
    if (!nextKeyword) {
      message.warning('Enter a tag name')
      return
    }
    if (nextKeyword.toLowerCase() === keywordRenameDraft.currentKeyword.toLowerCase()) {
      setKeywordRenameDraft(null)
      return
    }

    setKeywordManagerActionLoading(true)
    try {
      const renamed = await bgRequest<any>({
        path: `/api/v1/notes/keywords/${keywordRenameDraft.id}` as any,
        method: 'PATCH' as any,
        headers: {
          'Content-Type': 'application/json',
          'expected-version': String(keywordRenameDraft.expectedVersion)
        },
        body: {
          keyword: nextKeyword
        }
      })
      setKeywordRenameDraft(null)
      message.success(`Renamed tag to "${String(renamed?.keyword || nextKeyword)}"`)
      await refreshKeywordDataAfterManagement()
    } catch (error: any) {
      const statusCode = getRequestStatusCode(error)
      if (statusCode === 404 || statusCode === 409) {
        message.warning('Tag rename conflict. Reloaded tag list.')
        setKeywordRenameDraft(null)
        await loadKeywordManagementItems()
        return
      }
      message.error(String(error?.message || 'Could not rename tag'))
    } finally {
      setKeywordManagerActionLoading(false)
    }
  }, [
    getRequestStatusCode,
    keywordManagerActionLoading,
    keywordRenameDraft,
    loadKeywordManagementItems,
    message,
    refreshKeywordDataAfterManagement
  ])

  const submitKeywordMerge = React.useCallback(async () => {
    if (!keywordMergeDraft || keywordManagerActionLoading) return
    if (keywordMergeDraft.targetKeywordId == null) {
      message.warning('Select a target tag')
      return
    }

    const target = keywordMergeTargetOptions.find(
      (item) => item.id === keywordMergeDraft.targetKeywordId
    )
    if (!target) {
      message.warning('Target tag no longer exists. Reloaded tag list.')
      await loadKeywordManagementItems()
      return
    }

    setKeywordManagerActionLoading(true)
    try {
      await bgRequest({
        path: `/api/v1/notes/keywords/${keywordMergeDraft.source.id}/merge` as any,
        method: 'POST' as any,
        headers: {
          'Content-Type': 'application/json',
          'expected-version': String(keywordMergeDraft.source.version)
        },
        body: {
          target_keyword_id: target.id,
          expected_target_version: target.version
        }
      })
      setKeywordMergeDraft(null)
      message.success(`Merged "${keywordMergeDraft.source.keyword}" into "${target.keyword}"`)
      await refreshKeywordDataAfterManagement()
    } catch (error: any) {
      const statusCode = getRequestStatusCode(error)
      if (statusCode === 404 || statusCode === 409) {
        message.warning('Tag merge conflict. Reloaded tag list.')
        setKeywordMergeDraft(null)
        await loadKeywordManagementItems()
        return
      }
      message.error(String(error?.message || 'Could not merge tags'))
    } finally {
      setKeywordManagerActionLoading(false)
    }
  }, [
    getRequestStatusCode,
    keywordManagerActionLoading,
    keywordMergeDraft,
    keywordMergeTargetOptions,
    loadKeywordManagementItems,
    message,
    refreshKeywordDataAfterManagement
  ])

  const openKeywordPicker = React.useCallback(() => {
    keywordPickerReturnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    setKeywordPickerQuery('')
    setKeywordPickerSelection(keywordTokens)
    setKeywordPickerOpen(true)
    void loadAllKeywords()
  }, [keywordTokens, loadAllKeywords])

  const closeKeywordSuggestionModal = React.useCallback(() => {
    setKeywordSuggestionOptions([])
    setKeywordSuggestionSelection([])
    restoreFocusAfterOverlayClose(keywordSuggestionReturnFocusRef.current)
  }, [restoreFocusAfterOverlayClose])

  const applySelectedSuggestedKeywords = React.useCallback(() => {
    const selectedSuggestions = keywordSuggestionSelection
      .map((keyword) => String(keyword || '').trim())
      .filter(Boolean)
    if (selectedSuggestions.length === 0) {
      message.warning(
        t('option:notesSearch.assistKeywordsSelectAtLeastOne', {
          defaultValue: 'Select at least one suggested tag to apply.'
        })
      )
      return
    }

    const merged: string[] = []
    const seen = new Set<string>()
    const append = (value: string) => {
      const text = String(value || '').trim()
      if (!text) return
      const key = text.toLowerCase()
      if (seen.has(key)) return
      seen.add(key)
      merged.push(text)
    }
    editorKeywords.forEach(append)
    selectedSuggestions.forEach(append)

    setEditorKeywords(merged)
    setIsDirty(true)
    setSaveIndicator('dirty')
    setMonitoringNotice(null)
    markGeneratedEdit('suggest_keywords')
    rememberRecentKeywords(selectedSuggestions)
    closeKeywordSuggestionModal()
    message.success(
      t('option:notesSearch.assistKeywordsApplied', {
        defaultValue: 'Applied suggested tags.'
      })
    )
  }, [
    closeKeywordSuggestionModal,
    editorKeywords,
    keywordSuggestionSelection,
    markGeneratedEdit,
    message,
    rememberRecentKeywords,
    setIsDirty,
    setMonitoringNotice,
    setSaveIndicator,
    t
  ])

  const loadKeywordSuggestions = React.useCallback(async (text?: string) => {
    try {
      if (text && text.trim().length > 0) {
        const arr = await searchNoteKeywords(text, 10)
        setKeywordOptions(arr)
      } else if (allKeywords.length > 0) {
        setKeywordOptions(allKeywords)
      } else {
        setKeywordOptions([])
      }
    } catch {
      // Keyword load failed - feature will use empty suggestions
      console.debug('[NotesManagerPage] Keyword suggestions load failed')
    }
  }, [allKeywords])

  const debouncedLoadKeywordSuggestions = React.useCallback(
    (text?: string) => {
      if (typeof window === 'undefined') {
        void loadKeywordSuggestions(text)
        return
      }
      if (keywordSearchTimeoutRef.current != null) {
        window.clearTimeout(keywordSearchTimeoutRef.current)
      }
      keywordSearchTimeoutRef.current = window.setTimeout(() => {
        void loadKeywordSuggestions(text)
      }, 300)
    },
    [loadKeywordSuggestions]
  )

  const handleKeywordFilterSearch = React.useCallback(
    (text: string) => {
      if (isOnline) void debouncedLoadKeywordSuggestions(text)
    },
    [debouncedLoadKeywordSuggestions, isOnline]
  )

  const handleKeywordFilterChange = React.useCallback(
    (vals: string[] | string) => {
      const nextValues = Array.isArray(vals) ? vals : [vals]
      setKeywordTokens(nextValues)
      rememberRecentKeywords(nextValues)
      setPage(1)
    },
    [rememberRecentKeywords, setPage]
  )

  return {
    // state
    keywordTokens, setKeywordTokens,
    keywordOptions, setKeywordOptions,
    allKeywords, allKeywordsRef,
    keywordNoteCountByKey, setKeywordNoteCountByKey,
    editorKeywords, setEditorKeywords,
    keywordPickerOpen, setKeywordPickerOpen,
    keywordPickerQuery, setKeywordPickerQuery,
    keywordPickerSelection, setKeywordPickerSelection,
    keywordPickerSortMode, setKeywordPickerSortMode,
    recentKeywordHistory,
    keywordManagerOpen,
    keywordManagerLoading,
    keywordManagerQuery, setKeywordManagerQuery,
    keywordManagerItems,
    keywordRenameDraft, setKeywordRenameDraft,
    keywordMergeDraft, setKeywordMergeDraft,
    keywordManagerActionLoading,
    keywordSuggestionOptions, setKeywordSuggestionOptions,
    keywordSuggestionSelection, setKeywordSuggestionSelection,
    // refs
    keywordPickerReturnFocusRef, keywordManagerReturnFocusRef, keywordSuggestionReturnFocusRef,
    keywordSearchTimeoutRef,
    // computed
    availableKeywords,
    filteredKeywordPickerOptions, sortedKeywordPickerOptions, recentKeywordPickerOptions,
    maxKeywordNoteCount, getKeywordFrequencyTone, renderKeywordLabelWithFrequency,
    keywordManagerVisibleItems, keywordMergeTargetOptions,
    // callbacks
    rememberRecentKeywords,
    loadAllKeywords, loadKeywordManagementItems,
    refreshKeywordDataAfterManagement,
    openKeywordManager, closeKeywordManager, openKeywordManagerFromPicker,
    openKeywordPicker,
    handleKeywordManagerDelete, submitKeywordRename, submitKeywordMerge,
    loadKeywordSuggestions, debouncedLoadKeywordSuggestions,
    closeKeywordSuggestionModal, applySelectedSuggestedKeywords,
    handleKeywordFilterSearch, handleKeywordFilterChange,
  }
}
