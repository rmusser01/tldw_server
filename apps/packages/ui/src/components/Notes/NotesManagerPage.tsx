import React from 'react'
import { Input, Typography, Button } from 'antd'
import {
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { bgRequest } from '@/services/background-proxy'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useServerOnline } from '@/hooks/useServerOnline'
import { useConfirmDanger } from '@/components/Common/confirm-danger'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useDemoMode } from '@/context/demo-mode'
import { useServerCapabilities } from '@/hooks/useServerCapabilities'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { useAntdMessage } from '@/hooks/useAntdMessage'
import { useStoreMessageOption } from "@/store/option"
import { shallow } from "zustand/shallow"
import { updatePageTitle } from "@/utils/update-page-title"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import NotesEditorPane from "@/components/Notes/NotesEditorPane"
import NotesStudioCreateModal from "@/components/Notes/NotesStudioCreateModal"
import NotesSidebar from "@/components/Notes/NotesSidebar"
import {
  useNotesKeywords,
  useNotesListManagement,
  useNotesEditorState,
  useNotesExport,
  useNotesImport,
  useNotesWikilinks,
} from "@/components/Notes/hooks"
import type { NoteListItem } from "@/components/Notes/notes-manager-types"
import { clearSetting, getSetting } from "@/services/settings/registry"
import { buildFlashcardsGenerateRoute } from "@/services/tldw/flashcards-generate-handoff"
import { buildStudyPackRoute } from "@/services/tldw/study-pack-handoff"
import { deriveNoteStudio, getNoteStudioState, regenerateNoteStudio } from "@/services/notes-studio"
import { useMobile } from "@/hooks/useMediaQuery"
import {
  LAST_NOTE_ID_SETTING,
} from "@/services/settings/ui-settings"

import type {
  NotesAssistAction,
  MarkdownToolbarAction,
  NotesInputMode,
  NotesTocEntry,
  KeywordPickerSortMode,
} from './notes-manager-types'
import type {
  NoteStudioState,
  NotesStudioHandwritingMode,
  NotesStudioPaperSize,
  NotesStudioTemplateType,
} from './notes-studio-types'
import { getDefaultStudioPaperSizeFromLocale } from './export-utils'
import {
  normalizeConversationId,
  toConversationLabel,
  toAttachmentMarkdown,
  normalizeGraphNoteId,
  parseSourceNodeId,
  MIN_SIDEBAR_HEIGHT,
  NOTES_LIST_REGION_ID,
  NOTES_EDITOR_REGION_ID,
  NOTES_SHORTCUTS_SUMMARY_ID,
  shouldIgnoreGlobalShortcut,
  calculateSidebarHeight,
  NOTE_TEMPLATES,
  toSortableTimestamp,
  toNoteVersion,
  markdownToWysiwygHtml,
  wysiwygHtmlToMarkdown,
  LARGE_NOTES_PAGINATION_THRESHOLD,
  TRASH_LOOKUP_PAGE_SIZE,
  TRASH_LOOKUP_MAX_PAGES,
  sortNotesByPinnedIds,
} from './notes-manager-utils'

const LazyNotesManagerOverlays = React.lazy(() => import("./NotesManagerOverlays"))

const isMissingConversationLookupError = (error: unknown): boolean => {
  if (!error) return false
  const status =
    typeof error === 'object' && 'status' in error ? Number((error as { status?: unknown }).status) : Number.NaN
  if (status === 404) return true
  const code =
    typeof error === 'object' && 'code' in error ? String((error as { code?: unknown }).code || '') : ''
  if (code.toUpperCase() === 'NOT_FOUND') return true
  const message = error instanceof Error ? error.message : String(error)
  return /\b(not found|404)\b/i.test(message)
}

const UUID_LIKE_CONVERSATION_ID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

const shouldAutoResolveConversationLabel = (conversationId: string): boolean =>
  !UUID_LIKE_CONVERSATION_ID.test(conversationId)

const CONVERSATION_LABEL_MAX_RETRIES = 3
const CONVERSATION_LABEL_RETRY_DELAY_MS = 1500

const NotesManagerPage: React.FC = () => {
  const { t } = useTranslation(['option', 'common'])
  const isOnline = useServerOnline()
  const isMobileViewport = useMobile()
  const { demoEnabled } = useDemoMode()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const message = useAntdMessage()
  const confirmDanger = useConfirmDanger()
  const {
    setHistory,
    setMessages,
    setHistoryId,
    setServerChatId,
    setServerChatState,
    setServerChatTopic,
    setServerChatClusterId,
    setServerChatSource,
    setServerChatExternalRef
  } = useStoreMessageOption(
    (state) => ({
      setHistory: state.setHistory,
      setMessages: state.setMessages,
      setHistoryId: state.setHistoryId,
      setServerChatId: state.setServerChatId,
      setServerChatState: state.setServerChatState,
      setServerChatTopic: state.setServerChatTopic,
      setServerChatClusterId: state.setServerChatClusterId,
      setServerChatSource: state.setServerChatSource,
      setServerChatExternalRef: state.setServerChatExternalRef
    }),
    shallow
  )

  const capabilityDisabled = !capsLoading && capabilities && !capabilities.hasNotes
  const editorDisabled = Boolean(capabilityDisabled)

  const [mobileSidebarOpen, setMobileSidebarOpen] = React.useState(false)
  const [shortcutHelpOpen, setShortcutHelpOpen] = React.useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false)
  const desktopSidebarCollapsedRef = React.useRef(false)
  const [sidebarHeight, setSidebarHeight] = React.useState(calculateSidebarHeight())
  const [conversationLabelById, setConversationLabelById] = React.useState<Record<string, string>>({})
  const conversationLabelByIdRef = React.useRef<Record<string, string>>({})
  conversationLabelByIdRef.current = conversationLabelById
  const pendingConversationLabelRequestsRef = React.useRef<Set<string>>(new Set())
  const missingConversationLabelIdsRef = React.useRef<Set<string>>(new Set())
  const conversationLabelRetryAttemptsRef = React.useRef<Record<string, number>>({})
  const conversationLabelRetryTimeoutRef = React.useRef<number | null>(null)
  const [conversationLabelRetryTick, setConversationLabelRetryTick] = React.useState(0)

  // ---- Notebook keyword tokens (needed before list hook) ----
  // We compute this after list hook provides selectedNotebook

  // ---- List hook ----
  // We need keywordTokens from keyword hook, but keyword hook needs listMode from list hook.
  // Solution: manage keywordTokens at this level.

  // Temporary: we need to break the circular dep by keeping keywordTokens here
  const [keywordTokensLocal, setKeywordTokensLocal] = React.useState<string[]>([])

  // Placeholder for notebookKeywordTokens (computed after list hook)
  const notebookKeywordTokensRef = React.useRef<string[]>([])

  // Refs for editor state setters needed by keyword hook (editor hook initialized later)
  const setIsDirtyRef = React.useRef<React.Dispatch<React.SetStateAction<boolean>>>(() => {})
  const setSaveIndicatorRef = React.useRef<(state: any) => void>(() => {})
  const setMonitoringNoticeRef = React.useRef<(notice: any) => void>(() => {})
  const markGeneratedEditRef = React.useRef<(action: NotesAssistAction) => void>(() => {})

  const list = useNotesListManagement({
    isOnline,
    message,
    confirmDanger,
    queryClient,
    t,
    keywordTokens: keywordTokensLocal,
    setKeywordTokens: setKeywordTokensLocal,
    notebookKeywordTokens: notebookKeywordTokensRef.current,
  })

  // Compute notebook keyword tokens from list hook's selectedNotebook
  const notebookKeywordTokens = React.useMemo(
    () =>
      list.selectedNotebook == null
        ? []
        : list.selectedNotebook.keywords
            .map((keyword) => String(keyword || '').trim().toLowerCase())
            .filter((keyword) => keyword.length > 0),
    [list.selectedNotebook]
  )
  notebookKeywordTokensRef.current = notebookKeywordTokens

  // ---- Keyword hook ----
  const kw = useNotesKeywords({
    isOnline,
    listMode: list.listMode,
    message,
    confirmDanger,
    restoreFocusAfterOverlayClose: (target) => {
      if (!target) return
      window.requestAnimationFrame(() => {
        if (target.isConnected) target.focus()
      })
    },
    notebookKeywordTokens,
    queryClient,
    t,
    setPage: list.setPage,
    setIsDirty: (val) => setIsDirtyRef.current(val),
    setSaveIndicator: (val) => setSaveIndicatorRef.current(val),
    setMonitoringNotice: (val) => setMonitoringNoticeRef.current(val),
    markGeneratedEdit: (val) => markGeneratedEditRef.current(val),
  })

  // Sync keywordTokens between kw hook and local state
  React.useEffect(() => {
    setKeywordTokensLocal(kw.keywordTokens)
  }, [kw.keywordTokens])

  // ---- Editor hook (use actual deps now that list and kw are available) ----
  const ed = useNotesEditorState({
    isOnline,
    isMobileViewport,
    message,
    confirmDanger,
    queryClient,
    t,
    listMode: list.listMode,
    setListMode: list.setListMode,
    data: list.data,
    refetch: list.refetch,
    setPage: list.setPage,
    setQuery: list.setQuery,
    setQueryInput: list.setQueryInput,
    setKeywordTokens: kw.setKeywordTokens,
    setSelectedNotebookId: list.setSelectedNotebookId,
    setMobileSidebarOpen,
    editorKeywords: kw.editorKeywords,
    setEditorKeywords: kw.setEditorKeywords,
    keywordSuggestionReturnFocusRef: kw.keywordSuggestionReturnFocusRef,
    setKeywordSuggestionOptions: kw.setKeywordSuggestionOptions,
    setKeywordSuggestionSelection: kw.setKeywordSuggestionSelection,
    editorDisabled,
  })

  const [notesStudioCreateOpen, setNotesStudioCreateOpen] = React.useState(false)
  const [notesStudioCreateLoading, setNotesStudioCreateLoading] = React.useState(false)
  const [notesStudioMarkdownOnlyNoticeOpen, setNotesStudioMarkdownOnlyNoticeOpen] = React.useState(false)
  const [notesStudioExcerptText, setNotesStudioExcerptText] = React.useState('')
  const [notesStudioTemplateType, setNotesStudioTemplateType] =
    React.useState<NotesStudioTemplateType>('lined')
  const [notesStudioHandwritingMode, setNotesStudioHandwritingMode] =
    React.useState<NotesStudioHandwritingMode>('accented')
  const [selectedStudioState, setSelectedStudioState] = React.useState<NoteStudioState | null>(null)
  const [notesStudioRegenerating, setNotesStudioRegenerating] = React.useState(false)
  const defaultStudioPaperSize = React.useMemo(
    () => getDefaultStudioPaperSizeFromLocale(typeof navigator !== 'undefined' ? navigator.language : ''),
    []
  )
  const [selectedStudioPaperSize, setSelectedStudioPaperSize] =
    React.useState<NotesStudioPaperSize>(defaultStudioPaperSize)

  // Wire the keyword hook's cross-cutting refs now that editor is available
  setIsDirtyRef.current = ed.setIsDirty
  setSaveIndicatorRef.current = ed.setSaveIndicator
  setMonitoringNoticeRef.current = ed.setMonitoringNotice
  markGeneratedEditRef.current = ed.markGeneratedEdit

  // ---- Visible notes (applies pin sorting from editor on top of list data) ----
  const visibleNotes = React.useMemo(() => {
    if (!Array.isArray(list.data)) return []
    if (list.listMode !== 'active') return list.data
    if (list.listViewMode === 'moodboard') return list.data
    return sortNotesByPinnedIds(list.data, ed.pinnedNoteIdSet)
  }, [list.data, list.listMode, list.listViewMode, ed.pinnedNoteIdSet])

  const filteredCount = visibleNotes.length
  const orderedVisibleNoteIds = React.useMemo(
    () => visibleNotes.map((note) => String(note.id)),
    [visibleNotes]
  )

  const selectedStudioSummaryNoteId = ed.selectedStudioSummary?.note_id ?? null
  const currentNoteStudioSummary =
    ed.selectedId != null &&
    selectedStudioSummaryNoteId != null &&
    String(selectedStudioSummaryNoteId) === String(ed.selectedId)
      ? ed.selectedStudioSummary
      : null

  React.useEffect(() => {
    if (ed.selectedId == null || !currentNoteStudioSummary?.note_id) {
      setSelectedStudioState(null)
      return
    }

    let cancelled = false
    void getNoteStudioState(String(ed.selectedId))
      .then((studioState) => {
        if (cancelled) return
        setSelectedStudioState(studioState)
        ed.setEditorMode('preview')
      })
      .catch(() => {
        if (cancelled) return
        setSelectedStudioState(null)
      })

    return () => {
      cancelled = true
    }
  }, [currentNoteStudioSummary?.note_id, ed.selectedId])

  const effectiveStudioState = React.useMemo(() => {
    if (!selectedStudioState || ed.selectedId == null) return null
    if (String(selectedStudioState.note.id) !== String(ed.selectedId)) {
      return null
    }
    if (String(ed.content || '') === String(selectedStudioState.note.content || '')) {
      return selectedStudioState
    }
    return {
      ...selectedStudioState,
      is_stale: true,
      stale_reason: selectedStudioState.stale_reason || 'companion_content_hash_mismatch'
    }
  }, [ed.content, ed.selectedId, selectedStudioState])

  React.useEffect(() => {
    setNotesStudioMarkdownOnlyNoticeOpen(false)
    setSelectedStudioPaperSize(defaultStudioPaperSize)
  }, [defaultStudioPaperSize, ed.selectedId])

  // ---- Note graph neighbors ----
  const {
    data: noteNeighborsData,
    isLoading: noteNeighborsLoading,
    isError: noteNeighborsError
  } = useQuery({
    queryKey: ['note-graph-neighbors', ed.selectedId, ed.graphMutationTick],
    enabled: isOnline && ed.selectedId != null,
    queryFn: async () => {
      const noteId = encodeURIComponent(String(ed.selectedId))
      const graph = await bgRequest<any>({
        path: `/api/v1/notes/${noteId}/neighbors?edge_types=manual,wikilink,backlink,source_membership&max_nodes=80&max_edges=200` as any,
        method: 'GET' as any
      })
      return graph
    }
  })

  const noteRelations = React.useMemo(() => {
    const selectedNormalized = normalizeGraphNoteId(ed.selectedId)
    const nodes = Array.isArray(noteNeighborsData?.nodes) ? noteNeighborsData.nodes : []
    const edges = Array.isArray(noteNeighborsData?.edges) ? noteNeighborsData.edges : []

    const noteNodeMap = new Map<string, { id: string; title: string }>()
    const sourceNodeMap = new Map<string, { id: string; label: string }>()
    for (const node of nodes) {
      const nodeType = String(node?.type || '')
      if (!nodeType || nodeType === 'note') {
        const normalizedId = normalizeGraphNoteId(node?.id)
        if (!normalizedId) continue
        noteNodeMap.set(normalizedId, {
          id: normalizedId,
          title: String(node?.label || node?.title || `Note ${normalizedId}`)
        })
        continue
      }
      if (nodeType === 'source') {
        const sourceId = String(node?.id || '').trim()
        if (!sourceId) continue
        sourceNodeMap.set(sourceId, {
          id: sourceId,
          label: String(node?.label || sourceId)
        })
      }
    }

    const relatedIds = new Set<string>()
    const backlinkIds = new Set<string>()
    const sourceIds = new Set<string>()
    const manualLinkByEdgeId = new Map<
      string,
      { edgeId: string; noteId: string; title: string; directed: boolean; outgoing: boolean }
    >()

    for (const edge of edges) {
      const source = normalizeGraphNoteId(edge?.source)
      const target = normalizeGraphNoteId(edge?.target)
      if (!source || !target || source === target) continue

      if (source === selectedNormalized) relatedIds.add(target)
      if (target === selectedNormalized) relatedIds.add(source)

      const type = String(edge?.type || '').toLowerCase()
      const directed = Boolean(edge?.directed)
      if (type === 'wikilink' && target === selectedNormalized) {
        backlinkIds.add(source)
      }
      if (type === 'backlink' && source === selectedNormalized) {
        backlinkIds.add(target)
      }
      if (type === 'manual' && directed && target === selectedNormalized) {
        backlinkIds.add(source)
      }
      if (type === 'source_membership') {
        if (source === selectedNormalized && sourceNodeMap.has(target)) {
          sourceIds.add(target)
        }
        if (target === selectedNormalized && sourceNodeMap.has(source)) {
          sourceIds.add(source)
        }
      }
      if (type === 'manual') {
        const touchesSelected = source === selectedNormalized || target === selectedNormalized
        if (!touchesSelected) continue
        const counterpartId = source === selectedNormalized ? target : source
        if (!counterpartId) continue
        const edgeId = String(edge?.id || '')
        if (!edgeId) continue
        const node = noteNodeMap.get(counterpartId)
        manualLinkByEdgeId.set(edgeId, {
          edgeId,
          noteId: counterpartId,
          title: node?.title || `Note ${counterpartId}`,
          directed,
          outgoing: source === selectedNormalized
        })
      }
    }

    for (const normalizedId of noteNodeMap.keys()) {
      if (normalizedId !== selectedNormalized) {
        relatedIds.add(normalizedId)
      }
    }

    const toItems = (ids: Set<string>) =>
      Array.from(ids)
        .filter((id) => id !== selectedNormalized)
        .map((id) => {
          const node = noteNodeMap.get(id)
          return {
            id,
            title: node?.title || `Note ${id}`
          }
        })
        .sort((a, b) => a.title.localeCompare(b.title))

    const sourceItems = Array.from(sourceIds)
      .map((id) => {
        const sourceNode = sourceNodeMap.get(id)
        if (!sourceNode) return null
        return {
          id: sourceNode.id,
          label: sourceNode.label
        }
      })
      .filter(
        (item): item is { id: string; label: string } => item != null
      )
      .sort((a, b) => a.label.localeCompare(b.label))

    return {
      related: toItems(relatedIds),
      backlinks: toItems(backlinkIds),
      manualLinks: Array.from(manualLinkByEdgeId.values()).sort((a, b) =>
        a.title.localeCompare(b.title)
      ),
      sources: sourceItems
    }
  }, [noteNeighborsData, ed.selectedId])

  const manualLinkOptions = React.useMemo(() => {
    const selectedNormalized = normalizeGraphNoteId(ed.selectedId)
    const seen = new Set<string>()
    const options: Array<{ value: string; label: string }> = []
    const append = (id: string, noteTitle: string) => {
      const normalized = normalizeGraphNoteId(id)
      if (!normalized || normalized === selectedNormalized) return
      if (seen.has(normalized)) return
      seen.add(normalized)
      options.push({
        value: normalized,
        label: noteTitle || `Note ${normalized}`
      })
    }
    if (Array.isArray(list.data)) {
      for (const item of list.data) {
        append(String(item.id), String(item.title || `Note ${item.id}`))
      }
    }
    for (const item of noteRelations.related) {
      append(item.id, item.title)
    }
    for (const item of noteRelations.backlinks) {
      append(item.id, item.title)
    }
    return options.sort((a, b) => a.label.localeCompare(b.label))
  }, [list.data, noteRelations.backlinks, noteRelations.related, ed.selectedId])

  // ---- Wikilinks hook ----
  const wl = useNotesWikilinks({
    selectedId: ed.selectedId,
    title: ed.title,
    content: ed.content,
    editorCursorIndex: ed.editorCursorIndex,
    setEditorCursorIndex: ed.setEditorCursorIndex,
    editorDisabled,
    editorMode: ed.editorMode,
    contentTextareaRef: ed.contentTextareaRef,
    resizeEditorTextarea: ed.resizeEditorTextarea,
    setContentDirty: ed.setContentDirty,
    data: list.data,
    noteRelations,
  })

  // ---- Export hook ----
  const exp = useNotesExport({
    message,
    confirmDanger,
    t,
    listMode: list.listMode,
    query: list.query,
    effectiveKeywordTokens: list.effectiveKeywordTokens,
    total: list.total,
    filteredCount: filteredCount,
    hasActiveFilters: list.hasActiveFilters,
    selectedBulkNotes: list.selectedBulkNotes,
    fetchFilteredNotesRaw: list.fetchFilteredNotesRaw,
    selectedId: ed.selectedId,
    title: ed.title,
    content: ed.content,
    editorKeywords: kw.editorKeywords,
    selectedStudioState: effectiveStudioState,
    studioPaperSize: selectedStudioPaperSize,
  })

  // ---- Import hook ----
  const imp = useNotesImport({
    isOnline,
    message,
    t,
    listMode: list.listMode,
    refetch: list.refetch,
  })
  const hasDeferredOverlayOpen =
    kw.keywordSuggestionOptions.length > 0 ||
    kw.keywordPickerOpen ||
    kw.keywordManagerOpen ||
    kw.keywordRenameDraft != null ||
    kw.keywordMergeDraft != null ||
    imp.importModalOpen ||
    ed.graphModalOpen ||
    shortcutHelpOpen

  // ---- Remaining logic that stays in the component ----

  // Conversation label resolution
  const backlinkConversationLabel = React.useMemo(() => {
    const id = normalizeConversationId(ed.backlinkConversationId)
    if (!id) return null
    return conversationLabelById[id] || null
  }, [ed.backlinkConversationId, conversationLabelById])

  const conversationIdsToResolve = React.useMemo(() => {
    const ids = new Set<string>()
    const selectedConversationId = normalizeConversationId(ed.backlinkConversationId)
    for (const note of visibleNotes) {
      const normalized = normalizeConversationId(note?.conversation_id)
      if (!normalized) continue
      if (shouldAutoResolveConversationLabel(normalized)) {
        ids.add(normalized)
        continue
      }
      if (selectedConversationId && normalized === selectedConversationId) {
        ids.add(normalized)
      }
    }
    if (selectedConversationId) ids.add(selectedConversationId)
    return Array.from(ids)
  }, [ed.backlinkConversationId, visibleNotes])

  const resolveConversationLabels = React.useCallback(
    async (conversationIds: string[]) => {
      const pending = conversationIds.filter((conversationId) => {
        if (!conversationId) return false
        if (conversationLabelByIdRef.current[conversationId]) return false
        if (pendingConversationLabelRequestsRef.current.has(conversationId)) return false
        if (missingConversationLabelIdsRef.current.has(conversationId)) return false
        return true
      })
      if (pending.length === 0) return
      pending.forEach((conversationId) =>
        pendingConversationLabelRequestsRef.current.add(conversationId)
      )
      try {
        await tldwClient.initialize().catch(() => null)
        const settled = await Promise.allSettled(
          pending.map(async (conversationId) => {
            const chat = await tldwClient.getChat(conversationId)
            const label = toConversationLabel(chat)
            return { conversationId, label }
          })
        )
        setConversationLabelById((current) => {
          let next: Record<string, string> | null = null
          for (const result of settled) {
            if (result.status !== 'fulfilled') continue
            const { conversationId, label } = result.value
            if (!label) continue
            missingConversationLabelIdsRef.current.delete(conversationId)
            if (current[conversationId]) continue
            if (next == null) next = { ...current }
            next[conversationId] = label
          }
          return next ?? current
        })
        const retryableFailures: string[] = []
        settled.forEach((result, index) => {
          const conversationId = pending[index]
          if (!conversationId) return
          if (result.status === 'fulfilled') {
            delete conversationLabelRetryAttemptsRef.current[conversationId]
            return
          }
          if (isMissingConversationLookupError(result.reason)) {
            missingConversationLabelIdsRef.current.add(conversationId)
            delete conversationLabelRetryAttemptsRef.current[conversationId]
            return
          }
          const nextAttempt =
            (conversationLabelRetryAttemptsRef.current[conversationId] ?? 0) + 1
          conversationLabelRetryAttemptsRef.current[conversationId] = nextAttempt
          if (nextAttempt <= CONVERSATION_LABEL_MAX_RETRIES) {
            retryableFailures.push(conversationId)
          }
        })
        if (retryableFailures.length > 0) {
          if (conversationLabelRetryTimeoutRef.current != null) {
            window.clearTimeout(conversationLabelRetryTimeoutRef.current)
          }
          conversationLabelRetryTimeoutRef.current = window.setTimeout(() => {
            conversationLabelRetryTimeoutRef.current = null
            setConversationLabelRetryTick((current) => current + 1)
          }, CONVERSATION_LABEL_RETRY_DELAY_MS)
        }
      } finally {
        pending.forEach((conversationId) =>
          pendingConversationLabelRequestsRef.current.delete(conversationId)
        )
      }
    },
    []
  )

  React.useEffect(
    () => () => {
      if (conversationLabelRetryTimeoutRef.current != null) {
        window.clearTimeout(conversationLabelRetryTimeoutRef.current)
      }
    },
    []
  )

  React.useEffect(() => {
    const activeConversationIds = new Set(conversationIdsToResolve)
    for (const conversationId of Object.keys(conversationLabelRetryAttemptsRef.current)) {
      if (!activeConversationIds.has(conversationId)) {
        delete conversationLabelRetryAttemptsRef.current[conversationId]
      }
    }
    if (
      activeConversationIds.size === 0 &&
      conversationLabelRetryTimeoutRef.current != null
    ) {
      window.clearTimeout(conversationLabelRetryTimeoutRef.current)
      conversationLabelRetryTimeoutRef.current = null
    }
  }, [conversationIdsToResolve])

  React.useEffect(() => {
    if (!isOnline) return
    if (conversationIdsToResolve.length === 0) return
    void resolveConversationLabels(conversationIdsToResolve)
  }, [conversationIdsToResolve, conversationLabelRetryTick, isOnline, resolveConversationLabels])

  // ---- Handlers remaining in component ----

  const handleNewNote = React.useCallback(async (templateId?: string) => {
    const ok = await ed.confirmDiscardIfDirty()
    if (!ok) return
    if (list.listMode !== 'active') list.setListMode('active')
    if (isMobileViewport) setMobileSidebarOpen(false)
    ed.resetEditor()
    const template = NOTE_TEMPLATES.find((entry) => entry.id === templateId)
    if (template) {
      ed.setTitle(template.title)
      ed.setContent(template.content)
      ed.setIsDirty(true)
      ed.setSaveIndicator('dirty')
      message.success(`Applied template: ${template.label}`)
    }
    setTimeout(() => {
      ed.titleInputRef.current?.focus()
    }, 0)
  }, [ed, isMobileViewport, list, message])

  const handleCreateStudyPackFromNote = React.useCallback(() => {
    const selectedNoteId = ed.selectedId
    if (selectedNoteId == null) {
      message.warning(
        t('option:notesSearch.createStudyPackMissingSelection', {
          defaultValue: 'Select a note before creating a study pack.'
        })
      )
      return
    }

    if (ed.isDirty) {
      message.warning(
        t('option:notesSearch.createStudyPackDirty', {
          defaultValue: 'Save this note before creating a study pack.'
        })
      )
      return
    }

    const noteTitle = ed.title.trim()
    if (!noteTitle) {
      message.warning(
        t('option:notesSearch.createStudyPackEmpty', {
          defaultValue: 'Save and title this note before creating a study pack.'
        })
      )
      return
    }

    navigate(
      buildStudyPackRoute({
        title: noteTitle,
        sourceItems: [
          {
            sourceType: 'note',
            sourceId: String(selectedNoteId),
            sourceTitle: noteTitle
          }
        ]
      })
    )
  }, [ed.isDirty, ed.selectedId, ed.title, message, navigate, t])

  const duplicateSelectedNote = React.useCallback(async () => {
    if (editorDisabled) return
    const hasDraft = ed.title.trim().length > 0 || ed.content.trim().length > 0
    if (!hasDraft) {
      message.warning('Add a title or content before duplicating.')
      return
    }

    if (list.listMode !== 'active') list.setListMode('active')
    if (isMobileViewport) setMobileSidebarOpen(false)
    const baseTitle = ed.title.trim() || (ed.selectedId != null ? `Note ${ed.selectedId}` : 'Untitled note')
    const duplicateTitle = /\(copy\)$/i.test(baseTitle) ? baseTitle : `${baseTitle} (Copy)`
    const duplicateContent = ed.content
    const duplicateKeywords = [...kw.editorKeywords]

    ed.resetEditor()
    ed.setTitle(duplicateTitle)
    ed.setContent(duplicateContent)
    kw.setEditorKeywords(duplicateKeywords)
    ed.setIsDirty(true)
    ed.setSaveIndicator('dirty')
    message.success('Created duplicate draft. Save to keep it.')
    setTimeout(() => {
      ed.titleInputRef.current?.focus()
    }, 0)
  }, [
    ed,
    editorDisabled,
    isMobileViewport,
    kw,
    list,
    message,
  ])

  // Manual links
  const createManualLink = React.useCallback(async () => {
    if (ed.manualLinkSaving) return
    if (ed.selectedId == null || !ed.manualLinkTargetId) return
    const fromId = normalizeGraphNoteId(ed.selectedId)
    const toId = normalizeGraphNoteId(ed.manualLinkTargetId)
    if (!fromId || !toId) return
    if (fromId === toId) {
      message.warning('Cannot link a note to itself')
      return
    }
    ed.setManualLinkSaving(true)
    try {
      await bgRequest<any>({
        path: `/api/v1/notes/${encodeURIComponent(fromId)}/links` as any,
        method: 'POST' as any,
        headers: { 'Content-Type': 'application/json' },
        body: {
          to_note_id: toId,
          directed: false,
          weight: 1.0
        }
      })
      message.success('Manual link created')
      ed.setManualLinkTargetId(null)
      ed.setGraphMutationTick((current) => current + 1)
    } catch (error: any) {
      const status = Number(error?.status ?? error?.response?.status)
      if (status === 409) {
        message.warning('Manual link already exists')
      } else {
        message.error(String(error?.message || 'Could not create manual link'))
      }
    } finally {
      ed.setManualLinkSaving(false)
    }
  }, [ed, message])

  const removeManualLink = React.useCallback(
    async (edgeId: string) => {
      if (!edgeId || ed.manualLinkDeletingEdgeId) return
      const ok = await confirmDanger({
        title: 'Remove link?',
        content: 'This removes the manual relationship between these notes.',
        okText: 'Remove',
        cancelText: 'Cancel'
      })
      if (!ok) return
      ed.setManualLinkDeletingEdgeId(edgeId)
      try {
        await bgRequest<any>({
          path: `/api/v1/notes/links/${encodeURIComponent(edgeId)}` as any,
          method: 'DELETE' as any
        })
        message.success('Manual link removed')
        ed.setGraphMutationTick((current) => current + 1)
      } catch (error: any) {
        message.error(String(error?.message || 'Could not remove manual link'))
      } finally {
        ed.setManualLinkDeletingEdgeId(null)
      }
    },
    [confirmDanger, ed, message]
  )

  // Delete note
  const lookupDeletedNoteVersion = React.useCallback(async (noteId: string) => {
    if (list.listMode === 'trash' && Array.isArray(list.data)) {
      const existing = list.data.find((note) => String(note.id) === noteId)
      if (existing?.version != null) {
        return Number(existing.version)
      }
    }
    try {
      let offset = 0
      let pagesChecked = 0
      while (pagesChecked < TRASH_LOOKUP_MAX_PAGES) {
        const res = await bgRequest<any>({
          path: `/api/v1/notes/trash?limit=${TRASH_LOOKUP_PAGE_SIZE}&offset=${offset}` as any,
          method: 'GET' as any
        })
        const items = Array.isArray(res?.items) ? res.items : Array.isArray(res) ? res : []
        const match = items.find((item: any) => String(item?.id) === noteId)
        const version = toNoteVersion(match)
        if (version != null) return version
        if (items.length === 0) break
        const paginationTotalRaw =
          res && typeof res === 'object' ? Number((res as any)?.pagination?.total_items) : NaN
        const hasPaginationTotal = Number.isFinite(paginationTotalRaw)
        offset += items.length
        pagesChecked += 1
        if (hasPaginationTotal && offset >= paginationTotalRaw) break
        if (items.length < TRASH_LOOKUP_PAGE_SIZE) break
      }
      return null
    } catch {
      return null
    }
  }, [list.data, list.listMode])

  const showDeleteUndoToast = React.useCallback(
    (noteId: string) => {
      const toastKey = `notes-delete-${noteId}`
      const handleUndo = async () => {
        if (typeof (message as any).destroy === 'function') {
          ;(message as any).destroy(toastKey)
        }
        try {
          const resolvedVersion = await lookupDeletedNoteVersion(noteId)
          if (resolvedVersion == null) {
            message.warning('Undo unavailable. Open Trash to restore this note.')
            return
          }
          await bgRequest<any>({
            path: `/api/v1/notes/${encodeURIComponent(noteId)}/restore?expected_version=${encodeURIComponent(
              String(resolvedVersion)
            )}` as any,
            method: 'POST' as any
          })
          message.success('Note restored')
          list.setListMode('active')
          list.setPage(1)
          await list.refetch()
          await ed.loadDetail(noteId)
        } catch {
          message.warning('Undo failed. Open Trash to restore this note manually.')
        }
      }

      if (typeof (message as any).open === 'function') {
        ;(message as any).open({
          key: toastKey,
          duration: 10,
          content: (
            <span className="inline-flex items-center gap-2">
              <span>Note deleted</span>
              <Button
                type="link"
                size="small"
                className="!px-0"
                onClick={() => { void handleUndo() }}
                data-testid={`notes-delete-undo-${noteId.replace(/[^a-z0-9_-]/gi, '_')}`}
              >
                Undo
              </Button>
            </span>
          )
        })
        return
      }
      message.success('Note deleted')
    },
    [ed, list, lookupDeletedNoteVersion, message]
  )

  const countInboundDeleteReferences = React.useCallback(async (noteId: string) => {
    try {
      const encodedId = encodeURIComponent(noteId)
      const graph = await bgRequest<any>({
        path: `/api/v1/notes/${encodedId}/neighbors?edge_types=manual,wikilink,backlink&max_nodes=120&max_edges=240` as any,
        method: 'GET' as any
      })
      const edges = Array.isArray(graph?.edges) ? graph.edges : []
      const inbound = new Set<string>()
      for (const edge of edges) {
        const source = normalizeGraphNoteId(edge?.source)
        const target = normalizeGraphNoteId(edge?.target)
        if (!source || !target || source === target) continue
        const type = String(edge?.type || '').toLowerCase()
        const directed = Boolean(edge?.directed)
        if (type === 'wikilink' && target === noteId) { inbound.add(source); continue }
        if (type === 'backlink' && source === noteId) { inbound.add(target); continue }
        if (type === 'manual' && directed && target === noteId) { inbound.add(source) }
      }
      return inbound.size
    } catch {
      return 0
    }
  }, [])

  const deleteNote = async (id?: string | number | null) => {
    const target = id ?? ed.selectedId
    if (target == null) { message.warning('No note selected'); return }
    const targetId = String(target)
    const inboundReferenceCount = await countInboundDeleteReferences(targetId)
    const warningSuffix =
      inboundReferenceCount > 0
        ? `\n\nThis note is referenced by ${inboundReferenceCount} other note${
            inboundReferenceCount === 1 ? '' : 's'
          }. Deleting it will break those links.`
        : ''
    const ok = await confirmDanger({
      title: 'Please confirm',
      content: `Delete this note?${warningSuffix}`,
      okText: 'Delete',
      cancelText: 'Cancel'
    })
    if (!ok) return
    try {
      let expectedVersion: number | null = null
      if (ed.selectedId != null && String(ed.selectedId) === targetId) {
        expectedVersion = ed.selectedVersion
      }
      if (expectedVersion == null && Array.isArray(list.data)) {
        const match = list.data.find((note) => String(note.id) === targetId)
        if (typeof match?.version === 'number') expectedVersion = match.version
      }
      if (expectedVersion == null) {
        try {
          const detail = await bgRequest<any>({ path: `/api/v1/notes/${target}` as any, method: 'GET' as any })
          expectedVersion = toNoteVersion(detail)
        } catch (e: any) {
          message.error(e?.message || 'Could not verify note version')
          return
        }
      }
      if (expectedVersion == null) {
        message.error('Missing version; reload and try again')
        return
      }
      await bgRequest<any>({
        path: `/api/v1/notes/${target}?expected_version=${encodeURIComponent(
          String(expectedVersion)
        )}` as any,
        method: 'DELETE' as any,
        headers: { "expected-version": String(expectedVersion) }
      })
      showDeleteUndoToast(targetId)
      if (ed.selectedId != null && String(ed.selectedId) === targetId) ed.resetEditor()
      await list.refetch()
    } catch (e: any) {
      if (ed.isVersionConflictError(e)) {
        ed.handleVersionConflict(target)
      } else {
        message.error(String(e?.message || '') || 'Operation failed')
      }
    }
  }

  const restoreNote = async (id: string | number, version?: number) => {
    const target = String(id)
    let expectedVersion: number | null =
      typeof version === 'number' && Number.isFinite(version) ? version : null
    if (expectedVersion == null && Array.isArray(list.data)) {
      const match = list.data.find((note) => String(note.id) === target)
      if (typeof match?.version === 'number') expectedVersion = match.version
    }
    if (expectedVersion == null) {
      message.error('Missing version; reload trash and try again')
      return
    }
    try {
      const restored = await bgRequest<any>({
        path: `/api/v1/notes/${encodeURIComponent(target)}/restore?expected_version=${encodeURIComponent(
          String(expectedVersion)
        )}` as any,
        method: 'POST' as any
      })
      message.success('Note restored')
      list.setListMode('active')
      list.setPage(1)
      await list.refetch()
      const restoredId = String(restored?.id || target)
      await ed.loadDetail(restoredId)
    } catch (error: any) {
      if (ed.isVersionConflictError(error)) {
        message.error('Restore conflict. Refresh trash and retry.')
      } else {
        message.error(String(error?.message || 'Could not restore note'))
      }
    }
  }

  // Bulk operations
  const deleteSelectedBulk = React.useCallback(async () => {
    if (list.selectedBulkNotes.length === 0) {
      message.info('No selected notes to delete')
      return
    }
    const okToLeave = await ed.confirmDiscardIfDirty()
    if (!okToLeave) return
    const confirmed = await confirmDanger({
      title: 'Delete selected notes?',
      content: `Delete ${list.selectedBulkNotes.length} selected notes? This moves them to trash.`,
      okText: 'Delete selected',
      cancelText: 'Cancel'
    })
    if (!confirmed) return

    let deleted = 0
    let failed = 0
    const deletedIds = new Set<string>()

    for (const note of list.selectedBulkNotes) {
      const noteId = String(note.id)
      const expectedVersion = await ed.getExpectedVersionForNoteId(noteId)
      if (expectedVersion == null) { failed += 1; continue }
      try {
        await bgRequest<any>({
          path: `/api/v1/notes/${encodeURIComponent(noteId)}?expected_version=${encodeURIComponent(
            String(expectedVersion)
          )}` as any,
          method: 'DELETE' as any,
          headers: { 'expected-version': String(expectedVersion) }
        })
        deleted += 1
        deletedIds.add(noteId)
      } catch { failed += 1 }
    }

    if (deleted > 0) {
      message.success(`Deleted ${deleted} selected note${deleted === 1 ? '' : 's'}`)
      if (ed.selectedId != null && deletedIds.has(String(ed.selectedId))) {
        ed.resetEditor()
      }
      list.setBulkSelectedIds((current) => current.filter((id) => !deletedIds.has(id)))
      await list.refetch()
    }
    if (failed > 0) {
      message.warning(`${failed} selected note${failed === 1 ? '' : 's'} failed to delete`)
    }
  }, [confirmDanger, ed, list, message])

  const assignKeywordsToSelectedBulk = React.useCallback(async () => {
    if (list.selectedBulkNotes.length === 0) {
      message.info('No selected notes to update')
      return
    }
    const okToLeave = await ed.confirmDiscardIfDirty()
    if (!okToLeave) return
    const suggested = kw.keywordTokens.join(', ')
    const rawInput = window.prompt(
      'Assign keywords to selected notes (comma-separated):',
      suggested
    )
    if (rawInput == null) return
    const keywords = rawInput.split(',').map((entry) => entry.trim()).filter(Boolean)
    if (keywords.length === 0) {
      message.warning('Enter at least one keyword to assign')
      return
    }
    const confirmed = await confirmDanger({
      title: 'Apply keywords to selected notes?',
      content: `Apply ${keywords.join(', ')} to ${list.selectedBulkNotes.length} selected notes?`,
      okText: 'Apply keywords',
      cancelText: 'Cancel'
    })
    if (!confirmed) return

    let updated = 0
    let failed = 0
    for (const note of list.selectedBulkNotes) {
      const noteId = String(note.id)
      const expectedVersion = await ed.getExpectedVersionForNoteId(noteId)
      if (expectedVersion == null) { failed += 1; continue }
      try {
        await bgRequest<any>({
          path: `/api/v1/notes/${encodeURIComponent(noteId)}?expected_version=${encodeURIComponent(
            String(expectedVersion)
          )}` as any,
          method: 'PATCH' as any,
          headers: {
            'Content-Type': 'application/json',
            'expected-version': String(expectedVersion)
          },
          body: { keywords }
        })
        updated += 1
      } catch { failed += 1 }
    }

    if (updated > 0) {
      message.success(`Updated keywords on ${updated} selected note${updated === 1 ? '' : 's'}`)
      await list.refetch()
      if (ed.selectedId != null && list.selectedBulkNotes.some((note) => String(note.id) === String(ed.selectedId))) {
        await ed.loadDetail(ed.selectedId)
      }
    }
    if (failed > 0) {
      message.warning(`${failed} selected note${failed === 1 ? '' : 's'} failed keyword update`)
    }
  }, [confirmDanger, ed, kw.keywordTokens, list, message])

  // Editor input handlers
  const handleEditorChange = React.useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const nextContent = event.target.value
      ed.setContentDirty(nextContent)
      ed.setEditorCursorIndex(event.target.selectionStart ?? nextContent.length)
    },
    [ed]
  )

  const handleEditorSelectionUpdate = React.useCallback(
    (event: React.SyntheticEvent<HTMLTextAreaElement>) => {
      const target = event.currentTarget
      ed.setEditorCursorIndex(target.selectionStart ?? target.value.length)
    },
    [ed]
  )

  const applyMarkdownToolbarAction = React.useCallback(
    (action: MarkdownToolbarAction) => {
      if (editorDisabled) return
      if (ed.editorInputMode === 'wysiwyg') {
        const richEditor = ed.richEditorRef.current
        if (!richEditor) return
        richEditor.focus()
        const execute = (command: string, value?: string) => {
          if (typeof document === 'undefined') return
          if (typeof document.execCommand !== 'function') return
          document.execCommand(command, false, value)
        }
        if (action === 'bold') execute('bold')
        else if (action === 'italic') execute('italic')
        else if (action === 'heading') execute('formatBlock', '<h2>')
        else if (action === 'list') execute('insertUnorderedList')
        else if (action === 'link') {
          const href = typeof window !== 'undefined' ? window.prompt('Link URL', 'https://') : 'https://'
          const normalizedHref = String(href || '').trim()
          if (!normalizedHref) return
          execute('createLink', normalizedHref)
        } else if (action === 'code') execute('insertText', '`code`')

        const nextHtml = richEditor.innerHTML
        ed.setWysiwygHtml(nextHtml)
        ed.setWysiwygSessionDirty(true)
        const nextMarkdown = wysiwygHtmlToMarkdown(nextHtml)
        ed.setContentDirty(nextMarkdown)
        ed.setEditorCursorIndex(nextMarkdown.length)
        return
      }
      const textarea = ed.contentTextareaRef.current
      if (!textarea) return
      textarea.focus()
      const start = textarea.selectionStart ?? 0
      const end = textarea.selectionEnd ?? start
      const selected = ed.content.slice(start, end)
      let replacement = selected
      let nextSelectionStart = start
      let nextSelectionEnd = end
      if (action === 'bold') {
        const inner = selected || 'bold text'
        replacement = `**${inner}**`
        nextSelectionStart = start + 2
        nextSelectionEnd = nextSelectionStart + inner.length
      } else if (action === 'italic') {
        const inner = selected || 'italic text'
        replacement = `*${inner}*`
        nextSelectionStart = start + 1
        nextSelectionEnd = nextSelectionStart + inner.length
      } else if (action === 'code') {
        const inner = selected || 'code'
        replacement = `\`${inner}\``
        nextSelectionStart = start + 1
        nextSelectionEnd = nextSelectionStart + inner.length
      } else if (action === 'heading') {
        if (selected) {
          replacement = selected.split('\n').map((line) => (line.startsWith('#') ? line : `# ${line}`)).join('\n')
          nextSelectionStart = start
          nextSelectionEnd = start + replacement.length
        } else {
          replacement = '# Heading'
          nextSelectionStart = start + 2
          nextSelectionEnd = start + replacement.length
        }
      } else if (action === 'list') {
        if (selected) {
          replacement = selected.split('\n').map((line) => {
            const trimmed = line.trim()
            if (!trimmed) return line
            if (trimmed.startsWith('- ')) return line
            return line.replace(trimmed, `- ${trimmed}`)
          }).join('\n')
          nextSelectionStart = start
          nextSelectionEnd = start + replacement.length
        } else {
          replacement = '- List item'
          nextSelectionStart = start + 2
          nextSelectionEnd = start + replacement.length
        }
      } else if (action === 'link') {
        const text = selected || 'link text'
        const url = 'https://'
        replacement = `[${text}](${url})`
        const urlStart = start + replacement.indexOf(url)
        nextSelectionStart = urlStart
        nextSelectionEnd = urlStart + url.length
      }
      const nextContent = `${ed.content.slice(0, start)}${replacement}${ed.content.slice(end)}`
      ed.setContentDirty(nextContent)
      window.requestAnimationFrame(() => {
        const activeTextarea = ed.contentTextareaRef.current
        if (!activeTextarea) return
        activeTextarea.focus()
        activeTextarea.setSelectionRange(nextSelectionStart, nextSelectionEnd)
        ed.setEditorCursorIndex(nextSelectionEnd)
        ed.resizeEditorTextarea()
      })
    },
    [ed, editorDisabled]
  )

  const openAttachmentPicker = React.useCallback(() => {
    if (editorDisabled) return
    if (ed.selectedId == null) {
      message.warning(t('option:notesSearch.attachmentSaveFirstWarning', {
        defaultValue: 'Save this note once before adding attachments.'
      }))
      return
    }
    ed.attachmentInputRef.current?.click()
  }, [ed, editorDisabled, message, t])

  const handleAttachmentInputChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files
      if (!files || files.length === 0) return
      const inputElement = event.target
      if (ed.selectedId == null) { inputElement.value = ''; return }
      const noteId = String(ed.selectedId)
      const selectedFiles = Array.from(files)
      void (async () => {
        const uploadedMarkdown: string[] = []
        let failedUploads = 0
        for (const file of selectedFiles) {
          const formData = new FormData()
          formData.append('file', file)
          try {
            const result = await bgRequest<any>({
              path: `/api/v1/notes/${encodeURIComponent(noteId)}/attachments` as any,
              method: 'POST' as any,
              body: formData
            })
            const attachmentUrl = String(result?.url || '').trim()
            if (!attachmentUrl) { failedUploads += 1; continue }
            const attachmentName = String(result?.file_name || file.name || '').trim() || file.name
            const attachmentContentType = typeof result?.content_type === 'string' ? result.content_type : file.type
            uploadedMarkdown.push(toAttachmentMarkdown(attachmentName, attachmentUrl, attachmentContentType))
          } catch { failedUploads += 1 }
        }
        if (uploadedMarkdown.length === 0) {
          message.error(t('option:notesSearch.attachmentUploadFailed', { defaultValue: 'Attachment upload failed. Please try again.' }))
          return
        }
        const markdown = uploadedMarkdown.join('\n')
        if (ed.editorInputMode === 'wysiwyg') {
          const nextContent = ed.content.trim().length > 0 ? `${ed.content}\n${markdown}` : markdown
          ed.setContentDirty(nextContent)
          ed.setWysiwygHtml(markdownToWysiwygHtml(nextContent))
          ed.setWysiwygSessionDirty(true)
        } else {
          const activeTextarea = ed.contentTextareaRef.current
          if (!activeTextarea) return
          activeTextarea.focus()
          const start = activeTextarea.selectionStart ?? ed.content.length
          const end = activeTextarea.selectionEnd ?? start
          const nextContent = `${ed.content.slice(0, start)}${markdown}${ed.content.slice(end)}`
          ed.setContentDirty(nextContent)
          const cursor = start + markdown.length
          window.requestAnimationFrame(() => {
            const refreshedTextarea = ed.contentTextareaRef.current
            if (!refreshedTextarea) return
            refreshedTextarea.focus()
            refreshedTextarea.setSelectionRange(cursor, cursor)
            ed.setEditorCursorIndex(cursor)
            ed.resizeEditorTextarea()
          })
        }
        if (failedUploads > 0) {
          message.warning(t('option:notesSearch.attachmentUploadPartial', {
            defaultValue: 'Uploaded {{uploaded}} attachment(s); {{failed}} failed.',
            uploaded: uploadedMarkdown.length, failed: failedUploads
          }))
        } else {
          message.success(t('option:notesSearch.attachmentUploadSuccess', {
            defaultValue: 'Uploaded {{count}} attachment(s).', count: uploadedMarkdown.length
          }))
        }
      })().finally(() => { inputElement.value = '' })
    },
    [ed, message, t]
  )

  // WYSIWYG handlers
  const enterWysiwygMode = React.useCallback(() => {
    ed.markdownBeforeWysiwygRef.current = ed.content
    ed.setWysiwygHtml(markdownToWysiwygHtml(ed.content))
    ed.setWysiwygSessionDirty(false)
    ed.setEditorInputMode('wysiwyg')
    ed.setEditorCursorIndex(null)
  }, [ed])

  const exitWysiwygMode = React.useCallback(() => {
    const originalMarkdown = ed.markdownBeforeWysiwygRef.current
    if (!ed.wysiwygSessionDirty && originalMarkdown != null && originalMarkdown !== ed.content) {
      ed.setContent(originalMarkdown)
    }
    ed.markdownBeforeWysiwygRef.current = null
    ed.setEditorInputMode('markdown')
    ed.setWysiwygSessionDirty(false)
  }, [ed])

  const handleEditorInputModeChange = React.useCallback(
    (nextMode: NotesInputMode) => {
      if (nextMode === ed.editorInputMode) return
      if (nextMode === 'wysiwyg') { enterWysiwygMode(); return }
      setNotesStudioMarkdownOnlyNoticeOpen(false)
      exitWysiwygMode()
      window.requestAnimationFrame(() => {
        const textarea = ed.contentTextareaRef.current
        if (!textarea) return
        textarea.focus()
        const cursor = Math.min(ed.content.length, textarea.selectionStart ?? ed.content.length)
        textarea.setSelectionRange(cursor, cursor)
        ed.setEditorCursorIndex(cursor)
      })
    },
    [ed, enterWysiwygMode, exitWysiwygMode]
  )

  const handleWysiwygInput = React.useCallback(
    (event: React.FormEvent<HTMLDivElement>) => {
      const nextHtml = event.currentTarget.innerHTML
      ed.setWysiwygHtml(nextHtml)
      ed.setWysiwygSessionDirty(true)
      const nextMarkdown = wysiwygHtmlToMarkdown(nextHtml)
      ed.setContentDirty(nextMarkdown)
      ed.setEditorCursorIndex(nextMarkdown.length)
    },
    [ed]
  )

  const handleWysiwygPaste = React.useCallback((event: React.ClipboardEvent<HTMLDivElement>) => {
    if (editorDisabled) return
    const plain = event.clipboardData.getData('text/plain')
    if (!plain) return
    event.preventDefault()
    if (typeof document !== 'undefined' && typeof document.execCommand === 'function') {
      document.execCommand('insertText', false, plain)
    }
  }, [editorDisabled])

  // TOC jump
  const handleTocJump = React.useCallback(
    (entry: NotesTocEntry) => {
      if (ed.editorInputMode === 'wysiwyg') {
        const richEditor = ed.richEditorRef.current
        if (!richEditor) return
        const headingMatch = Array.from(
          richEditor.querySelectorAll<HTMLElement>('[data-md-slug]')
        ).find((element) => element.getAttribute('data-md-slug') === entry.id)
        if (headingMatch) {
          headingMatch.scrollIntoView({ block: 'center', behavior: 'smooth' })
          const range = document.createRange()
          range.selectNodeContents(headingMatch)
          range.collapse(true)
          const selection = window.getSelection()
          selection?.removeAllRanges()
          selection?.addRange(range)
        }
        richEditor.focus()
        return
      }
      const focusAtOffset = () => {
        const textarea = ed.contentTextareaRef.current
        if (!textarea) return
        const cursor = Math.max(0, Math.min(entry.offset, ed.content.length))
        textarea.focus()
        textarea.setSelectionRange(cursor, cursor)
        ed.setEditorCursorIndex(cursor)
      }
      if (ed.editorMode === 'preview') {
        ed.setEditorMode('split')
        window.requestAnimationFrame(() => { window.requestAnimationFrame(focusAtOffset) })
        return
      }
      window.requestAnimationFrame(focusAtOffset)
    },
    [ed]
  )

  // Preview link click
  const handlePreviewLinkClick = React.useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const target = event.target as HTMLElement | null
      if (!target) return
      const anchor = target.closest('a')
      if (!(anchor instanceof HTMLAnchorElement)) return
      const href = String(anchor.getAttribute('href') || '')
      if (!href.startsWith('note://')) return
      event.preventDefault()
      const noteId = decodeURIComponent(href.slice('note://'.length))
      if (!noteId) return
      void ed.handleSelectNote(noteId)
    },
    [ed]
  )

  const closeNotesStudioCreateModal = React.useCallback(() => {
    setNotesStudioCreateOpen(false)
    setNotesStudioExcerptText('')
    setNotesStudioTemplateType('lined')
    setNotesStudioHandwritingMode('accented')
  }, [])

  const handleOpenNotesStudio = React.useCallback(() => {
    if (editorDisabled || ed.selectedId == null) return
    if (ed.isDirty) {
      message.warning(t('option:notesSearch.notesStudioSaveFirstWarning', {
        defaultValue: 'Save this note before opening Notes Studio.'
      }))
      return
    }
    if (ed.editorInputMode !== 'markdown') {
      setNotesStudioMarkdownOnlyNoticeOpen(true)
      return
    }
    const textarea = ed.contentTextareaRef.current
    if (!textarea) {
      message.warning(t('option:notesSearch.notesStudioSelectionRequired', {
        defaultValue: 'Select Markdown text before opening Notes Studio.'
      }))
      return
    }
    const start = textarea.selectionStart ?? 0
    const end = textarea.selectionEnd ?? start
    const excerptText = ed.content.slice(start, end)
    if (excerptText.trim().length === 0) {
      message.warning(t('option:notesSearch.notesStudioSelectionRequired', {
        defaultValue: 'Select Markdown text before opening Notes Studio.'
      }))
      return
    }
    setNotesStudioMarkdownOnlyNoticeOpen(false)
    setNotesStudioExcerptText(excerptText)
    setNotesStudioTemplateType('lined')
    setNotesStudioHandwritingMode('accented')
    setNotesStudioCreateOpen(true)
  }, [ed, editorDisabled, message, t])

  const switchNotesStudioToMarkdown = React.useCallback(() => {
    setNotesStudioMarkdownOnlyNoticeOpen(false)
    handleEditorInputModeChange('markdown')
  }, [handleEditorInputModeChange])

  const handleCreateNotesStudio = React.useCallback(async () => {
    if (ed.selectedId == null || !notesStudioExcerptText.trim()) return
    setNotesStudioCreateLoading(true)
    try {
      const studioState = await deriveNoteStudio({
        source_note_id: String(ed.selectedId),
        excerpt_text: notesStudioExcerptText,
        template_type: notesStudioTemplateType,
        handwriting_mode: notesStudioHandwritingMode,
      })
      closeNotesStudioCreateModal()
      await list.refetch()
      const opened = await ed.handleSelectNote(studioState.note.id)
      if (!opened) return
      setSelectedStudioState(studioState)
      ed.setEditorMode('preview')
    } catch (error: any) {
      message.error(
        error?.message || t('option:notesSearch.notesStudioCreateError', {
          defaultValue: 'Failed to create Notes Studio note.'
        })
      )
    } finally {
      setNotesStudioCreateLoading(false)
    }
  }, [
    closeNotesStudioCreateModal,
    ed,
    list,
    message,
    notesStudioExcerptText,
    notesStudioHandwritingMode,
    notesStudioTemplateType,
    t,
  ])

  const handleRegenerateStudioView = React.useCallback(async () => {
    if (ed.selectedId == null) return
    setNotesStudioRegenerating(true)
    try {
      const selectedNoteId = String(ed.selectedId)
      const refreshed = await regenerateNoteStudio(String(ed.selectedId), {
        current_markdown: String(ed.content || ''),
      })
      const detailReloaded = await ed.loadDetail(selectedNoteId)
      if (!detailReloaded) {
        ed.setTitle(String(refreshed.note.title || ''))
        ed.setContent(String(refreshed.note.content || ''))
        ed.setWysiwygHtml(markdownToWysiwygHtml(String(refreshed.note.content || '')))
        ed.setIsDirty(false)
      }
      setSelectedStudioState(refreshed)
      ed.setEditorMode('preview')
    } catch (error: any) {
      message.error(
        error?.message || t('option:notesSearch.notesStudioRegenerateError', {
          defaultValue: 'Failed to regenerate Notes Studio view.'
        })
      )
    } finally {
      setNotesStudioRegenerating(false)
    }
  }, [ed, message, t])

  // Flashcards
  const handleGenerateFlashcardsFromNote = React.useCallback(() => {
    const sourceText = ed.content.trim()
    if (!sourceText) {
      message.warning(t("option:notesSearch.generateFlashcardsEmpty", {
        defaultValue: "Add note content before generating flashcards."
      }))
      return
    }
    navigate(buildFlashcardsGenerateRoute({
      text: sourceText,
      sourceType: "note",
      sourceId: ed.selectedId != null ? String(ed.selectedId) : undefined,
      sourceTitle: ed.title.trim() || undefined,
      conversationId: ed.backlinkConversationId || undefined,
      messageId: ed.backlinkMessageId || undefined
    }))
  }, [ed, message, navigate, t])

  // Open linked conversation
  const openLinkedConversation = async () => {
    const okToLeave = await ed.confirmDiscardIfDirty()
    if (!okToLeave) return
    if (!ed.backlinkConversationId) {
      message.warning(t("option:notesSearch.noLinkedConversation", { defaultValue: "No linked conversation to open." }))
      return
    }
    try {
      ed.setOpeningLinkedChat(true)
      await tldwClient.initialize().catch(() => null)
      const chat = await tldwClient.getChat(ed.backlinkConversationId)
      const resolvedLabel = toConversationLabel(chat)
      if (resolvedLabel) {
        setConversationLabelById((current) =>
          current[ed.backlinkConversationId!] ? current : { ...current, [ed.backlinkConversationId!]: resolvedLabel }
        )
      }
      setHistoryId(null)
      setServerChatId(String(ed.backlinkConversationId))
      setServerChatState((chat as any)?.state ?? (chat as any)?.conversation_state ?? "in-progress")
      setServerChatTopic((chat as any)?.topic_label ?? null)
      setServerChatClusterId((chat as any)?.cluster_id ?? null)
      setServerChatSource((chat as any)?.source ?? null)
      setServerChatExternalRef((chat as any)?.external_ref ?? null)
      let assistantName = "Assistant"
      if ((chat as any)?.character_id != null) {
        try {
          const c = await tldwClient.getCharacter((chat as any)?.character_id)
          assistantName = c?.name || c?.title || c?.slug || assistantName
        } catch {}
      }
      const messages = await tldwClient.listChatMessages(ed.backlinkConversationId, { include_deleted: "false" } as any)
      const historyArr = messages.map((m) => ({ role: normalizeChatRole(m.role), content: m.content }))
      const mappedMessages = messages.map((m) => {
        const createdAt = Date.parse(m.created_at)
        const normalizedRole = normalizeChatRole(m.role)
        return {
          createdAt: Number.isNaN(createdAt) ? undefined : createdAt,
          isBot: normalizedRole === "assistant",
          role: normalizedRole,
          name: normalizedRole === "assistant" ? assistantName : normalizedRole === "system" ? "System" : "You",
          message: m.content,
          sources: [],
          images: [],
          serverMessageId: m.id,
          serverMessageVersion: m.version
        }
      })
      setHistory(historyArr)
      setMessages(mappedMessages)
      updatePageTitle((chat as any)?.title || "")
      navigate("/")
      setTimeout(() => { try { window.dispatchEvent(new CustomEvent("tldw:focus-composer")) } catch {} }, 0)
    } catch (e: any) {
      message.error(e?.message || t("option:notesSearch.openConversationError", { defaultValue: "Failed to open linked conversation." }))
    } finally {
      ed.setOpeningLinkedChat(false)
    }
  }

  const openLinkedSource = React.useCallback(
    (sourceId: string, sourceLabel: string) => {
      const parsed = parseSourceNodeId(sourceId)
      const externalRef = parsed?.externalRef || null
      if (externalRef && /^https?:\/\//i.test(externalRef)) {
        if (typeof window !== 'undefined') window.open(externalRef, '_blank', 'noopener,noreferrer')
        return
      }
      if (externalRef) { navigate(`/media?id=${encodeURIComponent(externalRef)}`); return }
      navigate('/media')
      message.info(t('option:notesSearch.sourceNavigationFallback', {
        defaultValue: 'Opened media library for source "{{label}}".', label: sourceLabel
      }))
    },
    [message, navigate, t]
  )

  // Graph modal
  const openGraphModal = React.useCallback(() => {
    ed.graphModalReturnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    ed.setGraphModalOpen(true)
  }, [ed])

  const closeGraphModal = React.useCallback(() => {
    ed.setGraphModalOpen(false)
    ed.restoreFocusAfterOverlayClose(ed.graphModalReturnFocusRef.current)
  }, [ed])

  // Keyword picker handlers
  const handleKeywordPickerCancel = React.useCallback(() => {
    kw.setKeywordPickerOpen(false)
    ed.restoreFocusAfterOverlayClose(kw.keywordPickerReturnFocusRef.current)
  }, [ed, kw])

  const handleKeywordPickerApply = React.useCallback(() => {
    kw.setKeywordTokens(kw.keywordPickerSelection)
    kw.rememberRecentKeywords(kw.keywordPickerSelection)
    list.setPage(1)
    kw.setKeywordPickerOpen(false)
    ed.restoreFocusAfterOverlayClose(kw.keywordPickerReturnFocusRef.current)
  }, [ed, kw, list])

  const handleKeywordPickerQueryChange = React.useCallback((value: string) => { kw.setKeywordPickerQuery(value) }, [kw])
  const handleKeywordPickerSelectionChange = React.useCallback((vals: string[]) => { kw.setKeywordPickerSelection(vals) }, [kw])
  const handleKeywordPickerSortModeChange = React.useCallback((mode: KeywordPickerSortMode) => { kw.setKeywordPickerSortMode(mode) }, [kw])
  const handleToggleRecentKeyword = React.useCallback((keyword: string) => {
    kw.setKeywordPickerSelection((current) => current.includes(keyword) ? current.filter((entry) => entry !== keyword) : [...current, keyword])
  }, [kw])
  const handleKeywordPickerSelectAll = React.useCallback(() => { kw.setKeywordPickerSelection(kw.availableKeywords) }, [kw])
  const handleKeywordPickerClear = React.useCallback(() => { kw.setKeywordPickerSelection([]) }, [kw])

  // Active filter summary
  const activeFilterSummary = React.useMemo(() => {
    if (!list.hasActiveFilters || list.listMode !== 'active') return null
    const effectiveQuery = list.query.trim() || list.queryInput.trim()
    const details: string[] = []
    if (effectiveQuery) details.push(`${t('option:notesSearch.summaryQueryLabel', { defaultValue: 'Query' })}: "${effectiveQuery}"`)
    if (list.selectedNotebook != null) details.push(`${t('option:notesSearch.summaryNotebookLabel', { defaultValue: 'Smart collection' })}: ${list.selectedNotebook.name}`)
    if (kw.keywordTokens.length > 0) details.push(`${t('option:notesSearch.summaryKeywordsLabel', { defaultValue: 'Keywords' })}: ${kw.keywordTokens.join(', ')}`)
    const countText = `${t('option:notesSearch.summaryShowing', { defaultValue: 'Showing' })} ${filteredCount} ${t('option:notesSearch.summaryOf', { defaultValue: 'of' })} ${list.total} ${t('option:notesSearch.summaryNotes', { defaultValue: 'notes' })}`
    return { countText, detailsText: details.join(' + ') }
  }, [kw.keywordTokens, list, t])

  const showLargeListPaginationHint =
    list.listMode === 'active' && list.listViewMode !== 'moodboard' && list.total >= LARGE_NOTES_PAGINATION_THRESHOLD

  // Timeline sections
  const timelineSections = React.useMemo(() => {
    if (list.listMode !== 'active') return [] as Array<{ key: string; label: string; notes: NoteListItem[] }>
    const bucketMap = new Map<string, NoteListItem[]>()
    for (const note of visibleNotes) {
      const rawTimestamp = toSortableTimestamp(note.updated_at)
      if (rawTimestamp <= 0) {
        const fallback = bucketMap.get('unknown') || []
        fallback.push(note)
        bucketMap.set('unknown', fallback)
        continue
      }
      const parsed = new Date(rawTimestamp)
      const key = `${parsed.getUTCFullYear()}-${String(parsed.getUTCMonth() + 1).padStart(2, '0')}`
      const bucket = bucketMap.get(key) || []
      bucket.push(note)
      bucketMap.set(key, bucket)
    }
    const orderedKeys = Array.from(bucketMap.keys()).sort((a, b) => {
      if (a === 'unknown') return 1
      if (b === 'unknown') return -1
      return b.localeCompare(a)
    })
    return orderedKeys.map((key) => {
      const notesForKey = bucketMap.get(key) || []
      notesForKey.sort((a, b) => toSortableTimestamp(b.updated_at) - toSortableTimestamp(a.updated_at))
      if (key === 'unknown') return { key, label: t('option:notesSearch.timelineUnknownDate', { defaultValue: 'Unknown date' }), notes: notesForKey }
      const monthLabel = `${key}-01T00:00:00.000Z`
      const parsed = new Date(monthLabel)
      const label = Number.isNaN(parsed.getTime()) ? key : parsed.toLocaleDateString(undefined, { month: 'long', year: 'numeric' })
      return { key, label, notes: notesForKey }
    })
  }, [list.listMode, visibleNotes, t])

  // Search tips
  const searchableTips = React.useMemo(() => [
    { id: 'phrase', text: t('option:notesSearch.searchTipPhrase', { defaultValue: 'Use quotes for phrases, e.g. "project roadmap".' }) },
    { id: 'prefix', text: t('option:notesSearch.searchTipPrefix', { defaultValue: 'Use prefix terms (like analy*) for broader matches.' }) },
    { id: 'and', text: t('option:notesSearch.searchTipAnd', { defaultValue: 'Text query + selected keywords are combined with AND.' }) },
    { id: 'in-note', text: t('option:notesSearch.searchTipInNote', { defaultValue: 'To find text inside the open note, use browser Ctrl/Cmd+F.' }) }
  ], [t])

  const filteredSearchTips = React.useMemo(() => {
    const queryLower = list.searchTipsQuery.trim().toLowerCase()
    if (!queryLower) return searchableTips
    return searchableTips.filter((tip) => tip.text.toLowerCase().includes(queryLower))
  }, [list.searchTipsQuery, searchableTips])

  const searchTipsContent = React.useMemo(
    () => (
      <div className="max-w-[300px] space-y-2 text-xs text-text">
        <Input size="small" allowClear placeholder={t('option:notesSearch.searchTipsFilterPlaceholder', { defaultValue: 'Filter tips...' })} value={list.searchTipsQuery} onChange={(event) => list.setSearchTipsQuery(event.target.value)} data-testid="notes-search-tips-filter" />
        <div className="space-y-1">
          {filteredSearchTips.length === 0 ? (
            <Typography.Text type="secondary" className="block text-[11px] text-text-muted" data-testid="notes-search-tips-empty">
              {t('option:notesSearch.searchTipsEmpty', { defaultValue: 'No matching tips.' })}
            </Typography.Text>
          ) : filteredSearchTips.map((tip) => (<div key={tip.id} data-testid={`notes-search-tip-${tip.id}`}>{tip.text}</div>))}
        </div>
      </div>
    ),
    [filteredSearchTips, list.searchTipsQuery, list.setSearchTipsQuery, t]
  )

  // Deep-link support
  const [pendingNoteId, setPendingNoteId] = React.useState<string | null>(null)

  React.useEffect(() => {
    let cancelled = false
    void (async () => {
      const lastNoteId = await getSetting(LAST_NOTE_ID_SETTING)
      if (!cancelled && lastNoteId) setPendingNoteId(lastNoteId)
    })()
    return () => { cancelled = true }
  }, [])

  React.useEffect(() => {
    if (!isOnline || list.listMode !== 'active' || !pendingNoteId || !Array.isArray(list.data) || ed.selectedId != null) return
    let cancelled = false
    ;(async () => {
      const opened = await ed.handleSelectNote(pendingNoteId)
      if (cancelled) return
      if (!opened) return
      setPendingNoteId(null)
      void clearSetting(LAST_NOTE_ID_SETTING)
    })()
    return () => { cancelled = true }
  }, [list.data, ed, isOnline, list.listMode, pendingNoteId])

  // Sidebar layout
  React.useEffect(() => {
    if (typeof window === 'undefined') return
    const handleResize = () => setSidebarHeight(calculateSidebarHeight())
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  React.useEffect(() => {
    if (!isMobileViewport) desktopSidebarCollapsedRef.current = sidebarCollapsed
  }, [isMobileViewport, sidebarCollapsed])

  React.useEffect(() => {
    if (isMobileViewport) { setSidebarCollapsed(true); setMobileSidebarOpen(false); return }
    setMobileSidebarOpen(false)
    setSidebarCollapsed(desktopSidebarCollapsedRef.current)
  }, [isMobileViewport])

  // Shortcut help
  React.useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.key !== '?' || event.metaKey || event.ctrlKey || event.altKey) return
      if (shouldIgnoreGlobalShortcut(event.target)) return
      event.preventDefault()
      setShortcutHelpOpen(true)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // Cleanup
  React.useEffect(() => {
    return () => {
      list.clearSearchQueryTimeout()
      if (kw.keywordSearchTimeoutRef.current != null) clearTimeout(kw.keywordSearchTimeoutRef.current)
      ed.clearAutosaveTimeout()
    }
  }, [ed, kw, list])

  const handleSkipLinkActivate = React.useCallback(
    (targetId: string) => (event: React.MouseEvent<HTMLAnchorElement>) => {
      event.preventDefault()
      const sidebarHidden = isMobileViewport ? !mobileSidebarOpen : sidebarCollapsed
      if (targetId === NOTES_LIST_REGION_ID && sidebarHidden) {
        if (isMobileViewport) setMobileSidebarOpen(true)
        else setSidebarCollapsed(false)
      }
      window.requestAnimationFrame(() => {
        const target = document.getElementById(targetId) as HTMLElement | null
        if (!target) return
        target.focus()
        if (typeof window !== 'undefined') window.location.hash = targetId
      })
    },
    [isMobileViewport, mobileSidebarOpen, sidebarCollapsed]
  )

  return (
    <div className="relative flex h-full w-full bg-bg p-2 sm:p-4 mt-16">
      <a href={`#${NOTES_LIST_REGION_ID}`} onClick={handleSkipLinkActivate(NOTES_LIST_REGION_ID)} className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:border focus:border-border focus:bg-surface focus:px-3 focus:py-2 focus:text-sm focus:text-text focus:shadow">
        {t('option:notesSearch.skipToNotesList', { defaultValue: 'Skip to notes list' })}
      </a>
      <a href={`#${NOTES_EDITOR_REGION_ID}`} onClick={handleSkipLinkActivate(NOTES_EDITOR_REGION_ID)} className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-12 focus:z-50 focus:rounded-md focus:border focus:border-border focus:bg-surface focus:px-3 focus:py-2 focus:text-sm focus:text-text focus:shadow">
        {t('option:notesSearch.skipToEditor', { defaultValue: 'Skip to editor' })}
      </a>
      <p id={NOTES_SHORTCUTS_SUMMARY_ID} className="sr-only">
        {t('option:notesSearch.shortcutSummaryText', { defaultValue: 'Keyboard shortcuts: Ctrl or Command plus S to save, question mark to open keyboard shortcuts help, Escape to close dialogs.' })}
      </p>
      <div className="absolute right-4 top-4 z-20">
        <Button
          type="primary"
          size="small"
          onClick={handleCreateStudyPackFromNote}
          disabled={ed.selectedId == null || ed.isDirty || !ed.title.trim()}
          data-testid="notes-create-study-pack-button"
        >
          {t('option:notesSearch.createStudyPack', { defaultValue: 'Create study pack' })}
        </Button>
      </div>
      {isMobileViewport && mobileSidebarOpen && (
        <button type="button" aria-label={t('option:notesSearch.closeMobileSidebar', { defaultValue: 'Close notes list' })} data-testid="notes-mobile-sidebar-backdrop" className="absolute inset-0 z-30 bg-black/35" onClick={() => setMobileSidebarOpen(false)} />
      )}
      <NotesSidebar
        isMobileViewport={isMobileViewport}
        mobileSidebarOpen={mobileSidebarOpen}
        sidebarCollapsed={sidebarCollapsed}
        sidebarHeight={sidebarHeight}
        listMode={list.listMode}
        listViewMode={list.listViewMode}
        page={list.page}
        pageSize={list.pageSize}
        total={list.total}
        sortOption={list.sortOption}
        selectedId={ed.selectedId}
        visibleNotes={visibleNotes}
        filteredCount={filteredCount}
        timelineSections={timelineSections}
        recentNotes={ed.recentNotes}
        pinnedNoteIds={ed.pinnedNoteIds}
        pinnedNoteIdSet={ed.pinnedNoteIdSet}
        queryInput={list.queryInput}
        hasActiveFilters={list.hasActiveFilters}
        activeFilterSummary={activeFilterSummary}
        keywordTokens={kw.keywordTokens}
        keywordOptions={kw.keywordOptions}
        availableKeywords={kw.availableKeywords}
        notebookOptions={list.notebookOptions}
        selectedNotebookId={list.selectedNotebookId}
        selectedNotebook={list.selectedNotebook}
        moodboards={list.moodboards}
        selectedMoodboardId={list.selectedMoodboardId}
        selectedMoodboard={list.selectedMoodboard}
        isMoodboardsFetching={list.isMoodboardsFetching}
        moodboardTotalPages={list.moodboardTotalPages}
        moodboardCanGoPrev={list.moodboardCanGoPrev}
        moodboardCanGoNext={list.moodboardCanGoNext}
        moodboardRangeStart={list.moodboardRangeStart}
        moodboardRangeEnd={list.moodboardRangeEnd}
        bulkSelectedIds={list.bulkSelectedIds}
        searchTipsContent={searchTipsContent}
        query={list.query}
        isFetching={list.isFetching}
        isOnline={isOnline}
        demoEnabled={demoEnabled}
        capsLoading={capsLoading}
        capabilities={capabilities || null}
        queuedOfflineDraftCount={ed.queuedOfflineDraftCount}
        showLargeListPaginationHint={showLargeListPaginationHint}
        conversationLabelById={conversationLabelById}
        importSubmitting={imp.importSubmitting}
        exportProgress={exp.exportProgress}
        setMobileSidebarOpen={setMobileSidebarOpen}
        setListViewMode={list.setListViewMode}
        setPage={list.setPage}
        setPageSize={list.setPageSize}
        setSortOption={list.setSortOption}
        setQueryInput={list.setQueryInput}
        setSelectedMoodboardId={list.setSelectedMoodboardId}
        setSelectedNotebookId={list.setSelectedNotebookId}
        setSearchTipsQuery={list.setSearchTipsQuery}
        handleNewNote={handleNewNote}
        switchListMode={ed.switchListMode}
        handleSelectNote={ed.handleSelectNote}
        handleClearFilters={list.handleClearFilters}
        handleKeywordFilterSearch={kw.handleKeywordFilterSearch}
        handleKeywordFilterChange={kw.handleKeywordFilterChange}
        handleToggleBulkSelection={list.handleToggleBulkSelection}
        clearSearchQueryTimeout={list.clearSearchQueryTimeout}
        setQuery={list.setQuery}
        openKeywordPicker={kw.openKeywordPicker}
        createNotebookFromCurrentKeywords={list.createNotebookFromCurrentKeywords}
        removeSelectedNotebook={list.removeSelectedNotebook}
        createMoodboard={list.createMoodboard}
        renameMoodboard={list.renameMoodboard}
        deleteMoodboard={list.deleteMoodboard}
        clearBulkSelection={list.clearBulkSelection}
        exportSelectedBulk={exp.exportSelectedBulk}
        assignKeywordsToSelectedBulk={assignKeywordsToSelectedBulk}
        deleteSelectedBulk={deleteSelectedBulk}
        toggleNotePinned={ed.toggleNotePinned}
        restoreNote={restoreNote}
        exportAll={exp.exportAll}
        exportAllCSV={exp.exportAllCSV}
        exportAllJSON={exp.exportAllJSON}
        openImportPicker={imp.openImportPicker}
        resetEditor={ed.resetEditor}
        renderKeywordLabelWithFrequency={kw.renderKeywordLabelWithFrequency}
        onOpenSettings={() => navigate('/settings/tldw')}
        onOpenHealth={() => navigate('/settings/health')}
      />
      {!isMobileViewport && (
        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="relative w-6 bg-surface border-y border-r border-border hover:bg-surface2 flex items-center justify-center group transition-colors rounded-r-lg"
          style={{ minHeight: `${MIN_SIDEBAR_HEIGHT}px`, height: `${sidebarHeight}px` }}
          aria-label={sidebarCollapsed ? t('option:notesSearch.expandSidebar', { defaultValue: 'Expand sidebar' }) : t('option:notesSearch.collapseSidebar', { defaultValue: 'Collapse sidebar' })}
          data-testid="notes-desktop-sidebar-toggle"
        >
          <div className="flex items-center justify-center w-full h-full">
            {sidebarCollapsed ? <ChevronRight className="w-4 h-4 text-text-subtle group-hover:text-text" /> : <ChevronLeft className="w-4 h-4 text-text-subtle group-hover:text-text" />}
          </div>
        </button>
      )}
      <NotesEditorPane
        isMobileViewport={isMobileViewport}
        setMobileSidebarOpen={setMobileSidebarOpen}
        selectedId={ed.selectedId}
        title={ed.title}
        content={ed.content}
        editorDisabled={editorDisabled}
        isDirty={ed.isDirty}
        isOnline={isOnline}
        loadingDetail={ed.loadingDetail}
        saving={ed.saving}
        backlinkConversationId={ed.backlinkConversationId}
        backlinkConversationLabel={backlinkConversationLabel}
        backlinkMessageId={ed.backlinkMessageId}
        noteRelations={noteRelations}
        noteNeighborsLoading={noteNeighborsLoading}
        noteNeighborsError={noteNeighborsError}
        selectedNotePinned={ed.selectedNotePinned}
        editorMode={ed.editorMode}
        editorInputMode={ed.editorInputMode}
        openingLinkedChat={ed.openingLinkedChat}
        editorKeywords={kw.editorKeywords}
        keywordOptions={kw.keywordOptions}
        saveIndicator={ed.saveIndicator}
        saveIndicatorText={ed.saveIndicatorText}
        selectedLastSavedAt={ed.selectedLastSavedAt}
        offlineStatusText={ed.offlineStatusText}
        currentOfflineDraft={ed.currentOfflineDraft}
        remoteVersionInfo={ed.remoteVersionInfo}
        monitoringNotice={ed.monitoringNotice}
        monitoringNoticeClasses={ed.monitoringNoticeClasses}
        titleSuggestionLoading={ed.titleSuggestionLoading}
        canSwitchTitleStrategy={ed.canSwitchTitleStrategy}
        effectiveTitleSuggestStrategy={ed.effectiveTitleSuggestStrategy}
        titleStrategyOptions={ed.titleStrategyOptions}
        studioBadgeLabel={effectiveStudioState || currentNoteStudioSummary ? t('option:notesSearch.notesStudioAction', {
          defaultValue: 'Notes Studio'
        }) : null}
        showStudioMarkdownOnlyNotice={notesStudioMarkdownOnlyNoticeOpen}
        selectedStudioState={effectiveStudioState}
        studioPaperSize={selectedStudioPaperSize}
        onStudioPaperSizeChange={setSelectedStudioPaperSize}
        onRegenerateStudioView={() => {
          void handleRegenerateStudioView()
        }}
        studioRegenerating={notesStudioRegenerating}
        setTitleSuggestStrategy={ed.setTitleSuggestStrategy}
        manualLinkTargetId={ed.manualLinkTargetId}
        setManualLinkTargetId={ed.setManualLinkTargetId}
        manualLinkSaving={ed.manualLinkSaving}
        manualLinkOptions={manualLinkOptions}
        manualLinkDeletingEdgeId={ed.manualLinkDeletingEdgeId}
        assistLoadingAction={ed.assistLoadingAction}
        shouldShowToc={wl.shouldShowToc}
        tocEntries={wl.tocEntries}
        previewContent={wl.previewContent}
        usesLargePreviewGuardrails={wl.usesLargePreviewGuardrails}
        largePreviewReady={wl.largePreviewReady}
        wysiwygHtml={ed.wysiwygHtml}
        activeWikilinkQuery={wl.activeWikilinkQuery}
        wikilinkSuggestions={wl.wikilinkSuggestions}
        wikilinkSuggestionDisplayCounts={wl.wikilinkSuggestionDisplayCounts}
        wikilinkSelectionIndex={wl.wikilinkSelectionIndex}
        metricSummaryText={ed.metricSummaryText}
        revisionSummaryText={ed.revisionSummaryText}
        provenanceSummaryText={ed.provenanceSummaryText}
        queuedOfflineDraftCount={ed.queuedOfflineDraftCount}
        titleInputRef={ed.titleInputRef}
        contentTextareaRef={ed.contentTextareaRef}
        richEditorRef={ed.richEditorRef}
        attachmentInputRef={ed.attachmentInputRef}
        setTitle={ed.setTitle}
        setIsDirty={ed.setIsDirty}
        setSaveIndicator={ed.setSaveIndicator}
        setMonitoringNotice={ed.setMonitoringNotice}
        setEditorMode={ed.setEditorMode}
        setEditorKeywords={kw.setEditorKeywords}
        setEditorCursorIndex={ed.setEditorCursorIndex}
        setShortcutHelpOpen={setShortcutHelpOpen}
        markManualEdit={ed.markManualEdit}
        suggestTitle={ed.suggestTitle}
        openLinkedConversation={openLinkedConversation}
        openLinkedSource={openLinkedSource}
        handleNewNote={handleNewNote}
        duplicateSelectedNote={duplicateSelectedNote}
        toggleNotePinned={ed.toggleNotePinned}
        copySelected={exp.copySelected}
        handleGenerateFlashcardsFromNote={handleGenerateFlashcardsFromNote}
        handleCreateStudyPackFromNote={handleCreateStudyPackFromNote}
        handleOpenNotesStudio={handleOpenNotesStudio}
        exportSelected={exp.exportSelected}
        saveNote={ed.saveNote}
        deleteNote={deleteNote}
        handleSelectNote={ed.handleSelectNote}
        openGraphModal={openGraphModal}
        createManualLink={createManualLink}
        removeManualLink={removeManualLink}
        debouncedLoadKeywordSuggestions={kw.debouncedLoadKeywordSuggestions}
        renderKeywordLabelWithFrequency={kw.renderKeywordLabelWithFrequency}
        handleEditorInputModeChange={handleEditorInputModeChange}
        applyMarkdownToolbarAction={applyMarkdownToolbarAction}
        openAttachmentPicker={openAttachmentPicker}
        handleAttachmentInputChange={handleAttachmentInputChange}
        runAssistAction={ed.runAssistAction}
        switchStudioNoticeToMarkdown={switchNotesStudioToMarkdown}
        dismissStudioMarkdownOnlyNotice={() => setNotesStudioMarkdownOnlyNoticeOpen(false)}
        handleTocJump={handleTocJump}
        handlePreviewLinkClick={handlePreviewLinkClick}
        handleWysiwygInput={handleWysiwygInput}
        handleWysiwygPaste={handleWysiwygPaste}
        handleEditorChange={handleEditorChange}
        handleEditorKeyDown={wl.handleEditorKeyDown}
        handleEditorSelectionUpdate={handleEditorSelectionUpdate}
        applyWikilinkSuggestion={wl.applyWikilinkSuggestion}
      />
      <NotesStudioCreateModal
        open={notesStudioCreateOpen}
        excerptText={notesStudioExcerptText}
        templateType={notesStudioTemplateType}
        handwritingMode={notesStudioHandwritingMode}
        loading={notesStudioCreateLoading}
        onClose={closeNotesStudioCreateModal}
        onTemplateChange={setNotesStudioTemplateType}
        onHandwritingChange={setNotesStudioHandwritingMode}
        onSubmit={() => {
          void handleCreateNotesStudio()
        }}
      />
      <input ref={imp.importInputRef} type="file" multiple accept=".json,.md,.markdown,application/json,text/markdown,text/plain" className="hidden" data-testid="notes-import-input" onChange={(event) => { void imp.handleImportInputChange(event) }} />
      {hasDeferredOverlayOpen && (
        <React.Suspense fallback={null}>
          <LazyNotesManagerOverlays
            kw={kw}
            imp={imp}
            graph={{
              graphModalOpen: ed.graphModalOpen,
              selectedId: ed.selectedId,
              graphMutationTick: ed.graphMutationTick,
            }}
            isOnline={isOnline}
            shortcutHelpOpen={shortcutHelpOpen}
            setShortcutHelpOpen={setShortcutHelpOpen}
            closeGraphModal={closeGraphModal}
            handleSelectNote={ed.handleSelectNote}
            handleKeywordPickerCancel={handleKeywordPickerCancel}
            handleKeywordPickerApply={handleKeywordPickerApply}
            handleKeywordPickerSortModeChange={handleKeywordPickerSortModeChange}
            handleToggleRecentKeyword={handleToggleRecentKeyword}
            handleKeywordPickerQueryChange={handleKeywordPickerQueryChange}
            handleKeywordPickerSelectionChange={handleKeywordPickerSelectionChange}
            handleKeywordPickerSelectAll={handleKeywordPickerSelectAll}
            handleKeywordPickerClear={handleKeywordPickerClear}
            t={t}
          />
        </React.Suspense>
      )}
    </div>
  )
}

export default NotesManagerPage
