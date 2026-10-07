import {
  readKnowledgeNoteProvenance,
  retainKnowledgeNoteProvenance,
  stripKnowledgeNoteProvenance
} from '@/utils/knowledge-note-provenance'
import React from 'react'
import type { InputRef } from 'antd'
import { Button, Modal } from 'antd'
import type { MessageInstance } from 'antd/es/message/interface'
import type { QueryClient } from '@tanstack/react-query'
import { useQuery } from '@tanstack/react-query'
import { bgRequest, type BgRequestInit } from '@/services/background-proxy'
import { useCallerCapabilities } from '@/hooks/useCallerCapabilities'
import { tldwAuth } from '@/services/tldw/TldwAuth'
import type { ServicePromptTargetConfig, TldwConfig } from '@/services/tldw/TldwApiClient'
import { connectionAuthoritiesMatch, deriveSingleUserApiKeyCredentialScope } from '@/services/chat-surface-scope'
import {
  listNoteTasks,
  listTaskActivity,
  markTaskActivityRead,
  setNoteTaskStatus,
  type NoteTask,
  type NoteTaskActivityEvent,
  type NoteTaskReconciliationSummary,
} from '@/services/notes-tasks'
import { getSetting, setSetting, clearSetting } from '@/services/settings/registry'
import type { TaskChecklistTogglePayload } from '@/components/Notes/TaskChecklistPreview'
import { toggleChecklistItemMarker } from '@/components/Notes/task-markdown'
import {
  LAST_NOTE_ID_SETTING,
  NOTES_RECENT_OPENED_SETTING,
  NOTES_PINNED_IDS_SETTING,
  NOTES_TITLE_SUGGEST_STRATEGY_SETTING,
  type NotesRecentOpenedEntry,
  type NotesTitleSuggestStrategy,
} from '@/services/settings/ui-settings'
import type { ConfirmDangerOptions } from '@/components/Common/confirm-danger'
import type {
  SaveNoteOptions,
  SaveIndicatorState,
  NotesEditorMode,
  NotesInputMode,
  OfflineDraftEntry,
  OfflineDraftSyncResult,
  RemoteVersionInfo,
  NotesAssistAction,
  EditProvenanceState,
  MonitoringAlertSeverity,
  MonitoringNoticeState,
  MarkdownToolbarAction,
  NotesTitleSettingsResponse,
  KeywordSyncWarning,
} from '../notes-manager-types'
import {
  INITIAL_NOTE_SAVE_STATE,
  NOTE_SAVE_OWNER_ERROR_CODE,
  canStartNoteSave,
  classifyNoteSaveError,
  isRetryableNoteSaveFailure,
  nextAutosaveDelayMs,
  noteSaveIssueKind,
  toSaveIndicator,
  transitionNoteSave,
  type NoteSaveEvent,
  type NoteSaveFailure,
  type NoteSaveState,
  type NoteSaveTrigger,
} from '../notes-save-machine'
import {
  extractBacklink,
  extractKeywords,
  toNoteVersion,
  toNoteLastModified,
  toKeywordSyncWarning,
  NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY,
  NOTES_OFFLINE_NEW_DRAFT_KEY,
  isEditorSaveShortcutContext,
  normalizeOfflineDraftQueue,
  NOTES_TITLE_STRATEGIES,
  normalizeNotesTitleStrategy,
  deriveAllowedTitleStrategies,
  markdownToWysiwygHtml,
  wysiwygHtmlToMarkdown,
  EMPTY_WYSIWYG_HTML,
  buildSummaryDraft,
  buildOutlineDraft,
  suggestKeywordsDraft,
  NOTE_ASSIST_STOP_WORDS,
  toAttachmentMarkdown,
} from '../notes-manager-utils'
import type { NoteStudioDocumentSummary } from '../notes-studio-types'
import { buildSingleNoteCopyText } from '../export-utils'
import { notesAuthoritySetting } from '../notes-authority-storage'
import { useNotesAuthorityState } from './useNotesAuthorityState'
import { createNotesGraphAuthorityScope } from './useNotesGraphAuthorityScope'
import type { NoteRenamedEvent } from './useNotesWikilinkRename'

type ConfirmDanger = (options: ConfirmDangerOptions) => Promise<boolean>
type NotesOwnedRequestOptions = {
  servicePromptConfig: ServicePromptTargetConfig
  abortSignal: AbortSignal
  headers: Record<string, string>
}

export interface UseNotesEditorStateDeps {
  authorityScope?: string | null
  connectionConfig?: TldwConfig | null
  isOnline: boolean
  isMobileViewport: boolean
  message: MessageInstance
  confirmDanger: ConfirmDanger
  queryClient: QueryClient
  t: (key: string, opts?: Record<string, any>) => string
  /** From list hook */
  listMode: 'active' | 'trash'
  setListMode: React.Dispatch<React.SetStateAction<'active' | 'trash'>>
  data: any[] | undefined
  refetch: () => Promise<any>
  setPage: React.Dispatch<React.SetStateAction<number>>
  setQuery: React.Dispatch<React.SetStateAction<string>>
  setQueryInput: React.Dispatch<React.SetStateAction<string>>
  setKeywordTokens: React.Dispatch<React.SetStateAction<string[]>>
  setSelectedNotebookId: React.Dispatch<React.SetStateAction<number | null>>
  setMobileSidebarOpen: React.Dispatch<React.SetStateAction<boolean>>
  /** From keyword hook */
  editorKeywords: string[]
  setEditorKeywords: React.Dispatch<React.SetStateAction<string[]>>
  keywordSuggestionReturnFocusRef: React.MutableRefObject<HTMLElement | null>
  setKeywordSuggestionOptions: React.Dispatch<React.SetStateAction<string[]>>
  setKeywordSuggestionSelection: React.Dispatch<React.SetStateAction<string[]>>
  /** Capability */
  editorDisabled: boolean
  /** Called after a save changed an existing note's title on the server. */
  onNoteRenamed?: (event: NoteRenamedEvent) => void
}

const noteResourcePath = (id: string | number) =>
  `/api/v1/notes/${encodeURIComponent(String(id))}`

const toNoteTitle = (note: unknown): string | null => {
  const value = note && typeof note === 'object' ? (note as { title?: unknown }).title : null
  return typeof value === 'string' ? value : null
}

const offlineDraftKeyFor = (noteId: string | number | null | undefined) =>
  noteId == null ? NOTES_OFFLINE_NEW_DRAFT_KEY : `note:${String(noteId)}`

/** How often the offline queue retries once it owns a draft (NS-05). */
const NOTES_OFFLINE_QUEUE_RETRY_MS = 60_000

const browserReportsOffline = () =>
  typeof navigator !== 'undefined' && navigator.onLine === false

const ownerError = (message: string) =>
  Object.assign(new Error(message), { code: NOTE_SAVE_OWNER_ERROR_CODE })

/** The problem the editor's single save-issue surface shows (NS-03, NS-06). */
export type NotesSaveIssue = {
  kind: 'conflict' | 'error' | 'retrying' | 'remote-newer'
  message: string
  /** A save for this issue is in flight right now. */
  busy: boolean
}

/** What the route-level leave guard needs from the editor (NS-01). */
export type NotesLeaveGuardState = {
  /** Unsaved or in-flight edits exist. */
  when: boolean
  /** Flush them; resolve true to continue the navigation, false to stay. */
  onLeave: () => Promise<boolean>
}

/**
 * Copy text to the clipboard, reporting whether it worked. Call it before the
 * first `await` of a click handler: browsers only allow the write while the
 * click's user activation is still active.
 */
const writeClipboardText = async (text: string): Promise<boolean> => {
  try {
    if (typeof navigator === 'undefined' || typeof navigator.clipboard?.writeText !== 'function') {
      return false
    }
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}

export function useNotesEditorState(deps: UseNotesEditorStateDeps) {
  const {
    authorityScope,
    connectionConfig,
    isOnline,
    isMobileViewport,
    message,
    confirmDanger,
    t,
    listMode,
    setListMode,
    data,
    refetch,
    setPage,
    setQuery,
    setQueryInput,
    setKeywordTokens,
    setSelectedNotebookId,
    setMobileSidebarOpen,
    editorKeywords,
    setEditorKeywords,
    keywordSuggestionReturnFocusRef,
    setKeywordSuggestionOptions,
    setKeywordSuggestionSelection,
    editorDisabled,
  } = deps
  const callerCapabilities = useCallerCapabilities()
  const callerCapabilitiesRef = React.useRef(callerCapabilities)
  callerCapabilitiesRef.current = callerCapabilities
  const onNoteRenamedRef = React.useRef(deps.onNoteRenamed)
  onNoteRenamedRef.current = deps.onNoteRenamed
  // The open note's title as the server last returned it, so a save can tell a rename from an edit.
  const savedTitleRef = React.useRef<{ noteId: string; title: string } | null>(null)
  /** A save landed for this owner: announce the rename when the note's saved title changed. */
  const announceSavedTitle = React.useCallback(
    (
      noteId: string | number,
      previousTitle: string | null,
      savedTitle: string | null,
      savedForScope: string | null | undefined
    ) => {
      const id = String(noteId)
      if (savedTitle != null && savedTitleRef.current?.noteId === id) {
        savedTitleRef.current = { noteId: id, title: savedTitle }
      }
      if (savedForScope && previousTitle && savedTitle && previousTitle !== savedTitle) {
        onNoteRenamedRef.current?.({
          noteId: id,
          oldTitle: previousTitle,
          newTitle: savedTitle,
          authorityScope: savedForScope
        })
      }
    },
    []
  )
  const authorityScopeRef = React.useRef(authorityScope)
  const connectionConfigRef = React.useRef(connectionConfig)
  const connectionEpochRef = React.useRef(0)
  const authorityEpochRef = React.useRef(0)
  const authorityRequestsRef = React.useRef(new Set<AbortController>())
  const noteSelectionEpochRef = React.useRef(0)
  const pendingSelectionEpochRef = React.useRef<number | null>(null)
  if (!connectionAuthoritiesMatch(connectionConfig, connectionConfigRef.current)) {
    connectionEpochRef.current += 1
    authorityEpochRef.current += 1
  }
  const connectionEpoch = connectionEpochRef.current
  connectionConfigRef.current = connectionConfig ? { ...connectionConfig } : connectionConfig
  if (authorityScopeRef.current !== authorityScope) authorityEpochRef.current += 1
  authorityScopeRef.current = authorityScope
  // Set below once the save machine exists; tells it an in-flight save was cut off.
  const settleCancelledSaveRef = React.useRef<() => void>(() => {})
  React.useEffect(() => {
    const cancelRequests = () => {
      authorityEpochRef.current += 1
      pendingSelectionEpochRef.current = null
      for (const controller of authorityRequestsRef.current) controller.abort()
      authorityRequestsRef.current.clear()
      activeSaveRef.current = null
      activeSavePromiseRef.current = null
      savingInFlightRef.current = false
      setSaving(false)
      settleCancelledSaveRef.current()
    }
    const configChanged = (event: Event) => {
      const detail = (event as CustomEvent<{ authorityChanged?: boolean; refreshSessionInvalidated?: boolean }>).detail
      if (detail?.authorityChanged !== false || detail.refreshSessionInvalidated) cancelRequests()
    }
    window.addEventListener('tldw:config-updated', configChanged)
    window.addEventListener('tldw:auth-principal-changed', cancelRequests)
    return () => {
      cancelRequests()
      window.removeEventListener('tldw:config-updated', configChanged)
      window.removeEventListener('tldw:auth-principal-changed', cancelRequests)
    }
  }, [authorityScope, connectionEpoch])

  // ---- editor state ----
  const [selectedId, setSelectedId] = React.useState<string | number | null>(null)
  const selectedIdRef = React.useRef(selectedId)
  selectedIdRef.current = selectedId
  const [title, setTitle] = React.useState('')
  const [content, setContent] = React.useState('')
  const setEditorTitle = React.useCallback((value: React.SetStateAction<string>) => {
    if (pendingSelectionEpochRef.current == null || selectedIdRef.current == null) {
      setTitle(value)
    }
  }, [])
  const setEditorContent = React.useCallback((value: React.SetStateAction<string>) => {
    if (pendingSelectionEpochRef.current == null || selectedIdRef.current == null) {
      setContent(value)
    }
  }, [])
  const [loadingDetail, setLoadingDetail] = useNotesAuthorityState(authorityScope, false)
  const [saving, setSaving] = React.useState(false)
  // ---- save state machine (NS-01, NS-03, NS-N2, NS-05, NS-06) ----
  // One source of truth for autosave, retries, conflicts, the offline hand-off
  // and the status pill. Async code reads the ref; renders read the state.
  const [saveState, setSaveStateValue] =
    useNotesAuthorityState<NoteSaveState>(authorityScope, INITIAL_NOTE_SAVE_STATE)
  const saveStateRef = React.useRef(saveState)
  saveStateRef.current = saveState
  const dispatchSave = React.useCallback((event: NoteSaveEvent) => {
    const current = saveStateRef.current
    const next = transitionNoteSave(current, event)
    // Skip equal states: callers mark dirty from effects, and a fresh object
    // per call would re-render (and re-run those effects) forever.
    if (
      next === current ||
      (next.status === current.status &&
        next.retryAttempt === current.retryAttempt &&
        next.failure === current.failure)
    ) {
      return
    }
    saveStateRef.current = next
    setSaveStateValue(next)
  }, [setSaveStateValue])
  const lastEditAtRef = React.useRef(0)
  const dirtySinceRef = React.useRef(0)
  const lastFailureAtRef = React.useRef(0)
  const [originalMetadata, setOriginalMetadata] = React.useState<Record<string, any> | null>(null)
  const [selectedStudioSummary, setSelectedStudioSummary] =
    React.useState<NoteStudioDocumentSummary | null>(null)
  const [selectedVersion, setSelectedVersion] = React.useState<number | null>(null)
  // Saves read the base version from this ref so a save chained right after
  // another (the leave flush) never sends the version from a stale render.
  const selectedVersionRef = React.useRef(selectedVersion)
  selectedVersionRef.current = selectedVersion
  const assignSelectedVersion = React.useCallback((version: number | null) => {
    selectedVersionRef.current = version
    setSelectedVersion(version)
  }, [])
  const [selectedLastSavedAt, setSelectedLastSavedAt] = React.useState<string | null>(null)
  const [isDirty, setIsDirtyState] = React.useState(false)
  const isDirtyRef = React.useRef(isDirty)
  isDirtyRef.current = isDirty
  const dirtyRevisionRef = React.useRef(0)
  /** Set the unsaved-edits flag without counting it as a new edit. */
  const setDirtyFlag = React.useCallback((value: boolean) => {
    isDirtyRef.current = value
    setIsDirtyState(value)
  }, [])
  const setIsDirty = React.useCallback((value: React.SetStateAction<boolean>) => {
    if (value !== false) dirtyRevisionRef.current += 1
    const next = typeof value === 'function' ? value(isDirtyRef.current) : value
    if (next) {
      const now = Date.now()
      if (!isDirtyRef.current) dirtySinceRef.current = now
      lastEditAtRef.current = now
      dispatchSave({ type: 'edited' })
    }
    setDirtyFlag(next)
  }, [dispatchSave, setDirtyFlag])
  // Compatibility for callers that still pair setIsDirty(true) with
  // setSaveIndicator('dirty'); the pill itself is derived from the machine.
  const setSaveIndicator = React.useCallback((value: SaveIndicatorState) => {
    if (value === 'dirty') dispatchSave({ type: 'edited' })
    else if (value === 'idle') dispatchSave({ type: 'loaded', acknowledged: false })
  }, [dispatchSave])
  const saveIndicator = toSaveIndicator(saveState, isDirty)
  // An in-flight save cut off by an account change or unmount.
  const interruptedSaveRef = React.useRef(false)
  settleCancelledSaveRef.current = () => {
    if (saveStateRef.current.status !== 'saving') return
    interruptedSaveRef.current = true
    dispatchSave({ type: 'save-cancelled', hasPendingEdits: isDirtyRef.current })
  }
  const [backlinkConversationId, setBacklinkConversationId] = React.useState<string | null>(null)
  const [backlinkMessageId, setBacklinkMessageId] = React.useState<string | null>(null)
  const [remoteVersionInfo, setRemoteVersionInfo] = React.useState<RemoteVersionInfo | null>(null)
  const [editorMode, setEditorMode] = React.useState<NotesEditorMode>('edit')
  const [editorInputMode, setEditorInputMode] = React.useState<NotesInputMode>('markdown')
  // The WYSIWYG contentEditable is uncontrolled (NE-01, #3102). `wysiwygHtml` is
  // the latest document, from external replacements and from the user's own
  // edits. The editor writes it into the DOM only when `wysiwygRevision` changes
  // (an external replacement) or when the editor node mounts, never on input.
  const [wysiwygHtml, setWysiwygHtmlState] = React.useState<string>(EMPTY_WYSIWYG_HTML)
  const [wysiwygRevision, setWysiwygRevision] = React.useState(0)
  /** Replace the WYSIWYG document for an external reason (load, reload, template, mode switch). */
  const replaceWysiwygHtml = React.useCallback((html: string) => {
    setWysiwygHtmlState(html)
    setWysiwygRevision((current) => current + 1)
  }, [])
  // The user's own WYSIWYG edits, counted. A space at the end of a line or an
  // empty new line leaves the Markdown as it was, yet it is an edit: it is part
  // of the edit snapshot below, so a reload that was already on its way (a new
  // note's first save) does not write the server copy over it (NE-01).
  const [wysiwygInputCount, setWysiwygInputCount] = React.useState(0)
  /** Record HTML the editor already shows (user input or an in-place command); the DOM is left alone. */
  const recordWysiwygEditorHtml = React.useCallback((html: string) => {
    setWysiwygHtmlState(html)
    setWysiwygInputCount((count) => count + 1)
  }, [])
  const [wysiwygSessionDirty, setWysiwygSessionDirty] = React.useState(false)
  const [editorCursorIndex, setEditorCursorIndex] = React.useState<number | null>(null)
  const [titleSuggestionLoading, setTitleSuggestionLoading] = React.useState(false)
  const [assistLoadingAction, setAssistLoadingAction] = React.useState<NotesAssistAction | null>(null)
  const [editProvenance, setEditProvenance] = React.useState<EditProvenanceState>({ mode: 'manual' })
  const [monitoringNotice, setMonitoringNotice] = useNotesAuthorityState<MonitoringNoticeState | null>(authorityScope, null)
  const [noteTasks, setNoteTasks] = React.useState<NoteTask[]>([])
  const [taskReconciliation, setTaskReconciliation] =
    React.useState<NoteTaskReconciliationSummary | null>(null)
  const [taskActivityEvents, setTaskActivityEvents] = React.useState<NoteTaskActivityEvent[]>([])
  const [taskConflictNotice, setTaskConflictNotice] = React.useState<string | null>(null)
  const [recentNotes, setRecentNotes] = useNotesAuthorityState<NotesRecentOpenedEntry[]>(authorityScope, [])
  const recentNotesRef = React.useRef<NotesRecentOpenedEntry[]>([])
  recentNotesRef.current = recentNotes
  const recentRevisionRef = React.useRef(0)
  const [pinnedNoteIds, setPinnedNoteIds] = useNotesAuthorityState<string[]>(authorityScope, [])
  const recentNotesSetting = React.useMemo(() => notesAuthoritySetting(NOTES_RECENT_OPENED_SETTING, authorityScope), [authorityScope])
  const pinnedNotesSetting = React.useMemo(() => notesAuthoritySetting(NOTES_PINNED_IDS_SETTING, authorityScope), [authorityScope])
  const [titleSuggestStrategy, setTitleSuggestStrategy] =
    React.useState<NotesTitleSuggestStrategy>('heuristic')
  const [graphMutationTick, setGraphMutationTick] = React.useState(0)
  const [manualLinkTargetId, setManualLinkTargetId] = React.useState<string | null>(null)
  const [manualLinkSaving, setManualLinkSaving] = React.useState(false)
  const [manualLinkDeletingEdgeId, setManualLinkDeletingEdgeId] = React.useState<string | null>(null)
  const [openingLinkedChat, setOpeningLinkedChat] = React.useState(false)

  // ---- offline draft state ----
  const [offlineDraftQueue, setOfflineDraftQueue] = useNotesAuthorityState<Record<string, OfflineDraftEntry>>(authorityScope, {})
  const [offlineDraftQueueHydrated, setOfflineDraftQueueHydrated] = useNotesAuthorityState(authorityScope, false)
  const offlineDraftStorageKey = authorityScope ? `${NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY}:${authorityScope}` : null
  const offlineDraftQueueRef = React.useRef<Record<string, OfflineDraftEntry>>({})
  offlineDraftQueueRef.current = offlineDraftQueue
  const offlineSyncInFlightRef = React.useRef<number | null>(null)
  const restoredInitialOfflineDraftRef = React.useRef(false)

  // ---- refs ----
  const autosaveTimeoutRef = React.useRef<number | null>(null)
  const saveNoteRef = React.useRef<((opts?: SaveNoteOptions) => Promise<boolean>) | null>(null)
  const savingInFlightRef = React.useRef(false)
  const activeSaveRef = React.useRef<AbortController | null>(null)
  /** The save in flight, so a flush can wait for it before deciding. */
  const activeSavePromiseRef = React.useRef<Promise<boolean> | null>(null)
  /** Edit revision the user chose to discard (skip the unmount backstop). */
  const discardedRevisionRef = React.useRef<number | null>(null)
  const saveEpochRef = React.useRef(0)
  const titleInputRef = React.useRef<InputRef | null>(null)
  const contentTextareaRef = React.useRef<HTMLTextAreaElement | null>(null)
  const contentRef = React.useRef('')
  const richEditorRef = React.useRef<HTMLDivElement | null>(null)
  const attachmentInputRef = React.useRef<HTMLInputElement | null>(null)
  const markdownBeforeWysiwygRef = React.useRef<string | null>(null)
  contentRef.current = content
  const draftSnapshot = { title, content, editorKeywords, originalMetadata, backlinkConversationId, backlinkMessageId, wysiwygInputCount }
  const editRevisionRef = React.useRef({ ...draftSnapshot, revision: 0 })
  if ((Object.keys(draftSnapshot) as Array<keyof typeof draftSnapshot>).some((key) => editRevisionRef.current[key] !== draftSnapshot[key])) {
    editRevisionRef.current = { ...draftSnapshot, revision: editRevisionRef.current.revision + 1 }
  }

  // ---- AI assist undo ----
  const contentBeforeAssistRef = React.useRef<string | null>(null)
  const assistUndoTimerRef = React.useRef<number | null>(null)
  const [canUndoAssist, setCanUndoAssist] = React.useState(false)

  const clearAssistUndoState = React.useCallback(() => {
    if (assistUndoTimerRef.current != null) {
      window.clearTimeout(assistUndoTimerRef.current)
      assistUndoTimerRef.current = null
    }
    contentBeforeAssistRef.current = null
    setCanUndoAssist(false)
  }, [])

  const pinnedNoteIdSet = React.useMemo(() => new Set(pinnedNoteIds), [pinnedNoteIds])

  const clearAutosaveTimeout = React.useCallback(() => {
    if (autosaveTimeoutRef.current != null) {
      window.clearTimeout(autosaveTimeoutRef.current)
      autosaveTimeoutRef.current = null
    }
  }, [])

  const markManualEdit = React.useCallback(() => {
    setEditProvenance((current) => (current.mode === 'manual' ? current : { mode: 'manual' }))
  }, [])

  const markGeneratedEdit = React.useCallback((action: NotesAssistAction) => {
    setEditProvenance({
      mode: 'generated',
      action,
      at: Date.now()
    })
  }, [])

  const restoreFocusAfterOverlayClose = React.useCallback((target: HTMLElement | null) => {
    if (!target) return
    window.requestAnimationFrame(() => {
      if (target.isConnected) {
        target.focus()
      }
    })
  }, [])

  // ---- offline draft helpers ----
  const currentOfflineDraftKey = React.useMemo(() => {
    if (selectedId == null) return NOTES_OFFLINE_NEW_DRAFT_KEY
    return `note:${String(selectedId)}`
  }, [selectedId])

  // Reads the latest editor snapshot from refs, so the unmount and pagehide
  // backstops and a chained leave flush never persist a stale render.
  const buildCurrentOfflineDraft = React.useCallback(
    (
      overrides?: Partial<Pick<OfflineDraftEntry, 'syncState' | 'lastError' | 'updatedAt' | 'baseVersion'>>
    ): OfflineDraftEntry => {
      const nowIso = overrides?.updatedAt || new Date().toISOString()
      const noteId = selectedIdRef.current
      const snapshot = editRevisionRef.current
      return {
        key: offlineDraftKeyFor(noteId),
        noteId: noteId != null ? String(noteId) : null,
        baseVersion:
          overrides?.baseVersion !== undefined
            ? overrides.baseVersion
            : selectedVersionRef.current,
        title: snapshot.title,
        content: retainKnowledgeNoteProvenance(snapshot.content, snapshot.originalMetadata),
        keywords: [...snapshot.editorKeywords],
        metadata: snapshot.originalMetadata ? { ...snapshot.originalMetadata } : null,
        backlinkConversationId: snapshot.backlinkConversationId,
        backlinkMessageId: snapshot.backlinkMessageId,
        updatedAt: nowIso,
        syncState: overrides?.syncState || 'queued',
        lastError: overrides?.lastError ?? null
      }
    },
    []
  )

  const applyOfflineDraftToEditor = React.useCallback((draft: OfflineDraftEntry) => {
    setTitle(String(draft.title || ''))
    setContent(stripKnowledgeNoteProvenance(String(draft.content || '')))
    setEditorKeywords(Array.isArray(draft.keywords) ? [...draft.keywords] : [])
    const provenance = readKnowledgeNoteProvenance(String(draft.content || ''))
    setOriginalMetadata(provenance
      ? { ...(draft.metadata || {}), ...provenance, knowledge_provenance: provenance }
      : draft.metadata && typeof draft.metadata === 'object' ? { ...draft.metadata } : null
    )
    setBacklinkConversationId(draft.backlinkConversationId)
    setBacklinkMessageId(draft.backlinkMessageId)
    if (draft.baseVersion != null) {
      assignSelectedVersion(draft.baseVersion)
    }
    setSelectedLastSavedAt(draft.updatedAt)
    setDirtyFlag(false)
    // The draft lives on this device; the queue (not autosave) syncs it.
    dispatchSave({ type: 'draft-restored', conflict: draft.syncState === 'conflict' })
    setEditProvenance({ mode: 'manual' })
    setMonitoringNotice(null)
    setRemoteVersionInfo(null)
    setEditorCursorIndex(null)
    replaceWysiwygHtml(markdownToWysiwygHtml(String(draft.content || '')))
    setWysiwygSessionDirty(false)
    markdownBeforeWysiwygRef.current = String(draft.content || '')
  }, [assignSelectedVersion, dispatchSave, replaceWysiwygHtml, setDirtyFlag, setEditorKeywords, setMonitoringNotice])

  const upsertOfflineDraft = React.useCallback(
    (
      overrides?: Partial<Pick<OfflineDraftEntry, 'syncState' | 'lastError' | 'updatedAt' | 'baseVersion'>>
    ) => {
      const nextDraft = buildCurrentOfflineDraft(overrides)
      setOfflineDraftQueue((current) => ({
        ...current,
        [nextDraft.key]: nextDraft
      }))
    },
    [buildCurrentOfflineDraft, setOfflineDraftQueue]
  )

  const persistOfflineDraft = React.useCallback(
    (
      overrides?: Partial<Pick<OfflineDraftEntry, 'syncState' | 'lastError' | 'updatedAt' | 'baseVersion'>>
    ) => {
      if (!authorityScope || authorityScopeRef.current !== authorityScope) return false
      if (!offlineDraftStorageKey || !offlineDraftQueueHydrated || typeof window === 'undefined') return false
      const nextDraft = buildCurrentOfflineDraft(overrides)
      const nextQueue = {
        ...offlineDraftQueueRef.current,
        [nextDraft.key]: nextDraft
      }
      try {
        const serializedQueue = JSON.stringify(nextQueue)
        window.localStorage.setItem(offlineDraftStorageKey, serializedQueue)
        if (window.localStorage.getItem(offlineDraftStorageKey) !== serializedQueue) return false
      } catch {
        return false
      }
      offlineDraftQueueRef.current = nextQueue
      setOfflineDraftQueue(nextQueue)
      return true
    },
    [
      authorityScope,
      buildCurrentOfflineDraft,
      offlineDraftQueueHydrated,
      offlineDraftStorageKey,
      setOfflineDraftQueue
    ]
  )

  const removeOfflineDraftByKey = React.useCallback((key: string) => {
    const normalized = String(key || '').trim()
    if (!normalized) return
    setOfflineDraftQueue((current) => {
      if (!current[normalized]) return current
      const next = { ...current }
      delete next[normalized]
      return next
    })
  }, [setOfflineDraftQueue])

  /** Remove a queued draft before the next await (loadDetail re-applies queued drafts). */
  const dropOfflineDraftNow = React.useCallback((key: string) => {
    const current = offlineDraftQueueRef.current
    if (!current[key]) return
    const next = { ...current }
    delete next[key]
    offlineDraftQueueRef.current = next
    setOfflineDraftQueue(next)
  }, [setOfflineDraftQueue])

  const queuedOfflineDraftCount = React.useMemo(() => {
    return Object.values(offlineDraftQueue).filter((draft) => {
      return (
        draft.syncState === 'queued' ||
        draft.syncState === 'syncing' ||
        draft.syncState === 'error' ||
        draft.syncState === 'conflict'
      )
    }).length
  }, [offlineDraftQueue])

  const currentOfflineDraft = offlineDraftQueue[currentOfflineDraftKey] || null

  // ---- content helpers ----
  const resizeEditorTextarea = React.useCallback(() => {
    const textarea = contentTextareaRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    const viewportHeight = typeof window !== 'undefined' ? window.innerHeight : 900
    const maxHeight = Math.max(
      280,
      Math.floor(viewportHeight * (editorMode === 'split' ? 0.42 : 0.62))
    )
    const minHeight = editorMode === 'split' ? 220 : 280
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight)
    textarea.style.height = `${nextHeight}px`
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }, [editorMode])

  const setContentDirty = React.useCallback(
    (
      nextContent: string,
      options?: {
        provenance?: 'manual' | NotesAssistAction
      }
    ) => {
      // Freeze an existing note during a switch; preserve edits to a new draft.
      if (pendingSelectionEpochRef.current != null && selectedIdRef.current != null) {
        return
      }
      contentRef.current = nextContent
      setContent(nextContent)
      setIsDirty(true)
      setMonitoringNotice(null)
      setTaskConflictNotice(null)
      if (options?.provenance && options.provenance !== 'manual') {
        markGeneratedEdit(options.provenance)
      } else {
        markManualEdit()
      }
    },
    [markGeneratedEdit, markManualEdit, setIsDirty, setMonitoringNotice]
  )

  const clearTaskState = React.useCallback(() => {
    setNoteTasks([])
    setTaskReconciliation(null)
    setTaskActivityEvents([])
    setTaskConflictNotice(null)
  }, [])

  const loadTaskActivityForNote = React.useCallback(async (noteId: string | number, ownedRequest?: NotesOwnedRequestOptions) => {
    const requestEpoch = authorityEpochRef.current
    const selectionEpoch = noteSelectionEpochRef.current
    try {
      if (ownedRequest?.abortSignal.aborted) return
      const response = await listTaskActivity({ note_id: noteId, limit: 50 }, ownedRequest)
      if (authorityEpochRef.current !== requestEpoch || noteSelectionEpochRef.current !== selectionEpoch || ownedRequest?.abortSignal.aborted) return
      const noteIdText = String(noteId)
      const events = Array.isArray(response.events)
        ? response.events.filter((event) => String(event.note_id || '') === noteIdText)
        : []
      setTaskActivityEvents(events)
    } catch {
      if (authorityEpochRef.current !== requestEpoch || noteSelectionEpochRef.current !== selectionEpoch || ownedRequest?.abortSignal.aborted) return
      setTaskActivityEvents([])
    }
  }, [])

  const refreshTaskStateForNote = React.useCallback(async (noteId: string | number, ownedRequest?: NotesOwnedRequestOptions) => {
    const requestEpoch = authorityEpochRef.current
    const selectionEpoch = noteSelectionEpochRef.current
    try {
      if (ownedRequest?.abortSignal.aborted) return
      const response = await listNoteTasks(noteId, { limit: 500 }, ownedRequest)
      if (authorityEpochRef.current !== requestEpoch || noteSelectionEpochRef.current !== selectionEpoch || ownedRequest?.abortSignal.aborted) return
      setNoteTasks(Array.isArray(response.tasks) ? response.tasks : [])
      setTaskReconciliation(response.reconciliation ?? null)
    } catch {
      if (authorityEpochRef.current !== requestEpoch || noteSelectionEpochRef.current !== selectionEpoch || ownedRequest?.abortSignal.aborted) return
      setNoteTasks([])
      setTaskReconciliation(null)
    }
    await loadTaskActivityForNote(noteId, ownedRequest)
  }, [loadTaskActivityForNote])

  const toggleTaskCheckboxLocal = React.useCallback(
    (payload: TaskChecklistTogglePayload) => {
      const currentContent = contentRef.current
      const nextContent = toggleChecklistItemMarker(
        currentContent,
        payload.lineNumber,
        payload.nextStatus === 'done'
      )
      if (nextContent === currentContent) return
      setContentDirty(nextContent, { provenance: 'manual' })
    },
    [setContentDirty]
  )

  // ---- load/reset ----
  const updateRecentNotes = React.useCallback(
    (getNext: (current: NotesRecentOpenedEntry[]) => NotesRecentOpenedEntry[]) => {
      if (!recentNotesSetting || authorityScopeRef.current !== authorityScope) return
      const current = recentNotesRef.current
      const next = getNext(current)
      if (next === current) return
      recentNotesRef.current = next
      recentRevisionRef.current += 1
      setRecentNotes(next)
      void setSetting(recentNotesSetting, next).catch(() => {
        // Keep the in-memory recent list even if settings persistence fails.
      })
    },
    [authorityScope, recentNotesSetting, setRecentNotes]
  )

  const rememberRecentNote = React.useCallback((noteId: string | number, noteTitle: string) => {
    const normalizedId = String(noteId || '').trim()
    const normalizedTitle = String(noteTitle || '').trim()
    if (!normalizedId || !normalizedTitle) return
    updateRecentNotes((current) =>
      [
        { id: normalizedId, title: normalizedTitle },
        ...current.filter((entry) => entry.id !== normalizedId)
      ].slice(0, 5)
    )
  }, [updateRecentNotes])

  const removeRecentNotes = React.useCallback((noteIds: Array<string | number>) => {
    const removedIds = new Set(
      noteIds
        .map((noteId) => String(noteId || '').trim())
        .filter(Boolean)
    )
    if (removedIds.size === 0) return
    updateRecentNotes((current) => {
      const next = current.filter((entry) => !removedIds.has(String(entry.id)))
      if (next.length === current.length) return current
      return next
    })
  }, [updateRecentNotes])

  const loadDetail = React.useCallback(async (id: string | number, ownedRequest?: NotesOwnedRequestOptions, savedEditRevision?: number): Promise<boolean> => {    if (activeSaveRef.current && ownedRequest?.abortSignal !== activeSaveRef.current.signal) {
      activeSaveRef.current.abort()
      activeSaveRef.current = null
      activeSavePromiseRef.current = null
      savingInFlightRef.current = false
      setSaving(false)
      if (saveStateRef.current.status === 'saving') {
        dispatchSave({ type: 'save-cancelled', hasPendingEdits: isDirtyRef.current })
      }
    }
    const requestAuthorityScope = authorityScope
    const requestEpoch = authorityEpochRef.current
    const noteEpoch = ++noteSelectionEpochRef.current
    const editRevision = savedEditRevision ?? editRevisionRef.current.revision
    const isCurrent = () => !ownedRequest?.abortSignal.aborted && authorityEpochRef.current === requestEpoch && authorityScopeRef.current === requestAuthorityScope && noteSelectionEpochRef.current === noteEpoch
    if (requestAuthorityScope === null) return false
    pendingSelectionEpochRef.current = String(selectedIdRef.current) !== String(id) ? noteEpoch : null
    clearAssistUndoState()
    setLoadingDetail(true)
    try {
      const d = await bgRequest<any>({ ...ownedRequest, path: noteResourcePath(id) as any, method: 'GET' as any })
      if (!isCurrent() || editRevisionRef.current.revision !== editRevision) return false
      const loadedTitle = String(d?.title || `Note ${id}`)
      savedTitleRef.current = { noteId: String(id), title: String(d?.title || '') }
      selectedIdRef.current = id
      setSelectedId(id)
      setTitle(String(d?.title || ''))
      setContent(stripKnowledgeNoteProvenance(String(d?.content || '')))
      setEditorKeywords(extractKeywords(d))
      assignSelectedVersion(toNoteVersion(d))
      setSelectedLastSavedAt(toNoteLastModified(d))
      const rawMeta = d && typeof d === "object" ? (d as any).metadata : null
      const provenance = readKnowledgeNoteProvenance(String(d?.content || ''))
      setOriginalMetadata(
        provenance
          ? { ...(rawMeta && typeof rawMeta === 'object' ? rawMeta : {}), ...provenance, knowledge_provenance: provenance }
          : rawMeta && typeof rawMeta === "object" ? { ...(rawMeta as Record<string, any>) } : null
      )
      const rawStudio = d && typeof d === 'object' ? (d as any).studio : null
      setSelectedStudioSummary(
        rawStudio && typeof rawStudio === 'object'
          ? { ...(rawStudio as NoteStudioDocumentSummary) }
          : null
      )
      const links = extractBacklink(d)
      setBacklinkConversationId(links.conversation_id)
      setBacklinkMessageId(links.message_id)
      setDirtyFlag(false)
      interruptedSaveRef.current = false
      // A 2xx read of the server copy is an acknowledged state.
      dispatchSave({
        type: 'loaded',
        acknowledged: isOnline && toNoteVersion(d) != null && Boolean(toNoteLastModified(d))
      })
      setEditProvenance({ mode: 'manual' })
      setMonitoringNotice(null)
      clearTaskState()
      setRemoteVersionInfo(null)
      setEditorCursorIndex(0)
      replaceWysiwygHtml(markdownToWysiwygHtml(String(d?.content || '')))
      setWysiwygSessionDirty(false)
      markdownBeforeWysiwygRef.current = String(d?.content || '')
      rememberRecentNote(id, loadedTitle)
      const queuedDraft = offlineDraftQueueRef.current[`note:${String(id)}`]
      if (queuedDraft) {
        applyOfflineDraftToEditor(queuedDraft)
      }
      await refreshTaskStateForNote(id, ownedRequest)
      return isCurrent()
    } catch {
      if (isCurrent()) message.error('Failed to load note')
      return false
    } finally {
      if (pendingSelectionEpochRef.current === noteEpoch) pendingSelectionEpochRef.current = null
      if (isCurrent()) setLoadingDetail(false)
    }
  }, [applyOfflineDraftToEditor, assignSelectedVersion, authorityScope, clearAssistUndoState, clearTaskState, dispatchSave, isOnline, message, refreshTaskStateForNote, rememberRecentNote, replaceWysiwygHtml, setDirtyFlag, setEditorKeywords, setLoadingDetail, setMonitoringNotice])

  const dismissTaskActivity = React.useCallback(async (eventId: string) => {
    const normalizedEventId = String(eventId || '').trim()
    if (!normalizedEventId) return
    try {
      await markTaskActivityRead(normalizedEventId, { dismissed: true })
    } catch {
      // Non-critical notice state. Keep the UI moving if the server already marked it.
    }
    setTaskActivityEvents((current) =>
      current.filter((event) => String(event.id) !== normalizedEventId)
    )
  }, [])

  const inspectTaskActivity = React.useCallback(() => {
    if (selectedId == null) return
    void refreshTaskStateForNote(selectedId)
  }, [refreshTaskStateForNote, selectedId])

  const resetEditor = React.useCallback(() => {
    noteSelectionEpochRef.current += 1
    pendingSelectionEpochRef.current = null
    activeSaveRef.current?.abort()
    activeSaveRef.current = null
    activeSavePromiseRef.current = null
    savingInFlightRef.current = false
    setSaving(false)
    clearAssistUndoState()
    savedTitleRef.current = null
    selectedIdRef.current = null
    setSelectedId(null)
    setTitle('')
    setContent('')
    setEditorKeywords([])
    setOriginalMetadata(null)
    setSelectedStudioSummary(null)
    assignSelectedVersion(null)
    setSelectedLastSavedAt(null)
    setBacklinkConversationId(null)
    setBacklinkMessageId(null)
    setDirtyFlag(false)
    interruptedSaveRef.current = false
    dispatchSave({ type: 'loaded', acknowledged: false })
    setEditProvenance({ mode: 'manual' })
    setMonitoringNotice(null)
    clearTaskState()
    setRemoteVersionInfo(null)
    setEditorCursorIndex(null)
    replaceWysiwygHtml(EMPTY_WYSIWYG_HTML)
    setWysiwygSessionDirty(false)
    markdownBeforeWysiwygRef.current = null
  }, [assignSelectedVersion, clearAssistUndoState, clearTaskState, dispatchSave, replaceWysiwygHtml, setDirtyFlag, setEditorKeywords, setMonitoringNotice])

  /** User-facing reason a save did not reach the server (NS-02, NS-03). */
  const describeSaveFailure = React.useCallback(
    (failure: NoteSaveFailure | null, keptOnDevice: boolean): string => {
      switch (failure?.kind) {
        case 'conflict':
          return t('option:notesSearch.saveConflictExplained', {
            defaultValue:
              'This note was changed in another tab or device. Your edits are still here: keep your version, use theirs, or copy your text.'
          })
        case 'validation':
          return [
            t('option:notesSearch.saveRejected', { defaultValue: 'The server did not accept this note.' }),
            failure.message
          ].filter(Boolean).join(' ')
        case 'auth':
          return t('option:notesSearch.saveNotAllowed', {
            defaultValue: 'You are signed out or no longer have access to this note. Sign in again, then retry.'
          })
        case 'missing':
          return t('option:notesSearch.saveNoteMissing', {
            defaultValue: 'This note no longer exists on the server. Copy your text to keep it.'
          })
        case 'server':
          return keptOnDevice
            ? t('option:notesSearch.saveServerErrorKept', {
                defaultValue: 'The server had a problem saving. A copy is kept on this device and saving will retry automatically.'
              })
            : t('option:notesSearch.saveServerError', {
                defaultValue: 'The server had a problem saving. Saving will retry automatically.'
              })
        case 'network':
          return keptOnDevice
            ? t('option:notesSearch.saveUnreachableKept', {
                defaultValue: 'Could not reach the server. A copy is kept on this device and saving will retry automatically.'
              })
            : t('option:notesSearch.saveUnreachable', {
                defaultValue: 'Could not reach the server. Saving will retry automatically.'
              })
        default:
          return t('option:notesSearch.saveErrorRecovery', {
            defaultValue: 'Save failed. Your edits are still in the editor.'
          })
      }
    },
    [t]
  )

  /** True when the offline queue holds exactly what the editor shows. */
  const isDraftKeptOnDevice = React.useCallback(() => {
    const snapshot = editRevisionRef.current
    const queued = offlineDraftQueueRef.current[offlineDraftKeyFor(selectedIdRef.current)]
    return Boolean(
      queued &&
        queued.syncState !== 'conflict' &&
        queued.title === snapshot.title &&
        // A queued draft carries the note's knowledge provenance marker; the editor does not.
        stripKnowledgeNoteProvenance(queued.content) === snapshot.content
    )
  }, [])

  /**
   * Let an in-flight save finish, then save what is still pending. The leave
   * flush keeps going while newer edits arrive; a note switch saves once and
   * lets the caller decide about edits typed meanwhile.
   */
  const settlePendingEdits = React.useCallback(
    async (trigger: 'switch' | 'leave'): Promise<'clean' | 'saved' | 'failed'> => {
      clearAutosaveTimeout()
      let saved = false
      for (let round = 0; round < 4; round += 1) {
        const inFlight = activeSavePromiseRef.current
        if (inFlight) {
          saved = (await inFlight.catch(() => false)) || saved
          continue
        }
        if (!isDirtyRef.current) return saved ? 'saved' : 'clean'
        const snapshot = editRevisionRef.current
        if (!snapshot.title.trim() && !snapshot.content.trim()) return 'failed'
        if (!saveNoteRef.current || !canStartNoteSave(saveStateRef.current, trigger)) return 'failed'
        const ok = await saveNoteRef.current({ showSuccessMessage: false, trigger }).catch(() => false)
        if (!ok) return 'failed'
        saved = true
        if (trigger === 'switch') return 'saved'
      }
      return isDirtyRef.current ? 'failed' : saved ? 'saved' : 'clean'
    },
    [clearAutosaveTimeout]
  )

  const markEditsDiscarded = React.useCallback(() => {
    discardedRevisionRef.current = editRevisionRef.current.revision
    dropOfflineDraftNow(offlineDraftKeyFor(selectedIdRef.current))
  }, [dropOfflineDraftNow])

  /** Ask what to do with edits that could not be saved (cause-specific, NS-02/NS-03). */
  const askAboutUnsavedEdits = React.useCallback(
    (trigger: 'switch' | 'leave') =>
      new Promise<'saved' | 'continue' | 'discard' | 'stay'>((resolve) => {
        const state = saveStateRef.current
        const keptOnDevice = isDraftKeptOnDevice()
        const title = t('option:notesSearch.unsavedEditsTitle', {
          defaultValue: 'Your latest edits are not saved'
        })
        const stayText = t('option:notesSearch.stayOnNote', { defaultValue: 'Stay on this note' })
        if (state.status === 'conflict') {
          Modal.confirm({
            title,
            content: describeSaveFailure({ kind: 'conflict', status: 409, message: '' }, false),
            okText: t('option:notesSearch.discardMyChanges', { defaultValue: 'Discard my changes' }),
            okButtonProps: { danger: true },
            cancelText: stayText,
            onOk: () => resolve('discard'),
            onCancel: () => resolve('stay')
          })
          return
        }
        if (keptOnDevice) {
          Modal.confirm({
            title,
            content: describeSaveFailure(state.failure, true),
            okText: t('option:notesSearch.continueSyncLater', { defaultValue: 'Continue, sync later' }),
            cancelText: stayText,
            onOk: () => resolve('continue'),
            onCancel: () => resolve('stay')
          })
          return
        }
        const modalRef = Modal.confirm({
          title,
          content: describeSaveFailure(state.failure, false),
          okText: t('option:notesSearch.retrySave', { defaultValue: 'Try saving again' }),
          cancelText: stayText,
          okButtonProps: { type: 'primary' },
          onOk: async () => {
            const saved = await (saveNoteRef.current?.({ showSuccessMessage: true, trigger }) ?? Promise.resolve(false))
              .catch(() => false)
            resolve(saved ? 'saved' : 'stay')
          },
          onCancel: () => resolve('stay'),
          footer: (_, { OkBtn, CancelBtn }) => (
            <>
              <CancelBtn />
              <Button
                danger
                onClick={() => {
                  modalRef.destroy()
                  resolve('discard')
                }}
              >
                {t('option:notesSearch.discardChanges', { defaultValue: 'Discard changes' })}
              </Button>
              <OkBtn />
            </>
          ),
        })
      }),
    [describeSaveFailure, isDraftKeptOnDevice, t]
  )

  /**
   * Before the editor changes note or the page goes away: flush pending edits
   * and only ask when they could not be saved (NS-01).
   */
  const confirmDiscardIfDirty = React.useCallback(
    async (onSaved?: () => void, options?: { trigger?: 'switch' | 'leave' }) => {
      const trigger = options?.trigger ?? 'switch'
      if (!isDirtyRef.current && !activeSavePromiseRef.current) return true
      const outcome = await settlePendingEdits(trigger)
      if (outcome === 'clean') return true
      if (outcome === 'saved') {
        onSaved?.()
        return true
      }
      const snapshot = editRevisionRef.current
      if (!snapshot.title.trim() && !snapshot.content.trim()) {
        const ok = await confirmDanger({
          title: t('option:notesSearch.unsavedChangesTitle', { defaultValue: 'Save changes?' }),
          content: t('option:notesSearch.unsavedChangesContent', {
            defaultValue: 'Your changes could not be saved automatically. What would you like to do?'
          }),
          okText: t('option:notesSearch.discardChanges', { defaultValue: 'Discard changes' }),
          cancelText: t('common:cancel', { defaultValue: 'Cancel' })
        })
        if (ok) markEditsDiscarded()
        return ok
      }
      const choice = await askAboutUnsavedEdits(trigger)
      if (choice === 'saved') {
        onSaved?.()
        return true
      }
      if (choice === 'discard') markEditsDiscarded()
      return choice !== 'stay'
    },
    [askAboutUnsavedEdits, confirmDanger, markEditsDiscarded, settlePendingEdits, t]
  )
  const confirmDiscardIfDirtyRef = React.useRef(confirmDiscardIfDirty)
  confirmDiscardIfDirtyRef.current = confirmDiscardIfDirty

  const switchListMode = React.useCallback(
    async (nextMode: 'active' | 'trash') => {
      if (nextMode === listMode) return
      const ok = await confirmDiscardIfDirty()
      if (!ok) return
      resetEditor()
      setListMode(nextMode)
      setPage(1)
      if (nextMode === 'trash') {
        setQuery('')
        setQueryInput('')
        setKeywordTokens([])
        setSelectedNotebookId(null)
      }
    },
    [confirmDiscardIfDirty, listMode, resetEditor, setListMode, setPage, setQuery, setQueryInput, setKeywordTokens, setSelectedNotebookId]
  )

  const openSourceNote = React.useCallback(async (id: string, signal: AbortSignal): Promise<'opened' | 'cancelled' | 'unavailable'> => {
    const requestEpoch = authorityEpochRef.current
    const requestAuthorityScope = authorityScope
    const capturedConfig = connectionConfig ? { ...connectionConfig } : null
    let selectionEpoch = noteSelectionEpochRef.current
    let editRevision = editRevisionRef.current.revision
    const dirtyRevision = dirtyRevisionRef.current
    const controller = new AbortController()
    const abort = () => controller.abort()
    signal.addEventListener('abort', abort, { once: true })
    authorityRequestsRef.current.add(controller)
    const isCurrent = () => !signal.aborted && !controller.signal.aborted && authorityEpochRef.current === requestEpoch && authorityScopeRef.current === requestAuthorityScope
    try {
      const confirmed = await confirmDiscardIfDirty(() => {
        selectionEpoch = noteSelectionEpochRef.current
      })
      if (!confirmed || !isCurrent() || dirtyRevisionRef.current !== dirtyRevision || noteSelectionEpochRef.current !== selectionEpoch) return 'cancelled'
      if (!requestAuthorityScope || !capturedConfig) return 'unavailable'
      editRevision = editRevisionRef.current.revision
      if (!id || id.length > 200) {
        resetEditor()
        return 'unavailable'
      }
      const user = await tldwAuth.getCurrentUser()
      if (!isCurrent() || noteSelectionEpochRef.current !== selectionEpoch || editRevisionRef.current.revision !== editRevision) return 'cancelled'
      if (!user?.is_active || user.id == null || createNotesGraphAuthorityScope(capturedConfig.serverUrl, user.id) !== requestAuthorityScope) {
        resetEditor()
        return 'unavailable'
      }
      selectionEpoch += 1
      const opened = await loadDetail(id, {
        servicePromptConfig: {
          serverUrl: capturedConfig.serverUrl,
          authMode: capturedConfig.authMode,
          authSource: capturedConfig.authSource,
          orgId: capturedConfig.orgId,
          expectedUserId: user.id,
          expectedSingleUserApiKeyScope: capturedConfig.authMode === 'single-user'
            ? deriveSingleUserApiKeyCredentialScope('single-user', capturedConfig.apiKey) : undefined
        },
        headers: { 'X-TLDW-Expected-User-ID': String(user.id) },
        abortSignal: controller.signal
      })
      if (!isCurrent()) return 'cancelled'
      if (!opened) {
        if (noteSelectionEpochRef.current !== selectionEpoch || editRevisionRef.current.revision !== editRevision) return 'cancelled'
        resetEditor()
      }
      if (opened && isMobileViewport) setMobileSidebarOpen(false)
      return opened ? 'opened' : 'unavailable'
    } catch {
      if (isCurrent() && noteSelectionEpochRef.current === selectionEpoch && editRevisionRef.current.revision === editRevision) resetEditor()
      return isCurrent() ? 'unavailable' : 'cancelled'
    } finally {
      if (authorityEpochRef.current === requestEpoch && authorityScopeRef.current === requestAuthorityScope && noteSelectionEpochRef.current === selectionEpoch) setLoadingDetail(false)
      signal.removeEventListener('abort', abort)
      authorityRequestsRef.current.delete(controller)
    }
  }, [authorityScope, connectionConfig, confirmDiscardIfDirty, isMobileViewport, loadDetail, resetEditor, setLoadingDetail, setMobileSidebarOpen])

  const handleSelectNote = React.useCallback(
    async (id: string | number): Promise<boolean> => {
      const ok = await confirmDiscardIfDirty()
      if (!ok) return false
      const opened = await loadDetail(id)
      if (!opened) return false
      if (isMobileViewport) {
        setMobileSidebarOpen(false)
      }
      return true
    },
    [confirmDiscardIfDirty, isMobileViewport, loadDetail, setMobileSidebarOpen]
  )

  // ---- version conflict ----
  const isVersionConflictError = React.useCallback((error: any) => {
    const msg = String(error?.message || '')
    const lower = msg.toLowerCase()
    const status = error?.status ?? error?.response?.status
    return (
      status === 409 ||
      lower.includes('expected-version') ||
      lower.includes('expected_version') ||
      lower.includes('version mismatch')
    )
  }, [])

  const reloadSelectedNoteAfterConflict = React.useCallback(async () => {
    if (selectedId == null) return
    const ok = await confirmDanger({
      title: t('option:notesSearch.reloadConflictTitle', {
        defaultValue: 'Reload server version?'
      }),
      content: t('option:notesSearch.reloadConflictContent', {
        defaultValue:
          'Reloading replaces the current editor contents with the server version. Your unsaved local edits will be discarded.'
      }),
      okText: t('option:notesSearch.reloadConflictAction', {
        defaultValue: 'Reload server version'
      }),
      cancelText: t('option:notesSearch.keepEditingAction', {
        defaultValue: 'Keep editing'
      })
    })
    if (!ok) return
    dropOfflineDraftNow(offlineDraftKeyFor(selectedId))
    await loadDetail(selectedId)
  }, [confirmDanger, dropOfflineDraftNow, loadDetail, selectedId, t])

  // "Reload notes" from the conflict toast. The editor's base version must
  // never move without the content it belongs to: if it did, the next autosave
  // would send the stale text with the new version and overwrite the other
  // tab's change (NS-N1). The toast keeps an old copy of this callback, so the
  // live editor state is read from refs.
  const reloadNotes = React.useCallback(async (noteId?: string | number | null) => {
    const currentId = selectedIdRef.current
    const target = noteId ?? currentId
    if (target == null || currentId == null || String(target) !== String(currentId)) {
      // The conflict was on another note: refresh the list, leave the editor alone.
      await refetch()
      return
    }
    clearAutosaveTimeout()
    // A conflicting draft restored from this device is local text too.
    const hadLocalEdits = isDirtyRef.current || saveStateRef.current.status === 'conflict'
    if (hadLocalEdits) {
      // Keep the unsaved text recoverable before the server copy replaces it.
      // This must run before any other await (see writeClipboardText).
      const local = editRevisionRef.current
      const copied = await writeClipboardText(
        buildSingleNoteCopyText(
          { id: currentId, title: local.title, content: local.content, keywords: local.editorKeywords },
          'markdown'
        )
      )
      if (!copied) {
        message.warning(
          t('option:notesSearch.reloadConflictCopyFailed', {
            defaultValue: 'Could not copy your unsaved edits to the clipboard.'
          })
        )
        // Fall back to the explicit "your edits will be discarded" confirm.
        await reloadSelectedNoteAfterConflict()
        await refetch()
        return
      }
    }
    // The local text is on the clipboard; a queued copy must not come back over the server's.
    dropOfflineDraftNow(offlineDraftKeyFor(target))
    const loaded = await loadDetail(target)
    await refetch()
    if (loaded && hadLocalEdits) {
      message.info({
        content: t('option:notesSearch.reloadConflictCopied', {
          defaultValue:
            'Loaded the latest version from the server. Your unsaved edits were copied to the clipboard.'
        }),
        duration: 8
      })
    }
  }, [clearAutosaveTimeout, dropOfflineDraftNow, loadDetail, message, refetch, reloadSelectedNoteAfterConflict, t])

  const handleVersionConflict = React.useCallback((noteId?: string | number | null) => {
    message.error({
      content: (
        <span
          className="inline-flex items-center gap-2"
          role="alert"
          aria-live="assertive"
          aria-atomic="true"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              void reloadNotes(noteId)
            }
          }}
        >
          <span>This note changed on the server.</span>
          <Button
            type="link"
            size="small"
            onClick={() => void reloadNotes(noteId)}
            aria-label="Reload notes"
          >
            Reload notes
          </Button>
        </span>
      ),
      duration: 6
    })
  }, [message, reloadNotes])

  const toggleTaskCheckboxStatus = React.useCallback(
    async (payload: TaskChecklistTogglePayload) => {
      const task = payload.task
      if (!task || selectedId == null) {
        toggleTaskCheckboxLocal(payload)
        return
      }
      const expectedNoteVersion = selectedVersion ?? task.projection?.note_version ?? null
      if (expectedNoteVersion == null) {
        setTaskConflictNotice(
          t('option:notesSearch_taskConflictNotice', {
            defaultValue: 'Task changed on the server. Reload the note and try again.'
          })
        )
        return
      }

      try {
        await setNoteTaskStatus([
          {
            task_id: task.id,
            status: payload.nextStatus,
            expected_task_version: task.version,
            expected_note_version: expectedNoteVersion
          }
        ])
        setTaskConflictNotice(null)
        await loadDetail(selectedId)
      } catch (error: any) {
        setTaskConflictNotice(
          t('option:notesSearch_taskConflictNotice', {
            defaultValue: 'Task changed on the server. Reload the note and try again.'
          })
        )
        if (isVersionConflictError(error)) {
          handleVersionConflict(selectedId)
        } else {
          message.error(String(error?.message || '') || 'Task update failed')
        }
      }
    },
    [
      handleVersionConflict,
      isVersionConflictError,
      loadDetail,
      message,
      selectedId,
      selectedVersion,
      t,
      toggleTaskCheckboxLocal
    ]
  )

  // ---- monitoring ----
  const loadMonitoringNoticeForSavedNote = React.useCallback(
    async (
      noteId: string | number,
      source: 'notes.create' | 'notes.update',
      saveStartedAtMs: number,
      context: {
        isCurrent: () => boolean
        ownerId: string | number
        request: NotesOwnedRequestOptions
        capability: ReturnType<typeof useCallerCapabilities>
      }
    ) => {
      const isCurrent = () => context.isCurrent() &&
        context.capability.refreshAfterForbidden === callerCapabilitiesRef.current.refreshAfterForbidden
      if (!isCurrent() || context.capability.monitoringAlerts !== 'allowed' ||
        String(context.capability.userId ?? '') !== String(context.ownerId)) return
      const controller = new AbortController()
      authorityRequestsRef.current.add(controller)
      try {
        const params = new URLSearchParams()
        params.set('user_id', String(context.ownerId))
        params.set('source', source)
        params.set('unread_only', 'true')
        params.set('limit', '50')
        const response = await bgRequest<any>({
          ...context.request,
          abortSignal: controller.signal,
          path: `/api/v1/monitoring/alerts?${params.toString()}` as any,
          method: 'GET' as any
        })
        if (!isCurrent() || controller.signal.aborted) return
        const items = Array.isArray(response?.items) ? response.items : []
        const noteIdText = String(noteId)
        const minCreatedAtMs = saveStartedAtMs - 5000
        const matchedAlert = items.find((item: any) => {
          if (String(item?.user_id ?? '') !== String(context.ownerId)) return false
          if (String(item?.source_id || '') !== noteIdText) return false
          const createdAtMs = Date.parse(String(item?.created_at || ''))
          if (Number.isFinite(createdAtMs) && createdAtMs < minCreatedAtMs) return false
          return true
        })
        if (!matchedAlert) return

        const severityRaw = String(matchedAlert?.rule_severity || 'info').toLowerCase()
        const severity: MonitoringAlertSeverity =
          severityRaw === 'critical'
            ? 'critical'
            : severityRaw === 'warning'
              ? 'warning'
              : 'info'

        const titleCopy =
          severity === 'critical'
            ? t('option:notesSearch.monitoringAlertCriticalTitle', {
                defaultValue: 'Sensitive-topic alert detected'
              })
            : severity === 'warning'
              ? t('option:notesSearch.monitoringAlertWarningTitle', {
                  defaultValue: 'Monitoring warning detected'
                })
              : t('option:notesSearch.monitoringAlertInfoTitle', {
                  defaultValue: 'Monitored topic detected'
                })

        const guidanceCopy = t('option:notesSearch.monitoringAlertGuidance', {
          defaultValue:
            'Review this note for sensitive material before sharing. You can edit and save again.'
        })

        setMonitoringNotice({
          severity,
          title: titleCopy,
          guidance: guidanceCopy
        })
      } catch (error) {
        if (isCurrent() && !controller.signal.aborted) {
          await context.capability.refreshAfterForbidden(error)
        }
      } finally {
        authorityRequestsRef.current.delete(controller)
      }
    },
    [t, setMonitoringNotice]
  )

  const showKeywordSyncWarning = React.useCallback(
    (warning: KeywordSyncWarning, action: 'created' | 'updated') => {
      if (warning.failedCount <= 0) return
      const keywordSuffix =
        warning.failedKeywords.length > 0 ? ` (${warning.failedKeywords.join(', ')})` : ''
      message.warning(
        `Note ${action}, but ${warning.failedCount} tag${
          warning.failedCount === 1 ? '' : 's'
        } failed to attach${keywordSuffix}.`
      )
    },
    [message]
  )

  // ---- save note ----
  // Every exit of a started save reports to the save machine: a 2xx is the
  // only way to "saved", 409 is a conflict, 400/401/404/422 are fatal and
  // network/5xx back off and keep a copy in the offline queue.
  const saveNote = React.useCallback(
    async ({
      showSuccessMessage = true,
      trigger = 'manual',
      expectedVersion: expectedVersionOverride
    }: SaveNoteOptions = {}) => {
      if (pendingSelectionEpochRef.current != null) return false
      if (savingInFlightRef.current) return false
      const machine = saveStateRef.current
      if (!canStartNoteSave(machine, trigger as NoteSaveTrigger)) {
        if (machine.status === 'conflict' && showSuccessMessage) {
          message.warning(
            t('option:notesSearch.saveConflictChooseFirst', {
              defaultValue: 'This note changed elsewhere. Keep your version, use theirs, or copy your text first.'
            })
          )
        }
        return false
      }
      if (
        offlineSyncInFlightRef.current != null &&
        offlineDraftQueueRef.current[offlineDraftKeyFor(selectedIdRef.current)]?.syncState === 'syncing'
      ) {
        // The offline queue is sending this note's draft right now; its result
        // re-arms autosave, and saving in parallel would race it into a 409.
        return false
      }
      // Read identity and version from refs: a save chained right after
      // another (the leave flush) must not use a stale render's values.
      const noteId = selectedIdRef.current
      const baseVersion = expectedVersionOverride ?? selectedVersionRef.current
      const snapshot = editRevisionRef.current
      const saveEpoch = ++saveEpochRef.current
      const savedEditRevision = snapshot.revision
      const hasNewerEdits = () => editRevisionRef.current.revision !== savedEditRevision
      const requestAuthorityScope = authorityScope
      const requestEpoch = authorityEpochRef.current
      let requestNoteEpoch = noteSelectionEpochRef.current
      const capturedConfig = connectionConfig ? { ...connectionConfig } : null
      const capability = callerCapabilitiesRef.current
      const isCurrent = () => saveEpochRef.current === saveEpoch && authorityEpochRef.current === requestEpoch &&
        authorityScopeRef.current === requestAuthorityScope && noteSelectionEpochRef.current === requestNoteEpoch
      if (!snapshot.content.trim() && !snapshot.title.trim()) {
        if (showSuccessMessage) {
          message.warning('Nothing to save')
        }
        return false
      }
      if (!isOnline || browserReportsOffline()) {
        const queuedAt = new Date().toISOString()
        const persisted = persistOfflineDraft({
          syncState: 'queued',
          lastError: null,
          updatedAt: queuedAt
        })
        if (!persisted || !isCurrent()) {
          if (isCurrent()) {
            const unavailableMessage = t('option:notesSearch.offlineSaveUnavailable', {
              defaultValue: 'Offline saving is unavailable until your account and local storage are confirmed. Your changes remain unsaved.'
            })
            lastFailureAtRef.current = Date.now()
            dispatchSave({
              type: 'save-failed',
              failure: { kind: 'network', status: null, message: unavailableMessage },
              queuedOffline: false,
              hasNewerEdits: false
            })
            if (showSuccessMessage) message.error(unavailableMessage)
          }
          return false
        }
        setDirtyFlag(false)
        dispatchSave({ type: 'queued-offline' })
        if (showSuccessMessage) {
          message.info(
            t('option:notesSearch.offlineSavedLocally', {
              defaultValue: 'Saved locally. Sync will resume when connection returns.'
            })
          )
        }
        return true
      }
      if (
        trigger !== 'keep-mine' &&
        noteId != null &&
        baseVersion != null &&
        remoteVersionInfo &&
        remoteVersionInfo.version > baseVersion
      ) {
        // Saving now would send a stale version: never ask from autosave
        // (NS-N2); the conflict panel offers the explicit choices (NS-03).
        dispatchSave({ type: 'remote-changed' })
        if (showSuccessMessage) {
          message.warning(
            t('option:notesSearch.saveConflictChooseFirst', {
              defaultValue: 'This note changed elsewhere. Keep your version, use theirs, or copy your text first.'
            })
          )
        }
        return false
      }
      let settled = false
      let resolveActiveSave!: (saved: boolean) => void
      const activeSave = new Promise<boolean>((resolve) => {
        resolveActiveSave = resolve
      })
      activeSavePromiseRef.current = activeSave
      savingInFlightRef.current = true
      interruptedSaveRef.current = false
      setSaving(true)
      dispatchSave({ type: 'save-started' })
      setMonitoringNotice(null)
      const saveStartedAtMs = Date.now()
      const controller = new AbortController()
      activeSaveRef.current = controller
      authorityRequestsRef.current.add(controller)
      const acknowledge = () => {
        const newer = hasNewerEdits()
        setDirtyFlag(newer)
        if (newer) dirtySinceRef.current = Date.now()
        setRemoteVersionInfo(null)
        dispatchSave({ type: 'save-succeeded', hasNewerEdits: newer })
        settled = true
      }
      let result = false
      try {
        if (!requestAuthorityScope || !capturedConfig) {
          throw ownerError('The authenticated note owner is unavailable. Reconnect and try again.')
        }
        const user = await tldwAuth.getCurrentUser()
        if (!isCurrent() || controller.signal.aborted) return false
        if (!user?.is_active || user.id == null ||
          createNotesGraphAuthorityScope(capturedConfig.serverUrl, user.id) !== requestAuthorityScope) {
          throw ownerError('The authenticated note owner changed. Reopen the note before saving.')
        }
        const ownedRequest: NotesOwnedRequestOptions = {
          servicePromptConfig: {
            serverUrl: capturedConfig.serverUrl,
            authMode: capturedConfig.authMode,
            authSource: capturedConfig.authSource,
            orgId: capturedConfig.orgId,
            expectedUserId: user.id,
            expectedSingleUserApiKeyScope: capturedConfig.authMode === 'single-user'
              ? deriveSingleUserApiKeyCredentialScope('single-user', capturedConfig.apiKey)
              : undefined
          },
          abortSignal: controller.signal,
          headers: { 'X-TLDW-Expected-User-ID': String(user.id) }
        }
        const request = async (
          init: BgRequestInit<`/${string}`, 'GET' | 'POST' | 'PUT'>,
          onResponse?: (response: { id?: string | number }) => void
        ) => {
          if (!isCurrent()) throw new DOMException('Note selection changed', 'AbortError')
          const response = await bgRequest<{ id?: string | number }>({ ...init, ...ownedRequest, headers: { ...init.headers, ...ownedRequest.headers } })
          // The server has answered, even if the editor has since moved on.
          onResponse?.(response)
          if (!isCurrent() || controller.signal.aborted) throw new DOMException('Note selection changed', 'AbortError')
          return response
        }
        const monitoringContext = { isCurrent, ownerId: user.id, request: ownedRequest, capability }
        const draftKey = offlineDraftKeyFor(noteId)
        const acknowledgeOfflineDraft = (savedNoteId: string | number, version: number | null) => {
          if (!hasNewerEdits()) {
            dropOfflineDraftNow(draftKey)
            return
          }
          setOfflineDraftQueue((current) => {
            const queued = current[draftKey]
            if (!queued) return current
            const latest = editRevisionRef.current
            const next = { ...current }
            delete next[draftKey]
            const key = `note:${String(savedNoteId)}`
            next[key] = {
              ...queued, key, noteId: String(savedNoteId), baseVersion: version ?? queued.baseVersion,
              title: latest.title, content: latest.content, keywords: [...latest.editorKeywords],
              metadata: latest.originalMetadata, backlinkConversationId: latest.backlinkConversationId,
              backlinkMessageId: latest.backlinkMessageId
            }
            return next
          })
        }
        const metadata: Record<string, any> = {
          ...(snapshot.originalMetadata || {}),
          keywords: snapshot.editorKeywords
        }
        if (snapshot.backlinkConversationId) metadata.conversation_id = snapshot.backlinkConversationId
        if (snapshot.backlinkMessageId) metadata.message_id = snapshot.backlinkMessageId
        const payload: Record<string, any> = {
          title: snapshot.title || undefined,
          content: retainKnowledgeNoteProvenance(snapshot.content, snapshot.originalMetadata),
          metadata,
          keywords: snapshot.editorKeywords
        }
        if (snapshot.backlinkConversationId) payload.conversation_id = snapshot.backlinkConversationId
        if (snapshot.backlinkMessageId) payload.message_id = snapshot.backlinkMessageId
        if (noteId == null) {
          if (!snapshot.title.trim()) {
            // Let the server name an untitled note from its content (NS-02).
            delete payload.title
            payload.auto_title = true
          }
          const created = await request({
            path: '/api/v1/notes/' as any,
            method: 'POST' as any,
            headers: { 'Content-Type': 'application/json' },
            body: payload
          })
          const createdKeywordWarning = toKeywordSyncWarning(created)
          const createdVersion = toNoteVersion(created)
          const createdLastSaved = toNoteLastModified(created)
          if (created?.id != null) {
            savedTitleRef.current = { noteId: String(created.id), title: toNoteTitle(created) ?? snapshot.title }
            selectedIdRef.current = created.id
            setSelectedId(created.id)
            acknowledgeOfflineDraft(created.id, createdVersion)
          }
          if (createdVersion != null) assignSelectedVersion(createdVersion)
          if (createdLastSaved) setSelectedLastSavedAt(createdLastSaved)
          acknowledge()
          if (showSuccessMessage) {
            message.success('Note created')
          }
          if (createdKeywordWarning) {
            showKeywordSyncWarning(createdKeywordWarning, 'created')
          }
          result = true
          // The page is going away: the acknowledgement is all the user needs.
          if (trigger === 'leave') return true
          await refetch()
          if (!isCurrent()) return (result = false)
          if (created?.id != null) {
            if (!hasNewerEdits()) {
              requestNoteEpoch = noteSelectionEpochRef.current + 1
              const loaded = await loadDetail(created.id, ownedRequest, savedEditRevision)
              if (!isCurrent()) return (result = false)
              if (loaded && createdLastSaved) setSelectedLastSavedAt(createdLastSaved)
            }
            void loadMonitoringNoticeForSavedNote(created.id, 'notes.create', saveStartedAtMs, monitoringContext)
          }
          return true
        } else {
          let expectedVersion = baseVersion
          if (expectedVersion == null) {
            const latest = await request({
              path: noteResourcePath(noteId) as any,
              method: 'GET' as any
            })
            expectedVersion = toNoteVersion(latest)
          }
          if (expectedVersion == null) {
            throw Object.assign(new Error('Missing version; reload and try again'), { status: 428 })
          }
          const previousSavedTitle =
            savedTitleRef.current?.noteId === String(noteId) ? savedTitleRef.current.title : null
          const updated = await request(
            {
              path: noteResourcePath(noteId) as any,
              method: 'PUT' as any,
              headers: {
                'Content-Type': 'application/json',
                'expected-version': String(expectedVersion)
              },
              body: payload
            },
            // A rename is saved once the server answers, whatever the editor shows by then.
            // A blank title is not sent, so the server keeps the previous one.
            (saved) =>
              announceSavedTitle(
                noteId,
                previousSavedTitle,
                toNoteTitle(saved) ?? (snapshot.title.trim() || previousSavedTitle),
                requestAuthorityScope
              )
          )
          const updatedKeywordWarning = toKeywordSyncWarning(updated)
          const updatedVersion = toNoteVersion(updated)
          const updatedLastSaved = toNoteLastModified(updated)
          // Record the acknowledged version before anything else can save.
          if (updatedVersion != null) assignSelectedVersion(updatedVersion)
          setSelectedLastSavedAt(updatedLastSaved || new Date().toISOString())
          acknowledgeOfflineDraft(noteId, updatedVersion)
          acknowledge()
          if (showSuccessMessage) {
            message.success('Note updated')
          }
          if (updatedKeywordWarning) {
            showKeywordSyncWarning(updatedKeywordWarning, 'updated')
          }
          result = true
          if (trigger === 'leave') return true
          await refetch()
          if (!isCurrent()) return (result = false)
          if (updatedVersion == null) {
            try {
              const latest = await request({
                path: noteResourcePath(noteId) as any,
                method: 'GET' as any
              })
              assignSelectedVersion(toNoteVersion(latest))
            } catch (err) {
              if (!isCurrent() || controller.signal.aborted) return (result = false)
              console.debug('[NotesManagerPage] Version refresh after save failed:', err)
            }
          }
          await refreshTaskStateForNote(noteId)
          if (!isCurrent()) return (result = false)
          void loadMonitoringNoticeForSavedNote(noteId, 'notes.update', saveStartedAtMs, monitoringContext)
          return true
        }
      } catch (e: any) {
        if (!isCurrent() || controller.signal.aborted) return (result = false)
        if (settled) {
          // The server already acknowledged; only the follow-up refresh failed.
          console.debug('[NotesManagerPage] Refresh after save failed:', e)
          return result
        }
        const failure = classifyNoteSaveError(e)
        if (!failure) {
          dispatchSave({ type: 'save-cancelled', hasPendingEdits: isDirtyRef.current })
          settled = true
          return (result = false)
        }
        let queuedOffline = false
        if (isRetryableNoteSaveFailure(failure.kind)) {
          // Keep the edits on this device while the server is unreachable (NS-05).
          queuedOffline = persistOfflineDraft({ syncState: 'queued', lastError: failure.message })
          lastFailureAtRef.current = Date.now()
        }
        dispatchSave({ type: 'save-failed', failure, queuedOffline, hasNewerEdits: hasNewerEdits() })
        settled = true
        if (showSuccessMessage && !requestAuthorityScope) {
          // Without a verified owner there is no save state to show the problem in.
          message.error(describeSaveFailure(failure, false))
        }
        // Out of automatic retries: the offline queue owns the draft now.
        if (saveStateRef.current.status === 'offline-queued') setDirtyFlag(false)
        return (result = false)
      } finally {
        authorityRequestsRef.current.delete(controller)
        if (activeSaveRef.current === controller) {
          activeSaveRef.current = null
          savingInFlightRef.current = false
          setSaving(false)
        }
        if (activeSavePromiseRef.current === activeSave) activeSavePromiseRef.current = null
        if (!settled && isCurrent() && saveStateRef.current.status === 'saving') {
          dispatchSave({ type: 'save-cancelled', hasPendingEdits: isDirtyRef.current })
        }
        resolveActiveSave(result)
      }
    },
    [
      announceSavedTitle,
      assignSelectedVersion,
      authorityScope,
      connectionConfig,
      describeSaveFailure,
      dispatchSave,
      dropOfflineDraftNow,
      isOnline,
      loadDetail,
      loadMonitoringNoticeForSavedNote,
      message,
      persistOfflineDraft,
      refetch,
      refreshTaskStateForNote,
      remoteVersionInfo,
      setDirtyFlag,
      setMonitoringNotice,
      setOfflineDraftQueue,
      showKeywordSyncWarning,
      t
    ]
  )

  // Keep ref in sync
  saveNoteRef.current = saveNote

  // ---- offline sync ----
  const syncOfflineDraftEntry = React.useCallback(
    async (draft: OfflineDraftEntry): Promise<OfflineDraftSyncResult> => {
      const requestScope = authorityScope
      const requestEpoch = authorityEpochRef.current
      const capturedConfig = connectionConfig ? { ...connectionConfig } : null
      const authorityChanged = () => !requestScope || authorityEpochRef.current !== requestEpoch || authorityScopeRef.current !== requestScope
      const cancelledResult: OfflineDraftSyncResult = { status: 'error', key: draft.key, message: 'Account changed during draft sync.' }
      if (authorityChanged()) return cancelledResult
      if (!capturedConfig) return cancelledResult
      const metadata: Record<string, any> = {
        ...(draft.metadata || {}),
        keywords: draft.keywords
      }
      if (draft.backlinkConversationId) metadata.conversation_id = draft.backlinkConversationId
      if (draft.backlinkMessageId) metadata.message_id = draft.backlinkMessageId
      const payload: Record<string, any> = {
        title: draft.title || undefined,
        content: retainKnowledgeNoteProvenance(draft.content, draft.metadata),
        metadata,
        keywords: draft.keywords
      }
      if (draft.backlinkConversationId) payload.conversation_id = draft.backlinkConversationId
      if (draft.backlinkMessageId) payload.message_id = draft.backlinkMessageId
      if (!draft.noteId && !String(draft.title || '').trim()) {
        // Let the server name an untitled note from its content (NS-02).
        delete payload.title
        payload.auto_title = true
      }

      const controller = new AbortController()
      authorityRequestsRef.current.add(controller)
      try {
        const user = await tldwAuth.getCurrentUser()
        if (
          authorityChanged() || controller.signal.aborted || !user?.is_active ||
          user.id == null || createNotesGraphAuthorityScope(capturedConfig.serverUrl, user.id) !== requestScope
        ) return cancelledResult
        // Reuse the guarded transport in both the browser and extension worker.
        // It checks every credential load and refresh against this verified owner.
        const servicePromptConfig: ServicePromptTargetConfig = {
          serverUrl: capturedConfig.serverUrl,
          authMode: capturedConfig.authMode,
          authSource: capturedConfig.authSource,
          orgId: capturedConfig.orgId,
          expectedUserId: user.id,
          expectedSingleUserApiKeyScope: capturedConfig.authMode === 'single-user'
            ? deriveSingleUserApiKeyCredentialScope('single-user', capturedConfig.apiKey)
            : undefined
        }
        const requestScopeOptions = { servicePromptConfig, abortSignal: controller.signal }
        const ownerHeader = { 'X-TLDW-Expected-User-ID': String(user.id) }
        if (!draft.noteId) {
          const created = await bgRequest<any>({
            ...requestScopeOptions,
            path: '/api/v1/notes/' as any,
            method: 'POST' as any,
            headers: { ...ownerHeader, 'Content-Type': 'application/json' },
            body: payload
          })
          const createdId = String(created?.id || '').trim()
          if (!createdId) {
            return {
              status: 'error',
              key: draft.key,
              message: 'Queued create sync did not return a note id.'
            }
          }
          return {
            status: 'synced',
            key: draft.key,
            noteId: createdId,
            version: toNoteVersion(created),
            lastSavedAt: toNoteLastModified(created)
          }
        }

        const draftNotePath = noteResourcePath(draft.noteId)
        const remote = await bgRequest<any>({
          ...requestScopeOptions,
          headers: ownerHeader,
          path: draftNotePath as any,
          method: 'GET' as any
        })
        if (authorityChanged()) return cancelledResult
        const remoteVersion = toNoteVersion(remote)
        if (
          draft.baseVersion != null &&
          remoteVersion != null &&
          remoteVersion > draft.baseVersion
        ) {
          return {
            status: 'conflict',
            key: draft.key,
            message: `Remote version advanced from ${draft.baseVersion} to ${remoteVersion}.`
          }
        }
        const expectedVersion = draft.baseVersion ?? remoteVersion
        if (expectedVersion == null) {
          return {
            status: 'error',
            key: draft.key,
            message: 'Missing expected version for queued sync.'
          }
        }

        const updated = await bgRequest<any>({
          ...requestScopeOptions,
          path: draftNotePath as any,
          method: 'PUT' as any,
          headers: {
            ...ownerHeader,
            'Content-Type': 'application/json',
            'expected-version': String(expectedVersion)
          },
          body: payload
        })
        // A queued draft can carry a rename too.
        const remoteTitle = toNoteTitle(remote)
        announceSavedTitle(
          draft.noteId,
          remoteTitle,
          toNoteTitle(updated) ?? (String(draft.title || '').trim() || remoteTitle),
          requestScope
        )

        return {
          status: 'synced',
          key: draft.key,
          noteId: String(draft.noteId),
          version: toNoteVersion(updated),
          lastSavedAt: toNoteLastModified(updated)
        }
      } catch (error: any) {
        if (authorityChanged() || controller.signal.aborted) return cancelledResult
        if (isVersionConflictError(error)) {
          return {
            status: 'conflict',
            key: draft.key,
            message: String(error?.message || 'Server version conflict while syncing queued draft.')
          }
        }
        return {
          status: 'error',
          key: draft.key,
          message: String(error?.message || 'Queued sync failed.')
        }
      } finally {
        authorityRequestsRef.current.delete(controller)
      }
    },
    [announceSavedTitle, authorityScope, connectionConfig, isVersionConflictError]
  )

  const syncOfflineDraftQueue = React.useCallback(async () => {
    if (!isOnline || browserReportsOffline()) return
    if (!authorityScope || !offlineDraftQueueHydrated) return
    const requestScope = authorityScope
    const requestEpoch = authorityEpochRef.current
    if (offlineSyncInFlightRef.current === requestEpoch) return
    // The editor's own save in flight already carries the open note's draft.
    const editorDraftKey = savingInFlightRef.current ? offlineDraftKeyFor(selectedIdRef.current) : null
    const queuedEntries = Object.values(offlineDraftQueueRef.current)
      .filter((entry) => entry.syncState !== 'conflict' && entry.key !== editorDraftKey)
      .sort((a, b) => new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime())
    if (queuedEntries.length === 0) return

    offlineSyncInFlightRef.current = requestEpoch
    let successfulSyncs = 0
    try {
      for (const queuedEntry of queuedEntries) {
        if (authorityEpochRef.current !== requestEpoch || authorityScopeRef.current !== requestScope) return
        setOfflineDraftQueue((current) => {
          const existing = current[queuedEntry.key]
          if (!existing) return current
          return {
            ...current,
            [queuedEntry.key]: {
              ...existing,
              syncState: 'syncing',
              lastError: null
            }
          }
        })

        const latestEntry = offlineDraftQueueRef.current[queuedEntry.key] || queuedEntry
        const syncResult = await syncOfflineDraftEntry(latestEntry)
        if (authorityEpochRef.current !== requestEpoch || authorityScopeRef.current !== requestScope) return
        if (syncResult.status === 'synced' && syncResult.noteId) {
          successfulSyncs += 1
          setOfflineDraftQueue((current) => {
            if (!current[syncResult.key]) return current
            const next = { ...current }
            delete next[syncResult.key]
            return next
          })
          if (selectedIdRef.current == null && syncResult.key === NOTES_OFFLINE_NEW_DRAFT_KEY) {
            await loadDetail(syncResult.noteId)
          } else if (
            selectedIdRef.current != null &&
            syncResult.noteId &&
            String(selectedIdRef.current) === String(syncResult.noteId)
          ) {
            if (syncResult.version != null) {
              assignSelectedVersion(syncResult.version)
            }
            setSelectedLastSavedAt(syncResult.lastSavedAt || new Date().toISOString())
            // A 2xx from the queue is an acknowledgement too.
            dispatchSave({ type: 'save-succeeded', hasNewerEdits: isDirtyRef.current })
          }
          continue
        }

        if (syncResult.status === 'conflict') {
          setOfflineDraftQueue((current) => {
            const existing = current[syncResult.key]
            if (!existing) return current
            return {
              ...current,
              [syncResult.key]: {
                ...existing,
                syncState: 'conflict',
                lastError: syncResult.message
              }
            }
          })
          if (syncResult.key === offlineDraftKeyFor(selectedIdRef.current)) {
            dispatchSave({
              type: 'save-failed',
              failure: { kind: 'conflict', status: 409, message: syncResult.message },
              queuedOffline: true,
              hasNewerEdits: false
            })
          }
          continue
        }

        if (syncResult.status === 'error') {
          setOfflineDraftQueue((current) => {
            const existing = current[syncResult.key]
            if (!existing) return current
            return {
              ...current,
              [syncResult.key]: {
                ...existing,
                syncState: 'error',
                lastError: syncResult.message
              }
            }
          })
        }
      }
    } finally {
      if (offlineSyncInFlightRef.current === requestEpoch) offlineSyncInFlightRef.current = null
    }

    if (successfulSyncs > 0) {
      void refetch()
      message.success(
        t('option:notesSearch.offlineSyncCompleteToast', {
          defaultValue: 'Synced {{count}} queued offline draft(s).',
          count: successfulSyncs
        })
      )
    }
  }, [
    assignSelectedVersion,
    authorityScope,
    dispatchSave,
    offlineDraftQueueHydrated,
    setOfflineDraftQueue,
    isOnline,
    loadDetail,
    message,
    refetch,
    syncOfflineDraftEntry,
    t
  ])
  const syncOfflineDraftQueueRef = React.useRef(syncOfflineDraftQueue)
  syncOfflineDraftQueueRef.current = syncOfflineDraftQueue

  // ---- title suggestion ----
  const { data: scopedNotesTitleSettings } = useQuery({
    queryKey: ['notes-title-settings', authorityScope],
    enabled: isOnline && authorityScope !== null,
    staleTime: 5 * 60 * 1000,
    queryFn: async () => {
      const requestAuthorityScope = authorityScope
      try {
        // This policy endpoint is administrator-only; ordinary note editing and
        // heuristic title suggestions work without reading server admin settings.
        const user = await tldwAuth.getCurrentUser()
        if (
          authorityScopeRef.current !== requestAuthorityScope ||
          !user?.is_active || user.role?.trim().toLowerCase() !== 'admin'
        ) {
          return null
        }
        const settings = await bgRequest<NotesTitleSettingsResponse>({
          path: '/api/v1/admin/notes/title-settings' as any,
          method: 'GET' as any
        })
        return authorityScopeRef.current === requestAuthorityScope ? settings : null
      } catch {
        return null
      }
    }
  })
  const notesTitleSettings = authorityScope === null ? null : scopedNotesTitleSettings

  const allowedTitleStrategies = React.useMemo(
    () => deriveAllowedTitleStrategies(notesTitleSettings),
    [notesTitleSettings]
  )

  const canSwitchTitleStrategy = allowedTitleStrategies.length > 1

  const effectiveTitleSuggestStrategy = React.useMemo<NotesTitleSuggestStrategy>(() => {
    const preferred = normalizeNotesTitleStrategy(titleSuggestStrategy)
    if (preferred && allowedTitleStrategies.includes(preferred)) {
      return preferred
    }
    const fromServer = normalizeNotesTitleStrategy(
      notesTitleSettings?.effective_strategy ?? notesTitleSettings?.default_strategy
    )
    if (fromServer && allowedTitleStrategies.includes(fromServer)) {
      return fromServer
    }
    return allowedTitleStrategies[0] ?? 'heuristic'
  }, [allowedTitleStrategies, notesTitleSettings, titleSuggestStrategy])

  const titleStrategyOptions = React.useMemo(() => {
    return allowedTitleStrategies.map((strategy) => {
      if (strategy === 'llm') {
        return {
          value: strategy,
          label: t('option:notesSearch.titleStrategyLlm', {
            defaultValue: 'AI-powered'
          })
        }
      }
      if (strategy === 'llm_fallback') {
        return {
          value: strategy,
          label: t('option:notesSearch.titleStrategyLlmFallback', {
            defaultValue: 'AI-powered with fallback'
          })
        }
      }
      return {
        value: strategy,
        label: t('option:notesSearch.titleStrategyHeuristic', {
          defaultValue: 'Quick (from content)'
        })
      }
    })
  }, [allowedTitleStrategies, t])

  const suggestTitle = React.useCallback(async () => {
    if (editorDisabled || titleSuggestionLoading) return
    const sourceContent = content.trim()
    if (!sourceContent) {
      message.warning(
        t('option:notesSearch.titleSuggestEmptyContent', {
          defaultValue: 'Write some note content before generating a title.'
        })
      )
      return
    }

    setTitleSuggestionLoading(true)
    try {
      const response = await bgRequest<any>({
        path: '/api/v1/notes/title/suggest' as any,
        method: 'POST' as any,
        headers: { 'Content-Type': 'application/json' },
        body: {
          content: sourceContent,
          title_strategy: effectiveTitleSuggestStrategy
        }
      })
      const suggested = String(response?.title || '').trim()
      if (!suggested) {
        message.warning(
          t('option:notesSearch.titleSuggestNoResult', {
            defaultValue: 'No title suggestion was returned.'
          })
        )
        return
      }
      const apply = await confirmDanger({
        title: t('option:notesSearch.titleSuggestApplyTitle', {
          defaultValue: 'Apply suggested title?'
        }),
        content: suggested,
        okText: t('option:notesSearch.titleSuggestApplyAction', {
          defaultValue: 'Apply'
        }),
        cancelText: t('option:notesSearch.titleSuggestKeepCurrentAction', {
          defaultValue: 'Keep current'
        })
      })
      if (!apply) return
      setTitle(suggested)
      setIsDirty(true)
      setMonitoringNotice(null)
    } catch (error: any) {
      message.error(String(error?.message || 'Could not generate title'))
    } finally {
      setTitleSuggestionLoading(false)
    }
  }, [
    confirmDanger,
    content,
    editorDisabled,
    effectiveTitleSuggestStrategy,
    setIsDirty,
    setMonitoringNotice,
    message,
    t,
    titleSuggestionLoading
  ])

  // ---- assist actions ----
  const runAssistAction = React.useCallback(
    async (action: NotesAssistAction) => {
      if (editorDisabled || assistLoadingAction) return
      const sourceContent = content.trim()
      if (!sourceContent) {
        message.warning(
          t('option:notesSearch.assistEmptyContentWarning', {
            defaultValue: 'Write some content before using assist actions.'
          })
        )
        return
      }

      setAssistLoadingAction(action)
      try {
        if (action === 'suggest_keywords') {
          const suggestedKeywords = suggestKeywordsDraft(sourceContent, editorKeywords)
          if (suggestedKeywords.length === 0) {
            message.warning(
              t('option:notesSearch.assistKeywordsNoResult', {
                defaultValue: 'No additional tag suggestions were found.'
              })
            )
            return
          }
          keywordSuggestionReturnFocusRef.current =
            document.activeElement instanceof HTMLElement ? document.activeElement : null
          setKeywordSuggestionOptions(suggestedKeywords)
          setKeywordSuggestionSelection(suggestedKeywords)
          return
        }

        const generatedContent =
          action === 'summarize'
            ? buildSummaryDraft(sourceContent)
            : buildOutlineDraft(sourceContent)
        if (!generatedContent.trim()) {
          message.warning(
            t('option:notesSearch.assistNoResult', {
              defaultValue: 'No assist output was generated.'
            })
          )
          return
        }

        const apply = await confirmDanger({
          title:
            action === 'summarize'
              ? t('option:notesSearch.assistSummaryApplyTitle', {
                  defaultValue: 'Apply generated summary?'
                })
              : t('option:notesSearch.assistOutlineApplyTitle', {
                  defaultValue: 'Apply expanded outline?'
                }),
          content: generatedContent,
          okText: t('option:notesSearch.assistApplyAction', {
            defaultValue: 'Apply'
          }),
          cancelText: t('option:notesSearch.assistKeepCurrentAction', {
            defaultValue: 'Keep current'
          })
        })
        if (!apply) return
        // Store content before replacement so the user can undo the AI change
        contentBeforeAssistRef.current = sourceContent
        setCanUndoAssist(true)
        if (assistUndoTimerRef.current != null) window.clearTimeout(assistUndoTimerRef.current)
        assistUndoTimerRef.current = window.setTimeout(() => {
          contentBeforeAssistRef.current = null
          setCanUndoAssist(false)
          assistUndoTimerRef.current = null
        }, 30_000)
        setContentDirty(generatedContent, { provenance: action })
        message.success(
          action === 'summarize'
            ? t('option:notesSearch.assistSummaryApplied', {
                defaultValue: 'Applied generated summary.'
              })
            : t('option:notesSearch.assistOutlineApplied', {
                defaultValue: 'Applied expanded outline.'
              })
        )
      } catch (error: any) {
        message.error(String(error?.message || 'Assist action failed'))
      } finally {
        setAssistLoadingAction(null)
      }
    },
    [
      assistLoadingAction,
      confirmDanger,
      content,
      editorDisabled,
      editorKeywords,
      keywordSuggestionReturnFocusRef,
      message,
      setContentDirty,
      setKeywordSuggestionOptions,
      setKeywordSuggestionSelection,
      t
    ]
  )

  const undoAssist = React.useCallback(() => {
    if (contentBeforeAssistRef.current == null) return
    setContentDirty(contentBeforeAssistRef.current, { provenance: 'manual' })
    contentBeforeAssistRef.current = null
    setCanUndoAssist(false)
    if (assistUndoTimerRef.current != null) {
      window.clearTimeout(assistUndoTimerRef.current)
      assistUndoTimerRef.current = null
    }
    message.info(
      t('option:notesSearch.undoAssistApplied', {
        defaultValue: 'Reverted AI change.'
      })
    )
  }, [setContentDirty, message, t])

  // Clean up assist undo timer on unmount
  React.useEffect(() => {
    return () => {
      if (assistUndoTimerRef.current != null) window.clearTimeout(assistUndoTimerRef.current)
    }
  }, [])

  // ---- pinned notes ----
  const selectedNotePinned =
    selectedId != null && pinnedNoteIdSet.has(String(selectedId))

  const toggleNotePinned = React.useCallback(
    async (id: string | number) => {
      if (!pinnedNotesSetting) return
      const targetId = String(id || '').trim()
      if (!targetId) return
      const currentlyPinned = pinnedNoteIdSet.has(targetId)
      const nextPinnedIds = currentlyPinned
        ? pinnedNoteIds.filter((entry) => entry !== targetId)
        : [targetId, ...pinnedNoteIds.filter((entry) => entry !== targetId)].slice(0, 500)
      setPinnedNoteIds(nextPinnedIds)
      try {
        await setSetting(pinnedNotesSetting, nextPinnedIds)
      } catch {
        // Keep UI state even if persistence fails.
      }
      if (currentlyPinned) {
        message.info('Note unpinned')
      } else {
        message.success('Note pinned to top')
      }
    },
    [message, pinnedNoteIdSet, pinnedNoteIds, pinnedNotesSetting, setPinnedNoteIds]
  )

  // ---- effects ----

  // Ctrl+S save shortcut
  React.useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const lowered = event.key.toLowerCase()
      const hasModifier = event.ctrlKey || event.metaKey
      if (!hasModifier || lowered !== 's' || event.altKey || event.shiftKey) return
      if (!isEditorSaveShortcutContext(event.target)) return
      event.preventDefault()
      if (editorDisabled) return
      void saveNote()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [editorDisabled, saveNote])

  // Autosave: debounce (with a max wait) while dirty, back off after transient
  // failures, and never run during a conflict, a fatal error or while the
  // offline queue owns the draft (NS-N2). The machine decides; this only times.
  const offlineSyncStateForEditor = currentOfflineDraft?.syncState
  React.useEffect(() => {
    clearAutosaveTimeout()
    if (editorDisabled || saving) return
    if (!content.trim() && !title.trim()) return
    const delay = nextAutosaveDelayMs(saveState, {
      now: Date.now(),
      lastEditAt: lastEditAtRef.current,
      dirtySince: dirtySinceRef.current,
      lastFailureAt: lastFailureAtRef.current
    })
    if (delay == null) return
    autosaveTimeoutRef.current = window.setTimeout(() => {
      autosaveTimeoutRef.current = null
      void saveNoteRef.current?.({ showSuccessMessage: false, trigger: 'auto' })
    }, delay)
    return clearAutosaveTimeout
  }, [clearAutosaveTimeout, content, editorDisabled, isDirty, offlineSyncStateForEditor, saveState, saving, title])

  // A newer server version while local edits are pending is a conflict now,
  // not a modal at the next autosave (NS-N2, NS-03).
  React.useEffect(() => {
    if (!isDirty || remoteVersionInfo == null || selectedVersion == null) return
    if (remoteVersionInfo.version <= selectedVersion) return
    dispatchSave({ type: 'remote-changed' })
  }, [dispatchSave, isDirty, remoteVersionInfo, selectedVersion])

  // beforeunload: a full reload or tab close while edits are unsaved or saving.
  React.useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (!isDirtyRef.current && !savingInFlightRef.current) return
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [isDirty, saving])

  // Backstop for anything the leave guard did not hold (an account switch,
  // a layout teardown, the tab being closed after the beforeunload prompt):
  // keep unsaved edits in the offline queue so they come back (NS-01).
  const persistLeftoverDraftRef = React.useRef<() => void>(() => {})
  persistLeftoverDraftRef.current = () => {
    if (!isDirtyRef.current && !savingInFlightRef.current && !interruptedSaveRef.current) return
    if (discardedRevisionRef.current === editRevisionRef.current.revision) return
    const snapshot = editRevisionRef.current
    if (!snapshot.title.trim() && !snapshot.content.trim()) return
    persistOfflineDraft({
      syncState: saveStateRef.current.status === 'conflict' ? 'conflict' : 'queued',
      lastError: null
    })
  }
  React.useEffect(() => {
    const onPageHide = () => persistLeftoverDraftRef.current()
    window.addEventListener('pagehide', onPageHide)
    return () => {
      window.removeEventListener('pagehide', onPageHide)
      persistLeftoverDraftRef.current()
    }
  }, [])

  // Reset editor mode on note change. The WYSIWYG session flag is reset by the
  // paths that load a document (loadDetail, resetEditor, offline drafts), not
  // here: a new note's first save also changes selectedId, and clearing the
  // flag then would let exitWysiwygMode discard text typed during that save.
  React.useEffect(() => {
    setEditorMode('edit')
    setManualLinkTargetId(null)
    setRemoteVersionInfo(null)
    setEditorCursorIndex(null)
  }, [selectedId])

  // Wysiwyg sync
  React.useEffect(() => {
    if (editorInputMode !== 'wysiwyg') return
    if (wysiwygSessionDirty) return
    replaceWysiwygHtml(markdownToWysiwygHtml(content))
  }, [content, editorInputMode, replaceWysiwygHtml, wysiwygSessionDirty])

  // Resize textarea
  React.useEffect(() => {
    resizeEditorTextarea()
  }, [content, editorInputMode, editorMode, resizeEditorTextarea])

  React.useEffect(() => {
    if (typeof window === 'undefined') return
    const onResize = () => resizeEditorTextarea()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [resizeEditorTextarea])

  // Freshness check
  const checkSelectedNoteFreshness = React.useCallback(async () => {
    if (!isOnline || listMode !== 'active') return
    if (selectedId == null || selectedVersion == null) return
    if (saving) return
    try {
      const detail = await bgRequest<any>({
        path: noteResourcePath(selectedId) as any,
        method: 'GET' as any
      })
      const remoteVersion = toNoteVersion(detail)
      if (remoteVersion != null && remoteVersion > selectedVersion) {
        setRemoteVersionInfo({
          version: remoteVersion,
          lastModified: toNoteLastModified(detail)
        })
      } else {
        setRemoteVersionInfo(null)
      }
    } catch {
      // Ignore transient freshness-check failures.
    }
  }, [isOnline, listMode, saving, selectedId, selectedVersion])

  React.useEffect(() => {
    if (listMode !== 'active') {
      setRemoteVersionInfo(null)
      return
    }
    if (selectedId == null || selectedVersion == null) return
    let cancelled = false
    const runCheck = async () => {
      if (cancelled) return
      await checkSelectedNoteFreshness()
    }
    void runCheck()
    const intervalId = window.setInterval(() => {
      void runCheck()
    }, 30_000)
    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [checkSelectedNoteFreshness, listMode, selectedId, selectedVersion])

  // Offline draft persistence
  React.useEffect(() => {
    restoredInitialOfflineDraftRef.current = false
    if (!offlineDraftStorageKey) return
    if (typeof window === 'undefined') {
      setOfflineDraftQueueHydrated(true)
      return
    }
    try {
      const raw = window.localStorage.getItem(offlineDraftStorageKey)
      if (!raw) {
        setOfflineDraftQueue({})
        return
      }
      const parsed = JSON.parse(raw)
      setOfflineDraftQueue(normalizeOfflineDraftQueue(parsed))
    } catch {
      setOfflineDraftQueue({})
    } finally {
      setOfflineDraftQueueHydrated(true)
    }
  }, [offlineDraftStorageKey, setOfflineDraftQueue, setOfflineDraftQueueHydrated])

  React.useEffect(() => {
    if (!offlineDraftQueueHydrated) return
    if (!offlineDraftStorageKey) return
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(
        offlineDraftStorageKey,
        JSON.stringify(offlineDraftQueue)
      )
    } catch {
      // Ignore localStorage quota/transient persistence failures.
    }
  }, [offlineDraftQueue, offlineDraftQueueHydrated, offlineDraftStorageKey])

  React.useEffect(() => {
    if (!offlineDraftQueueHydrated) return
    if (restoredInitialOfflineDraftRef.current) return
    if (selectedId != null) return
    if (title.trim().length > 0 || content.trim().length > 0 || editorKeywords.length > 0) return
    const draft = offlineDraftQueue[NOTES_OFFLINE_NEW_DRAFT_KEY]
    if (!draft) return
    restoredInitialOfflineDraftRef.current = true
    applyOfflineDraftToEditor(draft)
  }, [
    applyOfflineDraftToEditor,
    content,
    editorKeywords,
    offlineDraftQueue,
    offlineDraftQueueHydrated,
    selectedId,
    title
  ])

  // While offline, or while saves keep failing to reach the server, keep the
  // newest edits on this device instead of only in memory (NS-05).
  const serverUnreachable = !isOnline || saveState.retryAttempt > 0
  React.useEffect(() => {
    if (!offlineDraftQueueHydrated) return
    if (!serverUnreachable) return
    if (editorDisabled) return
    if (!isDirty) return
    if (!content.trim() && !title.trim() && editorKeywords.length === 0) return
    const timeoutId = window.setTimeout(() => {
      upsertOfflineDraft({
        syncState: 'queued',
        lastError: null
      })
    }, 250)
    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [
    content,
    editorDisabled,
    editorKeywords,
    isDirty,
    offlineDraftQueueHydrated,
    serverUnreachable,
    title,
    upsertOfflineDraft
  ])

  // Sync once the queue is loaded and whenever the server comes back online.
  // Never re-run on callback identity: a draft that keeps failing would
  // otherwise resync on every render.
  React.useEffect(() => {
    if (!offlineDraftQueueHydrated) return
    if (!isOnline) return
    void syncOfflineDraftQueueRef.current()
  }, [authorityScope, isOnline, offlineDraftQueueHydrated])

  // Once the queue owns drafts, keep trying: when the browser reports it is
  // back online and on a slow timer (the 30 s health poll may never flip).
  const hasQueuedDrafts = queuedOfflineDraftCount > 0
  React.useEffect(() => {
    if (!offlineDraftQueueHydrated || !hasQueuedDrafts) return
    const retry = () => {
      void syncOfflineDraftQueueRef.current()
    }
    const intervalId = window.setInterval(retry, NOTES_OFFLINE_QUEUE_RETRY_MS)
    window.addEventListener('online', retry)
    return () => {
      window.clearInterval(intervalId)
      window.removeEventListener('online', retry)
    }
  }, [hasQueuedDrafts, offlineDraftQueueHydrated])

  // Persist title strategy and recent/pinned
  React.useEffect(() => {
    let cancelled = false
    void (async () => {
      const savedStrategy = await getSetting(NOTES_TITLE_SUGGEST_STRATEGY_SETTING)
      if (cancelled) return
      const normalized = normalizeNotesTitleStrategy(savedStrategy)
      if (!normalized) return
      setTitleSuggestStrategy(normalized)
    })()
    return () => {
      cancelled = true
    }
  }, [])

  React.useEffect(() => {
    let cancelled = false
    if (!recentNotesSetting) return
    const revision = recentRevisionRef.current
    void (async () => {
      const savedRecent = await getSetting(recentNotesSetting)
      if (cancelled || recentRevisionRef.current !== revision) return
      if (!Array.isArray(savedRecent)) return
      recentNotesRef.current = savedRecent
      setRecentNotes(savedRecent)
    })()
    return () => {
      cancelled = true
    }
  }, [recentNotesSetting, setRecentNotes])

  React.useEffect(() => {
    let cancelled = false
    if (!pinnedNotesSetting) return
    void (async () => {
      const savedPinned = await getSetting(pinnedNotesSetting)
      if (cancelled) return
      if (!Array.isArray(savedPinned)) return
      const normalized = savedPinned
        .map((entry) => String(entry || '').trim())
        .filter((entry) => entry.length > 0)
        .filter((entry, index, arr) => arr.indexOf(entry) === index)
        .slice(0, 500)
      setPinnedNoteIds(normalized)
    })()
    return () => {
      cancelled = true
    }
  }, [pinnedNotesSetting, setPinnedNoteIds])

  // Cleanup
  React.useEffect(() => {
    return () => {
      clearAutosaveTimeout()
    }
  }, [clearAutosaveTimeout])

  // ---- computed values ----
  const offlineStatusText = React.useMemo(() => {
    if (!offlineDraftQueueHydrated) return null
    if (!isOnline) {
      if (currentOfflineDraft) {
        return t('option:notesSearch.offlineDraftQueuedStatus', {
          defaultValue: 'Offline: changes stored locally and queued for sync.'
        })
      }
      return t('option:notesSearch.offlineEditingStatus', {
        defaultValue: 'Offline: local draft persistence is active.'
      })
    }
    if (!currentOfflineDraft && queuedOfflineDraftCount <= 0) return null
    if (currentOfflineDraft?.syncState === 'syncing') {
      return t('option:notesSearch.offlineSyncingStatus', {
        defaultValue: 'Syncing queued offline draft...'
      })
    }
    if (currentOfflineDraft?.syncState === 'conflict') {
      return t('option:notesSearch.offlineConflictStatus', {
        defaultValue:
          'Offline sync conflict: server has a newer version. Your local draft is still saved; copy it before reloading.'
      })
    }
    if (currentOfflineDraft?.syncState === 'error') {
      return t('option:notesSearch.offlineSyncErrorStatus', {
        defaultValue: 'Queued sync failed. Will retry automatically on reconnect.'
      })
    }
    if (currentOfflineDraft?.syncState === 'queued' && saveState.status === 'offline-queued') {
      return t('option:notesSearch.offlineSavedOnDeviceStatus', {
        defaultValue: 'Saved on this device. It will sync when the server is reachable.'
      })
    }
    if (queuedOfflineDraftCount > 0) {
      return t('option:notesSearch.offlineQueuedCountStatus', {
        defaultValue: '{{count}} offline draft(s) pending sync.',
        count: queuedOfflineDraftCount
      })
    }
    return null
  }, [currentOfflineDraft, isOnline, offlineDraftQueueHydrated, queuedOfflineDraftCount, saveState.status, t])

  // The one save-problem surface (NS-03): conflict, error, retrying, or a
  // newer server copy of a note without local edits. Offline is the pill plus
  // the offline status line.
  const remoteIsNewer =
    remoteVersionInfo != null && selectedVersion != null && remoteVersionInfo.version > selectedVersion
  const saveIssueKind = noteSaveIssueKind(saveState)
  const draftKeptOnDevice = currentOfflineDraft != null && currentOfflineDraft.syncState !== 'conflict'
  const saveIssue = React.useMemo<NotesSaveIssue | null>(() => {
    const busy = saveState.status === 'saving'
    if (saveIssueKind === 'conflict') {
      const changedAt = remoteVersionInfo?.lastModified ? new Date(remoteVersionInfo.lastModified) : null
      const when = changedAt && !Number.isNaN(changedAt.getTime())
        ? ` ${t('option:notesSearch.saveConflictChangedAt', { defaultValue: 'Changed at' })} ${changedAt.toLocaleTimeString()}.`
        : ''
      return {
        kind: 'conflict',
        message: `${describeSaveFailure({ kind: 'conflict', status: 409, message: '' }, false)}${when}`,
        busy
      }
    }
    if (saveIssueKind === 'error') {
      return { kind: 'error', message: describeSaveFailure(saveState.failure, false), busy }
    }
    if (saveIssueKind === 'retrying') {
      return { kind: 'retrying', message: describeSaveFailure(saveState.failure, draftKeptOnDevice), busy }
    }
    if (saveIssueKind == null && remoteIsNewer && !isDirty) {
      return {
        kind: 'remote-newer',
        message: t('option:notesSearch.staleVersionWarning', {
          defaultValue: 'This note was updated elsewhere. Reload to see the latest version.'
        }),
        busy: false
      }
    }
    return null
  }, [
    describeSaveFailure,
    draftKeptOnDevice,
    isDirty,
    remoteIsNewer,
    remoteVersionInfo,
    saveIssueKind,
    saveState.failure,
    saveState.status,
    t
  ])

  /** Pill tooltip: where the note is stored and which version (NS-06). */
  const serverHost = React.useMemo(() => {
    try {
      return connectionConfig?.serverUrl ? new URL(connectionConfig.serverUrl).host : null
    } catch {
      return null
    }
  }, [connectionConfig?.serverUrl])

  // ---- conflict choices (NS-03) ----
  const copyMyText = React.useCallback(async (): Promise<boolean> => {
    const local = editRevisionRef.current
    const copied = await writeClipboardText(
      buildSingleNoteCopyText(
        { id: selectedIdRef.current ?? 'draft', title: local.title, content: local.content, keywords: local.editorKeywords },
        'markdown'
      )
    )
    if (copied) {
      message.success(t('option:notesSearch.copiedMyText', { defaultValue: 'Copied your version to the clipboard.' }))
    } else {
      message.warning(
        t('option:notesSearch.reloadConflictCopyFailed', {
          defaultValue: 'Could not copy your unsaved edits to the clipboard.'
        })
      )
    }
    return copied
  }, [message, t])

  /** Explicit overwrite: save over the server's latest version, never a stale one. */
  const keepMyVersion = React.useCallback(async (): Promise<boolean> => {
    const noteId = selectedIdRef.current
    if (noteId == null || !saveNoteRef.current) return false
    clearAutosaveTimeout()
    let serverVersion: number | null = null
    try {
      serverVersion = toNoteVersion(
        await bgRequest<any>({ path: noteResourcePath(noteId) as any, method: 'GET' as any })
      )
    } catch {
      serverVersion = null
    }
    if (serverVersion == null) {
      message.error(
        t('option:notesSearch.keepMineLoadFailed', {
          defaultValue: 'Could not read the latest server version. Check the connection and try again.'
        })
      )
      return false
    }
    return saveNoteRef.current({ showSuccessMessage: true, trigger: 'keep-mine', expectedVersion: serverVersion })
  }, [clearAutosaveTimeout, message, t])

  /** Load the server copy; the local text goes to the clipboard first. */
  const takeTheirVersion = React.useCallback(async (): Promise<boolean> => {
    const noteId = selectedIdRef.current
    if (noteId == null) return false
    clearAutosaveTimeout()
    const local = editRevisionRef.current
    // Must run before any other await (see writeClipboardText).
    const copied = await writeClipboardText(
      buildSingleNoteCopyText(
        { id: noteId, title: local.title, content: local.content, keywords: local.editorKeywords },
        'markdown'
      )
    )
    if (!copied) {
      const ok = await confirmDanger({
        title: t('option:notesSearch.useTheirVersionTitle', { defaultValue: 'Use their version?' }),
        content: t('option:notesSearch.useTheirVersionNoCopy', {
          defaultValue: 'Your text could not be copied to the clipboard. Using their version discards your edits.'
        }),
        okText: t('option:notesSearch.useTheirVersion', { defaultValue: 'Use their version' }),
        cancelText: t('option:notesSearch.keepEditingAction', { defaultValue: 'Keep editing' })
      })
      if (!ok) return false
    }
    dropOfflineDraftNow(offlineDraftKeyFor(noteId))
    const loaded = await loadDetail(noteId)
    if (loaded) {
      void refetch()
      message.info({
        content: copied
          ? t('option:notesSearch.reloadConflictCopied', {
              defaultValue: 'Loaded the latest version from the server. Your unsaved edits were copied to the clipboard.'
            })
          : t('option:notesSearch.loadedTheirVersion', { defaultValue: 'Loaded the latest version from the server.' }),
        duration: 8
      })
    }
    return loaded
  }, [clearAutosaveTimeout, confirmDanger, dropOfflineDraftNow, loadDetail, message, refetch, t])

  /** "Reload" for a newer server copy when there is nothing local to lose. */
  const loadLatestVersion = React.useCallback(async (): Promise<boolean> => {
    const noteId = selectedIdRef.current
    if (noteId == null) return false
    if (isDirtyRef.current) return takeTheirVersion()
    return loadDetail(noteId)
  }, [loadDetail, takeTheirVersion])

  const retrySave = React.useCallback(
    () => saveNoteRef.current?.({ showSuccessMessage: true, trigger: 'retry' }) ?? Promise.resolve(false),
    []
  )

  // ---- leave guard (NS-01) ----
  const flushBeforeLeave = React.useCallback(
    () => confirmDiscardIfDirtyRef.current(undefined, { trigger: 'leave' }),
    []
  )
  const leaveGuard = React.useMemo<NotesLeaveGuardState>(
    () => ({ when: isDirty || saving, onLeave: flushBeforeLeave }),
    [flushBeforeLeave, isDirty, saving]
  )

  const editorMetrics = React.useMemo(() => {
    const chars = content.length
    const words = content.trim().length > 0 ? content.trim().split(/\s+/).filter(Boolean).length : 0
    const readingTimeMinutes = words === 0 ? 0 : Math.max(1, Math.ceil(words / 200))
    return { chars, words, readingTimeMinutes }
  }, [content])

  const metricSummaryText = React.useMemo(() => {
    const wordLabel = editorMetrics.words === 1 ? 'word' : 'words'
    const charLabel = editorMetrics.chars === 1 ? 'char' : 'chars'
    const readLabel = editorMetrics.readingTimeMinutes === 1 ? 'min read' : 'mins read'
    return `${editorMetrics.words} ${wordLabel} · ${editorMetrics.chars} ${charLabel} · ${editorMetrics.readingTimeMinutes} ${readLabel}`
  }, [editorMetrics])

  const revisionSummaryText = React.useMemo(() => {
    const versionText =
      selectedVersion != null
        ? `${t('option:notesSearch.versionMetadata', {
            defaultValue: 'Version'
          })} ${selectedVersion}`
        : t('option:notesSearch.versionMetadataPending', {
            defaultValue: 'Version pending'
          })

    const lastSavedText = selectedLastSavedAt
      ? `${t('option:notesSearch.lastSavedMetadata', {
          defaultValue: 'Last saved'
        })} ${new Date(selectedLastSavedAt).toLocaleString()}`
      : t('option:notesSearch.lastSavedMetadataPending', {
          defaultValue: 'Not saved yet'
        })

    return `${versionText} · ${lastSavedText}`
  }, [selectedLastSavedAt, selectedVersion, t])

  const saveStatusDetail = React.useMemo(() => {
    const where = serverHost
      ? `${t('option:notesSearch.storedOnServer', { defaultValue: 'Stored on' })} ${serverHost}`
      : null
    return [revisionSummaryText, where].filter(Boolean).join(' · ')
  }, [revisionSummaryText, serverHost, t])

  const provenanceSummaryText = React.useMemo(() => {
    if (originalMetadata?.origin === 'knowledge_qa') {
      return [
        t('option:notesSearch.provenanceKnowledgeQa', {
          defaultValue: 'Origin: Knowledge QA',
        }),
        originalMetadata.trust_state,
        originalMetadata.evidence_origin,
        originalMetadata.thread_id
          ? `Session: ${originalMetadata.thread_id}`
          : null,
      ]
        .filter(Boolean)
        .join(' · ')
    }
    if (editProvenance.mode === 'manual') {
      if (backlinkConversationId) {
        return t('option:notesSearch.provenanceChat', { defaultValue: 'Origin: Saved from Chat' })
      }
      return t('option:notesSearch.provenanceManual', {
        defaultValue: 'Origin: Typed manually'
      })
    }
    const actionLabel =
      editProvenance.action === 'summarize'
        ? t('option:notesSearch.assistSummarizeAction', { defaultValue: 'Summarize' })
        : editProvenance.action === 'expand_outline'
          ? t('option:notesSearch.assistExpandOutlineAction', { defaultValue: 'Expand outline' })
          : t('option:notesSearch.assistSuggestKeywordsAction', { defaultValue: 'Suggest tags' })
    const generatedAt = new Date(editProvenance.at).toLocaleTimeString()
    const generatedPrefix = t('option:notesSearch.provenanceGeneratedPrefix', {
      defaultValue: 'Origin: AI-generated'
    })
    return `${generatedPrefix} (${actionLabel} at ${generatedAt})`
  }, [backlinkConversationId, editProvenance, originalMetadata, t])

  const monitoringNoticeClasses = React.useMemo(() => {
    if (!monitoringNotice) return ''
    if (monitoringNotice.severity === 'critical') {
      return 'border-danger/50 bg-danger/10 text-danger'
    }
    if (monitoringNotice.severity === 'warning') {
      return 'border-warn/50 bg-warn/10 text-warn'
    }
    return 'border-primary/40 bg-primary/10 text-primary'
  }, [monitoringNotice])

  return {
    // state
    selectedId, setSelectedId,
    title, setTitle: setEditorTitle,
    content, setContent: setEditorContent,
    loadingDetail,
    loadingSelection: loadingDetail && pendingSelectionEpochRef.current != null,
    saving,
    saveIndicator, setSaveIndicator,
    saveStatus: saveState.status,
    saveIssue,
    saveStatusDetail,
    leaveGuard,
    originalMetadata,
    selectedStudioSummary,
    selectedVersion,
    selectedLastSavedAt,
    isDirty, setIsDirty,
    backlinkConversationId, setBacklinkConversationId,
    backlinkMessageId, setBacklinkMessageId,
    remoteVersionInfo,
    editorMode, setEditorMode,
    editorInputMode, setEditorInputMode,
    wysiwygHtml, wysiwygRevision, replaceWysiwygHtml, recordWysiwygEditorHtml,
    wysiwygSessionDirty, setWysiwygSessionDirty,
    editorCursorIndex, setEditorCursorIndex,
    titleSuggestionLoading,
    assistLoadingAction,
    canUndoAssist,
    editProvenance,
    monitoringNotice, setMonitoringNotice,
    noteTasks,
    taskReconciliation,
    taskActivityEvents,
    taskConflictNotice,
    recentNotes,
    pinnedNoteIds, pinnedNoteIdSet,
    titleSuggestStrategy, setTitleSuggestStrategy,
    graphMutationTick, setGraphMutationTick,
    manualLinkTargetId, setManualLinkTargetId,
    manualLinkSaving, setManualLinkSaving,
    manualLinkDeletingEdgeId, setManualLinkDeletingEdgeId,
    openingLinkedChat, setOpeningLinkedChat,
    offlineDraftQueue,
    offlineDraftQueueHydrated,
    queuedOfflineDraftCount,
    currentOfflineDraft,
    // computed
    offlineStatusText,
    metricSummaryText,
    revisionSummaryText,
    provenanceSummaryText,
    monitoringNoticeClasses,
    selectedNotePinned,
    canSwitchTitleStrategy,
    effectiveTitleSuggestStrategy,
    titleStrategyOptions,
    // refs
    titleInputRef,
    contentTextareaRef,
    richEditorRef,
    attachmentInputRef,
    markdownBeforeWysiwygRef,
    saveNoteRef,
    // callbacks
    clearAutosaveTimeout,
    markManualEdit,
    markGeneratedEdit,
    restoreFocusAfterOverlayClose,
    setContentDirty,
    resizeEditorTextarea,
    loadDetail,
    refreshTaskStateForNote,
    toggleTaskCheckboxLocal,
    toggleTaskCheckboxStatus,
    dismissTaskActivity,
    inspectTaskActivity,
    resetEditor,
    removeRecentNotes,
    confirmDiscardIfDirty,
    switchListMode,
    handleSelectNote,
    openSourceNote,
    saveNote,
    reloadNotes,
    reloadSelectedNoteAfterConflict,
    keepMyVersion,
    takeTheirVersion,
    copyMyText,
    loadLatestVersion,
    retrySave,
    suggestTitle,
    runAssistAction,
    undoAssist,
    toggleNotePinned,
    isVersionConflictError,
    handleVersionConflict,
    getExpectedVersionForNoteId: React.useCallback(
      async (noteId: string): Promise<number | null> => {
        if (selectedId != null && String(selectedId) === noteId && selectedVersion != null) {
          return selectedVersion
        }
        if (Array.isArray(data)) {
          const match = data.find((note) => String(note.id) === noteId)
          if (typeof match?.version === 'number' && Number.isFinite(match.version)) {
            return match.version
          }
        }
        try {
          const detail = await bgRequest<any>({
            path: noteResourcePath(noteId) as any,
            method: 'GET' as any
          })
          return toNoteVersion(detail)
        } catch {
          return null
        }
      },
      [data, selectedId, selectedVersion]
    ),
  }
}
