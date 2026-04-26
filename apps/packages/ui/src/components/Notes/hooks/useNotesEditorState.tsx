import React from 'react'
import type { InputRef } from 'antd'
import { Button, Modal } from 'antd'
import type { MessageInstance } from 'antd/es/message/interface'
import type { QueryClient } from '@tanstack/react-query'
import { useQuery } from '@tanstack/react-query'
import { bgRequest } from '@/services/background-proxy'
import { getSetting, setSetting, clearSetting } from '@/services/settings/registry'
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
  extractBacklink,
  extractKeywords,
  toNoteVersion,
  toNoteLastModified,
  toKeywordSyncWarning,
  NOTE_AUTOSAVE_DELAY_MS,
  NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY,
  NOTES_OFFLINE_NEW_DRAFT_KEY,
  isEditorSaveShortcutContext,
  normalizeOfflineDraftQueue,
  NOTES_TITLE_STRATEGIES,
  normalizeNotesTitleStrategy,
  deriveAllowedTitleStrategies,
  markdownToWysiwygHtml,
  wysiwygHtmlToMarkdown,
  buildSummaryDraft,
  buildOutlineDraft,
  suggestKeywordsDraft,
  NOTE_ASSIST_STOP_WORDS,
  toAttachmentMarkdown,
} from '../notes-manager-utils'
import type { NoteStudioDocumentSummary } from '../notes-studio-types'

type ConfirmDanger = (options: ConfirmDangerOptions) => Promise<boolean>

export interface UseNotesEditorStateDeps {
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
}

export function useNotesEditorState(deps: UseNotesEditorStateDeps) {
  const {
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

  // ---- editor state ----
  const [selectedId, setSelectedId] = React.useState<string | number | null>(null)
  const [title, setTitle] = React.useState('')
  const [content, setContent] = React.useState('')
  const [loadingDetail, setLoadingDetail] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [saveIndicator, setSaveIndicator] = React.useState<SaveIndicatorState>('idle')
  const [originalMetadata, setOriginalMetadata] = React.useState<Record<string, any> | null>(null)
  const [selectedStudioSummary, setSelectedStudioSummary] =
    React.useState<NoteStudioDocumentSummary | null>(null)
  const [selectedVersion, setSelectedVersion] = React.useState<number | null>(null)
  const [selectedLastSavedAt, setSelectedLastSavedAt] = React.useState<string | null>(null)
  const [isDirty, setIsDirty] = React.useState(false)
  const [backlinkConversationId, setBacklinkConversationId] = React.useState<string | null>(null)
  const [backlinkMessageId, setBacklinkMessageId] = React.useState<string | null>(null)
  const [remoteVersionInfo, setRemoteVersionInfo] = React.useState<RemoteVersionInfo | null>(null)
  const [editorMode, setEditorMode] = React.useState<NotesEditorMode>('edit')
  const [editorInputMode, setEditorInputMode] = React.useState<NotesInputMode>('markdown')
  const [wysiwygHtml, setWysiwygHtml] = React.useState<string>('<p><br/></p>')
  const [wysiwygSessionDirty, setWysiwygSessionDirty] = React.useState(false)
  const [editorCursorIndex, setEditorCursorIndex] = React.useState<number | null>(null)
  const [titleSuggestionLoading, setTitleSuggestionLoading] = React.useState(false)
  const [assistLoadingAction, setAssistLoadingAction] = React.useState<NotesAssistAction | null>(null)
  const [editProvenance, setEditProvenance] = React.useState<EditProvenanceState>({ mode: 'manual' })
  const [monitoringNotice, setMonitoringNotice] = React.useState<MonitoringNoticeState | null>(null)
  const [recentNotes, setRecentNotes] = React.useState<NotesRecentOpenedEntry[]>([])
  const [pinnedNoteIds, setPinnedNoteIds] = React.useState<string[]>([])
  const [titleSuggestStrategy, setTitleSuggestStrategy] =
    React.useState<NotesTitleSuggestStrategy>('heuristic')
  const [graphModalOpen, setGraphModalOpen] = React.useState(false)
  const [graphMutationTick, setGraphMutationTick] = React.useState(0)
  const [manualLinkTargetId, setManualLinkTargetId] = React.useState<string | null>(null)
  const [manualLinkSaving, setManualLinkSaving] = React.useState(false)
  const [manualLinkDeletingEdgeId, setManualLinkDeletingEdgeId] = React.useState<string | null>(null)
  const [openingLinkedChat, setOpeningLinkedChat] = React.useState(false)

  // ---- offline draft state ----
  const [offlineDraftQueue, setOfflineDraftQueue] = React.useState<Record<string, OfflineDraftEntry>>({})
  const [offlineDraftQueueHydrated, setOfflineDraftQueueHydrated] = React.useState(false)
  const offlineDraftQueueRef = React.useRef<Record<string, OfflineDraftEntry>>({})
  offlineDraftQueueRef.current = offlineDraftQueue
  const offlineSyncInFlightRef = React.useRef(false)
  const restoredInitialOfflineDraftRef = React.useRef(false)

  // ---- refs ----
  const autosaveTimeoutRef = React.useRef<number | null>(null)
  const saveNoteRef = React.useRef<((opts?: { showSuccessMessage?: boolean }) => Promise<void>) | null>(null)
  const titleInputRef = React.useRef<InputRef | null>(null)
  const contentTextareaRef = React.useRef<HTMLTextAreaElement | null>(null)
  const richEditorRef = React.useRef<HTMLDivElement | null>(null)
  const attachmentInputRef = React.useRef<HTMLInputElement | null>(null)
  const markdownBeforeWysiwygRef = React.useRef<string | null>(null)
  const graphModalReturnFocusRef = React.useRef<HTMLElement | null>(null)

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

  const buildCurrentOfflineDraft = React.useCallback(
    (
      overrides?: Partial<Pick<OfflineDraftEntry, 'syncState' | 'lastError' | 'updatedAt' | 'baseVersion'>>
    ): OfflineDraftEntry => {
      const nowIso = overrides?.updatedAt || new Date().toISOString()
      return {
        key: currentOfflineDraftKey,
        noteId: selectedId != null ? String(selectedId) : null,
        baseVersion:
          overrides?.baseVersion !== undefined
            ? overrides.baseVersion
            : selectedVersion != null
              ? selectedVersion
              : null,
        title,
        content,
        keywords: [...editorKeywords],
        metadata: originalMetadata ? { ...originalMetadata } : null,
        backlinkConversationId,
        backlinkMessageId,
        updatedAt: nowIso,
        syncState: overrides?.syncState || 'queued',
        lastError: overrides?.lastError ?? null
      }
    },
    [
      backlinkConversationId,
      backlinkMessageId,
      content,
      currentOfflineDraftKey,
      editorKeywords,
      originalMetadata,
      selectedId,
      selectedVersion,
      title
    ]
  )

  const applyOfflineDraftToEditor = React.useCallback((draft: OfflineDraftEntry) => {
    setTitle(String(draft.title || ''))
    setContent(String(draft.content || ''))
    setEditorKeywords(Array.isArray(draft.keywords) ? [...draft.keywords] : [])
    setOriginalMetadata(
      draft.metadata && typeof draft.metadata === 'object'
        ? { ...(draft.metadata as Record<string, any>) }
        : null
    )
    setBacklinkConversationId(draft.backlinkConversationId)
    setBacklinkMessageId(draft.backlinkMessageId)
    if (draft.baseVersion != null) {
      setSelectedVersion(draft.baseVersion)
    }
    setSelectedLastSavedAt(draft.updatedAt)
    setIsDirty(false)
    setSaveIndicator(draft.syncState === 'conflict' || draft.syncState === 'error' ? 'error' : 'saved')
    setEditProvenance({ mode: 'manual' })
    setMonitoringNotice(null)
    setRemoteVersionInfo(null)
    setEditorCursorIndex(null)
    setWysiwygHtml(markdownToWysiwygHtml(String(draft.content || '')))
    setWysiwygSessionDirty(false)
    markdownBeforeWysiwygRef.current = String(draft.content || '')
  }, [setEditorKeywords])

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
    [buildCurrentOfflineDraft]
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
  }, [])

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
      setContent(nextContent)
      setIsDirty(true)
      setSaveIndicator('dirty')
      setMonitoringNotice(null)
      if (options?.provenance && options.provenance !== 'manual') {
        markGeneratedEdit(options.provenance)
      } else {
        markManualEdit()
      }
    },
    [markGeneratedEdit, markManualEdit]
  )

  // ---- load/reset ----
  const rememberRecentNote = React.useCallback((noteId: string | number, noteTitle: string) => {
    const normalizedId = String(noteId || '').trim()
    const normalizedTitle = String(noteTitle || '').trim()
    if (!normalizedId || !normalizedTitle) return
    setRecentNotes((current) => {
      const next = [
        { id: normalizedId, title: normalizedTitle },
        ...current.filter((entry) => entry.id !== normalizedId)
      ].slice(0, 5)
      void setSetting(NOTES_RECENT_OPENED_SETTING, next)
      return next
    })
  }, [])

  const loadDetail = React.useCallback(async (id: string | number): Promise<boolean> => {
    clearAssistUndoState()
    setLoadingDetail(true)
    try {
      const d = await bgRequest<any>({ path: `/api/v1/notes/${id}` as any, method: 'GET' as any })
      const loadedTitle = String(d?.title || `Note ${id}`)
      setSelectedId(id)
      setTitle(String(d?.title || ''))
      setContent(String(d?.content || ''))
      setEditorKeywords(extractKeywords(d))
      setSelectedVersion(toNoteVersion(d))
      setSelectedLastSavedAt(toNoteLastModified(d))
      const rawMeta = d && typeof d === "object" ? (d as any).metadata : null
      setOriginalMetadata(
        rawMeta && typeof rawMeta === "object" ? { ...(rawMeta as Record<string, any>) } : null
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
      setIsDirty(false)
      setSaveIndicator('idle')
      setEditProvenance({ mode: 'manual' })
      setMonitoringNotice(null)
      setRemoteVersionInfo(null)
      setEditorCursorIndex(0)
      setWysiwygHtml(markdownToWysiwygHtml(String(d?.content || '')))
      setWysiwygSessionDirty(false)
      markdownBeforeWysiwygRef.current = String(d?.content || '')
      rememberRecentNote(id, loadedTitle)
      const queuedDraft = offlineDraftQueueRef.current[`note:${String(id)}`]
      if (queuedDraft) {
        applyOfflineDraftToEditor(queuedDraft)
      }
      return true
    } catch {
      message.error('Failed to load note')
      return false
    } finally { setLoadingDetail(false) }
  }, [applyOfflineDraftToEditor, clearAssistUndoState, message, rememberRecentNote, setEditorKeywords])

  const resetEditor = React.useCallback(() => {
    clearAssistUndoState()
    setSelectedId(null)
    setTitle('')
    setContent('')
    setEditorKeywords([])
    setOriginalMetadata(null)
    setSelectedStudioSummary(null)
    setSelectedVersion(null)
    setSelectedLastSavedAt(null)
    setBacklinkConversationId(null)
    setBacklinkMessageId(null)
    setIsDirty(false)
    setSaveIndicator('idle')
    setEditProvenance({ mode: 'manual' })
    setMonitoringNotice(null)
    setRemoteVersionInfo(null)
    setEditorCursorIndex(null)
    setWysiwygHtml('<p><br/></p>')
    setWysiwygSessionDirty(false)
    markdownBeforeWysiwygRef.current = null
  }, [clearAssistUndoState, setEditorKeywords])

  const confirmDiscardIfDirty = React.useCallback(async () => {
    if (!isDirty) return true
    if (!saving && (content.trim() || title.trim()) && saveNoteRef.current) {
      try {
        await saveNoteRef.current({ showSuccessMessage: false })
        return true
      } catch {
        // Save failed — show 3-button dialog: retry / discard / cancel
        const retryResult = await new Promise<'saved' | 'discard' | 'cancel'>((resolve) => {
          const modalRef = Modal.confirm({
            title: t('option:notesSearch.unsavedChangesTitle', { defaultValue: 'Save changes?' }),
            content: t('option:notesSearch.unsavedChangesContent', {
              defaultValue: 'Auto-save could not reach the server. You can try saving again or discard your changes.'
            }),
            okText: t('option:notesSearch.retrySave', { defaultValue: 'Try saving again' }),
            cancelText: t('common:cancel', { defaultValue: 'Cancel' }),
            okButtonProps: { type: 'primary' },
            onOk: async () => {
              try {
                if (saveNoteRef.current) {
                  await saveNoteRef.current({ showSuccessMessage: true })
                }
                resolve('saved')
              } catch {
                // Save still failed — stay on current note
                resolve('cancel')
              }
            },
            onCancel: () => resolve('cancel'),
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
        })
        return retryResult === 'saved' || retryResult === 'discard'
      }
    }
    // Fallback for edge cases (empty content or save in progress)
    const ok = await confirmDanger({
      title: t('option:notesSearch.unsavedChangesTitle', { defaultValue: 'Save changes?' }),
      content: t('option:notesSearch.unsavedChangesContent', {
        defaultValue: 'Your changes could not be saved automatically. What would you like to do?'
      }),
      okText: t('option:notesSearch.discardChanges', { defaultValue: 'Discard changes' }),
      cancelText: t('common:cancel', { defaultValue: 'Cancel' })
    })
    return ok
  }, [confirmDanger, content, isDirty, saving, t, title])

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

  const reloadNotes = React.useCallback(async (noteId?: string | number | null) => {
    await refetch()
    const target = noteId ?? selectedId
    if (target == null) return
    try {
      const detail = await bgRequest<any>({ path: `/api/v1/notes/${target}` as any, method: 'GET' as any })
      const version = toNoteVersion(detail)
      if (version != null) setSelectedVersion(version)
      setSelectedLastSavedAt(toNoteLastModified(detail))
      setRemoteVersionInfo(null)
    } catch {
      // Ignore refresh errors for reload action
    }
  }, [refetch, selectedId])

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

  // ---- monitoring ----
  const loadMonitoringNoticeForSavedNote = React.useCallback(
    async (
      noteId: string | number,
      source: 'notes.create' | 'notes.update',
      saveStartedAtMs: number
    ) => {
      try {
        const params = new URLSearchParams()
        params.set('source', source)
        params.set('unread_only', 'true')
        params.set('limit', '50')
        const response = await bgRequest<any>({
          path: `/api/v1/monitoring/alerts?${params.toString()}` as any,
          method: 'GET' as any
        })
        const items = Array.isArray(response?.items) ? response.items : []
        const noteIdText = String(noteId)
        const minCreatedAtMs = saveStartedAtMs - 5000
        const matchedAlert = items.find((item: any) => {
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
      } catch {
        // Endpoint may be disabled or permission-gated
      }
    },
    [t]
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
  const saveNote = React.useCallback(
    async ({ showSuccessMessage = true }: SaveNoteOptions = {}) => {
      if (saving) return
      if (!content.trim() && !title.trim()) {
        if (showSuccessMessage) {
          message.warning('Nothing to save')
        }
        setSaveIndicator('idle')
        return
      }
      if (!isOnline) {
        const queuedAt = new Date().toISOString()
        upsertOfflineDraft({
          syncState: 'queued',
          lastError: null,
          updatedAt: queuedAt
        })
        setIsDirty(false)
        setSaveIndicator('saved')
        setSelectedLastSavedAt(queuedAt)
        if (showSuccessMessage) {
          message.info(
            t('option:notesSearch.offlineSavedLocally', {
              defaultValue: 'Saved locally. Sync will resume when connection returns.'
            })
          )
        }
        return
      }
      if (
        selectedId != null &&
        selectedVersion != null &&
        remoteVersionInfo &&
        remoteVersionInfo.version > selectedVersion
      ) {
        const proceed = await confirmDanger({
          title: 'Remote changes detected',
          content:
            `This note changed in another tab or session (local v${selectedVersion}, remote v${remoteVersionInfo.version}). ` +
            'Saving now may cause a conflict. Continue anyway?',
          okText: 'Save anyway',
          cancelText: 'Cancel'
        })
        if (!proceed) {
          return
        }
      }
      setSaving(true)
      setSaveIndicator('saving')
      setMonitoringNotice(null)
      const saveStartedAtMs = Date.now()
      try {
        const metadata: Record<string, any> = {
          ...(originalMetadata || {}),
          keywords: editorKeywords
        }
        if (backlinkConversationId) metadata.conversation_id = backlinkConversationId
        if (backlinkMessageId) metadata.message_id = backlinkMessageId
        const payload: Record<string, any> = {
          title: title || undefined,
          content,
          metadata,
          keywords: editorKeywords
        }
        if (backlinkConversationId) payload.conversation_id = backlinkConversationId
        if (backlinkMessageId) payload.message_id = backlinkMessageId
        if (selectedId == null) {
          const created = await bgRequest<any>({
            path: '/api/v1/notes/' as any,
            method: 'POST' as any,
            headers: { 'Content-Type': 'application/json' },
            body: payload
          })
          const createdKeywordWarning = toKeywordSyncWarning(created)
          const createdVersion = toNoteVersion(created)
          const createdLastSaved = toNoteLastModified(created)
          if (showSuccessMessage) {
            message.success('Note created')
          }
          if (createdKeywordWarning) {
            showKeywordSyncWarning(createdKeywordWarning, 'created')
          }
          setIsDirty(false)
          setSaveIndicator('saved')
          setRemoteVersionInfo(null)
          removeOfflineDraftByKey(currentOfflineDraftKey)
          if (createdVersion != null) setSelectedVersion(createdVersion)
          if (createdLastSaved) setSelectedLastSavedAt(createdLastSaved)
          await refetch()
          if (created?.id != null) {
            await loadDetail(created.id)
            void loadMonitoringNoticeForSavedNote(created.id, 'notes.create', saveStartedAtMs)
          }
        } else {
          let expectedVersion = selectedVersion
          if (expectedVersion == null) {
            try {
              const latest = await bgRequest<any>({
                path: `/api/v1/notes/${selectedId}` as any,
                method: 'GET' as any
              })
              expectedVersion = toNoteVersion(latest)
            } catch (e: any) {
              setSaveIndicator('error')
              if (showSuccessMessage) {
                message.error(e?.message || 'Save failed')
              }
              return
            }
          }
          if (expectedVersion == null) {
            setSaveIndicator('error')
            if (showSuccessMessage) {
              message.error('Missing version; reload and try again')
            }
            return
          }
          const updated = await bgRequest<any>({
            path: `/api/v1/notes/${selectedId}?expected_version=${encodeURIComponent(
              String(expectedVersion)
            )}` as any,
            method: 'PUT' as any,
            headers: { 'Content-Type': 'application/json' },
            body: payload
          })
          const updatedKeywordWarning = toKeywordSyncWarning(updated)
          const updatedVersion = toNoteVersion(updated)
          const updatedLastSaved = toNoteLastModified(updated)
          if (showSuccessMessage) {
            message.success('Note updated')
          }
          if (updatedKeywordWarning) {
            showKeywordSyncWarning(updatedKeywordWarning, 'updated')
          }
          setIsDirty(false)
          setSaveIndicator('saved')
          setRemoteVersionInfo(null)
          removeOfflineDraftByKey(currentOfflineDraftKey)
          await refetch()
          if (updatedVersion != null) {
            setSelectedVersion(updatedVersion)
          } else if (selectedId != null) {
            try {
              const latest = await bgRequest<any>({
                path: `/api/v1/notes/${selectedId}` as any,
                method: 'GET' as any
              })
              setSelectedVersion(toNoteVersion(latest))
            } catch (err) {
              console.debug('[NotesManagerPage] Version refresh after save failed:', err)
            }
          }
          if (updatedLastSaved) {
            setSelectedLastSavedAt(updatedLastSaved)
          } else {
            setSelectedLastSavedAt(new Date().toISOString())
          }
          if (selectedId != null) {
            void loadMonitoringNoticeForSavedNote(selectedId, 'notes.update', saveStartedAtMs)
          }
        }
      } catch (e: any) {
        setSaveIndicator('error')
        if (isVersionConflictError(e)) {
          if (showSuccessMessage) {
            handleVersionConflict(selectedId)
          }
        } else if (showSuccessMessage) {
          message.error(String(e?.message || '') || 'Operation failed')
        }
      } finally {
        setSaving(false)
      }
    },
    [
      backlinkConversationId,
      backlinkMessageId,
      confirmDanger,
      content,
      editorKeywords,
      handleVersionConflict,
      isVersionConflictError,
      isOnline,
      loadDetail,
      loadMonitoringNoticeForSavedNote,
      message,
      originalMetadata,
      currentOfflineDraftKey,
      refetch,
      removeOfflineDraftByKey,
      remoteVersionInfo,
      saving,
      selectedId,
      selectedVersion,
      showKeywordSyncWarning,
      t,
      title,
      upsertOfflineDraft
    ]
  )

  // Keep ref in sync
  saveNoteRef.current = saveNote

  // ---- offline sync ----
  const syncOfflineDraftEntry = React.useCallback(
    async (draft: OfflineDraftEntry): Promise<OfflineDraftSyncResult> => {
      const metadata: Record<string, any> = {
        ...(draft.metadata || {}),
        keywords: draft.keywords
      }
      if (draft.backlinkConversationId) metadata.conversation_id = draft.backlinkConversationId
      if (draft.backlinkMessageId) metadata.message_id = draft.backlinkMessageId
      const payload: Record<string, any> = {
        title: draft.title || undefined,
        content: draft.content,
        metadata,
        keywords: draft.keywords
      }
      if (draft.backlinkConversationId) payload.conversation_id = draft.backlinkConversationId
      if (draft.backlinkMessageId) payload.message_id = draft.backlinkMessageId

      try {
        if (!draft.noteId) {
          const created = await bgRequest<any>({
            path: '/api/v1/notes/' as any,
            method: 'POST' as any,
            headers: { 'Content-Type': 'application/json' },
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

        const encodedId = encodeURIComponent(String(draft.noteId))
        const remote = await bgRequest<any>({
          path: `/api/v1/notes/${encodedId}` as any,
          method: 'GET' as any
        })
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
          path: `/api/v1/notes/${encodedId}?expected_version=${encodeURIComponent(
            String(expectedVersion)
          )}` as any,
          method: 'PUT' as any,
          headers: { 'Content-Type': 'application/json' },
          body: payload
        })

        return {
          status: 'synced',
          key: draft.key,
          noteId: String(draft.noteId),
          version: toNoteVersion(updated),
          lastSavedAt: toNoteLastModified(updated)
        }
      } catch (error: any) {
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
      }
    },
    [isVersionConflictError]
  )

  const syncOfflineDraftQueue = React.useCallback(async () => {
    if (!isOnline) return
    if (offlineSyncInFlightRef.current) return
    const queuedEntries = Object.values(offlineDraftQueueRef.current)
      .filter((entry) => entry.syncState !== 'conflict')
      .sort((a, b) => new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime())
    if (queuedEntries.length === 0) return

    offlineSyncInFlightRef.current = true
    let successfulSyncs = 0
    try {
      for (const queuedEntry of queuedEntries) {
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
        if (syncResult.status === 'synced' && syncResult.noteId) {
          successfulSyncs += 1
          setOfflineDraftQueue((current) => {
            if (!current[syncResult.key]) return current
            const next = { ...current }
            delete next[syncResult.key]
            return next
          })
          if (selectedId == null && syncResult.key === NOTES_OFFLINE_NEW_DRAFT_KEY) {
            await loadDetail(syncResult.noteId)
          } else if (
            selectedId != null &&
            syncResult.noteId &&
            String(selectedId) === String(syncResult.noteId)
          ) {
            if (syncResult.version != null) {
              setSelectedVersion(syncResult.version)
            }
            setSelectedLastSavedAt(syncResult.lastSavedAt || new Date().toISOString())
            setSaveIndicator('saved')
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
          if (syncResult.key === currentOfflineDraftKey) {
            setSaveIndicator('error')
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
      offlineSyncInFlightRef.current = false
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
    currentOfflineDraftKey,
    isOnline,
    loadDetail,
    message,
    refetch,
    selectedId,
    syncOfflineDraftEntry,
    t
  ])

  // ---- title suggestion ----
  const { data: notesTitleSettings } = useQuery({
    queryKey: ['notes-title-settings'],
    enabled: isOnline,
    staleTime: 5 * 60 * 1000,
    queryFn: async () => {
      try {
        const settings = await bgRequest<NotesTitleSettingsResponse>({
          path: '/api/v1/admin/notes/title-settings' as any,
          method: 'GET' as any
        })
        return settings
      } catch {
        return null
      }
    }
  })

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
      setSaveIndicator('dirty')
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
      const targetId = String(id || '').trim()
      if (!targetId) return
      const currentlyPinned = pinnedNoteIdSet.has(targetId)
      const nextPinnedIds = currentlyPinned
        ? pinnedNoteIds.filter((entry) => entry !== targetId)
        : [targetId, ...pinnedNoteIds.filter((entry) => entry !== targetId)].slice(0, 500)
      setPinnedNoteIds(nextPinnedIds)
      try {
        await setSetting(NOTES_PINNED_IDS_SETTING, nextPinnedIds)
      } catch {
        // Keep UI state even if persistence fails.
      }
      if (currentlyPinned) {
        message.info('Note unpinned')
      } else {
        message.success('Note pinned to top')
      }
    },
    [message, pinnedNoteIdSet, pinnedNoteIds]
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

  // Autosave
  React.useEffect(() => {
    if (!isDirty || editorDisabled || saving) {
      clearAutosaveTimeout()
      return
    }
    if (!content.trim() && !title.trim()) {
      clearAutosaveTimeout()
      return
    }
    clearAutosaveTimeout()
    autosaveTimeoutRef.current = window.setTimeout(() => {
      void saveNote({ showSuccessMessage: false })
    }, NOTE_AUTOSAVE_DELAY_MS)
    return () => {
      clearAutosaveTimeout()
    }
  }, [clearAutosaveTimeout, content, editorDisabled, isDirty, saveNote, saving, title])

  // beforeunload
  React.useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (!isDirty) return
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [isDirty])

  // Reset editor mode on note change
  React.useEffect(() => {
    setEditorMode('edit')
    setManualLinkTargetId(null)
    setRemoteVersionInfo(null)
    setEditorCursorIndex(null)
    setWysiwygSessionDirty(false)
  }, [selectedId])

  React.useEffect(() => {
    if (selectedId == null) {
      setGraphModalOpen(false)
    }
  }, [selectedId])

  // Wysiwyg sync
  React.useEffect(() => {
    if (editorInputMode !== 'wysiwyg') return
    if (wysiwygSessionDirty) return
    setWysiwygHtml(markdownToWysiwygHtml(content))
  }, [content, editorInputMode, wysiwygSessionDirty])

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
        path: `/api/v1/notes/${selectedId}` as any,
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
    if (typeof window === 'undefined') {
      setOfflineDraftQueueHydrated(true)
      return
    }
    try {
      const raw = window.localStorage.getItem(NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY)
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
  }, [])

  React.useEffect(() => {
    if (!offlineDraftQueueHydrated) return
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(
        NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY,
        JSON.stringify(offlineDraftQueue)
      )
    } catch {
      // Ignore localStorage quota/transient persistence failures.
    }
  }, [offlineDraftQueue, offlineDraftQueueHydrated])

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

  React.useEffect(() => {
    if (!offlineDraftQueueHydrated) return
    if (isOnline) return
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
    isOnline,
    offlineDraftQueueHydrated,
    title,
    upsertOfflineDraft
  ])

  React.useEffect(() => {
    if (!offlineDraftQueueHydrated) return
    if (!isOnline) return
    void syncOfflineDraftQueue()
  }, [isOnline, offlineDraftQueueHydrated, syncOfflineDraftQueue])

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
    void (async () => {
      const savedRecent = await getSetting(NOTES_RECENT_OPENED_SETTING)
      if (cancelled) return
      if (!Array.isArray(savedRecent)) return
      setRecentNotes(savedRecent)
    })()
    return () => {
      cancelled = true
    }
  }, [])

  React.useEffect(() => {
    let cancelled = false
    void (async () => {
      const savedPinned = await getSetting(NOTES_PINNED_IDS_SETTING)
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
  }, [])

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
        defaultValue: 'Offline sync conflict: server has a newer version.'
      })
    }
    if (currentOfflineDraft?.syncState === 'error') {
      return t('option:notesSearch.offlineSyncErrorStatus', {
        defaultValue: 'Queued sync failed. Will retry automatically on reconnect.'
      })
    }
    if (queuedOfflineDraftCount > 0) {
      return t('option:notesSearch.offlineQueuedCountStatus', {
        defaultValue: '{{count}} offline draft(s) pending sync.',
        count: queuedOfflineDraftCount
      })
    }
    return null
  }, [currentOfflineDraft, isOnline, offlineDraftQueueHydrated, queuedOfflineDraftCount, t])

  const saveIndicatorText = React.useMemo(() => {
    if (saveIndicator === 'saving') {
      return t('option:notesSearch.saving', { defaultValue: 'Saving...' })
    }
    if (saveIndicator === 'error') {
      return t('option:notesSearch.autosaveFailed', {
        defaultValue: 'Could not save — check your connection and try again.'
      })
    }
    if (saveIndicator === 'saved' && !isDirty) {
      return t('option:notesSearch.saved', { defaultValue: 'All changes saved' })
    }
    return null
  }, [isDirty, saveIndicator, t])

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

  const provenanceSummaryText = React.useMemo(() => {
    if (editProvenance.mode === 'manual') {
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
  }, [editProvenance, t])

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
    title, setTitle,
    content, setContent,
    loadingDetail,
    saving,
    saveIndicator, setSaveIndicator,
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
    wysiwygHtml, setWysiwygHtml,
    wysiwygSessionDirty, setWysiwygSessionDirty,
    editorCursorIndex, setEditorCursorIndex,
    titleSuggestionLoading,
    assistLoadingAction,
    canUndoAssist,
    editProvenance,
    monitoringNotice, setMonitoringNotice,
    recentNotes,
    pinnedNoteIds, pinnedNoteIdSet,
    titleSuggestStrategy, setTitleSuggestStrategy,
    graphModalOpen, setGraphModalOpen,
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
    saveIndicatorText,
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
    graphModalReturnFocusRef,
    saveNoteRef,
    // callbacks
    clearAutosaveTimeout,
    markManualEdit,
    markGeneratedEdit,
    restoreFocusAfterOverlayClose,
    setContentDirty,
    resizeEditorTextarea,
    loadDetail,
    resetEditor,
    confirmDiscardIfDirty,
    switchListMode,
    handleSelectNote,
    saveNote,
    reloadNotes,
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
            path: `/api/v1/notes/${encodeURIComponent(noteId)}` as any,
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
