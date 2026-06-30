import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Button, Empty, Input, Modal, Spin, Tag, Tooltip } from "antd"
import {
  ChevronLeft,
  ChevronRight,
  FolderOpen,
  Plus,
  RefreshCcw,
  Save,
  StickyNote,
  X
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useNotesDockStore, type NotesDockNote } from "@/store/notes-dock"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { classNames } from "@/libs/class-name"
import { useNavigate } from "react-router-dom"
import { getQueryClient } from "@/services/query-client"
import TaskChecklistPreview, {
  type TaskChecklistTogglePayload
} from "@/components/Notes/TaskChecklistPreview"
import TaskActivityNotice from "@/components/Notes/TaskActivityNotice"
import { toggleChecklistItemMarker } from "@/components/Notes/task-markdown"
import {
  listNoteTasks,
  listTaskActivity,
  markTaskActivityRead,
  setNoteTaskStatus,
  type NoteTask,
  type NoteTaskActivityEvent,
  type NoteTaskReconciliationSummary
} from "@/services/notes-tasks"

const { TextArea } = Input

type NoteListItem = {
  id: number
  title?: string
  content?: string
  keywords?: any[]
  metadata?: { keywords?: any[] }
  version?: number | null
}

type NotesSearchResponse = {
  notes?: NoteListItem[]
  results?: NoteListItem[]
  items?: NoteListItem[]
}

type CloseChoice = "save" | "discard" | "cancel"

type DragState = {
  offsetX: number
  offsetY: number
}

const MIN_WIDTH = 360
const MIN_HEIGHT = 360

const clamp = (value: number, min: number, max: number) =>
  Math.min(Math.max(value, min), max)

const clampPosition = (
  x: number,
  y: number,
  width: number,
  height: number
) => {
  if (typeof window === "undefined") return { x, y }
  const maxX = Math.max(0, window.innerWidth - width)
  const maxY = Math.max(0, window.innerHeight - height)
  return {
    x: clamp(x, 0, maxX),
    y: clamp(y, 0, maxY)
  }
}

const readObservedBorderBoxSize = (entry: ResizeObserverEntry) => {
  const borderBoxSize = Array.isArray(entry.borderBoxSize)
    ? entry.borderBoxSize[0]
    : entry.borderBoxSize

  if (
    borderBoxSize &&
    Number.isFinite(borderBoxSize.inlineSize) &&
    Number.isFinite(borderBoxSize.blockSize)
  ) {
    return {
      width: borderBoxSize.inlineSize,
      height: borderBoxSize.blockSize
    }
  }

  const rect = entry.target.getBoundingClientRect()
  if (
    Number.isFinite(rect.width) &&
    rect.width > 0 &&
    Number.isFinite(rect.height) &&
    rect.height > 0
  ) {
    return {
      width: rect.width,
      height: rect.height
    }
  }

  return {
    width: entry.contentRect.width,
    height: entry.contentRect.height
  }
}

const parseKeywords = (input: string): string[] =>
  input
    .split(",")
    .map((keyword) => keyword.trim())
    .filter((keyword) => keyword.length > 0)

const extractKeywords = (note: NoteListItem | any): string[] => {
  const rawKeywords = (Array.isArray(note?.metadata?.keywords)
    ? note.metadata.keywords
    : Array.isArray(note?.keywords)
      ? note.keywords
      : []) as any[]

  return rawKeywords
    .map((item: any) => {
      if (typeof item === "string") return item
      const raw =
        item?.keyword ??
        item?.keyword_text ??
        item?.text ??
        item?.name ??
        item
      return typeof raw === "string" ? raw : null
    })
    .filter((keyword): keyword is string => Boolean(keyword && keyword.trim()))
}

const pickNotesArray = (response: NotesSearchResponse | NoteListItem[] | any): NoteListItem[] => {
  if (!response) return []
  if (Array.isArray(response)) return response
  if (Array.isArray(response.notes)) return response.notes
  if (Array.isArray(response.results)) return response.results
  if (Array.isArray(response.items)) return response.items
  return []
}

const ensurePortalRoot = () => {
  if (typeof document === "undefined") return null
  return document.getElementById("tldw-portal-root") || document.body
}

export const NotesDockPanel: React.FC = () => {
  const { t } = useTranslation(["option", "common", "playground"])
  const navigate = useNavigate()
  const message = useAntdMessage()
  const isOnline = useServerOnline()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const editorDisabled = !isOnline || (!capsLoading && !capabilities?.hasNotes)

  const {
    isOpen,
    position,
    size,
    notes,
    activeNoteId,
    setOpen,
    setPosition,
    setSize,
    createDraft,
    openNote,
    setActiveNote,
    updateNote,
    markSaved,
    discardNoteChanges,
    removeNote,
    discardAll
  } = useNotesDockStore()

  const [searchQuery, setSearchQuery] = useState("")
  const [notesList, setNotesList] = useState<NoteListItem[]>([])
  const [loadingList, setLoadingList] = useState(false)
  const [unsavedModalOpen, setUnsavedModalOpen] = useState(false)
  const [pendingCloseNoteId, setPendingCloseNoteId] = useState<string | null>(null)
  const [savingNoteId, setSavingNoteId] = useState<string | null>(null)
  const [syncingNotesList, setSyncingNotesList] = useState(false)
  const [keywordsInput, setKeywordsInput] = useState<Record<string, string>>({})
  const [archiveCollapsed, setArchiveCollapsed] = useState(false)
  const [activeNoteTasks, setActiveNoteTasks] = useState<NoteTask[]>([])
  const [taskReconciliation, setTaskReconciliation] =
    useState<NoteTaskReconciliationSummary | null>(null)
  const [taskActivityEvents, setTaskActivityEvents] = useState<NoteTaskActivityEvent[]>([])
  const [dismissingTaskActivityId, setDismissingTaskActivityId] = useState<string | null>(null)

  const dockRef = useRef<HTMLDivElement | null>(null)
  const dragStateRef = useRef<DragState | null>(null)
  const searchTimeoutRef = useRef<number | null>(null)
  const fetchRequestIdRef = useRef(0)
  const taskRequestIdRef = useRef(0)
  const activeServerNoteIdRef = useRef<string | null>(null)
  const cacheSyncInFlightRef = useRef(0)
  const unsavedModalReturnFocusRef = useRef<HTMLElement | null>(null)

  const hasDirtyNotes = useMemo(
    () => notes.some((note) => note.isDirty),
    [notes]
  )

  const activeNote = useMemo(
    () => notes.find((note) => note.localId === activeNoteId) ?? null,
    [notes, activeNoteId]
  )

  useEffect(() => {
    activeServerNoteIdRef.current = activeNote?.id != null ? String(activeNote.id) : null
  }, [activeNote?.id])

  const openNoteIds = useMemo(() => {
    return new Set(notes.map((note) => note.id).filter(Boolean))
  }, [notes])

  const ensureActiveNote = useCallback(() => {
    if (activeNoteId || notes.length === 0) return
    setActiveNote(notes[notes.length - 1]?.localId ?? null)
  }, [activeNoteId, notes, setActiveNote])

  useEffect(() => {
    ensureActiveNote()
  }, [ensureActiveNote])

  useEffect(() => {
    setKeywordsInput((prev) => {
      let changed = false
      const next = { ...prev }
      notes.forEach((note) => {
        if (note.isDirty) return
        const text = note.keywords.join(", ")
        if (next[note.localId] !== text) {
          next[note.localId] = text
          changed = true
        }
      })
      return changed ? next : prev
    })
  }, [notes])

  const handleRequestClose = useCallback(() => {
    if (hasDirtyNotes) {
      unsavedModalReturnFocusRef.current =
        document.activeElement instanceof HTMLElement ? document.activeElement : null
      setUnsavedModalOpen(true)
      return
    }
    setOpen(false)
  }, [hasDirtyNotes, setOpen])

  useEffect(() => {
    const handler = () => handleRequestClose()
    window.addEventListener("tldw:notes-dock-request-close", handler)
    return () => {
      window.removeEventListener("tldw:notes-dock-request-close", handler)
    }
  }, [handleRequestClose])

  const restoreFocusAfterUnsavedModalClose = useCallback(() => {
    const target = unsavedModalReturnFocusRef.current
    if (!target) return
    window.requestAnimationFrame(() => {
      if (target.isConnected) target.focus()
    })
  }, [])

  const handleSaveChoice = async () => {
    const ok = await saveAllDirtyNotes()
    if (ok) {
      setUnsavedModalOpen(false)
      setOpen(false)
    }
  }

  const handleDiscardChoice = () => {
    discardAll()
    setUnsavedModalOpen(false)
    setOpen(false)
  }

  const handleCancelChoice = () => {
    setUnsavedModalOpen(false)
    restoreFocusAfterUnsavedModalClose()
  }

  const fetchNotesList = useCallback(
    async (query: string) => {
      const requestId = ++fetchRequestIdRef.current
      if (editorDisabled) {
        setLoadingList(false)
        setNotesList([])
        return
      }
      setLoadingList(true)
      try {
        let response: NotesSearchResponse
        if (query.trim()) {
          response = await bgRequest<NotesSearchResponse>({
            path: `/api/v1/notes/search/?query=${encodeURIComponent(
              query.trim()
            )}&limit=20&include_keywords=true` as AllowedPath,
            method: "GET"
          })
        } else {
          response = await bgRequest<NotesSearchResponse>({
            path: "/api/v1/notes/?page=1&results_per_page=20&include_keywords=true" as AllowedPath,
            method: "GET"
          })
        }
        if (requestId !== fetchRequestIdRef.current) return
        const normalized = pickNotesArray(response).map((note) => ({
          ...note,
          keywords: extractKeywords(note)
        }))
        setNotesList(normalized)
      } catch (error) {
        if (requestId !== fetchRequestIdRef.current) return
        setNotesList([])
        message.error(t("option:notesDock.loadError", "Failed to load notes"))
      } finally {
        if (requestId === fetchRequestIdRef.current) {
          setLoadingList(false)
        }
      }
    },
    [editorDisabled, message, t]
  )

  useEffect(() => {
    fetchNotesList("")
  }, [fetchNotesList])

  useEffect(() => {
    if (searchTimeoutRef.current) {
      window.clearTimeout(searchTimeoutRef.current)
    }
    searchTimeoutRef.current = window.setTimeout(() => {
      fetchNotesList(searchQuery)
    }, 250)
    return () => {
      if (searchTimeoutRef.current) {
        window.clearTimeout(searchTimeoutRef.current)
      }
    }
  }, [fetchNotesList, searchQuery])

  const handleOpenNote = async (noteId: number) => {
    if (editorDisabled) return
    try {
      const detail = await bgRequest<NoteListItem>({
        path: `/api/v1/notes/${noteId}` as AllowedPath,
        method: "GET"
      })
      const keywords = extractKeywords(detail)
      openNote({
        id: detail.id,
        title: detail.title || "",
        content: detail.content || "",
        keywords,
        version: detail.version ?? null
      })
    } catch (error) {
      message.error(t("option:notesDock.loadDetailError", "Failed to open note"))
    }
  }

  const refreshTaskStateForNote = useCallback(
    async (noteId: string | number | null | undefined) => {
      const noteIdText = noteId != null ? String(noteId) : null
      if (noteIdText && activeServerNoteIdRef.current !== noteIdText) {
        return
      }
      const requestId = ++taskRequestIdRef.current
      if (!noteIdText || editorDisabled) {
        setActiveNoteTasks([])
        setTaskReconciliation(null)
        setTaskActivityEvents([])
        return
      }

      try {
        const response = await listNoteTasks(noteId, { limit: 500 })
        if (requestId !== taskRequestIdRef.current || activeServerNoteIdRef.current !== noteIdText) return
        setActiveNoteTasks(Array.isArray(response.tasks) ? response.tasks : [])
        setTaskReconciliation(response.reconciliation ?? null)
      } catch {
        if (requestId !== taskRequestIdRef.current || activeServerNoteIdRef.current !== noteIdText) return
        setActiveNoteTasks([])
        setTaskReconciliation(null)
      }

      try {
        const response = await listTaskActivity({ note_id: noteId, limit: 50 })
        if (requestId !== taskRequestIdRef.current || activeServerNoteIdRef.current !== noteIdText) return
        const events = Array.isArray(response.events)
          ? response.events.filter((event) => {
              return (
                String(event.note_id || "") === noteIdText &&
                !event.dismissed_at &&
                event.actor_type === "agent"
              )
            })
          : []
        setTaskActivityEvents(events)
      } catch {
        if (requestId !== taskRequestIdRef.current || activeServerNoteIdRef.current !== noteIdText) return
        setTaskActivityEvents([])
      }
    },
    [editorDisabled]
  )

  useEffect(() => {
    void refreshTaskStateForNote(activeNote?.id)
  }, [activeNote?.id, refreshTaskStateForNote])

  const syncNotesPageCache = useCallback(async () => {
    cacheSyncInFlightRef.current += 1
    setSyncingNotesList(true)
    try {
      await getQueryClient().invalidateQueries({ queryKey: ["notes"] })
    } catch {
      message.warning(
        t(
          "option:notesDock.syncRefreshWarning",
          "Saved note, but notes page refresh may be delayed."
        )
      )
    } finally {
      cacheSyncInFlightRef.current = Math.max(0, cacheSyncInFlightRef.current - 1)
      if (cacheSyncInFlightRef.current === 0) {
        setSyncingNotesList(false)
      }
    }
  }, [message, t])

  const saveNote = async (note: NotesDockNote) => {
    if (editorDisabled) return false
    if (!note.title.trim() && !note.content.trim()) {
      message.warning(
        t(
          "option:notesDock.emptyNote",
          "Add a title or content before saving."
        )
      )
      return false
    }

    setSavingNoteId(note.localId)
    try {
      const payload: Record<string, unknown> = {
        title: note.title.trim() || "Untitled Note",
        content: note.content,
        keywords: note.keywords ?? []
      }

      let saved: NoteListItem
      if (note.id) {
        const path = note.version
          ? (`/api/v1/notes/${note.id}?expected_version=${encodeURIComponent(
              note.version
            )}` as AllowedPath)
          : (`/api/v1/notes/${note.id}` as AllowedPath)

        saved = await bgRequest<NoteListItem>({
          path,
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: payload
        })
      } else {
        saved = await bgRequest<NoteListItem>({
          path: "/api/v1/notes/" as AllowedPath,
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload
        })
      }

      const extractedKeywords = extractKeywords(saved)
      markSaved(note.localId, {
        id: saved.id ?? note.id,
        title: saved.title ?? note.title,
        content: saved.content ?? note.content,
        keywords: extractedKeywords.length > 0 ? extractedKeywords : note.keywords,
          version: saved.version ?? note.version ?? null
      })

      const savedId = saved.id ?? note.id
      if (savedId != null) {
        await refreshTaskStateForNote(savedId)
      }
      void syncNotesPageCache()
      message.success(t("option:notesDock.saved", "Note saved"))
      return true
    } catch (error: any) {
      if (error?.status === 409 || error?.message?.includes("version")) {
        message.error(
          t(
            "option:notesDock.versionConflict",
            "Note was modified elsewhere. Reload and try again."
          )
        )
      } else {
        message.error(t("option:notesDock.saveError", "Failed to save note"))
      }
      return false
    } finally {
      setSavingNoteId(null)
    }
  }

  const toggleTaskCheckboxLocal = useCallback(
    (payload: TaskChecklistTogglePayload) => {
      if (!activeNote) return
      const nextContent = toggleChecklistItemMarker(
        activeNote.content,
        payload.lineNumber,
        payload.nextStatus === "done"
      )
      if (nextContent === activeNote.content) return
      updateNote(activeNote.localId, { content: nextContent })
    },
    [activeNote, updateNote]
  )

  const toggleTaskCheckboxStatus = useCallback(
    async (payload: TaskChecklistTogglePayload) => {
      if (!activeNote) return
      const task = payload.task
      if (activeNote.isDirty || !task || !activeNote.id) {
        toggleTaskCheckboxLocal(payload)
        return
      }

      const expectedNoteVersion = activeNote.version ?? task.projection?.note_version ?? null
      if (expectedNoteVersion == null) {
        message.error(
          t(
            "option:notesSearch_taskConflictNotice",
            "Task changed on the server. Reload the note and try again."
          )
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
        const detail = await bgRequest<NoteListItem>({
          path: `/api/v1/notes/${activeNote.id}` as AllowedPath,
          method: "GET"
        })
        const keywords = extractKeywords(detail)
        markSaved(activeNote.localId, {
          id: detail.id ?? activeNote.id,
          title: detail.title ?? activeNote.title,
          content: detail.content ?? activeNote.content,
          keywords,
          version: detail.version ?? activeNote.version ?? null
        })
        await refreshTaskStateForNote(activeNote.id)
        void syncNotesPageCache()
      } catch (error: any) {
        if (error?.status === 409 || error?.message?.includes("version")) {
          message.error(
            t(
              "option:notesSearch_taskConflictNotice",
              "Task changed on the server. Reload the note and try again."
            )
          )
        } else {
          message.error(t("option:notesSearch_taskUpdateError", "Task update failed"))
        }
      }
    },
    [
      activeNote,
      markSaved,
      message,
      refreshTaskStateForNote,
      syncNotesPageCache,
      t,
      toggleTaskCheckboxLocal
    ]
  )

  const dismissTaskActivity = useCallback(
    async (eventId: string) => {
      setDismissingTaskActivityId(eventId)
      try {
        await markTaskActivityRead(eventId, { dismissed: true })
        setTaskActivityEvents((current) => current.filter((event) => event.id !== eventId))
      } catch {
        message.error(t("option:notesSearch_taskActivityDismissError", "Failed to dismiss task activity"))
      } finally {
        setDismissingTaskActivityId(null)
      }
    },
    [message, t]
  )

  const saveAllDirtyNotes = async () => {
    const dirtyNotes = notes.filter((note) => note.isDirty)
    for (const note of dirtyNotes) {
      const ok = await saveNote(note)
      if (!ok) return false
    }
    return true
  }

  const handleNewNote = () => {
    createDraft()
  }

  const handleKeywordsChange = (localId: string, value: string) => {
    setKeywordsInput((prev) => ({ ...prev, [localId]: value }))
    updateNote(localId, { keywords: parseKeywords(value) })
  }

  const handleCloseNoteTab = async (localId: string) => {
    const note = notes.find((item) => item.localId === localId)
    if (!note) return

    if (!note.isDirty) {
      removeNote(localId)
      return
    }

    setPendingCloseNoteId(localId)
  }

  const handleConfirmClose = async (choice: CloseChoice) => {
    if (!pendingCloseNoteId) return
    const note = notes.find((item) => item.localId === pendingCloseNoteId)
    if (!note) {
      setPendingCloseNoteId(null)
      return
    }

    if (choice === "save") {
      const ok = await saveNote(note)
      if (ok) {
        removeNote(note.localId)
      }
    }

    if (choice === "discard") {
      discardNoteChanges(note.localId)
      removeNote(note.localId)
    }

    setPendingCloseNoteId(null)
  }

  const startDrag = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button !== 0) return
    const target = event.target as HTMLElement
    if (target.closest("[data-no-drag]") != null) return
    const rect = dockRef.current?.getBoundingClientRect()
    if (!rect) return

    dragStateRef.current = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top
    }
    document.body.style.userSelect = "none"

    const handleMove = (moveEvent: MouseEvent) => {
      if (!dragStateRef.current) return
      const nextX = moveEvent.clientX - dragStateRef.current.offsetX
      const nextY = moveEvent.clientY - dragStateRef.current.offsetY
      const clamped = clampPosition(nextX, nextY, size.width, size.height)
      setPosition(clamped)
    }

    const handleUp = () => {
      dragStateRef.current = null
      document.body.style.userSelect = ""
      window.removeEventListener("mousemove", handleMove)
      window.removeEventListener("mouseup", handleUp)
    }

    window.addEventListener("mousemove", handleMove)
    window.addEventListener("mouseup", handleUp)
  }

  useEffect(() => {
    const handleResize = () => {
      const clamped = clampPosition(position.x, position.y, size.width, size.height)
      if (clamped.x !== position.x || clamped.y !== position.y) {
        setPosition(clamped)
      }
    }
    window.addEventListener("resize", handleResize)
    handleResize()
    return () => window.removeEventListener("resize", handleResize)
  }, [position.x, position.y, size.width, size.height, setPosition])

  useEffect(() => {
    const dockElement = dockRef.current
    if (!dockElement) return

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      const observedSize = readObservedBorderBoxSize(entry)
      const nextWidth = Math.max(MIN_WIDTH, Math.round(observedSize.width))
      const nextHeight = Math.max(MIN_HEIGHT, Math.round(observedSize.height))
      if (nextWidth !== size.width || nextHeight !== size.height) {
        setSize({ width: nextWidth, height: nextHeight })
        const clamped = clampPosition(position.x, position.y, nextWidth, nextHeight)
        if (clamped.x !== position.x || clamped.y !== position.y) {
          setPosition(clamped)
        }
      }
    })
    try {
      observer.observe(dockElement, { box: "border-box" })
    } catch {
      observer.observe(dockElement)
    }
    return () => observer.disconnect()
  }, [position.x, position.y, setPosition, setSize, size.height, size.width])

  if (!isOpen) return null

  const portalRoot = ensurePortalRoot()
  if (!portalRoot) return null
  const hasIncompleteTaskReconciliation = taskReconciliation?.status === "incomplete"

  const panel = (
    <div
      ref={dockRef}
      role="dialog"
      aria-label={t("option:notesDock.title", "Notes Dock")}
      className={classNames(
        "fixed z-[70] flex flex-col rounded-xl border border-border bg-surface shadow-xl",
        "overflow-hidden"
      )}
      style={{
        width: size.width,
        height: size.height,
        minWidth: MIN_WIDTH,
        minHeight: MIN_HEIGHT,
        top: position.y,
        left: position.x,
        resize: "both"
      }}
    >
      <div
        className={classNames(
          "flex items-center justify-between border-b border-border",
          "px-3 py-2 cursor-move select-none"
        )}
        onMouseDown={startDrag}
      >
        <div className="flex items-center gap-2 text-sm font-semibold text-text">
          <StickyNote className="size-4" />
          {t("option:notesDock.title", "Notes Dock")}
        </div>
        <div className="flex items-center gap-1" data-no-drag>
          <Tooltip title={t("option:notesDock.new", "New note")}>
            <Button
              type="text"
              size="small"
              icon={<Plus className="h-3.5 w-3.5" />}
              onClick={handleNewNote}
              disabled={editorDisabled}
              aria-label={t("option:notesDock.new", "New note")}
            />
          </Tooltip>
          <Tooltip title={t("option:notesDock.close", "Close") }>
            <Button
              type="text"
              size="small"
              icon={<X className="h-3.5 w-3.5" />}
              onClick={handleRequestClose}
              aria-label={t("option:notesDock.close", "Close")}
              data-testid="notes-dock-close-button"
            />
          </Tooltip>
        </div>
      </div>

      <div className="flex min-h-0 flex-1">
        {/* Archive column */}
        <div
          className={classNames(
            "flex shrink-0 flex-col border-r border-border bg-surface2",
            archiveCollapsed ? "w-12" : "w-64"
          )}
        >
          <div className="flex items-center justify-between px-3 py-2">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase text-text-subtle">
              <FolderOpen className="h-3.5 w-3.5" />
              {!archiveCollapsed &&
                t("option:notesDock.archive", "Archive")}
            </div>
            {!archiveCollapsed && (
              <Tooltip title={t("common:refresh", "Refresh")}>
                <Button
                  type="text"
                  size="small"
                  icon={<RefreshCcw className="h-3.5 w-3.5" />}
                  onClick={() => fetchNotesList(searchQuery)}
                  disabled={editorDisabled || loadingList}
                />
              </Tooltip>
            )}
            <Tooltip
              title={
                archiveCollapsed
                  ? t("option:notesDock.expandArchive", "Expand archive")
                  : t("option:notesDock.collapseArchive", "Collapse archive")
              }
            >
              <Button
                type="text"
                size="small"
                icon={
                  archiveCollapsed ? (
                    <ChevronRight className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronLeft className="h-3.5 w-3.5" />
                  )
                }
                onClick={() => setArchiveCollapsed((prev) => !prev)}
              />
            </Tooltip>
          </div>
          {!archiveCollapsed && (
            <>
              <div className="px-3 pb-2">
                <Input
                  size="small"
                  allowClear
                  placeholder={t("option:notesDock.search", "Search notes...")}
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  disabled={editorDisabled}
                />
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
                {editorDisabled ? (
                  <div className="px-2 py-3 text-xs text-text-subtle">
                    {t(
                      "option:notesDock.disabled",
                      "Connect to a server that supports notes to browse your archive."
                    )}
                  </div>
                ) : loadingList ? (
                  <div className="flex items-center justify-center py-6">
                    <Spin size="small" />
                  </div>
                ) : notesList.length === 0 ? (
                  <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description={
                      <span className="text-text-subtle text-xs">
                        {searchQuery
                          ? t("option:notesDock.noResults", "No notes found")
                          : t("option:notesDock.noNotes", "No notes yet")}
                      </span>
                    }
                  />
                ) : (
                  <div className="space-y-2">
                    {notesList.map((note) => {
                      const isOpenNote = note.id != null && openNoteIds.has(note.id)
                      return (
                        <button
                          key={note.id}
                          type="button"
                          onClick={() => handleOpenNote(note.id)}
                          className={classNames(
                            "flex w-full items-start gap-2 rounded-lg border border-border bg-surface px-2 py-2 text-left",
                            "transition hover:border-primary/50 hover:bg-primary/5",
                            isOpenNote ? "border-primary/60" : ""
                          )}
                        >
                          <StickyNote className="mt-0.5 h-3.5 w-3.5 text-text-muted" />
                          <div className="min-w-0 flex-1">
                            <p className="truncate text-xs font-medium text-text">
                              {note.title || t("option:notesDock.untitled", "Untitled")}
                            </p>
                            <p className="line-clamp-2 text-[11px] text-text-subtle">
                              {note.content?.slice(0, 80) ||
                                t("option:notesDock.empty", "No content")}
                            </p>
                            {note.keywords && note.keywords.length > 0 && (
                              <div className="mt-1 flex flex-wrap gap-1">
                                {note.keywords.slice(0, 3).map((keyword, idx) => (
                                  <Tag key={`${note.id}-${idx}`} className="text-[10px]">
                                    {keyword}
                                  </Tag>
                                ))}
                                {note.keywords.length > 3 && (
                                  <span className="text-[10px] text-text-muted">
                                    +{note.keywords.length - 3}
                                  </span>
                                )}
                              </div>
                            )}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>
              <div className="border-t border-border px-3 py-2">
                <button
                  type="button"
                  onClick={() => navigate("/notes")}
                  className="text-xs text-primary hover:text-primaryStrong"
                >
                  {t("option:notesDock.openNotes", "Open Notes page")}
                </button>
              </div>
            </>
          )}
        </div>

        {/* Editor column */}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="border-b border-border px-3 py-2">
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex-1 min-w-[200px]">
                <Input
                  size="small"
                  placeholder={t("option:notesDock.titlePlaceholder", "Note title...")}
                  value={activeNote?.title ?? ""}
                  onChange={(event) =>
                    activeNote &&
                    updateNote(activeNote.localId, { title: event.target.value })
                  }
                  disabled={!activeNote || editorDisabled}
                />
              </div>
              <Button
                size="small"
                type="primary"
                icon={<Save className="h-3.5 w-3.5" />}
                disabled={!activeNote || editorDisabled}
                loading={savingNoteId === activeNote?.localId}
                onClick={() => activeNote && saveNote(activeNote)}
              >
                {activeNote?.id
                  ? t("option:notesDock.update", "Update")
                  : t("option:notesDock.save", "Save")}
              </Button>
            </div>
            <div className="mt-2">
              <Input
                size="small"
                placeholder={t(
                  "option:notesDock.tagsPlaceholder",
                  "Keywords (comma-separated)"
                )}
                value={
                  activeNote
                    ? keywordsInput[activeNote.localId] ??
                      activeNote.keywords.join(", ")
                    : ""
                }
                onChange={(event) =>
                  activeNote &&
                  handleKeywordsChange(activeNote.localId, event.target.value)
                }
                disabled={!activeNote || editorDisabled}
              />
            </div>
            {activeNote && activeNote.keywords.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {activeNote.keywords.map((keyword, idx) => (
                  <Tag
                    key={`${activeNote.localId}-kw-${idx}`}
                    closable
                    onClose={(event) => {
                      event.preventDefault()
                      const nextKeywords = activeNote.keywords.filter(
                        (_, index) => index !== idx
                      )
                      updateNote(activeNote.localId, { keywords: nextKeywords })
                      setKeywordsInput((prev) => ({
                        ...prev,
                        [activeNote.localId]: nextKeywords.join(", ")
                      }))
                    }}
                    className="text-[11px]"
                  >
                    {keyword}
                  </Tag>
                ))}
              </div>
            )}
            {activeNote?.isDirty && (
              <div className="mt-2 text-[11px] text-warning">
                {t("option:notesDock.unsaved", "Unsaved changes")}
              </div>
            )}
            <TaskActivityNotice
              events={taskActivityEvents}
              noteTitle={activeNote?.title || null}
              testId="notes-dock-task-activity-notice"
              compact
              dismissingEventId={dismissingTaskActivityId}
              onInspect={() => navigate("/notes")}
              onDismiss={(eventId) => void dismissTaskActivity(eventId)}
            />
            {hasIncompleteTaskReconciliation && (
              <div className="mt-2 text-[11px] text-warning" data-testid="notes-dock-task-reconciliation-warning">
                {t(
                  "option:notesSearch_taskIncompleteReconciliationWarning",
                  "Some task updates are still reconciling."
                )}
              </div>
            )}
            {syncingNotesList && (
              <div className="mt-2 text-[11px] text-text-subtle" data-testid="notes-dock-sync-indicator">
                {t("option:notesDock.syncingNotesList", "Syncing notes list...")}
              </div>
            )}
          </div>

          <div className="min-h-0 flex-1 p-3">
            {activeNote ? (
              <div className="flex h-full min-h-0 flex-col gap-2">
                <TaskChecklistPreview
                  content={activeNote.content}
                  tasks={activeNoteTasks}
                  isDirty={activeNote.isDirty}
                  disabled={editorDisabled}
                  compact
                  onToggleLocal={toggleTaskCheckboxLocal}
                  onToggleTaskStatus={toggleTaskCheckboxStatus}
                />
                <TextArea
                  value={activeNote.content}
                  onChange={(event) =>
                    updateNote(activeNote.localId, { content: event.target.value })
                  }
                  placeholder={t(
                    "option:notesDock.contentPlaceholder",
                    "Jot down notes, ideas, or observations..."
                  )}
                  className="min-h-0 flex-1 resize-none text-sm"
                  style={{ height: "100%" }}
                  disabled={editorDisabled}
                />
              </div>
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-text-subtle">
                {t(
                  "option:notesDock.emptyState",
                  "Select a note from the archive or create a new one."
                )}
              </div>
            )}
          </div>

          {notes.length > 0 && (
            <div className="border-t border-border px-3 py-2">
              <div className="flex flex-wrap gap-2">
                {notes.map((note) => (
                  <div
                    key={note.localId}
                    className={classNames(
                      "flex items-center gap-2 rounded-full border px-3 py-1 text-xs",
                      note.localId === activeNoteId
                        ? "border-primary text-primary"
                        : "border-border text-text-muted"
                    )}
                  >
                    <button
                      type="button"
                      onClick={() => setActiveNote(note.localId)}
                      aria-pressed={note.localId === activeNoteId}
                      className={classNames(
                        "flex min-w-0 items-center gap-2",
                        note.localId === activeNoteId
                          ? "text-primary"
                          : "text-text-muted hover:text-text"
                      )}
                    >
                      <span className="max-w-[120px] truncate">
                        {note.title || t("option:notesDock.untitled", "Untitled")}
                      </span>
                      {note.isDirty && (
                        <span className="h-1.5 w-1.5 rounded-full bg-warning" />
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleCloseNoteTab(note.localId)}
                      className={classNames(
                        "rounded-full text-text-muted hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50",
                        note.localId === activeNoteId && "text-primary hover:text-primary"
                      )}
                      aria-label={t("option:notesDock.closeTab", "Close note")}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <Modal
        open={unsavedModalOpen}
        onCancel={handleCancelChoice}
        title={t("option:notesDock.unsavedTitle", "Unsaved notes")}
        keyboard
        destroyOnHidden
        footer={
          <div className="flex items-center justify-end gap-2">
            <Button
              onClick={handleCancelChoice}
              data-testid="notes-dock-unsaved-cancel-button"
            >
              {t("common:cancel", "Cancel")}
            </Button>
            <Button danger onClick={handleDiscardChoice}>
              {t("option:notesDock.discard", "Discard")}
            </Button>
            <Button type="primary" onClick={handleSaveChoice}>
              {t("option:notesDock.saveAndClose", "Save & Close")}
            </Button>
          </div>
        }
      >
        <p className="text-sm text-text-subtle" data-testid="notes-dock-unsaved-modal-body">
          {t(
            "option:notesDock.unsavedBody",
            "You have unsaved notes. Save them before closing the dock?"
          )}
        </p>
      </Modal>

      <Modal
        open={pendingCloseNoteId != null}
        onCancel={() => handleConfirmClose("cancel")}
        title={t("option:notesDock.closeNoteTitle", "Unsaved note")}
        footer={
          <div className="flex items-center justify-end gap-2">
            <Button onClick={() => handleConfirmClose("cancel")}>
              {t("common:cancel", "Cancel")}
            </Button>
            <Button danger onClick={() => handleConfirmClose("discard")}>
              {t("option:notesDock.discard", "Discard")}
            </Button>
            <Button type="primary" onClick={() => handleConfirmClose("save")}>
              {t("option:notesDock.save", "Save")}
            </Button>
          </div>
        }
      >
        <p className="text-sm text-text-subtle">
          {t(
            "option:notesDock.closeNoteBody",
            "Save changes to this note before closing it?"
          )}
        </p>
      </Modal>
    </div>
  )

  return createPortal(panel, portalRoot)
}

export default NotesDockPanel
