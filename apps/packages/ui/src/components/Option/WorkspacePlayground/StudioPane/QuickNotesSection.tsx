import React, { useState, useCallback, useRef, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { Input, Button, Modal, AutoComplete, message, Tag, Empty, Spin } from "antd"
import type { InputRef } from "antd"
import type { TextAreaRef } from "antd/es/input/TextArea"
import {
  Save,
  FolderOpen,
  X,
  Search,
  FileText,
  AlertCircle,
  Check,
  ChevronUp,
  Eye,
  PencilLine,
  Download
} from "lucide-react"
import { useWorkspaceStore } from "@/store/workspace"
import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import { MarkdownPreview } from "@/components/Common/MarkdownPreview"
import { getNoteKeywords } from "@/services/note-keywords"
import {
  WORKSPACE_UNDO_WINDOW_MS,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction
} from "../undo-manager"

const { TextArea } = Input

type NoteKeyword =
  | string
  | {
      keyword?: string
      keyword_text?: string
      text?: string
      name?: string
    }

interface NoteListItem {
  id: number
  title?: string
  content?: string
  keywords?: NoteKeyword[]
  metadata?: {
    keywords?: NoteKeyword[]
  }
  version?: number
  created_at?: string
  last_modified?: string
  workspace_tag?: string
}

interface NotesSearchResponse {
  notes?: NoteListItem[]
  results?: NoteListItem[]
  items?: NoteListItem[]
  total?: number
}

const DEFAULT_NOTES_LIMIT = 20
const WORKSPACE_NOTES_LIMIT = 8

const parseKeywordValue = (keyword: NoteKeyword | null | undefined): string | null => {
  if (!keyword) return null
  if (typeof keyword === "string") return keyword.trim() || null
  const value =
    keyword.keyword ??
    keyword.keyword_text ??
    keyword.text ??
    keyword.name ??
    null
  return typeof value === "string" && value.trim().length > 0 ? value.trim() : null
}

const normalizeKeywords = (keywords: string[]): string[] => {
  const seen = new Set<string>()
  const normalized: string[] = []

  for (const keyword of keywords) {
    const cleaned = keyword.trim()
    if (!cleaned) continue
    const dedupeKey = cleaned.toLowerCase()
    if (seen.has(dedupeKey)) continue
    seen.add(dedupeKey)
    normalized.push(cleaned)
  }

  return normalized
}

export const extractNoteKeywords = (note?: NoteListItem | null): string[] => {
  if (!note) return []
  const raw = Array.isArray(note.metadata?.keywords)
    ? note.metadata?.keywords
    : Array.isArray(note.keywords)
      ? note.keywords
      : []
  return normalizeKeywords(
    raw
      .map((keyword) => parseKeywordValue(keyword))
      .filter((value): value is string => Boolean(value))
  )
}

const isKeywordMatch = (keyword: string, target: string): boolean =>
  keyword.trim().toLowerCase() === target.trim().toLowerCase()

const stripWorkspaceTagFromKeywords = (
  keywords: string[],
  workspaceTag: string
): string[] =>
  keywords.filter((keyword) => !isKeywordMatch(keyword, workspaceTag))

const buildPersistedKeywords = (
  keywords: string[],
  workspaceTag: string
): string[] => {
  const normalized = normalizeKeywords(keywords)
  if (!workspaceTag.trim()) return normalized
  return normalizeKeywords([...normalized, workspaceTag.trim()])
}

export const isWorkspaceTaggedNote = (
  note: NoteListItem,
  workspaceTag: string
): boolean => {
  if (!workspaceTag.trim()) return false
  if (
    typeof note.workspace_tag === "string" &&
    isKeywordMatch(note.workspace_tag, workspaceTag)
  ) {
    return true
  }
  return extractNoteKeywords(note).some((keyword) =>
    isKeywordMatch(keyword, workspaceTag)
  )
}

const pickNotesArray = (
  response: NotesSearchResponse | NoteListItem[] | null | undefined
): NoteListItem[] => {
  if (!response) return []
  if (Array.isArray(response)) return response
  if (Array.isArray(response.notes)) return response.notes
  if (Array.isArray(response.results)) return response.results
  if (Array.isArray(response.items)) return response.items
  return []
}

const parseNoteTimestamp = (note: NoteListItem): number => {
  const candidate = note.last_modified || note.created_at
  if (!candidate) return 0
  const value = new Date(candidate).getTime()
  return Number.isNaN(value) ? 0 : value
}

export const prioritizeWorkspaceNotes = (
  notes: NoteListItem[],
  workspaceTag: string
): NoteListItem[] => {
  return [...notes].sort((a, b) => {
    const aWorkspace = isWorkspaceTaggedNote(a, workspaceTag) ? 1 : 0
    const bWorkspace = isWorkspaceTaggedNote(b, workspaceTag) ? 1 : 0
    if (aWorkspace !== bWorkspace) return bWorkspace - aWorkspace
    return parseNoteTimestamp(b) - parseNoteTimestamp(a)
  })
}

const normalizeNotesForDisplay = (
  notes: NoteListItem[],
  workspaceTag: string
): NoteListItem[] =>
  notes.map((note) => {
    const normalizedKeywords = extractNoteKeywords(note)
    return {
      ...note,
      keywords: workspaceTag.trim()
        ? stripWorkspaceTagFromKeywords(normalizedKeywords, workspaceTag)
        : normalizedKeywords
    }
  })

const mergeUniqueNotes = (
  prioritized: NoteListItem[],
  fallback: NoteListItem[]
): NoteListItem[] => {
  const mergedMap = new Map<number, NoteListItem>()
  for (const note of [...prioritized, ...fallback]) {
    if (!mergedMap.has(note.id)) {
      mergedMap.set(note.id, note)
    }
  }
  return Array.from(mergedMap.values())
}

const buildSearchPath = ({
  query,
  workspaceToken,
  limit = DEFAULT_NOTES_LIMIT
}: {
  query?: string
  workspaceToken?: string
  limit?: number
}) => {
  const params = new URLSearchParams()
  if (query?.trim()) {
    params.set("query", query.trim())
  }
  if (workspaceToken?.trim()) {
    params.append("tokens", workspaceToken.trim())
  }
  params.set("limit", String(limit))
  params.set("include_keywords", "true")
  return `/api/v1/notes/search/?${params.toString()}` as AllowedPath
}

const buildListPath = (limit: number = DEFAULT_NOTES_LIMIT) =>
  `/api/v1/notes/?page=1&results_per_page=${limit}&include_keywords=true` as AllowedPath

const getCurrentKeywordFragment = (value: string): string => {
  const segments = value.split(",")
  return segments[segments.length - 1]?.trim() || ""
}

const sanitizeFilename = (value: string): string => {
  const normalized = value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  return normalized || "note"
}

const downloadTextFile = (
  content: string,
  filename: string,
  mimeType: string
): void => {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export const rankKeywordSuggestions = (
  query: string,
  keywords: string[]
): string[] => {
  const normalizedQuery = query.trim().toLowerCase()
  if (!normalizedQuery) return []

  const deduped = normalizeKeywords(keywords)
  const startsWithMatches = deduped.filter((keyword) =>
    keyword.toLowerCase().startsWith(normalizedQuery)
  )
  const containsMatches = deduped.filter((keyword) => {
    const lower = keyword.toLowerCase()
    return !lower.startsWith(normalizedQuery) && lower.includes(normalizedQuery)
  })

  return [...startsWithMatches, ...containsMatches]
}

interface QuickNotesSectionProps {
  /** Callback to collapse this section */
  onCollapse?: () => void
}

/**
 * QuickNotesSection - Enhanced notes editor with load/save functionality
 */
export const QuickNotesSection: React.FC<QuickNotesSectionProps> = ({ onCollapse }) => {
  const { t } = useTranslation(["playground", "common"])
  const [messageApi, contextHolder] = message.useMessage()

  // Store state
  const currentNote = useWorkspaceStore((s) => s.currentNote)
  const workspaceTag = useWorkspaceStore((s) => s.workspaceTag)
  const noteFocusTarget = useWorkspaceStore((s) => s.noteFocusTarget)

  // Store actions
  const updateNoteTitle = useWorkspaceStore((s) => s.updateNoteTitle)
  const updateNoteContent = useWorkspaceStore((s) => s.updateNoteContent)
  const updateNoteKeywords = useWorkspaceStore((s) => s.updateNoteKeywords)
  const setCurrentNote = useWorkspaceStore((s) => s.setCurrentNote)
  const clearCurrentNote = useWorkspaceStore((s) => s.clearCurrentNote)
  const loadNote = useWorkspaceStore((s) => s.loadNote)
  const clearNoteFocusTarget = useWorkspaceStore((s) => s.clearNoteFocusTarget)

  // Local state
  const [isSaving, setIsSaving] = useState(false)
  const [showSavedIndicator, setShowSavedIndicator] = useState(false)
  const savedIndicatorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [isLoadModalOpen, setIsLoadModalOpen] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [notesList, setNotesList] = useState<NoteListItem[]>([])
  const [workspaceNotes, setWorkspaceNotes] = useState<NoteListItem[]>([])
  const [isLoadingWorkspaceNotes, setIsLoadingWorkspaceNotes] = useState(false)
  const [editorMode, setEditorMode] = useState<"edit" | "preview">("edit")
  const [keywordsInput, setKeywordsInput] = useState("")
  const [keywordCatalog, setKeywordCatalog] = useState<string[]>([])
  const [keywordSuggestions, setKeywordSuggestions] = useState<string[]>([])
  const titleInputRef = useRef<InputRef | null>(null)
  const contentInputRef = useRef<TextAreaRef | null>(null)

  const clearSavedIndicatorTimer = useCallback(() => {
    if (savedIndicatorTimerRef.current) {
      clearTimeout(savedIndicatorTimerRef.current)
      savedIndicatorTimerRef.current = null
    }
  }, [])

  const hideSavedIndicator = useCallback(() => {
    clearSavedIndicatorTimer()
    setShowSavedIndicator(false)
  }, [clearSavedIndicatorTimer])

  const showSavedIndicatorTemporarily = useCallback(() => {
    hideSavedIndicator()
    setShowSavedIndicator(true)
    savedIndicatorTimerRef.current = setTimeout(() => {
      setShowSavedIndicator(false)
      savedIndicatorTimerRef.current = null
    }, 2000)
  }, [hideSavedIndicator])

  // Clean up saved indicator timer on unmount
  useEffect(() => {
    return () => {
      clearSavedIndicatorTimer()
    }
  }, [clearSavedIndicatorTimer])

  useEffect(() => {
    if (!currentNote.isDirty) return
    hideSavedIndicator()
  }, [currentNote.isDirty, hideSavedIndicator])

  // Parse keywords from input
  const parseKeywords = (input: string): string[] =>
    normalizeKeywords(input.split(","))

  const updateKeywordSuggestions = useCallback(
    (value: string) => {
      const fragment = getCurrentKeywordFragment(value)
      if (!fragment) {
        setKeywordSuggestions([])
        return
      }

      const ranked = rankKeywordSuggestions(fragment, keywordCatalog)
      const filtered = workspaceTag.trim()
        ? ranked.filter((keyword) => !isKeywordMatch(keyword, workspaceTag))
        : ranked
      setKeywordSuggestions(filtered.slice(0, 8))
    },
    [keywordCatalog, workspaceTag]
  )

  // Handle keywords input change
  const handleKeywordsChange = (value: string) => {
    hideSavedIndicator()
    setKeywordsInput(value)
    updateNoteKeywords(parseKeywords(value))
    updateKeywordSuggestions(value)
  }

  const handleKeywordSelect = (selectedKeyword: string) => {
    hideSavedIndicator()
    const segments = keywordsInput.split(",")
    if (segments.length === 0) {
      segments.push(selectedKeyword)
    } else {
      segments[segments.length - 1] = selectedKeyword
    }
    const joined = normalizeKeywords(segments).join(", ")
    const nextValue = joined ? `${joined}, ` : ""
    setKeywordsInput(nextValue)
    updateNoteKeywords(parseKeywords(nextValue))
    setKeywordSuggestions([])
  }

  // Track if keywords were just loaded from a note (to avoid sync loops)
  const lastLoadedNoteId = useRef<number | undefined>(undefined)

  // Sync keywords input when note is loaded or cleared
  useEffect(() => {
    // Only sync when the note ID changes (load or clear)
    if (currentNote.id !== lastLoadedNoteId.current) {
      lastLoadedNoteId.current = currentNote.id
      if (currentNote.keywords.length > 0) {
        setKeywordsInput(currentNote.keywords.join(", "))
      } else {
        setKeywordsInput("")
      }
      setKeywordSuggestions([])
      hideSavedIndicator()
    }
  }, [currentNote.id, currentNote.keywords, hideSavedIndicator])

  useEffect(() => {
    if (!noteFocusTarget) return

    const timer = window.setTimeout(() => {
      if (noteFocusTarget.field === "title") {
        titleInputRef.current?.focus()
        titleInputRef.current?.input?.select()
      } else {
        const textArea = contentInputRef.current?.resizableTextArea?.textArea
        textArea?.focus()
      }
    }, 0)

    clearNoteFocusTarget()

    return () => {
      window.clearTimeout(timer)
    }
  }, [clearNoteFocusTarget, noteFocusTarget])

  // Debounce timer ref for search
  const searchDebounceRef = useRef<NodeJS.Timeout | null>(null)

  const serializeNoteForEditor = useCallback(
    (note: NoteListItem) => ({
      id: note.id,
      title: note.title || "",
      content: note.content || "",
      keywords: workspaceTag
        ? stripWorkspaceTagFromKeywords(extractNoteKeywords(note), workspaceTag)
        : extractNoteKeywords(note),
      version: note.version
    }),
    [workspaceTag]
  )

  // Search/list notes
  const searchNotes = useCallback(async (query?: string) => {
    setIsSearching(true)
    try {
      const normalizedQuery = query?.trim() || ""
      const fallbackPath = normalizedQuery
        ? buildSearchPath({ query: normalizedQuery })
        : buildListPath(DEFAULT_NOTES_LIMIT)

      const fallbackResponse = await bgRequest<NotesSearchResponse | NoteListItem[]>({
        path: fallbackPath,
        method: "GET"
      })
      const fallbackNotes = pickNotesArray(fallbackResponse)

      let mergedNotes = fallbackNotes
      if (workspaceTag.trim()) {
        try {
          const workspaceResponse = await bgRequest<NotesSearchResponse | NoteListItem[]>({
            path: buildSearchPath({
              query: normalizedQuery || undefined,
              workspaceToken: workspaceTag,
              limit: DEFAULT_NOTES_LIMIT
            }),
            method: "GET"
          })
          const workspaceMatches = pickNotesArray(workspaceResponse)
          mergedNotes = mergeUniqueNotes(workspaceMatches, fallbackNotes)
        } catch {
          mergedNotes = fallbackNotes
        }
      }

      const prioritized = prioritizeWorkspaceNotes(mergedNotes, workspaceTag)
      setNotesList(normalizeNotesForDisplay(prioritized, workspaceTag))
    } catch (error) {
      messageApi.error(
        t("playground:studio.loadNotesError", "Failed to load notes")
      )
      setNotesList([])
    } finally {
      setIsSearching(false)
    }
  }, [messageApi, t, workspaceTag])

  const loadWorkspaceNotes = useCallback(async () => {
    if (!workspaceTag.trim()) {
      setWorkspaceNotes([])
      return
    }

    setIsLoadingWorkspaceNotes(true)
    try {
      const response = await bgRequest<NotesSearchResponse | NoteListItem[]>({
        path: buildSearchPath({
          workspaceToken: workspaceTag,
          limit: WORKSPACE_NOTES_LIMIT
        }),
        method: "GET"
      })
      const notes = pickNotesArray(response)
      const prioritized = prioritizeWorkspaceNotes(notes, workspaceTag)
      setWorkspaceNotes(normalizeNotesForDisplay(prioritized, workspaceTag))
    } catch (error) {
      setWorkspaceNotes([])
    } finally {
      setIsLoadingWorkspaceNotes(false)
    }
  }, [workspaceTag])

  // Debounced search for typing
  const debouncedSearch = useCallback((query: string) => {
    if (searchDebounceRef.current) {
      clearTimeout(searchDebounceRef.current)
    }
    searchDebounceRef.current = setTimeout(() => {
      searchNotes(query)
    }, 300)
  }, [searchNotes])

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (searchDebounceRef.current) {
        clearTimeout(searchDebounceRef.current)
      }
    }
  }, [])

  useEffect(() => {
    loadWorkspaceNotes()
  }, [loadWorkspaceNotes])

  useEffect(() => {
    let isMounted = true

    const loadKeywordCatalog = async () => {
      try {
        const keywords = await getNoteKeywords(200)
        if (!isMounted) return
        const filtered = workspaceTag.trim()
          ? keywords.filter((keyword) => !isKeywordMatch(keyword, workspaceTag))
          : keywords
        setKeywordCatalog(normalizeKeywords(filtered))
      } catch {
        if (isMounted) {
          setKeywordCatalog([])
        }
      }
    }

    loadKeywordCatalog()

    return () => {
      isMounted = false
    }
  }, [workspaceTag])

  // Open load modal and fetch notes
  const handleOpenLoadModal = () => {
    setIsLoadModalOpen(true)
    setSearchQuery("")
    searchNotes()
  }

  // Handle note selection
  const handleSelectNote = useCallback(async (note: NoteListItem) => {
    try {
      // Fetch full note details
      const fullNote = await bgRequest<NoteListItem>({
        path: `/api/v1/notes/${note.id}` as AllowedPath,
        method: "GET"
      })
      hideSavedIndicator()
      loadNote(serializeNoteForEditor(fullNote))
      setIsLoadModalOpen(false)
      messageApi.success(
        t("playground:studio.noteLoaded", "Note loaded")
      )
    } catch (error) {
      messageApi.error(
        t("playground:studio.loadNoteError", "Failed to load note")
      )
    }
  }, [hideSavedIndicator, loadNote, messageApi, serializeNoteForEditor, t])

  const handleReloadLatestAfterConflict = useCallback(async () => {
    if (!currentNote.id) return

    const localDraft = {
      title: currentNote.title.trim(),
      content: currentNote.content.trim(),
      keywords: [...currentNote.keywords]
    }

    try {
      const latest = await bgRequest<NoteListItem>({
        path: `/api/v1/notes/${currentNote.id}` as AllowedPath,
        method: "GET"
      })

      const latestForEditor = serializeNoteForEditor(latest)
      const latestKeywords = normalizeKeywords(latestForEditor.keywords || [])
      const mergedKeywords = normalizeKeywords([
        ...latestKeywords,
        ...localDraft.keywords
      ])

      const titleChanged =
        localDraft.title.length > 0 && localDraft.title !== latestForEditor.title.trim()
      const contentChanged =
        localDraft.content.length > 0 &&
        localDraft.content !== latestForEditor.content.trim()
      const keywordsChanged = localDraft.keywords.some(
        (keyword) =>
          !latestKeywords.some((existing) => isKeywordMatch(existing, keyword))
      )

      const draftBlock = `## Local Draft (Unsaved)\n\n${localDraft.content}`
      const mergedContent = contentChanged
        ? latestForEditor.content.trim()
          ? `${latestForEditor.content.trim()}\n\n---\n\n${draftBlock}`
          : draftBlock
        : latestForEditor.content

      hideSavedIndicator()
      setCurrentNote({
        id: latestForEditor.id,
        title: titleChanged ? localDraft.title : latestForEditor.title,
        content: mergedContent,
        keywords: mergedKeywords,
        version: latestForEditor.version,
        isDirty: titleChanged || contentChanged || keywordsChanged
      })
      setKeywordsInput(mergedKeywords.join(", "))
      messageApi.success(
        t(
          "playground:studio.noteReloadedWithDraft",
          "Loaded latest note and preserved your unsaved draft."
        )
      )
    } catch (error) {
      messageApi.error(
        t(
          "playground:studio.reloadLatestFailed",
          "Failed to load the latest note version."
        )
      )
    }
  }, [currentNote, hideSavedIndicator, messageApi, serializeNoteForEditor, setCurrentNote, t])

  // Save note (create or update)
  const handleSave = async () => {
    if (!currentNote.content.trim() && !currentNote.title.trim()) {
      messageApi.warning(
        t("playground:studio.emptyNoteWarning", "Please add some content or a title")
      )
      return
    }

    setIsSaving(true)
    try {
      const persistedKeywords = buildPersistedKeywords(
        currentNote.keywords,
        workspaceTag
      )

      const payload: Record<string, unknown> = {
        title: currentNote.title || "Untitled Note",
        content: currentNote.content,
        keywords: persistedKeywords.length > 0 ? persistedKeywords : undefined
      }

      // Add workspace tag if available
      if (workspaceTag) {
        payload.workspace_tag = workspaceTag
      }

      if (currentNote.id) {
        // Update existing note with version check
        const path = currentNote.version
          ? `/api/v1/notes/${currentNote.id}?expected_version=${currentNote.version}` as AllowedPath
          : `/api/v1/notes/${currentNote.id}` as AllowedPath

        const updated = await bgRequest<NoteListItem>({
          path,
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: payload
        })

        loadNote(
          serializeNoteForEditor({
            ...updated,
            keywords: updated.keywords || persistedKeywords
          })
        )
        loadWorkspaceNotes()
        messageApi.success(
          t("playground:studio.noteUpdated", "Note updated")
        )
        showSavedIndicatorTemporarily()
      } else {
        // Create new note
        const created = await bgRequest<NoteListItem>({
          path: "/api/v1/notes/" as AllowedPath,
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload
        })

        loadNote(
          serializeNoteForEditor({
            ...created,
            keywords: created.keywords || persistedKeywords
          })
        )
        loadWorkspaceNotes()
        messageApi.success(
          t("playground:studio.noteSaved", "Note saved")
        )
        showSavedIndicatorTemporarily()
      }
    } catch (error: any) {
      // Handle version conflict
      if (error?.message?.includes("version") || error?.status === 409) {
        messageApi.open({
          type: "error",
          key: "workspace-note-version-conflict",
          duration: 8,
          content: (
            <div className="flex items-center gap-2">
              <span>
                {t(
                  "playground:studio.versionConflict",
                  "Note was modified elsewhere. Reload the latest version to merge your draft."
                )}
              </span>
              {currentNote.id ? (
                <Button
                  size="small"
                  type="link"
                  onClick={() => {
                    messageApi.destroy("workspace-note-version-conflict")
                    void handleReloadLatestAfterConflict()
                  }}
                >
                  {t("common:reload", "Reload latest")}
                </Button>
              ) : null}
            </div>
          )
        })
      } else {
        messageApi.error(
          t("playground:studio.noteSaveError", "Failed to save note")
        )
      }
    } finally {
      setIsSaving(false)
    }
  }

  // Handle clear
  const clearNoteWithUndo = () => {
    const previousNote = {
      ...currentNote,
      keywords: [...currentNote.keywords]
    }
    const previousKeywordsInput = keywordsInput
    const undoHandle = scheduleWorkspaceUndoAction({
      apply: () => {
        hideSavedIndicator()
        clearCurrentNote()
        setKeywordsInput("")
      },
      undo: () => {
        setCurrentNote(previousNote)
        setKeywordsInput(previousKeywordsInput)
      }
    })

    const undoMessageKey = `workspace-note-clear-undo-${undoHandle.id}`
    const maybeOpen = (messageApi as { open?: (config: unknown) => void }).open
    const messageConfig = {
      key: undoMessageKey,
      type: "warning",
      duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
      content: t(
        "playground:studio.noteCleared",
        "Note cleared."
      ),
      btn: (
        <Button
          size="small"
          type="link"
          onClick={() => {
            if (undoWorkspaceAction(undoHandle.id)) {
              messageApi.success(
                t("playground:studio.noteRestored", "Note restored")
              )
            }
            messageApi.destroy(undoMessageKey)
          }}
        >
          {t("common:undo", "Undo")}
        </Button>
      )
    }
    if (typeof maybeOpen === "function") {
      maybeOpen(messageConfig)
    } else {
      const maybeWarning = (
        messageApi as { warning?: (content: string) => void }
      ).warning
      if (typeof maybeWarning === "function") {
        maybeWarning(t("playground:studio.noteCleared", "Note cleared."))
      }
    }
  }

  const handleClear = () => {
    if (currentNote.isDirty) {
      Modal.confirm({
        title: t("playground:studio.unsavedChanges", "Unsaved Changes"),
        content: t(
          "playground:studio.unsavedChangesWarning",
          "You have unsaved changes. Are you sure you want to clear?"
        ),
        onOk: () => {
          clearNoteWithUndo()
        }
      })
    } else {
      clearNoteWithUndo()
    }
  }

  const handleExportNote = () => {
    const title = currentNote.title.trim() || t("playground:studio.untitledNote", "Untitled")
    if (!title && !currentNote.content.trim()) {
      messageApi.warning(
        t("playground:studio.emptyNoteWarning", "Please add some content or a title")
      )
      return
    }

    const keywordLine =
      currentNote.keywords.length > 0
        ? `Tags: ${currentNote.keywords.map((keyword) => `#${keyword.replace(/\s+/g, "-")}`).join(" ")}`
        : ""

    const sections = [`# ${title}`]
    if (keywordLine) sections.push(keywordLine)
    if (currentNote.content.trim()) sections.push(currentNote.content)
    const markdown = `${sections.join("\n\n").trim()}\n`
    const filename = `${sanitizeFilename(title)}.md`

    downloadTextFile(markdown, filename, "text/markdown;charset=utf-8")
    messageApi.success(
      t("playground:studio.noteExported", "Note downloaded as Markdown")
    )
  }

  return (
    <div className="flex h-full flex-col border-t border-border p-4">
      {contextHolder}

      {/* Header */}
      <div className="mb-3 flex shrink-0 items-center justify-between">
        <h3 className="text-xs font-semibold uppercase text-text-muted">
          {t("playground:studio.quickNotes", "Quick Notes")}
          {currentNote.id && (
            <span className="ml-2 font-normal normal-case text-primary">
              (ID: {currentNote.id})
            </span>
          )}
        </h3>
        <div className="flex items-center gap-1">
          <Button
            type="text"
            size="small"
            icon={<FolderOpen className="h-3.5 w-3.5" />}
            onClick={handleOpenLoadModal}
            aria-label={t("playground:studio.loadNote", "Load note")}
            title={t("playground:studio.loadNote", "Load note")}
          />
          <Button
            type="text"
            size="small"
            icon={<Download className="h-3.5 w-3.5" />}
            onClick={handleExportNote}
            aria-label={t("playground:studio.exportNote", "Download .md")}
            title={t("playground:studio.exportNote", "Download .md")}
            disabled={!currentNote.content.trim() && !currentNote.title.trim()}
          />
          <Button
            type="text"
            size="small"
            icon={<X className="h-3.5 w-3.5" />}
            onClick={handleClear}
            aria-label={t("common:clear", "Clear")}
            title={t("common:clear", "Clear")}
            disabled={!currentNote.content && !currentNote.title && !currentNote.id}
          />
          {onCollapse && (
            <Button
              type="text"
              size="small"
              icon={<ChevronUp className="h-3.5 w-3.5" />}
              onClick={onCollapse}
              aria-label={t("common:collapse", "Collapse")}
              title={t("common:collapse", "Collapse")}
            />
          )}
        </div>
      </div>

      {workspaceTag && (
        <div className="mb-3 shrink-0 rounded-md border border-border/80 bg-surface2/40 p-2">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">
              {t("playground:studio.workspaceNotes", "Workspace notes")}
            </p>
            <button
              type="button"
              className="text-xs text-primary hover:underline"
              onClick={handleOpenLoadModal}
            >
              {t("playground:studio.viewAllNotes", "View all")}
            </button>
          </div>
          {isLoadingWorkspaceNotes ? (
            <div className="flex items-center justify-center py-2">
              <Spin size="small" />
            </div>
          ) : workspaceNotes.length > 0 ? (
            <div
              data-testid="workspace-notes-list"
              className="custom-scrollbar flex gap-1 overflow-x-auto pb-1"
            >
              {workspaceNotes.map((note) => (
                <button
                  key={note.id}
                  type="button"
                  onClick={() => handleSelectNote(note)}
                  aria-pressed={currentNote.id === note.id}
                  className={`shrink-0 rounded-md border px-2 py-1 text-xs transition ${
                    currentNote.id === note.id
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border text-text hover:border-primary/50 hover:bg-primary/5"
                  }`}
                >
                  {note.title || t("playground:studio.untitledNote", "Untitled")}
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs text-text-muted">
              {t(
                "playground:studio.noWorkspaceNotesYet",
                "No workspace notes yet. Save your first note to pin it here."
              )}
            </p>
          )}
        </div>
      )}

      {/* Title input */}
          <Input
        ref={titleInputRef}
        value={currentNote.title}
        onChange={(e) => {
          hideSavedIndicator()
          updateNoteTitle(e.target.value)
        }}
        placeholder={t("playground:studio.noteTitlePlaceholder", "Note title...")}
        size="small"
        className="mb-2 shrink-0"
      />

      {/* Keywords input */}
      <AutoComplete
        value={keywordsInput}
        onChange={handleKeywordsChange}
        onSelect={handleKeywordSelect}
        options={keywordSuggestions.map((keyword) => ({
          value: keyword,
          label: keyword
        }))}
        filterOption={false}
        className="mb-2 shrink-0"
      >
        <Input
          placeholder={t(
            "playground:studio.noteKeywordsPlaceholder",
            "Keywords (comma-separated)..."
          )}
          size="small"
          prefix={
            <span className="text-xs text-text-muted">
              {t("playground:studio.tags", "Tags")}:
            </span>
          }
        />
      </AutoComplete>

      {/* Display keywords as tags - horizontally scrollable */}
      {currentNote.keywords.length > 0 && (
        <div className="custom-scrollbar mb-2 flex shrink-0 gap-1 overflow-x-auto pb-1">
          {currentNote.keywords.map((kw, idx) => (
            <Tag
              key={idx}
              closable
              onClose={() => {
                hideSavedIndicator()
                const newKeywords = currentNote.keywords.filter((_, i) => i !== idx)
                updateNoteKeywords(newKeywords)
                setKeywordsInput(newKeywords.join(", "))
              }}
              className="shrink-0 text-xs"
            >
              {kw}
            </Tag>
          ))}
        </div>
      )}

      <div className="mb-2 flex shrink-0 items-center justify-end gap-1">
        <Button
          size="small"
          type={editorMode === "edit" ? "primary" : "text"}
          icon={<PencilLine className="h-3.5 w-3.5" />}
          onClick={() => setEditorMode("edit")}
          aria-pressed={editorMode === "edit"}
        >
          {t("playground:studio.notesEditMode", "Edit")}
        </Button>
        <Button
          size="small"
          type={editorMode === "preview" ? "primary" : "text"}
          icon={<Eye className="h-3.5 w-3.5" />}
          onClick={() => setEditorMode("preview")}
          aria-pressed={editorMode === "preview"}
        >
          {t("playground:studio.notesPreviewMode", "Preview")}
        </Button>
      </div>

      {/* Content area - fills remaining space */}
      <div className="min-h-0 flex-1">
        {editorMode === "edit" ? (
          <TextArea
            ref={contentInputRef}
            value={currentNote.content}
            onChange={(e) => {
              hideSavedIndicator()
              updateNoteContent(e.target.value)
            }}
            placeholder={t(
              "playground:studio.notesPlaceholder",
              "Jot down notes, ideas, or observations..."
            )}
            className="h-full !resize-none text-sm [&_.ant-input]:!h-full"
            style={{ height: "100%", minHeight: "80px" }}
          />
        ) : (
          <div
            data-testid="quick-notes-markdown-preview"
            className="custom-scrollbar h-full overflow-y-auto rounded-md border border-border bg-surface2/40 p-3"
          >
            {currentNote.content.trim() ? (
              <MarkdownPreview content={currentNote.content} size="sm" />
            ) : (
              <p className="text-xs text-text-muted">
                {t(
                  "playground:studio.notesPreviewEmpty",
                  "Nothing to preview yet. Start writing in Edit mode."
                )}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Save button */}
      {(currentNote.content.trim() || currentNote.title.trim() || currentNote.isDirty) && (
        <div className="mt-2 flex shrink-0 items-center justify-between">
          <div className="flex items-center gap-2">
            {currentNote.isDirty && !showSavedIndicator && (
              <span className="flex items-center gap-1 text-xs text-warning">
                <AlertCircle className="h-3 w-3" />
                {t("playground:studio.unsaved", "Unsaved")}
              </span>
            )}
            {showSavedIndicator && !currentNote.isDirty && (
              <span className="flex items-center gap-1 text-xs text-success" data-testid="quick-notes-saved-indicator">
                <Check className="h-3 w-3" />
                {t("playground:studio.saved", "Saved")}
              </span>
            )}
          </div>
          <Button
            size="small"
            type="primary"
            icon={<Save className="h-3.5 w-3.5" />}
            onClick={handleSave}
            loading={isSaving}
          >
            {currentNote.id
              ? t("playground:studio.updateNote", "Update")
              : t("playground:studio.saveNote", "Save")}
          </Button>
        </div>
      )}

      {/* Load Note Modal */}
      <Modal
        title={
          <span className="flex items-center gap-2">
            <FolderOpen className="h-4 w-4" />
            {t("playground:studio.loadNoteTitle", "Load Note")}
          </span>
        }
        open={isLoadModalOpen}
        onCancel={() => setIsLoadModalOpen(false)}
        footer={null}
        width={500}
      >
        {/* Search input */}
        <Input
          prefix={<Search className="h-4 w-4 text-text-muted" />}
          placeholder={t("playground:studio.searchNotes", "Search notes...")}
          value={searchQuery}
          onChange={(e) => {
            const value = e.target.value
            setSearchQuery(value)
            // If cleared (empty), search immediately; otherwise debounce
            if (!value) {
              if (searchDebounceRef.current) {
                clearTimeout(searchDebounceRef.current)
              }
              searchNotes("")
            } else {
              debouncedSearch(value)
            }
          }}
          allowClear
          className="mb-4"
        />

        {/* Notes list */}
        <div className="max-h-80 overflow-y-auto">
          {isSearching ? (
            <div className="flex items-center justify-center py-8">
              <Spin />
            </div>
          ) : notesList.length === 0 ? (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                <span className="text-text-muted">
                  {searchQuery
                    ? t("playground:studio.noNotesFound", "No notes found")
                    : t("playground:studio.noNotesYet", "No notes yet")}
                </span>
              }
            />
          ) : (
            <div className="space-y-2">
              {notesList.map((note) => {
                const noteKeywords = extractNoteKeywords(note)
                const workspaceScoped = isWorkspaceTaggedNote(note, workspaceTag)
                return (
                  <button
                    key={note.id}
                    type="button"
                    onClick={() => handleSelectNote(note)}
                    className="flex w-full items-start gap-3 rounded-lg border border-border p-3 text-left transition hover:border-primary/50 hover:bg-primary/5"
                  >
                    <FileText className="mt-0.5 h-4 w-4 shrink-0 text-text-muted" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-text">
                        {note.title || "Untitled"}
                      </p>
                      <p className="line-clamp-2 text-xs text-text-muted">
                        {note.content?.slice(0, 100) || "No content"}
                      </p>
                      <div className="mt-1 flex flex-wrap items-center gap-1">
                        {workspaceScoped && (
                          <Tag color="blue" className="text-xs">
                            {t("playground:studio.workspaceScoped", "Workspace")}
                          </Tag>
                        )}
                        {noteKeywords.slice(0, 3).map((kw, idx) => (
                          <Tag key={idx} className="text-xs">
                            {kw}
                          </Tag>
                        ))}
                        {noteKeywords.length > 3 && (
                          <span className="text-xs text-text-muted">
                            +{noteKeywords.length - 3}
                          </span>
                        )}
                      </div>
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </Modal>
    </div>
  )
}

export default QuickNotesSection
