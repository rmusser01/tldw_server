import { canonicalWikilinkNoteId } from "@/components/Notes/wikilinks"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import { createNotesGraphAuthorityScope } from "@/components/Notes/hooks/useNotesGraphAuthorityScope"
import {
  readSurfaceOfflineDraftQueue,
  checkpointQuickNotesOfflineDraft,
  quickNotesDraftMatchesWrite,
  isQuickNotesRetainedDraft,
  retainSurfaceOfflineDraft,
  retireSurfaceOfflineDraft,
  type OfflineDraftEntry
} from "@/components/Notes/notes-manager-utils"
import { isDefinitiveWriteRejection, isNotesProvenancePolicyUnavailable, NOTES_PROVENANCE_UNAVAILABLE_MESSAGE } from "@/services/tldw/api-error"
import { KnowledgeNoteHistory } from "@/components/Notes/KnowledgeNoteHistory"
import {
  knowledgeNoteHead,
  knowledgeNoteWriteFields,
  knowledgeNoteProvenanceMatches,
  resolveKnowledgeNoteProvenance,
  type KnowledgeNoteHead,
  retainKnowledgeNoteProvenance,
  stripKnowledgeNoteProvenance
} from "@/utils/knowledge-note-provenance"
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
import {
  hasResearchWorkspaceMigrationTombstone,
  useWorkspaceStore,
} from "@/store/workspace";
import { bgRequest } from "@/services/background-proxy"
import type { BgRequestInit } from "@/services/background-proxy"
import {
  loadServicePromptSnapshot,
  type ServicePromptSnapshot
} from "@/services/service-prompts"
import {
  requestScopeFields,
  servicePromptAuthorityKey,
} from "@/services/tldw/domains/service-prompts";
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import { MarkdownPreview } from "@/components/Common/MarkdownPreview"
import { getNoteKeywords } from "@/services/note-keywords"
import {
  WORKSPACE_UNDO_WINDOW_MS,
  scheduleWorkspaceUndoAction,
  undoWorkspaceAction
} from "../undo-manager"
import { renderWorkspaceMessageActionContent } from "../workspace-message-content"

const { TextArea } = Input

type NoteKeyword =
  | string
  | {
      keyword?: string
      keyword_text?: string
      text?: string
      name?: string
    }

interface NoteListItem extends KnowledgeNoteHead {
  id: string | number
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
  const mergedMap = new Map<string | number, NoteListItem>()
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
  const messageApiRef = useRef(messageApi)
  messageApiRef.current = messageApi

  // Store state
  const currentNote = useWorkspaceStore((s) => s.currentNote)
  const workspaceTag = useWorkspaceStore((s) => s.workspaceTag)
  const activeWorkspaceId = useWorkspaceStore((s) => s.workspaceId)
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
  const saveControllerRef = useRef<AbortController | null>(null)
  const readControllerRef = useRef<AbortController | null>(null)
  const pendingSaveRef = useRef<{
    authorityScope: string;
    entry: OfflineDraftEntry;
    scopeKey: string;
    workspaceId: string | null;
    noteId: string | number | undefined;
    draft: typeof currentNote;
    path: AllowedPath;
    method: "POST" | "PUT";
    body: Record<string, unknown>;
    headers: Record<string, string>;
  } | null>(null);
  const draftBindingRef = useRef<{
    authorityScope: string;
    authorityId: string;
    key: string;
    pendingKey: string;
    workspaceId: string | null;
    workspaceTag: string;
    noteId: typeof currentNote.id;
    active: boolean;
    latest: typeof currentNote;
  } | null>(null);
  const [recoverableDrafts, setRecoverableDrafts] = useState<
    OfflineDraftEntry[]
  >([]);
  const [recoveryRevision, setRecoveryRevision] = useState(0);
  const draftSnapshot = (
    binding: NonNullable<typeof draftBindingRef.current>,
  ) =>
    binding.active
      ? {
          title: binding.latest.title,
          content: binding.latest.content,
          keywords: binding.latest.keywords,
          isDirty: binding.latest.isDirty,
          metadata: {
            ...knowledgeNoteHead(binding.latest),
            pendingKnowledgeProvenance:
              binding.latest.pendingKnowledgeProvenance,
            quickNotesAuthorityId: binding.authorityId,
            quickNotesWorkspaceId: binding.workspaceId,
            quickNotesWorkspaceTag: binding.workspaceTag,
          },
        }
      : null;
  const flushBoundDraft = useCallback(() => {
    const binding = draftBindingRef.current;
    if (!binding?.active) return Promise.resolve(null);
    return checkpointQuickNotesOfflineDraft(
      binding.authorityScope,
      binding.key,
      binding.pendingKey,
      () => draftSnapshot(binding),
    ).catch(() => {
      messageApiRef.current.error(
        "Could not retain the current local note draft on this device. Keep the editor open and retry after storage is available.",
      );
      return null;
    });
  }, []);
  useEffect(() => {
    const stop = useWorkspaceStore.subscribe((state) => {
      const binding = draftBindingRef.current;
      if (!binding?.active) return;
      if (
        state.workspaceId !== binding.workspaceId ||
        state.workspaceTag !== binding.workspaceTag ||
        state.currentNote.id !== binding.noteId ||
        state.currentNote.pendingNoteWriteKey !== binding.key
      ) {
        binding.active = false;
        draftBindingRef.current = null;
        return;
      }
      binding.latest = state.currentNote;
      void flushBoundDraft();
    });
    const stopOwner = watchChatAccountChanges((invalidated) => {
      if (!invalidated) return;
      if (draftBindingRef.current) draftBindingRef.current.active = false;
      draftBindingRef.current = null;
      setRecoverableDrafts([]);
      setRecoveryRevision((value) => value + 1);
    });
    const flush = () => {
      void flushBoundDraft();
    };
    window.addEventListener("pagehide", flush);
    return () => {
      flush();
      stop();
      stopOwner();
      window.removeEventListener("pagehide", flush);
    };
  }, [flushBoundDraft]);

  const blankDraft =
    !currentNote.id && !currentNote.title && !currentNote.content;
  useEffect(() => {
    if (
      !blankDraft ||
      !activeWorkspaceId ||
      !hasResearchWorkspaceMigrationTombstone(activeWorkspaceId)
    )
      return;
    const controller = new AbortController();
    const stopOwner = watchChatAccountChanges((invalidated) => {
      if (invalidated) controller.abort();
    });
    void (async () => {
      const scope = await loadServicePromptSnapshot([], {
        signal: controller.signal,
      });
      try {
        const user =
          scope.requestScope.userId == null
            ? await tldwAuth.getCurrentUser()
            : null;
        const ownerId =
          scope.requestScope.userId ?? (user?.is_active ? user.id : null);
        if (ownerId == null) return;
        const authority = servicePromptAuthorityKey(scope.requestScope);
        const queue = await readSurfaceOfflineDraftQueue(
          createNotesGraphAuthorityScope(
            scope.requestScope.config.serverUrl,
            ownerId,
          ),
        );
        if (
          controller.signal.aborted ||
          scope.scopeSignal.aborted ||
          scope.scopeInvalidatedSignal.aborted
        )
          return;
        const prefix = `surface:quick-notes:${JSON.stringify([activeWorkspaceId, workspaceTag])}:`;
        setRecoverableDrafts(
          Object.values(queue).filter(
            (entry) =>
              entry.key.startsWith(prefix) &&
              (entry.pendingWrite?.authorityId === authority ||
                (isQuickNotesRetainedDraft(entry) &&
                  entry.metadata?.quickNotesAuthorityId === authority &&
                  entry.metadata.quickNotesDirty)),
          ),
        );
      } finally {
        scope.release();
      }
    })().catch(() => {
      if (!controller.signal.aborted)
        messageApiRef.current.error(
          "Could not read retained local notes on this device.",
        );
    });
    return () => {
      controller.abort();
      stopOwner();
    };
  }, [activeWorkspaceId, workspaceTag, blankDraft, recoveryRevision]);

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
      saveControllerRef.current?.abort()
      readControllerRef.current?.abort()
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
  const lastLoadedNoteId = useRef<string | number | undefined>(undefined)

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
      content: retainKnowledgeNoteProvenance(note.content || "", note),
      ...knowledgeNoteHead(note),
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

  const loadWorkspaceNotes = useCallback(
    async (operation?: {
      request: Pick<
        BgRequestInit,
        "servicePromptConfig" | "headers" | "abortSignal"
      >
      isCurrent: () => boolean
    }) => {
      const isCurrent = operation?.isCurrent ?? (() => true)
      if (!isCurrent()) return
      if (!workspaceTag.trim()) {
        setWorkspaceNotes([])
        return
      }

      setIsLoadingWorkspaceNotes(true)
      try {
        const response = await bgRequest<NotesSearchResponse | NoteListItem[]>({
          ...operation?.request,
          path: buildSearchPath({
            workspaceToken: workspaceTag,
            limit: WORKSPACE_NOTES_LIMIT
          }),
          method: "GET"
        })
        if (!isCurrent()) return
        const notes = pickNotesArray(response)
        const prioritized = prioritizeWorkspaceNotes(notes, workspaceTag)
        setWorkspaceNotes(normalizeNotesForDisplay(prioritized, workspaceTag))
      } catch {
        if (isCurrent()) setWorkspaceNotes([])
      } finally {
        setIsLoadingWorkspaceNotes(false)
      }
    },
    [workspaceTag]
  )

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

  const readCurrentNoteDetails = useCallback(
    async (
      id: string | number,
      expected: ReturnType<typeof useWorkspaceStore.getState>,
      apply: (note: NoteListItem) => void,
    ) => {
      const unchanged = () => {
        const state = useWorkspaceStore.getState();
        return (
          state.currentNote === expected.currentNote &&
          state.workspaceId === expected.workspaceId &&
          state.workspaceTag === expected.workspaceTag
        );
      };
      if (!unchanged()) return;
      readControllerRef.current?.abort();
      const controller = new AbortController();
      readControllerRef.current = controller;
      const stopDraft = useWorkspaceStore.subscribe(() => {
        if (!unchanged()) controller.abort();
      });
      const stopOwner = watchChatAccountChanges((invalidated) => {
        if (invalidated) controller.abort();
      });
      let scope: ServicePromptSnapshot | undefined;
      try {
        scope = await loadServicePromptSnapshot([], {
          signal: controller.signal,
        });
        if (controller.signal.aborted || scope.scopeInvalidatedSignal.aborted)
          return;
        const note = await bgRequest<NoteListItem>({
          ...requestScopeFields(scope.requestScope),
          abortSignal: scope.scopeSignal,
          path: `/api/v1/notes/${encodeURIComponent(String(id))}` as AllowedPath,
          method: "GET",
        });
        if (
          !unchanged() ||
          controller.signal.aborted ||
          scope.scopeSignal.aborted ||
          scope.scopeInvalidatedSignal.aborted
        )
          return;
        apply(note);
      } catch (error) {
        if (!controller.signal.aborted) throw error;
      } finally {
        stopDraft();
        stopOwner();
        scope?.release();
      }
    },
    [],
  );

  // Confirmation and its eventual GET both belong to the exact draft being replaced.
  const handleSelectNote = useCallback(
    (note: NoteListItem) => {
      const expected = useWorkspaceStore.getState();
      const select = async () => {
        try {
          await readCurrentNoteDetails(note.id, expected, (fullNote) => {
            hideSavedIndicator();
            loadNote(serializeNoteForEditor(fullNote));
            setIsLoadModalOpen(false);
            messageApi.success(
              t("playground:studio.noteLoaded", "Note loaded"),
            );
          });
        } catch {
          messageApi.error(
            t("playground:studio.loadNoteError", "Failed to load note"),
          );
        }
      };
      if (expected.currentNote.isDirty)
        Modal.confirm({
          title: t("playground:studio.unsavedChanges", "Unsaved Changes"),
          content: "Replace the current unsaved draft with the saved note?",
          onOk: select,
        });
      else void select();
    },
    [
      hideSavedIndicator,
      loadNote,
      messageApi,
      readCurrentNoteDetails,
      serializeNoteForEditor,
      t,
    ],
  );

  const handleReloadLatestAfterConflict = useCallback(async () => {
    const expected = useWorkspaceStore.getState();
    if (!expected.currentNote.id) return;
    const localDraft = {
      title: expected.currentNote.title.trim(),
      content: expected.currentNote.content.trim(),
      keywords: [...expected.currentNote.keywords],
    };

    try {
      await readCurrentNoteDetails(
        expected.currentNote.id,
        expected,
        (latest) => {
          const latestForEditor = serializeNoteForEditor(latest);
          const latestKeywords = normalizeKeywords(
            latestForEditor.keywords || [],
          );
          const mergedKeywords = normalizeKeywords([
            ...latestKeywords,
            ...localDraft.keywords,
          ]);

          const titleChanged =
            localDraft.title.length > 0 &&
            localDraft.title !== latestForEditor.title.trim();
          const contentChanged =
            localDraft.content.length > 0 &&
            localDraft.content !== latestForEditor.content.trim();
          const keywordsChanged = localDraft.keywords.some(
            (keyword) =>
              !latestKeywords.some((existing) =>
                isKeywordMatch(existing, keyword),
              ),
          );

          const draftBlock = `## Local Draft (Unsaved)\n\n${localDraft.content}`;
          const mergedContent = contentChanged
            ? latestForEditor.content.trim()
              ? `${latestForEditor.content.trim()}\n\n---\n\n${draftBlock}`
              : draftBlock
            : latestForEditor.content;

          hideSavedIndicator();
          setCurrentNote({
            ...knowledgeNoteHead(latestForEditor),
            id: latestForEditor.id,
            title: titleChanged ? localDraft.title : latestForEditor.title,
            content: mergedContent,
            keywords: mergedKeywords,
            version: latestForEditor.version,
            isDirty: titleChanged || contentChanged || keywordsChanged,
          });
          setKeywordsInput(mergedKeywords.join(", "));
          messageApi.success(
            t(
              "playground:studio.noteReloadedWithDraft",
              "Loaded latest note and preserved your unsaved draft.",
            ),
          );
        },
      );
    } catch (error) {
      messageApi.error(
        t(
          "playground:studio.reloadLatestFailed",
          "Failed to load the latest note version.",
        ),
      );
    }
  }, [
    hideSavedIndicator,
    messageApi,
    readCurrentNoteDetails,
    serializeNoteForEditor,
    setCurrentNote,
    t,
  ]);

  // Save one captured draft under one owner through version lookup and write.
  const handleSave = async (recovery?: { key: string; workspaceId: string | null; workspaceTag: string | null }) => {
    if (saveControllerRef.current && !saveControllerRef.current.signal.aborted)
      return
    const {
      currentNote: draft,
      workspaceId,
      workspaceTag: draftWorkspaceTag
    } = useWorkspaceStore.getState()
    if (
      recovery &&
      (recovery.workspaceId !== workspaceId ||
        recovery.workspaceTag !== draftWorkspaceTag)
    ) {
      messageApi.error(
        "Restore the previous Workspace before retrying its save.",
      );
      return;
    }
    if (!recovery && !draft.content.trim() && !draft.title.trim()) {
      messageApi.warning(
        t(
          "playground:studio.emptyNoteWarning",
          "Please add some content or a title",
        ),
      );
      return;
    }

    const controller = new AbortController()
    saveControllerRef.current = controller
    let acknowledged = false;
    let rejectedOperationCleared = false;
    let attemptedOperation: typeof pendingSaveRef.current = null;
    let scope: ServicePromptSnapshot | undefined
    let noteId = draft.id
    const operationPrefix = `surface:quick-notes:${JSON.stringify([workspaceId, draftWorkspaceTag])}:`;
    let operationKey = draft.pendingNoteWriteKey?.startsWith(operationPrefix)
      ? draft.pendingNoteWriteKey
      : undefined;
    const hasCurrentIdentity = () => {
      const latest = useWorkspaceStore.getState()
      return (
        latest.workspaceId === workspaceId &&
        latest.workspaceTag === draftWorkspaceTag &&
        (!!recovery || latest.currentNote.id === noteId) &&
        (!!recovery ||
          !operationKey ||
          latest.currentNote.pendingNoteWriteKey === operationKey)
      );
    }
    const isCurrent = () =>
      !controller.signal.aborted &&
      !scope?.scopeSignal.aborted &&
      !scope?.scopeInvalidatedSignal.aborted &&
      hasCurrentIdentity()
    // Latch replacement/clear events, including a switch away and back.
    const stopWatchingNote = useWorkspaceStore.subscribe((state) => {
      const clearedNewDraft =
        !recovery &&
        noteId == null &&
        state.currentNote !== draft &&
        !state.currentNote.title &&
        !state.currentNote.content;
      if (!hasCurrentIdentity() || clearedNewDraft) { controller.abort(); pendingSaveRef.current = null }
    })
    const stopWatchingOwner = watchChatAccountChanges((invalidated) => {
      if (invalidated) { controller.abort(); pendingSaveRef.current = null }
    })
    setIsSaving(true)
    try {
      scope = await loadServicePromptSnapshot([], { signal: controller.signal })
      if (!isCurrent()) return
      const user =
        scope.requestScope.userId == null
          ? await tldwAuth.getCurrentUser()
          : null;
      const ownerId =
        scope.requestScope.userId ?? (user?.is_active ? user.id : null);
      if (!isCurrent()) return;
      if (ownerId == null)
        throw new Error("Verify your account before saving a note.");
      const authorityScope = createNotesGraphAuthorityScope(
        scope.requestScope.config.serverUrl,
        ownerId,
      );
      const authorityId = servicePromptAuthorityKey(scope.requestScope);
      const scopedFields = requestScopeFields(scope.requestScope)
      const request = { ...scopedFields, abortSignal: scope.scopeSignal }
      const persistedKeywords = buildPersistedKeywords(
        [
          ...draft.keywords,
          ...(resolveKnowledgeNoteProvenance(draft, draft.content).provenance?.research
            ? [
                `workspace:${resolveKnowledgeNoteProvenance(draft, draft.content).provenance!.research!.workspace_id}`
              ]
            : [])
        ],
        draftWorkspaceTag
      )
      const payload: Record<string, unknown> = {
        title: draft.title || "Untitled Note",
        content: retainKnowledgeNoteProvenance(draft.content, draft),
        ...knowledgeNoteWriteFields(draft.content, draft, {
          create: !draft.id,
          replacement: draft.pendingKnowledgeProvenance
        }),
        keywords: persistedKeywords.length > 0 ? persistedKeywords : undefined
      }
      if (payload.knowledge_provenance)
        payload.content = retainKnowledgeNoteProvenance(draft.content, {
          knowledge_provenance_state: "active",
          knowledge_provenance: payload.knowledge_provenance
        })
      if (draftWorkspaceTag) payload.workspace_tag = draftWorkspaceTag

      let pending = recovery ? null : pendingSaveRef.current;
      if (pending && (pending.scopeKey !== scope.scopeKey || pending.workspaceId !== workspaceId || pending.noteId !== draft.id)) pending = null
      const canonicalNoteId = canonicalWikilinkNoteId(
        typeof draft.id === "string" ? draft.id : null,
      );
      const migrationSuppressed =
        !!workspaceId && hasResearchWorkspaceMigrationTombstone(workspaceId);
      if (
        !pending &&
        (recovery || operationKey || canonicalNoteId || migrationSuppressed)
      ) {
        const queue = await readSurfaceOfflineDraftQueue(authorityScope);
        if (!isCurrent()) return;
        const workspaceEntries = Object.values(queue).filter(
          (entry) =>
            entry.key.startsWith(operationPrefix) && entry.pendingWrite,
        );
        const candidates = operationKey
          ? []
          : workspaceEntries.filter(
              (entry) =>
                canonicalNoteId &&
                canonicalWikilinkNoteId(entry.noteId) === canonicalNoteId,
            );
        if (
          candidates.length > 1 ||
          (recovery && workspaceEntries.length !== 1)
        )
          throw new Error(
            "Multiple or missing unresolved operations prevent a safe retry; no save was sent.",
          );
        if (recovery && !migrationSuppressed)
          throw new Error(
            "The previous save requires its durable Workspace binding before retrying.",
          );
        if (
          !recovery &&
          !operationKey &&
          !candidates.length &&
          migrationSuppressed &&
          workspaceEntries.length
        ) {
          if (workspaceEntries.length !== 1)
            throw new Error(
              "Multiple unresolved operations are retained for this Workspace; no save was sent.",
            );
          const retained = workspaceEntries[0];
          if (retained.pendingWrite?.authorityId !== authorityId)
            throw new Error(
              "The retained note operation belongs to a different service authority. Restore that connection before retrying.",
            );
          messageApi.open({
            type: "warning",
            key: "workspace-note-previous-save",
            duration: 0,
            content: (
              <div className="flex items-center gap-2">
                <span>
                  Previous save retained: {retained.title || "Untitled Note"}.
                  Resolve it before saving this draft.
                </span>
                <Button
                  size="small"
                  type="link"
                  onClick={() => {
                    messageApi.destroy("workspace-note-previous-save");
                    void handleSave({
                      key: retained.key,
                      workspaceId,
                      workspaceTag: draftWorkspaceTag,
                    });
                  }}
                >
                  Retry previous save
                </Button>
              </div>
            ),
          });
          return;
        }
        const entry = recovery
          ? queue[recovery.key]
          : operationKey
            ? queue[operationKey]
            : candidates[0];
        if (
          recovery &&
          (!entry ||
            !entry.key.startsWith(operationPrefix) ||
            !entry.pendingWrite)
        )
          throw new Error(
            "The previous save is no longer available in this Workspace; no save was sent.",
          );
        if (
          !operationKey &&
          !recovery &&
          entry?.pendingWrite &&
          entry.pendingWrite.body.id != null &&
          canonicalWikilinkNoteId(String(entry.pendingWrite.body.id)) !==
            canonicalNoteId
        )
          throw new Error(
            "The retained update does not match this canonical Note; no save was sent.",
          );
        if (
          entry?.pendingWrite &&
          entry.pendingWrite.authorityId !== authorityId
        )
          throw new Error(
            "The retained note operation belongs to a different service authority. Restore that connection before retrying.",
          );
        if (
          entry?.pendingWrite &&
          (recovery ||
            entry.noteId === (draft.id == null ? null : String(draft.id)) ||
            (entry.noteId == null && entry.pendingWrite.body.id === draft.id))
        ) {
          pending = {
            authorityScope,
            entry,
            scopeKey: scope.scopeKey,
            workspaceId,
            noteId: draft.id,
            draft: {
              ...entry.metadata,
              id: draft.id,
              title: entry.title,
              content: entry.content,
              keywords: entry.keywords,
              version: entry.baseVersion ?? undefined,
              isDirty: true,
            },
            path: entry.noteId
              ? (`/api/v1/notes/${encodeURIComponent(entry.noteId)}` as AllowedPath)
              : "/api/v1/notes/",
            method: entry.noteId ? "PUT" : "POST",
            body: entry.pendingWrite.body,
            headers: {
              "Content-Type": "application/json",
              "Idempotency-Key": entry.pendingWrite.key,
              ...(entry.pendingWrite.expectedVersion != null
                ? {
                    "expected-version": String(
                      entry.pendingWrite.expectedVersion,
                    ),
                  }
                : {}),
            },
          };
        }
      }
      if (!pending) {
        const path = draft.id ? `/api/v1/notes/${encodeURIComponent(String(draft.id))}` as AllowedPath : "/api/v1/notes/"
        let expectedVersion = draft.id ? draft.version : undefined;
        if (draft.id && expectedVersion == null) {
          const remote = await bgRequest<NoteListItem>({ ...request, path, method: "GET" })
          if (!isCurrent()) return
          expectedVersion = remote.version
          delete payload.knowledge_provenance
          delete payload.expected_provenance_version
          Object.assign(
            payload,
            knowledgeNoteWriteFields(draft.content, remote, {
              replacement: draft.pendingKnowledgeProvenance
            })
          )
          payload.content = retainKnowledgeNoteProvenance(
            draft.content,
            payload.knowledge_provenance
              ? {
                  knowledge_provenance_state: "active",
                  knowledge_provenance: payload.knowledge_provenance
                }
              : remote
          )
        }
        if (draft.id && expectedVersion == null) throw new Error("Missing note version; reload before saving.")
        const queueKey =
          operationKey || `${operationPrefix}${crypto.randomUUID()}`;
        if (!draft.id) payload.id = crypto.randomUUID();
        const key = crypto.randomUUID();
        const body = JSON.parse(JSON.stringify(payload)) as Record<
          string,
          unknown
        >;
        const entry: OfflineDraftEntry = {
          key: queueKey,
          noteId: draft.id == null ? null : String(draft.id),
          baseVersion: expectedVersion ?? null,
          title: draft.title,
          content: draft.content,
          keywords: [...draft.keywords],
          metadata: {
            ...knowledgeNoteHead(draft),
            quickNotesAuthorityId: authorityId,
            quickNotesWorkspaceId: workspaceId,
            quickNotesWorkspaceTag: draftWorkspaceTag,
            pendingKnowledgeProvenance: draft.pendingKnowledgeProvenance,
          },
          backlinkConversationId: null,
          backlinkMessageId: null,
          updatedAt: new Date().toISOString(),
          syncState: "queued",
          lastError: null,
          pendingWrite: {
            authorityId,
            key,
            body,
            expectedVersion: expectedVersion ?? null,
          },
        };
        pending = {
          authorityScope,
          entry,
          scopeKey: scope.scopeKey,
          workspaceId,
          noteId: draft.id,
          draft,
          path,
          method: draft.id ? "PUT" : "POST",
          body,
          headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": key,
            ...(draft.id
              ? { "expected-version": String(expectedVersion) }
              : {}),
          },
        };
      }
      if (pending.entry.pendingWrite?.authorityId !== authorityId)
        throw new Error(
          "The retained note operation belongs to a different service authority. Restore that connection before retrying.",
        );
      if (!recovery) pendingSaveRef.current = pending;
      pending.entry.metadata = {
        ...pending.entry.metadata,
        quickNotesAuthorityId: authorityId,
        quickNotesWorkspaceId: workspaceId,
        quickNotesWorkspaceTag: draftWorkspaceTag,
      };
      await retainSurfaceOfflineDraft(authorityScope, pending.entry);
      if (!isCurrent()) return;
      if (!recovery) operationKey = pending.entry.key;
      if (
        !recovery &&
        useWorkspaceStore.getState().currentNote.pendingNoteWriteKey !==
          operationKey
      )
        setCurrentNote({
          ...useWorkspaceStore.getState().currentNote,
          pendingNoteWriteKey: operationKey,
        });
      if (!recovery) {
        draftBindingRef.current = {
          authorityScope,
          authorityId,
          key: pending.entry.key,
          pendingKey: pending.entry.pendingWrite!.key,
          workspaceId,
          workspaceTag: draftWorkspaceTag,
          noteId: useWorkspaceStore.getState().currentNote.id,
          active: true,
          latest: useWorkspaceStore.getState().currentNote,
        };
        if (!(await flushBoundDraft()))
          throw new Error("Could not retain the current note draft safely.");
        if (!isCurrent()) return;
      }
      const canonicalUpdateBound =
        canonicalNoteId &&
        canonicalWikilinkNoteId(pending.entry.noteId) === canonicalNoteId &&
        pending.method === "PUT" &&
        pending.path ===
          `/api/v1/notes/${encodeURIComponent(pending.entry.noteId!)}` &&
        (pending.body.id == null ||
          canonicalWikilinkNoteId(String(pending.body.id)) === canonicalNoteId);
      // Migration intentionally removes local Workspace snapshots. The owned immutable
      // operation remains recoverable only through the explicit previous-save action.
      if (!canonicalUpdateBound && !migrationSuppressed) {
        const persistence = useWorkspaceStore.persist.getOptions();
        const persistedWorkspace = await persistence.storage?.getItem(
          persistence.name,
        );
        if (!isCurrent()) return;
        const persistedSnapshot = workspaceId
          ? persistedWorkspace?.state?.workspaceSnapshots?.[workspaceId]
          : undefined;
        if (
          persistedSnapshot?.workspaceTag !== draftWorkspaceTag ||
          persistedSnapshot.currentNote?.pendingNoteWriteKey !== operationKey
        )
          throw new Error(
            "Could not retain the note recovery pointer in this Workspace. Retry after storage is available.",
          );
      }
      attemptedOperation = pending;
      const saved = await bgRequest<NoteListItem>({ ...request, path: pending.path, method: pending.method, headers: { ...request.headers, ...pending.headers }, body: pending.body })
      if (!isCurrent()) return
      const latest = useWorkspaceStore.getState().currentNote
      if (pending.draft.pendingKnowledgeProvenance && pending.body.knowledge_provenance &&
          !knowledgeNoteProvenanceMatches(resolveKnowledgeNoteProvenance(saved).provenance, pending.body.knowledge_provenance)) {
        throw new Error("Capture source history was not confirmed; retry the saved draft.")
      }
      if (recovery) {
        acknowledged = true;
        const retained = await checkpointQuickNotesOfflineDraft(
          authorityScope,
          pending.entry.key,
          pending.entry.pendingWrite!.key,
          () => null,
          saved,
        );
        if (!isCurrent()) return;
        setRecoverableDrafts(retained?.metadata?.quickNotesDirty ? [retained] : []);
        pendingSaveRef.current = null;
        messageApi.open({
          type: "success",
          duration: 8,
          content: (
            <div className="flex items-center gap-2">
              <span>
                Previous note saved:{" "}
                {saved.title || pending.entry.title || "Untitled Note"}.
              </span>
              <Button size="small" type="link" onClick={handleOpenLoadModal}>
                Load saved note
              </Button>
            </div>
          ),
        });
        await loadWorkspaceNotes({ request, isCurrent });
        return;
      }
      // The acknowledgment may assign the first canonical ID to this draft.
      noteId = saved.id
      pending.noteId = saved.id;
      if (draftBindingRef.current) draftBindingRef.current.noteId = saved.id;
      const unchanged = quickNotesDraftMatchesWrite(
        {
          ...latest,
          metadata: {
            ...knowledgeNoteHead(latest),
            quickNotesWorkspaceTag: draftWorkspaceTag,
            pendingKnowledgeProvenance: latest.pendingKnowledgeProvenance,
          },
        },
        pending.body,
      );
      if (unchanged) {
        setCurrentNote({
          ...serializeNoteForEditor({
            ...saved,
            keywords: saved.keywords || persistedKeywords,
          }),
          pendingNoteWriteKey: operationKey,
          isDirty: false,
        });
      } else {
        setCurrentNote({
          ...latest,
          pendingNoteWriteKey: operationKey,
          ...(latest.pendingKnowledgeProvenance &&
          knowledgeNoteProvenanceMatches(
            latest.pendingKnowledgeProvenance,
            pending.body.knowledge_provenance,
          )
            ? { pendingKnowledgeProvenance: undefined }
            : {}),
          id: saved.id,
          version: saved.version,
          ...((saved.knowledge_provenance_version || 0) >=
          (latest.knowledge_provenance_version || 0)
            ? knowledgeNoteHead(saved)
            : {}),
          isDirty: true,
        });
      }
      acknowledged = true;
      // An accepted create still needs a recoverable canonical binding before
      // its immutable operation can be removed. Known UUID updates already
      // have that independent identity; migration never recreates snapshots.
      if (
        !(
          canonicalUpdateBound &&
          canonicalWikilinkNoteId(String(saved.id)) === canonicalNoteId
        )
      ) {
        const persistence = useWorkspaceStore.persist.getOptions();
        const persistedWorkspace = await persistence.storage?.getItem(
          persistence.name,
        );
        if (!isCurrent()) return;
        const persistedSnapshot = workspaceId
          ? persistedWorkspace?.state?.workspaceSnapshots?.[workspaceId]
          : undefined;
        if (
          saved.id == null ||
          persistedSnapshot?.workspaceTag !== draftWorkspaceTag ||
          String(persistedSnapshot.currentNote?.id) !== String(saved.id) ||
          persistedSnapshot.currentNote?.pendingNoteWriteKey !== pending.entry.key
        )
          throw new Error(
            "Could not retain the accepted note identity in this Workspace. Retry the same saved operation.",
          );
      }
      const binding = draftBindingRef.current;
      await checkpointQuickNotesOfflineDraft(
        authorityScope,
        pending.entry.key,
        pending.entry.pendingWrite!.key,
        () => (binding ? draftSnapshot(binding) : null),
        saved,
      );
      if (!isCurrent()) return;
      pendingSaveRef.current = null;
      const acknowledgedDraft = useWorkspaceStore.getState().currentNote;
      if (!acknowledgedDraft.isDirty) showSavedIndicatorTemporarily();
      messageApi.success(
        draft.id
          ? t("playground:studio.noteUpdated", "Note updated")
          : t("playground:studio.noteSaved", "Note saved")
      )
      await loadWorkspaceNotes({ request, isCurrent })
    } catch (error: any) {
      if (!isCurrent()) return
      if (isDefinitiveWriteRejection(error) && attemptedOperation) {
        const pending = attemptedOperation;
        try {
          await retireSurfaceOfflineDraft(
            pending.authorityScope,
            pending.entry.key,
            pending.entry.pendingWrite!.key,
          );
          rejectedOperationCleared = true;
          if (!isCurrent()) return;
          pendingSaveRef.current = null;
        } catch {
          messageApi.error(
            "Could not retire the rejected note operation on this device. Retry before changing the draft.",
          );
          return;
        }
      }
      if (recovery) {
        messageApi.error(
          isNotesProvenancePolicyUnavailable(error)
            ? t(
                "playground:studio.sourceHistoryUnavailable",
                NOTES_PROVENANCE_UNAVAILABLE_MESSAGE,
              )
            : rejectedOperationCleared
              ? "The previous save was rejected and its retained operation cleared. Save your current draft when ready."
              : "The previous save could not be confirmed. Retry the same previous save.",
        );
        return;
      }
      // A policy-blocked receipt remains uncertain and must retain its key/body.
      if (isNotesProvenancePolicyUnavailable(error)) {
        messageApi.error(t("playground:studio.sourceHistoryUnavailable", NOTES_PROVENANCE_UNAVAILABLE_MESSAGE))
      } else if (isDefinitiveWriteRejection(error) && (error?.message?.includes("version") || error?.status === 409)) {
        pendingSaveRef.current = null
        if (draft.id && scope && !String(error?.message).includes("encryption_unsupported")) {
          try {
            const remote = await bgRequest<NoteListItem>({ ...requestScopeFields(scope.requestScope), abortSignal: scope.scopeSignal, path: `/api/v1/notes/${encodeURIComponent(String(draft.id))}` as AllowedPath, method: "GET" })
            if (!isCurrent()) return
            const latest = useWorkspaceStore.getState().currentNote
            setCurrentNote({
              ...latest,
              ...knowledgeNoteHead(remote),
              isDirty: true,
            });
          } catch { /* Keep the original draft and require another explicit retry. */ }
        }
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
          acknowledged
            ? "Note saved, but its retry checkpoint could not be cleared. Retry the same save."
            : t("playground:studio.noteSaveError", "Failed to save note"),
        );
      }
    } finally {
      stopWatchingNote()
      stopWatchingOwner()
      scope?.release()
      if (saveControllerRef.current === controller) {
        saveControllerRef.current = null
        setIsSaving(false)
      }
    }
  }

  const handleResumeLocalDraft = (key: string) => {
    const expected = useWorkspaceStore.getState();
    const resume = async () => {
      const current = useWorkspaceStore.getState();
      if (
        current.currentNote !== expected.currentNote ||
        current.workspaceId !== expected.workspaceId ||
        current.workspaceTag !== expected.workspaceTag
      )
        return;
      const controller = new AbortController();
      readControllerRef.current?.abort();
      readControllerRef.current = controller;
      const stop = watchChatAccountChanges((invalidated) => {
        if (invalidated) controller.abort();
      });
      const stopDraft = useWorkspaceStore.subscribe((state) => {
        if (
          state.currentNote !== expected.currentNote ||
          state.workspaceId !== expected.workspaceId ||
          state.workspaceTag !== expected.workspaceTag
        )
          controller.abort();
      });
      let scope: ServicePromptSnapshot | undefined;
      try {
        scope = await loadServicePromptSnapshot([], {
          signal: controller.signal,
        });
        const user =
          scope.requestScope.userId == null
            ? await tldwAuth.getCurrentUser()
            : null;
        const ownerId =
          scope.requestScope.userId ?? (user?.is_active ? user.id : null);
        if (ownerId == null)
          throw new Error("Verify your account before resuming a note.");
        const authorityScope = createNotesGraphAuthorityScope(
          scope.requestScope.config.serverUrl,
          ownerId,
        );
        const authorityId = servicePromptAuthorityKey(scope.requestScope);
        const entry = (await readSurfaceOfflineDraftQueue(authorityScope))[key];
        if (
          controller.signal.aborted ||
          scope.scopeSignal.aborted ||
          scope.scopeInvalidatedSignal.aborted
        )
          return;
        if (
          !isQuickNotesRetainedDraft(entry) ||
          !entry.metadata?.quickNotesDirty ||
          entry.metadata.quickNotesAuthorityId !== authorityId ||
          entry.metadata.quickNotesWorkspaceId !== expected.workspaceId ||
          entry.metadata.quickNotesWorkspaceTag !== expected.workspaceTag
        )
          throw new Error(
            "The retained local draft no longer matches this Workspace and service.",
          );
        if (draftBindingRef.current) draftBindingRef.current.active = false;
        draftBindingRef.current = null;
        stopDraft();
        const resumed = {
          ...knowledgeNoteHead(entry.metadata),
          id: entry.noteId!,
          title: entry.title,
          content: entry.content,
          keywords: [...entry.keywords],
          version: entry.baseVersion!,
          isDirty: true,
          pendingNoteWriteKey: key,
          pendingKnowledgeProvenance: entry.metadata.pendingKnowledgeProvenance,
        };
        setCurrentNote(resumed);
        setKeywordsInput(resumed.keywords.join(", "));
        draftBindingRef.current = {
          authorityScope,
          authorityId,
          key,
          pendingKey:
            entry.metadata.quickNotesAcceptedKey ||
            entry.metadata.quickNotesRejectedKey,
          workspaceId: expected.workspaceId,
          workspaceTag: expected.workspaceTag,
          noteId: resumed.id,
          active: true,
          latest: resumed,
        };
        setRecoverableDrafts([]);
      } catch {
        if (!controller.signal.aborted)
          messageApi.error(
            "Could not resume the retained local draft. Restore its Workspace and service and retry.",
          );
      } finally {
        stop();
        stopDraft();
        scope?.release();
      }
    };
    if (expected.currentNote.isDirty)
      Modal.confirm({
        title: t("playground:studio.unsavedChanges", "Unsaved Changes"),
        content:
          "Replace the current unsaved draft with the retained local draft?",
        onOk: resume,
      });
    else void resume();
  };

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
    const clearContent = t(
      "playground:studio.noteCleared",
      "Note cleared."
    )
    const messageConfig = {
      key: undoMessageKey,
      type: "warning",
      duration: WORKSPACE_UNDO_WINDOW_MS / 1000,
      content: renderWorkspaceMessageActionContent(
        clearContent,
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
        maybeWarning(clearContent)
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
    if (currentNote.content.trim()) sections.push(retainKnowledgeNoteProvenance(currentNote.content, currentNote))
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
            aria-label={t(
              "playground:studio.clearCurrentNote",
              "Clear current note"
            )}
            title={t(
              "playground:studio.clearCurrentNote",
              "Clear current note"
            )}
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

      {resolveKnowledgeNoteProvenance(currentNote, currentNote.content).state === "deleted" ? (
        <p role="status" className="mb-2 text-xs text-text-muted">{t("playground:studio.sourceHistoryRemoved", "Source history was removed. Ordinary edits keep it removed.")}</p>
      ) : resolveKnowledgeNoteProvenance(currentNote, currentNote.content).reconciliation ? (
        <p role="status" className="mb-2 text-xs text-text-muted">{t("playground:studio.sourceHistoryReconciled", "Retained source history differs from the portable copy. The retained history is used.")}</p>
      ) : resolveKnowledgeNoteProvenance(currentNote, currentNote.content).provenance ? (
        <p className="mb-2 text-xs text-text-muted">{t("playground:studio.sourceHistoryRetained", "Original source history is retained with this note.")}</p>
      ) : null}

      <KnowledgeNoteHistory note={currentNote} />

      {/* Title input */}
          <Input
        ref={titleInputRef}
        value={currentNote.title}
        onChange={(e) => {
          hideSavedIndicator()
          updateNoteTitle(e.target.value)
        }}
        aria-label={t("playground:studio.noteTitleLabel", "Note title")}
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
          aria-label={t("playground:studio.noteKeywordsLabel", "Note keywords")}
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
            value={stripKnowledgeNoteProvenance(currentNote.content)}
            onChange={(e) => {
              hideSavedIndicator()
              updateNoteContent(
                retainKnowledgeNoteProvenance(
                  e.target.value,
                  currentNote
                )
              )
            }}
            aria-label={t("playground:studio.noteContentLabel", "Note content")}
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
              <MarkdownPreview
                content={stripKnowledgeNoteProvenance(currentNote.content)}
                size="sm"
              />
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

      {recoverableDrafts.map((entry) => (
        <Button
          key={entry.key}
          size="small"
          disabled={isSaving}
          onClick={() => {
            if (entry.pendingWrite)
              void handleSave({
                key: entry.key,
                workspaceId: activeWorkspaceId,
                workspaceTag,
              });
            else handleResumeLocalDraft(entry.key);
          }}
        >
          {entry.pendingWrite
            ? "Retry previous save"
            : "Resume local unsaved draft"}
        </Button>
      ))}

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
            onClick={() => void handleSave()}
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
          aria-label={t("playground:studio.searchNotesLabel", "Search notes")}
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
                        {stripKnowledgeNoteProvenance(note.content || "").slice(0, 100) ||
                          "No content"}
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
