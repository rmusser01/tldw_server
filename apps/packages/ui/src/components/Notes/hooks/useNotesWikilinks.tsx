import React from 'react'
import { useQuery } from '@tanstack/react-query'
import { bgRequest } from '@/services/background-proxy'
import { useCanonicalConnectionConfig } from '@/hooks/useCanonicalConnectionConfig'
import { tldwAuth } from '@/services/tldw/TldwAuth'
import { requestScopeFields } from '@/services/tldw/domains/service-prompts'
import { createServicePromptScopeChangedError } from '@/services/tldw/service-prompt-scope-error'
import { connectionAuthoritiesMatch, deriveConnectionAuthorityId, deriveSingleUserApiKeyCredentialScope } from '@/services/chat-surface-scope'
import { createNotesGraphAuthorityScope } from './useNotesGraphAuthorityScope'
import { useDebounce } from '@/hooks/useDebounce'
import type { NoteListItem } from '@/components/Notes/notes-manager-types'
import type { ActiveWikilinkQuery, WikilinkCandidate, WikilinkResolutions } from '@/components/Notes/wikilinks'
import {
  buildWikilinkIndex,
  collectWikilinkTargets,
  getActiveWikilinkQuery,
  insertWikilinkAtCursor,
  parseWikilinkResolutions,
  renderContentWithResolvedWikilinks,
} from '@/components/Notes/wikilinks'
import {
  normalizeGraphNoteId,
  extractMarkdownHeadings,
  LARGE_NOTE_PREVIEW_THRESHOLD,
  LARGE_NOTE_PREVIEW_DELAY_MS,
} from '../notes-manager-utils'

const WIKILINK_SUGGESTION_LIMIT = 8
const WIKILINK_SERVER_SUGGESTION_LIMIT = 20
const WIKILINK_SUGGEST_DEBOUNCE_MS = 200
const WIKILINK_RESOLVE_DEBOUNCE_MS = 250
const NO_WIKILINK_TARGETS_KEY = JSON.stringify([[], []])

const toWikilinkCandidates = (payload: unknown): WikilinkCandidate[] => {
  const body = payload as { items?: unknown; notes?: unknown; results?: unknown } | null
  const rows = Array.isArray(payload)
    ? payload
    : Array.isArray(body?.items)
      ? body.items
      : Array.isArray(body?.notes)
        ? body.notes
        : Array.isArray(body?.results)
          ? body.results
          : []
  const candidates: WikilinkCandidate[] = []
  for (const row of rows as Array<{ id?: unknown; title?: unknown } | null>) {
    const id = normalizeGraphNoteId(row?.id as string | number | null | undefined)
    const title = String(row?.title || '').trim()
    if (id && title) candidates.push({ id, title })
  }
  return candidates
}

export interface UseNotesWikilinksDeps {
  /** Server lookups run only while online and for a verified notes owner. */
  isOnline: boolean
  authorityScope: string | null | undefined
  /** Tooltip for an unresolved link, e.g. `Create note "Title"`. */
  createNoteLabel?: (title: string) => string
  /** From editor hook */
  selectedId: string | number | null
  title: string
  content: string
  editorCursorIndex: number | null
  setEditorCursorIndex: React.Dispatch<React.SetStateAction<number | null>>
  editorDisabled: boolean
  editorMode: string
  contentTextareaRef: React.MutableRefObject<HTMLTextAreaElement | null>
  resizeEditorTextarea: () => void
  setContentDirty: (
    nextContent: string,
    options?: { provenance?: 'manual' | string }
  ) => void
  /** From list hook */
  data: NoteListItem[] | undefined
  /** From note relations (computed in main component) */
  noteRelations: {
    related: Array<{ id: string; title: string; available?: boolean; unavailableReason?: string | null }>
    backlinks: Array<{ id: string; title: string; available?: boolean; unavailableReason?: string | null }>
    manualLinks: Array<{
      noteId: string
      title: string
      edgeId: string
      directed: boolean
      outgoing: boolean
      available?: boolean
      unavailableReason?: string | null
    }>
  }
}

export function useNotesWikilinks(deps: UseNotesWikilinksDeps) {
  const {
    isOnline,
    authorityScope,
    createNoteLabel,
    selectedId,
    title,
    content,
    editorCursorIndex,
    setEditorCursorIndex,
    editorDisabled,
    editorMode,
    contentTextareaRef,
    resizeEditorTextarea,
    setContentDirty,
    data,
    noteRelations,
  } = deps

  const [wikilinkSelectionIndex, setWikilinkSelectionIndex] = React.useState(0)
  const [largePreviewReady, setLargePreviewReady] = React.useState(true)
  const { config, loading, authorityLoading } = useCanonicalConnectionConfig()
  const capturedConfig = config ? { ...config } : null
  const configReady = !(authorityLoading ?? loading) && capturedConfig !== null
  const [, setAuthorityRevision] = React.useState(0)
  const authorityEpochRef = React.useRef(0)
  const authorityRef = React.useRef({ authorityScope, config: capturedConfig, configReady, isOnline })
  const previousAuthority = authorityRef.current
  if (
    previousAuthority.authorityScope !== authorityScope ||
    !connectionAuthoritiesMatch(capturedConfig, previousAuthority.config) ||
    previousAuthority.configReady !== configReady || previousAuthority.isOnline !== isOnline
  ) authorityEpochRef.current += 1
  authorityRef.current = { authorityScope, config: capturedConfig, configReady, isOnline }
  const authorityEpoch = authorityEpochRef.current
  const connectionAuthority = deriveConnectionAuthorityId(capturedConfig)

  // Invalidate immediately, including an A-B-A change between React renders.
  React.useEffect(() => {
    const invalidate = () => {
      authorityEpochRef.current += 1
      setAuthorityRevision(revision => revision + 1)
    }
    const configChanged = (event: Event) => {
      const detail = (event as CustomEvent<{ authorityChanged?: boolean; refreshSessionInvalidated?: boolean }>).detail
      if (detail?.authorityChanged !== false || detail.refreshSessionInvalidated) invalidate()
    }
    window.addEventListener('tldw:config-updated', configChanged)
    window.addEventListener('tldw:auth-principal-changed', invalidate)
    return () => {
      authorityEpochRef.current += 1
      window.removeEventListener('tldw:config-updated', configChanged)
      window.removeEventListener('tldw:auth-principal-changed', invalidate)
    }
  }, [])

  const assertOwnerCurrent = (
    user: { id?: string | number | null; is_active?: boolean } | null | undefined,
    signal: AbortSignal
  ) => {
    if (
      signal.aborted || authorityEpochRef.current !== authorityEpoch ||
      !configReady || !isOnline || !capturedConfig || !authorityScope ||
      user?.is_active !== true || user.id == null ||
      createNotesGraphAuthorityScope(capturedConfig.serverUrl, user.id) !== authorityScope
    ) throw createServicePromptScopeChangedError()
  }

  const wikilinkCandidates = React.useMemo(() => {
    const seen = new Set<string>()
    const candidates: WikilinkCandidate[] = []

    const append = (id: string | number, candidateTitle: string) => {
      const normalizedId = normalizeGraphNoteId(id)
      const normalizedTitle = String(candidateTitle || '').trim()
      if (!normalizedId || !normalizedTitle) return
      const dedupeKey = `${normalizedId}::${normalizedTitle.toLowerCase()}`
      if (seen.has(dedupeKey)) return
      seen.add(dedupeKey)
      candidates.push({ id: normalizedId, title: normalizedTitle })
    }

    if (selectedId != null) {
      append(selectedId, title || `Note ${selectedId}`)
    }

    if (Array.isArray(data)) {
      for (const note of data) {
        append(String(note.id), String(note.title || `Note ${note.id}`))
      }
    }

    for (const note of noteRelations.related) {
      if (note.unavailableReason) continue
      append(note.id, note.title)
    }
    for (const note of noteRelations.backlinks) {
      if (note.unavailableReason) continue
      append(note.id, note.title)
    }
    for (const link of noteRelations.manualLinks) {
      if (link.unavailableReason) continue
      append(link.noteId, link.title)
    }

    return candidates.sort((a, b) => a.title.localeCompare(b.title) || a.id.localeCompare(b.id))
  }, [data, noteRelations.backlinks, noteRelations.manualLinks, noteRelations.related, selectedId, title])

  const wikilinkIndex = React.useMemo(
    () => buildWikilinkIndex(wikilinkCandidates),
    [wikilinkCandidates]
  )

  const activeWikilinkQuery = React.useMemo<ActiveWikilinkQuery | null>(() => {
    if (editorDisabled) return null
    if (editorCursorIndex == null) return null
    return getActiveWikilinkQuery(content, editorCursorIndex)
  }, [content, editorCursorIndex, editorDisabled])

  const serverLookupsEnabled = isOnline && authorityScope != null && configReady

  // Autocomplete searches titles across the whole library, not just the
  // loaded list page (NE-02).
  const wikilinkQueryText = activeWikilinkQuery?.query.trim() ?? ''
  const debouncedWikilinkQueryText = useDebounce(wikilinkQueryText, WIKILINK_SUGGEST_DEBOUNCE_MS)
  const { data: serverWikilinkCandidates } = useQuery<WikilinkCandidate[]>({
    queryKey: ['notes-wikilink-suggest', authorityScope, debouncedWikilinkQueryText, connectionAuthority, authorityEpoch],
    enabled: serverLookupsEnabled && !editorDisabled && debouncedWikilinkQueryText.length > 0,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 30_000,
    // Keep the previous owner's titles out of a new owner's suggestions.
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === authorityScope &&
      previousQuery.queryKey.at(-2) === connectionAuthority &&
      previousQuery.queryKey.at(-1) === authorityEpoch ? previous : undefined,
    queryFn: async ({ signal }) => {
      const user = await tldwAuth.getCurrentUser()
      assertOwnerCurrent(user, signal)
      const params = new URLSearchParams({
        query: debouncedWikilinkQueryText,
        title_only: 'true',
        limit: String(WIKILINK_SERVER_SUGGESTION_LIMIT)
      })
      const response = await bgRequest<unknown>({
        path: `/api/v1/notes/search/?${params.toString()}`,
        method: 'GET',
        abortSignal: signal,
        ...requestScopeFields({
          config: {
            serverUrl: capturedConfig!.serverUrl,
            authMode: capturedConfig!.authMode,
            authSource: capturedConfig!.authSource,
            orgId: capturedConfig!.orgId,
            expectedSingleUserApiKeyScope: capturedConfig!.authMode === 'single-user'
              ? deriveSingleUserApiKeyCredentialScope('single-user', capturedConfig!.apiKey)
              : undefined
          },
          userId: user.id
        })
      })
      assertOwnerCurrent(await tldwAuth.getCurrentUser(), signal)
      return toWikilinkCandidates(response)
    }
  })

  const wikilinkSuggestions = React.useMemo(() => {
    if (!activeWikilinkQuery) return [] as WikilinkCandidate[]
    const queryLower = activeWikilinkQuery.query.trim().toLowerCase()
    const selectedNormalized = normalizeGraphNoteId(selectedId)
    const seenIds = new Set<string>()
    const pool = queryLower
      ? [...wikilinkCandidates, ...(serverLookupsEnabled ? serverWikilinkCandidates ?? [] : [])]
      : wikilinkCandidates
    const filtered = pool.filter((candidate) => {
      if (!candidate.title) return false
      if (selectedNormalized && candidate.id === selectedNormalized) return false
      if (seenIds.has(candidate.id)) return false
      if (queryLower && !candidate.title.toLowerCase().includes(queryLower)) return false
      seenIds.add(candidate.id)
      return true
    })
    return filtered
      .sort((a, b) => {
        const aTitle = a.title.toLowerCase()
        const bTitle = b.title.toLowerCase()
        const aStarts = queryLower.length > 0 && aTitle.startsWith(queryLower)
        const bStarts = queryLower.length > 0 && bTitle.startsWith(queryLower)
        if (aStarts !== bStarts) return aStarts ? -1 : 1
        return a.title.localeCompare(b.title) || a.id.localeCompare(b.id)
      })
      .slice(0, WIKILINK_SUGGESTION_LIMIT)
  }, [activeWikilinkQuery, selectedId, serverLookupsEnabled, serverWikilinkCandidates, wikilinkCandidates])

  const wikilinkSuggestionDisplayCounts = React.useMemo(() => {
    const counts = new Map<string, number>()
    for (const candidate of wikilinkSuggestions) {
      const key = candidate.title.toLowerCase()
      counts.set(key, (counts.get(key) || 0) + 1)
    }
    return counts
  }, [wikilinkSuggestions])

  // The server resolves each link the same way it builds backlinks, so a link
  // opens the note its backlink points to, even beyond the loaded list page.
  const wikilinkTargetsKey = React.useMemo(() => {
    const targets = collectWikilinkTargets(content)
    return JSON.stringify([targets.titles, targets.ids])
  }, [content])
  const debouncedWikilinkTargetsKey = useDebounce(wikilinkTargetsKey, WIKILINK_RESOLVE_DEBOUNCE_MS)
  const sourceNoteId = normalizeGraphNoteId(selectedId)
  const { data: serverWikilinkResolutions } = useQuery<WikilinkResolutions | null>({
    queryKey: ['notes-wikilink-resolve', authorityScope, sourceNoteId, debouncedWikilinkTargetsKey, connectionAuthority, authorityEpoch],
    enabled: serverLookupsEnabled && debouncedWikilinkTargetsKey !== NO_WIKILINK_TARGETS_KEY,
    retry: false,
    refetchOnWindowFocus: false,
    // Keep earlier answers while a newly typed link resolves, but never
    // another owner's.
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === authorityScope &&
      previousQuery.queryKey.at(-2) === connectionAuthority &&
      previousQuery.queryKey.at(-1) === authorityEpoch ? previous : undefined,
    queryFn: async ({ signal }) => {
      const user = await tldwAuth.getCurrentUser()
      assertOwnerCurrent(user, signal)
      const [titles, ids] = JSON.parse(debouncedWikilinkTargetsKey) as [string[], string[]]
      const response = await bgRequest<unknown>({
        path: '/api/v1/notes/wikilinks/resolve',
        method: 'POST',
        abortSignal: signal,
        ...requestScopeFields({
          config: {
            serverUrl: capturedConfig!.serverUrl,
            authMode: capturedConfig!.authMode,
            authSource: capturedConfig!.authSource,
            orgId: capturedConfig!.orgId,
            expectedSingleUserApiKeyScope: capturedConfig!.authMode === 'single-user'
              ? deriveSingleUserApiKeyCredentialScope('single-user', capturedConfig!.apiKey)
              : undefined
          },
          userId: user.id
        }),
        body: {
          titles,
          ids,
          ...(sourceNoteId ? { source_note_id: sourceNoteId } : {})
        }
      })
      assertOwnerCurrent(await tldwAuth.getCurrentUser(), signal)
      return parseWikilinkResolutions(response)
    }
  })
  const wikilinkResolutions = serverLookupsEnabled ? serverWikilinkResolutions ?? null : null

  const previewContent = React.useMemo(
    () =>
      renderContentWithResolvedWikilinks(content, wikilinkIndex, {
        resolutions: wikilinkResolutions,
        labels: { createNote: createNoteLabel }
      }),
    [content, createNoteLabel, wikilinkIndex, wikilinkResolutions]
  )
  const tocEntries = React.useMemo(() => extractMarkdownHeadings(content), [content])
  const shouldShowToc = tocEntries.length >= 3

  const usesLargePreviewGuardrails = React.useMemo(
    () =>
      previewContent.trim().length >= LARGE_NOTE_PREVIEW_THRESHOLD &&
      (editorMode === 'preview' || editorMode === 'split'),
    [editorMode, previewContent]
  )

  const applyWikilinkSuggestion = React.useCallback(
    (candidate: WikilinkCandidate) => {
      if (!activeWikilinkQuery) return
      // A title shared by several notes is ambiguous, so link the picked note by id.
      const titleIsAmbiguous =
        (wikilinkSuggestionDisplayCounts.get(candidate.title.toLowerCase()) || 0) > 1
      const next = insertWikilinkAtCursor(
        content,
        activeWikilinkQuery,
        candidate.title,
        titleIsAmbiguous ? candidate.id : null
      )
      setContentDirty(next.content)
      setEditorCursorIndex(next.cursor)
      setWikilinkSelectionIndex(0)
      window.requestAnimationFrame(() => {
        const textarea = contentTextareaRef.current
        if (!textarea) return
        textarea.focus()
        textarea.setSelectionRange(next.cursor, next.cursor)
        resizeEditorTextarea()
      })
    },
    [activeWikilinkQuery, content, contentTextareaRef, resizeEditorTextarea, setContentDirty, setEditorCursorIndex, wikilinkSuggestionDisplayCounts]
  )

  const handleEditorKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (!activeWikilinkQuery || wikilinkSuggestions.length === 0) return
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        setWikilinkSelectionIndex((current) => (current + 1) % wikilinkSuggestions.length)
        return
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        setWikilinkSelectionIndex((current) =>
          current === 0 ? wikilinkSuggestions.length - 1 : current - 1
        )
        return
      }
      if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault()
        const candidate =
          wikilinkSuggestions[Math.max(0, Math.min(wikilinkSelectionIndex, wikilinkSuggestions.length - 1))]
        if (!candidate) return
        applyWikilinkSuggestion(candidate)
        return
      }
      if (event.key === 'Escape') {
        event.preventDefault()
        const closeCursor = activeWikilinkQuery.start
        setEditorCursorIndex(closeCursor)
        window.requestAnimationFrame(() => {
          const textarea = contentTextareaRef.current
          if (!textarea) return
          textarea.focus()
          textarea.setSelectionRange(closeCursor, closeCursor)
        })
      }
    },
    [activeWikilinkQuery, applyWikilinkSuggestion, contentTextareaRef, setEditorCursorIndex, wikilinkSelectionIndex, wikilinkSuggestions]
  )

  // Effects
  React.useEffect(() => {
    setWikilinkSelectionIndex(0)
  }, [activeWikilinkQuery?.start, activeWikilinkQuery?.query])

  React.useEffect(() => {
    if (wikilinkSuggestions.length === 0) return
    if (wikilinkSelectionIndex < wikilinkSuggestions.length) return
    setWikilinkSelectionIndex(0)
  }, [wikilinkSelectionIndex, wikilinkSuggestions.length])

  React.useEffect(() => {
    if (!usesLargePreviewGuardrails) {
      setLargePreviewReady(true)
      return
    }
    setLargePreviewReady(false)
    if (typeof window === 'undefined') {
      setLargePreviewReady(true)
      return
    }
    const timeoutId = window.setTimeout(() => {
      setLargePreviewReady(true)
    }, LARGE_NOTE_PREVIEW_DELAY_MS)
    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [previewContent, usesLargePreviewGuardrails])

  return {
    wikilinkSelectionIndex, setWikilinkSelectionIndex,
    wikilinkCandidates,
    wikilinkIndex,
    activeWikilinkQuery,
    wikilinkSuggestions,
    wikilinkSuggestionDisplayCounts,
    previewContent,
    tocEntries,
    shouldShowToc,
    usesLargePreviewGuardrails,
    largePreviewReady,
    applyWikilinkSuggestion,
    handleEditorKeyDown,
  }
}
