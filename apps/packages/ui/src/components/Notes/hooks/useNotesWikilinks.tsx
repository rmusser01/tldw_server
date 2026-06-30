import React from 'react'
import type { NoteListItem } from '@/components/Notes/notes-manager-types'
import type { ActiveWikilinkQuery, WikilinkCandidate } from '@/components/Notes/wikilinks'
import {
  buildWikilinkIndex,
  getActiveWikilinkQuery,
  insertWikilinkAtCursor,
  renderContentWithResolvedWikilinks,
} from '@/components/Notes/wikilinks'
import {
  normalizeGraphNoteId,
  extractMarkdownHeadings,
  LARGE_NOTE_PREVIEW_THRESHOLD,
  LARGE_NOTE_PREVIEW_DELAY_MS,
} from '../notes-manager-utils'

export interface UseNotesWikilinksDeps {
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

  const wikilinkSuggestions = React.useMemo(() => {
    if (!activeWikilinkQuery) return [] as WikilinkCandidate[]
    const queryLower = activeWikilinkQuery.query.trim().toLowerCase()
    const selectedNormalized = normalizeGraphNoteId(selectedId)
    const filtered = wikilinkCandidates.filter((candidate) => {
      if (!candidate.title) return false
      if (selectedNormalized && candidate.id === selectedNormalized) return false
      if (!queryLower) return true
      return candidate.title.toLowerCase().includes(queryLower)
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
      .slice(0, 8)
  }, [activeWikilinkQuery, selectedId, wikilinkCandidates])

  const wikilinkSuggestionDisplayCounts = React.useMemo(() => {
    const counts = new Map<string, number>()
    for (const candidate of wikilinkSuggestions) {
      const key = candidate.title.toLowerCase()
      counts.set(key, (counts.get(key) || 0) + 1)
    }
    return counts
  }, [wikilinkSuggestions])

  const previewContent = React.useMemo(
    () => renderContentWithResolvedWikilinks(content, wikilinkIndex),
    [content, wikilinkIndex]
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
      const next = insertWikilinkAtCursor(content, activeWikilinkQuery, candidate.title)
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
    [activeWikilinkQuery, content, contentTextareaRef, resizeEditorTextarea, setContentDirty, setEditorCursorIndex]
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
