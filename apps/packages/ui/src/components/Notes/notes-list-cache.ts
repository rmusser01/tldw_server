import type { QueryClient } from '@tanstack/react-query'
import type { NoteListItem } from './notes-manager-types'

/** Query-key prefix shared by every notes list query (see useNotesListManagement). */
export const NOTES_LIST_QUERY_KEY_PREFIX = ['notes'] as const

type SavedNoteResponse = {
  title?: unknown
  content?: unknown
  version?: unknown
  last_modified?: unknown
  updated_at?: unknown
  keywords?: unknown
} | null | undefined

type SavedNoteFallback = {
  title?: string
  content?: string
  keywords?: string[]
}

type PatchSavedNoteOptions = {
  /**
   * The server and account the note was saved to. When set, only lists cached
   * for that scope are patched: note ids repeat across servers, so another
   * server's cached list may hold an unrelated note with the same id.
   */
  authorityScope?: string | null
}

const AUTHORITY_KEY_SEGMENT = 'authority'

/**
 * The kind of rows a notes list query holds: active notes, Trash, or one
 * collection's notes. Reads the key built in useNotesListManagement:
 * ['notes', listMode, viewMode, moodboardId, ...].
 */
const notesListRowsKind = (queryKey: readonly unknown[]): string => {
  const [, listMode, viewMode, moodboardId] = queryKey
  if (listMode === 'trash') return 'trash'
  if (viewMode === 'moodboard') return `moodboard:${String(moodboardId)}`
  return 'active'
}

/**
 * Whether rows cached for one list query may stand in while another loads
 * (NS-04). A new page, search or sort of the same list may keep the current
 * rows on screen; active notes must never stand in for Trash (where they
 * would show Restore buttons), nor one collection's notes for another's.
 */
export const isSameNotesListRowsKind = (
  previousKey: readonly unknown[],
  nextKey: readonly unknown[]
): boolean => notesListRowsKind(previousKey) === notesListRowsKind(nextKey)

const isListForScope = (
  queryKey: readonly unknown[],
  authorityScope: string | null | undefined
): boolean => {
  if (authorityScope === undefined) return true
  const index = queryKey.indexOf(AUTHORITY_KEY_SEGMENT)
  return index >= 0 && queryKey[index + 1] === authorityScope
}

const asNonEmptyString = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim().length > 0 ? value : undefined

const toKeywordList = (value: unknown): string[] | undefined => {
  if (!Array.isArray(value)) return undefined
  const keywords = value
    .map((entry) => {
      if (typeof entry === 'string') return entry
      if (entry && typeof entry === 'object') {
        const record = entry as { keyword?: unknown; name?: unknown }
        return asNonEmptyString(record.keyword) ?? asNonEmptyString(record.name)
      }
      return undefined
    })
    .filter((entry): entry is string => Boolean(entry && entry.trim()))
  return keywords
}

/**
 * Patch the saved note's row in every cached notes list for the note's
 * server and account (NS-04).
 *
 * Saving used to await a full list refetch, which blanked the list behind a
 * spinner and reset its scroll position. The PUT response already carries the
 * new title, content, version and timestamp, so the row is updated in place
 * and the list is refreshed quietly in the background.
 *
 * Returns the number of cached lists that contained the note.
 */
export const patchSavedNoteInListCache = (
  queryClient: QueryClient,
  noteId: string | number,
  saved: SavedNoteResponse,
  fallback: SavedNoteFallback = {},
  options: PatchSavedNoteOptions = {}
): number => {
  const targetId = String(noteId)
  const title = asNonEmptyString(saved?.title) ?? asNonEmptyString(fallback.title)
  const content =
    typeof saved?.content === 'string' ? saved.content : fallback.content
  const updatedAt =
    asNonEmptyString(saved?.last_modified) ?? asNonEmptyString(saved?.updated_at)
  const version =
    typeof saved?.version === 'number' && Number.isFinite(saved.version)
      ? saved.version
      : undefined
  const keywords = toKeywordList(saved?.keywords) ?? fallback.keywords

  let patchedLists = 0
  queryClient.setQueriesData<unknown>(
    {
      queryKey: [...NOTES_LIST_QUERY_KEY_PREFIX],
      predicate: (query) => isListForScope(query.queryKey, options.authorityScope)
    },
    (current) => {
      if (!Array.isArray(current)) return current
      let changed = false
      const next = current.map((row) => {
        if (!row || typeof row !== 'object' || String((row as NoteListItem).id) !== targetId) {
          return row
        }
        changed = true
        const patched: NoteListItem = { ...(row as NoteListItem) }
        if (title !== undefined) patched.title = title
        if (content !== undefined) patched.content = content
        if (updatedAt !== undefined) patched.updated_at = updatedAt
        if (version !== undefined) patched.version = version
        if (keywords !== undefined) patched.keywords = keywords
        return patched
      })
      if (!changed) return current
      patchedLists += 1
      return next
    }
  )
  return patchedLists
}
