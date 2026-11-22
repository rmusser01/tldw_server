import type { NotesSortOption } from './notes-manager-utils'

type Translate = (key: string, options?: Record<string, unknown>) => string

/**
 * The notes sort menu. "Best match" (`relevance`) leads it while a text search
 * is active and is absent otherwise: the plain list has no relevance (NL-07).
 */
export const buildNotesSortOptions = (
  t: Translate,
  { bestMatchAvailable }: { bestMatchAvailable: boolean }
): Array<{ value: NotesSortOption; label: string }> => [
  ...(bestMatchAvailable
    ? [
        {
          value: 'relevance' as const,
          label: t('option:notesSearch.sortRelevance', { defaultValue: 'Best match' })
        }
      ]
    : []),
  {
    value: 'modified_desc',
    label: t('option:notesSearch.sortModifiedDesc', {
      defaultValue: 'Date modified (newest first)'
    })
  },
  {
    value: 'created_desc',
    label: t('option:notesSearch.sortCreatedDesc', {
      defaultValue: 'Date created (newest first)'
    })
  },
  {
    value: 'title_asc',
    label: t('option:notesSearch.sortTitleAsc', { defaultValue: 'Title (A-Z)' })
  },
  {
    value: 'title_desc',
    label: t('option:notesSearch.sortTitleDesc', { defaultValue: 'Title (Z-A)' })
  }
]
