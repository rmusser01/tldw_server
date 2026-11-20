type Translate = (key: string, options?: Record<string, any>) => string

/** The title to name in the Moved to Trash offer: the open note's, else the list row's. */
export const findNoteTitleForConfirm = (
  noteId: string,
  sources: {
    selectedId?: string | number | null
    selectedTitle?: string | null
    rows?: ReadonlyArray<{ id: string | number; title?: string | null }> | null
  }
): string => {
  if (sources.selectedId != null && String(sources.selectedId) === noteId) {
    const title = String(sources.selectedTitle || '').trim()
    if (title) return title
  }
  const row = sources.rows?.find((entry) => String(entry.id) === noteId)
  return String(row?.title || '').trim()
}

/**
 * The inline Undo offer's text after one note moves to Trash (NL-15, XP-17).
 * Deleting no longer asks first (#3101, decision 8), so the offer names the
 * note and, when other notes link to it, says those links stop working
 * until it is restored.
 */
export const buildMovedToTrashMessage = (
  t: Translate,
  { noteTitle, inboundReferenceCount }: { noteTitle: string; inboundReferenceCount: number }
): string => {
  const title = noteTitle.trim()
  let content = title
    ? t('option:notesSearch.movedToTrashNamed', {
        defaultValue: 'Moved "{{title}}" to Trash.',
        title
      }).replace('{{title}}', () => title)
    : t('option:notesSearch.movedToTrashUnnamed', {
        defaultValue: 'Moved this note to Trash.'
      })
  if (inboundReferenceCount > 0) {
    const count = String(inboundReferenceCount)
    const warning =
      inboundReferenceCount === 1
        ? t('option:notesSearch.moveToTrashLinkedOne', {
            defaultValue:
              'This note is linked from 1 other note. That link will not work while the note is in Trash.'
          })
        : t('option:notesSearch.moveToTrashLinkedMany', {
            defaultValue:
              'This note is linked from {{count}} other notes. Those links will not work while the note is in Trash.',
            count: inboundReferenceCount
          }).replace('{{count}}', count)
    content = `${content} ${warning}`
  }
  return content
}
