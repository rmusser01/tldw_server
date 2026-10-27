type Translate = (key: string, options?: Record<string, any>) => string

/** The message key of a note's "Note deleted · Undo" toast. */
export const notesDeleteToastKey = (noteId: string | number): string =>
  `notes-delete-${String(noteId)}`

/**
 * Close a note's "Note deleted · Undo" toast (NL-15). Once the note is back
 * from Trash the toast is stale, and it used to sit above "Note restored".
 */
export const dismissNoteDeletedToast = (
  message: { destroy?: (key?: string) => void } | null | undefined,
  noteId: string | number
): void => {
  if (typeof message?.destroy === 'function') message.destroy(notesDeleteToastKey(noteId))
}

/** The title to name in the Move to Trash confirm: the open note's, else the list row's. */
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
 * Copy for the delete confirm (NL-15). Deleting moves a note to Trash, so the
 * confirm says that, names the note, and says it can be restored.
 */
export const buildMoveToTrashConfirm = (
  t: Translate,
  { noteTitle, inboundReferenceCount }: { noteTitle: string; inboundReferenceCount: number }
) => {
  const title = noteTitle.trim()
  let content = title
    ? t('option:notesSearch.moveToTrashConfirmNamed', {
        defaultValue: 'Move "{{title}}" to Trash? You can restore it from Trash.',
        title
      }).replace('{{title}}', () => title)
    : t('option:notesSearch.moveToTrashConfirmUnnamed', {
        defaultValue: 'Move this note to Trash? You can restore it from Trash.'
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
    content = `${content}\n\n${warning}`
  }
  return {
    title: t('option:notesSearch.moveToTrashConfirmTitle', { defaultValue: 'Move to Trash?' }),
    content,
    okText: t('option:notesSearch.moveToTrashConfirmOk', { defaultValue: 'Move to Trash' }),
    cancelText: t('common:cancel', { defaultValue: 'Cancel' })
  }
}
