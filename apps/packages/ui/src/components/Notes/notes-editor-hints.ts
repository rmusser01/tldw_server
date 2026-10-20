import type { TFunction } from 'i18next'
import type { NotesInputMode } from './notes-manager-types'

/**
 * The one-line hint under the note body.
 *
 * With no note open and nothing typed yet, the editor is already a usable
 * draft, so the hint says how to start instead of showing a separate empty
 * state on top of the editor (NO-04). The hint sits below the body, so
 * swapping it on the first keystroke never moves the text area.
 */
export const notesEditorSupportHint = (
  t: TFunction | ((key: string, options?: Record<string, unknown>) => string),
  inputMode: NotesInputMode,
  isPristineDraft: boolean
): string => {
  if (isPristineDraft) {
    return t('option:notesSearch.editorDraftHint', {
      defaultValue:
        'Start typing to create a note. Ctrl/Cmd+S saves it to your server. Type [[ to link another note.'
    })
  }
  return inputMode === 'wysiwyg'
    ? t('option:notesSearch.wysiwygSupportHint', {
        defaultValue: 'WYSIWYG mode keeps markdown structure while you edit.'
      })
    : t('option:notesSearch.editorSupportHint', {
        defaultValue: 'Markdown + LaTeX supported'
      })
}
