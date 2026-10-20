import React from 'react'
import { Button } from 'antd'
import { Trash2 as TrashIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

type NotesTrashGuidanceProps = {
  isMobileViewport: boolean
  onShowTrashList: () => void
  onBackToNotes: () => void
}

/**
 * What the editor area shows in Trash (NL-15). Trashed notes are restored
 * from the list, so a blank "New note" editor here only invited mistakes.
 */
const NotesTrashGuidance: React.FC<NotesTrashGuidanceProps> = ({
  isMobileViewport,
  onShowTrashList,
  onBackToNotes
}) => {
  const { t } = useTranslation(['option'])
  return (
    <section
      className="flex min-w-0 flex-1 flex-col items-center justify-center gap-3 rounded-lg border border-border bg-surface p-6 text-center"
      aria-labelledby="notes-trash-guidance-title"
      data-testid="notes-trash-guidance"
    >
      <TrashIcon className="h-8 w-8 text-text-subtle" aria-hidden="true" />
      <h2 id="notes-trash-guidance-title" className="text-lg font-medium text-text">
        {t('option:notesSearch.trashGuidanceTitle', { defaultValue: 'Trash' })}
      </h2>
      <p className="max-w-sm text-sm text-text-muted">
        {t('option:notesSearch.trashGuidanceDescription', {
          defaultValue:
            'Deleted notes wait here. Choose Restore next to a note to put it back with your notes.'
        })}
      </p>
      <div className="flex flex-wrap items-center justify-center gap-2">
        {isMobileViewport && (
          <Button onClick={onShowTrashList} data-testid="notes-trash-guidance-show-list">
            {t('option:notesSearch.trashGuidanceShowList', { defaultValue: 'Show Trash' })}
          </Button>
        )}
        <Button type="primary" onClick={onBackToNotes} data-testid="notes-trash-guidance-back">
          {t('option:notesSearch.trashGuidanceBack', { defaultValue: 'Back to notes' })}
        </Button>
      </div>
    </section>
  )
}

export default NotesTrashGuidance
