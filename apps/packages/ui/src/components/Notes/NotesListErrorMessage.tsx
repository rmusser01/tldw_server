import React from 'react'
import { useTranslation } from 'react-i18next'
import { describeNotesListError } from './notes-list-error'

type NotesListErrorMessageProps = {
  message?: string | null
  fallback: string
}

/**
 * The body of a notes list load error (NL-16): plain words first, with the
 * raw message and request path tucked behind a "Details" disclosure.
 */
const NotesListErrorMessage: React.FC<NotesListErrorMessageProps> = ({ message, fallback }) => {
  const { t } = useTranslation(['option'])
  const { summary, details, unreachable } = describeNotesListError(message)
  const text = unreachable
    ? t('option:notesSearch.loadErrorUnreachable', {
        defaultValue: 'Could not reach the notes server.'
      })
    : summary || fallback
  const showDetails = details != null || (unreachable && summary.length > 0)

  return (
    <>
      <span>{text}</span>
      {showDetails && (
        <details className="mt-2 text-left text-xs text-text-muted" data-testid="notes-list-error-details">
          <summary className="cursor-pointer select-none">
            {t('option:notesSearch.loadErrorDetails', { defaultValue: 'Details' })}
          </summary>
          <code className="mt-1 block break-all font-mono text-[11px]">{details ?? summary}</code>
        </details>
      )}
    </>
  )
}

export default NotesListErrorMessage
