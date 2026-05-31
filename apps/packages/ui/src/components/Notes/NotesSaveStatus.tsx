import React from 'react'
import { useTranslation } from 'react-i18next'
import type { SaveIndicatorState } from './notes-manager-types'

interface NotesSaveStatusProps {
  state: SaveIndicatorState
  lastSavedAt?: string | null
  onRetry?: () => void
}

const NotesSaveStatus: React.FC<NotesSaveStatusProps> = ({ state, lastSavedAt, onRetry }) => {
  const { t } = useTranslation(['option'])

  if (state === 'idle') return null

  const config = (() => {
    switch (state) {
      case 'dirty':
        return {
          dotClass: 'bg-amber-400',
          text: t('option:notesSearch.saveStatusDirty', { defaultValue: 'Unsaved changes' }),
          textClass: 'text-amber-600 dark:text-amber-400'
        }
      case 'saving':
        return {
          dotClass: 'bg-blue-400 animate-pulse',
          text: t('option:notesSearch.saving', { defaultValue: 'Saving...' }),
          textClass: 'text-blue-600 dark:text-blue-400'
        }
      case 'saved':
        return {
          dotClass: 'bg-emerald-500',
          text: formatSavedText(lastSavedAt, t),
          textClass: 'text-emerald-600 dark:text-emerald-400'
        }
      case 'error':
        return {
          dotClass: 'bg-red-500',
          text: t('option:notesSearch.saveStatusError', { defaultValue: 'Save failed' }),
          textClass: 'text-red-600 dark:text-red-400'
        }
    }
  })()

  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[11px] ${config.textClass}`}
      role="status"
      aria-live="polite"
      aria-atomic="true"
      data-testid="notes-save-status"
      data-state={state}
    >
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${config.dotClass}`} />
      <span>{config.text}</span>
      {state === 'error' && onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="underline hover:no-underline ml-0.5"
          data-testid="notes-save-retry"
        >
          {t('option:notesSearch.saveStatusRetry', { defaultValue: 'Retry' })}
        </button>
      )}
    </span>
  )
}

function formatSavedText(
  lastSavedAt: string | null | undefined,
  t: (key: string, opts?: Record<string, unknown>) => string
): string {
  if (!lastSavedAt) {
    return t('option:notesSearch.saveStatusSaved', { defaultValue: 'Saved' })
  }
  const elapsed = Date.now() - new Date(lastSavedAt).getTime()
  if (elapsed < 10_000) {
    return t('option:notesSearch.saveStatusSavedJustNow', { defaultValue: 'Saved just now' })
  }
  if (elapsed < 60_000) {
    return t('option:notesSearch.saveStatusSavedSecondsAgo', {
      defaultValue: 'Saved a few seconds ago'
    })
  }
  const minutes = Math.floor(elapsed / 60_000)
  if (minutes < 60) {
    return t('option:notesSearch.saveStatusSavedMinutesAgo', {
      defaultValue: `Saved ${minutes}m ago`,
      count: minutes
    })
  }
  return t('option:notesSearch.saveStatusSaved', { defaultValue: 'Saved' })
}

export default React.memo(NotesSaveStatus)
