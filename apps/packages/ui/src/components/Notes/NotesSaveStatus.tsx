import React from 'react'
import { Tooltip } from 'antd'
import { useTranslation } from 'react-i18next'
import type { SaveIndicatorState } from './notes-manager-types'

interface NotesSaveStatusProps {
  state: SaveIndicatorState
  lastSavedAt?: string | null
  onRetry?: () => void
  /** Version, last save and where the note is stored (shown on hover/focus). */
  detail?: string | null
}

type Translate = (key: string, opts?: Record<string, unknown>) => string

/**
 * The editor's one save status (NS-06). The pill is visible but not live;
 * a separate polite region announces only outcomes (saved, problems), never
 * "Unsaved changes" or "Saving..." on every keystroke.
 */
const NotesSaveStatus: React.FC<NotesSaveStatusProps> = ({ state, lastSavedAt, onRetry, detail }) => {
  const { t } = useTranslation(['option'])
  const config = statusConfig(state, lastSavedAt, t)
  const announcement = ANNOUNCED_STATES.has(state) && config ? config.text : ''
  const showRetry = (state === 'error' || state === 'retrying') && Boolean(onRetry)

  return (
    <>
      <span
        className={`inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap text-[11px] ${config?.textClass ?? ''}`}
        data-testid="notes-save-status"
        data-state={state}
      >
        {config ? (
          <Tooltip title={detail || undefined}>
            <span className="inline-flex items-center gap-1.5" tabIndex={detail ? 0 : undefined}>
              <span className={`inline-block w-1.5 h-1.5 rounded-full ${config.dotClass}`} aria-hidden="true" />
              <span>{config.text}</span>
            </span>
          </Tooltip>
        ) : null}
        {detail ? (
          <span className="sr-only" data-testid="notes-editor-revision-meta">
            {detail}
          </span>
        ) : null}
        {showRetry && (
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
      <span
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        data-testid="notes-save-status-announcement"
      >
        {announcement}
      </span>
    </>
  )
}

const ANNOUNCED_STATES = new Set<SaveIndicatorState>(['saved', 'error', 'retrying', 'conflict', 'offline'])

// Text colours are theme tokens so they follow the palette and stay at
// 4.5:1 on every surface (AX-07); the dot carries the status hue.
function statusConfig(
  state: SaveIndicatorState,
  lastSavedAt: string | null | undefined,
  t: Translate
): { dotClass: string; text: string; textClass: string } | null {
  switch (state) {
    case 'idle':
      return null
    case 'dirty':
      return {
        dotClass: 'bg-amber-400',
        text: t('option:notesSearch.saveStatusDirty', { defaultValue: 'Unsaved changes' }),
        textClass: 'text-warn'
      }
    case 'saving':
      return {
        dotClass: 'bg-blue-400 animate-pulse',
        text: t('option:notesSearch.saving', { defaultValue: 'Saving...' }),
        textClass: 'text-primaryStrong dark:text-primary'
      }
    case 'saved':
      return {
        dotClass: 'bg-emerald-500',
        text: formatSavedText(lastSavedAt, t),
        textClass: 'text-success'
      }
    case 'offline':
      return {
        dotClass: 'bg-sky-500',
        text: t('option:notesSearch.saveStatusOffline', { defaultValue: 'Saved on this device' }),
        textClass: 'text-primaryStrong dark:text-primary'
      }
    case 'retrying':
      return {
        dotClass: 'bg-amber-500 animate-pulse',
        text: t('option:notesSearch.saveStatusRetrying', { defaultValue: 'Not saved, retrying' }),
        textClass: 'text-warn'
      }
    case 'conflict':
      return {
        dotClass: 'bg-amber-500',
        text: t('option:notesSearch.saveStatusConflict', { defaultValue: 'Changed elsewhere' }),
        textClass: 'text-warn'
      }
    case 'error':
      return {
        dotClass: 'bg-red-500',
        text: t('option:notesSearch.saveStatusError', { defaultValue: 'Not saved, needs attention' }),
        textClass: 'text-danger'
      }
  }
}

function formatSavedText(lastSavedAt: string | null | undefined, t: Translate): string {
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
