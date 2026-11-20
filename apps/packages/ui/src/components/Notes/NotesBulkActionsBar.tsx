import React from 'react'
import { Button, Spin, Typography } from 'antd'
import { useTranslation } from 'react-i18next'
import type { NotesBulkAction, NotesBulkFailures, NotesBulkProgress } from './hooks/useNotesBulkRun'
import type { NotesBulkSelectionControls } from './hooks/useNotesMatchingSelection'

/** What the bar needs from the page's bulk run (NL-05). */
export type NotesBulkRunControls = {
  progress: NotesBulkProgress | null
  failures: NotesBulkFailures | null
  onRetryFailures: () => void
  onDismissFailures: () => void
}

type NotesBulkActionsBarProps = {
  listMode: 'active' | 'trash'
  selectedCount: number
  isMobileViewport: boolean
  run?: NotesBulkRunControls
  selection?: NotesBulkSelectionControls
  onClearSelection: () => void
  onExport: () => void
  onAddTags: () => void
  onDelete: () => void
  /** Trash: restore the selected notes (NL-05). */
  onRestore?: () => void
}

const MAX_LISTED_FAILURES = 5

/** Fills `{{name}}` placeholders a translation left in place. */
const fill = (text: string, values: Record<string, string | number>) =>
  text.replace(/\{\{(\w+)\}\}/g, (match, name: string) => (name in values ? String(values[name]) : match))

// A bulk delete's Undo restores from the notes list, so its progress and
// failures show there, not in Trash.
const actionMode = (action: NotesBulkAction): 'active' | 'trash' => (action === 'restore' ? 'trash' : 'active')

const noteLabel = (note: { id: string | number; title?: string | null }) => note.title || `Note ${note.id}`

/**
 * The sticky bar under the notes list: the selection, its actions, a running
 * action's n/N progress, and the notes the last action could not change, with
 * Retry. While an action runs every control here is disabled (NL-05).
 */
export const NotesBulkActionsBar: React.FC<NotesBulkActionsBarProps> = ({
  listMode,
  selectedCount,
  isMobileViewport,
  run,
  selection,
  onClearSelection,
  onExport,
  onAddTags,
  onDelete,
  onRestore
}) => {
  const { t } = useTranslation(['option'])
  const progress = run?.progress ?? null
  const failures = run?.failures && actionMode(run.failures.action) === listMode ? run.failures : null
  const running = progress != null
  // Actions wait while "Select all matching" is still reading the matches.
  const actionsLocked = running || Boolean(selection?.selectingAllMatching)
  const hasSelection = selectedCount > 0
  if (!hasSelection && !running && !failures) return null

  const progressText = (value: NotesBulkProgress) => {
    const values = { done: value.done, total: value.total }
    const text = {
      delete: () => t('option:notesSearch.bulkProgressDelete', { defaultValue: 'Deleting {{done}}/{{total}}…', ...values }),
      addTags: () => t('option:notesSearch.bulkProgressAddTags', { defaultValue: 'Adding tags {{done}}/{{total}}…', ...values }),
      restore: () => t('option:notesSearch.bulkProgressRestore', { defaultValue: 'Restoring {{done}}/{{total}}…', ...values }),
      undoDelete: () => t('option:notesSearch.bulkProgressRestore', { defaultValue: 'Restoring {{done}}/{{total}}…', ...values })
    }[value.action]()
    return fill(text, values)
  }
  const failureText = (value: NotesBulkFailures) => {
    const values = { count: value.notes.length }
    const text = {
      delete: () => t('option:notesSearch.bulkFailedDelete', { defaultValue: 'Delete failed for {{count}}:', ...values }),
      addTags: () => t('option:notesSearch.bulkFailedAddTags', { defaultValue: 'Adding tags failed for {{count}}:', ...values }),
      restore: () => t('option:notesSearch.bulkFailedRestore', { defaultValue: 'Restore failed for {{count}}:', ...values }),
      undoDelete: () => t('option:notesSearch.bulkFailedRestore', { defaultValue: 'Restore failed for {{count}}:', ...values })
    }[value.action]()
    return fill(text, values)
  }
  const buttonClass = isMobileViewport ? 'min-h-[44px]' : undefined
  const renderMatchingSelection = () => {
    if (!selection) return null
    const { matching } = selection
    if (matching) {
      const values = { count: matching.count, total: matching.total }
      const text =
        matching.total > matching.count
          ? t('option:notesSearch.bulkFirstMatchingSelected', {
              defaultValue: 'The first {{count}} of {{total}} matching notes are selected.',
              ...values
            })
          : t('option:notesSearch.bulkAllMatchingSelected', {
              defaultValue: 'All {{count}} matching notes are selected.',
              ...values
            })
      return (
        <div className="mt-1 text-xs text-text-muted" data-testid="notes-bulk-matching-status">
          {fill(text, values)}
        </div>
      )
    }
    if (!selection.allOnPageSelected || selection.total <= selection.pageCount) return null
    const count = Math.min(selection.total, selection.matchingCap)
    const label =
      selection.total > selection.matchingCap
        ? t('option:notesSearch.bulkSelectFirstMatching', {
            defaultValue: 'Select the first {{count}} matching',
            count
          })
        : t('option:notesSearch.bulkSelectAllMatching', {
            defaultValue: 'Select all {{count}} matching',
            count
          })
    return (
      <Button
        size="small"
        type="link"
        className="!px-0 text-xs"
        loading={selection.selectingAllMatching}
        disabled={running}
        onClick={selection.onSelectAllMatching}
        data-testid="notes-bulk-select-all-matching"
      >
        {fill(label, { count })}
      </Button>
    )
  }

  return (
    // Only the status text is a live region: the count, progress and failures
    // are announced, the buttons around them are not.
    <div
      className="flex-shrink-0 border-t border-border bg-surface2 px-4 py-2"
      data-testid="notes-bulk-actions-bar"
    >
      {hasSelection && (
        <div className="flex items-center justify-between gap-2">
          <Typography.Text
            className="text-xs font-medium text-text"
            role="status"
            aria-live="polite"
            data-testid="notes-bulk-selected-count"
          >
            {fill(
              t('option:notesSearch.bulkSelectedCount', {
                defaultValue: '{{count}} selected',
                count: selectedCount
              }),
              { count: selectedCount }
            )}
          </Typography.Text>
          <Button
            size="small"
            type="link"
            className="!px-0 text-xs"
            disabled={running}
            onClick={onClearSelection}
            data-testid="notes-bulk-clear-selection"
          >
            {t('option:notesSearch.bulkClearSelection', { defaultValue: 'Clear selection' })}
          </Button>
        </div>
      )}
      {hasSelection && renderMatchingSelection()}
      {progress && progress.total > 0 && (
        <div
          className="mt-1 flex items-center gap-2 text-xs text-text-muted"
          role="status"
          aria-live="polite"
          data-testid="notes-bulk-progress"
        >
          <Spin size="small" />
          <span>{progressText(progress)}</span>
        </div>
      )}
      {hasSelection && listMode === 'trash' && (
        <div className="mt-2">
          <Button
            size="small"
            type="primary"
            className={buttonClass}
            disabled={actionsLocked || !onRestore}
            onClick={onRestore}
            data-testid="notes-bulk-restore"
          >
            {t('option:notesSearch.bulkRestore', { defaultValue: 'Restore selected' })}
          </Button>
        </div>
      )}
      {hasSelection && listMode === 'active' && (
        <div className="mt-2 grid grid-cols-3 gap-2">
          <Button
            size="small"
            className={buttonClass}
            disabled={actionsLocked}
            onClick={onExport}
            data-testid="notes-bulk-export"
          >
            {t('option:notesSearch.bulkExport', { defaultValue: 'Export selected' })}
          </Button>
          <Button
            size="small"
            className={buttonClass}
            disabled={actionsLocked}
            onClick={onAddTags}
            data-testid="notes-bulk-assign-keywords"
          >
            {t('option:notesSearch.bulkAddTags', { defaultValue: 'Add tags' })}
          </Button>
          <Button
            size="small"
            className={buttonClass}
            danger
            disabled={actionsLocked}
            onClick={onDelete}
            data-testid="notes-bulk-delete"
          >
            {t('option:notesSearch.bulkDelete', { defaultValue: 'Move selected to Trash' })}
          </Button>
        </div>
      )}
      {failures && (
        <div
          className="mt-2 rounded border border-warn/30 bg-warn/10 px-2 py-1.5 text-xs text-text"
          data-testid="notes-bulk-failures"
        >
          <span className="font-medium" role="status" aria-live="polite">
            {failureText(failures)}
          </span>
          <ul className="mt-0.5 flex flex-wrap gap-x-2 text-text-muted">
            {failures.notes.slice(0, MAX_LISTED_FAILURES).map((note) => (
              <li key={String(note.id)} className="max-w-full truncate">
                {noteLabel(note)}
              </li>
            ))}
            {failures.notes.length > MAX_LISTED_FAILURES && (
              <li>
                {fill(
                  t('option:notesSearch.bulkFailedMore', {
                    defaultValue: '+{{count}} more',
                    count: failures.notes.length - MAX_LISTED_FAILURES
                  }),
                  { count: failures.notes.length - MAX_LISTED_FAILURES }
                )}
              </li>
            )}
          </ul>
          <div className="mt-1 flex items-center gap-3">
            <Button
              size="small"
              type="link"
              className="!px-0 text-xs"
              disabled={running}
              onClick={run?.onRetryFailures}
              data-testid="notes-bulk-retry"
            >
              {t('option:notesSearch.bulkRetry', { defaultValue: 'Retry' })}
            </Button>
            <Button
              size="small"
              type="link"
              className="!px-0 text-xs"
              onClick={run?.onDismissFailures}
              data-testid="notes-bulk-dismiss-failures"
            >
              {t('option:notesSearch.bulkDismissFailures', { defaultValue: 'Dismiss' })}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

export default NotesBulkActionsBar
