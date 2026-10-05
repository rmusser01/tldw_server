import { useRef, useState } from 'react'
import { Modal } from 'antd'
import { useNavigate } from 'react-router-dom'
import { CheckSquare, Download, Tags, Trash2, X } from 'lucide-react'
import type { TFunction } from 'i18next'

import type { MediaResultItem } from '@/components/Media/types'

interface MediaBulkToolbarSelection {
  bulkSelectedItems: MediaResultItem[]
  bulkKeywordsDraft: string
  setBulkKeywordsDraft: (value: string) => void
  handleBulkAddKeywords: () => Promise<void> | void
  handleBulkDelete: () => Promise<void> | void
  collectionDraftName: string
  setCollectionDraftName: (value: string) => void
  handleAddSelectionToCollection: () => void
  handleOpenSelectionInMultiReview: () => Promise<void> | void
  bulkExportFormat: 'json' | 'markdown' | 'text'
  setBulkExportFormat: (value: 'json' | 'markdown' | 'text') => void
  handleBulkExport: () => void
  handleSelectAllVisibleItems: () => void
  handleClearBulkSelection: () => void
}

interface MediaBulkToolbarProps {
  selection: MediaBulkToolbarSelection
  t: TFunction
  deleteDisabledReason?: string
}

export function MediaBulkToolbar({ selection, t, deleteDisabledReason }: MediaBulkToolbarProps) {
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [moving, setMoving] = useState(false)
  const [showRecovery, setShowRecovery] = useState(false)
  const navigate = useNavigate()
  const pendingAction = useRef<MediaBulkToolbarSelection['handleBulkDelete'] | null>(null)
  const deleteButton = useRef<HTMLButtonElement>(null)
  const closeConfirmation = () => {
    setConfirmOpen(false)
    pendingAction.current = null
  }
  return (
    <div
      className="sticky top-0 z-10 shrink-0 border-b border-border bg-surface2 px-3 py-2 space-y-2"
      role="region"
      aria-label={t('review:mediaPage.selectionActions', { defaultValue: 'Selection actions' })}
      data-testid="media-bulk-toolbar"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium text-text-muted">
          {t('review:mediaPage.bulkSelectedCount', {
            defaultValue: '{{count}} selected',
            count: selection.bulkSelectedItems.length
          })}
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={selection.handleSelectAllVisibleItems}
            className="inline-flex min-h-[44px] md:min-h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface"
            data-testid="media-bulk-select-all"
          >
            <CheckSquare className="h-3.5 w-3.5" />
            {t('review:mediaPage.selectAllVisible', {
              defaultValue: 'Select this page'
            })}
          </button>
          <button
            type="button"
            onClick={selection.handleClearBulkSelection}
            className="inline-flex min-h-[44px] md:min-h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface"
            data-testid="media-bulk-clear"
          >
            <X className="h-3.5 w-3.5" />
            {t('review:mediaPage.clearSelection', {
              defaultValue: 'Clear selection'
            })}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => void selection.handleOpenSelectionInMultiReview()} disabled={!selection.bulkSelectedItems.some(item => item.kind === 'media')} className="min-h-[44px] md:min-h-8 rounded-md border border-border px-2 text-xs disabled:opacity-60" data-testid="media-bulk-open-multi">{t('review:mediaPage.bulkOpenMultiReview', { defaultValue: 'Open selection' })}</button>
        <button ref={deleteButton} type="button" onClick={() => { pendingAction.current = selection.handleBulkDelete; setConfirmOpen(true) }} disabled={!selection.bulkSelectedItems.length || Boolean(deleteDisabledReason) || moving} title={deleteDisabledReason} className="min-h-[44px] md:min-h-8 rounded-md border border-danger/50 px-2 text-xs text-danger disabled:opacity-60" data-testid="media-bulk-delete"><Trash2 className="inline h-4 w-4 mr-1" />{t('review:mediaPage.moveSelectionToTrash', { defaultValue: 'Move {{count}} items to trash', count: selection.bulkSelectedItems.length })}</button>
      </div>
      {selection.bulkSelectedItems.some(item => item.kind === 'note') ? (
        <p className="text-xs text-text-muted">{t('review:mediaPage.notesReviewScope', { defaultValue: 'Review opens Media only; Notes stay selected.' })}</p>
      ) : null}
      {showRecovery ? <button type="button" onClick={() => navigate('/media-trash')} className="min-h-[44px] md:min-h-8 rounded-md border border-border px-2 text-xs">{t('review:mediaPage.openTrashRecovery', { defaultValue: 'Open Trash' })}</button> : null}
      <details className="max-h-48 overflow-auto">
        <summary className="cursor-pointer min-h-[44px] md:min-h-8 py-3 md:py-1 text-xs">{t('review:mediaPage.moreSelectionActions', { defaultValue: 'Tags, collections and export' })}</summary>
      <div className="flex flex-wrap items-center gap-2">
        <input
          aria-label={t('review:mediaPage.bulkKeywordsPlaceholder', { defaultValue: 'Keywords (comma separated)' })}
          value={selection.bulkKeywordsDraft}
          onChange={(event) => selection.setBulkKeywordsDraft(event.target.value)}
          placeholder={t('review:mediaPage.bulkKeywordsPlaceholder', {
            defaultValue: 'Keywords (comma separated)'
          })}
          className="min-h-[44px] md:min-h-8 min-w-0 flex-1 rounded-md border border-border bg-surface px-2 text-[11px] text-text"
          data-testid="media-bulk-keywords-input"
        />
        <button
          type="button"
          onClick={() => void selection.handleBulkAddKeywords()}
          disabled={selection.bulkSelectedItems.length === 0}
          className="inline-flex min-h-[44px] md:min-h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
          data-testid="media-bulk-tag"
        >
          <Tags className="h-3.5 w-3.5" />
          {t('review:mediaPage.bulkAddKeywords', { defaultValue: 'Add tags' })}
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <input
          aria-label={t('review:mediaPage.collectionNamePlaceholder', { defaultValue: 'Collection name' })}
          value={selection.collectionDraftName}
          onChange={(event) => selection.setCollectionDraftName(event.target.value)}
          placeholder={t('review:mediaPage.collectionNamePlaceholder', {
            defaultValue: 'Collection name'
          })}
          className="min-h-[44px] md:min-h-8 min-w-0 w-36 rounded-md border border-border bg-surface px-2 text-[11px] text-text"
          data-testid="media-bulk-collection-name"
        />
        <button
          type="button"
          onClick={selection.handleAddSelectionToCollection}
          disabled={selection.bulkSelectedItems.length === 0}
          className="inline-flex min-h-[44px] md:min-h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
          data-testid="media-bulk-add-collection"
        >
          {t('review:mediaPage.collectionAddSelection', {
            defaultValue: 'Add to collection'
          })}
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <select
          aria-label={t('review:mediaPage.exportFormat', { defaultValue: 'Export format' })}
          value={selection.bulkExportFormat}
          onChange={(event) =>
            selection.setBulkExportFormat(
              event.target.value as 'json' | 'markdown' | 'text'
            )
          }
          className="min-h-[44px] md:min-h-8 rounded-md border border-border bg-surface px-2 text-[11px] text-text"
          data-testid="media-bulk-export-format"
        >
          <option value="json">
            {t('review:mediaPage.bulkExportJson', {
              defaultValue: 'JSON'
            })}
          </option>
          <option value="markdown">
            {t('review:mediaPage.bulkExportMarkdown', {
              defaultValue: 'Markdown'
            })}
          </option>
          <option value="text">
            {t('review:mediaPage.bulkExportText', {
              defaultValue: 'Plain text'
            })}
          </option>
        </select>
        <button
          type="button"
          onClick={selection.handleBulkExport}
          disabled={selection.bulkSelectedItems.length === 0}
          className="inline-flex min-h-[44px] md:min-h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
          data-testid="media-bulk-export"
        >
          <Download className="h-3.5 w-3.5" />
          {t('review:mediaPage.bulkExport', { defaultValue: 'Export' })}
        </button>
      </div>
      </details>
      <Modal
        open={confirmOpen}
        afterClose={() => deleteButton.current?.focus()}
        title={t('review:mediaPage.moveSelectionToTrash', { defaultValue: 'Move {{count}} items to trash', count: selection.bulkSelectedItems.length })}
        onCancel={closeConfirmation}
        onOk={async () => {
        const action = pendingAction.current
        if (!action) return
        setMoving(true)
        try {
          await action()
          setShowRecovery(true)
          closeConfirmation()
        } finally {
          setMoving(false)
        }
      }}
        confirmLoading={moving}
        okText={t('review:mediaPage.moveToTrash', { defaultValue: 'Move to trash' })}
        cancelText={t('common:cancel', { defaultValue: 'Cancel' })}>
        <p>{t('review:mediaPage.trashRecoveryHint', { defaultValue: 'You can restore items from Trash. Items that fail remain selected.' })}</p>
      </Modal>
    </div>
  )
}
