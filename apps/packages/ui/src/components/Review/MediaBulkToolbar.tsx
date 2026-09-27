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
  return (
    <div
      className="border-b border-border bg-surface2 px-4 py-3 space-y-2.5"
      data-testid="media-bulk-toolbar"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] font-medium text-text-muted">
          {t('review:mediaPage.bulkSelectedCount', {
            defaultValue: '{{count}} selected',
            count: selection.bulkSelectedItems.length
          })}
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={selection.handleSelectAllVisibleItems}
            className="inline-flex h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface"
            data-testid="media-bulk-select-all"
          >
            <CheckSquare className="h-3.5 w-3.5" />
            {t('review:mediaPage.selectAllVisible', {
              defaultValue: 'Select visible'
            })}
          </button>
          <button
            type="button"
            onClick={selection.handleClearBulkSelection}
            className="inline-flex h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface"
            data-testid="media-bulk-clear"
          >
            <X className="h-3.5 w-3.5" />
            {t('review:mediaPage.clearSelection', {
              defaultValue: 'Clear'
            })}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <input
          value={selection.bulkKeywordsDraft}
          onChange={(event) => selection.setBulkKeywordsDraft(event.target.value)}
          placeholder={t('review:mediaPage.bulkKeywordsPlaceholder', {
            defaultValue: 'Keywords (comma separated)'
          })}
          className="h-8 min-w-[180px] flex-1 rounded-md border border-border bg-surface px-2 text-[11px] text-text"
          data-testid="media-bulk-keywords-input"
        />
        <button
          type="button"
          onClick={() => void selection.handleBulkAddKeywords()}
          disabled={selection.bulkSelectedItems.length === 0}
          className="inline-flex h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
          data-testid="media-bulk-tag"
        >
          <Tags className="h-3.5 w-3.5" />
          {t('review:mediaPage.bulkAddKeywords', { defaultValue: 'Add tags' })}
        </button>
        <button
          type="button"
          onClick={() => void selection.handleBulkDelete()}
          disabled={selection.bulkSelectedItems.length === 0 || Boolean(deleteDisabledReason)}
          title={deleteDisabledReason}
          className="inline-flex h-8 items-center gap-1 rounded-md border border-danger/50 px-2 text-[11px] text-danger hover:bg-danger/10 disabled:cursor-not-allowed disabled:opacity-60"
          data-testid="media-bulk-delete"
        >
          <Trash2 className="h-3.5 w-3.5" />
          {t('review:mediaPage.bulkDelete', { defaultValue: 'Delete' })}
        </button>
      </div>

      <div className="flex items-center gap-2">
        <input
          value={selection.collectionDraftName}
          onChange={(event) => selection.setCollectionDraftName(event.target.value)}
          placeholder={t('review:mediaPage.collectionNamePlaceholder', {
            defaultValue: 'Collection name'
          })}
          className="h-8 min-w-[140px] rounded-md border border-border bg-surface px-2 text-[11px] text-text"
          data-testid="media-bulk-collection-name"
        />
        <button
          type="button"
          onClick={selection.handleAddSelectionToCollection}
          disabled={selection.bulkSelectedItems.length === 0}
          className="inline-flex h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
          data-testid="media-bulk-add-collection"
        >
          {t('review:mediaPage.collectionAddSelection', {
            defaultValue: 'Add to collection'
          })}
        </button>
        <button
          type="button"
          onClick={() => {
            void selection.handleOpenSelectionInMultiReview()
          }}
          disabled={selection.bulkSelectedItems.length === 0}
          className="inline-flex h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
          data-testid="media-bulk-open-multi"
        >
          {t('review:mediaPage.bulkOpenMultiReview', {
            defaultValue: 'Open selection'
          })}
        </button>
      </div>

      <div className="flex items-center gap-2">
        <select
          value={selection.bulkExportFormat}
          onChange={(event) =>
            selection.setBulkExportFormat(
              event.target.value as 'json' | 'markdown' | 'text'
            )
          }
          className="h-8 rounded-md border border-border bg-surface px-2 text-[11px] text-text"
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
          className="inline-flex h-8 items-center gap-1 rounded-md border border-border px-2 text-[11px] text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
          data-testid="media-bulk-export"
        >
          <Download className="h-3.5 w-3.5" />
          {t('review:mediaPage.bulkExport', { defaultValue: 'Export' })}
        </button>
      </div>
    </div>
  )
}
