import React from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import {
  ANNOTATION_COLOR_OPTIONS,
  DOCUMENT_INTELLIGENCE_TABS,
  type MediaAnnotationColor
} from './hooks/useContentViewerModals'

type DocumentIntelligenceTab =
  | 'outline'
  | 'insights'
  | 'references'
  | 'figures'
  | 'annotations'

type IntelligencePanelState = {
  loading: boolean
  error: string | null
  data: any[]
}

type DocumentIntelligenceModals = {
  intelligenceSectionCollapsed: boolean
  activeIntelligenceTab: DocumentIntelligenceTab
  setActiveIntelligenceTab: (tab: DocumentIntelligenceTab) => void
  activeDocumentIntelligencePanel: IntelligencePanelState | null | undefined
  fetchDocumentIntelligence: () => Promise<void>
  annotationSelectionText: string
  annotationManualText: string
  setAnnotationManualText: (value: string) => void
  annotationDraftNote: string
  setAnnotationDraftNote: (value: string) => void
  annotationDraftColor: MediaAnnotationColor
  setAnnotationDraftColor: React.Dispatch<React.SetStateAction<MediaAnnotationColor>>
  handleCreateAnnotation: () => Promise<void>
  annotationCreating: boolean
  handleSyncAnnotations: () => Promise<void>
  annotationSyncing: boolean
  clearAnnotationDraft: () => void
  annotationUpdatingId: string | null
  handleUpdateAnnotationNote: (entry: any) => Promise<void>
  annotationDeletingId: string | null
  handleDeleteAnnotation: (annotationId: string) => Promise<void>
}

interface ContentViewerDocumentIntelligenceSectionProps {
  modals: DocumentIntelligenceModals
  onToggleCollapsed: () => void
  t: (key: string, opts?: Record<string, any>) => string
}

export function ContentViewerDocumentIntelligenceSection({
  modals,
  onToggleCollapsed,
  t
}: ContentViewerDocumentIntelligenceSectionProps) {
  const renderDocumentIntelligencePanel = () => {
    if (!modals.activeDocumentIntelligencePanel) {
      return null
    }

    if (modals.activeDocumentIntelligencePanel.loading) {
      return (
        <div
          className="text-xs text-text-muted"
          data-testid="media-intelligence-loading"
        >
          {t('review:mediaPage.intelligenceLoading', {
            defaultValue: 'Loading intelligence data...'
          })}
        </div>
      )
    }

    if (modals.activeDocumentIntelligencePanel.error) {
      return (
        <div className="space-y-2" data-testid="media-intelligence-error">
          <p className="text-xs text-danger">{modals.activeDocumentIntelligencePanel.error}</p>
          <button
            type="button"
            onClick={() => {
              void modals.fetchDocumentIntelligence()
            }}
            className="rounded border border-border bg-surface2 px-2 py-1 text-xs text-text hover:bg-surface"
            data-testid="media-intelligence-retry"
          >
            {t('common:retry', { defaultValue: 'Retry' })}
          </button>
        </div>
      )
    }

    if (
      modals.activeIntelligenceTab !== 'annotations' &&
      (!Array.isArray(modals.activeDocumentIntelligencePanel.data) ||
        modals.activeDocumentIntelligencePanel.data.length === 0)
    ) {
      return (
        <div
          className="text-xs text-text-muted"
          data-testid="media-intelligence-empty"
        >
          {t(`review:mediaPage.intelligenceEmpty.${modals.activeIntelligenceTab}`, {
            defaultValue: `No ${modals.activeIntelligenceTab} available for this item.`
          })}
        </div>
      )
    }

    if (modals.activeIntelligenceTab === 'outline') {
      return (
        <ul className="space-y-1 text-xs" data-testid="media-intelligence-outline-list">
          {modals.activeDocumentIntelligencePanel.data.map((entry: any, index: number) => (
            <li
              key={`${entry?.title || 'entry'}-${entry?.page || index}-${index}`}
              className="flex items-start justify-between gap-2 rounded bg-surface2 px-2 py-1 text-text"
              data-testid="media-intelligence-outline-item"
            >
              <span className="truncate">{entry?.title || `Section ${index + 1}`}</span>
              <span className="shrink-0 text-text-muted">{entry?.page ?? '\u2014'}</span>
            </li>
          ))}
        </ul>
      )
    }

    if (modals.activeIntelligenceTab === 'insights') {
      return (
        <ul className="space-y-2 text-xs" data-testid="media-intelligence-insights-list">
          {modals.activeDocumentIntelligencePanel.data.map((entry: any, index: number) => (
            <li
              key={`${entry?.category || 'insight'}-${index}`}
              className="rounded bg-surface2 px-2 py-1.5"
              data-testid="media-intelligence-insight-item"
            >
              <p className="font-medium text-text">{entry?.title || `Insight ${index + 1}`}</p>
              <p className="mt-1 whitespace-pre-wrap text-text-muted">
                {entry?.content || ''}
              </p>
            </li>
          ))}
        </ul>
      )
    }

    if (modals.activeIntelligenceTab === 'references') {
      return (
        <ul className="space-y-1 text-xs" data-testid="media-intelligence-references-list">
          {modals.activeDocumentIntelligencePanel.data.map((entry: any, index: number) => {
            const label =
              entry?.title ||
              entry?.raw_text ||
              t('review:mediaPage.referenceLabel', {
                defaultValue: `Reference ${index + 1}`
              })
            return (
              <li
                key={`${entry?.doi || entry?.url || 'reference'}-${index}`}
                className="rounded bg-surface2 px-2 py-1 text-text"
                data-testid="media-intelligence-reference-item"
              >
                {label}
              </li>
            )
          })}
        </ul>
      )
    }

    if (modals.activeIntelligenceTab === 'figures') {
      return (
        <ul className="space-y-1 text-xs" data-testid="media-intelligence-figures-list">
          {modals.activeDocumentIntelligencePanel.data.map((entry: any, index: number) => (
            <li
              key={`${entry?.id || 'figure'}-${index}`}
              className="rounded bg-surface2 px-2 py-1 text-text"
              data-testid="media-intelligence-figure-item"
            >
              {entry?.caption || `Figure ${index + 1}`} (p.{entry?.page ?? '\u2014'})
            </li>
          ))}
        </ul>
      )
    }

    const annotationEntries = Array.isArray(modals.activeDocumentIntelligencePanel.data)
      ? modals.activeDocumentIntelligencePanel.data
      : []

    return (
      <div className="space-y-2 text-xs" data-testid="media-intelligence-annotations-panel">
        <div className="space-y-2 rounded border border-border bg-surface2 p-2">
          <p className="text-[11px] text-text-muted">
            {modals.annotationSelectionText
              ? t('review:mediaPage.annotationSelectionCaptured', {
                  defaultValue: 'Selection captured. Add details and save.'
                })
              : t('review:mediaPage.annotationManualHint', {
                  defaultValue: 'Create an annotation from selected text or enter text manually.'
                })}
          </p>
          {modals.annotationSelectionText ? (
            <p
              className="max-h-20 overflow-y-auto rounded border border-border bg-surface px-2 py-1 text-text"
              data-testid="media-annotation-selection-preview"
            >
              {modals.annotationSelectionText}
            </p>
          ) : null}
          <textarea
            value={modals.annotationManualText}
            onChange={(event) => modals.setAnnotationManualText(event.target.value)}
            placeholder={t('review:mediaPage.annotationTextPlaceholder', {
              defaultValue: 'Annotation text'
            })}
            className="min-h-[56px] w-full rounded border border-border bg-surface px-2 py-1 text-xs text-text"
            data-testid="media-annotation-manual-text"
          />
          <input
            value={modals.annotationDraftNote}
            onChange={(event) => modals.setAnnotationDraftNote(event.target.value)}
            placeholder={t('review:mediaPage.annotationNotePlaceholder', {
              defaultValue: 'Optional note'
            })}
            className="h-8 w-full rounded border border-border bg-surface px-2 text-xs text-text"
            data-testid="media-annotation-note-input"
          />
          <div className="flex items-center gap-2">
            <select
              value={modals.annotationDraftColor}
              onChange={(event) =>
                modals.setAnnotationDraftColor(
                  event.target.value as MediaAnnotationColor
                )
              }
              className="h-8 rounded border border-border bg-surface px-2 text-xs text-text"
              data-testid="media-annotation-color"
            >
              {ANNOTATION_COLOR_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => {
                void modals.handleCreateAnnotation()
              }}
              disabled={modals.annotationCreating}
              className="inline-flex h-8 items-center rounded border border-border px-2 text-xs text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
              data-testid="media-annotation-create"
            >
              {modals.annotationCreating
                ? t('review:mediaPage.annotationSaving', {
                    defaultValue: 'Saving...'
                  })
                : t('review:mediaPage.annotationSave', {
                    defaultValue: 'Save annotation'
                  })}
            </button>
            <button
              type="button"
              onClick={() => {
                void modals.handleSyncAnnotations()
              }}
              disabled={modals.annotationSyncing || annotationEntries.length === 0}
              className="inline-flex h-8 items-center rounded border border-border px-2 text-xs text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
              data-testid="media-annotation-sync"
            >
              {modals.annotationSyncing
                ? t('review:mediaPage.annotationSyncing', {
                    defaultValue: 'Syncing...'
                  })
                : t('review:mediaPage.annotationSync', {
                    defaultValue: 'Sync now'
                  })}
            </button>
            <button
              type="button"
              onClick={modals.clearAnnotationDraft}
              className="inline-flex h-8 items-center rounded border border-border px-2 text-xs text-text hover:bg-surface"
              data-testid="media-annotation-clear-draft"
            >
              {t('common:clear', { defaultValue: 'Clear' })}
            </button>
          </div>
        </div>

        {annotationEntries.length > 0 ? (
          <ul className="space-y-1 text-xs" data-testid="media-intelligence-annotations-list">
            {annotationEntries.map((entry: any, index: number) => (
              <li
                key={`${entry?.id || 'annotation'}-${index}`}
                className="rounded bg-surface2 px-2 py-1 text-text"
                data-testid="media-intelligence-annotation-item"
              >
                <p>{entry?.text || entry?.note || `Annotation ${index + 1}`}</p>
                {entry?.note ? (
                  <p className="mt-1 text-[11px] text-text-muted">{entry.note}</p>
                ) : null}
                <div className="mt-1 flex items-center gap-2">
                  <span className="rounded bg-surface px-1.5 py-0.5 text-[10px] uppercase text-text-muted">
                    {entry.color || 'yellow'}
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      void modals.handleUpdateAnnotationNote(entry)
                    }}
                    disabled={modals.annotationUpdatingId === entry.id}
                    className="rounded border border-border px-2 py-0.5 text-[11px] text-text hover:bg-surface disabled:cursor-not-allowed disabled:opacity-60"
                    data-testid={`media-annotation-edit-${entry.id}`}
                  >
                    {modals.annotationUpdatingId === entry.id
                      ? t('review:mediaPage.annotationUpdating', {
                          defaultValue: 'Updating...'
                        })
                      : t('common:edit', { defaultValue: 'Edit' })}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      void modals.handleDeleteAnnotation(entry.id)
                    }}
                    disabled={modals.annotationDeletingId === entry.id}
                    className="rounded border border-danger/50 px-2 py-0.5 text-[11px] text-danger hover:bg-danger/10 disabled:cursor-not-allowed disabled:opacity-60"
                    data-testid={`media-annotation-delete-${entry.id}`}
                  >
                    {modals.annotationDeletingId === entry.id
                      ? t('review:mediaPage.annotationDeleting', {
                          defaultValue: 'Deleting...'
                        })
                      : t('common:delete', { defaultValue: 'Delete' })}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    )
  }

  return (
    <div
      className="bg-surface border border-border rounded-lg mb-2 overflow-hidden"
      data-testid="media-intelligence-section"
    >
      <button
        onClick={onToggleCollapsed}
        className="w-full flex items-center justify-between px-3 py-2 bg-surface2 hover:bg-surface transition-colors"
        title={t('review:mediaPage.documentIntelligence', {
          defaultValue: 'Document Intelligence'
        })}
        data-testid="media-intelligence-toggle"
      >
        <span className="text-sm font-medium text-text">
          {t('review:mediaPage.documentIntelligence', {
            defaultValue: 'Document Intelligence'
          })}
        </span>
        {modals.intelligenceSectionCollapsed ? (
          <ChevronDown className="w-4 h-4 text-text-subtle" />
        ) : (
          <ChevronUp className="w-4 h-4 text-text-subtle" />
        )}
      </button>
      {!modals.intelligenceSectionCollapsed ? (
        <div
          className="space-y-2 p-3 bg-surface animate-in fade-in slide-in-from-top-1 duration-150"
          data-testid="media-intelligence-panel"
        >
          <div className="flex flex-wrap gap-1">
            {DOCUMENT_INTELLIGENCE_TABS.map((tab) => {
              const isActive = tab.key === modals.activeIntelligenceTab
              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => modals.setActiveIntelligenceTab(tab.key)}
                  className={`rounded border px-2 py-1 text-xs transition-colors ${
                    isActive
                      ? 'border-primary bg-primary text-white'
                      : 'border-border bg-surface2 text-text hover:bg-surface'
                  }`}
                  aria-pressed={isActive}
                  data-testid={`media-intelligence-tab-${tab.key}`}
                >
                  {t(`review:mediaPage.documentIntelligenceTab.${tab.key}`, {
                    defaultValue: tab.label
                  })}
                </button>
              )
            })}
          </div>
          <div data-testid="media-intelligence-content">
            {renderDocumentIntelligencePanel()}
          </div>
        </div>
      ) : null}
    </div>
  )
}
