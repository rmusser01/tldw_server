import React from 'react'
import { Button, Dropdown, Pagination, Spin, Tooltip } from 'antd'
import { Clock3 as ClockIcon, Link2 as LinkIcon, Star as StarIcon, Tag as TagIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import type { ServerCapabilities } from '@/services/tldw/server-capabilities'
import type { NoteListItem } from '@/components/Notes/types'

const LazyNotesListPanelEmptyStates = React.lazy(() => import('./NotesListPanelEmptyStates'))

const MAX_TITLE_LENGTH = 80
const MAX_PREVIEW_LENGTH = 140
const RECENT_EDIT_WINDOW_MS = 24 * 60 * 60 * 1000

const truncateText = (value?: string | null, max?: number) => {
  if (!value) return ''
  if (!max || value.length <= max) return value
  return `${value.slice(0, max)}...`
}

const derivePreviewText = (content?: string | null, title?: string | null) => {
  const source = String(content || '').replace(/\r\n/g, '\n').trim()
  if (!source) return ''
  const normalizedTitle = String(title || '')
    .trim()
    .replace(/^#+\s*/, '')
    .toLowerCase()
  const lines = source
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
  const firstNonTitleLine = lines.find((line) => {
    if (!normalizedTitle) return true
    return line.replace(/^#+\s*/, '').toLowerCase() !== normalizedTitle
  })
  return truncateText(firstNonTitleLine || lines[0] || source, MAX_PREVIEW_LENGTH)
}

const wasEditedRecently = (updatedAt?: string | null) => {
  if (!updatedAt) return false
  const parsed = new Date(updatedAt)
  const timestamp = parsed.getTime()
  if (Number.isNaN(timestamp)) return false
  return Date.now() - timestamp <= RECENT_EDIT_WINDOW_MS
}

const SEARCH_TERM_PATTERN = /"([^"]+)"|(\S+)/g

const extractSearchTerms = (query?: string): string[] => {
  const text = String(query || '').trim()
  if (!text) return []
  const terms: string[] = []
  const seen = new Set<string>()
  for (const match of text.matchAll(SEARCH_TERM_PATTERN)) {
    const raw = (match[1] || match[2] || '').trim()
    if (!raw) continue
    const key = raw.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    terms.push(raw)
  }
  return terms.sort((a, b) => b.length - a.length)
}

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

const renderHighlightedText = (text: string, terms: string[]) => {
  if (!text || terms.length === 0) return text
  const pattern = terms.map((term) => escapeRegExp(term)).join('|')
  if (!pattern) return text
  const regex = new RegExp(`(${pattern})`, 'ig')
  const parts = text.split(regex)
  return parts.map((part, index) => {
    const isMatch = terms.some((term) => term.toLowerCase() === part.toLowerCase())
    if (!isMatch) return <React.Fragment key={`text-${index}`}>{part}</React.Fragment>
    return (
      <mark
        key={`mark-${index}`}
        className="rounded-sm bg-primary/20 px-0.5 text-text"
      >
        {part}
      </mark>
    )
  })
}

type NotesListPanelProps = {
  listMode: 'active' | 'trash'
  searchQuery?: string
  conversationLabelById?: Record<string, string>
  bulkSelectedIds?: string[]
  pinnedNoteIds?: string[]
  isOnline: boolean
  isFetching: boolean
  demoEnabled: boolean
  capsLoading: boolean
  capabilities: ServerCapabilities | null
  notes: NoteListItem[] | undefined
  total: number
  page: number
  pageSize: number
  selectedId: string | number | null
  onSelectNote: (id: string | number) => void
  onToggleBulkSelection?: (id: string | number, checked: boolean, shiftKey: boolean) => void
  onTogglePinned?: (id: string | number) => void
  onChangePage: (page: number, pageSize: number) => void
  onResetEditor: () => void
  onOpenSettings: () => void
  onOpenHealth: () => void
  onRestoreNote: (id: string | number, version?: number) => void
  onExportAllMd: () => void
  onExportAllCsv: () => void
  onExportAllJson: () => void
  onImportNotes?: () => void
  importInProgress?: boolean
  exportProgress?: {
    format: 'md' | 'csv' | 'json'
    fetchedNotes: number
    fetchedPages: number
    failedBatches: number
  } | null
}

const NotesListPanel: React.FC<NotesListPanelProps> = ({
  listMode,
  searchQuery,
  conversationLabelById = {},
  bulkSelectedIds = [],
  pinnedNoteIds = [],
  isOnline,
  isFetching,
  demoEnabled,
  capsLoading,
  capabilities,
  notes,
  total,
  page,
  pageSize,
  selectedId,
  onSelectNote,
  onToggleBulkSelection,
  onTogglePinned,
  onChangePage,
  onResetEditor,
  onOpenSettings,
  onOpenHealth,
  onRestoreNote,
  onExportAllMd,
  onExportAllCsv,
  onExportAllJson,
  onImportNotes,
  importInProgress = false,
  exportProgress = null
}) => {
  const { t } = useTranslation(['option', 'settings'])
  const isTrashView = listMode === 'trash'
  const hasNotes = Array.isArray(notes) && notes.length > 0
  const bulkSelectedIdSet = React.useMemo(
    () => new Set((bulkSelectedIds || []).map((id) => String(id))),
    [bulkSelectedIds]
  )
  const pinnedNoteIdSet = React.useMemo(
    () => new Set((pinnedNoteIds || []).map((id) => String(id))),
    [pinnedNoteIds]
  )
  const searchTerms = React.useMemo(() => extractSearchTerms(searchQuery), [searchQuery])
  const startItem = hasNotes ? (page - 1) * pageSize + 1 : 0
  const endItem = hasNotes ? Math.min(page * pageSize, total) : 0
  const isExporting = exportProgress != null
  const exportDisabled = !isOnline || !hasNotes || isTrashView || isExporting
  const importDisabled = !isOnline || isTrashView || importInProgress

  const renderEmptyStateSurface = (
    variant: 'demo' | 'connect' | 'unsupported' | 'empty',
  ) => (
    <React.Suspense fallback={null}>
      <LazyNotesListPanelEmptyStates
        variant={variant}
        isTrashView={isTrashView}
        onOpenSettings={onOpenSettings}
        onOpenHealth={onOpenHealth}
        onResetEditor={onResetEditor}
      />
    </React.Suspense>
  )

  return (
    <div className="flex flex-col h-full">
      {/* Export header */}
      <div className="flex-shrink-0 px-4 py-2 border-b border-border bg-surface2">
        <div className="flex items-center justify-between">
          <span className="text-xs uppercase tracking-[0.14em] text-text-muted">
            {isTrashView
              ? t('option:notesSearch.trashResultsLabel', { defaultValue: 'Trash' })
              : t('option:notesSearch.resultsLabel', { defaultValue: 'Results' })}
          </span>
          <div className="flex items-center gap-1">
            <Tooltip
              title={
                importDisabled
                  ? t('option:notesSearch.importDisabled', {
                      defaultValue: isOnline
                        ? 'Switch to Notes view to import'
                        : 'Connect to import notes'
                    })
                  : undefined
              }
            >
              <Button
                size="small"
                type="text"
                className="text-xs"
                disabled={importDisabled}
                loading={importInProgress}
                onClick={() => onImportNotes?.()}
              >
                {t('option:notesSearch.importMenuTrigger', {
                  defaultValue: 'Import'
                })}
              </Button>
            </Tooltip>
            <Dropdown
              menu={{
                items: [
                  {
                    key: 'md',
                    label: t('option:notesSearch.exportMdTooltip', {
                      defaultValue: 'Export matching notes as Markdown (.md)'
                    })
                  },
                  {
                    key: 'csv',
                    label: t('option:notesSearch.exportCsvTooltip', {
                      defaultValue: 'Export matching notes as CSV'
                    })
                  },
                  {
                    key: 'json',
                    label: t('option:notesSearch.exportJsonTooltip', {
                      defaultValue: 'Export matching notes as JSON'
                    })
                  }
                ],
                onClick: ({ key }) => {
                  if (exportDisabled) return
                  if (key === 'md') onExportAllMd()
                  if (key === 'csv') onExportAllCsv()
                  if (key === 'json') onExportAllJson()
                }
              }}
              disabled={exportDisabled}
            >
              <span className="inline-flex">
                <Tooltip
                  title={
                    exportDisabled
                      ? isExporting
                        ? t('option:notesSearch.exportProgressDisabled', {
                            defaultValue: 'Export in progress'
                          })
                        : t('option:notesSearch.exportDisabled', {
                            defaultValue: isOnline
                              ? 'No results to export'
                              : 'Connect to export notes'
                          })
                      : undefined
                  }
                >
                  <Button
                    size="small"
                    type="text"
                    className="text-xs"
                    disabled={exportDisabled}
                  >
                    {t('option:notesSearch.exportMenuTrigger', {
                      defaultValue: 'Export'
                    })}
                  </Button>
                </Tooltip>
              </span>
            </Dropdown>
          </div>
        </div>
        {exportProgress && (
          <div
            className="mt-2 inline-flex items-center gap-2 text-[11px] text-text-muted"
            role="status"
            aria-live="polite"
            data-testid="notes-export-progress"
          >
            <Spin size="small" />
            <span>
              {t('option:notesSearch.exportProgressLabel', {
                defaultValue: 'Exporting {{format}}'
              })
                .replace('{{format}}', exportProgress.format.toUpperCase())}
              {`: ${exportProgress.fetchedNotes} notes across ${exportProgress.fetchedPages} batch${
                exportProgress.fetchedPages === 1 ? '' : 'es'
              }`}
            </span>
            {exportProgress.failedBatches > 0 && (
              <span>
                {` · ${exportProgress.failedBatches} failed`}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Content area */}
      <div className="flex-1 overflow-auto p-4">
      {isFetching ? (
        <div className="flex items-center justify-center py-10">
          <Spin />
        </div>
      ) : !isOnline ? (
        demoEnabled ? (
          renderEmptyStateSurface('demo')
        ) : (
          renderEmptyStateSurface('connect')
        )
      ) : !capsLoading && capabilities && !capabilities.hasNotes ? (
        renderEmptyStateSurface('unsupported')
      ) : Array.isArray(notes) && notes.length > 0 ? (
        <>
          <div className="divide-y divide-border">
            {notes.map((item) => (
              (() => {
                const itemIdText = String(item.id)
                const isBulkSelected = bulkSelectedIdSet.has(itemIdText)
                const isSelectedNote =
                  selectedId != null && String(selectedId) === itemIdText
                const isPinned = pinnedNoteIdSet.has(itemIdText)
                return (
                  <div
                    key={itemIdText}
                    className={`w-full py-3 text-left transition-colors ${
                      isTrashView
                        ? 'px-4'
                        : isSelectedNote
                          ? 'bg-surface2 border-l-4 border-l-primary px-3'
                          : isBulkSelected
                            ? 'bg-surface2/70 px-4'
                            : 'px-4 hover:bg-surface2'
                    }`}
                    data-testid={isTrashView ? `notes-trash-row-${itemIdText}` : undefined}
                  >
                {!isTrashView && (
                  <div className="flex items-start gap-2">
                    <input
                      type="checkbox"
                      checked={isBulkSelected}
                      aria-label={t('option:notesSearch.bulkSelectAria', {
                        defaultValue: 'Select note {{title}} for bulk actions',
                        title: item.title || `Note ${item.id}`
                      })}
                      data-testid={`notes-select-checkbox-${itemIdText.replace(/[^a-z0-9_-]/gi, '_')}`}
                      onChange={() => {}}
                      onClick={(event) => {
                        event.stopPropagation()
                        onToggleBulkSelection?.(item.id, !isBulkSelected, event.shiftKey)
                      }}
                    />
                    <button
                      type="button"
                      aria-label={
                        isPinned
                          ? t('option:notesSearch.unpinNoteAria', {
                              defaultValue: 'Unpin note {{title}}',
                              title: item.title || `Note ${item.id}`
                            })
                          : t('option:notesSearch.pinNoteAria', {
                              defaultValue: 'Pin note {{title}}',
                              title: item.title || `Note ${item.id}`
                            })
                      }
                      className="mt-0.5 inline-flex h-6 w-6 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text"
                      data-testid={`notes-pin-toggle-${itemIdText.replace(/[^a-z0-9_-]/gi, '_')}`}
                      onClick={(event) => {
                        event.stopPropagation()
                        onTogglePinned?.(item.id)
                      }}
                    >
                      <StarIcon
                        className={`h-4 w-4 ${isPinned ? 'fill-current text-amber-500' : ''}`}
                        aria-hidden="true"
                      />
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        onSelectNote(item.id)
                      }}
                      className="w-full text-left"
                      aria-selected={isSelectedNote}
                      aria-current={isSelectedNote ? "true" : undefined}
                      data-testid={`notes-open-button-${itemIdText.replace(/[^a-z0-9_-]/gi, '_')}`}
                    >
                      <div className="w-full">
                        {(() => {
                          const titleText = truncateText(
                            item.title || `Note ${item.id}`,
                            MAX_TITLE_LENGTH
                          )
                          const hasKeywords = Array.isArray(item.keywords) && item.keywords.length > 0
                          const hasBacklink = Boolean(item.conversation_id)
                          const editedRecently = wasEditedRecently(item.updated_at)
                          return (
                            <div className="flex items-start justify-between gap-2">
                              <div className="text-sm font-medium text-text truncate">
                                {renderHighlightedText(titleText, searchTerms)}
                              </div>
                              {(isPinned || hasKeywords || hasBacklink || editedRecently) && (
                                <div
                                  className="mt-0.5 flex items-center gap-1 text-text-muted"
                                  data-testid={`notes-item-badges-${String(item.id)}`}
                                >
                                  {isPinned && (
                                    <Tooltip
                                      title={t('option:notesSearch.badgePinned', {
                                        defaultValue: 'Pinned note'
                                      })}
                                    >
                                      <StarIcon className="h-3.5 w-3.5 fill-current text-amber-500" aria-hidden="true" />
                                    </Tooltip>
                                  )}
                                  {hasKeywords && (
                                    <Tooltip
                                      title={t('option:notesSearch.badgeHasKeywords', {
                                        defaultValue: 'Has keywords'
                                      })}
                                    >
                                      <TagIcon className="h-3.5 w-3.5" aria-hidden="true" />
                                    </Tooltip>
                                  )}
                                  {hasBacklink && (
                                    <Tooltip
                                      title={t('option:notesSearch.badgeHasBacklink', {
                                        defaultValue: 'Linked conversation'
                                      })}
                                    >
                                      <LinkIcon className="h-3.5 w-3.5" aria-hidden="true" />
                                    </Tooltip>
                                  )}
                                  {editedRecently && (
                                    <Tooltip
                                      title={t('option:notesSearch.badgeEditedRecently', {
                                        defaultValue: 'Edited in last 24 hours'
                                      })}
                                    >
                                      <ClockIcon className="h-3.5 w-3.5" aria-hidden="true" />
                                    </Tooltip>
                                  )}
                                </div>
                              )}
                            </div>
                          )
                        })()}
                        {item.content && (
                          <div className="text-xs text-text-muted truncate mt-1">
                            {renderHighlightedText(
                              derivePreviewText(item.content, item.title),
                              searchTerms
                            )}
                          </div>
                        )}
                        {Array.isArray(item.keywords) && item.keywords.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {item.keywords.slice(0, 5).map((keyword, idx) => (
                              <span
                                key={`${keyword}-${idx}`}
                                className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-surface2 text-text"
                              >
                                {keyword}
                              </span>
                            ))}
                            {item.keywords.length > 5 && (
                              <Tooltip
                                title={t('option:notesSearch.moreTagsTooltip', {
                                  defaultValue: '+{{count}} more tags',
                                  count: item.keywords.length - 5
                                })}
                              >
                                <span className="inline-flex items-center px-2 py-0.5 text-xs text-text-muted">
                                  +{item.keywords.length - 5}
                                </span>
                              </Tooltip>
                            )}
                          </div>
                        )}
                        {item.conversation_id &&
                          (() => {
                            const conversationId = String(item.conversation_id)
                            const conversationLabel = conversationLabelById[conversationId] || conversationId
                            return (
                              <div className="text-xs text-primary mt-1">
                                {t('option:notesSearch.linkedConversation', {
                                  defaultValue: 'Linked to conversation'
                                })}
                                {': '}
                                <Tooltip
                                  title={`${t('option:notesSearch.linkedConversationIdLabel', {
                                    defaultValue: 'Conversation ID'
                                  })}: ${conversationId}`}
                                >
                                  <span className="font-medium">{conversationLabel}</span>
                                </Tooltip>
                                {item.message_id ? ` · msg ${String(item.message_id)}` : ''}
                              </div>
                            )
                          })()}
                        <div className="text-xs text-text-subtle mt-1">
                          {item.updated_at
                            ? (() => {
                                const d = new Date(item.updated_at)
                                return isNaN(d.getTime()) ? '' : d.toLocaleString()
                              })()
                            : ''}
                        </div>
                      </div>
                    </button>
                  </div>
                )}
                {isTrashView && (
                  <div className="w-full">
                    <div className="text-sm font-medium text-text truncate">
                      {truncateText(
                        item.title || `Note ${item.id}`,
                        MAX_TITLE_LENGTH
                      )}
                    </div>
                    {item.content && (
                      <div className="text-xs text-text-muted truncate mt-1">
                        {derivePreviewText(item.content, item.title)}
                      </div>
                    )}
                    <div className="mt-2 flex items-center justify-between gap-2">
                      <div className="text-xs text-text-subtle">
                        {t('option:notesSearch.trashedAtLabel', {
                          defaultValue: 'Deleted'
                        })}
                        {item.updated_at
                          ? (() => {
                              const d = new Date(item.updated_at)
                              return isNaN(d.getTime())
                                ? ''
                                : ` · ${d.toLocaleString()}`
                            })()
                          : ''}
                      </div>
                      <Button
                        size="small"
                        type="primary"
                        onClick={() => onRestoreNote(item.id, item.version)}
                        data-testid={`notes-restore-${String(item.id)}`}
                      >
                        {t('option:notesSearch.restoreAction', {
                          defaultValue: 'Restore'
                        })}
                      </Button>
                    </div>
                  </div>
                )}
                  </div>
                )
              })()
            ))}
          </div>
        </>
      ) : (
        renderEmptyStateSurface('empty')
      )}
      </div>

      {/* Pagination Footer */}
      {hasNotes && (
        <div className="flex-shrink-0 px-4 py-3 border-t border-border bg-surface">
          <div className="flex items-center justify-between text-xs text-text-muted">
            <div>
              {t('option:notesSearch.showingRange', {
                defaultValue: 'Showing {{start}}-{{end}} of {{total}}',
                start: startItem,
                end: endItem,
                total
              })}
            </div>
            <Pagination
              size="small"
              current={page}
              pageSize={pageSize}
              total={total}
              onChange={(p, ps) => {
                onChangePage(p, ps)
              }}
              showSizeChanger
              pageSizeOptions={["20", "50", "100"]}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default NotesListPanel
