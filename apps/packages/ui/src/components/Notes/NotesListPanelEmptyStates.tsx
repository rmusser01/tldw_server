import React from 'react'
import { useTranslation } from 'react-i18next'
import { AlertTriangle as AlertTriangleIcon, SearchX as SearchXIcon } from 'lucide-react'
import FeatureEmptyState from '@/components/Common/FeatureEmptyState'
import ConnectionProblemBanner from '@/components/Common/ConnectionProblemBanner'
import { useConnectionActions } from '@/hooks/useConnectionState'
import { getDemoNotes } from '@/utils/demo-content'

type NotesListPanelEmptyStatesProps = {
  variant: 'demo' | 'connect' | 'unsupported' | 'empty' | 'no-results' | 'error'
  isTrashView: boolean
  searchQuery?: string
  errorMessage?: string | null
  onOpenSettings: () => void
  onOpenHealth: () => void
  onCreateNote: () => void
  onResetEditor: () => void
  onClearFilters?: () => void
  onRetry?: () => void
}

const NotesListPanelEmptyStates: React.FC<NotesListPanelEmptyStatesProps> = ({
  variant,
  isTrashView,
  searchQuery,
  errorMessage,
  onOpenSettings,
  onOpenHealth,
  onCreateNote,
  onResetEditor,
  onClearFilters,
  onRetry,
}) => {
  const { t } = useTranslation(['option', 'settings'])
  const { checkOnce } = useConnectionActions()
  const demoNotes = React.useMemo(() => getDemoNotes(t), [t])
  const normalizedSearchQuery = String(searchQuery || '').trim()

  if (variant === 'demo') {
    return (
      <div className="space-y-4">
        <FeatureEmptyState
          title={
            <span className="inline-flex items-center gap-2">
              <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primaryStrong">
                Demo
              </span>
              <span>
                {t('option:notesEmpty.demoTitle', {
                  defaultValue: 'Explore Notes in demo mode'
                })}
              </span>
            </span>
          }
          description={t('option:notesEmpty.demoDescription', {
            defaultValue:
              'This demo shows how Notes can organize your insights. Connect your own server later to create and save real notes.'
          })}
          examples={[
            t('option:notesEmpty.demoExample1', {
              defaultValue:
                'See how note titles, previews, and timestamps appear in this list.'
            }),
            t('option:notesEmpty.demoExample2', {
              defaultValue:
                'When you connect, you’ll be able to create notes from meetings, reviews, and more.'
            }),
            t('option:notesEmpty.demoExample3', {
              defaultValue:
                'Use Notes alongside Media and Review to keep track of your findings.'
            })
          ]}
          primaryActionLabel={t('settings:tldw.setupLink', 'Set up server')}
          onPrimaryAction={onOpenSettings}
        />
        <div className="rounded-lg border border-dashed border-border bg-surface p-3 text-xs text-text">
          <div className="mb-2 font-semibold">
            {t('option:notesEmpty.demoPreviewHeading', {
              defaultValue: 'Example notes (preview only)'
            })}
          </div>
          <div className="divide-y divide-border">
            {demoNotes.map((note) => (
              <div key={note.id} className="py-2">
                <div className="text-sm font-medium text-text">
                  {note.title}
                </div>
                <div className="mt-1 text-[11px] text-text-muted">
                  {note.preview}
                </div>
                <div className="mt-1 text-[11px] text-text-subtle">
                  {note.updated_at}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (variant === 'connect') {
    return (
      <ConnectionProblemBanner
        badgeLabel="Not connected"
        title={t('option:notesEmpty.connectTitle', {
          defaultValue: 'Connect to use Notes'
        })}
        description={t('option:notesEmpty.connectDescription', {
          defaultValue:
            'This view needs a connected server. Use the server connection card above to fix your connection, then return here to capture and organize notes.'
        })}
        examples={[
          t('option:notesEmpty.connectExample1', {
            defaultValue:
              'Use the connection card at the top of this page to add your server URL and API key.'
          })
        ]}
        primaryActionLabel={t('settings:tldw.setupLink', 'Set up server')}
        onPrimaryAction={onOpenSettings}
        retryActionLabel={t('option:buttonRetry', 'Retry connection')}
        onRetry={() => {
          void checkOnce()
        }}
      />
    )
  }

  if (variant === 'unsupported') {
    return (
      <FeatureEmptyState
        title={
          <span className="inline-flex items-center gap-2">
            <span className="rounded-full bg-warn/10 px-2 py-0.5 text-[11px] font-medium text-warn">
              Feature unavailable
            </span>
            <span>
              {t('option:notesEmpty.offlineTitle', {
                defaultValue: 'Notes API not available on this server'
              })}
            </span>
          </span>
        }
        description={t('option:notesEmpty.offlineDescription', {
          defaultValue:
            'This tldw server does not advertise the Notes endpoints (for example, /api/v1/notes/). Upgrade your server to a version that includes the Notes API to use this workspace.'
        })}
        examples={[
          t('option:notesEmpty.offlineExample1', {
            defaultValue:
              'Open Health & diagnostics to confirm your server version and available APIs.'
          }),
          t('option:notesEmpty.offlineExample2', {
            defaultValue:
              'After upgrading, reload the extension and return to Notes.'
          })
        ]}
        primaryActionLabel={t('settings:healthSummary.diagnostics', {
          defaultValue: 'Health & diagnostics'
        })}
        onPrimaryAction={onOpenHealth}
      />
    )
  }

  if (variant === 'error') {
    return (
      <div data-testid="notes-list-error-state">
        <FeatureEmptyState
          title={
            <span className="inline-flex items-center gap-2">
              <span className="rounded-full bg-danger/10 px-2 py-0.5 text-[11px] font-medium text-danger">
                Load error
              </span>
              <span>
                {t('option:notesSearch.loadErrorTitle', {
                  defaultValue: 'Could not load notes'
                })}
              </span>
            </span>
          }
          description={
            errorMessage ||
            t('option:notesSearch.loadErrorDescription', {
              defaultValue: 'The notes list failed to load. Retry the request or check server health.'
            })
          }
          examples={[
            t('option:notesSearch.loadErrorExampleRetry', {
              defaultValue: 'Retry keeps your current search and filters.'
            }),
            t('option:notesSearch.loadErrorExampleHealth', {
              defaultValue: 'Open diagnostics if the server keeps returning errors.'
            })
          ]}
          primaryActionLabel={t('option:buttonRetry', 'Retry')}
          onPrimaryAction={onRetry}
          secondaryActionLabel={t('settings:healthSummary.diagnostics', {
            defaultValue: 'Health & diagnostics'
          })}
          onSecondaryAction={onOpenHealth}
          icon={AlertTriangleIcon}
          iconClassName="text-danger"
        />
      </div>
    )
  }

  if (variant === 'no-results') {
    return (
      <div data-testid="notes-list-no-results-state">
        <FeatureEmptyState
          title={
            <span className="inline-flex items-center gap-2">
              <span className="rounded-full bg-surface2 px-2 py-0.5 text-[11px] font-medium text-text">
                No results
              </span>
              <span>
                {t('option:notesSearch.noResultsTitle', {
                  defaultValue: 'No notes match your filters'
                })}
              </span>
            </span>
          }
          description={
            normalizedSearchQuery
              ? t('option:notesSearch.noResultsDescriptionWithQuery', {
                  defaultValue: 'No notes match "{{query}}".'
                }).replace('{{query}}', normalizedSearchQuery)
              : t('option:notesSearch.noResultsDescription', {
                  defaultValue: 'No notes match the current tags or saved filter.'
                })
          }
          examples={[
            t('option:notesSearch.noResultsExampleSearch', {
              defaultValue: 'Try a broader phrase or remove exact-match quotes.'
            }),
            t('option:notesSearch.noResultsExampleTags', {
              defaultValue: 'Clear tag filters to return to all notes.'
            })
          ]}
          primaryActionLabel={t('option:notesSearch.clear', {
            defaultValue: 'Clear search & filters'
          })}
          onPrimaryAction={onClearFilters}
          secondaryActionLabel={t('option:notesSearch.emptyCreateSecondary', {
            defaultValue: 'Create note'
          })}
          onSecondaryAction={onResetEditor}
          icon={SearchXIcon}
        />
      </div>
    )
  }

  return (
    <FeatureEmptyState
      title={
        <span className="inline-flex items-center gap-2">
          <span className="rounded-full bg-surface2 px-2 py-0.5 text-[11px] font-medium text-text">
            Getting started
          </span>
          <span>
            {isTrashView
              ? t('option:notesSearch.emptyTrashTitle', {
                  defaultValue: 'Trash is empty'
                })
              : t('option:notesEmpty.title', { defaultValue: 'No notes yet' })}
          </span>
        </span>
      }
      description={isTrashView
        ? t('option:notesSearch.emptyTrashDescription', {
            defaultValue: 'Deleted notes will appear here until restored.'
          })
        : t('option:notesEmpty.description', {
            defaultValue:
              'Capture and organize free-form notes connected to your tldw insights.'
          })}
      examples={isTrashView
        ? [
            t('option:notesSearch.emptyTrashExample', {
              defaultValue:
                'Restore a note from trash to return it to your active notes list.'
            })
          ]
        : [
            t('option:notesEmpty.exampleCreate', {
              defaultValue:
                'Create a new note for a recent meeting or transcript.'
            }),
            t('option:notesEmpty.exampleLink', {
              defaultValue:
                'Save review outputs into Notes so you can revisit them later.'
            }),
            t('option:notesEmpty.exampleQuickSaveFromChat', {
              defaultValue:
                'You can also create notes directly from chat messages using quick save.'
            })
          ]}
      primaryActionLabel={isTrashView
        ? t('option:notesSearch.switchToActiveNotes', {
            defaultValue: 'Back to notes'
          })
        : t('option:notesEmpty.primaryCta', {
            defaultValue: 'Create note'
          })}
      onPrimaryAction={isTrashView ? onResetEditor : onCreateNote}
    />
  )
}

export default NotesListPanelEmptyStates
