import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useStorage } from '@plasmohq/storage/hook'
import { ChevronDown, Loader2, RefreshCw } from 'lucide-react'
import { tldwClient } from '@/services/tldw/TldwApiClient'
import { formatRelativeTime } from '@/utils/dateFormatters'
import { requestQuickIngestOpen } from '@/utils/quick-ingest-open'

type MediaIngestJobStatus = {
  id: number
  status: string
  source?: string | null
  source_kind?: string | null
  progress_percent?: number | null
  progress_message?: string | null
  error_message?: string | null
  created_at?: string | null
  completed_at?: string | null
}

const MEDIA_INGEST_PANEL_COLLAPSED_KEY = 'media:ingest:panelCollapsed'
const MEDIA_INGEST_LAST_BATCH_ID_KEY = 'media:ingest:lastBatchId'
const MEDIA_INGEST_AUTO_REFRESH_KEY = 'media:ingest:autoRefresh'
const MEDIA_INGEST_POLL_INTERVAL_MS = 8000

const statusToneClass = (status: string | null | undefined): string => {
  const normalized = String(status || '')
    .toLowerCase()
    .trim()
  if (normalized === 'completed' || normalized === 'succeeded') {
    return 'bg-success/10 text-success'
  }
  if (
    normalized === 'running' ||
    normalized === 'processing' ||
    normalized === 'started' ||
    normalized === 'queued'
  ) {
    return 'bg-primary/10 text-primaryStrong'
  }
  if (normalized === 'failed' || normalized === 'error' || normalized === 'cancelled') {
    return 'bg-danger/10 text-danger'
  }
  return 'bg-surface2 text-text-muted'
}

export function MediaIngestJobsPanel() {
  const { t } = useTranslation(['review'])
  const [collapsed, setCollapsed] = useStorage<boolean>(
    MEDIA_INGEST_PANEL_COLLAPSED_KEY,
    true
  )
  const [savedBatchId, setSavedBatchId] = useStorage<string>(
    MEDIA_INGEST_LAST_BATCH_ID_KEY,
    ''
  )
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useStorage<boolean>(
    MEDIA_INGEST_AUTO_REFRESH_KEY,
    true
  )
  const [batchDraft, setBatchDraft] = useState(savedBatchId || '')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [jobs, setJobs] = useState<MediaIngestJobStatus[]>([])
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null)
  const requestSequenceRef = useRef(0)
  const panelCollapsed = collapsed === true
  const persistedBatchId = String(savedBatchId || '').trim()

  useEffect(() => {
    setBatchDraft(String(savedBatchId || ''))
  }, [savedBatchId])

  // Auto-link batch ID and expand panel when Quick Ingest completes
  useEffect(() => {
    const handleIngestComplete = (e: Event) => {
      const detail = (e as CustomEvent)?.detail
      const batchId = detail?.batchId
      if (batchId) {
        setBatchDraft(batchId)
        void setSavedBatchId(batchId)
        void setCollapsed(false)
      }
    }
    window.addEventListener("tldw:quick-ingest-complete", handleIngestComplete)
    return () => window.removeEventListener("tldw:quick-ingest-complete", handleIngestComplete)
  }, [setSavedBatchId, setCollapsed])

  const summary = useMemo(() => {
    return jobs.reduce(
      (acc, job) => {
        const status = String(job.status || '')
          .toLowerCase()
          .trim()
        if (status === 'completed' || status === 'succeeded') {
          acc.completed += 1
          return acc
        }
        if (
          status === 'running' ||
          status === 'started' ||
          status === 'queued' ||
          status === 'processing'
        ) {
          acc.active += 1
          return acc
        }
        if (status === 'failed' || status === 'error' || status === 'cancelled') {
          acc.failed += 1
          return acc
        }
        acc.other += 1
        return acc
      },
      { completed: 0, active: 0, failed: 0, other: 0 }
    )
  }, [jobs])

  const loadJobs = useCallback(async () => {
    if (!persistedBatchId) {
      setJobs([])
      setError(null)
      setLastUpdatedAt(null)
      setLoading(false)
      return
    }

    const requestSequence = ++requestSequenceRef.current
    setLoading(true)
    setError(null)

    try {
      const response = await tldwClient.listMediaIngestJobs({
        batch_id: persistedBatchId,
        limit: 50
      })
      if (requestSequence !== requestSequenceRef.current) return
      setJobs(Array.isArray(response?.jobs) ? response.jobs : [])
      setLastUpdatedAt(new Date().toISOString())
    } catch (err: any) {
      if (requestSequence !== requestSequenceRef.current) return
      setError('Unable to load ingest jobs. Check the batch ID and try again.')
      setJobs([])
    } finally {
      if (requestSequence === requestSequenceRef.current) {
        setLoading(false)
      }
    }
  }, [persistedBatchId])

  useEffect(() => {
    if (panelCollapsed) return
    void loadJobs()
  }, [loadJobs, panelCollapsed])

  useEffect(() => {
    if (panelCollapsed || !persistedBatchId || autoRefreshEnabled === false) return
    const intervalId = window.setInterval(() => {
      void loadJobs()
    }, MEDIA_INGEST_POLL_INTERVAL_MS)
    return () => window.clearInterval(intervalId)
  }, [autoRefreshEnabled, loadJobs, panelCollapsed, persistedBatchId])

  const applyBatchId = useCallback(() => {
    const normalizedBatchId = batchDraft.trim()
    void setSavedBatchId(normalizedBatchId)
  }, [batchDraft, setSavedBatchId])

  return (
    <div className="border-b border-border px-4 py-3">
      <button
        type="button"
        onClick={() => setCollapsed(!panelCollapsed)}
        className="flex w-full items-center justify-between text-sm text-text hover:text-text"
        aria-expanded={!panelCollapsed}
        aria-controls="media-ingest-jobs-panel"
        data-testid="media-ingest-jobs-toggle"
      >
        <span>{t('review:mediaPage.ingestJobsTitle', { defaultValue: 'Ingest jobs' })}</span>
        <ChevronDown
          className={`h-4 w-4 transition-transform ${panelCollapsed ? '' : 'rotate-180'}`}
        />
      </button>

      {!panelCollapsed && (
        <div id="media-ingest-jobs-panel" className="mt-3 space-y-3" data-testid="media-ingest-jobs-panel">
          <div className="space-y-2">
            <label className="block text-xs text-text-muted">
              {t('review:mediaPage.ingestBatchId', { defaultValue: 'Batch ID' })}
            </label>
            <div className="flex items-center gap-2">
              <input
                value={batchDraft}
                onChange={(event) => setBatchDraft(event.target.value)}
                placeholder={t('review:mediaPage.ingestBatchIdPlaceholder', {
                  defaultValue: 'Paste batch ID from ingest response'
                })}
                className="h-8 flex-1 rounded-md border border-border bg-surface px-2 text-xs text-text"
                data-testid="media-ingest-batch-input"
              />
              <button
                type="button"
                onClick={applyBatchId}
                disabled={!batchDraft.trim()}
                className="h-8 rounded-md border border-border px-2 text-xs text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-60"
                data-testid="media-ingest-batch-apply"
              >
                {t('review:mediaPage.applyBatch', { defaultValue: 'Apply' })}
              </button>
              <button
                type="button"
                onClick={() => void loadJobs()}
                className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-border text-text hover:bg-surface2"
                aria-label={t('review:mediaPage.refreshIngestJobs', { defaultValue: 'Refresh ingest jobs' })}
                title={t('review:mediaPage.refreshIngestJobs', { defaultValue: 'Refresh ingest jobs' })}
                data-testid="media-ingest-jobs-refresh"
              >
                <RefreshCw className="h-3.5 w-3.5" />
              </button>
            </div>
            <label className="inline-flex items-center gap-2 text-xs text-text-muted">
              <input
                type="checkbox"
                checked={autoRefreshEnabled !== false}
                onChange={(event) => setAutoRefreshEnabled(event.target.checked)}
                className="h-3.5 w-3.5 rounded border-border bg-surface"
                data-testid="media-ingest-auto-refresh"
              />
              <span>
                {t('review:mediaPage.autoRefreshJobs', {
                  defaultValue: 'Auto refresh every 8s'
                })}
              </span>
            </label>
          </div>

          {!persistedBatchId ? (
            <p className="text-xs text-text-muted" data-testid="media-ingest-jobs-empty-batch">
              {t('review:mediaPage.ingestJobsHint', {
                defaultValue: 'Enter a batch ID to track ingestion progress.'
              })}
            </p>
          ) : (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-text-muted">
                <span>{t('review:mediaPage.totalJobs', { defaultValue: 'Jobs: {{count}}', count: jobs.length })}</span>
                <span>
                  {t('review:mediaPage.activeJobs', {
                    defaultValue: 'Active: {{count}}',
                    count: summary.active
                  })}
                </span>
                <span>
                  {t('review:mediaPage.completedJobs', {
                    defaultValue: 'Completed: {{count}}',
                    count: summary.completed
                  })}
                </span>
                <span>
                  {t('review:mediaPage.failedJobs', {
                    defaultValue: 'Failed: {{count}}',
                    count: summary.failed
                  })}
                </span>
                {summary.other > 0 && (
                  <span>
                    {t('review:mediaPage.otherJobs', {
                      defaultValue: 'Other: {{count}}',
                      count: summary.other
                    })}
                  </span>
                )}
                {lastUpdatedAt && (
                  <span data-testid="media-ingest-jobs-updated">
                    {t('review:mediaPage.lastUpdated', {
                      defaultValue: 'Updated {{time}}',
                      time: formatRelativeTime(lastUpdatedAt, t, { compact: true })
                    })}
                  </span>
                )}
              </div>

              {loading && (
                <div
                  className="inline-flex items-center gap-2 text-xs text-text-muted"
                  data-testid="media-ingest-jobs-loading"
                >
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>
                    {t('review:mediaPage.loadingIngestJobs', {
                      defaultValue: 'Loading ingest jobs...'
                    })}
                  </span>
                </div>
              )}

              {error && (
                <div
                  className="rounded-md border border-danger/30 bg-danger/5 px-2 py-2 text-xs text-danger"
                  data-testid="media-ingest-jobs-error"
                >
                  <p>{error}</p>
                  <button
                    type="button"
                    onClick={() => void loadJobs()}
                    className="mt-2 rounded-md border border-danger/40 px-2 py-1 text-[11px] hover:bg-danger/10"
                    data-testid="media-ingest-jobs-retry"
                  >
                    {t('review:mediaPage.retryIngestJobs', { defaultValue: 'Retry' })}
                  </button>
                </div>
              )}

              {!loading && !error && jobs.length === 0 && (
                <p className="text-xs text-text-muted" data-testid="media-ingest-jobs-empty">
                  {t('review:mediaPage.ingestJobsEmpty', {
                    defaultValue: 'No jobs found for this batch.'
                  })}
                </p>
              )}

              {jobs.length > 0 && (
                <ul className="max-h-52 space-y-2 overflow-y-auto" data-testid="media-ingest-jobs-list">
                  {jobs.map((job) => (
                    <li
                      key={job.id}
                      className="rounded-md border border-border bg-surface px-2 py-2"
                      data-testid={`media-ingest-job-row-${job.id}`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <p className="truncate text-xs font-medium text-text">
                            {job.source || t('review:mediaPage.unknownSource', { defaultValue: 'Unknown source' })}
                          </p>
                          <p className="text-[11px] text-text-muted">
                            {t('review:mediaPage.jobIdLabel', {
                              defaultValue: 'Job #{{id}}',
                              id: job.id
                            })}
                            {job.source_kind
                              ? ` • ${String(job.source_kind).toUpperCase()}`
                              : ''}
                          </p>
                        </div>
                        <span
                          className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium ${statusToneClass(job.status)}`}
                          data-testid={`media-ingest-job-status-${job.id}`}
                        >
                          {job.status || 'unknown'}
                        </span>
                      </div>

                      {(typeof job.progress_percent === 'number' || job.progress_message) && (
                        <p className="mt-1 text-[11px] text-text-muted">
                          {typeof job.progress_percent === 'number'
                            ? `${Math.max(0, Math.min(100, Math.round(job.progress_percent)))}%`
                            : null}
                          {job.progress_message
                            ? `${typeof job.progress_percent === 'number' ? ' • ' : ''}${job.progress_message}`
                            : null}
                        </p>
                      )}

                      {job.error_message && (
                        <p className="mt-1 text-[11px] text-danger">{job.error_message}</p>
                      )}
                      {/* Retry button for failed/error/cancelled jobs */}
                      {(() => {
                        const s = String(job.status || '').toLowerCase().trim()
                        if (s !== 'failed' && s !== 'error' && s !== 'cancelled') return null
                        return (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              if (job.source) {
                                requestQuickIngestOpen({
                                  source: job.source,
                                  sourceKind: job.source_kind,
                                })
                              }
                            }}
                            className="mt-1.5 inline-flex items-center gap-1 rounded-md border border-danger/40 px-2 py-0.5 text-[11px] text-danger hover:bg-danger/10 transition-colors"
                            data-testid={`media-ingest-job-retry-${job.id}`}
                          >
                            <RefreshCw className="h-3 w-3" />
                            {t('review:mediaPage.retryJob', { defaultValue: 'Retry' })}
                          </button>
                        )
                      })()}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
