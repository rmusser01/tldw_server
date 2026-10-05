import { setSetting } from "@/services/settings/registry"
import {
  MEDIA_REVIEW_SELECTION_SETTING,
  MEDIA_REVIEW_SELECTION_SNAPSHOT_SETTING
} from "@/services/settings/ui-settings"
import { tldwClient } from '@/services/tldw/TldwApiClient'
import {
  quickIngestAuthority,
  useQuickIngestAuthority
} from "@/services/tldw/quick-ingest-authority"
import {
  type RecentImport,
  recentImportSourceLabel,
  useQuickIngestSessionStore
} from "@/store/quick-ingest-session"
import { formatRelativeTime } from '@/utils/dateFormatters'
import { ChevronDown, Loader2, RefreshCw } from 'lucide-react'
import React, { useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from 'react-i18next'
import { useNavigate } from "react-router-dom"

import { useStorage } from '@plasmohq/storage/hook'

type MediaIngestJobStatus = {
  id: number
  status: string
  source?: string | null
  progress_percent?: number | null
  progress_message?: string | null
  error_message?: string | null
  result?: { media_id?: number | string; persisted?: boolean }
}
const activeJob = (status: string) =>
  ["running", "processing", "started", "queued"].includes(status.toLowerCase())

export function MediaIngestJobsPanel() {
  const { t } = useTranslation(["review"])
  const navigate = useNavigate()
  const authorityKey = useQuickIngestAuthority()
  const recentImports = useQuickIngestSessionStore(
    (state) => state.recentImports
  )
  const session = useQuickIngestSessionStore((state) => state.session)
  const [collapsed, setCollapsed] = useStorage<boolean>(
    "media:ingest:panelCollapsed",
    true
  )
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useStorage<boolean>(
    "media:ingest:autoRefresh",
    true
  )
  // Raw IDs are explicit, temporary diagnostics. Old unowned persisted IDs cannot start requests.
  const [batchDraft, setBatchDraft] = useState("")
  const [diagnosticBatch, setDiagnosticBatch] = useState("")
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [jobs, setJobs] = useState<MediaIngestJobStatus[]>([])
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null)
  const sequence = useRef(0)
  const imports = authorityKey
    ? recentImports.filter((item) => item.authorityKey === authorityKey)
    : []
  const selected = imports.find((item) => item.id === selectedId)
  const batchIds = selected?.batchIds.join("\n") || diagnosticBatch
  const panelCollapsed = collapsed === true
  const qi = useCallback(
    (key: string, defaultValue: string, options?: Record<string, unknown>) =>
      t(`review:mediaPage.${key}`, { defaultValue, ...options }),
    [t]
  )

  useEffect(() => {
    sequence.current += 1
    setSelectedId(null)
    setDiagnosticBatch("")
    setBatchDraft("")
    setJobs([])
    setError(null)
    setLoading(false)
    setLastUpdatedAt(null)
  }, [authorityKey])

  const loadError = qi(
    "ingestJobsLoadError",
    "Unable to refresh this import. Try again."
  )
  const loadJobs = useCallback(async () => {
    if (!authorityKey || !batchIds) return
    const request = ++sequence.current
    let operation: ReturnType<typeof quickIngestAuthority.capture>
    try {
      operation = quickIngestAuthority.capture({ sessionBound: false })
    } catch {
      return
    }
    const current = () => request === sequence.current && operation.isCurrent()
    setLoading(true)
    setError(null)
    try {
      const responses = await Promise.all(
        batchIds.split("\n").map(async (batch_id) => {
          const collected: MediaIngestJobStatus[] = []
          let offset: number | undefined
          while (current()) {
            const response = await tldwClient.listMediaIngestJobs(
              { batch_id, limit: 50, ...(offset ? { offset } : {}) },
              {
                signal: operation.signal,
                requestScope: operation.requestScope
              }
            )
            if (!current()) return { jobs: [] }
            collected.push(
              ...(Array.isArray(response?.jobs) ? response.jobs : [])
            )
            const nextOffset =
              response?.next_offset ?? response?.pagination?.next_offset
            if (!(response?.has_more ?? response?.pagination?.has_more)) break
            if (
              !Number.isSafeInteger(nextOffset) ||
              nextOffset <= (offset || 0)
            )
              throw new Error("Invalid import pagination")
            offset = nextOffset
          }
          return { jobs: collected }
        })
      )
      if (!current()) return
      const next: MediaIngestJobStatus[] = Array.from(
        new Map(
          responses
            .flatMap((response) =>
              Array.isArray(response?.jobs) ? response.jobs : []
            )
            .map((job) => [job.id, job])
        ).values()
      )
      setJobs(next)
      setLastUpdatedAt(new Date().toISOString())
      if (selectedId && next.length) {
        const anyActive = next.some(
          (job) =>
            activeJob(job.status) ||
            ![
              "completed",
              "succeeded",
              "failed",
              "error",
              "cancelled"
            ].includes(job.status.toLowerCase())
        )
        const failed = next.some((job) =>
          ["failed", "error"].includes(job.status.toLowerCase())
        )
        const cancelled = next.every(
          (job) => job.status.toLowerCase() === "cancelled"
        )
        useQuickIngestSessionStore.getState().updateRecentImport(selectedId, {
          lifecycle: anyActive
            ? "processing"
            : failed
              ? "partial_failure"
              : cancelled
                ? "cancelled"
                : "completed",
          savedMediaIds: next
            .filter(
              (job) =>
                ["completed", "succeeded"].includes(job.status.toLowerCase()) &&
                job.result?.persisted !== false
            )
            .map((job) => Number(job.result?.media_id))
            .filter((id) => Number.isSafeInteger(id) && id > 0)
        })
      }
    } catch {
      if (current()) {
        setJobs([])
        setError(loadError)
      }
    } finally {
      if (current()) setLoading(false)
    }
  }, [authorityKey, batchIds, selectedId, loadError])

  useEffect(() => {
    if (!panelCollapsed) void loadJobs()
    return () => {
      sequence.current += 1
    }
  }, [loadJobs, panelCollapsed])
  useEffect(() => {
    if (
      panelCollapsed ||
      !batchIds ||
      autoRefreshEnabled === false ||
      !(
        jobs.some((job) => activeJob(job.status)) ||
        (!jobs.length && selected?.lifecycle === "processing")
      )
    )
      return
    const timer = window.setInterval(() => {
      void loadJobs()
    }, 8000)
    return () => window.clearInterval(timer)
  }, [
    panelCollapsed,
    batchIds,
    autoRefreshEnabled,
    jobs,
    selected?.lifecycle,
    loadJobs
  ])

  const reviewSaved = async (item: RecentImport) => {
    if (item.authorityKey !== authorityKey || !item.savedMediaIds.length) return
    let operation: ReturnType<typeof quickIngestAuthority.capture>
    try {
      operation = quickIngestAuthority.capture({ sessionBound: false })
    } catch {
      return
    }
    setError(null)
    try {
      await setSetting(MEDIA_REVIEW_SELECTION_SNAPSHOT_SETTING, {
        version: 1,
        authorityKey: operation.authorityKey,
        selectedIds: item.savedMediaIds
      })
      if (!operation.isCurrent()) return
      try {
        await setSetting(MEDIA_REVIEW_SELECTION_SETTING, item.savedMediaIds)
      } catch {
        console.warn(
          "[media-review] Compatibility selection mirror could not be written."
        )
      }
      if (operation.isCurrent()) navigate("/media-multi")
    } catch {
      if (operation.isCurrent())
        setError(
          qi(
            "ingestReviewError",
            "Unable to open the saved review set. Try again."
          )
        )
    }
  }
  const actionClass =
    "min-h-10 rounded-md border border-border px-3 py-2 text-xs text-text hover:bg-surface2 disabled:opacity-50"
  return (
    <div className="border-b border-border px-4 py-3">
      <button
        type="button"
        onClick={() => setCollapsed(!panelCollapsed)}
        className="flex min-h-10 w-full items-center justify-between text-sm text-text"
        aria-expanded={!panelCollapsed}
        aria-controls="media-ingest-jobs-panel"
        data-testid="media-ingest-jobs-toggle">
        <span>
          {qi("recentImportsTitle", "Recent imports")}
          {imports.length ? ` (${imports.length})` : ""}
        </span>
        <ChevronDown
          className={`h-4 w-4 ${panelCollapsed ? "" : "rotate-180"}`}
        />
      </button>
      {!panelCollapsed && (
        <div id="media-ingest-jobs-panel" className="mt-3 space-y-3" data-testid="media-ingest-jobs-panel">
          {!imports.length && (
            <p className="text-xs text-text-muted" data-testid="media-ingest-jobs-empty-batch">
              {qi(
                "recentImportsHint",
                "Imports appear here when processing starts. Add media to begin."
              )}
            </p>
          )}
          <ul className="space-y-2">
            {imports.map((item) => (
              <li
                key={item.id}
                className="rounded-md border border-border bg-surface px-3 py-2">
                <p className="truncate text-sm font-medium text-text">
                  {item.sourceLabel}
                </p>
                <p className="text-xs text-text-muted">
                  {qi("importSourceCount", "{{count}} sources", {
                    count: item.sourceCount
                  })}{" "}
                  ·{" "}
                  {qi(
                    `importState.${item.lifecycle}`,
                    item.lifecycle.replace(/_/g, " ")
                  )}{" "}
                  ·{" "}
                  {formatRelativeTime(
                    new Date(item.updatedAt).toISOString(),
                    t,
                    { compact: true }
                  )}
                </p>
                {item.id === session?.id &&
                  session.authorityKey === authorityKey && (
                    <p className="text-xs text-text-muted" role="status">
                      {qi(
                        "importLiveProgress",
                        "{{done}} processing outcomes received",
                        {
                          done: session.results.length,
                          count: item.sourceCount
                        }
                      )}
                    </p>
                  )}
                <div className="mt-2 flex flex-wrap gap-2">
                  {item.id === session?.id &&
                    session.authorityKey === authorityKey && (
                      <button
                        className={actionClass}
                        onClick={() =>
                          useQuickIngestSessionStore.getState().showSession()
                        }>
                        {qi("resumeImport", "Resume import")}
                      </button>
                    )}
                  {!!item.batchIds.length && (
                    <button
                      className={actionClass}
                      onClick={() => {
                        setDiagnosticBatch("")
                        if (selectedId === item.id) void loadJobs()
                        else setSelectedId(item.id)
                      }}>
                      <RefreshCw className="mr-1 inline h-3 w-3" />
                      {qi("refreshImport", "Refresh import")}
                    </button>
                  )}
                  {!!item.savedMediaIds.length && (
                    <button
                      className={actionClass}
                      onClick={() => void reviewSaved(item)}>
                      {qi("reviewImportSaved", "Review {{count}} saved items", {
                        count: item.savedMediaIds.length
                      })}
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
          <details>
            <summary className="cursor-pointer text-xs text-text-muted">
              {qi("importDiagnostics", "Optional diagnostics")}
            </summary>
            <div className="mt-2 flex gap-2">
              <label className="min-w-0 flex-1 text-xs">
                {qi("ingestBatchId", "Batch ID")}
                <input
                  value={batchDraft}
                  onChange={(event) => setBatchDraft(event.target.value)}
                  className="mt-1 h-10 w-full rounded border border-border bg-surface px-2"
                  data-testid="media-ingest-batch-input"
                />
              </label>
              <button
                disabled={!authorityKey || !batchDraft.trim()}
                className={actionClass}
                data-testid="media-ingest-batch-apply"
                onClick={() => {
                  setSelectedId(null)
                  setDiagnosticBatch(batchDraft.trim())
                }}>
                {qi("applyBatch", "Apply")}
              </button>
            </div>
          </details>
          {!!batchIds && (
            <div className="space-y-2">
              <label className="inline-flex min-h-10 items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={autoRefreshEnabled !== false}
                  onChange={(event) => setAutoRefreshEnabled(event.target.checked)}
                  data-testid="media-ingest-auto-refresh"
                />
                {qi("autoRefreshJobs", "Auto refresh every 8s")}
              </label>
              {lastUpdatedAt && (
                <p
                  className="text-xs text-text-muted"
                  data-testid="media-ingest-jobs-updated">
                  {qi("lastUpdated", "Updated {{time}}", {
                    time: formatRelativeTime(lastUpdatedAt, t, { compact: true })
                  })}
                </p>
              )}
              {loading && (
                <p className="text-xs" data-testid="media-ingest-jobs-loading">
                  <Loader2 className="mr-1 inline h-3 w-3 animate-spin" />
                  {qi("loadingIngestJobs", "Loading ingest jobs...")}
                </p>
              )}
              {!loading && !error && !jobs.length && (
                <p className="text-xs text-text-muted" data-testid="media-ingest-jobs-empty">
                  {qi("ingestJobsEmpty", "No jobs found for this batch.")}
                </p>
              )}
              {!!jobs.length && (
                <ul className="max-h-52 space-y-2 overflow-y-auto" data-testid="media-ingest-jobs-list">
                  {jobs.map((job) => (
                    <li
                      key={job.id}
                      className="rounded border border-border p-2 text-xs"
                      data-testid={`media-ingest-job-row-${job.id}`}>
                      <p>
                        {recentImportSourceLabel({
                          url: job.source?.startsWith("http")
                            ? job.source
                            : undefined,
                          fileName: job.source || undefined
                        })}{" "}
                        ·{" "}
                        <span data-testid={`media-ingest-job-status-${job.id}`}>
                          {job.status}
                        </span>
                      </p>
                      {(typeof job.progress_percent === 'number' || job.progress_message) && (
                        <p>
                          {typeof job.progress_percent === "number"
                            ? `${Math.max(0, Math.min(100, Math.round(job.progress_percent)))}%`
                            : ""}
                          {job.progress_message
                            ? ` • ${job.progress_message}`
                            : ""}
                        </p>
                      )}
                      {job.error_message && (
                        <p className="text-danger">{job.error_message}</p>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
          {error && (
            <div
              role="alert"
              className="rounded border border-danger/30 p-2 text-xs text-danger"
              data-testid="media-ingest-jobs-error">
              <p>{error}</p>
              {!!batchIds && (
                <button
                  className={actionClass}
                  onClick={() => void loadJobs()}
                  data-testid="media-ingest-jobs-retry">
                  {qi("retryIngestJobs", "Retry")}
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
