import React from "react"
import type { TFunction } from "i18next"
import { formatRelativeTime } from "@/utils/dateFormatters"
import type { Result } from "./ResultsList"

/**
 * Inspector-tooltip construction for Media ResultsList rows.
 *
 * Extracted from ResultsList (WP-11) so the memoized row component can build
 * its tooltip inside a `useMemo` (per `result` identity) and tests can spy on
 * the builder through a cross-module import.
 */

/** Normalize processing status string to a display category. */
export const normalizeProcessingStatus = (status: any): 'complete' | 'processing' | 'failed' | 'unknown' => {
  if (status == null) return 'unknown'
  const s = String(status).toLowerCase().trim()
  if (s === 'completed' || s === 'succeeded' || s === 'done' || s === 'ready') return 'complete'
  if (s === 'running' || s === 'processing' || s === 'started' || s === 'queued' || s === 'pending' || s === 'in_progress') return 'processing'
  if (s === 'failed' || s === 'error' || s === 'cancelled') return 'failed'
  return 'unknown'
}

/** Format duration seconds to human readable. */
export const formatDuration = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

/** Format publication date to short display. */
export const formatPublishedDate = (dateStr: string): string => {
  try {
    const d = new Date(dateStr)
    if (Number.isNaN(d.getTime())) return dateStr
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short' })
  } catch {
    return dateStr
  }
}

export const buildInspectorTooltip = (
  result: Result,
  t: TFunction,
  readingProgress?: Map<string, number>
) => {
  const title = result.title || `${result.kind} ${result.id}`
  const lines: Array<{
    label: string
    value: string
    multiline?: boolean
    preserveWhitespace?: boolean
  }> = [
    {
      label: t('mediaPage.titleLabel', 'Title'),
      value: title
    },
    {
      label: t('mediaPage.typeLabel', 'Type'),
      value: result.meta?.type || result.kind
    }
  ]
  if (result.meta?.author) {
    lines.push({
      label: t('mediaPage.authorLabel', 'Author'),
      value: result.meta.author
    })
  }
  if (result.meta?.source) {
    lines.push({
      label: t('mediaPage.source', 'Source'),
      value: result.meta.source
    })
  }
  if (result.meta?.published_at) {
    lines.push({
      label: t('mediaPage.publishedLabel', 'Published'),
      value: formatPublishedDate(result.meta.published_at)
    })
  }
  if (result.meta?.created_at) {
    lines.push({
      label: t('mediaPage.ingested', 'Ingested'),
      value: formatRelativeTime(result.meta.created_at, t, { compact: true })
    })
  }
  if (result.meta?.duration != null && result.meta.duration > 0) {
    lines.push({
      label: t('mediaPage.durationLabel', 'Duration'),
      value: formatDuration(result.meta.duration)
    })
  }
  if (result.meta?.page_count != null) {
    lines.push({
      label: t('mediaPage.pagesLabel', 'Pages'),
      value: String(result.meta.page_count)
    })
  }
  if (result.meta?.transcription_model) {
    lines.push({
      label: t('mediaPage.transcriptionModelLabel', 'Transcription'),
      value: result.meta.transcription_model
    })
  }
  {
    const progress = readingProgress?.get(String(result.id))
    if (progress != null && progress > 0) {
      lines.push({
        label: t('mediaPage.readingProgressLabel', 'Read'),
        value: `${Math.round(progress)}%`
      })
    }
  }
  if (result.snippet) {
    lines.push({
      label: t('mediaPage.previewLabel', 'Preview'),
      value: result.snippet,
      multiline: true,
      preserveWhitespace: true
    })
  }
  if (Array.isArray(result.keywords) && result.keywords.length > 0) {
    lines.push({
      label: t('mediaPage.keywords', 'Keywords'),
      value: result.keywords.join(', '),
      multiline: true
    })
  }

  return (
    <div className="space-y-1 text-xs max-w-xs">
      {lines.map((line, index) => (
        <div
          key={`${line.label}-${index}`}
          className={line.multiline ? "flex flex-col gap-1" : "flex gap-1"}
        >
          <span className="font-medium text-text">{line.label}:</span>
          <span
            className={
              line.multiline
                ? `text-text-subtle break-words line-clamp-4 ${
                    line.preserveWhitespace ? "whitespace-pre-wrap" : "whitespace-normal"
                  }`
                : "text-text-subtle break-words"
            }
          >
            {line.value}
          </span>
        </div>
      ))}
    </div>
  )
}
