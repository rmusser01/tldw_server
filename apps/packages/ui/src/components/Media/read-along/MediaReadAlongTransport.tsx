import React from 'react'
import {
  AlertCircle,
  Pause,
  Play,
  RefreshCw,
  SkipForward,
  Square
} from 'lucide-react'
import { Tooltip } from 'antd'

import type { ReadAlongSessionState } from './types'

type Translate = (key: string, opts?: Record<string, any>) => string

interface MediaReadAlongTransportProps {
  state: ReadAlongSessionState
  anchorRect: DOMRect | null
  onToggle: () => void
  onStop: () => void
  onRetry: () => void
  onSkip: () => void
  t: Translate
}

const visibleStatuses = new Set([
  'preparing',
  'playing',
  'paused',
  'segment-error'
])

export function MediaReadAlongTransport({
  state,
  anchorRect,
  onToggle,
  onStop,
  onRetry,
  onSkip,
  t
}: MediaReadAlongTransportProps) {
  if (!visibleStatuses.has(state.status)) return null

  const isPaused = state.status === 'paused'
  const isPreparing = state.status === 'preparing'
  const hasError = state.status === 'segment-error'
  const progress =
    state.totalSegments > 0 && state.activeIndex >= 0
      ? `${state.activeIndex + 1}/${state.totalSegments}`
      : isPreparing
        ? t('review:mediaPage.readAlongPreparing', { defaultValue: 'Preparing' })
        : '0/0'
  const viewportWidth = typeof window !== 'undefined' ? window.innerWidth : 1024

  return (
    <div
      className="fixed z-40 inline-flex max-w-[calc(100vw-16px)] items-center gap-1 rounded border border-border bg-surface px-2 py-1 text-xs text-text shadow-lg"
      style={{
        left: Math.max(8, Math.min(anchorRect?.left ?? 16, viewportWidth - 260)),
        top: Math.max(8, (anchorRect?.bottom ?? 88) + 8)
      }}
      data-testid="media-read-along-transport"
      role="group"
      aria-label={t('review:mediaPage.readAlongTransport', {
        defaultValue: 'Read-along controls'
      })}
    >
      <Tooltip
        title={
          isPaused
            ? t('review:mediaPage.readAlongResume', { defaultValue: 'Resume' })
            : t('review:mediaPage.readAlongPause', { defaultValue: 'Pause' })
        }
      >
        <button
          type="button"
          className="inline-flex h-7 w-7 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
          onClick={onToggle}
          disabled={isPreparing || hasError}
          aria-label={
            isPaused
              ? t('review:mediaPage.readAlongResume', { defaultValue: 'Resume' })
              : t('review:mediaPage.readAlongPause', { defaultValue: 'Pause' })
          }
          data-testid="media-read-along-toggle"
        >
          {isPaused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
        </button>
      </Tooltip>
      <Tooltip title={t('review:mediaPage.readAlongStop', { defaultValue: 'Stop' })}>
        <button
          type="button"
          className="inline-flex h-7 w-7 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text"
          onClick={onStop}
          aria-label={t('review:mediaPage.readAlongStop', { defaultValue: 'Stop' })}
          data-testid="media-read-along-stop"
        >
          <Square className="h-3.5 w-3.5" />
        </button>
      </Tooltip>
      <span
        className="min-w-[44px] rounded bg-surface2 px-1.5 py-0.5 text-center tabular-nums text-text-muted"
        data-testid="media-read-along-progress"
      >
        {progress}
      </span>
      <Tooltip title={t('review:mediaPage.readAlongRetry', { defaultValue: 'Retry' })}>
        <button
          type="button"
          className="inline-flex h-7 w-7 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
          onClick={onRetry}
          disabled={!hasError}
          aria-label={t('review:mediaPage.readAlongRetry', { defaultValue: 'Retry' })}
          data-testid="media-read-along-retry"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </Tooltip>
      <Tooltip title={t('review:mediaPage.readAlongSkip', { defaultValue: 'Skip' })}>
        <button
          type="button"
          className="inline-flex h-7 w-7 items-center justify-center rounded text-text-muted hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
          onClick={onSkip}
          disabled={!hasError}
          aria-label={t('review:mediaPage.readAlongSkip', { defaultValue: 'Skip' })}
          data-testid="media-read-along-skip"
        >
          <SkipForward className="h-3.5 w-3.5" />
        </button>
      </Tooltip>
      {hasError ? (
        <span className="inline-flex max-w-[180px] items-center gap-1 truncate text-warn">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate">{state.error || 'Playback failed'}</span>
        </span>
      ) : null}
    </div>
  )
}
