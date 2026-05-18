import React from 'react'
import { FileText, List, SkipForward, StickyNote, Volume2 } from 'lucide-react'
import { Tooltip } from 'antd'

import type { ReadAlongScope } from './types'

type Translate = (key: string, opts?: Record<string, any>) => string

interface MediaReadAlongPopoverProps {
  anchorRect: DOMRect
  viewportRect?: DOMRect | null
  supportedScopes?: ReadAlongScope[]
  onReadScope: (scope: ReadAlongScope) => void
  onAnnotate: () => void
  t: Translate
}

interface PopoverAction {
  scope?: ReadAlongScope
  testId: string
  label: string
  title: string
  icon: React.ReactNode
  onClick?: () => void
}

const EDGE_MARGIN_PX = 8
const ESTIMATED_POPOVER_WIDTH_PX = 560
const ESTIMATED_POPOVER_HEIGHT_PX = 40

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(value, max))

const getViewportRect = (viewportRect?: DOMRect | null): DOMRect => {
  const width = typeof window !== 'undefined' ? window.innerWidth : 1024
  const height = typeof window !== 'undefined' ? window.innerHeight : 768
  const windowViewport = {
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    width,
    height,
    right: width,
    bottom: height,
    toJSON: () => ({})
  } as DOMRect
  if (!viewportRect) return windowViewport

  const left = Math.max(windowViewport.left, viewportRect.left)
  const top = Math.max(windowViewport.top, viewportRect.top)
  const right = Math.min(windowViewport.right, viewportRect.right)
  const bottom = Math.min(windowViewport.bottom, viewportRect.bottom)
  if (right - left <= EDGE_MARGIN_PX * 2 || bottom - top <= EDGE_MARGIN_PX * 2) {
    return windowViewport
  }

  return {
    x: left,
    y: top,
    left,
    top,
    width: right - left,
    height: bottom - top,
    right,
    bottom,
    toJSON: () => ({})
  } as DOMRect
}

const getPopoverPosition = (
  anchorRect: DOMRect,
  viewportRect?: DOMRect | null
): { left: number; maxWidth: number; top: number } => {
  const viewport = getViewportRect(viewportRect)
  const viewportWidth = Math.max(0, viewport.right - viewport.left)
  const viewportHeight = Math.max(0, viewport.bottom - viewport.top)
  const popoverWidth = Math.min(
    ESTIMATED_POPOVER_WIDTH_PX,
    Math.max(0, viewportWidth - EDGE_MARGIN_PX * 2)
  )
  const popoverHeight = Math.min(
    ESTIMATED_POPOVER_HEIGHT_PX,
    Math.max(0, viewportHeight - EDGE_MARGIN_PX * 2)
  )
  const minLeft = viewport.left + EDGE_MARGIN_PX
  const maxLeft = Math.max(minLeft, viewport.right - popoverWidth - EDGE_MARGIN_PX)
  const minTop = viewport.top + EDGE_MARGIN_PX
  const maxTop = Math.max(minTop, viewport.bottom - popoverHeight - EDGE_MARGIN_PX)

  const left = clamp(anchorRect.left, minLeft, maxLeft)

  return {
    left,
    maxWidth: Math.max(0, viewport.right - left - EDGE_MARGIN_PX),
    top: clamp(anchorRect.top - ESTIMATED_POPOVER_HEIGHT_PX, minTop, maxTop)
  }
}

export function MediaReadAlongPopover({
  anchorRect,
  viewportRect,
  supportedScopes = ['selection', 'from-here', 'current-section', 'full-item'],
  onReadScope,
  onAnnotate,
  t
}: MediaReadAlongPopoverProps) {
  const supportedScopeSet = new Set(supportedScopes)
  const actions: PopoverAction[] = [
    {
      scope: 'selection',
      testId: 'media-selection-action-read-selection',
      label: t('review:mediaPage.readSelection', { defaultValue: 'Read selection' }),
      title: t('review:mediaPage.readSelectionTooltip', {
        defaultValue: 'Read the selected text'
      }),
      icon: <Volume2 className="h-3.5 w-3.5" />
    },
    {
      scope: 'from-here',
      testId: 'media-selection-action-read-from-here',
      label: t('review:mediaPage.readFromHere', { defaultValue: 'Read from here' }),
      title: t('review:mediaPage.readFromHereTooltip', {
        defaultValue: 'Read from this point forward'
      }),
      icon: <SkipForward className="h-3.5 w-3.5" />
    },
    {
      scope: 'current-section',
      testId: 'media-selection-action-read-current-section',
      label: t('review:mediaPage.readCurrentSection', {
        defaultValue: 'Read current section'
      }),
      title: t('review:mediaPage.readCurrentSectionTooltip', {
        defaultValue: 'Read the current section'
      }),
      icon: <List className="h-3.5 w-3.5" />
    },
    {
      scope: 'full-item',
      testId: 'media-selection-action-read-full-item',
      label: t('review:mediaPage.readFullItem', { defaultValue: 'Read full item' }),
      title: t('review:mediaPage.readFullItemTooltip', {
        defaultValue: 'Read the full media item'
      }),
      icon: <FileText className="h-3.5 w-3.5" />
    },
    {
      testId: 'media-selection-action-annotate',
      label: t('review:mediaPage.annotate', { defaultValue: 'Annotate' }),
      title: t('review:mediaPage.annotateSelection', {
        defaultValue: 'Annotate selection'
      }),
      icon: <StickyNote className="h-3.5 w-3.5" />,
      onClick: onAnnotate
    }
  ].filter((action) => !action.scope || supportedScopeSet.has(action.scope))

  const position = getPopoverPosition(anchorRect, viewportRect)

  return (
    <div
      className="fixed z-50 flex flex-wrap items-center gap-1 rounded border border-border bg-surface px-2 py-1 shadow-lg"
      style={{
        left: position.left,
        maxWidth: position.maxWidth,
        top: position.top
      }}
      data-testid="media-selection-actions-popover"
    >
      {actions.map((action) => (
        <Tooltip title={action.title} key={action.testId}>
          <button
            type="button"
            className="inline-flex h-7 items-center gap-1 rounded px-2 text-xs font-medium text-text hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
            onPointerDown={(event) => event.preventDefault()}
            onMouseDown={(event) => event.preventDefault()}
            onClick={action.onClick || (() => action.scope && onReadScope(action.scope))}
            data-testid={action.testId}
          >
            {action.icon}
            <span>{action.label}</span>
          </button>
        </Tooltip>
      ))}
    </div>
  )
}
