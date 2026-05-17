import React from 'react'
import { FileText, List, SkipForward, StickyNote, Volume2 } from 'lucide-react'
import { Tooltip } from 'antd'

import type { ReadAlongScope } from './types'

type Translate = (key: string, opts?: Record<string, any>) => string

interface MediaReadAlongPopoverProps {
  anchorRect: DOMRect
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

export function MediaReadAlongPopover({
  anchorRect,
  onReadScope,
  onAnnotate,
  t
}: MediaReadAlongPopoverProps) {
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
  ]

  return (
    <div
      className="fixed z-50 flex max-w-[min(92vw,560px)] flex-wrap items-center gap-1 rounded border border-border bg-surface px-2 py-1 shadow-lg"
      style={{
        left: Math.max(8, anchorRect.left),
        top: Math.max(8, anchorRect.top - 40)
      }}
      data-testid="media-selection-actions-popover"
    >
      {actions.map((action) => (
        <Tooltip title={action.title} key={action.testId}>
          <button
            type="button"
            className="inline-flex h-7 items-center gap-1 rounded px-2 text-xs font-medium text-text hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
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
