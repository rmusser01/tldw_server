import React from "react"
import { X } from "lucide-react"

import type { SidepanelChatHandoffPageContext } from "@/services/sidepanel-chat-handoff"

export function SidepanelImportedContextBanner({
  context,
  onRemove,
  labels,
}: {
  context: SidepanelChatHandoffPageContext
  onRemove: () => void
  labels?: {
    defaultTitle?: string
    regionLabel?: string
    removeContext?: (title: string) => string
    snippetSummary?: (count: number) => string
    noSnippets?: string
  }
}) {
  const title =
    context.title || labels?.defaultTitle || "Imported sidepanel context"
  const snippetCount = context.snippets.length
  const snippetSummary =
    snippetCount > 0
      ? labels?.snippetSummary?.(snippetCount) ??
        `${snippetCount} snippet${snippetCount === 1 ? "" : "s"}`
      : labels?.noSnippets ?? "No snippets"
  const firstSnippetPreview = context.snippets[0]?.text.trim()
  const preview =
    firstSnippetPreview && firstSnippetPreview.length > 0
      ? firstSnippetPreview.slice(0, 120)
      : null
  const removeLabel =
    labels?.removeContext?.(title) || `Remove imported context from ${title}`

  return (
    <section
      aria-label={labels?.regionLabel || "Imported sidepanel context"}
      className="mb-2 rounded-lg border border-border bg-surface2/70 px-3 py-2 text-sm text-text"
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate font-medium">{title}</div>
          {context.url ? (
            <div className="truncate text-xs text-text-muted">{context.url}</div>
          ) : null}
          <div className="truncate text-xs text-text-muted">
            {preview ? `${snippetSummary} - ${preview}` : snippetSummary}
          </div>
        </div>
        <button
          type="button"
          onClick={onRemove}
          aria-label={removeLabel}
          className="rounded p-1 text-text-subtle hover:bg-surface hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </section>
  )
}
