import React from "react"
import { FilePlus2, X } from "lucide-react"

type FirstSourceMilestonePromptProps = {
  onAddSource: () => void
  onDismiss: () => void
}

export function FirstSourceMilestonePrompt({
  onAddSource,
  onDismiss
}: FirstSourceMilestonePromptProps) {
  return (
    <section
      aria-labelledby="first-source-milestone-title"
      className="mx-auto mb-4 w-full max-w-5xl rounded-md border border-border bg-surface px-4 py-4 shadow-sm"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <span className="inline-flex size-10 shrink-0 items-center justify-center rounded-md bg-surface2 text-primary">
            <FilePlus2 className="size-5" aria-hidden="true" />
          </span>
          <div>
            <h2
              id="first-source-milestone-title"
              className="text-base font-semibold text-text"
            >
              Add your first source
            </h2>
            <p className="mt-1 max-w-2xl text-sm text-text-muted">
              First chat is working. Add a source next so chat can use your own
              material.
            </p>
          </div>
        </div>
        <div className="flex shrink-0 gap-2">
          <button
            type="button"
            onClick={onAddSource}
            className="rounded-md bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground"
          >
            Add source
          </button>
          <button
            type="button"
            aria-label="Dismiss"
            onClick={onDismiss}
            className="inline-flex size-9 items-center justify-center rounded-md border border-border bg-surface text-text hover:bg-surface2"
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        </div>
      </div>
    </section>
  )
}
