import React from "react"
import type { DynamicUISurface } from "@/types/dynamic-ui"

export const DynamicUISourceFallback = ({
  title = "OpenUI source",
  source,
  error,
  surface
}: {
  title?: string
  source: string
  error?: string
  surface?: DynamicUISurface
}) => (
  <details
    data-testid="dynamic-ui-source-fallback"
    data-dynamic-ui-surface={surface}
    className="rounded-md border border-border bg-surface2 p-3 text-sm text-text"
    open>
    <summary className="cursor-pointer font-medium text-text">{title}</summary>
    {error ? (
      <p role="alert" className="mt-2 text-danger">
        {error}
      </p>
    ) : null}
    <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded bg-surface p-2 text-xs text-text-muted">
      {source}
    </pre>
  </details>
)
