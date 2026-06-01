import React from "react"

export const DynamicUISourceFallback = ({
  title = "OpenUI source",
  source,
  error
}: {
  title?: string
  source: string
  error?: string
}) => (
  <details
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
