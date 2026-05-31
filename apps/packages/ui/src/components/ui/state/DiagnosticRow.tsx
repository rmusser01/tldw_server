import React from "react"
import { cn } from "@/libs/utils"

export interface DiagnosticRowProps {
  label: React.ReactNode
  value: React.ReactNode
  code?: boolean
  copyLabel?: string
  className?: string
  "data-testid"?: string
}

export function DiagnosticRow({
  label,
  value,
  code = false,
  copyLabel,
  className,
  "data-testid": dataTestId
}: DiagnosticRowProps) {
  const valueNode = code ? (
    <code className="break-words rounded-sm bg-surface2 px-1.5 py-0.5 font-mono text-xs text-text">
      {value}
    </code>
  ) : (
    <div className="break-words text-text">{value}</div>
  )

  return (
    <div
      className={cn("grid gap-1 py-1 text-sm sm:grid-cols-[9rem_minmax(0,1fr)]", className)}
      data-testid={dataTestId}
    >
      <dt className="font-medium text-text-muted">{label}</dt>
      <dd className="min-w-0">
        {valueNode}
        {copyLabel ? <span className="sr-only">{copyLabel}</span> : null}
      </dd>
    </div>
  )
}
