import React from "react"
import { Alert } from "@/components/ui/primitives"

// Cross-platform dev mode detection (Vite and Next.js compatible)
const isDevMode = typeof import.meta !== "undefined" && (
  Boolean(import.meta.env?.DEV) ||
  import.meta.env?.MODE === "development" ||
  (typeof process !== "undefined" && process.env.NODE_ENV === "development")
)

export type LocaleIssue = {
  path: string
  message: string
  line?: number
  column?: number
}

export interface LocaleJsonDiagnosticsPanelProps {
  issues: LocaleIssue[]
}

export const LocaleJsonDiagnosticsPanel: React.FC<
  LocaleJsonDiagnosticsPanelProps
> = ({ issues }) => (
  <div className="mb-4">
    <Alert variant="error" title="Locale JSON errors detected">
      <div className="space-y-1 text-xs">
        {issues.map((issue) => (
          <div key={issue.path} className="break-all">
            <span className="font-mono">{issue.path}</span>
            {issue.line != null && issue.column != null
              ? ` (line ${issue.line}, col ${issue.column})`
              : ""}
            {": "}
            {issue.message}
          </div>
        ))}
      </div>
    </Alert>
  </div>
)

const findErrorPosition = (message: string): number | null => {
  const match = message.match(/position (\d+)/i)
  if (!match) return null
  const pos = Number(match[1])
  return Number.isFinite(pos) ? pos : null
}

const findLineColumn = (text: string, position: number) => {
  if (position < 0 || position > text.length) return null
  const before = text.slice(0, position)
  const line = before.split("\n").length
  const column = position - before.lastIndexOf("\n")
  return { line, column }
}

export const LocaleJsonDiagnostics: React.FC = () => {
  const issues = React.useMemo<LocaleIssue[]>(() => {
    if (!isDevMode) return []

    // import.meta.glob is Vite-only - skip in Next.js/non-Vite environments
    if (typeof import.meta.glob !== "function") return []

    const rawModules = import.meta.glob("../../assets/locale/*/*.json", {
      query: "?raw",
      import: "default",
      eager: true
    }) as Record<string, string>

    const next: LocaleIssue[] = []
    Object.entries(rawModules).forEach(([path, raw]) => {
      if (typeof raw !== "string") return
      try {
        JSON.parse(raw)
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        const position = findErrorPosition(message)
        const loc = position != null ? findLineColumn(raw, position) : null
        next.push({
          path,
          message,
          line: loc?.line,
          column: loc?.column
        })
      }
    })

    return next
  }, [])

  if (!isDevMode || issues.length === 0) return null

  return <LocaleJsonDiagnosticsPanel issues={issues} />
}

export default LocaleJsonDiagnostics
