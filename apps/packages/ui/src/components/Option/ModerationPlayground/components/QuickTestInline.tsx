import React from "react"
import { X, Zap } from "lucide-react"
import type { ModerationTestResponse } from "@/services/moderation"

interface QuickTestInlineProps {
  open: boolean
  onClose: () => void
  onRunTest: (text: string, phase: "input" | "output") => Promise<ModerationTestResponse | undefined>
  onOpenFull: () => void
  userId?: string
}

const resultLabel: Record<string, { text: string; color: string }> = {
  pass: { text: "PASS", color: "text-green-600 dark:text-green-400" },
  block: { text: "BLOCKED", color: "text-red-600 dark:text-red-400" },
  redact: { text: "REDACTED", color: "text-orange-600 dark:text-orange-400" },
  warn: { text: "WARNED", color: "text-yellow-600 dark:text-yellow-400" }
}

export const QuickTestInline: React.FC<QuickTestInlineProps> = ({
  open,
  onClose,
  onRunTest,
  onOpenFull,
  userId
}) => {
  const [text, setText] = React.useState("")
  const [phase, setPhase] = React.useState<"input" | "output">("input")
  const [result, setResult] = React.useState<ModerationTestResponse | null>(null)
  const [running, setRunning] = React.useState(false)
  const inputRef = React.useRef<HTMLInputElement>(null)
  const textInputId = React.useId()
  const phaseSelectId = React.useId()

  React.useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [open])

  React.useEffect(() => {
    if (!open) return
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", handleEsc)
    return () => window.removeEventListener("keydown", handleEsc)
  }, [open, onClose])

  const handleRun = async () => {
    if (!text.trim()) return
    setRunning(true)
    try {
      const res = await onRunTest(text, phase)
      if (res) setResult(res)
    } finally {
      setRunning(false)
    }
  }

  if (!open) return null

  const label = result ? resultLabel[result.action] ?? resultLabel.pass : null

  return (
    <div className="border-b border-border bg-surface/80 backdrop-blur-sm px-4 py-3">
      <div className="flex max-w-7xl flex-col gap-2 mx-auto sm:flex-row sm:items-center">
        <Zap className="h-4 w-4 text-text-muted flex-shrink-0" />
        <label htmlFor={textInputId} className="sr-only">
          Quick test text
        </label>
        <input
          id={textInputId}
          ref={inputRef}
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleRun()}
          placeholder="Quick test text..."
          className="w-full min-w-0 flex-1 px-2 py-1 text-sm border border-border rounded bg-bg text-text placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <label htmlFor={phaseSelectId} className="sr-only">
          Quick test phase
        </label>
        <select
          id={phaseSelectId}
          value={phase}
          onChange={(e) => setPhase(e.target.value as "input" | "output")}
          className="w-full px-2 py-1 text-sm border border-border rounded bg-bg text-text sm:w-auto"
        >
          <option value="input">Input</option>
          <option value="output">Output</option>
        </select>
        <button
          type="button"
          onClick={handleRun}
          disabled={running || !text.trim()}
          className="w-full px-3 py-1 text-sm font-medium rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 sm:w-auto"
        >
          {running ? "..." : "Run"}
        </button>
        {result && (
          <span className="flex min-w-0 flex-wrap items-center gap-1.5 text-sm" role="status" aria-live="polite">
            <span className={`font-semibold ${label?.color}`}>{label?.text}</span>
            {result.category && (
              <span className="text-text-muted">· {result.category}</span>
            )}
            <button
              type="button"
              onClick={onOpenFull}
              className="text-blue-500 hover:underline text-xs ml-1"
            >
              Full results
            </button>
          </span>
        )}
        <button
          type="button"
          onClick={onClose}
          className="self-start p-1 hover:bg-surface rounded sm:self-auto"
          aria-label="Close quick test"
        >
          <X className="h-4 w-4 text-text-muted" />
        </button>
      </div>
    </div>
  )
}
