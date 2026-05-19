import React from "react"
import { Tooltip } from "antd"
import { Play, RotateCcw, Upload, Trash2 } from "lucide-react"
import type { ModerationTestState, TestHistoryEntry } from "./hooks/useModerationTest"
import type { ModerationTestResponse } from "@/services/moderation"

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface TestSandboxPanelProps {
  tester: ModerationTestState
  messageApi: {
    success: (msg: string) => void
    error: (msg: string) => void
    warning: (msg: string) => void
  }
}

// ---------------------------------------------------------------------------
// Quick sample data
// ---------------------------------------------------------------------------

const QUICK_SAMPLES = [
  { label: "PII: email", text: "Contact me at john.doe@example.com for details" },
  { label: "PII: phone", text: "My phone number is 555-123-4567" },
  { label: "Profanity", text: "That was a damn stupid thing to do" },
  { label: "Violence", text: "I want to kill the process and bomb the deployment" },
  { label: "Clean text", text: "The weather is nice today and I enjoy reading books" }
] as const

const PHASE_OPTIONS = [
  { value: "input" as const, label: "User message" },
  { value: "output" as const, label: "AI response" }
] as const

// ---------------------------------------------------------------------------
// Status badge mapping
// ---------------------------------------------------------------------------

const STATUS_CONFIG: Record<
  string,
  { label: string; bg: string; text: string }
> = {
  pass: {
    label: "Content Allowed",
    bg: "bg-green-100 dark:bg-green-900/30",
    text: "text-green-800 dark:text-green-300"
  },
  block: {
    label: "Content Blocked",
    bg: "bg-red-100 dark:bg-red-900/30",
    text: "text-red-800 dark:text-red-300"
  },
  redact: {
    label: "Content Redacted",
    bg: "bg-orange-100 dark:bg-orange-900/30",
    text: "text-orange-800 dark:text-orange-300"
  },
  warn: {
    label: "Warning Logged",
    bg: "bg-yellow-100 dark:bg-yellow-900/30",
    text: "text-yellow-800 dark:text-yellow-300"
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function highlightMatch(original: string, sample: string | null | undefined): React.ReactNode {
  if (!sample) return original
  const idx = original.indexOf(sample)
  if (idx === -1) return original
  return (
    <>
      {original.slice(0, idx)}
      <mark className="bg-red-200 dark:bg-red-800/50 px-0.5 rounded">{sample}</mark>
      {original.slice(idx + sample.length)}
    </>
  )
}

function truncate(str: string, max: number): string {
  return str.length > max ? str.slice(0, max) + "..." : str
}

function hasUserOverride(effective: Record<string, any>): boolean {
  return Boolean(
    effective.user_override ||
    effective.user_overrides ||
    effective.override_applied ||
    effective.per_user_override_applied
  )
}

function explainModerationResult(
  result: ModerationTestResponse,
  phase: string
): string {
  const effective = result.effective || {}
  if (effective.enabled === false) {
    return "The moderation engine is disabled in the effective policy, so the sample was allowed without rule checks."
  }
  if (phase === "input" && effective.input_enabled === false) {
    return "User message moderation is disabled for the effective policy, so input rules were skipped."
  }
  if (phase === "output" && effective.output_enabled === false) {
    return "AI response moderation is disabled for the effective policy, so output rules were skipped."
  }
  if (!result.flagged && result.action === "pass") {
    return "No active rule matched this sample, so the content passed moderation."
  }
  if (hasUserOverride(effective)) {
    return `A per-user override influenced this ${phase} decision and produced the ${result.action} action.`
  }
  if (result.sample || result.category) {
    const category = result.category ? `${result.category} ` : ""
    const sample = result.sample ? ` with sample "${result.sample}"` : ""
    return `The sample matched an active ${category}rule${sample}, so the policy returned ${result.action}.`
  }
  return `No specific matched rule was returned; the global policy fallback produced the ${result.action} action.`
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const ResultBadge: React.FC<{ action: string }> = ({ action }) => {
  const cfg = STATUS_CONFIG[action] ?? STATUS_CONFIG.pass
  return (
    <span
      className={`inline-flex items-center px-3 py-1.5 rounded-lg text-sm font-semibold ${cfg.bg} ${cfg.text}`}
      data-testid="result-badge"
    >
      {cfg.label}
    </span>
  )
}

const ResultDetails: React.FC<{ result: ModerationTestResponse; text: string; phase: string }> = ({
  result,
  text,
  phase
}) => (
  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
    {/* Left: Match Details */}
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-text">Match Details</h4>
      <div className="space-y-2 text-sm">
        {result.category && (
          <div>
            <span className="text-text-muted">Category: </span>
            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300">
              {result.category}
            </span>
          </div>
        )}
        {result.sample && (
          <div>
            <span className="text-text-muted">Matched pattern: </span>
            <code className="px-1.5 py-0.5 bg-surface-secondary rounded text-xs">{result.sample}</code>
          </div>
        )}
        <div>
          <span className="text-text-muted">Action: </span>
          <span className="font-medium">{result.action}</span>
        </div>
        <div>
          <span className="text-text-muted">Phase: </span>
          <span className="font-medium">{phase === "input" ? "User message" : "AI response"}</span>
        </div>
      </div>
    </div>

    {/* Right: Before/After */}
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-text">Before / After</h4>
      <div className="space-y-2">
        <div>
          <span className="text-xs text-text-muted block mb-1">Original:</span>
          <div className="p-2 bg-surface-secondary rounded text-sm break-words">
            {highlightMatch(text, result.sample)}
          </div>
        </div>
        {result.redacted_text && (
          <div>
            <span className="text-xs text-text-muted block mb-1">Redacted:</span>
            <div className="p-2 bg-surface-secondary rounded text-sm break-words">
              {result.redacted_text}
            </div>
          </div>
        )}
      </div>
    </div>
  </div>
)

const ResultExplanation: React.FC<{
  result: ModerationTestResponse
  phase: string
}> = ({ result, phase }) => (
  <div className="rounded-lg border border-border bg-surface/50 p-3">
    <h4 className="text-sm font-semibold text-text">Why this result?</h4>
    <p className="mt-1 text-sm text-text-muted">
      {explainModerationResult(result, phase)}
    </p>
  </div>
)

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const TestSandboxPanel: React.FC<TestSandboxPanelProps> = ({ tester, messageApi }) => {
  const { phase, setPhase, text, setText, userId, setUserId, result, running, runTest, runTestWith, history, clearHistory, loadFromHistory } = tester
  const phaseLabelId = React.useId()
  const userIdInputId = React.useId()
  const sampleTextId = React.useId()

  const handlePhaseKeyDown = (
    event: React.KeyboardEvent<HTMLButtonElement>,
    currentPhase: "input" | "output"
  ) => {
    const currentIndex = PHASE_OPTIONS.findIndex((option) => option.value === currentPhase)
    let nextIndex: number | null = null
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1) % PHASE_OPTIONS.length
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex = (currentIndex - 1 + PHASE_OPTIONS.length) % PHASE_OPTIONS.length
    } else if (event.key === "Home") {
      nextIndex = 0
    } else if (event.key === "End") {
      nextIndex = PHASE_OPTIONS.length - 1
    }
    if (nextIndex == null) return
    event.preventDefault()
    setPhase(PHASE_OPTIONS[nextIndex].value)
  }

  const handleRunTest = async () => {
    try {
      await runTest()
    } catch (err: any) {
      messageApi.error(err?.message || "Test failed")
    }
  }

  const handleRerun = async (entry: TestHistoryEntry) => {
    try {
      await runTestWith({ text: entry.text, phase: entry.phase, userId: entry.userId })
    } catch (err: any) {
      messageApi.error(err?.message || "Test failed")
    }
  }

  return (
    <div className="space-y-6">
      {/* ---- Test Configuration ---- */}
      <section className="space-y-4">
        <h3 className="text-lg font-semibold text-text">Test Configuration</h3>

        {/* Phase selector */}
        <div>
          <span id={phaseLabelId} className="block text-sm font-medium text-text-muted mb-1">
            Phase
          </span>
          <div
            className="inline-flex rounded-lg border border-border overflow-hidden"
            role="radiogroup"
            aria-labelledby={phaseLabelId}
          >
            {PHASE_OPTIONS.map((option, index) => (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={phase === option.value}
                tabIndex={phase === option.value ? 0 : -1}
                onClick={() => setPhase(option.value)}
                onKeyDown={(event) => handlePhaseKeyDown(event, option.value)}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  index > 0 ? "border-l border-border" : ""
                } ${
                  phase === option.value
                    ? "bg-blue-500 text-white"
                    : "bg-surface text-text hover:bg-surface-secondary"
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        {/* User ID input */}
        <div>
          <label htmlFor={userIdInputId} className="block text-sm font-medium text-text-muted mb-1">User ID</label>
          <input
            id={userIdInputId}
            type="text"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="User ID (optional)"
            className="w-full max-w-sm px-3 py-2 border border-border rounded-lg bg-surface text-text text-sm placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Sample text */}
        <div>
          <label htmlFor={sampleTextId} className="block text-sm font-medium text-text-muted mb-1">Sample Text</label>
          <textarea
            id={sampleTextId}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to test against moderation rules..."
            rows={5}
            className="w-full px-3 py-2 border border-border rounded-lg bg-surface text-text text-sm placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
            data-testid="sample-text"
          />
        </div>

        {/* Quick sample buttons */}
        <div>
          <span className="block text-xs text-text-muted mb-2">Quick samples:</span>
          <div className="flex flex-wrap gap-2">
            {QUICK_SAMPLES.map((sample) => (
              <button
                key={sample.label}
                type="button"
                onClick={() => setText(sample.text)}
                className="px-2.5 py-1 text-xs rounded-md border border-border bg-surface hover:bg-surface-secondary text-text transition-colors"
              >
                {sample.label}
              </button>
            ))}
          </div>
        </div>

        {/* Run Test button */}
        <div>
          <button
            type="button"
            onClick={handleRunTest}
            disabled={running || !text.trim()}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500 text-white text-sm font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Play className="w-4 h-4" />
            {running ? "Running..." : "Run Test"}
          </button>
        </div>
      </section>

      {/* ---- Results Section ---- */}
      {result && (
        <section className="space-y-4" data-testid="results-section" aria-live="polite">
          <h3 className="text-lg font-semibold text-text">Results</h3>

          {/* Status badge */}
          <ResultBadge action={result.action} />

          {/* Match details + Before/After */}
          <ResultDetails result={result} text={text} phase={phase} />

          <ResultExplanation result={result} phase={phase} />

          {/* Effective Policy */}
          <details className="mt-4">
            <summary className="text-sm font-medium text-text-muted cursor-pointer hover:text-text">
              Effective Policy
            </summary>
            <textarea
              readOnly
              aria-label="Effective policy JSON"
              value={JSON.stringify(result.effective, null, 2)}
              className="mt-2 w-full h-48 px-3 py-2 border border-border rounded-lg bg-surface-secondary text-text text-xs font-mono resize-y"
            />
          </details>
        </section>
      )}

      {/* ---- Test History ---- */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-text">Test History</h3>
          {history.length > 0 && (
            <button
              type="button"
              onClick={clearHistory}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md border border-border bg-surface hover:bg-surface-secondary text-text-muted hover:text-text transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear history
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="text-sm text-text-muted py-4" data-testid="history-empty">
            No tests run yet
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-[680px] w-full text-sm" data-testid="history-table">
              <thead>
                <tr className="border-b border-border text-left text-text-muted">
                  <th className="py-2 pr-3 font-medium">#</th>
                  <th className="py-2 pr-3 font-medium">Input</th>
                  <th className="py-2 pr-3 font-medium">Phase</th>
                  <th className="py-2 pr-3 font-medium">Result</th>
                  <th className="py-2 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {history.map((entry, idx) => {
                  const entryCfg = STATUS_CONFIG[entry.result.action] ?? STATUS_CONFIG.pass
                  return (
                    <tr key={entry.timestamp} className="border-b border-border/50">
                      <td className="py-2 pr-3 text-text-muted">{idx + 1}</td>
                      <td className="py-2 pr-3 text-text max-w-[200px] truncate" title={entry.text}>
                        {truncate(entry.text, 50)}
                      </td>
                      <td className="py-2 pr-3 text-text-muted">
                        {entry.phase === "input" ? "User message" : "AI response"}
                      </td>
                      <td className="py-2 pr-3">
                        <span
                          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${entryCfg.bg} ${entryCfg.text}`}
                        >
                          {entryCfg.label}
                        </span>
                      </td>
                      <td className="py-2">
                        <div className="flex items-center gap-2">
                          <Tooltip title="Rerun this test">
                            <button
                              type="button"
                              onClick={() => handleRerun(entry)}
                              className="p-1 rounded hover:bg-surface-secondary text-text-muted hover:text-text transition-colors"
                              aria-label="Rerun"
                            >
                              <RotateCcw className="w-3.5 h-3.5" />
                            </button>
                          </Tooltip>
                          <Tooltip title="Load into editor">
                            <button
                              type="button"
                              onClick={() => loadFromHistory(entry)}
                              className="p-1 rounded hover:bg-surface-secondary text-text-muted hover:text-text transition-colors"
                              aria-label="Load"
                            >
                              <Upload className="w-3.5 h-3.5" />
                            </button>
                          </Tooltip>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}

export default TestSandboxPanel
