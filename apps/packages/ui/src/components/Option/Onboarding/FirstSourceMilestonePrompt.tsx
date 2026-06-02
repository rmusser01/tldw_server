import React from "react"
import { ClipboardPaste, FilePlus2, Link, Loader2, Upload, X } from "lucide-react"

export type FirstSourceKind = "web_url" | "file_upload" | "paste_text"

type FirstSourceMilestonePromptProps = {
  readinessStatus: "idle" | "processing" | "ready" | "error"
  lastSourceLabel?: string | null
  errorMessage?: string | null
  onAddSource: (kind: FirstSourceKind) => void
  onAskAboutSource?: () => void
  onRetry?: () => void
  onDismiss: () => void
}

const SOURCE_KIND_OPTIONS: Array<{
  kind: FirstSourceKind
  label: string
  Icon: typeof Link
}> = [
  { kind: "web_url", label: "Web URL", Icon: Link },
  { kind: "file_upload", label: "File", Icon: Upload },
  { kind: "paste_text", label: "Paste", Icon: ClipboardPaste },
]

export function FirstSourceMilestonePrompt({
  readinessStatus,
  lastSourceLabel,
  errorMessage,
  onAddSource,
  onAskAboutSource,
  onRetry,
  onDismiss
}: FirstSourceMilestonePromptProps) {
  const [selectedKind, setSelectedKind] =
    React.useState<FirstSourceKind>("web_url")
  const isProcessing = readinessStatus === "processing"
  const isError = readinessStatus === "error"
  const isReady = readinessStatus === "ready"

  return (
    <section
      aria-labelledby="first-source-milestone-title"
      className="mx-auto mb-4 w-full max-w-5xl rounded-md border border-border bg-surface px-4 py-4 shadow-sm"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <span className="inline-flex size-10 shrink-0 items-center justify-center rounded-md bg-surface2 text-primary">
            {isProcessing ? (
              <Loader2 className="size-5 animate-spin" aria-hidden="true" />
            ) : (
              <FilePlus2 className="size-5" aria-hidden="true" />
            )}
          </span>
          <div className="min-w-0">
            <h2
              id="first-source-milestone-title"
              className="text-base font-semibold text-text"
            >
              Add your first source
            </h2>
            <p className="mt-1 max-w-2xl text-sm text-text-muted">
              {isProcessing
                ? "Processing your source. Chat about it will unlock when it is ready."
                : isReady
                  ? "Your first source is ready for grounded chat."
                  : isError
                    ? "Source ingest did not finish. Retry when you are ready."
                    : "First chat is working. Add a source next so chat can use your own material."}
            </p>
            {lastSourceLabel ? (
              <p className="mt-2 max-w-2xl truncate text-sm font-medium text-text">
                {lastSourceLabel}
              </p>
            ) : null}
            {errorMessage ? (
              <p className="mt-2 max-w-2xl text-sm text-danger">
                {errorMessage}
              </p>
            ) : null}
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap justify-end gap-2">
          {isReady && onAskAboutSource ? (
            <button
              type="button"
              onClick={onAskAboutSource}
              className="rounded-md bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground"
            >
              Ask a question about this source
            </button>
          ) : null}
          {isError && onRetry ? (
            <button
              type="button"
              onClick={onRetry}
              className="rounded-md bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground"
            >
              Retry
            </button>
          ) : null}
          {!isProcessing && !isReady && !isError ? (
            <button
              type="button"
              onClick={() => onAddSource(selectedKind)}
              className="rounded-md bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground"
            >
              Add source
            </button>
          ) : null}
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
      {!isProcessing && !isReady && !isError ? (
        <div className="mt-4 grid gap-2 sm:grid-cols-3">
          {SOURCE_KIND_OPTIONS.map(({ kind, label, Icon }) => {
            const selected = selectedKind === kind
            return (
              <label
                key={kind}
                className={`flex min-h-12 cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-sm font-medium focus-within:outline-none focus-within:ring-2 focus-within:ring-primary focus-within:ring-offset-1 ${
                  selected
                    ? "border-primary bg-primary/10 text-text"
                    : "border-border bg-bg text-text-muted hover:bg-surface2"
                }`}
              >
                <input
                  type="radio"
                  name="first-source-kind"
                  value={kind}
                  checked={selected}
                  onChange={() => setSelectedKind(kind)}
                  className="sr-only"
                />
                <Icon className="size-4 shrink-0" aria-hidden="true" />
                <span>{label}</span>
              </label>
            )
          })}
        </div>
      ) : null}
    </section>
  )
}
