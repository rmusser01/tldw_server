import React from "react"
import { KeyRound, RotateCw } from "lucide-react"

type PostSetupApiRecoveryProps = {
  errorMessage?: string | null
  onRecover: (apiKey: string) => Promise<void>
  onRetry?: () => Promise<void>
}

export function PostSetupApiRecovery({
  errorMessage,
  onRecover,
  onRetry
}: PostSetupApiRecoveryProps) {
  const [apiKey, setApiKey] = React.useState("")
  const [submitting, setSubmitting] = React.useState(false)
  const [localError, setLocalError] = React.useState<string | null>(null)

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setLocalError(null)
    setSubmitting(true)
    try {
      await onRecover(apiKey)
      setApiKey("")
    } catch (error) {
      setLocalError(
        error instanceof Error && error.message
          ? error.message
          : "Could not save the API key."
      )
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section
      aria-labelledby="post-setup-api-recovery-title"
      className="mx-auto w-full max-w-2xl rounded-md border border-border bg-surface px-5 py-5 shadow-sm"
    >
      <div className="mb-4 flex items-start gap-3">
        <span className="inline-flex size-10 shrink-0 items-center justify-center rounded-md bg-surface2 text-primary">
          <KeyRound className="size-5" aria-hidden="true" />
        </span>
        <div className="min-w-0">
          <h2
            id="post-setup-api-recovery-title"
            className="text-base font-semibold text-text"
          >
            Restore media access
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            First chat is complete, but the WebUI needs a working server key
            before it can add sources.
          </p>
        </div>
      </div>

      {errorMessage ? (
        <p className="mb-3 rounded-md border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-text">
          {errorMessage}
        </p>
      ) : null}
      {localError ? (
        <p className="mb-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-text">
          {localError}
        </p>
      ) : null}

      <form className="space-y-3" onSubmit={submit}>
        <label
          htmlFor="post-setup-api-key"
          className="block text-sm font-medium text-text"
        >
          Single-user API key
        </label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            id="post-setup-api-key"
            type="password"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            autoComplete="off"
            className="min-h-10 min-w-0 flex-1 rounded-md border border-border bg-background px-3 py-2 text-sm text-text outline-none focus:border-primary"
          />
          <button
            type="submit"
            disabled={submitting}
            className="inline-flex min-h-10 items-center justify-center rounded-md bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-60"
          >
            {submitting ? "Saving..." : "Save API key"}
          </button>
          {onRetry ? (
            <button
              type="button"
              onClick={() => void onRetry()}
              className="inline-flex min-h-10 items-center justify-center rounded-md border border-border bg-surface px-3 py-2 text-sm font-semibold text-text hover:bg-surface2"
            >
              <RotateCw className="mr-2 size-4" aria-hidden="true" />
              Retry
            </button>
          ) : null}
        </div>
      </form>
    </section>
  )
}
