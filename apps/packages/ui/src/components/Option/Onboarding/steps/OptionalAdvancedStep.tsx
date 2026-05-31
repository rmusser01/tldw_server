import React from "react"
import { SlidersHorizontal } from "lucide-react"

import type {
  FirstRunStepSaveResponse,
  OptionalAdvancedRequest
} from "@/types/setup-onboarding"

type OptionalAdvancedStepProps = {
  saveOptionalAdvanced: (
    payload: OptionalAdvancedRequest
  ) => Promise<FirstRunStepSaveResponse>
  onContinue: () => void
  onBack: () => void
}

type OptionalChoice = "configure" | "defer" | "skip"

export function OptionalAdvancedStep({
  saveOptionalAdvanced,
  onContinue,
  onBack
}: OptionalAdvancedStepProps) {
  const [rag, setRag] = React.useState<OptionalChoice>("defer")
  const [storagePaths, setStoragePaths] = React.useState<OptionalChoice>("defer")
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const handleContinue = async () => {
    setSaving(true)
    setError(null)
    try {
      await saveOptionalAdvanced({
        rag,
        storage_paths: storagePaths
      })
      onContinue()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Advanced choices were not saved.")
    } finally {
      setSaving(false)
    }
  }

  const renderChoice = (
    label: string,
    value: OptionalChoice,
    current: OptionalChoice,
    setter: (value: OptionalChoice) => void,
    name: string
  ) => (
    <label className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text">
      <input
        type="radio"
        name={name}
        value={value}
        checked={current === value}
        onChange={() => setter(value)}
        className="mr-2"
      />
      {label}
    </label>
  )

  return (
    <section aria-labelledby="optional-advanced-title" className="space-y-5">
      <div className="flex items-start gap-3">
        <span className="inline-flex size-10 items-center justify-center rounded-md bg-surface2 text-primary">
          <SlidersHorizontal className="size-5" aria-hidden="true" />
        </span>
        <div>
          <h2
            id="optional-advanced-title"
            className="text-lg font-semibold text-text"
          >
            Optional advanced setup
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            RAG and storage paths are optional for the first chat.
          </p>
        </div>
      </div>

      {error ? (
        <div
          role="alert"
          className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          {error}
        </div>
      ) : null}

      <fieldset className="space-y-2">
        <legend className="text-sm font-medium text-text">RAG and embeddings</legend>
        <div className="flex flex-wrap gap-2">
          {renderChoice("Configure after setup", "configure", rag, setRag, "rag")}
          {renderChoice("Defer", "defer", rag, setRag, "rag")}
          {renderChoice("Skip", "skip", rag, setRag, "rag")}
        </div>
        {rag === "configure" ? (
          <a href="/settings/rag" className="text-sm font-medium text-primary">
            Open RAG settings after setup
          </a>
        ) : null}
      </fieldset>

      <fieldset className="space-y-2">
        <legend className="text-sm font-medium text-text">Storage paths</legend>
        <div className="flex flex-wrap gap-2">
          {renderChoice(
            "Configure after setup",
            "configure",
            storagePaths,
            setStoragePaths,
            "storage-paths"
          )}
          {renderChoice("Defer", "defer", storagePaths, setStoragePaths, "storage-paths")}
          {renderChoice("Skip", "skip", storagePaths, setStoragePaths, "storage-paths")}
        </div>
      </fieldset>

      <div className="flex flex-wrap justify-between gap-2">
        <button
          type="button"
          onClick={onBack}
          disabled={saving}
          className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
        >
          Back
        </button>
        <button
          type="button"
          onClick={handleContinue}
          disabled={saving}
          className="rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
        >
          {saving ? "Saving..." : "Continue"}
        </button>
      </div>
    </section>
  )
}
