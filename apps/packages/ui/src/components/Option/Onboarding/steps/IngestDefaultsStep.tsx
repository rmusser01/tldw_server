import React from "react"
import { FileText } from "lucide-react"

import type {
  FirstRunStepSaveResponse,
  IngestDefaultsRequest
} from "@/types/setup-onboarding"

type IngestDefaultsStepProps = {
  saveIngestDefaults: (
    payload: IngestDefaultsRequest
  ) => Promise<FirstRunStepSaveResponse>
  onContinue: () => void
  onBack: () => void
}

const splitRoots = (value: string) =>
  value
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean)

export function IngestDefaultsStep({
  saveIngestDefaults,
  onContinue,
  onBack
}: IngestDefaultsStepProps) {
  const [allowLocalFiles, setAllowLocalFiles] = React.useState(false)
  const [allowedRoots, setAllowedRoots] = React.useState("")
  const [chunkingProfile, setChunkingProfile] =
    React.useState<NonNullable<IngestDefaultsRequest["chunking_profile"]>>(
      "balanced"
    )
  const [metadataMode, setMetadataMode] =
    React.useState<NonNullable<IngestDefaultsRequest["metadata_mode"]>>(
      "automatic"
    )
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const handleContinue = async () => {
    setSaving(true)
    setError(null)
    try {
      await saveIngestDefaults({
        allow_local_file_ingest: allowLocalFiles,
        allowed_local_roots: allowLocalFiles ? splitRoots(allowedRoots) : [],
        chunking_profile: chunkingProfile,
        metadata_mode: metadataMode
      })
      onContinue()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ingest defaults were not saved.")
    } finally {
      setSaving(false)
    }
  }

  return (
    <section aria-labelledby="ingest-defaults-title" className="space-y-5">
      <div className="flex items-start gap-3">
        <span className="inline-flex size-10 items-center justify-center rounded-md bg-surface2 text-primary">
          <FileText className="size-5" aria-hidden="true" />
        </span>
        <div>
          <h2 id="ingest-defaults-title" className="text-lg font-semibold text-text">
            Ingest defaults
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            Choose conservative defaults for your first sources.
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

      <label className="flex items-start gap-3 rounded-md border border-border bg-surface px-4 py-3 text-sm text-text">
        <input
          type="checkbox"
          checked={allowLocalFiles}
          onChange={(event) => setAllowLocalFiles(event.currentTarget.checked)}
          className="mt-1"
        />
        <span>
          Allow local file ingest
          <span className="mt-1 block text-xs text-text-muted">
            Keep this off unless you want the WebUI to help ingest files from
            local paths.
          </span>
        </span>
      </label>

      <label className="block text-sm font-medium text-text">
        <span>Allowed local roots</span>
        <textarea
          value={allowedRoots}
          onChange={(event) => setAllowedRoots(event.currentTarget.value)}
          disabled={!allowLocalFiles}
          rows={3}
          className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text disabled:opacity-50"
          placeholder="/Users/me/Documents"
        />
      </label>

      <fieldset className="space-y-2">
        <legend className="text-sm font-medium text-text">Chunking profile</legend>
        <div className="flex flex-wrap gap-2">
          {["simple", "balanced", "advanced"].map((profile) => (
            <label
              key={profile}
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
            >
              <input
                type="radio"
                name="first-run-chunking-profile"
                value={profile}
                checked={chunkingProfile === profile}
                onChange={() => setChunkingProfile(profile)}
                className="mr-2"
              />
              {profile}
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset className="space-y-2">
        <legend className="text-sm font-medium text-text">Metadata mode</legend>
        <div className="flex flex-wrap gap-2">
          {["automatic", "ask", "minimal"].map((mode) => (
            <label
              key={mode}
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
            >
              <input
                type="radio"
                name="first-run-metadata-mode"
                value={mode}
                checked={metadataMode === mode}
                onChange={() => setMetadataMode(mode)}
                className="mr-2"
              />
              {mode}
            </label>
          ))}
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
