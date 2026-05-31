import React from "react"
import { Volume2 } from "lucide-react"

import type {
  AudioDefaultsRequest,
  FirstRunStepSaveResponse
} from "@/types/setup-onboarding"

type AudioSetupStepProps = {
  recommendations?: Array<Record<string, unknown>>
  saveAudioDefaults: (
    payload: AudioDefaultsRequest
  ) => Promise<FirstRunStepSaveResponse>
  onContinue: () => void
  onBack: () => void
}

const recommendationLabel = (recommendation: Record<string, unknown>) => {
  const bundle = recommendation.bundle
  if (bundle && typeof bundle === "object" && "name" in bundle) {
    return String((bundle as { name?: unknown }).name || "Recommended bundle")
  }
  return String(
    recommendation.label ||
      recommendation.bundle_id ||
      recommendation.id ||
      "Recommended bundle"
  )
}

export function AudioSetupStep({
  recommendations = [],
  saveAudioDefaults,
  onContinue,
  onBack
}: AudioSetupStepProps) {
  const [mode, setMode] =
    React.useState<NonNullable<AudioDefaultsRequest["mode"]>>("skip")
  const [sttProvider, setSttProvider] = React.useState("")
  const [ttsProvider, setTtsProvider] = React.useState("")
  const [ttsVoice, setTtsVoice] = React.useState("")
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const handleContinue = async () => {
    setSaving(true)
    setError(null)
    try {
      await saveAudioDefaults({
        mode,
        stt_provider: mode === "configure" ? sttProvider || null : null,
        tts_provider: mode === "configure" ? ttsProvider || null : null,
        tts_voice: mode === "configure" ? ttsVoice || null : null
      })
      onContinue()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Audio defaults were not saved.")
    } finally {
      setSaving(false)
    }
  }

  return (
    <section aria-labelledby="audio-setup-title" className="space-y-5">
      <div className="flex items-start gap-3">
        <span className="inline-flex size-10 items-center justify-center rounded-md bg-surface2 text-primary">
          <Volume2 className="size-5" aria-hidden="true" />
        </span>
        <div>
          <h2 id="audio-setup-title" className="text-lg font-semibold text-text">
            Audio, STT, and TTS
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            Choose whether to configure audio now or defer it.
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

      {recommendations.length > 0 ? (
        <div className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-text">
          Suggested: {recommendationLabel(recommendations[0])}
        </div>
      ) : null}

      <fieldset className="space-y-2">
        <legend className="text-sm font-medium text-text">Audio setup mode</legend>
        <div className="grid gap-2 md:grid-cols-3">
          {[
            ["defaults", "Use defaults"],
            ["configure", "Configure now"],
            ["skip", "Skip for now"]
          ].map(([value, label]) => (
            <label
              key={value}
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-text"
            >
              <input
                type="radio"
                name="first-run-audio-mode"
                value={value}
                checked={mode === value}
                onChange={() =>
                  setMode(value as NonNullable<AudioDefaultsRequest["mode"]>)
                }
                className="mr-2"
              />
              {label}
            </label>
          ))}
        </div>
      </fieldset>

      {mode === "configure" ? (
        <div className="grid gap-3 md:grid-cols-3">
          <label className="block text-sm font-medium text-text">
            <span>STT provider</span>
            <input
              value={sttProvider}
              onChange={(event) => setSttProvider(event.currentTarget.value)}
              className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
              placeholder="faster_whisper"
            />
          </label>
          <label className="block text-sm font-medium text-text">
            <span>TTS provider</span>
            <input
              value={ttsProvider}
              onChange={(event) => setTtsProvider(event.currentTarget.value)}
              className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
              placeholder="openai"
            />
          </label>
          <label className="block text-sm font-medium text-text">
            <span>TTS voice</span>
            <input
              value={ttsVoice}
              onChange={(event) => setTtsVoice(event.currentTarget.value)}
              className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
              placeholder="alloy"
            />
          </label>
        </div>
      ) : null}

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
