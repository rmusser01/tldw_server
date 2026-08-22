import React from "react"

import { Button } from "@/components/Common/Button"
import { Badge } from "@/components/ui"
import { useStandaloneHtmlGeneration } from "@/hooks/useStandaloneHtmlGeneration"
import type { SlidesCapabilities } from "@/services/tldw/TldwApiClient"

type StandaloneHtmlGenerationFormProps = {
  capabilities: SlidesCapabilities | null
  recoveryOnly?: boolean
  onCapabilitiesChanged?: () => Promise<unknown> | unknown
  onCompleted?: (presentationId: string) => void
  onStopWaiting?: () => void
}

const PRESENTATION_TYPES = [
  ["pitch-deck", "Pitch deck"],
  ["tech-sharing", "Tech sharing"],
  ["product-launch", "Product launch"],
  ["weekly-report", "Weekly report"],
  ["course-module", "Course module"],
  ["keynote", "Keynote"],
  ["data-report", "Data report"],
  ["training", "Training"],
  ["social-media", "Social media"],
  ["case-study", "Case study"],
  ["comparison", "Comparison"],
  ["roadmap", "Roadmap"]
] as const

const VISUAL_DIRECTIONS = [
  ["auto", "Auto"],
  ["dark-technical", "Dark technical"],
  ["minimal-light", "Minimal light"],
  ["editorial", "Editorial"],
  ["corporate", "Corporate"],
  ["soft-pastel", "Soft pastel"],
  ["bold-creative", "Bold creative"],
  ["neo-brutalist", "Neo-brutalist"]
] as const

const inputClass =
  "min-h-[44px] w-full rounded-md border border-border bg-surface px-3 py-2 text-sm text-text outline-none transition-colors motion-reduce:transition-none focus-visible:border-focus focus-visible:ring-2 focus-visible:ring-focus/30 disabled:cursor-not-allowed disabled:bg-surface2 disabled:text-text-muted"

const statusLabel = (status: string | null): string | null => {
  if (!status) return null
  if (status === "queued") return "Queued"
  if (status === "running") return "Running"
  if (status === "completed") return "Completed"
  if (status === "failed") return "Failed"
  if (status === "cancelled") return "Cancelled"
  return null
}

const phaseLabel = (phase: string): string | null => {
  if (phase === "failed") return "Failed"
  if (phase === "cancelled") return "Cancelled"
  if (phase === "completed_missing_binding") return "Completed"
  if (phase === "ambiguous") return "Submission outcome unknown"
  if (phase === "auth_lost") return "Sign in required"
  if (phase === "missing") return "Generation not found"
  if (phase === "throttled") return "Status checks paused"
  if (phase === "outage") return "Status unavailable"
  if (phase === "stopped") return "Waiting stopped"
  if (phase === "rejected") return "Request rejected"
  if (phase === "configuration_changed") return "Configuration changed"
  return null
}

export const StandaloneHtmlGenerationForm: React.FC<StandaloneHtmlGenerationFormProps> = ({
  capabilities,
  recoveryOnly = false,
  onCapabilitiesChanged = () => undefined,
  onCompleted = () => undefined,
  onStopWaiting = () => undefined
}) => {
  const generation = capabilities?.generation_modes.standalone_html ?? null
  const capability = generation?.enabled ? generation : null
  const contentMaxSlides = capabilities?.content_kinds.standalone_html.limits.max_slides ?? 30

  return (
    <EnabledStandaloneHtmlGenerationForm
      capability={capability}
      contentMaxSlides={contentMaxSlides}
      recoveryOnly={recoveryOnly}
      onCapabilitiesChanged={onCapabilitiesChanged}
      onCompleted={onCompleted}
      onStopWaiting={onStopWaiting}
    />
  )
}

type EnabledCapability = Extract<
  SlidesCapabilities["generation_modes"]["standalone_html"],
  { enabled: true }
>

const EnabledStandaloneHtmlGenerationForm: React.FC<{
  capability: EnabledCapability | null
  contentMaxSlides: number
  recoveryOnly: boolean
  onCapabilitiesChanged: () => Promise<unknown> | unknown
  onCompleted: (presentationId: string) => void
  onStopWaiting: () => void
}> = ({ capability, contentMaxSlides, recoveryOnly, onCapabilitiesChanged, onCompleted, onStopWaiting }) => {
  const generation = useStandaloneHtmlGeneration({
    capability,
    contentMaxSlides,
    onCapabilitiesChanged,
    onCompleted,
    onStopWaiting
  })
  const [confirmDifferent, setConfirmDifferent] = React.useState(false)
  const {
    draft,
    fieldErrors,
    editError,
    phase,
    locked,
    snapshot,
    backendStatus,
    progressText,
    safeError,
    recoveryAvailable,
    storageWarning
  } = generation

  const currentStatus = phaseLabel(phase) ?? statusLabel(backendStatus)
  const isSubmitting = phase === "submitting"
  const isPolling = phase === "polling"
  const canTryAgain = ["failed", "cancelled", "completed_missing_binding"].includes(phase)
  const canResume = recoveryAvailable && [
    "ambiguous", "stopped", "auth_lost", "missing", "throttled", "outage"
  ].includes(phase)
  const canStartDifferent = snapshot !== null && [
    "ambiguous", "stopped", "auth_lost", "missing", "throttled", "outage", "rejected"
  ].includes(phase)

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-border bg-surface p-4 sm:p-6" aria-labelledby="html-generation-heading">
        <div className="max-w-[72ch] space-y-2">
          <h2 id="html-generation-heading" className="text-lg font-semibold text-text">
            Standalone HTML + JavaScript
          </h2>
          <p className="text-sm text-text-muted">
            Paste direct material to generate a text-only outline and a downloadable presentation file.
          </p>
          <p className="text-sm text-text-muted">
            Generated JavaScript never runs in Presentation Studio. It runs only if you download and open the file outside tldw.
          </p>
        </div>

        <dl className="mt-4 grid gap-x-6 gap-y-2 border-t border-border pt-4 text-sm sm:grid-cols-2">
          <div><dt className="text-text-muted">Provider</dt><dd className="break-all font-medium text-text">{capability?.provider ?? "Unavailable"}</dd></div>
          <div><dt className="text-text-muted">Model</dt><dd className="break-all font-medium text-text">{capability?.model ?? "Unavailable"}</dd></div>
          <div><dt className="text-text-muted">Adapter</dt><dd className="break-all font-medium text-text">{capability?.adapter_id ?? "Unavailable"}</dd></div>
          <div><dt className="text-text-muted">Endpoint</dt><dd className="break-all font-medium text-text">{capability?.endpoint_identity ?? "Unavailable"}</dd></div>
          <div className="sm:col-span-2"><dt className="text-text-muted">Generation configuration revision</dt><dd className="break-all font-mono text-xs text-text">{capability?.generation_config_revision ?? "Unavailable"}</dd></div>
        </dl>

        <form className="mt-6 space-y-5" autoComplete="off" onSubmit={(event) => { event.preventDefault(); void generation.submit() }}>
          <div>
            <label htmlFor="standalone-html-source" className="mb-1 block text-sm font-medium text-text">
              Subject and material
            </label>
            <textarea
              id="standalone-html-source"
              value={draft.source}
              onChange={(event) => generation.updateField("source", event.target.value)}
              disabled={locked || recoveryOnly || !capability}
              rows={9}
              spellCheck={false}
              autoCorrect="off"
              autoCapitalize="off"
              autoComplete="off"
              data-1p-ignore="true"
              aria-invalid={Boolean(fieldErrors.source)}
              aria-describedby={fieldErrors.source ? "standalone-html-source-error" : "standalone-html-source-help"}
              className={`${inputClass} resize-y`}
            />
            <p id="standalone-html-source-help" className="mt-1 text-xs text-text-muted">
              Direct pasted material only. This version does not load chats, media, notes, or RAG sources.
            </p>
            {fieldErrors.source ? <p id="standalone-html-source-error" role="alert" className="mt-1 text-sm text-danger">{fieldErrors.source}</p> : null}
          </div>

          <div className="grid gap-5 sm:grid-cols-2">
            <div>
              <label htmlFor="standalone-html-type" className="mb-1 block text-sm font-medium text-text">Presentation type</label>
              <select
                id="standalone-html-type"
                value={draft.presentationType}
                disabled={locked || recoveryOnly || !capability}
                onChange={(event) => generation.updateField("presentationType", event.target.value as typeof draft.presentationType)}
                className={inputClass}
              >
                {PRESENTATION_TYPES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor="standalone-html-audience" className="mb-1 block text-sm font-medium text-text">Audience</label>
              <input
                id="standalone-html-audience"
                type="text"
                value={draft.audience}
                disabled={locked || recoveryOnly || !capability}
                onChange={(event) => generation.updateField("audience", event.target.value)}
                spellCheck={false}
                autoCorrect="off"
                autoCapitalize="off"
                autoComplete="off"
                data-1p-ignore="true"
                aria-invalid={Boolean(fieldErrors.audience)}
                aria-describedby={fieldErrors.audience ? "standalone-html-audience-error" : undefined}
                className={inputClass}
              />
              {fieldErrors.audience ? <p id="standalone-html-audience-error" role="alert" className="mt-1 text-sm text-danger">{fieldErrors.audience}</p> : null}
            </div>
            <div>
              <label htmlFor="standalone-html-slide-count" className="mb-1 block text-sm font-medium text-text">Approximate slide count</label>
              <input
                id="standalone-html-slide-count"
                type="number"
                min={1}
                max={Math.min(30, contentMaxSlides)}
                step={1}
                value={draft.slideCount}
                disabled={locked || recoveryOnly || !capability}
                onChange={(event) => generation.updateField("slideCount", Number(event.target.value))}
                aria-invalid={Boolean(fieldErrors.slideCount)}
                aria-describedby={fieldErrors.slideCount ? "standalone-html-slide-count-error" : undefined}
                className={inputClass}
              />
              {fieldErrors.slideCount ? <p id="standalone-html-slide-count-error" role="alert" className="mt-1 text-sm text-danger">{fieldErrors.slideCount}</p> : null}
            </div>
            <div>
              <label htmlFor="standalone-html-direction" className="mb-1 block text-sm font-medium text-text">Visual direction</label>
              <select
                id="standalone-html-direction"
                value={draft.visualDirection}
                disabled={locked || recoveryOnly || !capability}
                onChange={(event) => generation.updateField("visualDirection", event.target.value as typeof draft.visualDirection)}
                className={inputClass}
              >
                {VISUAL_DIRECTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </div>
          </div>

          <fieldset disabled={locked || recoveryOnly || !capability} className="space-y-2">
            <legend className="text-sm font-medium text-text">Delivery style</legend>
            <div className="grid gap-2 sm:grid-cols-2">
              <label className="flex min-h-[44px] cursor-pointer items-start gap-3 rounded-md border border-border p-3 focus-within:ring-2 focus-within:ring-focus/30">
                <input type="radio" aria-label="Speaker-led" checked={draft.deliveryStyle === "speaker-led"} onChange={() => generation.updateField("deliveryStyle", "speaker-led")} className="mt-1 h-4 w-4" />
                <span><span className="block text-sm font-medium text-text">Speaker-led</span><span className="block text-xs text-text-muted">Uses concise speaker notes to support a live presenter.</span></span>
              </label>
              <label className="flex min-h-[44px] cursor-pointer items-start gap-3 rounded-md border border-border p-3 focus-within:ring-2 focus-within:ring-focus/30">
                <input type="radio" aria-label="Self-guided" checked={draft.deliveryStyle === "self-guided"} onChange={() => generation.updateField("deliveryStyle", "self-guided")} className="mt-1 h-4 w-4" />
                <span><span className="block text-sm font-medium text-text">Self-guided</span><span className="block text-xs text-text-muted">Includes fuller context but does not autoplay or auto-advance.</span></span>
              </label>
            </div>
          </fieldset>

          {editError ? <p role="alert" className="text-sm text-danger">{editError}</p> : null}
          {generation.scopeError ? <p role="alert" className="text-sm text-danger">{generation.scopeError}</p> : null}
          {!snapshot && safeError ? <p role="alert" className="break-words font-mono text-sm text-danger">{safeError}</p> : null}
          {storageWarning ? <p role="status" className="text-sm text-state-degraded">{storageWarning}</p> : null}

          <Button
            type="submit"
            variant="primary"
            size="lg"
            disabled={!generation.scopeReady || snapshot !== null || recoveryOnly || !capability}
            loading={isSubmitting}
          >
            {isSubmitting ? "Submitting request" : "Generate standalone presentation"}
          </Button>
        </form>
      </section>

      {snapshot ? (
        <section className="rounded-lg border border-border bg-surface p-4 sm:p-6" aria-labelledby="submitted-request-heading">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 id="submitted-request-heading" className="text-lg font-semibold text-text">Submitted request</h2>
            {currentStatus ? (
              <span role="status" aria-live="polite">
                <Badge variant={phase === "failed" ? "warning" : "secondary"}>{currentStatus}</Badge>
              </span>
            ) : null}
          </div>
          <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
            <div className="sm:col-span-2"><dt className="text-text-muted">Subject and material</dt><dd><pre className="mt-1 whitespace-pre-wrap break-words font-sans text-text">{snapshot.source.kind === "prompt" ? snapshot.source.prompt : "Direct material"}</pre></dd></div>
            <div><dt className="text-text-muted">Audience</dt><dd className="text-text">{snapshot.html_options.audience}</dd></div>
            <div><dt className="text-text-muted">Slide count</dt><dd className="text-text">{snapshot.html_options.slide_count}</dd></div>
            <div><dt className="text-text-muted">Presentation type</dt><dd className="text-text">{snapshot.html_options.presentation_type}</dd></div>
            <div><dt className="text-text-muted">Visual direction</dt><dd className="text-text">{snapshot.html_options.visual_direction}</dd></div>
            <div><dt className="text-text-muted">Delivery style</dt><dd className="text-text">{snapshot.html_options.delivery_style}</dd></div>
            <div className="sm:col-span-2"><dt className="text-text-muted">Generation configuration revision</dt><dd className="break-all font-mono text-xs text-text">{snapshot.generation_config_revision}</dd></div>
          </dl>

          {progressText ? <p role="status" className="mt-4 text-sm text-text">{progressText}</p> : null}
          {safeError ? <p role="alert" className="mt-4 break-words font-mono text-sm text-danger">{safeError}</p> : null}

          <div className="mt-5 flex flex-wrap gap-2">
            {isPolling ? <Button variant="outline" size="lg" onClick={generation.stopWaiting}>Stop waiting</Button> : null}
            {canResume ? <Button variant="primary" size="lg" onClick={() => void generation.resume()}>Resume</Button> : null}
            {canTryAgain ? <Button variant="primary" size="lg" onClick={() => void generation.tryAgain()}>Try again</Button> : null}
            {canStartDifferent ? <Button variant="outline" size="lg" onClick={() => setConfirmDifferent(true)}>Start a different request</Button> : null}
            {recoveryAvailable ? <Button variant="outline" size="lg" onClick={generation.forget}>Forget this job; generation continues</Button> : null}
          </div>

          {recoveryAvailable ? (
            <p className="mt-3 text-xs text-text-muted">
              Forget removes only this browser tab&apos;s recovery record. It does not cancel generation or delete a presentation.
            </p>
          ) : null}

          {confirmDifferent ? (
            <div role="alert" className="mt-4 rounded-md border border-state-degraded/30 bg-state-degraded/10 p-3 text-sm text-text">
              <p>The original request may still complete. Starting a different request uses a new replay key.</p>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button variant="primary" size="lg" onClick={() => { setConfirmDifferent(false); generation.startDifferent() }}>Confirm different request</Button>
                <Button variant="outline" size="lg" onClick={() => setConfirmDifferent(false)}>Keep waiting</Button>
              </div>
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  )
}
