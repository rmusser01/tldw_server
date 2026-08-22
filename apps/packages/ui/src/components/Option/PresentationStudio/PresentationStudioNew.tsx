import React from "react"
import { useNavigate } from "react-router-dom"

import { Button } from "@/components/Common/Button"
import { Badge, LoadingState, StatePanel } from "@/components/ui"
import { useSlidesCapabilities } from "@/hooks/useSlidesCapabilities"
import { useStandaloneHtmlRecoveryProbe } from "@/hooks/useStandaloneHtmlGeneration"

import { PresentationStudioPage } from "./PresentationStudioPage"
import { StandaloneHtmlGenerationForm } from "./StandaloneHtmlGenerationForm"

type CreationMode = "structured" | "standalone_html"

export const PresentationStudioNew: React.FC = () => {
  const navigate = useNavigate()
  const slides = useSlidesCapabilities()
  const recovery = useStandaloneHtmlRecoveryProbe()
  const [creationMode, setCreationMode] = React.useState<CreationMode>("structured")
  const recoveryAvailable = recovery.status === "available"
  const capabilityConfirmed = ["ready", "generation_disabled", "validator_unavailable"].includes(slides.status)
  const htmlOptionEnabled = capabilityConfirmed || recoveryAvailable

  React.useEffect(() => {
    if (recoveryAvailable) setCreationMode("standalone_html")
  }, [recoveryAvailable])

  const standaloneContent = React.useMemo(() => {
    if (recoveryAvailable && !slides.canGenerate) {
      return (
        <StandaloneHtmlGenerationForm
          capabilities={slides.capabilities}
          recoveryOnly
          onCapabilitiesChanged={slides.retry}
          onCompleted={(presentationId) => navigate(`/presentation-studio/${presentationId}`, { replace: true })}
          onStopWaiting={() => navigate("/presentation-studio")}
        />
      )
    }
    if (slides.status === "loading") {
      return (
        <div role="status" aria-label="Loading generation capabilities" className="rounded-lg border border-border bg-surface p-4">
          <LoadingState mode="skeleton" rows={3} />
        </div>
      )
    }
    if (slides.status === "offline") {
      return <StatePanel state="unavailable" title="Presentation Studio is offline" message="Reconnect before starting generation." primaryAction={{ label: "Retry", onClick: () => void slides.retry() }} />
    }
    if (slides.status === "error" || slides.status === "auth_required" || slides.status === "forbidden") {
      return (
        <StatePanel
          state="error"
          title="Generation capabilities could not load"
          message="Generation stays unavailable until the server contract is confirmed."
          primaryAction={{ label: "Retry", onClick: () => void slides.retry() }}
          role="alert"
        />
      )
    }
    if (slides.status === "validator_unavailable") {
      return (
        <StatePanel
          state="blocked"
          title="Standalone validation is unavailable"
          message={<code className="break-all">{slides.reason ?? "validator_unavailable"}</code>}
          primaryAction={{ label: "Retry", onClick: () => void slides.retry() }}
        />
      )
    }
    if (slides.status === "generation_disabled" || !slides.canGenerate || !slides.capabilities) {
      return (
        <StatePanel
          state="blocked"
          title="Standalone generation is disabled"
          message={<code className="break-all">{slides.reason ?? "generation_disabled"}</code>}
          primaryAction={{ label: "Retry", onClick: () => void slides.retry() }}
        />
      )
    }
    return (
      <StandaloneHtmlGenerationForm
        capabilities={slides.capabilities}
        onCapabilitiesChanged={slides.retry}
        onCompleted={(presentationId) => navigate(`/presentation-studio/${presentationId}`, { replace: true })}
        onStopWaiting={() => navigate("/presentation-studio")}
      />
    )
  }, [navigate, recoveryAvailable, slides])

  return (
    <div className="space-y-6 py-6">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold text-text">New presentation</h1>
          <p className="max-w-[70ch] text-sm text-text-muted">
            Choose the durable project format that fits this presentation.
          </p>
        </div>
        <Button variant="outline" size="lg" onClick={() => navigate("/presentation-studio")}>Back to projects</Button>
      </header>

      <fieldset className="space-y-2">
        <legend className="text-sm font-semibold text-text">Creation mode</legend>
        <div className="grid gap-2 sm:grid-cols-2">
          <label className="flex min-h-[64px] cursor-pointer items-start gap-3 rounded-lg border border-border bg-surface p-4 focus-within:ring-2 focus-within:ring-focus/30">
            <input
              type="radio"
              name="presentation-creation-mode"
              value="structured"
              checked={creationMode === "structured"}
              onChange={() => setCreationMode("structured")}
              className="mt-1 h-4 w-4"
            />
            <span>
              <span className="block text-sm font-semibold text-text">Structured slides</span>
              <span className="mt-1 block text-xs text-text-muted">Editable slides with the existing studio workflow.</span>
            </span>
          </label>
          <label className="flex min-h-[64px] cursor-pointer items-start gap-3 rounded-lg border border-border bg-surface p-4 focus-within:ring-2 focus-within:ring-focus/30">
            <input
              type="radio"
              name="presentation-creation-mode"
              value="standalone_html"
              checked={creationMode === "standalone_html"}
              onChange={() => setCreationMode("standalone_html")}
              disabled={!htmlOptionEnabled}
              className="mt-1 h-4 w-4"
            />
            <span>
              <span className="flex flex-wrap items-center gap-2 text-sm font-semibold text-text">
                Standalone HTML + JavaScript <Badge variant="secondary">Experimental</Badge>
              </span>
              <span className="mt-1 block text-xs text-text-muted">Generated as one downloadable file that can run only after you download and open it outside tldw.</span>
            </span>
          </label>
        </div>
      </fieldset>

      {!htmlOptionEnabled && slides.status !== "loading" ? (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface p-4" role="status">
          <p className="text-sm text-text-muted">Standalone generation stays unavailable until the server contract is confirmed.</p>
          <Button variant="outline" size="lg" onClick={() => void slides.retry()}>Retry generation capabilities</Button>
        </div>
      ) : null}

      {creationMode === "structured" ? <PresentationStudioPage mode="new" embedded /> : standaloneContent}
    </div>
  )
}
