import React from "react"

import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { useSetupOnboarding } from "@/hooks/useSetupOnboarding"
import type {
  FirstRunMetadata,
  FirstRunState,
  FirstRunStepUpdateRequest
} from "@/types/setup-onboarding"
import { SetupPathStep } from "./steps/SetupPathStep"
import { PrivacySecurityStep } from "./steps/PrivacySecurityStep"
import { MultiUserExitPanel } from "./steps/MultiUserExitPanel"

type WizardStep =
  | "setup_path"
  | "privacy_security"
  | "provider_setup"
  | "multi_user_exit"

type SoloSetupPath = "docker" | "local"

type UnifiedSetupWizardProps = {
  initialState?: FirstRunState | null
  initialMetadata?: FirstRunMetadata | null
  onStateChange?: (state: FirstRunState) => void
}

const setupPathToBackend = (path: SoloSetupPath) =>
  path === "docker" ? "docker_single_user" : "local_single_user"

export function UnifiedSetupWizard({
  initialState = null,
  initialMetadata = null,
  onStateChange
}: UnifiedSetupWizardProps = {}) {
  const {
    state,
    metadata,
    loading,
    error,
    saveStep,
    skip
  } = useSetupOnboarding({
    initialState,
    initialMetadata,
    autoLoad: !initialState || !initialMetadata
  })
  const [step, setStep] = React.useState<WizardStep>("setup_path")
  const [selectedPath, setSelectedPath] = React.useState<SoloSetupPath | null>(
    null
  )
  const [savingStep, setSavingStep] = React.useState(false)
  const [stepError, setStepError] = React.useState<string | null>(null)

  const persistStep = React.useCallback(
    async (payload: FirstRunStepUpdateRequest) => {
      setSavingStep(true)
      setStepError(null)
      try {
        const nextState = await saveStep(payload)
        onStateChange?.(nextState)
        return nextState
      } catch {
        setStepError("Setup progress could not be saved. Try again.")
        return null
      } finally {
        setSavingStep(false)
      }
    },
    [onStateChange, saveStep]
  )

  const handlePathSelect = React.useCallback(
    (path: "docker" | "local" | "multi_user") => {
      if (path === "multi_user") {
        setStepError(null)
        setStep("multi_user_exit")
        return
      }

      void (async () => {
        const nextState = await persistStep({
          step: "setup_path",
          data: {
            acknowledged: true,
            selected_path: setupPathToBackend(path),
            setup_path_key: setupPathToBackend(path),
            install_method: path,
            deployment_mode: "single_user"
          }
        })
        if (!nextState) return
        setSelectedPath(path)
        setStep("privacy_security")
      })()
    },
    [persistStep]
  )

  const handlePrivacyContinue = React.useCallback(() => {
    void (async () => {
      const nextState = await persistStep({
        step: "privacy_security",
        data: {
          acknowledged: true,
          local_only: metadata?.connection?.browser_access === "local",
          allow_remote_setup_access: Boolean(metadata?.remote_setup_enabled)
        }
      })
      if (!nextState) return
      setStep("provider_setup")
    })()
  }, [metadata, persistStep])

  const handleSkip = React.useCallback(() => {
    setStepError(null)
    void skip({ reason: "user_skip" })
      .then((nextState) => {
        onStateChange?.(nextState)
      })
      .catch(() => {
        setStepError("Setup skip could not be saved. Try again.")
      })
  }, [onStateChange, skip])

  if (loading && !state && !metadata) {
    return (
      <PageAssistLoader
        label="Loading setup..."
        description="Reading setup readiness from the server"
      />
    )
  }

  return (
    <div
      data-testid="unified-setup-shell"
      tabIndex={-1}
      className="mx-auto flex min-h-screen w-full max-w-4xl flex-col px-4 py-8"
    >
      <header className="mb-6">
        <p className="text-xs font-medium uppercase tracking-normal text-text-muted">
          Solo onboarding
        </p>
        <div className="mt-2 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold text-text">
              First-time setup
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-text-muted">
              Configure the minimum needed to reach a successful first chat.
            </p>
          </div>
          <button
            type="button"
            onClick={handleSkip}
            className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2"
          >
            Skip for now
          </button>
        </div>
      </header>

      {error ? (
        <div
          role="alert"
          className="mb-4 rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          Setup readiness could not be loaded.
        </div>
      ) : null}

      {stepError ? (
        <div
          role="alert"
          className="mb-4 rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          {stepError}
        </div>
      ) : null}

      <div className="rounded-md border border-border bg-bg px-4 py-5 shadow-sm md:px-6">
        {step === "setup_path" ? (
          <SetupPathStep onSelect={handlePathSelect} />
        ) : null}
        {step === "privacy_security" ? (
          <PrivacySecurityStep
            metadata={metadata}
            onBack={() => setStep("setup_path")}
            onContinue={handlePrivacyContinue}
            saving={savingStep}
          />
        ) : null}
        {step === "multi_user_exit" ? (
          <MultiUserExitPanel
            metadata={metadata}
            onBack={() => setStep("setup_path")}
          />
        ) : null}
        {step === "provider_setup" ? (
          <section aria-labelledby="provider-setup-title" className="space-y-3">
            <h2 id="provider-setup-title" className="text-lg font-semibold text-text">
              Chat provider
            </h2>
            <p className="text-sm text-text-muted">
              Next, connect a hosted API key or a local OpenAI-compatible
              endpoint for {selectedPath === "local" ? "local install" : "Docker"}.
            </p>
          </section>
        ) : null}
      </div>
    </div>
  )
}

export default UnifiedSetupWizard
