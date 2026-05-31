import React from "react"
import { useNavigate } from "react-router-dom"

import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { UnifiedSetupWizard } from "@/components/Option/Onboarding/UnifiedSetupWizard"
import { SetupRequiredPanel } from "@/components/ui/state"
import { useSetupOnboarding } from "@/hooks/useSetupOnboarding"
import OptionLayout from "~/components/Layouts/Layout"

const setupRequiredStatuses = new Set([
  "not_started",
  "in_progress",
  "blocked",
  "first_chat_complete"
])

const OptionSetup = () => {
  const navigate = useNavigate()
  const { state, metadata, loading, adoptState } = useSetupOnboarding()
  const status = state?.status
  const showWizard = !status || setupRequiredStatuses.has(status)

  return (
    <OptionLayout hideHeader hideSidebar>
      <SetupRequiredPanel
        className="mx-auto mb-4 w-full max-w-3xl"
        title="/setup operator recovery"
        message="Use this surface when first-run setup needs local operator recovery."
        primaryAction={{
          label: showWizard ? "Continue setup" : "Return home",
          onClick: () => {
            if (!showWizard) {
              navigate("/")
              return
            }
            const setupShell = document.querySelector<HTMLElement>(
              "[data-testid='unified-setup-shell']"
            )
            const fallbackButton = document.querySelector<HTMLElement>("button")
            const focusTarget = setupShell ?? fallbackButton
            focusTarget?.focus()
          }
        }}
      />
      {loading && !state ? (
        <PageAssistLoader
          label="Loading setup..."
          description="Reading first-run readiness from the server"
        />
      ) : showWizard ? (
        <UnifiedSetupWizard
          initialState={state}
          initialMetadata={metadata}
          onStateChange={adoptState}
        />
      ) : null}
    </OptionLayout>
  )
}

export default OptionSetup
