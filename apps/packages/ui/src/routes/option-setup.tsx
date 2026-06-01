import React from "react"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"

import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { UnifiedSetupWizard } from "@/components/Option/Onboarding/UnifiedSetupWizard"
import { SetupRequiredPanel } from "@/components/ui/state"
import { useSetupOnboarding } from "@/hooks/useSetupOnboarding"
import OptionLayout from "~/components/Layouts/Layout"
import { isSetupStatusRequiringWizard } from "./setup-status"

const OptionSetup = () => {
  const navigate = useNavigate()
  const { t } = useTranslation("option")
  const { state, metadata, loading, adoptState } = useSetupOnboarding()
  const status = state?.status
  const showWizard = isSetupStatusRequiringWizard(status)

  return (
    <OptionLayout hideHeader hideSidebar>
      <SetupRequiredPanel
        className="mx-auto mb-4 w-full max-w-3xl"
        title={t("setupRoute.recoveryTitle", "/setup operator recovery")}
        message={t(
          "setupRoute.recoveryMessage",
          "Use this surface when first-run setup needs local operator recovery."
        )}
        primaryAction={{
          label: showWizard
            ? t("setupRoute.continueAction", "Continue setup")
            : t("setupRoute.returnHomeAction", "Return home"),
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
          label={t("setupRoute.loadingLabel", "Loading setup...")}
          description={t(
            "setupRoute.loadingDescription",
            "Reading first-run readiness from the server"
          )}
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
