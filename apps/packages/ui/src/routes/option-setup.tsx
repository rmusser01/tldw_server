import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import OptionLayout from "~/components/Layouts/Layout"
import { OnboardingWizard } from "@/components/Option/Onboarding/OnboardingWizard"
import { ReadinessSetupScreen } from "@/components/Option/Setup/ReadinessSetupScreen"
import { SetupRequiredPanel } from "@/components/ui/state"
import {
  useConnectionState,
  useConnectionUxState
} from "@/hooks/useConnectionState"

const OptionSetup = () => {
  const { t } = useTranslation("option")
  const navigate = useNavigate()
  const { serverUrl } = useConnectionState()
  const { hasCompletedFirstRun, isConfigOrError } = useConnectionUxState()
  const [readinessUnavailable, setReadinessUnavailable] = React.useState(false)

  const handleFinish = React.useCallback(() => {
    navigate("/")
  }, [navigate])

  React.useEffect(() => {
    setReadinessUnavailable(false)
  }, [serverUrl, isConfigOrError, hasCompletedFirstRun])

  if (serverUrl && !isConfigOrError && !readinessUnavailable) {
    return (
      <OptionLayout hideHeader hideSidebar>
        <ReadinessSetupScreen
          mode={hasCompletedFirstRun ? "admin" : "first-run"}
          onComplete={handleFinish}
          onUnavailable={() => setReadinessUnavailable(true)}
        />
      </OptionLayout>
    )
  }

  return (
    <OptionLayout hideHeader hideSidebar>
      <div className="mx-auto mb-4 w-full max-w-3xl">
        <h1 className="text-lg font-semibold text-text">
          {t("setupRoute.title", "Setup Wizard")}
        </h1>
      </div>
      <SetupRequiredPanel
        className="mx-auto mb-4 w-full max-w-3xl"
        title={t("setupRoute.panelTitle", "Connect your server")}
        message={t(
          "setupRoute.panelMessage",
          "Guided connection setup for production use."
        )}
        primaryAction={{
          label: t("setupRoute.startAction", "Start setup"),
          onClick: () => {
            document
              .querySelector<HTMLElement>(
                "[data-testid='onboarding-server-url'] input, [data-testid='onboarding-server-url']"
              )
              ?.focus()
          }
        }}
      />
      <OnboardingWizard onFinish={handleFinish} />
    </OptionLayout>
  )
}

export default OptionSetup
