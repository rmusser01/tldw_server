import React from "react"
import { useNavigate } from "react-router-dom"
import OptionLayout from "~/components/Layouts/Layout"
import { OnboardingWizard } from "@/components/Option/Onboarding/OnboardingWizard"
import { SetupRequiredPanel } from "@/components/ui/state"

const OptionSetup = () => {
  const navigate = useNavigate()

  const handleFinish = React.useCallback(() => {
    navigate("/")
  }, [navigate])

  return (
    <OptionLayout hideHeader hideSidebar>
      <div className="mx-auto mb-4 w-full max-w-3xl">
        <h1 className="text-lg font-semibold text-text">Setup Wizard</h1>
      </div>
      <SetupRequiredPanel
        className="mx-auto mb-4 w-full max-w-3xl"
        title="Connect your server"
        message="Guided connection setup for production use."
        primaryAction={{
          label: "Start setup",
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
