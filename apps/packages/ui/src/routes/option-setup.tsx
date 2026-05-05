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
      <SetupRequiredPanel
        className="mx-auto mb-4 w-full max-w-3xl"
        title="Setup Wizard"
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
