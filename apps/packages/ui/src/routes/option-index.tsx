import React from "react"

import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { UnifiedSetupWizard } from "@/components/Option/Onboarding/UnifiedSetupWizard"
import {
  useConnectionActions,
  useConnectionState
} from "@/hooks/useConnectionState"
import { useFocusComposerOnConnect } from "@/hooks/useComposerFocus"
import { useSetupOnboarding } from "@/hooks/useSetupOnboarding"
import OptionLayout from "~/components/Layouts/Layout"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"

const LazyCompanionHomeShell = React.lazy(() =>
  import("@/components/Option/CompanionHome").then((module) => ({
    default: module.CompanionHomeShell
  }))
)

const LazyOptionHostedHome = React.lazy(() => import("./option-hosted-home"))

const setupRequiredStatuses = new Set([
  "not_started",
  "in_progress",
  "blocked",
  "first_chat_complete"
])

const OptionIndex = () => {
  const hostedMode = isHostedTldwDeployment()
  const { phase } = useConnectionState()
  const { checkOnce } = useConnectionActions()
  const {
    state: firstRunState,
    metadata: firstRunMetadata,
    loading: setupLoading,
    adoptState: adoptFirstRunState
  } = useSetupOnboarding()
  const [didHydrate, setDidHydrate] = React.useState(false)

  React.useEffect(() => {
    if (hostedMode) {
      setDidHydrate(true)
      return
    }
    let cancelled = false
    const run = async () => {
      try {
        await checkOnce()
      } finally {
        if (!cancelled) setDidHydrate(true)
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [checkOnce, hostedMode])

  useFocusComposerOnConnect(phase ?? null)

  if (hostedMode) {
    return (
      <OptionLayout hideHeader hideSidebar>
        <React.Suspense
          fallback={
            <PageAssistLoader
              label="Loading home..."
              description="Preparing your workspace"
            />
          }
        >
          <LazyOptionHostedHome />
        </React.Suspense>
      </OptionLayout>
    )
  }

  if ((setupLoading || !didHydrate) && !firstRunState) {
    return (
      <OptionLayout hideHeader hideSidebar>
        <PageAssistLoader
          label="Loading setup..."
          description="Reading first-run readiness from the server"
        />
      </OptionLayout>
    )
  }

  const setupStatus = firstRunState?.status
  if (!setupStatus || setupRequiredStatuses.has(setupStatus)) {
    return (
      <OptionLayout hideHeader hideSidebar>
        <UnifiedSetupWizard
          initialState={firstRunState}
          initialMetadata={firstRunMetadata}
          onStateChange={adoptFirstRunState}
        />
      </OptionLayout>
    )
  }

  return (
    <OptionLayout>
      <React.Suspense
        fallback={
          <PageAssistLoader
            label="Loading home..."
            description="Preparing your dashboard"
          />
        }
      >
        <LazyCompanionHomeShell surface="options" />
      </React.Suspense>
    </OptionLayout>
  )
}

export default OptionIndex
