import React from "react"
import { KeyRound, Settings } from "lucide-react"
import { useLocation, useNavigate } from "react-router-dom"

import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { PageShell } from "@/components/Common/PageShell"
import { useConnectionUxState } from "@/hooks/useConnectionState"
import { buildFirstRunOnboardingRoute } from "@/utils/onboarding-route-intent"

type WorkspaceConnectionGateProps = {
  featureName: string
  children: React.ReactNode
  maxWidthClassName?: string
  className?: string
  authTitle?: React.ReactNode
  authDescription?: React.ReactNode
  setupTitle?: React.ReactNode
  setupDescription?: React.ReactNode
  unreachableTitle?: React.ReactNode
  unreachableDescription?: React.ReactNode
  renderDemo?: () => React.ReactNode
  renderTesting?: () => React.ReactNode
}

const defaultTestingState = () => (
  <div
    role="status"
    className="rounded-3xl border border-border/80 bg-surface/90 p-6 text-sm text-text shadow-card"
  >
    Checking server connection...
  </div>
)

export const WorkspaceConnectionGate: React.FC<
  WorkspaceConnectionGateProps
> = ({
  featureName,
  children,
  maxWidthClassName = "max-w-6xl",
  className = "py-6",
  authTitle,
  authDescription,
  setupTitle,
  setupDescription,
  unreachableTitle,
  unreachableDescription,
  renderDemo,
  renderTesting
}) => {
  const navigate = useNavigate()
  const location = useLocation()
  const { uxState, hasCompletedFirstRun } = useConnectionUxState()

  const openSettings = () => navigate("/settings/tldw")
  const openDiagnostics = () => navigate("/settings/health")
  const finishSetup = () => navigate(buildFirstRunOnboardingRoute(location))

  if (uxState === "demo_mode") {
    return renderDemo ? <>{renderDemo()}</> : <>{children}</>
  }

  if (uxState === "testing") {
    return (
      <PageShell className={className} maxWidthClassName={maxWidthClassName}>
        {renderTesting ? renderTesting() : defaultTestingState()}
      </PageShell>
    )
  }

  if (uxState === "error_auth" || uxState === "configuring_auth") {
    return (
      <PageShell className={className} maxWidthClassName={maxWidthClassName}>
        <FeatureEmptyState
          title={
            authTitle ||
            `Add your credentials before ${featureName} can load data.`
          }
          description={
            authDescription ||
            `Open Settings to add or repair your tldw server credentials, then return to ${featureName}.`
          }
          primaryActionLabel="Open Settings"
          onPrimaryAction={openSettings}
          icon={KeyRound}
          iconClassName="h-8 w-8 text-warn"
        />
      </PageShell>
    )
  }

  if (uxState === "unconfigured" || uxState === "configuring_url") {
    const needsFirstRun = !hasCompletedFirstRun
    return (
      <PageShell className={className} maxWidthClassName={maxWidthClassName}>
        <FeatureEmptyState
          title={
            setupTitle || `Finish setup before using ${featureName}.`
          }
          description={
            setupDescription ||
            `${featureName} depends on your connected tldw server.`
          }
          primaryActionLabel={needsFirstRun ? "Finish Setup" : "Open Settings"}
          onPrimaryAction={needsFirstRun ? finishSetup : openSettings}
          icon={Settings}
        />
      </PageShell>
    )
  }

  if (uxState === "error_unreachable") {
    return (
      <PageShell className={className} maxWidthClassName={maxWidthClassName}>
        <FeatureEmptyState
          title={
            unreachableTitle || "Can't reach your tldw server right now."
          }
          description={
            unreachableDescription ||
            `Open Health & diagnostics to verify your server is reachable, or review your server URL in Settings before returning to ${featureName}.`
          }
          primaryActionLabel="Health & diagnostics"
          onPrimaryAction={openDiagnostics}
          secondaryActionLabel="Open Settings"
          onSecondaryAction={openSettings}
        />
      </PageShell>
    )
  }

  return <>{children}</>
}

export default WorkspaceConnectionGate
