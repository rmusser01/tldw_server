import React from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"

import { PageShell } from "@/components/Common/PageShell"
import WorkspaceConnectionGate from "@/components/Common/WorkspaceConnectionGate"
import { StatePanel, buildCapabilityState } from "@/components/ui/state"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import type { ServerCapabilities } from "@/services/tldw/server-capabilities"

type SourcesAvailabilityGateProps = {
  children: React.ReactNode
  capabilityState?: {
    capabilities: ServerCapabilities | null
    loading: boolean
  }
  maxWidthClassName?: string
}

export const SourcesAvailabilityGate: React.FC<SourcesAvailabilityGateProps> = ({
  children,
  capabilityState,
  maxWidthClassName = "max-w-6xl"
}) => {
  const { t } = useTranslation(["sources"])
  const navigate = useNavigate()
  const defaultCapabilityState = useServerCapabilities()
  const { capabilities, loading } = capabilityState ?? defaultCapabilityState
  const unsupportedState = buildCapabilityState({
    kind: "unavailable",
    featureName: t("sources:title", "Sources"),
    capabilityName: t("sources:capability.ingestionSources", "ingestion sources"),
    primaryAction: {
      label: t("sources:actions.checkServerSetup", "Check server setup"),
      onClick: () => {
        navigate("/settings/health")
      }
    }
  })

  return (
    <WorkspaceConnectionGate
      featureName={t("sources:title", "Sources")}
      setupDescription={t(
        "sources:setupRequired",
        "Sources depends on your connected tldw server to manage folders, archive snapshots, and sync rules."
      )}
      maxWidthClassName={maxWidthClassName}
    >
      {!loading && capabilities && !capabilities.hasIngestionSources ? (
        <PageShell className="py-6" maxWidthClassName={maxWidthClassName}>
          <StatePanel
            state={unsupportedState.state}
            title={unsupportedState.title}
            message={unsupportedState.message}
            primaryAction={unsupportedState.primaryAction}
          />
        </PageShell>
      ) : (
        <>{children}</>
      )}
    </WorkspaceConnectionGate>
  )
}
