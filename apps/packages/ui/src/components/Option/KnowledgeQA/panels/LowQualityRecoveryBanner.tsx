import React from "react"
import { Globe, Layers, Search, X, type LucideIcon } from "lucide-react"
import { RecoveryCallout } from "@/components/ui/state"
import type { KnowledgeSourceStatus } from "../types"

type LowQualityRecoveryBannerProps = {
  onRefine: () => void
  onEnableWeb: () => void
  onSelectSources: () => void
  onDismiss: () => void
  title?: string
  description?: string
  refineLabel?: string
  enableWebLabel?: string
  selectSourcesLabel?: string
  sourceStatus?: Record<string, KnowledgeSourceStatus>
}

type ActionLabelProps = {
  icon: LucideIcon
  children: React.ReactNode
}

function ActionLabel({ icon: Icon, children }: ActionLabelProps) {
  return (
    <span className="inline-flex items-center gap-1">
      <Icon className="h-3 w-3" />
      {children}
    </span>
  )
}

function formatSourceDiagnosticsSummary(
  sourceStatus?: Record<string, KnowledgeSourceStatus>
): string | null {
  const entries = Object.values(sourceStatus ?? {})
  if (entries.length === 0) return null

  const searched = entries.filter((entry) => entry.status === "searched").length
  const empty = entries.filter((entry) => entry.status === "empty").length
  const unavailable = entries.filter((entry) => entry.status === "unavailable").length
  const parts = [
    `${searched} searched`,
    `${empty} empty`,
    `${unavailable} unavailable`,
  ]
  return `Source diagnostics: ${parts.join(", ")}.`
}

export function LowQualityRecoveryBanner({
  onRefine,
  onEnableWeb,
  onSelectSources,
  onDismiss,
  title = "These sources may not closely match your question.",
  description = "Try refining your search:",
  refineLabel = "Use more specific terms",
  enableWebLabel = "Include web sources",
  selectSourcesLabel = "Select different sources",
  sourceStatus,
}: LowQualityRecoveryBannerProps) {
  const sourceDiagnosticsSummary = formatSourceDiagnosticsSummary(sourceStatus)

  return (
    <RecoveryCallout
      state="degraded"
      title={title}
      message={
        sourceDiagnosticsSummary ? (
          <>
            <p>{description}</p>
            <p className="mt-1">{sourceDiagnosticsSummary}</p>
          </>
        ) : (
          description
        )
      }
      role="status"
      aria-live="polite"
      aria-atomic="true"
      primaryAction={{
        label: <ActionLabel icon={Search}>{refineLabel}</ActionLabel>,
        onClick: onRefine
      }}
      secondaryActions={[
        {
          label: <ActionLabel icon={Globe}>{enableWebLabel}</ActionLabel>,
          onClick: onEnableWeb
        },
        {
          label: <ActionLabel icon={Layers}>{selectSourcesLabel}</ActionLabel>,
          onClick: onSelectSources
        },
        {
          label: <ActionLabel icon={X}>Dismiss</ActionLabel>,
          ariaLabel: "Dismiss recovery suggestions",
          onClick: onDismiss
        }
      ]}
    />
  )
}
