import React from "react"
import { Globe, Layers, Search, X, type LucideIcon } from "lucide-react"
import { RecoveryCallout } from "@/components/ui/state"

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
}: LowQualityRecoveryBannerProps) {
  return (
    <RecoveryCallout
      state="degraded"
      title={title}
      message={description}
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
