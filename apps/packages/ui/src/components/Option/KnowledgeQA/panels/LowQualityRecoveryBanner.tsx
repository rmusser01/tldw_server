import React from "react"
import { Globe, Layers, Search, X } from "lucide-react"
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
      primaryAction={{
        label: (
          <span className="inline-flex items-center gap-1">
            <Search className="h-3 w-3" />
            {refineLabel}
          </span>
        ),
        onClick: onRefine
      }}
      secondaryActions={[
        {
          label: (
            <span className="inline-flex items-center gap-1">
              <Globe className="h-3 w-3" />
              {enableWebLabel}
            </span>
          ),
          onClick: onEnableWeb
        },
        {
          label: (
            <span className="inline-flex items-center gap-1">
              <Layers className="h-3 w-3" />
              {selectSourcesLabel}
            </span>
          ),
          onClick: onSelectSources
        },
        {
          label: (
            <span className="inline-flex items-center gap-1">
              <X className="h-3 w-3" />
              Dismiss
            </span>
          ),
          onClick: onDismiss
        }
      ]}
    />
  )
}
