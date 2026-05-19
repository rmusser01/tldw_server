import React from "react"

import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system"
import {
  Badge,
  getBadgeVariantForDesignSystemSeverity
} from "@/components/ui/primitives"
import { cn } from "@/libs/utils"
import type { PresentationStudioAssetStatus } from "@/store/presentation-studio"

type PresentationStudioStatusBadgeProps = {
  status: PresentationStudioAssetStatus | null | undefined
  className?: string
}

const STATUS_STATES = {
  missing: "empty",
  ready: "ready",
  stale: "degraded",
  generating: "retrying",
  failed: "error"
} satisfies Record<PresentationStudioAssetStatus, DesignSystemStateKey>

export const PresentationStudioStatusBadge: React.FC<
  PresentationStudioStatusBadgeProps
> = ({ status, className }) => {
  const normalized: PresentationStudioAssetStatus = status || "missing"
  const state = getDesignSystemState(STATUS_STATES[normalized])

  return (
    <Badge
      className={cn("capitalize", className)}
      dot
      size="sm"
      variant={getBadgeVariantForDesignSystemSeverity(state.severity)}
    >
      {normalized}
    </Badge>
  )
}
