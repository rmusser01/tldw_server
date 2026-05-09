/**
 * StatusBadge component
 * Displays run status with color coding and optional loading animation
 */

import React from "react"
import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"
import { Loader2 } from "lucide-react"

export type RunStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | string

interface StatusBadgeProps {
  status: RunStatus
  className?: string
}

interface StatusConfig {
  stateKey: DesignSystemStateKey
  icon?: React.ReactNode
}

const STATUS_CONFIG: Record<string, StatusConfig> = {
  pending: { stateKey: "loading" },
  running: {
    stateKey: "retrying",
    icon: (
      <Loader2
        className="inline h-3 w-3 animate-spin"
        aria-hidden
        data-testid="evaluations-status-running-spinner"
      />
    )
  },
  completed: { stateKey: "ready" },
  failed: { stateKey: "error" },
  cancelled: { stateKey: "degraded" }
}

const SEVERITY_BADGE_VARIANTS = {
  success: "success",
  error: "danger",
  warning: "warning",
  info: "info",
  neutral: "secondary",
} satisfies Record<ReturnType<typeof getDesignSystemState>["severity"], BadgeVariant>

const UNKNOWN_STATUS_CONFIG = {
  stateKey: "empty",
} satisfies StatusConfig

function getStatusConfig(status: string): StatusConfig {
  if (Object.prototype.hasOwnProperty.call(STATUS_CONFIG, status)) {
    return STATUS_CONFIG[status]
  }

  return UNKNOWN_STATUS_CONFIG
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  className
}) => {
  const normalizedStatus = String(status || "").toLowerCase()
  const config = getStatusConfig(normalizedStatus)
  const state = getDesignSystemState(config.stateKey)

  return (
    <Badge
      variant={SEVERITY_BADGE_VARIANTS[state.severity]}
      className={className}
    >
      {config.icon}
      {status}
    </Badge>
  )
}

export default StatusBadge
