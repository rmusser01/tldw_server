import React from "react"
import { AlertTriangle, Ban, CheckCircle2, Circle, Clock3, LoaderCircle, type LucideIcon } from "lucide-react"
import { useTranslation } from "react-i18next"
import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"
import type { RunStatus } from "@/types/watchlists"

interface StatusTagProps {
  status: RunStatus | string
  size?: "small" | "default"
}

type StatusIconToken = "pending" | "running" | "completed" | "failed" | "cancelled" | "unknown"

const STATUS_CONFIG: Record<string, {
  stateKey: DesignSystemStateKey
  labelKey: string
  fallbackLabel: string
  iconToken: StatusIconToken
  icon: LucideIcon
}> = {
  pending: {
    stateKey: "loading",
    labelKey: "watchlists:runs.statusLabels.pending",
    fallbackLabel: "Pending",
    iconToken: "pending",
    icon: Clock3
  },
  queued: {
    stateKey: "loading",
    labelKey: "watchlists:runs.statusLabels.queued",
    fallbackLabel: "Queued",
    iconToken: "pending",
    icon: Clock3
  },
  running: {
    stateKey: "retrying",
    labelKey: "watchlists:runs.statusLabels.running",
    fallbackLabel: "Running",
    iconToken: "running",
    icon: LoaderCircle
  },
  completed: {
    stateKey: "ready",
    labelKey: "watchlists:runs.statusLabels.completed",
    fallbackLabel: "Completed",
    iconToken: "completed",
    icon: CheckCircle2
  },
  failed: {
    stateKey: "error",
    labelKey: "watchlists:runs.statusLabels.failed",
    fallbackLabel: "Failed",
    iconToken: "failed",
    icon: AlertTriangle
  },
  cancelled: {
    stateKey: "degraded",
    labelKey: "watchlists:runs.statusLabels.cancelled",
    fallbackLabel: "Cancelled",
    iconToken: "cancelled",
    icon: Ban
  }
}

const UNKNOWN_STATUS_CONFIG = {
  stateKey: "empty",
  iconToken: "unknown",
  icon: Circle
} satisfies {
  stateKey: DesignSystemStateKey
  iconToken: StatusIconToken
  icon: LucideIcon
}

const SEVERITY_BADGE_VARIANTS = {
  success: "success",
  error: "danger",
  warning: "warning",
  info: "info",
  neutral: "secondary",
} satisfies Record<ReturnType<typeof getDesignSystemState>["severity"], BadgeVariant>

const toTitleCase = (value: string): string =>
  value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase())

export const StatusTag: React.FC<StatusTagProps> = ({ status, size = "default" }) => {
  const { t } = useTranslation(["watchlists"])
  const normalizedStatus = String(status || "").trim().toLowerCase()
  const config = STATUS_CONFIG[normalizedStatus]
  const statusConfig = config || UNKNOWN_STATUS_CONFIG
  const fallbackLabel = normalizedStatus
    ? toTitleCase(normalizedStatus)
    : t("watchlists:runs.statusLabels.unknown", "Unknown")
  const label = config
    ? t(config.labelKey, config.fallbackLabel)
    : fallbackLabel
  const ariaLabel = t("watchlists:runs.statusAria", "Run status: {{status}}", { status: label })
  const state = getDesignSystemState(statusConfig.stateKey)
  const Icon = statusConfig.icon

  return (
    <Badge
      variant={SEVERITY_BADGE_VARIANTS[state.severity]}
      size={size === "small" ? "sm" : "md"}
      aria-label={ariaLabel}
      title={ariaLabel}
    >
      <span data-testid={`watchlists-status-icon-${statusConfig.iconToken}`}>
        <Icon className={size === "small" ? "h-3 w-3" : "h-3.5 w-3.5"} aria-hidden />
      </span>
      <span>{label}</span>
    </Badge>
  )
}
