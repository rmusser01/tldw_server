import React from "react"
import { Bookmark, BookOpen, CheckCircle, Archive, type LucideIcon } from "lucide-react"
import { useTranslation } from "react-i18next"
import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"
import type { ReadingStatus } from "@/types/collections"

interface StatusBadgeProps {
  status: ReadingStatus
  size?: "small" | "default"
}

const STATUS_CONFIG: Record<
  ReadingStatus,
  {
    stateKey: DesignSystemStateKey
    icon: LucideIcon
    labelKey: string
  }
> = {
  saved: {
    stateKey: "ready",
    icon: Bookmark,
    labelKey: "saved"
  },
  reading: {
    stateKey: "retrying",
    icon: BookOpen,
    labelKey: "reading"
  },
  read: {
    stateKey: "ready",
    icon: CheckCircle,
    labelKey: "read"
  },
  archived: {
    stateKey: "empty",
    icon: Archive,
    labelKey: "archived"
  }
}

const SEVERITY_BADGE_VARIANTS = {
  success: "success",
  error: "danger",
  warning: "warning",
  info: "info",
  neutral: "secondary",
} satisfies Record<ReturnType<typeof getDesignSystemState>["severity"], BadgeVariant>

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = "default"
}) => {
  const { t } = useTranslation("collections")
  const config = STATUS_CONFIG[status]
  const state = getDesignSystemState(config.stateKey)
  const Icon = config.icon

  return (
    <Badge
      variant={SEVERITY_BADGE_VARIANTS[state.severity]}
      size={size === "small" ? "sm" : "md"}
      className={size === "small" ? "py-0 text-xs" : undefined}
    >
      <Icon
        className={size === "small" ? "h-3 w-3" : "h-3.5 w-3.5"}
        data-testid={`collections-status-icon-${status}`}
        aria-hidden
      />
      <span>{t(`status.${config.labelKey}`, config.labelKey)}</span>
    </Badge>
  )
}
