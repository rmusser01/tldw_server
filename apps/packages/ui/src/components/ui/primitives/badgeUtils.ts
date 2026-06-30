import type { DesignSystemSeverity } from "@/design-system"
import type { BadgeVariant } from "./Badge"

export const DESIGN_SYSTEM_SEVERITY_BADGE_VARIANTS = {
  success: "success",
  error: "danger",
  warning: "warning",
  info: "info",
  neutral: "secondary",
} satisfies Record<DesignSystemSeverity, BadgeVariant>

export function getBadgeVariantForDesignSystemSeverity(
  severity: DesignSystemSeverity
): BadgeVariant {
  return DESIGN_SYSTEM_SEVERITY_BADGE_VARIANTS[severity]
}
