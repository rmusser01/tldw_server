import React from "react"
import { cn } from "@/libs/utils"

export type BadgeVariant =
  | "primary"
  | "secondary"
  | "success"
  | "warning"
  | "danger"
  | "info"
  | "demo"

export type BadgeSize = "sm" | "md"

export interface BadgeProps {
  /** Visual style variant */
  variant?: BadgeVariant
  /** Size variant */
  size?: BadgeSize
  /** Badge content */
  children: React.ReactNode
  /** Show as pill (fully rounded) - default true */
  pill?: boolean
  /** Show status dot before text */
  dot?: boolean
  /** Outline style instead of filled */
  outline?: boolean
  /** Screen reader label (if different from visible text) */
  srLabel?: string
  /** Accessible label for the badge itself */
  "aria-label"?: string
  /** Native tooltip text */
  title?: string
  /** Additional CSS classes */
  className?: string
  /** Test ID */
  "data-testid"?: string
}

const variantStyles = {
  primary: {
    filled: "bg-primary/10 text-primary",
    outline: "border-primary/30 text-primary",
    dot: "bg-primary",
  },
  secondary: {
    filled: "bg-surface2 text-text-muted",
    outline: "border-border text-text-muted",
    dot: "bg-muted",
  },
  success: {
    filled: "bg-success/10 text-success",
    outline: "border-success/30 text-success",
    dot: "bg-success",
  },
  warning: {
    filled: "bg-warn/10 text-warn",
    outline: "border-warn/30 text-warn",
    dot: "bg-warn",
  },
  danger: {
    filled: "bg-danger/10 text-danger",
    outline: "border-danger/30 text-danger",
    dot: "bg-danger",
  },
  info: {
    filled: "bg-primary/10 text-primary",
    outline: "border-primary/30 text-primary",
    dot: "bg-primary",
  },
  demo: {
    filled: "bg-primary/10 text-primary",
    outline: "border-primary/30 text-primary",
    dot: "bg-primary",
  },
} as const

const sizeStyles = {
  sm: "px-1.5 py-0.5 text-[10px]",
  md: "px-2 py-0.5 text-xs",
} as const

/**
 * Badge component for status indicators, labels, and counts.
 *
 * Extends and replaces the existing StatusBadge component with more variants.
 *
 * @example
 * ```tsx
 * <Badge variant="success">Complete</Badge>
 * <Badge variant="warning" dot>Pending</Badge>
 * <Badge variant="danger" outline>Error</Badge>
 * <Badge variant="primary" size="sm">New</Badge>
 * ```
 */
export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  (
    {
      variant = "secondary",
      size = "md",
      children,
      pill = true,
      dot = false,
      outline = false,
      srLabel,
      "aria-label": ariaLabel,
      title,
      className,
      "data-testid": dataTestId,
    },
    ref
  ) => {
    const styles = variantStyles[variant]

    return (
      <span
        ref={ref}
        className={cn(
          "inline-flex items-center gap-1 font-medium",
          sizeStyles[size],
          pill ? "rounded-full" : "rounded",
          outline ? `border ${styles.outline} bg-transparent` : styles.filled,
          className
        )}
        aria-label={ariaLabel}
        data-ds-component="Badge"
        data-ds-size={size}
        data-ds-variant={variant}
        data-testid={dataTestId}
        title={title}
      >
        {dot && (
          <span
            className={cn("size-1.5 rounded-full", styles.dot)}
            aria-hidden="true"
          />
        )}
        {children}
        {srLabel && <span className="sr-only">{srLabel}</span>}
      </span>
    )
  }
)

Badge.displayName = "Badge"

export default Badge
