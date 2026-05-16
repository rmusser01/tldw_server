import React from "react"
import { X, Info, CheckCircle, AlertTriangle, XCircle } from "lucide-react"
import { cn } from "@/libs/utils"
import { Button } from "@/components/Common/Button"

export type AlertVariant = "info" | "success" | "warning" | "error"

export interface AlertProps {
  /** Visual style variant */
  variant?: AlertVariant
  /** Optional title displayed prominently */
  title?: React.ReactNode
  /** Optional alert content/message */
  children?: React.ReactNode
  /** Custom icon (defaults to variant-appropriate icon) */
  icon?: React.ReactNode
  /** Primary action button */
  action?: {
    label: string
    onClick: () => void
    loading?: boolean
    disabled?: boolean
  }
  /** Secondary action (text link style) */
  secondaryAction?: {
    label: string
    onClick: () => void
  }
  /** Show dismiss button */
  dismissible?: boolean
  /** Callback when dismissed */
  onDismiss?: () => void
  /** Accessible label for the dismiss button */
  dismissLabel?: string
  /** ARIA role for alert urgency */
  role?: "alert" | "status"
  /** Live-region politeness for status-style alerts */
  "aria-live"?: "off" | "polite" | "assertive"
  /** Additional CSS classes */
  className?: string
  /** Test ID for testing */
  "data-testid"?: string
}

const variantConfig = {
  info: {
    container: "border-primary/30 bg-primary/10",
    text: "text-primary",
    icon: Info,
  },
  success: {
    container: "border-success/30 bg-success/10",
    text: "text-success",
    icon: CheckCircle,
  },
  warning: {
    container: "border-warn/30 bg-warn/10",
    text: "text-warn",
    icon: AlertTriangle,
  },
  error: {
    container: "border-danger/30 bg-danger/10",
    text: "text-danger",
    icon: XCircle,
  },
} as const

/**
 * Alert component for inline messages, warnings, and notifications.
 *
 * Replaces scattered `bg-{color}/10 border-{color}` patterns throughout the codebase.
 *
 * @example
 * ```tsx
 * <Alert variant="warning" title="Connection issue">
 *   Unable to reach server. Retrying...
 * </Alert>
 *
 * <Alert
 *   variant="error"
 *   action={{ label: "Retry", onClick: handleRetry, loading: isRetrying }}
 *   dismissible
 *   onDismiss={() => setDismissed(true)}
 * >
 *   Failed to save changes.
 * </Alert>
 * ```
 */
export const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
  (
    {
      variant = "info",
      title,
      children,
      icon,
      action,
      secondaryAction,
      dismissible = false,
      onDismiss,
      dismissLabel = "Dismiss",
      role = "alert",
      "aria-live": ariaLive,
      className,
      "data-testid": dataTestId,
    },
    ref
  ) => {
    const config = variantConfig[variant]
    const DefaultIcon = config.icon
    const hasContent = React.Children.toArray(children).some((child) => child !== "")

    return (
      <div
        ref={ref}
        className={cn(
          "flex items-start gap-3 rounded-lg border p-3",
          config.container,
          className
        )}
        role={role}
        aria-live={ariaLive}
        data-testid={dataTestId}
        data-ds-component="Alert"
      >
        <span
          className={cn("mt-0.5 flex-shrink-0", config.text)}
          aria-hidden="true"
        >
          {icon || <DefaultIcon className="size-4" />}
        </span>

        <div className="min-w-0 flex-1">
          {title && (
            <p className={cn("text-sm font-medium", config.text)}>{title}</p>
          )}
          {hasContent && (
            <div className={cn("text-sm", config.text, title && "mt-1")}>
              {children}
            </div>
          )}

          {(action || secondaryAction) && (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              {action && (
                <Button
                  size="sm"
                  onClick={action.onClick}
                  loading={action.loading}
                  disabled={action.disabled}
                >
                  {action.label}
                </Button>
              )}
              {secondaryAction && (
                <button
                  type="button"
                  onClick={secondaryAction.onClick}
                  className={cn(
                    "text-xs font-medium underline transition-colors duration-150",
                    config.text,
                    "hover:opacity-80"
                  )}
                >
                  {secondaryAction.label}
                </button>
              )}
            </div>
          )}
        </div>

        {dismissible && onDismiss && (
          <button
            type="button"
            onClick={onDismiss}
            className={cn(
              "flex-shrink-0 rounded p-1 transition-colors duration-150 hover:bg-surface2",
              config.text
            )}
            aria-label={dismissLabel}
          >
            <X className="size-4" />
          </button>
        )}
      </div>
    )
  }
)

Alert.displayName = "Alert"

export default Alert
