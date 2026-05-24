import React from "react"
import type { LucideIcon } from "lucide-react"
import { cn } from "@/libs/utils"
import { Button } from "@/components/Common/Button"

export type EmptyStateVariant = "card" | "inline" | "fullPage"
export type EmptyStateSize = "sm" | "md" | "lg"

export interface EmptyStateAction {
  /** Button label */
  label: React.ReactNode
  /** Click handler */
  onClick?: () => void
  /** Show loading spinner */
  loading?: boolean
  /** Disable the button */
  disabled?: boolean
}

export interface EmptyStateStep {
  /** Step icon */
  icon: LucideIcon
  /** Step text */
  text: string
}

export interface EmptyStateSuggestion {
  /** Suggestion icon */
  icon: LucideIcon
  /** Suggestion label */
  label: string
  /** Click handler */
  onClick: () => void
}

export interface EmptyStateProps {
  /** Main title */
  title: React.ReactNode
  /** Description text */
  description?: React.ReactNode
  /** Icon to display above title */
  icon?: LucideIcon
  /** Icon color class */
  iconClassName?: string
  /** Example items to display as bullets */
  examples?: React.ReactNode[]
  /** Steps to display */
  steps?: EmptyStateStep[]
  /** Quick action suggestions */
  suggestions?: EmptyStateSuggestion[]
  /** Primary action button */
  primaryAction?: EmptyStateAction
  /** Secondary action button */
  secondaryAction?: EmptyStateAction
  /** Layout variant */
  variant?: EmptyStateVariant
  /** Size variant */
  size?: EmptyStateSize
  /** Additional CSS classes */
  className?: string
  /** Test ID */
  "data-testid"?: string
}

const sizeConfig = {
  sm: {
    container: "p-4 max-w-sm",
    icon: "h-6 w-6",
    iconWrapper: "p-2",
    title: "text-sm",
    description: "text-xs",
  },
  md: {
    container: "p-6 max-w-md",
    icon: "h-8 w-8",
    iconWrapper: "p-3",
    title: "text-base",
    description: "text-sm",
  },
  lg: {
    container: "p-8 max-w-xl",
    icon: "h-10 w-10",
    iconWrapper: "p-4",
    title: "text-lg",
    description: "text-sm",
  },
} as const

const variantConfig = {
  card: "rounded-2xl border border-border/80 bg-surface/90 shadow-card backdrop-blur",
  inline: "bg-transparent",
  fullPage: "min-h-[400px] flex items-center justify-center",
} as const

/**
 * EmptyState component for displaying when there's no content.
 *
 * Consolidates FeatureEmptyState, PlaygroundEmpty, and EmptySidePanel patterns.
 *
 * @example
 * ```tsx
 * // Basic usage
 * <EmptyState
 *   icon={FileText}
 *   title="No documents"
 *   description="Upload a document to get started"
 *   primaryAction={{ label: "Upload", onClick: handleUpload }}
 * />
 *
 * // With examples
 * <EmptyState
 *   title="Try asking"
 *   examples={["Summarize this article", "What are the key points?"]}
 * />
 *
 * // Inline variant (no card styling)
 * <EmptyState
 *   variant="inline"
 *   title="No results"
 *   description="Try a different search term"
 * />
 * ```
 */
export const EmptyState = React.forwardRef<HTMLDivElement, EmptyStateProps>(
  (
    {
      title,
      description,
      icon: Icon,
      iconClassName,
      examples,
      steps,
      suggestions,
      primaryAction,
      secondaryAction,
      variant = "card",
      size = "md",
      className,
      "data-testid": dataTestId,
    },
    ref
  ) => {
    const sizeStyles = sizeConfig[size]
    const variantStyles = variantConfig[variant]

    return (
      <div
        ref={ref}
        className={cn(
          "mx-auto text-center",
          sizeStyles.container,
          variantStyles,
          className
        )}
        data-testid={dataTestId}
      >
        <div className="space-y-3">
          {Icon && (
            <div className="flex justify-center">
              <div
                className={cn(
                  "rounded-full bg-surface2/80",
                  sizeStyles.iconWrapper
                )}
              >
                <Icon
                  className={cn(
                    sizeStyles.icon,
                    iconClassName || "text-text-subtle"
                  )}
                  aria-hidden="true"
                />
              </div>
            </div>
          )}

          <h2
            className={cn(
              "font-semibold text-text",
              sizeStyles.title,
              Icon && "text-center"
            )}
          >
            {title}
          </h2>

          {description &&
            (typeof description === "string" || typeof description === "number" ? (
              <p className={cn("text-text-muted", sizeStyles.description)}>
                {description}
              </p>
            ) : (
              <div className={cn("text-text-muted", sizeStyles.description)}>
                {description}
              </div>
            ))}

          {examples && examples.length > 0 && (
            <div className="text-left">
              <ul
                className={cn(
                  "list-disc space-y-1 pl-4 text-text-muted",
                  sizeStyles.description
                )}
              >
                {examples.map((example, index) => (
                  <li key={index}>{example}</li>
                ))}
              </ul>
            </div>
          )}

          {steps && steps.length > 0 && (
            <div className="space-y-2 pt-2">
              {steps.map((step, index) => (
                <div
                  key={index}
                  className="flex items-center gap-2 text-left text-text-muted"
                >
                  <step.icon className="size-4 flex-shrink-0" aria-hidden />
                  <span className={sizeStyles.description}>{step.text}</span>
                </div>
              ))}
            </div>
          )}

          {suggestions && suggestions.length > 0 && (
            <div className="flex flex-wrap justify-center gap-2 pt-2">
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={suggestion.onClick}
                  className="inline-flex items-center gap-1.5 rounded-full bg-surface2 px-3 py-1.5 text-xs text-text-muted transition-colors hover:bg-surface hover:text-text"
                >
                  <suggestion.icon className="size-3.5" aria-hidden />
                  {suggestion.label}
                </button>
              ))}
            </div>
          )}

          {(primaryAction || secondaryAction) && (
            <div className="flex flex-wrap justify-center gap-2 pt-3">
              {primaryAction && (
                <Button
                  variant="primary"
                  size="sm"
                  onClick={primaryAction.onClick}
                  loading={primaryAction.loading}
                  disabled={primaryAction.disabled}
                >
                  {primaryAction.label}
                </Button>
              )}
              {secondaryAction && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={secondaryAction.onClick}
                  loading={secondaryAction.loading}
                  disabled={secondaryAction.disabled}
                >
                  {secondaryAction.label}
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    )
  }
)

EmptyState.displayName = "EmptyState"

export default EmptyState
