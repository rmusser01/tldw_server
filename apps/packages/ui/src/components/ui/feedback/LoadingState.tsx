import React from "react"
import { Skeleton } from "antd"
import { Loader2 } from "lucide-react"
import { cn } from "@/libs/utils"

export type LoadingMode = "spinner" | "skeleton" | "dots" | "inline"
export type LoadingSize = "sm" | "md" | "lg"

export interface LoadingSource {
  /** Unique key for the source */
  key: string
  /** Whether this source is currently loading */
  loading: boolean
  /** Optional label to display */
  label?: string
}

export interface LoadingStateProps {
  /** Display mode */
  mode?: LoadingMode
  /** For skeleton mode: number of rows */
  rows?: number
  /** For skeleton mode: show avatar placeholder */
  avatar?: boolean
  /** Loading sources to track (for multi-source loading) */
  sources?: LoadingSource[]
  /** Show individual source labels */
  showLabels?: boolean
  /** Loading text/label */
  label?: string
  /** Size variant */
  size?: LoadingSize
  /** Cover the entire screen */
  fullscreen?: boolean
  /** Show as overlay on content */
  overlay?: boolean
  /** Children to render when not loading */
  children?: React.ReactNode
  /** Force loading state (overrides sources) */
  loading?: boolean
  /** Additional CSS classes */
  className?: string
  /** Inline style for surface-specific sizing; ignored for fullscreen mode */
  style?: React.CSSProperties
  /** Test ID */
  "data-testid"?: string
}

const sizeConfig = {
  sm: {
    spinner: "size-4",
    text: "text-xs",
    skeleton: 2,
  },
  md: {
    spinner: "size-5",
    text: "text-sm",
    skeleton: 4,
  },
  lg: {
    spinner: "size-6",
    text: "text-base",
    skeleton: 6,
  },
} as const

/**
 * Unified loading state component.
 *
 * Consolidates Spin, Skeleton, Loader2, and UnifiedLoadingState patterns.
 *
 * @example
 * ```tsx
 * // Simple spinner
 * <LoadingState mode="spinner" label="Loading..." />
 *
 * // Skeleton with rows
 * <LoadingState mode="skeleton" rows={4} />
 *
 * // Multi-source loading
 * <LoadingState
 *   sources={[
 *     { key: "local", loading: isLocalLoading, label: "Local data" },
 *     { key: "server", loading: isServerLoading, label: "Server" },
 *   ]}
 *   showLabels
 * >
 *   <Content />
 * </LoadingState>
 *
 * // Inline spinner
 * <LoadingState mode="inline" size="sm" />
 * ```
 */
export const LoadingState = React.forwardRef<HTMLDivElement, LoadingStateProps>(
  (
    {
      mode = "skeleton",
      rows,
      avatar = false,
      sources,
      showLabels = false,
      label,
      size = "md",
      fullscreen = false,
      overlay = false,
      children,
      loading: forcedLoading,
      className,
      style,
      "data-testid": dataTestId,
    },
    ref
  ) => {
    const sizeStyles = sizeConfig[size]

    // Determine if we're loading based on sources or forced loading prop
    const loadingSources = sources?.filter((s) => s.loading) || []
    const isLoading =
      forcedLoading !== undefined
        ? forcedLoading
        : sources
          ? loadingSources.length > 0
          : true

    // If not loading and children provided, render children
    if (!isLoading && children) {
      return <>{children}</>
    }

    // If not loading and no children, render nothing
    if (!isLoading) {
      return null
    }

    const renderSpinner = () => (
      <Loader2
        className={cn(sizeStyles.spinner, "animate-spin text-primary")}
        aria-hidden="true"
      />
    )

    const renderDots = () => (
      <div className="flex items-center gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="size-1.5 animate-pulse rounded-full bg-primary"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </div>
    )

    const renderSkeleton = () => (
      <Skeleton
        active
        avatar={avatar}
        paragraph={{ rows: rows ?? sizeStyles.skeleton }}
        className="w-full"
      />
    )

    const renderContent = () => {
      switch (mode) {
        case "spinner":
          return (
            <div className="flex flex-col items-center gap-2">
              {renderSpinner()}
              {label && (
                <span className={cn("text-text-muted", sizeStyles.text)}>
                  {label}
                </span>
              )}
            </div>
          )
        case "dots":
          return (
            <div className="flex flex-col items-center gap-2">
              {renderDots()}
              {label && (
                <span className={cn("text-text-muted", sizeStyles.text)}>
                  {label}
                </span>
              )}
            </div>
          )
        case "inline":
          return renderSpinner()
        case "skeleton":
        default:
          return renderSkeleton()
      }
    }

    const renderSourceLabels = () => {
      if (!showLabels || loadingSources.length === 0) return null

      return (
        <div className="mb-3 flex flex-wrap justify-center gap-1">
          {loadingSources.map((source) => (
            <span
              key={source.key}
              className="inline-flex items-center gap-1 rounded-full bg-surface2 px-2 py-0.5 text-xs text-text-subtle"
            >
              <span className="size-1.5 animate-pulse rounded-full bg-primary" />
              {source.label || source.key}
            </span>
          ))}
        </div>
      )
    }

    if (mode === "inline" && !fullscreen && !overlay) {
      return (
        <div
          ref={ref}
          className={cn("inline-flex items-center gap-2", className)}
          style={style}
          data-testid={dataTestId}
          data-ds-component="LoadingState"
        >
          {renderContent()}
          {label && (
            <span className={cn("text-text-muted", sizeStyles.text)}>
              {label}
            </span>
          )}
        </div>
      )
    }

    // Fullscreen loading
    if (fullscreen) {
      return (
        <div
          ref={ref}
          className={cn(
            "fixed inset-0 z-50 flex items-center justify-center bg-bg/80 backdrop-blur-sm",
            className
          )}
          data-testid={dataTestId}
          data-ds-component="LoadingState"
        >
          {renderSourceLabels()}
          {renderContent()}
        </div>
      )
    }

    // Overlay loading
    if (overlay) {
      return (
        <div
          ref={ref}
          className={cn("relative", className)}
          style={style}
          data-ds-component="LoadingState"
        >
          {children}
          <div
            className="absolute inset-0 flex items-center justify-center bg-bg/60 backdrop-blur-[2px]"
            data-testid={dataTestId}
          >
            {renderSourceLabels()}
            {renderContent()}
          </div>
        </div>
      )
    }

    // Standard loading
    return (
      <div
        ref={ref}
        className={cn("flex flex-col items-center px-2 py-4", className)}
        style={style}
        data-testid={dataTestId}
        data-ds-component="LoadingState"
      >
        {renderSourceLabels()}
        {renderContent()}
      </div>
    )
  }
)

LoadingState.displayName = "LoadingState"

export default LoadingState
