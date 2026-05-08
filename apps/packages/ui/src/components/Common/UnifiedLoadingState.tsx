import React, { useEffect, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { translateMessage } from "@/i18n/translateMessage"
import { LoadingState } from "@/components/ui/feedback/LoadingState"

export interface LoadingSource {
  key: string
  loading: boolean
  label?: string
}

interface UnifiedLoadingStateProps {
  /** Array of loading sources to track */
  sources: LoadingSource[]
  /** Number of skeleton rows (default: 4) */
  rows?: number
  /** Custom className */
  className?: string
  /** Whether to show individual source labels during loading */
  showLabels?: boolean
  /** Children to render when all sources are done loading */
  children?: React.ReactNode
}

/**
 * Unified loading state component that combines multiple loading sources
 * into a single skeleton display. Shows loading until ALL sources complete.
 *
 * Usage:
 * ```tsx
 * <UnifiedLoadingState
 *   sources={[
 *     { key: "local", loading: isLocalLoading, label: "Local data" },
 *     { key: "server", loading: isServerLoading, label: "Server sync" },
 *     { key: "folders", loading: isFoldersLoading, label: "Folder structure" }
 *   ]}
 *   showLabels={true}
 * >
 *   <YourContent />
 * </UnifiedLoadingState>
 * ```
 */
export function UnifiedLoadingState({
  sources,
  rows = 4,
  className,
  showLabels = false,
  children
}: UnifiedLoadingStateProps) {
  const { t } = useTranslation(["common"])
  const loadingSources = useMemo(
    () => sources.filter((source) => source.loading),
    [sources]
  )

  useEffect(() => {
    if (!showLabels) return
    if (process.env.NODE_ENV === "production") return
    const missingLabels = loadingSources
      .filter((source) => !source.label?.trim())
      .map((source) => source.key)
    if (missingLabels.length > 0) {
      // eslint-disable-next-line no-console
      console.warn(
        "[UnifiedLoadingState] Missing labels for loading sources:",
        missingLabels
      )
    }
  }, [loadingSources, showLabels])

  const translatedLoadingSources = useMemo(
    () =>
      loadingSources.map((source) => {
        const label = showLabels
          ? source.label
            ? translateMessage(t, source.label, source.label)
            : translateMessage(
                t,
                `common:loadingSource.${source.key}`,
                `Loading: ${source.key}`
              )
          : source.label

        return {
          ...source,
          label
        }
      }),
    [loadingSources, showLabels, t]
  )

  if (loadingSources.length === 0) {
    return <>{children}</>
  }

  return (
    <LoadingState
      mode="skeleton"
      rows={rows}
      sources={translatedLoadingSources}
      showLabels={showLabels}
      className={className}
    >
      {children}
    </LoadingState>
  )
}

/**
 * Hook to manage multiple loading sources and determine unified loading state
 */
export function useUnifiedLoading(
  sources: Array<{ key: string; loading: boolean; label?: string }>
) {
  return useMemo(() => {
    const isLoading = sources.some((s) => s.loading)
    const loadingSources = sources.filter((s) => s.loading)
    const completedSources = sources.filter((s) => !s.loading)

    return {
      isLoading,
      loadingSources,
      completedSources,
      progress:
        sources.length > 0
          ? Math.round((completedSources.length / sources.length) * 100)
          : 100
    }
  }, [sources])
}

export default UnifiedLoadingState
