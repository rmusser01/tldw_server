import React from "react"
import { useTranslation } from "react-i18next"
import { Lightbulb } from "lucide-react"
import { Alert } from "@/components/ui/primitives/Alert"
import { useSetting } from "@/hooks/useSetting"
import { SEEN_HINTS_SETTING } from "@/services/settings/ui-settings"

type FeatureHintProps = {
  /** Unique key to track if this hint has been seen */
  featureKey: string
  /** Short title for the hint */
  title: string
  /** Description explaining the feature */
  description: string
  /** Additional class names */
  className?: string
  /** Whether the hint should be shown (in addition to not being dismissed) */
  show?: boolean
  /** Callback when hint is dismissed */
  onDismiss?: () => void
}

/**
 * First-time feature hint component.
 * Shows guidance in normal document flow that can be dismissed permanently.
 * Tracks seen hints in extension storage to only show once.
 */
export const FeatureHint: React.FC<FeatureHintProps> = ({
  featureKey,
  title,
  description,
  className,
  show = true,
  onDismiss
}) => {
  const { t } = useTranslation(["common"])
  const [seenHints, setSeenHints] = useSetting(SEEN_HINTS_SETTING)
  const [isVisible, setIsVisible] = React.useState(true)

  // Check if this hint has been seen
  const hasBeenSeen = seenHints?.[featureKey] === true

  // Don't render if already seen or explicitly hidden
  if (hasBeenSeen || !isVisible || !show) {
    return null
  }

  const handleDismiss = async () => {
    setIsVisible(false)
    // Mark as seen in storage
    await setSeenHints((prev) => ({
      ...(prev || {}),
      [featureKey]: true
    }))
    onDismiss?.()
  }

  return (
    <Alert
      role="status"
      title={title}
      icon={<Lightbulb className="size-4 text-warn" aria-hidden="true" />}
      dismissible
      dismissLabel={t("common:dismiss", "Dismiss")}
      onDismiss={handleDismiss}
      className={`w-full min-w-0 px-3 py-2 ${className || ""}`}
    >
      <span className="text-xs">{description}</span>
    </Alert>
  )
}

/**
 * Hook to check if a feature hint has been seen
 */
export const useFeatureHintSeen = (featureKey: string) => {
  const [seenHints] = useSetting(SEEN_HINTS_SETTING)

  return seenHints?.[featureKey] === true
}

/**
 * Hook to mark a feature hint as seen programmatically
 */
export const useMarkFeatureHintSeen = () => {
  const [, setSeenHints] = useSetting(SEEN_HINTS_SETTING)

  return React.useCallback(
    async (featureKey: string) => {
      await setSeenHints((prev) => ({
        ...(prev || {}),
        [featureKey]: true
      }))
    },
    [setSeenHints]
  )
}

/**
 * Hook to reset all feature hints (for testing/debugging)
 */
export const useResetFeatureHints = () => {
  const [, setSeenHints] = useSetting(SEEN_HINTS_SETTING)

  return React.useCallback(async () => {
    await setSeenHints({})
  }, [setSeenHints])
}

export default FeatureHint
