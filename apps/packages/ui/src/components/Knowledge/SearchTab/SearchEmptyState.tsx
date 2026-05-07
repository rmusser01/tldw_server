import React from "react"
import { AlertCircle, Search, RefreshCw } from "lucide-react"
import { useTranslation } from "react-i18next"
import { EmptyState } from "@/components/ui/feedback/EmptyState"

type EmptyStateVariant = "initial" | "no-results" | "timeout" | "disconnected"

type SearchEmptyStateProps = {
  variant: EmptyStateVariant
  onRetry?: () => void
  onDismissHint?: () => void
  showHint?: boolean
}

/**
 * Empty state display for various search states
 */
export const SearchEmptyState: React.FC<SearchEmptyStateProps> = ({
  variant,
  onRetry,
  onDismissHint,
  showHint = false
}) => {
  const { t } = useTranslation(["sidepanel"])

  if (variant === "initial") {
    return (
      <EmptyState
        title={t("sidepanel:rag.initialState", "No results yet")}
        description={t(
          "sidepanel:rag.hint.message",
          "Search your knowledge base and insert results into your message."
        )}
        icon={Search}
        iconClassName="text-text-muted"
        size="md"
        variant="inline"
        className="py-8"
        secondaryAction={
          showHint && onDismissHint
            ? {
                label: t("sidepanel:rag.hint.dismiss", "Dismiss"),
                onClick: onDismissHint
              }
            : undefined
        }
      />
    )
  }

  if (variant === "no-results") {
    return (
      <EmptyState
        title={t("sidepanel:rag.noResults", "No results found")}
        description={t(
          "sidepanel:rag.tryDifferentQuery",
          "Try a different search query or adjust your filters."
        )}
        icon={Search}
        iconClassName="text-text-muted"
        size="md"
        variant="inline"
        className="py-8"
      />
    )
  }

  if (variant === "timeout") {
    return (
      <EmptyState
        title={t("sidepanel:rag.timeout.message", "Request timed out.")}
        icon={AlertCircle}
        iconClassName="text-warn"
        size="md"
        variant="inline"
        className="py-8"
        primaryAction={
          onRetry
            ? {
                label: t("sidepanel:rag.timeout.retry", "Retry"),
                icon: <RefreshCw className="h-3.5 w-3.5" />,
                onClick: onRetry
              }
            : undefined
        }
      />
    )
  }

  if (variant === "disconnected") {
    return (
      <EmptyState
        title={t(
          "sidepanel:rag.disconnected",
          "Connect to server to search knowledge base"
        )}
        icon={AlertCircle}
        iconClassName="text-error"
        size="md"
        variant="inline"
        className="py-8"
      />
    )
  }

  return null
}
