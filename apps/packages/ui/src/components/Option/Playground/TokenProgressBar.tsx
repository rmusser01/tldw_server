import React from "react"
import { Tooltip } from "antd"
import { useTranslation } from "react-i18next"

interface TokenProgressBarProps {
  /** Current conversation token count (excluding draft) */
  conversationTokens: number
  /** Draft message token count */
  draftTokens: number
  /** Maximum context window size */
  maxTokens: number | null
  /** Model label for tooltip */
  modelLabel?: string
  /** Whether to show compact version */
  compact?: boolean
  /** Optional click handler for opening context settings */
  onClick?: () => void
}

const getProgressColor = (percentage: number): string => {
  if (percentage >= 80) return "bg-danger"
  if (percentage >= 50) return "bg-warn"
  return "bg-success"
}

const getProgressColorText = (percentage: number): string => {
  if (percentage >= 80) return "text-danger"
  if (percentage >= 50) return "text-warn"
  return "text-success"
}

export const TokenProgressBar: React.FC<TokenProgressBarProps> = ({
  conversationTokens,
  draftTokens,
  maxTokens,
  modelLabel,
  compact = false,
  onClick
}) => {
  const { t } = useTranslation(["playground", "common"])
  const isInteractive = typeof onClick === "function"

  const totalTokens = conversationTokens + draftTokens
  const percentage = maxTokens && maxTokens > 0
    ? Math.min(100, Math.round((totalTokens / maxTokens) * 100))
    : 0

  const formatNumber = React.useCallback((value: number | null) => {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—"
    return new Intl.NumberFormat().format(Math.round(value))
  }, [])

  const tooltipContent = React.useMemo(() => {
    const lines = [
      modelLabel && `${t("playground:tokens.model", "Model")}: ${modelLabel}`,
      `${t("playground:tokens.draft", "This message")}: ~${formatNumber(draftTokens)} ${t("playground:tokens.tokenUnit", "tokens")}`,
      `${t("playground:tokens.conversation", "Conversation")}: ${formatNumber(conversationTokens)} ${t("playground:tokens.tokenUnit", "tokens")}`,
      maxTokens && `${t("playground:tokens.contextWindow", "Context window")}: ${formatNumber(maxTokens)} ${t("playground:tokens.tokenUnit", "tokens")}`,
      maxTokens && `${t("playground:tokens.remaining", "Remaining")}: ${formatNumber(maxTokens - totalTokens)} ${t("playground:tokens.tokenUnit", "tokens")}`,
      isInteractive &&
        t(
          "playground:tokens.configureContextWindowHint",
          "Click to configure context window size."
        )
    ].filter(Boolean)
    return lines.join("\n")
  }, [
    modelLabel,
    draftTokens,
    conversationTokens,
    maxTokens,
    totalTokens,
    formatNumber,
    isInteractive,
    t
  ])

  const fallbackConversationLabel = React.useMemo(
    () =>
      `${formatNumber(conversationTokens)} + ~${formatNumber(
        draftTokens
      )} = ${formatNumber(totalTokens)} ${t("playground:tokens.tokenUnit", "tokens")}`,
    [conversationTokens, draftTokens, formatNumber, t, totalTokens]
  )

  if (!maxTokens || maxTokens <= 0) {
    if (isInteractive) {
      return (
        <button
          type="button"
          onClick={onClick}
          className="inline-flex items-center gap-1.5 rounded-sm bg-transparent p-0 text-[11px] text-text-muted transition hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          aria-label={t(
            "playground:tokens.configureContextWindow",
            "Configure context window size"
          )}
        >
          <span>{fallbackConversationLabel}</span>
        </button>
      )
    }
    return (
      <div className="inline-flex items-center gap-1.5 text-[11px] text-text-muted">
        <span>{fallbackConversationLabel}</span>
      </div>
    )
  }

  const progressColor = getProgressColor(percentage)
  const textColor = getProgressColorText(percentage)

  if (compact) {
    const compactContent = (
      <>
        <div className="relative h-1.5 w-16 sm:w-20 overflow-hidden rounded-full bg-border">
          <div
            className={`absolute inset-y-0 left-0 ${progressColor} transition-all duration-300`}
            style={{ width: `${percentage}%` }}
          />
        </div>
        <span className={`text-xs sm:text-[10px] font-medium ${textColor}`}>
          {fallbackConversationLabel}
        </span>
      </>
    )
    return (
      <Tooltip title={tooltipContent} placement="top">
        {isInteractive ? (
          <button
            type="button"
            onClick={onClick}
            className="inline-flex items-center gap-2 rounded-sm bg-transparent p-0 text-left transition hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
            aria-label={t(
              "playground:tokens.configureContextWindow",
              "Configure context window size"
            )}
          >
            {compactContent}
          </button>
        ) : (
          <div
            className="inline-flex items-center gap-2 cursor-help"
            aria-label={`${percentage}% ${t("playground:tokens.used", "used")}`}
          >
            {compactContent}
          </div>
        )}
      </Tooltip>
    )
  }

  const fullContent = (
    <>
      <div className="flex items-center justify-between gap-3">
        <div className="relative h-2 w-24 overflow-hidden rounded-full bg-border">
          <div
            className={`absolute inset-y-0 left-0 ${progressColor} transition-all duration-300`}
            style={{ width: `${percentage}%` }}
          />
        </div>
        <span className={`text-xs font-medium ${textColor}`}>
          {percentage}% {t("playground:tokens.used", "used")}
        </span>
      </div>
      <div className="text-[10px] text-text-muted">
        {t("playground:tokens.thisMessage", "This message")}: ~{formatNumber(draftTokens)}
      </div>
    </>
  )

  return (
    <Tooltip title={tooltipContent} placement="top">
      {isInteractive ? (
        <button
          type="button"
          onClick={onClick}
          className="inline-flex flex-col gap-1 rounded-lg border border-border bg-surface px-3 py-1.5 text-left transition hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          aria-label={t(
            "playground:tokens.configureContextWindow",
            "Configure context window size"
          )}
        >
          {fullContent}
        </button>
      ) : (
        <div
          className="inline-flex flex-col gap-1 cursor-help rounded-lg border border-border bg-surface px-3 py-1.5"
          aria-label={`${t("playground:tokens.contextUsage", "Context usage")}: ${percentage}%`}
        >
          {fullContent}
        </div>
      )}
    </Tooltip>
  )
}
