import React from "react"
import { useTranslation } from "react-i18next"
import type { BillingLimitErrorInfo } from "@/utils/billing-error"

type UpgradePromptProps = {
  errorInfo: BillingLimitErrorInfo
  isHighestPlan?: boolean
  onDismiss?: () => void
  onUpgrade?: () => void
  className?: string
}

const formatCategory = (category?: string): string => {
  if (!category) return ""
  return category.replace(/_/g, " ")
}

const UpgradePrompt: React.FC<UpgradePromptProps> = ({
  errorInfo,
  isHighestPlan = false,
  onDismiss,
  onUpgrade,
  className,
}) => {
  const { t } = useTranslation()

  const isFeatureGate = errorInfo.errorType === "feature_not_available"
  const hasUsageInfo =
    errorInfo.current != null && errorInfo.limit != null && errorInfo.limit > 0
  const percentUsed = hasUsageInfo
    ? Math.min(100, Math.round((errorInfo.current! / errorInfo.limit!) * 100))
    : null

  return (
    <div
      className={`rounded-xl border border-amber-300/40 bg-amber-50 px-4 py-3 dark:border-amber-600/30 dark:bg-amber-950/30 ${className ?? ""}`}
      role="alert"
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 text-amber-600 dark:text-amber-400" aria-hidden>
          {isFeatureGate ? "\u26A0" : "\u26A1"}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-amber-900 dark:text-amber-100">
            {isFeatureGate
              ? t("billing:upgradePrompt.featureTitle", "Feature not available")
              : t("billing:upgradePrompt.limitTitle", "Plan limit reached")}
          </p>
          <p className="mt-1 text-sm text-amber-800 dark:text-amber-200">
            {errorInfo.message}
          </p>

          {hasUsageInfo && (
            <div className="mt-2">
              <div className="flex justify-between text-xs text-amber-700 dark:text-amber-300">
                <span>
                  {formatCategory(errorInfo.category)}
                </span>
                <span>
                  {errorInfo.current!.toLocaleString()} / {errorInfo.limit!.toLocaleString()}
                </span>
              </div>
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-amber-200 dark:bg-amber-800">
                <div
                  className="h-full rounded-full bg-amber-500 transition-all dark:bg-amber-400"
                  style={{ width: `${percentUsed ?? 0}%` }}
                />
              </div>
            </div>
          )}

          <div className="mt-3 flex items-center gap-2">
            {isHighestPlan ? (
              <a
                href="mailto:support@tldw.dev"
                className="inline-flex items-center rounded-md bg-amber-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-amber-700 dark:bg-amber-500 dark:hover:bg-amber-600"
              >
                {t("billing:upgradePrompt.contactSupport", "Contact support")}
              </a>
            ) : onUpgrade ? (
              <button
                type="button"
                onClick={onUpgrade}
                className="inline-flex items-center rounded-md bg-amber-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-amber-700 dark:bg-amber-500 dark:hover:bg-amber-600"
              >
                {t("billing:upgradePrompt.upgradePlan", "Upgrade plan")}
              </button>
            ) : errorInfo.upgradeUrl ? (
              <a
                href={errorInfo.upgradeUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center rounded-md bg-amber-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-amber-700 dark:bg-amber-500 dark:hover:bg-amber-600"
              >
                {t("billing:upgradePrompt.upgradePlan", "Upgrade plan")}
              </a>
            ) : null}
            {onDismiss && (
              <button
                type="button"
                onClick={onDismiss}
                className="rounded-md px-3 py-1.5 text-xs text-amber-700 hover:bg-amber-100 dark:text-amber-300 dark:hover:bg-amber-900/50"
              >
                {t("common:dismiss", "Dismiss")}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default UpgradePrompt
