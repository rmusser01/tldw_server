/**
 * RateLimitsWidget component
 * Compact display of rate limits with progress bars
 */

import React, { useMemo } from "react"
import { Progress, Spin, Typography } from "antd"
import { useTranslation } from "react-i18next"
import type { EvaluationRateLimitStatus } from "@/services/evaluations"
import { Alert as DsAlert } from "@/components/ui/primitives"

const { Text } = Typography

interface RateLimitsWidgetProps {
  rateLimits?: EvaluationRateLimitStatus
  isLoading?: boolean
  isError?: boolean
  quotaSnapshot?: {
    limitDay?: number
    remainingDay?: number
    limitMinute?: number
    remainingMinute?: number
    reset?: string | null
  } | null
  compact?: boolean
  className?: string
}

export const RateLimitsWidget: React.FC<RateLimitsWidgetProps> = ({
  rateLimits,
  isLoading,
  isError,
  quotaSnapshot,
  compact = false,
  className = ""
}) => {
  const { t } = useTranslation(["evaluations", "common"])

  const quotaText = useMemo(() => {
    if (!quotaSnapshot) return null
    const parts: string[] = []
    if (
      quotaSnapshot.remainingDay != null &&
      quotaSnapshot.limitDay != null
    ) {
      parts.push(
        t("evaluations:rateLimitsDayRemaining", {
          defaultValue: "Day {{remaining}}/{{limit}} remaining",
          remaining: quotaSnapshot.remainingDay,
          limit: quotaSnapshot.limitDay
        })
      )
    }
    if (
      quotaSnapshot.remainingMinute != null &&
      quotaSnapshot.limitMinute != null
    ) {
      parts.push(
        t("evaluations:rateLimitsMinuteRemaining", {
          defaultValue: "Minute {{remaining}}/{{limit}} remaining",
          remaining: quotaSnapshot.remainingMinute,
          limit: quotaSnapshot.limitMinute
        })
      )
    }
    if (quotaSnapshot.reset) {
      parts.push(
        t("evaluations:rateLimitsResetAt", {
          defaultValue: "Resets at {{timestamp}}",
          timestamp: quotaSnapshot.reset
        })
      )
    }
    return parts.join(" • ")
  }, [quotaSnapshot, t])

  if (isLoading) {
    return (
      <div className={`flex justify-center py-4 ${className}`}>
        <Spin />
      </div>
    )
  }

  if (isError) {
    return (
      <DsAlert
        variant="warning"
        title={t("evaluations:rateLimitsErrorTitle", {
          defaultValue: "Unable to fetch rate limits"
        })}
        className={className}
      />
    )
  }

  // Show quota snapshot if available (from response headers)
  if (quotaSnapshot && quotaText) {
    return (
      <DsAlert
        variant="info"
        title={t("evaluations:rateLimitsTitle", {
          defaultValue: "Evaluation limits"
        })}
        className={className}
      >
        {quotaText}
      </DsAlert>
    )
  }

  if (!rateLimits) {
    return (
      <Text type="secondary" className={`text-xs ${className}`}>
        {t("evaluations:rateLimitsNoData", {
          defaultValue:
            "Rate limit information is not available for this server."
        })}
      </Text>
    )
  }

  const dailyUsed = rateLimits.usage.evaluations_today
  const dailyLimit = rateLimits.limits.evaluations_per_day
  const dailyPercent =
    dailyLimit > 0 ? Math.round((dailyUsed / dailyLimit) * 100) : 0

  const tokensUsed = rateLimits.usage.tokens_today
  const tokensLimit = rateLimits.limits.tokens_per_day
  const tokensPercent =
    tokensLimit > 0 ? Math.round((tokensUsed / tokensLimit) * 100) : 0

  if (compact) {
    return (
      <div className={`space-y-1 text-xs ${className}`}>
        <div className="flex items-center gap-2">
          <span className="w-16 text-text-subtle">
            {t("evaluations:rateLimitsDailyShort", {
              defaultValue: "Daily"
            })}
            :
          </span>
          <Progress
            percent={dailyPercent}
            size="small"
            className="flex-1"
            status={dailyPercent >= 90 ? "exception" : undefined}
          />
          <span className="w-16 text-right">
            {dailyUsed}/{dailyLimit}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-16 text-text-subtle">
            {t("evaluations:rateLimitsTokensShort", {
              defaultValue: "Tokens"
            })}
            :
          </span>
          <Progress
            percent={tokensPercent}
            size="small"
            className="flex-1"
            status={tokensPercent >= 90 ? "exception" : undefined}
          />
          <span className="w-16 text-right">
            {tokensUsed.toLocaleString()}/{tokensLimit.toLocaleString()}
          </span>
        </div>
      </div>
    )
  }

  return (
    <div className={`space-y-2 text-xs ${className}`}>
      <div>
        <Text type="secondary">
          {t("evaluations:tierLabel", { defaultValue: "Tier" })}
          {": "}
        </Text>
        <Text>{rateLimits.tier}</Text>
      </div>
      <div>
        <Text type="secondary">
          {t("evaluations:dailyLimitLabel", {
            defaultValue: "Daily evaluations"
          })}
          {": "}
        </Text>
        <Text>
          {dailyUsed}/{dailyLimit}
        </Text>
        <Progress
          percent={dailyPercent}
          size="small"
          status={dailyPercent >= 90 ? "exception" : undefined}
        />
      </div>
      <div>
        <Text type="secondary">
          {t("evaluations:tokensTodayLabel", {
            defaultValue: "Tokens today"
          })}
          {": "}
        </Text>
        <Text>
          {tokensUsed.toLocaleString()}/{tokensLimit.toLocaleString()}
        </Text>
        <Progress
          percent={tokensPercent}
          size="small"
          status={tokensPercent >= 90 ? "exception" : undefined}
        />
      </div>
      <div>
        <Text type="secondary">
          {t("evaluations:perMinuteLabel", {
            defaultValue: "Per-minute limit"
          })}
          {": "}
        </Text>
        <Text>{rateLimits.limits.evaluations_per_minute}</Text>
      </div>
      <div>
        <Text type="secondary">
          {t("evaluations:resetAtLabel", { defaultValue: "Resets at" })}
          {": "}
        </Text>
        <Text>{rateLimits.reset_at}</Text>
      </div>
    </div>
  )
}

export default RateLimitsWidget
