import React from "react"
import { Progress } from "antd"
import {
  getBudgetUtilizationPercent,
  getBudgetUtilizationBand,
  getBudgetUtilizationColor
} from "./worldBookStatsUtils"

type WorldBookBudgetBarProps = {
  estimatedTokens: number
  tokenBudget: number
  projectedTokens?: number
  className?: string
}

export const WorldBookBudgetBar: React.FC<WorldBookBudgetBarProps> = ({
  estimatedTokens,
  tokenBudget,
  projectedTokens,
  className
}) => {
  const percent = getBudgetUtilizationPercent(estimatedTokens, tokenBudget)

  // Render nothing when tokenBudget is 0 or not a finite number
  if (percent == null) return null

  const band = getBudgetUtilizationBand(percent)
  const color = getBudgetUtilizationColor(band)
  const exceeds = percent > 100

  return (
    <div
      role="meter"
      aria-valuenow={estimatedTokens}
      aria-valuemax={tokenBudget}
      aria-label="Token budget usage"
      className={className}
    >
      <Progress
        percent={Math.min(percent, 100)}
        strokeColor={color}
        size="small"
        showInfo={false}
      />
      <div className="text-xs text-text-muted">
        <span>
          {estimatedTokens}/{tokenBudget} tokens
        </span>
        {projectedTokens != null && (
          <span className="ml-2">
            After save: {projectedTokens}/{tokenBudget}
          </span>
        )}
      </div>
      {exceeds && (
        <div className="text-xs text-red-500">
          Estimated usage exceeds the configured budget.
        </div>
      )}
    </div>
  )
}
