import React from "react"
import { X } from "lucide-react"
import type {
  ModelRecommendation,
  ModelRecommendationAction
} from "./model-recommendations"

type Props = {
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
  recommendations: ModelRecommendation[]
  showOpenInsights: boolean
  onOpenInsights: () => void
  onRunAction: (action: ModelRecommendationAction) => void
  onDismiss: (id: string) => void
  getActionLabel: (action: ModelRecommendationAction) => string
}

export const ModelRecommendationsPanel: React.FC<Props> = ({
  t,
  recommendations,
  showOpenInsights,
  onOpenInsights,
  onRunAction,
  onDismiss,
  getActionLabel
}) => {
  if (recommendations.length === 0) return null

  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="model-recommendations-panel"
      className="mt-1 space-y-2 rounded-md border border-primary/30 bg-primary/10 px-2 py-2"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-semibold uppercase tracking-wide text-primaryStrong">
        <span>
          {t("playground:composer.recommendationsTitle", "Model recommendations")}
        </span>
        {showOpenInsights && (
          <button
            type="button"
            onClick={onOpenInsights}
            className="rounded border border-primary/40 bg-surface px-2 py-0.5 text-[10px] font-medium text-primaryStrong hover:bg-primary/10"
          >
            {t("playground:composer.recommendationsOpenInsights", "Open insights")}
          </button>
        )}
      </div>
      {recommendations.map((recommendation) => (
        <div
          key={recommendation.id}
          data-testid={`model-recommendation-${recommendation.id}`}
          className="flex items-start justify-between gap-2 rounded border border-primary/20 bg-surface/80 px-2 py-1.5 text-xs text-primaryStrong"
        >
          <div className="min-w-0">
            <p className="font-medium text-text">{recommendation.title}</p>
            <p className="mt-0.5 text-[11px] leading-4 text-text-muted">
              {recommendation.reason}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              onClick={() => onRunAction(recommendation.action)}
              className="rounded border border-primary/30 bg-surface px-2 py-0.5 text-[11px] font-medium text-primaryStrong hover:bg-primary/10"
            >
              {getActionLabel(recommendation.action)}
            </button>
            <button
              type="button"
              onClick={() => onDismiss(recommendation.id)}
              aria-label={t(
                "playground:composer.recommendationDismiss",
                "Dismiss recommendation"
              ) as string}
              title={t(
                "playground:composer.recommendationDismiss",
                "Dismiss recommendation"
              ) as string}
              className="rounded border border-border px-1.5 py-0.5 text-text-subtle hover:bg-surface2 hover:text-text"
            >
              <X className="h-3 w-3" aria-hidden="true" />
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
