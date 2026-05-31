import React from "react"
import { formatCost } from "@/utils/model-pricing"
import type {
  SessionInsights,
  SessionInsightsModelRow
} from "./session-insights"

type Props = {
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
  insights: SessionInsights
}

const formatCount = (value: number): string => value.toLocaleString()

export const SessionInsightsPanel: React.FC<Props> = ({ t, insights }) => {
  const [providerFilter, setProviderFilter] = React.useState<string>("all")
  const [expandedModelKey, setExpandedModelKey] = React.useState<string | null>(
    null
  )

  const filteredModels = React.useMemo(() => {
    if (providerFilter === "all") return insights.models
    return insights.models.filter((row) => row.providerKey === providerFilter)
  }, [insights.models, providerFilter])

  const activeModel = React.useMemo<SessionInsightsModelRow | null>(() => {
    if (!expandedModelKey) return null
    return (
      filteredModels.find((row) => row.key === expandedModelKey) ||
      insights.models.find((row) => row.key === expandedModelKey) ||
      null
    )
  }, [expandedModelKey, filteredModels, insights.models])

  return (
    <div className="space-y-3" data-testid="session-insights-panel">
      <div className="grid gap-2 sm:grid-cols-3">
        <div className="rounded border border-border bg-surface2 px-2 py-1.5">
          <p className="text-[11px] text-text-muted">
            {t("playground:insights.generatedMessages", "Generated messages")}
          </p>
          <p className="text-sm font-semibold text-text">
            {formatCount(insights.totals.generatedMessages)}
          </p>
        </div>
        <div className="rounded border border-border bg-surface2 px-2 py-1.5">
          <p className="text-[11px] text-text-muted">
            {t("playground:insights.totalTokens", "Total tokens")}
          </p>
          <p className="text-sm font-semibold text-text">
            {formatCount(insights.totals.totalTokens)}
          </p>
        </div>
        <div className="rounded border border-border bg-surface2 px-2 py-1.5">
          <p className="text-[11px] text-text-muted">
            {t("playground:insights.totalCost", "Estimated cost")}
          </p>
          <p className="text-sm font-semibold text-text">
            {insights.totals.estimatedCostUsd != null
              ? formatCost(insights.totals.estimatedCostUsd)
              : t("common:unknown", "Unknown")}
          </p>
        </div>
      </div>

      <div className="space-y-1">
        <p className="text-xs font-medium text-text-muted">
          {t("playground:insights.providerFilter", "Filter by provider")}
        </p>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => setProviderFilter("all")}
            className={`rounded-full border px-2 py-0.5 text-[11px] ${
              providerFilter === "all"
                ? "border-primary/40 bg-primary/10 text-primaryStrong"
                : "border-border bg-surface text-text-muted"
            }`}
          >
            {t("common:all", "All")}
          </button>
          {insights.providers.map((provider) => (
            <button
              key={provider.providerKey}
              type="button"
              onClick={() => setProviderFilter(provider.providerKey)}
              className={`rounded-full border px-2 py-0.5 text-[11px] ${
                providerFilter === provider.providerKey
                  ? "border-primary/40 bg-primary/10 text-primaryStrong"
                  : "border-border bg-surface text-text-muted"
              }`}
            >
              {provider.providerKey}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-1 rounded border border-border bg-surface2 p-2">
        <p className="text-xs font-medium text-text-muted">
          {t("playground:insights.modelsBreakdown", "Model breakdown")}
        </p>
        {filteredModels.length === 0 ? (
          <p className="text-xs text-text-muted">
            {t("playground:insights.noModels", "No model usage available.")}
          </p>
        ) : (
          <div className="space-y-1">
            {filteredModels.map((row) => (
              <div
                key={row.key}
                className="rounded border border-border bg-surface px-2 py-1"
              >
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="font-medium text-text">
                    {row.modelId}
                  </span>
                  <span className="text-text-muted">
                    {formatCount(row.totalTokens)}{" "}
                    {t("playground:tokens.tokenUnit", "tokens")}
                  </span>
                </div>
                <div className="mt-1 flex items-center justify-between gap-2">
                  <span className="text-[11px] text-text-muted">
                    {row.providerKey} • {row.messageCount}{" "}
                    {t("playground:insights.messagesShort", "msgs")}
                  </span>
                  <button
                    type="button"
                    onClick={() =>
                      setExpandedModelKey((prev) =>
                        prev === row.key ? null : row.key
                      )
                    }
                    className="rounded border border-border bg-surface2 px-2 py-0.5 text-[11px] text-text-subtle hover:bg-surface-hover hover:text-text"
                  >
                    {expandedModelKey === row.key
                      ? t("common:hideDetails", "Hide details")
                      : t("common:viewDetails", "View details")}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {activeModel && (
        <div
          className="rounded border border-primary/30 bg-primary/10 p-2 text-xs text-primaryStrong"
          data-testid="session-insights-drilldown"
        >
          <p className="font-semibold">
            {t("playground:insights.drilldownTitle", "Model detail")}:{" "}
            {activeModel.modelId}
          </p>
          <p>
            {t("playground:insights.provider", "Provider")}:{" "}
            {activeModel.providerKey}
          </p>
          <p>
            {t("playground:insights.messages", "Messages")}:{" "}
            {formatCount(activeModel.messageCount)}
          </p>
          <p>
            {t("playground:insights.tokenSplit", "Prompt/Completion")}:{" "}
            {formatCount(activeModel.inputTokens)} /{" "}
            {formatCount(activeModel.outputTokens)}
          </p>
          <p>
            {t("playground:insights.estimatedCost", "Estimated cost")}:{" "}
            {activeModel.estimatedCostUsd != null
              ? formatCost(activeModel.estimatedCostUsd)
              : t("common:unknown", "Unknown")}
          </p>
        </div>
      )}

      <div className="space-y-1 rounded border border-border bg-surface2 p-2">
        <p className="text-xs font-medium text-text-muted">
          {t("playground:insights.topics", "Topic distribution")}
        </p>
        {insights.topics.length === 0 ? (
          <p className="text-xs text-text-muted">
            {t(
              "playground:insights.noTopics",
              "No topic data yet. Continue the conversation to populate this."
            )}
          </p>
        ) : (
          <div className="space-y-1">
            {insights.topics.map((topic) => (
              <div
                key={topic.label}
                className="flex items-center justify-between gap-2 text-xs"
              >
                <span className="text-text">{topic.label}</span>
                <span className="text-text-muted">{topic.count}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
