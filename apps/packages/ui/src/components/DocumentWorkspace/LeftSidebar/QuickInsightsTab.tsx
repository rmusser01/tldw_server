import React from "react"
import { useTranslation } from "react-i18next"
import { Button, Empty, Skeleton, Collapse, Segmented } from "antd"
import {
  Lightbulb,
  AlertTriangle,
  RefreshCw,
  Sparkles,
} from "lucide-react"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import {
  useDocumentInsights,
  useGenerateInsightsMutation,
  INSIGHT_CATEGORY_INFO,
  type InsightItem,
} from "@/hooks/document-workspace"
import { useConnectionStore } from "@/store/connection"
import type { InsightDetailLevel } from "../types"
import { CATEGORY_ICONS } from "../config"

/**
 * Detail level configuration - maps to max_content_length parameter
 */
const DETAIL_LEVEL_CONFIG: Record<InsightDetailLevel, { label: string; maxLength: number; description: string }> = {
  brief: {
    label: "Brief",
    maxLength: 500,
    description: "Quick summary highlights"
  },
  standard: {
    label: "Standard",
    maxLength: 2000,
    description: "Balanced detail level"
  },
  detailed: {
    label: "Detailed",
    maxLength: 5000,
    description: "Comprehensive analysis"
  }
}

/**
 * Single insight item display.
 */
const InsightItemDisplay: React.FC<{ insight: InsightItem }> = ({ insight }) => {
  const icon = CATEGORY_ICONS[insight.category]
  const info = INSIGHT_CATEGORY_INFO[insight.category]

  return (
    <Collapse
      size="small"
      className="mb-2 [&_.ant-collapse-header]:!px-3 [&_.ant-collapse-header]:!py-2"
      items={[
        {
          key: insight.category,
          label: (
            <div className="flex items-center gap-2">
              <span className="text-primary">{icon}</span>
              <span className="font-medium text-sm">{insight.title}</span>
            </div>
          ),
          children: (
            <div className="text-sm text-text-secondary leading-relaxed">
              {insight.content}
            </div>
          ),
        },
      ]}
      defaultActiveKey={insight.category === "summary" ? ["summary"] : []}
    />
  )
}

/**
 * Empty state when no document is selected.
 */
const NoDocumentState: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  return (
    <div className="flex h-full items-center justify-center p-4">
      <Empty
        image={<Lightbulb className="h-12 w-12 text-muted mx-auto mb-2" />}
        description={t(
          "option:documentWorkspace.noDocumentForInsights",
          "AI-generated summaries, key findings, and research analysis. Open a document to get started."
        )}
      />
    </div>
  )
}

/**
 * State when insights haven't been generated yet.
 */
const GeneratePromptState: React.FC<{
  onGenerate: () => void
  isGenerating: boolean
  isServerAvailable: boolean
  detailLevel: InsightDetailLevel
  onDetailLevelChange: (value: string | number) => void
}> = ({ onGenerate, isGenerating, isServerAvailable, detailLevel, onDetailLevelChange }) => {
  const { t } = useTranslation(["option", "common"])

  return (
    <div className="flex flex-col items-center justify-center p-6 text-center">
      <Sparkles className="h-12 w-12 text-primary mb-4" />
      <h3 className="text-base font-medium mb-2">
        {t("option:documentWorkspace.insightsTitle", "AI Document Insights")}
      </h3>
      <p className="text-sm text-text-secondary mb-4 max-w-xs">
        {t(
          "option:documentWorkspace.insightsDescription",
          "Analyze this document to extract key research insights, findings, and summaries."
        )}
      </p>

      {/* Detail level selector */}
      <div className="mb-4 w-full max-w-xs">
        <label className="mb-1.5 block text-xs font-medium text-text-secondary">
          {t("option:documentWorkspace.detailLevel", "Detail Level")}
        </label>
        <Segmented
          value={detailLevel}
          onChange={onDetailLevelChange}
          options={Object.entries(DETAIL_LEVEL_CONFIG).map(([key, config]) => ({
            value: key,
            label: t(`option:documentWorkspace.detail${config.label}`, config.label)
          }))}
          block
          size="small"
        />
        <p className="mt-1 text-[11px] text-text-muted">
          {DETAIL_LEVEL_CONFIG[detailLevel].description}
        </p>
      </div>

      <Button
        type="primary"
        onClick={onGenerate}
        loading={isGenerating}
        disabled={!isServerAvailable}
        icon={<Sparkles className="h-4 w-4" />}
      >
        {isGenerating
          ? t("option:documentWorkspace.generating", "Generating...")
          : t("option:documentWorkspace.generateInsights", "Generate Insights")}
      </Button>
      {!isServerAvailable && (
        <p className="text-xs text-warning mt-2">
          {t(
            "option:documentWorkspace.serverUnavailable",
            "Connect to your server in Settings to use this feature"
          )}
        </p>
      )}
    </div>
  )
}

/**
 * Error state when generation fails.
 */
const ErrorState: React.FC<{
  error: Error
  onRetry: () => void
  isRetrying: boolean
}> = ({ error, onRetry, isRetrying }) => {
  const { t } = useTranslation(["option", "common"])

  return (
    <div className="flex flex-col items-center justify-center p-6 text-center">
      <AlertTriangle className="h-10 w-10 text-error mb-3" />
      <h3 className="text-sm font-medium text-error mb-2">
        {t("option:documentWorkspace.insightsError", "Failed to generate insights")}
      </h3>
      <p className="text-xs text-text-secondary mb-4 max-w-xs">
        {error.message || t("common:unknownError", "An unknown error occurred")}
      </p>
      <Button
        onClick={onRetry}
        loading={isRetrying}
        icon={<RefreshCw className="h-4 w-4" />}
      >
        {t("common:retry", "Retry")}
      </Button>
    </div>
  )
}

/**
 * Loading state during generation.
 */
const LoadingState: React.FC = () => {
  return (
    <div className="space-y-3 p-4">
      <Skeleton active paragraph={{ rows: 2 }} />
      <Skeleton active paragraph={{ rows: 2 }} />
      <Skeleton active paragraph={{ rows: 2 }} />
    </div>
  )
}

/**
 * QuickInsightsTab - Display AI-generated document insights.
 *
 * Features:
 * - On-demand LLM analysis of document content
 * - Collapsible sections for each insight category
 * - Loading and error states
 * - Server connection awareness
 */
export const QuickInsightsTab: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const activeDocumentId = useDocumentWorkspaceStore((s) => s.activeDocumentId)
  const insightDetailLevel = useDocumentWorkspaceStore((s) => s.insightDetailLevel)
  const setInsightDetailLevel = useDocumentWorkspaceStore((s) => s.setInsightDetailLevel)

  // Connection state
  const isConnected = useConnectionStore((s) => s.state.isConnected)
  const mode = useConnectionStore((s) => s.state.mode)
  const isServerAvailable = isConnected && mode !== "demo"

  // Query and mutation
  const { data: insights, isLoading, error } = useDocumentInsights(activeDocumentId)
  const generateMutation = useGenerateInsightsMutation()

  // Handle generate click
  const handleGenerate = (force: boolean = false) => {
    if (activeDocumentId) {
      const maxLength = DETAIL_LEVEL_CONFIG[insightDetailLevel].maxLength
      generateMutation.mutate({
        mediaId: activeDocumentId,
        options: {
          max_content_length: maxLength,
          ...(force ? { force: true } : {})
        },
      })
    }
  }

  // Handle detail level change
  const handleDetailLevelChange = (value: string | number) => {
    setInsightDetailLevel(value as InsightDetailLevel)
  }

  // No document selected
  if (!activeDocumentId) {
    return <NoDocumentState />
  }

  // Loading state during generation
  if (generateMutation.isPending || isLoading) {
    return <LoadingState />
  }

  // Error state
  if (generateMutation.error) {
    return (
      <ErrorState
        error={generateMutation.error as Error}
        onRetry={() => handleGenerate(true)}
        isRetrying={generateMutation.isPending}
      />
    )
  }

  // No insights generated yet
  if (!insights || insights.insights.length === 0) {
    return (
      <GeneratePromptState
        onGenerate={() => handleGenerate(false)}
        isGenerating={generateMutation.isPending}
        isServerAvailable={isServerAvailable}
        detailLevel={insightDetailLevel}
        onDetailLevelChange={handleDetailLevelChange}
      />
    )
  }

  // Display insights
  // Sort insights: summary first, then rest
  const sortedInsights = [...insights.insights].sort((a, b) => {
    if (a.category === "summary") return -1
    if (b.category === "summary") return 1
    return 0
  })

  return (
    <div className="h-full overflow-y-auto">
      <div className="space-y-1 p-3">
        {/* Header with model info and detail level */}
        <div className="mb-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-text-muted">
              {t("option:documentWorkspace.generatedWith", "Generated with")}{" "}
              <span className="text-text-secondary">{insights.model_used}</span>
            </span>
            <Button
              size="small"
              type="text"
              icon={<RefreshCw className="h-3.5 w-3.5" />}
              onClick={() => handleGenerate(true)}
              loading={generateMutation.isPending}
              title={t("option:documentWorkspace.regenerate", "Regenerate")}
            />
          </div>

          {/* Detail level selector for regeneration */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted shrink-0">
              {t("option:documentWorkspace.detailLevel", "Detail")}:
            </span>
            <Segmented
              value={insightDetailLevel}
              onChange={handleDetailLevelChange}
              options={Object.entries(DETAIL_LEVEL_CONFIG).map(([key, config]) => ({
                value: key,
                label: config.label
              }))}
              size="small"
              className="flex-1"
            />
          </div>
        </div>

        {/* Insights list */}
        {sortedInsights.map((insight) => (
          <InsightItemDisplay key={insight.category} insight={insight} />
        ))}
      </div>
    </div>
  )
}

export default QuickInsightsTab
