import { useQuery } from "@tanstack/react-query"
import { Drawer, Skeleton, Descriptions, Table, Progress, Statistic } from "antd"
import {
  BarChart3,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  Target,
  TrendingUp
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import { getEvaluation, type PromptStudioEvaluation } from "@/services/prompt-studio"
import { Badge } from "@/components/ui/primitives"

type EvaluationDetailPanelProps = {
  open: boolean
  evaluationId: number | null
  onClose: () => void
}

export const EvaluationDetailPanel: React.FC<EvaluationDetailPanelProps> = ({
  open,
  evaluationId,
  onClose
}) => {
  const { t } = useTranslation(["settings", "common"])

  // Fetch evaluation with auto-refresh for running evaluations
  const { data: evaluationResponse, isLoading } = useQuery({
    queryKey: ["prompt-studio", "evaluation", evaluationId],
    queryFn: () => getEvaluation(evaluationId!),
    enabled: open && evaluationId !== null,
    refetchInterval: (query) => {
      const data = query.state.data as any
      const evaluation = data?.data
      const isRunning = ["running", "pending"].includes(
        evaluation?.status?.toLowerCase()
      )
      return isRunning ? 3000 : false
    }
  })

  const evaluation: PromptStudioEvaluation | undefined = (evaluationResponse as any)?.data

  const metrics = evaluation?.aggregate_metrics ?? evaluation?.metrics
  const hasMetric = (key: string): boolean =>
    typeof metrics?.[key] === "number" && Number.isFinite(metrics[key])
  const hasMetrics = ["accuracy", "pass_rate", "f1", "precision", "recall"].some(hasMetric)
  const configs = evaluation?.model_configs === undefined
    ? evaluation?.config && Object.keys(evaluation.config).length ? [evaluation.config] : []
    : evaluation.model_configs ?? []

  const getStatusTag = (status?: string) => {
    const statusLower = status?.toLowerCase()
    switch (statusLower) {
      case "completed":
        return (
          <Badge variant="success" size="sm">
            <CheckCircle2 className="size-3" aria-hidden="true" />
            Completed
          </Badge>
        )
      case "running":
        return (
          <Badge variant="info" size="sm">
            <Loader2 className="size-3 animate-spin" aria-hidden="true" />
            Running
          </Badge>
        )
      case "pending":
        return (
          <Badge variant="secondary" size="sm">
            <Clock className="size-3" aria-hidden="true" />
            Pending
          </Badge>
        )
      case "failed":
        return (
          <Badge variant="danger" size="sm">
            <XCircle className="size-3" aria-hidden="true" />
            Failed
          </Badge>
        )
      default:
        return (
          <Badge variant="secondary" size="sm">
            {status}
          </Badge>
        )
    }
  }

  const formatDate = (dateStr?: string | null) => {
    if (!dateStr) return "-"
    return new Date(dateStr).toLocaleString()
  }

  const metricColor = (value: number): string => {
    if (value >= 0.8) return "rgb(var(--color-success))"
    if (value >= 0.5) return "rgb(var(--color-warn))"
    return "rgb(var(--color-danger))"
  }

  const formatMetricValue = (value: number | undefined, isPercent = false) => {
    if (value === undefined) return "-"
    if (isPercent) return `${(value * 100).toFixed(1)}%`
    return value.toFixed(3)
  }

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={
        <span className="flex items-center gap-2">
          <BarChart3 className="size-5" />
          {t("managePrompts.studio.evaluations.detailTitle", {
            defaultValue: "Evaluation Details"
          })}
        </span>
      }
      size={600}
      destroyOnHidden
    >
      {isLoading && <Skeleton paragraph={{ rows: 10 }} />}

      {!isLoading && evaluation && (
        <div className="space-y-6">
          {/* Header info */}
          <div className="p-4 bg-surface2 rounded-md">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-lg">
                {evaluation.name || `Evaluation #${evaluation.id}`}
              </h3>
              {getStatusTag(evaluation.status)}
            </div>
            {evaluation.description && (
              <p className="text-sm text-text-muted">{evaluation.description}</p>
            )}
          </div>

          {/* Error message if failed */}
          {evaluation.error_message && (
            <div className="p-4 bg-danger/10 border border-danger/30 rounded-md">
              <p className="text-sm text-danger font-medium">
                {t("managePrompts.studio.evaluations.error", {
                  defaultValue: "Error"
                })}
              </p>
              <p className="text-sm mt-1">{evaluation.error_message}</p>
            </div>
          )}

          {/* Metrics summary */}
          {hasMetrics && metrics && (
              <div>
                <h4 className="font-medium mb-3">
                  {t("managePrompts.studio.evaluations.metrics", {
                    defaultValue: "Metrics"
                  })}
                </h4>
                <div className="grid grid-cols-3 gap-4">
                  {hasMetric("accuracy") && (
                    <div className="p-3 bg-surface2 rounded-md text-center">
                      <Statistic
                        title={
                          <span className="flex items-center justify-center gap-1">
                            <Target className="size-4" />
                            Accuracy
                          </span>
                        }
                        value={metrics.accuracy * 100}
                        precision={1}
                        suffix="%"
                        valueStyle={{
                          color: metricColor(metrics.accuracy)
                        }}
                      />
                    </div>
                  )}
                  {hasMetric("pass_rate") && (
                    <div className="p-3 bg-surface2 rounded-md text-center">
                      <Statistic
                        title={
                          <span className="flex items-center justify-center gap-1">
                            <CheckCircle2 className="size-4" />
                            Pass Rate
                          </span>
                        }
                        value={metrics.pass_rate * 100}
                        precision={1}
                        suffix="%"
                        valueStyle={{
                          color: metricColor(metrics.pass_rate)
                        }}
                      />
                    </div>
                  )}
                  {hasMetric("f1") && (
                    <div className="p-3 bg-surface2 rounded-md text-center">
                      <Statistic
                        title={
                          <span className="flex items-center justify-center gap-1">
                            <TrendingUp className="size-4" />
                            F1 Score
                          </span>
                        }
                        value={metrics.f1}
                        precision={3}
                      />
                    </div>
                  )}
                </div>

                {/* Additional metrics */}
                {(hasMetric("precision") ||
                  hasMetric("recall")) && (
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    {hasMetric("precision") && (
                      <div className="p-3 bg-surface2 rounded-md">
                        <div className="text-sm text-text-muted mb-1">
                          Precision
                        </div>
                        <Progress
                          percent={Math.round(
                            metrics.precision * 100
                          )}
                          size="small"
                        />
                      </div>
                    )}
                    {hasMetric("recall") && (
                      <div className="p-3 bg-surface2 rounded-md">
                        <div className="text-sm text-text-muted mb-1">
                          Recall
                        </div>
                        <Progress
                          percent={Math.round(
                            metrics.recall * 100
                          )}
                          size="small"
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

          {/* Configuration details */}
          <Descriptions
            title={t("managePrompts.studio.evaluations.configuration", {
              defaultValue: "Configuration"
            })}
            column={2}
            size="small"
            bordered
          >
            <Descriptions.Item
              label={t("managePrompts.studio.evaluations.promptId", {
                defaultValue: "Prompt ID"
              })}
            >
              {evaluation.prompt_id}
            </Descriptions.Item>
            <Descriptions.Item
              label={t("managePrompts.studio.evaluations.testCaseCount", {
                defaultValue: "Test Cases"
              })}
            >
              {evaluation.test_case_ids?.length ?? "Unavailable"}
            </Descriptions.Item>
            <Descriptions.Item
              label={t("managePrompts.studio.evaluations.created", {
                defaultValue: "Created"
              })}
            >
              {formatDate(evaluation.created_at)}
            </Descriptions.Item>
            <Descriptions.Item
              label={t("managePrompts.studio.evaluations.completed", {
                defaultValue: "Completed"
              })}
            >
              {formatDate(evaluation.completed_at)}
            </Descriptions.Item>
          </Descriptions>

          {!hasMetrics && <p>Metrics: <span>Unavailable</span></p>}

          {/* Each recorded model has its own public names and parameters. */}
          {configs.length === 0 ? <p>Model Configuration: <span>Unavailable</span></p> : configs.map((config, index) => {
            const names = [
              { key: "provider", label: "Provider", children: config.provider ?? config.api_name ?? "Unavailable" },
              { key: "model", label: "Model", children: config.model_name ?? config.model ?? "Unavailable" }
            ]
            const parameters = ([
              ["temperature", "Temperature"], ["max_tokens", "Max Tokens"], ["top_p", "Top P"],
              ["frequency_penalty", "Frequency Penalty"], ["presence_penalty", "Presence Penalty"],
              ["timeout_seconds", "Timeout (seconds)"], ["retry_count", "Retry Count"], ["parallel_requests", "Parallel Requests"]
            ] as const).filter(([key]) => typeof config[key] === "number" && Number.isFinite(config[key]))
              .map(([key, label]) => ({ key, label, children: config[key] }))
            return <Descriptions key={index} title={configs.length > 1 ? `Model Configuration ${index + 1}` : "Model Configuration"}
              column={2} size="small" bordered items={[...names, ...parameters]} />
          })}
        </div>
      )}
    </Drawer>
  )
}
