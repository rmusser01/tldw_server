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
          {evaluation.aggregate_metrics &&
            Object.keys(evaluation.aggregate_metrics).length > 0 && (
              <div>
                <h4 className="font-medium mb-3">
                  {t("managePrompts.studio.evaluations.metrics", {
                    defaultValue: "Metrics"
                  })}
                </h4>
                <div className="grid grid-cols-3 gap-4">
                  {evaluation.aggregate_metrics.accuracy !== undefined && (
                    <div className="p-3 bg-surface2 rounded-md text-center">
                      <Statistic
                        title={
                          <span className="flex items-center justify-center gap-1">
                            <Target className="size-4" />
                            Accuracy
                          </span>
                        }
                        value={evaluation.aggregate_metrics.accuracy * 100}
                        precision={1}
                        suffix="%"
                        valueStyle={{
                          color: metricColor(evaluation.aggregate_metrics.accuracy)
                        }}
                      />
                    </div>
                  )}
                  {evaluation.aggregate_metrics.pass_rate !== undefined && (
                    <div className="p-3 bg-surface2 rounded-md text-center">
                      <Statistic
                        title={
                          <span className="flex items-center justify-center gap-1">
                            <CheckCircle2 className="size-4" />
                            Pass Rate
                          </span>
                        }
                        value={evaluation.aggregate_metrics.pass_rate * 100}
                        precision={1}
                        suffix="%"
                        valueStyle={{
                          color: metricColor(evaluation.aggregate_metrics.pass_rate)
                        }}
                      />
                    </div>
                  )}
                  {evaluation.aggregate_metrics.f1 !== undefined && (
                    <div className="p-3 bg-surface2 rounded-md text-center">
                      <Statistic
                        title={
                          <span className="flex items-center justify-center gap-1">
                            <TrendingUp className="size-4" />
                            F1 Score
                          </span>
                        }
                        value={evaluation.aggregate_metrics.f1}
                        precision={3}
                      />
                    </div>
                  )}
                </div>

                {/* Additional metrics */}
                {(evaluation.aggregate_metrics.precision !== undefined ||
                  evaluation.aggregate_metrics.recall !== undefined) && (
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    {evaluation.aggregate_metrics.precision !== undefined && (
                      <div className="p-3 bg-surface2 rounded-md">
                        <div className="text-sm text-text-muted mb-1">
                          Precision
                        </div>
                        <Progress
                          percent={Math.round(
                            evaluation.aggregate_metrics.precision * 100
                          )}
                          size="small"
                        />
                      </div>
                    )}
                    {evaluation.aggregate_metrics.recall !== undefined && (
                      <div className="p-3 bg-surface2 rounded-md">
                        <div className="text-sm text-text-muted mb-1">
                          Recall
                        </div>
                        <Progress
                          percent={Math.round(
                            evaluation.aggregate_metrics.recall * 100
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
              {evaluation.test_case_ids?.length ?? 0}
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

          {/* Model config */}
          {evaluation.config && Object.keys(evaluation.config).length > 0 && (
            <Descriptions
              title={t("managePrompts.studio.evaluations.modelConfig", {
                defaultValue: "Model Configuration"
              })}
              column={2}
              size="small"
              bordered
            >
              {evaluation.config.model_name && (
                <Descriptions.Item label="Model">
                  {evaluation.config.model_name}
                </Descriptions.Item>
              )}
              {evaluation.config.temperature !== undefined && (
                <Descriptions.Item label="Temperature">
                  {evaluation.config.temperature}
                </Descriptions.Item>
              )}
              {evaluation.config.max_tokens !== undefined && (
                <Descriptions.Item label="Max Tokens">
                  {evaluation.config.max_tokens}
                </Descriptions.Item>
              )}
            </Descriptions>
          )}
        </div>
      )}
    </Drawer>
  )
}
