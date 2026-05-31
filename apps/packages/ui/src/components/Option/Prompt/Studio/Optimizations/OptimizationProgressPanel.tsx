import { useQuery } from "@tanstack/react-query"
import {
  Drawer,
  Skeleton,
  Progress,
  Descriptions,
  Timeline,
  Statistic
} from "antd"
import {
  Sparkles,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  TrendingUp,
  Target,
  StopCircle
} from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import {
  getOptimization,
  getOptimizationIterations,
  type Optimization,
  type OptimizationIteration
} from "@/services/prompt-studio"
import {
  Alert as DsAlert,
  Badge,
  type BadgeVariant
} from "@/components/ui/primitives"

type OptimizationProgressPanelProps = {
  open: boolean
  optimizationId: number | null
  onClose: () => void
}

export const OptimizationProgressPanel: React.FC<
  OptimizationProgressPanelProps
> = ({ open, optimizationId, onClose }) => {
  const { t } = useTranslation(["settings", "common"])

  // Fetch optimization with auto-refresh for running jobs
  const { data: optimizationResponse, isLoading: isLoadingOptimization } =
    useQuery({
      queryKey: ["prompt-studio", "optimization", optimizationId],
      queryFn: () => getOptimization(optimizationId!),
      enabled: open && optimizationId !== null,
      refetchInterval: (query) => {
        const data = query.state.data as any
        const optimization = data?.data?.data
        const isRunning = ["running", "pending"].includes(
          optimization?.status?.toLowerCase()
        )
        return isRunning ? 3000 : false
      }
    })

  const optimization: Optimization | undefined =
    (optimizationResponse as any)?.data?.data

  // Fetch iterations
  const { data: iterationsResponse, isLoading: isLoadingIterations } = useQuery(
    {
      queryKey: ["prompt-studio", "optimization-iterations", optimizationId],
      queryFn: () => getOptimizationIterations(optimizationId!),
      enabled: open && optimizationId !== null,
      refetchInterval: (query) => {
        const isRunning = ["running", "pending"].includes(
          optimization?.status?.toLowerCase() ?? ""
        )
        return isRunning ? 5000 : false
      }
    }
  )

  const iterations: OptimizationIteration[] =
    (iterationsResponse as any)?.data?.data ?? []

  const renderStatusBadge = (status?: string) => {
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
      case "cancelled":
        return (
          <Badge variant="warning" size="sm">
            <StopCircle className="size-3" aria-hidden="true" />
            Cancelled
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

  const strategyBadgeVariant = (strategy?: string): BadgeVariant => {
    if (!strategy) return "secondary"
    if (strategy === "mipro" || strategy === "genetic") return "primary"
    if (strategy === "bootstrap") return "success"
    return "secondary"
  }

  const getStrategyLabel = (strategy?: string) => {
    if (!strategy) return "-"
    const labels: Record<string, string> = {
      iterative: "Iterative",
      mipro: "MIPRO",
      bootstrap: "Bootstrap",
      genetic: "Genetic Algorithm",
      beam_search: "Beam Search",
      random_search: "Random Search",
      hill_climbing: "Hill Climbing",
      simulated_annealing: "Simulated Annealing"
    }
    return labels[strategy] || strategy
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

  const progressPercent = optimization?.total_iterations
    ? Math.round(
        ((optimization.current_iteration ?? 0) /
          optimization.total_iterations) *
          100
      )
    : 0

  // Find best iteration
  const bestIteration = iterations.length > 0
    ? iterations.reduce((best, current) =>
        current.score > best.score ? current : best
      )
    : null

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={
        <span className="flex items-center gap-2">
          <Sparkles className="size-5" />
          {t("managePrompts.studio.optimizations.progressTitle", {
            defaultValue: "Optimization Progress"
          })}
        </span>
      }
      size={650}
      destroyOnHidden
    >
      {isLoadingOptimization && <Skeleton paragraph={{ rows: 10 }} />}

      {!isLoadingOptimization && optimization && (
        <div className="space-y-6">
          {/* Header */}
          <div className="p-4 bg-surface2 rounded-md">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-lg">
                {optimization.name || `Optimization #${optimization.id}`}
              </h3>
              {renderStatusBadge(optimization.status)}
            </div>
            {optimization.description && (
              <p className="text-sm text-text-muted">
                {optimization.description}
              </p>
            )}
          </div>

          {/* Error message if failed */}
          {optimization.error_message && (
            <DsAlert
              variant="error"
              title={t("managePrompts.studio.optimizations.error", {
                defaultValue: "Error"
              })}
            >
              {optimization.error_message}
            </DsAlert>
          )}

          {/* Cancel reason */}
          {optimization.cancel_reason && (
            <DsAlert
              variant="warning"
              title={t("managePrompts.studio.optimizations.cancelled", {
                defaultValue: "Cancelled"
              })}
            >
              {optimization.cancel_reason}
            </DsAlert>
          )}

          {/* Progress */}
          {optimization.total_iterations && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">
                  {t("managePrompts.studio.optimizations.progress", {
                    defaultValue: "Progress"
                  })}
                </span>
                <span className="text-sm text-text-muted">
                  {optimization.current_iteration ?? 0} /{" "}
                  {optimization.total_iterations} iterations
                </span>
              </div>
              <Progress
                percent={progressPercent}
                status={optimization.status === "running" ? "active" : undefined}
              />
            </div>
          )}

          {/* Stats */}
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-surface2 rounded-md text-center">
              <Statistic
                title={
                  <span className="flex items-center justify-center gap-1">
                    <Target className="size-4" />
                    {t("managePrompts.studio.optimizations.bestScore", {
                      defaultValue: "Best Score"
                    })}
                  </span>
                }
                value={
                  optimization.best_score !== undefined &&
                  optimization.best_score !== null
                    ? optimization.best_score * 100
                    : 0
                }
                precision={1}
                suffix="%"
                valueStyle={{
                  color: metricColor(optimization.best_score ?? 0)
                }}
              />
            </div>
            <div className="p-3 bg-surface2 rounded-md text-center">
              <Statistic
                title={
                  <span className="flex items-center justify-center gap-1">
                    <TrendingUp className="size-4" />
                    {t("managePrompts.studio.optimizations.iterations", {
                      defaultValue: "Iterations"
                    })}
                  </span>
                }
                value={optimization.current_iteration ?? 0}
                suffix={`/ ${optimization.total_iterations ?? "?"}`}
              />
            </div>
          </div>

          {/* Configuration */}
          <Descriptions
            title={t("managePrompts.studio.optimizations.configuration", {
              defaultValue: "Configuration"
            })}
            column={2}
            size="small"
            bordered
          >
            <Descriptions.Item label="Strategy">
              <Badge
                variant={strategyBadgeVariant(optimization.config?.strategy)}
                size="sm"
              >
                {getStrategyLabel(optimization.config?.strategy)}
              </Badge>
            </Descriptions.Item>
            <Descriptions.Item label="Prompt ID">
              {optimization.prompt_id}
            </Descriptions.Item>
            <Descriptions.Item label="Started">
              {formatDate(optimization.started_at)}
            </Descriptions.Item>
            <Descriptions.Item label="Completed">
              {formatDate(optimization.completed_at)}
            </Descriptions.Item>
          </Descriptions>

          {/* Best prompt info */}
          {optimization.best_prompt_id && (
            <DsAlert
              variant="success"
              icon={<CheckCircle2 className="size-4" />}
              title={t("managePrompts.studio.optimizations.bestPromptFound", {
                defaultValue: "Best prompt found"
              })}
            >
              {t(
                "managePrompts.studio.optimizations.bestPromptDesc",
                {
                  defaultValue:
                    "The optimized prompt has been saved as Prompt #{{promptId}}",
                  promptId: optimization.best_prompt_id
                }
              )}
            </DsAlert>
          )}

          {/* Iteration timeline */}
          {iterations.length > 0 && (
            <div>
              <h4 className="font-medium mb-3">
                {t("managePrompts.studio.optimizations.iterationHistory", {
                  defaultValue: "Iteration History"
                })}
              </h4>
              <div className="max-h-[300px] overflow-y-auto">
                <Timeline
                  items={iterations
                    .sort((a, b) => b.iteration - a.iteration)
                    .slice(0, 20)
                    .map((iteration) => {
                      const isBest =
                        bestIteration &&
                        iteration.iteration === bestIteration.iteration
                      return {
                        color: isBest ? "green" : "gray",
                        dot: isBest ? (
                          <CheckCircle2 className="size-4 text-success" />
                        ) : undefined,
                        children: (
                          <div
                            className={`p-2 rounded-md border ${
                              isBest
                                ? "border-success/30 bg-success/5"
                                : "border-border bg-surface2/50"
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium">
                                Iteration {iteration.iteration}
                              </span>
                              <span
                                className={`font-medium ${
                                  iteration.score >= 0.8
                                    ? "text-success"
                                    : iteration.score >= 0.5
                                    ? "text-warn"
                                    : "text-danger"
                                }`}
                              >
                                {(iteration.score * 100).toFixed(1)}%
                              </span>
                            </div>
                            {iteration.changes && (
                              <p className="text-xs text-text-muted mt-1 line-clamp-2">
                                {iteration.changes}
                              </p>
                            )}
                            {iteration.timestamp && (
                              <p className="text-xs text-text-muted mt-1">
                                {new Date(iteration.timestamp).toLocaleTimeString()}
                              </p>
                            )}
                          </div>
                        )
                      }
                    })}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </Drawer>
  )
}
