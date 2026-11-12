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
  const { data: optimizationResponse, isLoading: isLoadingOptimization, isError: isOptimizationError } =
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
  const { data: iterationsResponse, isLoading: isLoadingIterations, isError: isIterationsError } = useQuery(
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
    iterationsResponse?.data?.data?.iterations ?? []

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

  const progressPercent = optimization?.status?.toLowerCase() === "completed"
    ? 100
    : optimization?.max_iterations
    ? Math.round(
        ((optimization.iterations_completed ?? 0) /
          optimization.max_iterations) *
          100
      )
    : 0

  const bestScore = typeof optimization?.final_metrics?.score === "number" && Number.isFinite(optimization.final_metrics.score)
    ? optimization.final_metrics.score
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

      {isOptimizationError && <DsAlert variant="error" title={t("managePrompts.studio.optimizations.detailLoadError", {
        defaultValue: "Unable to load optimization"
      })} />}

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
          {optimization.max_iterations > 0 && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium">
                  {t("managePrompts.studio.optimizations.progress.title", {
                    defaultValue: "Progress"
                  })}
                </span>
                <span className="text-sm text-text-muted">
                  {optimization.iterations_completed ?? 0} /{" "}
                  {optimization.max_iterations} iterations
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
                  bestScore === null ? t("common:unavailable", { defaultValue: "Unavailable" }) : bestScore * 100
                }
                precision={1}
                suffix={bestScore === null ? undefined : "%"}
                valueStyle={{
                  color: bestScore === null ? undefined : metricColor(bestScore)
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
                value={optimization.iterations_completed ?? 0}
                suffix={`/ ${optimization.max_iterations ?? "?"}`}
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
                variant={strategyBadgeVariant(optimization.optimizer_type)}
                size="sm"
              >
                {getStrategyLabel(optimization.optimizer_type)}
              </Badge>
            </Descriptions.Item>
            <Descriptions.Item label="Prompt ID">
              {optimization.initial_prompt_id}
            </Descriptions.Item>
            <Descriptions.Item label="Started">
              {formatDate(optimization.started_at)}
            </Descriptions.Item>
            <Descriptions.Item label="Completed">
              {formatDate(optimization.completed_at)}
            </Descriptions.Item>
          </Descriptions>

          {/* Best prompt info */}
          {optimization.optimized_prompt_id && (
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
                  promptId: optimization.optimized_prompt_id
                }
              )}
            </DsAlert>
          )}

          {/* Iteration timeline */}
          {isLoadingIterations && <Skeleton paragraph={{ rows: 2 }} />}
          {isIterationsError && <DsAlert variant="error" title={t("managePrompts.studio.optimizations.iterationsLoadError", {
            defaultValue: "Unable to load stored iterations"
          })} />}
          {!isLoadingIterations && !isIterationsError && iterations.length === 0 && <p className="text-sm text-text-muted">
            {t("managePrompts.studio.optimizations.noStoredIterations", { defaultValue: "No stored iteration records" })}
          </p>}
          {!isLoadingIterations && !isIterationsError && iterations.length > 0 && (
            <div>
              <h4 className="font-medium mb-3">
                {t("managePrompts.studio.optimizations.iterationHistory", {
                  defaultValue: "Iteration History"
                })}
              </h4>
              <div className="max-h-[300px] overflow-y-auto">
                <Timeline
                  items={[...iterations]
                    .sort((a, b) => b.iteration_number - a.iteration_number)
                    .slice(0, 20)
                    .map((iteration) => {
                      const score = typeof iteration.metrics?.score === "number" && Number.isFinite(iteration.metrics.score)
                        ? iteration.metrics.score : null
                      return {
                        color: "gray",
                        children: (
                          <div
                            className="p-2 rounded-md border border-border bg-surface2/50"
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium">
                                Iteration {iteration.iteration_number}
                              </span>
                              <span
                                className={`font-medium ${
                                  score !== null && score >= 0.8
                                    ? "text-success"
                                    : score !== null && score >= 0.5
                                    ? "text-warn"
                                    : "text-danger"
                                }`}
                              >
                                {score === null ? t("managePrompts.studio.optimizations.scoreUnavailable", { defaultValue: "Score unavailable" }) : `${(score * 100).toFixed(1)}%`}
                              </span>
                            </div>
                            {iteration.note && (
                              <p className="text-xs text-text-muted mt-1 line-clamp-2">
                                {iteration.note}
                              </p>
                            )}
                            {iteration.created_at && (
                              <p className="text-xs text-text-muted mt-1">
                                {new Date(iteration.created_at).toLocaleTimeString()}
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
