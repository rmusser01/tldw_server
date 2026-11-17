import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Table,
  Skeleton,
  Tag,
  notification,
  Dropdown,
  Progress
} from "antd"
import type { MenuProps } from "antd"
import {
  Plus,
  Sparkles,
  MoreHorizontal,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Eye,
  StopCircle
} from "lucide-react"
import React, { useState } from "react"
import { useTranslation } from "react-i18next"
import { usePromptStudioStore } from "@/store/prompt-studio"
import {
  listOptimizations,
  cancelOptimization,
  type Optimization
} from "@/services/prompt-studio"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { Button } from "@/components/Common/Button"
import { Alert } from "@/components/ui/primitives"
import { CreateOptimizationWizard } from "./CreateOptimizationWizard"
import { OptimizationProgressPanel } from "./OptimizationProgressPanel"

export const OptimizationsTab: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()

  const [progressPanelOpen, setProgressPanelOpen] = useState(false)
  const [selectedOptForProgress, setSelectedOptForProgress] = useState<number | null>(null)

  const selectedProjectId = usePromptStudioStore((s) => s.selectedProjectId)
  const isOptimizationWizardOpen = usePromptStudioStore(
    (s) => s.isOptimizationWizardOpen
  )
  const setOptimizationWizardOpen = usePromptStudioStore(
    (s) => s.setOptimizationWizardOpen
  )

  // Fetch optimizations with auto-refresh for running jobs
  const { data: optimizationsResponse, status: optimizationsStatus } = useQuery({
    queryKey: ["prompt-studio", "optimizations", selectedProjectId],
    queryFn: () =>
      listOptimizations({
        project_id: selectedProjectId!,
        per_page: 100
      }),
    enabled: selectedProjectId !== null,
    refetchInterval: (query) => {
      const data = query.state.data as any
      const optimizations = data?.data?.data ?? []
      const hasRunning = optimizations.some((o: Optimization) =>
        ["running", "pending"].includes(o.status?.toLowerCase())
      )
      return hasRunning ? 5000 : false
    }
  })

  const optimizations: Optimization[] =
    optimizationsResponse?.data?.data ?? []

  // Cancel mutation
  const cancelMutation = useMutation({
    mutationFn: (id: number) => cancelOptimization(id),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "optimizations", selectedProjectId]
      })
      notification.success({
        message: t("managePrompts.studio.optimizations.cancelled", {
          defaultValue: "Optimization cancelled"
        })
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error.label", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  const handleCancel = async (optimization: Optimization) => {
    const ok = await confirmDanger({
      title: t("managePrompts.studio.optimizations.cancelConfirmTitle", {
        defaultValue: "Cancel optimization?"
      }),
      content: t("managePrompts.studio.optimizations.cancelConfirmContent", {
        defaultValue:
          "This will stop the optimization. Progress made so far will be preserved."
      }),
      okText: t("managePrompts.studio.optimizations.cancelBtn", {
        defaultValue: "Cancel Optimization"
      }),
      cancelText: t("common:keepRunning", { defaultValue: "Keep Running" })
    })
    if (ok) {
      cancelMutation.mutate(optimization.id)
    }
  }

  const handleViewProgress = (optimization: Optimization) => {
    setSelectedOptForProgress(optimization.id)
    setProgressPanelOpen(true)
  }

  const getStatusTag = (status: string) => {
    const statusLower = status?.toLowerCase()
    switch (statusLower) {
      case "completed":
        return (
          <Tag color="green" icon={<CheckCircle2 className="size-3" />}>
            {t("managePrompts.studio.optimizations.statusCompleted", {
              defaultValue: "Completed"
            })}
          </Tag>
        )
      case "running":
        return (
          <Tag color="blue" icon={<Loader2 className="size-3 animate-spin" />}>
            {t("managePrompts.studio.optimizations.statusRunning", {
              defaultValue: "Running"
            })}
          </Tag>
        )
      case "pending":
        return (
          <Tag color="default" icon={<Clock className="size-3" />}>
            {t("managePrompts.studio.optimizations.statusPending", {
              defaultValue: "Pending"
            })}
          </Tag>
        )
      case "failed":
        return (
          <Tag color="red" icon={<XCircle className="size-3" />}>
            {t("managePrompts.studio.optimizations.statusFailed", {
              defaultValue: "Failed"
            })}
          </Tag>
        )
      case "cancelled":
        return (
          <Tag color="orange" icon={<StopCircle className="size-3" />}>
            {t("managePrompts.studio.optimizations.statusCancelled", {
              defaultValue: "Cancelled"
            })}
          </Tag>
        )
      default:
        return <Tag>{status}</Tag>
    }
  }

  const getStrategyLabel = (strategy: string) => {
    const labels: Record<string, string> = {
      iterative: "Iterative",
      mipro: "MIPRO",
      bootstrap: "Bootstrap",
      hyperparameter: "Hyperparameter",
      genetic: "Genetic",
      beam_search: "Beam Search",
      simulated_annealing: "Simulated Annealing",
      random_search: "Random Search",
      hill_climbing: "Hill Climbing",
      mcts: "MCTS"
    }
    return labels[strategy] || strategy
  }

  const getOptimizationActions = (
    optimization: Optimization
  ): MenuProps["items"] => {
    const isRunning = ["running", "pending"].includes(
      optimization.status?.toLowerCase()
    )

    return [
      {
        key: "view",
        icon: <Eye className="size-4" />,
        label: t("managePrompts.studio.optimizations.viewProgress", {
          defaultValue: "View Progress"
        }),
        onClick: () => handleViewProgress(optimization)
      },
      ...(isRunning
        ? [
            { type: "divider" as const },
            {
              key: "cancel",
              icon: <StopCircle className="size-4" />,
              label: t("managePrompts.studio.optimizations.cancel", {
                defaultValue: "Cancel"
              }),
              danger: true,
              onClick: () => handleCancel(optimization)
            }
          ]
        : [])
    ]
  }

  if (!selectedProjectId) {
    return (
      <FeatureEmptyState
        title={t("managePrompts.studio.optimizations.noProjectSelected", {
          defaultValue: "Select a project first"
        })}
        description={t(
          "managePrompts.studio.optimizations.noProjectSelectedDesc",
          {
            defaultValue:
              "Go to the Projects tab and select a project to view its optimizations."
          }
        )}
        examples={[]}
      />
    )
  }

  if (optimizationsStatus === "pending") {
    return <Skeleton paragraph={{ rows: 8 }} />
  }

  if (optimizationsStatus === "error") {
    return <Alert variant="error" title={t("managePrompts.studio.optimizations.loadError", {
      defaultValue: "Unable to load optimizations"
    })} />
  }

  if (optimizationsStatus === "success" && optimizations.length === 0) {
    return (
      <>
        <FeatureEmptyState
          title={t("managePrompts.studio.optimizations.emptyTitle", {
            defaultValue: "No optimizations yet"
          })}
          description={t(
            "managePrompts.studio.optimizations.emptyDescription",
            {
              defaultValue:
                "Run optimization jobs to automatically improve your prompts using various strategies."
            }
          )}
          examples={[
            t("managePrompts.studio.optimizations.emptyExample1", {
              defaultValue:
                "Optimizations iteratively refine prompts to maximize evaluation scores."
            }),
            t("managePrompts.studio.optimizations.emptyExample2", {
              defaultValue:
                "Choose from the optimization strategies available on this server."
            })
          ]}
          primaryActionLabel={t(
            "managePrompts.studio.optimizations.createBtn",
            {
              defaultValue: "Start Optimization"
            }
          )}
          onPrimaryAction={() => setOptimizationWizardOpen(true)}
        />
        <CreateOptimizationWizard
          open={isOptimizationWizardOpen}
          projectId={selectedProjectId}
          onClose={() => setOptimizationWizardOpen(false)}
        />
      </>
    )
  }

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Button type="primary" onClick={() => setOptimizationWizardOpen(true)}>
          <Plus className="size-4 mr-1" />
          {t("managePrompts.studio.optimizations.createBtn", {
            defaultValue: "Start Optimization"
          })}
        </Button>
      </div>

      {/* Optimizations table */}
      <Table<Optimization>
        dataSource={optimizations}
        rowKey="id"
        size="middle"
        pagination={{
          pageSize: 10,
          showTotal: (total) =>
            t("managePrompts.studio.optimizations.totalCount", {
              defaultValue: "{{count}} optimizations",
              count: total
            })
        }}
        onRow={(record) => ({
          onDoubleClick: () => handleViewProgress(record),
          className: "cursor-pointer hover:bg-surface2"
        })}
        columns={[
          {
            title: "",
            width: 40,
            render: () => <Sparkles className="size-5 text-primary" />
          },
          {
            title: t("managePrompts.studio.optimizations.columns.name", {
              defaultValue: "Name"
            }),
            key: "name",
            render: (_, record) => (
              <div className="flex flex-col">
                <span className="font-medium">
                  {record.name || `Optimization #${record.id}`}
                </span>
                {record.description && (
                  <span className="text-xs text-text-muted line-clamp-1">
                    {record.description}
                  </span>
                )}
              </div>
            )
          },
          {
            title: t("managePrompts.studio.optimizations.columns.strategy", {
              defaultValue: "Strategy"
            }),
            key: "strategy",
            width: 140,
            render: (_, record) => (
              <Tag color="purple">
                {getStrategyLabel(record.optimizer_type)}
              </Tag>
            )
          },
          {
            title: t("managePrompts.studio.optimizations.columns.status", {
              defaultValue: "Status"
            }),
            key: "status",
            width: 120,
            render: (_, record) => getStatusTag(record.status)
          },
          {
            title: t("managePrompts.studio.optimizations.columns.progress", {
              defaultValue: "Progress"
            }),
            key: "progress",
            width: 150,
            render: (_, record) => {
              if (!record.max_iterations) return "-"
              const current = record.iterations_completed ?? 0
              const total = record.max_iterations
              const percent = record.status?.toLowerCase() === "completed"
                ? 100
                : Math.round((current / total) * 100)
              return (
                <div>
                  <Progress
                    percent={percent}
                    size="small"
                    status={
                      record.status === "running" ? "active" : undefined
                    }
                  />
                  <span className="text-xs text-text-muted">
                    {current} / {total}
                  </span>
                </div>
              )
            }
          },
          {
            title: t("managePrompts.studio.optimizations.columns.bestScore", {
              defaultValue: "Best Score"
            }),
            key: "best_score",
            width: 100,
            render: (_, record) =>
              typeof record.final_metrics?.score === "number" && Number.isFinite(record.final_metrics.score) ? (
                <span className="font-medium text-success">
                  {(record.final_metrics.score * 100).toFixed(1)}%
                </span>
              ) : (
                <span className="text-text-muted">-</span>
              )
          },
          {
            title: t("managePrompts.studio.optimizations.columns.created", {
              defaultValue: "Created"
            }),
            key: "created_at",
            width: 120,
            render: (_, record) => {
              if (!record.created_at) return "-"
              return new Date(record.created_at).toLocaleDateString()
            }
          },
          {
            title: "",
            key: "actions",
            width: 50,
            render: (_, record) => (
              <Dropdown
                menu={{ items: getOptimizationActions(record) }}
                trigger={["click"]}
                placement="bottomRight"
              >
                <button
                  onClick={(e) => e.stopPropagation()}
                  className="p-1 rounded hover:bg-surface2"
                >
                  <MoreHorizontal className="size-4" />
                </button>
              </Dropdown>
            )
          }
        ]}
      />

      {/* Wizard */}
      <CreateOptimizationWizard
        open={isOptimizationWizardOpen}
        projectId={selectedProjectId}
        onClose={() => setOptimizationWizardOpen(false)}
      />

      {/* Progress panel */}
      <OptimizationProgressPanel
        open={progressPanelOpen}
        optimizationId={selectedOptForProgress}
        onClose={() => {
          setProgressPanelOpen(false)
          setSelectedOptForProgress(null)
        }}
      />
    </div>
  )
}
