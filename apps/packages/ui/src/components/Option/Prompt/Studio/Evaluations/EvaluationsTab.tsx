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
  BarChart3,
  Trash2,
  MoreHorizontal,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Eye
} from "lucide-react"
import React, { useState } from "react"
import { useTranslation } from "react-i18next"
import { usePromptStudioStore } from "@/store/prompt-studio"
import {
  listEvaluations,
  deleteEvaluation,
  type PromptStudioEvaluation
} from "@/services/prompt-studio"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { Button } from "@/components/Common/Button"
import { CreateEvaluationWizard } from "./CreateEvaluationWizard"
import { EvaluationDetailPanel } from "./EvaluationDetailPanel"

export const EvaluationsTab: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()

  const [detailPanelOpen, setDetailPanelOpen] = useState(false)
  const [selectedEvalForDetail, setSelectedEvalForDetail] = useState<number | null>(null)

  const selectedProjectId = usePromptStudioStore((s) => s.selectedProjectId)
  const isEvaluationWizardOpen = usePromptStudioStore((s) => s.isEvaluationWizardOpen)
  const setEvaluationWizardOpen = usePromptStudioStore((s) => s.setEvaluationWizardOpen)

  // Fetch evaluations with auto-refresh for running evaluations
  const { data: evaluationsResponse, status: evaluationsStatus } = useQuery({
    queryKey: ["prompt-studio", "evaluations", selectedProjectId],
    queryFn: () =>
      listEvaluations({
        project_id: selectedProjectId!,
        limit: 100
      }),
    enabled: selectedProjectId !== null,
    refetchInterval: (query) => {
      const data = query.state.data as any
      const evaluations = data?.data?.evaluations ?? []
      const hasRunning = evaluations.some((e: PromptStudioEvaluation) =>
        ["running", "pending"].includes(e.status?.toLowerCase())
      )
      return hasRunning ? 5000 : false
    }
  })

  const evaluations: PromptStudioEvaluation[] =
    (evaluationsResponse as any)?.data?.evaluations ?? []

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: number) => deleteEvaluation(id),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "evaluations", selectedProjectId]
      })
      notification.success({
        message: t("managePrompts.studio.evaluations.deleted", {
          defaultValue: "Evaluation deleted"
        })
      })
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  const handleDelete = async (evaluation: PromptStudioEvaluation) => {
    const ok = await confirmDanger({
      title: t("managePrompts.studio.evaluations.deleteConfirmTitle", {
        defaultValue: "Delete evaluation?"
      }),
      content: t("managePrompts.studio.evaluations.deleteConfirmContent", {
        defaultValue: "This will permanently delete this evaluation and its results."
      }),
      okText: t("common:delete", { defaultValue: "Delete" }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" })
    })
    if (ok) {
      deleteMutation.mutate(evaluation.id)
    }
  }

  const handleViewDetail = (evaluation: PromptStudioEvaluation) => {
    setSelectedEvalForDetail(evaluation.id)
    setDetailPanelOpen(true)
  }

  const getStatusTag = (status: string) => {
    const statusLower = status?.toLowerCase()
    switch (statusLower) {
      case "completed":
        return (
          <Tag color="green" icon={<CheckCircle2 className="size-3" />}>
            {t("managePrompts.studio.evaluations.statusCompleted", {
              defaultValue: "Completed"
            })}
          </Tag>
        )
      case "running":
        return (
          <Tag color="blue" icon={<Loader2 className="size-3 animate-spin" />}>
            {t("managePrompts.studio.evaluations.statusRunning", {
              defaultValue: "Running"
            })}
          </Tag>
        )
      case "pending":
        return (
          <Tag color="default" icon={<Clock className="size-3" />}>
            {t("managePrompts.studio.evaluations.statusPending", {
              defaultValue: "Pending"
            })}
          </Tag>
        )
      case "failed":
        return (
          <Tag color="red" icon={<XCircle className="size-3" />}>
            {t("managePrompts.studio.evaluations.statusFailed", {
              defaultValue: "Failed"
            })}
          </Tag>
        )
      default:
        return <Tag>{status}</Tag>
    }
  }

  const getEvaluationActions = (
    evaluation: PromptStudioEvaluation
  ): MenuProps["items"] => [
    {
      key: "view",
      icon: <Eye className="size-4" />,
      label: t("managePrompts.studio.evaluations.viewDetails", {
        defaultValue: "View Details"
      }),
      onClick: () => handleViewDetail(evaluation)
    },
    { type: "divider" },
    {
      key: "delete",
      icon: <Trash2 className="size-4" />,
      label: t("common:delete", { defaultValue: "Delete" }),
      danger: true,
      onClick: () => handleDelete(evaluation)
    }
  ]

  const formatMetrics = (metrics?: Record<string, any>) => {
    if (!metrics || Object.keys(metrics).length === 0) return null

    const displayMetrics = []
    if (metrics.accuracy !== undefined) {
      displayMetrics.push({
        label: t("managePrompts.studio.evaluations.metrics.accuracy", {
          defaultValue: "Accuracy"
        }),
        tooltip: t("managePrompts.studio.evaluations.tooltip.accuracy", {
          defaultValue:
            "How often the model's output exactly matches the expected answer"
        }),
        value: `${(metrics.accuracy * 100).toFixed(1)}%`
      })
    }
    if (metrics.pass_rate !== undefined) {
      displayMetrics.push({
        label: t("managePrompts.studio.evaluations.metrics.passRate", {
          defaultValue: "Pass Rate"
        }),
        tooltip: t("managePrompts.studio.evaluations.tooltip.passRate", {
          defaultValue:
            "Percentage of test cases where the output met the quality threshold"
        }),
        value: `${(metrics.pass_rate * 100).toFixed(1)}%`
      })
    }
    if (metrics.f1 !== undefined) {
      displayMetrics.push({
        label: t("managePrompts.studio.evaluations.metrics.f1", {
          defaultValue: "F1 Score"
        }),
        tooltip: t("managePrompts.studio.evaluations.tooltip.f1", {
          defaultValue:
            "Balances precision (relevance of results) and recall (coverage of expected content). Higher is better, max 1.0"
        }),
        value: metrics.f1.toFixed(3)
      })
    }

    return displayMetrics.length > 0 ? displayMetrics : null
  }

  if (!selectedProjectId) {
    return (
      <FeatureEmptyState
        title={t("managePrompts.studio.evaluations.noProjectSelected", {
          defaultValue: "Select a project first"
        })}
        description={t(
          "managePrompts.studio.evaluations.noProjectSelectedDesc",
          {
            defaultValue:
              "Go to the Projects tab and select a project to view its evaluations."
          }
        )}
        examples={[]}
      />
    )
  }

  if (evaluationsStatus === "pending") {
    return <Skeleton paragraph={{ rows: 8 }} />
  }

  if (evaluationsStatus === "success" && evaluations.length === 0) {
    return (
      <>
        <FeatureEmptyState
          title={t("managePrompts.studio.evaluations.emptyTitle", {
            defaultValue: "No evaluations yet"
          })}
          description={t("managePrompts.studio.evaluations.emptyDescription", {
            defaultValue:
              "Run evaluations to measure how well your prompts perform on test cases."
          })}
          examples={[
            t("managePrompts.studio.evaluations.emptyExample1", {
              defaultValue:
                "Evaluations run your prompts against test cases and measure performance."
            }),
            t("managePrompts.studio.evaluations.emptyExample2", {
              defaultValue:
                "Compare different prompt versions or model configurations."
            })
          ]}
          primaryActionLabel={t(
            "managePrompts.studio.evaluations.createBtn",
            {
              defaultValue: "Run Evaluation"
            }
          )}
          onPrimaryAction={() => setEvaluationWizardOpen(true)}
        />
        <CreateEvaluationWizard
          open={isEvaluationWizardOpen}
          projectId={selectedProjectId}
          onClose={() => setEvaluationWizardOpen(false)}
        />
      </>
    )
  }

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Button type="primary" onClick={() => setEvaluationWizardOpen(true)}>
          <Plus className="size-4 mr-1" />
          {t("managePrompts.studio.evaluations.createBtn", {
            defaultValue: "Run Evaluation"
          })}
        </Button>
      </div>

      {/* Evaluations table */}
      <Table<PromptStudioEvaluation>
        dataSource={evaluations}
        rowKey="id"
        size="middle"
        pagination={{
          pageSize: 10,
          showTotal: (total) =>
            t("managePrompts.studio.evaluations.totalCount", {
              defaultValue: "{{count}} evaluations",
              count: total
            })
        }}
        onRow={(record) => ({
          onDoubleClick: () => handleViewDetail(record),
          className: "cursor-pointer hover:bg-surface2"
        })}
        columns={[
          {
            title: "",
            width: 40,
            render: () => <BarChart3 className="size-5 text-primary" />
          },
          {
            title: t("managePrompts.studio.evaluations.columns.name", {
              defaultValue: "Name"
            }),
            key: "name",
            render: (_, record) => (
              <div className="flex flex-col">
                <span className="font-medium">
                  {record.name || `Evaluation #${record.id}`}
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
            title: t("managePrompts.studio.evaluations.columns.status", {
              defaultValue: "Status"
            }),
            key: "status",
            width: 120,
            render: (_, record) => getStatusTag(record.status)
          },
          {
            title: t("managePrompts.studio.evaluations.columns.testCases", {
              defaultValue: "Test Cases"
            }),
            key: "test_cases",
            width: 100,
            render: (_, record) => record.test_case_ids?.length ?? 0
          },
          {
            title: t("managePrompts.studio.evaluations.columns.metrics", {
              defaultValue: "Metrics"
            }),
            key: "metrics",
            render: (_, record) => {
              const metrics = formatMetrics(record.aggregate_metrics)
              if (!metrics)
                return <span className="text-text-muted">-</span>
              return (
                <div className="flex gap-3">
                  {metrics.map((m) => (
                    <div key={m.label} className="text-sm" title={m.tooltip || undefined}>
                      <span className="text-text-muted">{m.label}:</span>{" "}
                      <span className="font-medium">{m.value}</span>
                    </div>
                  ))}
                </div>
              )
            }
          },
          {
            title: t("managePrompts.studio.evaluations.columns.created", {
              defaultValue: "Created"
            }),
            key: "created_at",
            width: 140,
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
                menu={{ items: getEvaluationActions(record) }}
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
      <CreateEvaluationWizard
        open={isEvaluationWizardOpen}
        projectId={selectedProjectId}
        onClose={() => setEvaluationWizardOpen(false)}
      />

      {/* Detail panel */}
      <EvaluationDetailPanel
        open={detailPanelOpen}
        evaluationId={selectedEvalForDetail}
        onClose={() => {
          setDetailPanelOpen(false)
          setSelectedEvalForDetail(null)
        }}
      />
    </div>
  )
}
