import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Table,
  Skeleton,
  Input,
  Tag,
  Tooltip,
  notification,
  Dropdown,
  Badge
} from "antd"
import type { MenuProps } from "antd"
import {
  Plus,
  Search,
  FileText,
  History,
  Play,
  Trash2,
  Pen,
  MoreHorizontal,
  Copy,
  Undo2
} from "lucide-react"
import React, { useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { usePromptStudioStore } from "@/store/prompt-studio"
import {
  listPrompts,
  createPrompt,
  type Prompt,
  type PromptCreatePayload
} from "@/services/prompt-studio"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { Button } from "@/components/Common/Button"
import { PromptEditorDrawer } from "./PromptEditorDrawer"
import { VersionHistoryDrawer } from "./VersionHistoryDrawer"
import { ExecutePlayground } from "./ExecutePlayground"

export const StudioPromptsTab: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()

  const [searchText, setSearchText] = useState("")
  const [historyDrawerOpen, setHistoryDrawerOpen] = useState(false)

  const selectedProjectId = usePromptStudioStore((s) => s.selectedProjectId)
  const selectedPromptId = usePromptStudioStore((s) => s.selectedPromptId)
  const setSelectedPromptId = usePromptStudioStore((s) => s.setSelectedPromptId)
  const isExecutePlaygroundOpen = usePromptStudioStore(
    (s) => s.isExecutePlaygroundOpen
  )
  const setExecutePlaygroundOpen = usePromptStudioStore(
    (s) => s.setExecutePlaygroundOpen
  )
  const isPromptEditorOpen = usePromptStudioStore((s) => s.isPromptEditorOpen)
  const setPromptEditorOpen = usePromptStudioStore((s) => s.setPromptEditorOpen)
  const editingPromptId = usePromptStudioStore((s) => s.editingPromptId)
  const setEditingPromptId = usePromptStudioStore((s) => s.setEditingPromptId)

  // Fetch prompts for selected project
  const { data: promptsResponse, status: promptsStatus } = useQuery({
    queryKey: ["prompt-studio", "prompts", selectedProjectId],
    queryFn: () => listPrompts(selectedProjectId!, { per_page: 100 }),
    enabled: selectedProjectId !== null
  })

  const prompts: Prompt[] = (promptsResponse as any)?.data?.data ?? []

  // Duplicate mutation
  const duplicateMutation = useMutation({
    mutationFn: async (prompt: Prompt) => {
      const payload: PromptCreatePayload = {
        project_id: prompt.project_id,
        name: `${prompt.name} (Copy)`,
        system_prompt: prompt.system_prompt,
        user_prompt: prompt.user_prompt,
        prompt_format: prompt.prompt_format,
        prompt_schema_version: prompt.prompt_schema_version,
        prompt_definition: prompt.prompt_definition,
        few_shot_examples: prompt.few_shot_examples,
        modules_config: prompt.modules_config,
        change_description: `Duplicated from "${prompt.name}"`
      }
      const response = await createPrompt(payload, crypto.randomUUID())
      if (response.ok === false || response.data?.success === false) {
        throw new Error(response.error || response.data?.error || "Prompt could not be duplicated")
      }
      return response
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "prompts", selectedProjectId]
      })
      notification.success({
        message: t("managePrompts.studio.prompts.duplicated", {
          defaultValue: "Prompt duplicated"
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

  // Filter prompts
  const filteredPrompts = useMemo(() => {
    let items = prompts
    if (searchText.trim()) {
      const q = searchText.toLowerCase()
      items = items.filter(
        (p) => {
          const structuredContent = Array.isArray(p.prompt_definition?.blocks)
            ? p.prompt_definition.blocks
                .map((block: any) =>
                  typeof block?.content === "string" ? block.content : ""
                )
                .join(" ")
                .toLowerCase()
            : ""
          return (
            p.name.toLowerCase().includes(q) ||
            p.system_prompt?.toLowerCase().includes(q) ||
            p.user_prompt?.toLowerCase().includes(q) ||
            structuredContent.includes(q)
          )
        }
      )
    }
    return items.sort(
      (a, b) =>
        new Date(b.updated_at || b.created_at || 0).getTime() -
        new Date(a.updated_at || a.created_at || 0).getTime()
    )
  }, [prompts, searchText])

  const handleOpenCreate = () => {
    setEditingPromptId(null)
    setPromptEditorOpen(true)
  }

  const handleOpenEdit = (prompt: Prompt) => {
    setEditingPromptId(prompt.id)
    setPromptEditorOpen(true)
  }

  const handleViewHistory = (prompt: Prompt) => {
    setSelectedPromptId(prompt.id)
    setHistoryDrawerOpen(true)
  }

  const handleExecute = (prompt: Prompt) => {
    setSelectedPromptId(prompt.id)
    setExecutePlaygroundOpen(true)
  }

  const handleDuplicate = (prompt: Prompt) => {
    duplicateMutation.mutate(prompt)
  }

  const getPromptActions = (prompt: Prompt): MenuProps["items"] => [
    {
      key: "edit",
      icon: <Pen className="size-4" />,
      label: t("common:edit", { defaultValue: "Edit" }),
      onClick: () => handleOpenEdit(prompt)
    },
    {
      key: "execute",
      icon: <Play className="size-4" />,
      label: t("managePrompts.studio.prompts.execute", {
        defaultValue: "Execute"
      }),
      onClick: () => handleExecute(prompt)
    },
    {
      key: "history",
      icon: <History className="size-4" />,
      label: t("managePrompts.studio.prompts.viewHistory", {
        defaultValue: "Version History"
      }),
      onClick: () => handleViewHistory(prompt)
    },
    { type: "divider" },
    {
      key: "duplicate",
      icon: <Copy className="size-4" />,
      label: t("common:duplicate", { defaultValue: "Duplicate" }),
      onClick: () => handleDuplicate(prompt)
    }
  ]

  if (!selectedProjectId) {
    return (
      <FeatureEmptyState
        title={t("managePrompts.studio.prompts.noProjectSelected", {
          defaultValue: "Select a project first"
        })}
        description={t("managePrompts.studio.prompts.noProjectSelectedDesc", {
          defaultValue:
            "Go to the Projects tab and select a project to manage its prompts."
        })}
        examples={[]}
      />
    )
  }

  if (promptsStatus === "pending") {
    return <Skeleton paragraph={{ rows: 8 }} />
  }

  if (promptsStatus === "success" && prompts.length === 0) {
    return (
      <>
        <FeatureEmptyState
          title={t("managePrompts.studio.prompts.emptyTitle", {
            defaultValue: "No prompts in this project"
          })}
          description={t("managePrompts.studio.prompts.emptyDescription", {
            defaultValue:
              "Create your first prompt to get started with evaluations and optimizations."
          })}
          examples={[
            t("managePrompts.studio.prompts.emptyExample1", {
              defaultValue:
                "Prompts can have system instructions and user message templates."
            }),
            t("managePrompts.studio.prompts.emptyExample2", {
              defaultValue:
                "Add few-shot examples to improve model performance on specific tasks."
            })
          ]}
          primaryActionLabel={t("managePrompts.studio.prompts.createBtn", {
            defaultValue: "Create Prompt"
          })}
          onPrimaryAction={handleOpenCreate}
        />
        <PromptEditorDrawer
          open={isPromptEditorOpen}
          promptId={editingPromptId}
          projectId={selectedProjectId}
          onClose={() => {
            setPromptEditorOpen(false)
            setEditingPromptId(null)
          }}
        />
      </>
    )
  }

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Button type="primary" onClick={handleOpenCreate}>
            <Plus className="size-4 mr-1" />
            {t("managePrompts.studio.prompts.createBtn", {
              defaultValue: "Create Prompt"
            })}
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Input
            placeholder={t("managePrompts.studio.prompts.searchPlaceholder", {
              defaultValue: "Search prompts..."
            })}
            prefix={<Search className="size-4 text-text-muted" />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            allowClear
            style={{ width: 220 }}
          />
        </div>
      </div>

      {/* Prompts table */}
      <Table<Prompt>
        dataSource={filteredPrompts}
        rowKey="id"
        size="middle"
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showTotal: (total) =>
            t("managePrompts.studio.prompts.totalCount", {
              defaultValue: "{{count}} prompts",
              count: total
            })
        }}
        rowClassName={(record) =>
          record.id === selectedPromptId
            ? "bg-primary/10"
            : "cursor-pointer hover:bg-surface2"
        }
        onRow={(record) => ({
          onClick: () => setSelectedPromptId(record.id),
          onDoubleClick: () => handleOpenEdit(record)
        })}
        columns={[
          {
            title: "",
            width: 40,
            render: () => <FileText className="size-5 text-primary" />
          },
          {
            title: t("managePrompts.studio.prompts.columns.name", {
              defaultValue: "Name"
            }),
            dataIndex: "name",
            key: "name",
            render: (name: string, record) => (
              <div className="flex flex-col">
                <span className="font-medium">{name}</span>
                {record.change_description && (
                  <span className="text-xs text-text-muted line-clamp-1">
                    {record.change_description}
                  </span>
                )}
              </div>
            )
          },
          {
            title: t("managePrompts.studio.prompts.columns.content", {
              defaultValue: "Content"
            }),
            key: "content",
            render: (_, record) => {
              const hasSystem = !!record.system_prompt?.trim()
              const hasUser = !!record.user_prompt?.trim()
              const hasFewShot =
                record.few_shot_examples && record.few_shot_examples.length > 0
              const isStructured = record.prompt_format === "structured"
              return (
                <div className="flex flex-col gap-1 max-w-md">
                  {isStructured && <Tag color="geekblue">Structured</Tag>}
                  {hasSystem && (
                    <div className="flex items-start gap-2">
                      <Tag color="volcano" className="shrink-0">
                        System
                      </Tag>
                      <span className="line-clamp-1 text-sm">
                        {record.system_prompt}
                      </span>
                    </div>
                  )}
                  {hasUser && (
                    <div className="flex items-start gap-2">
                      <Tag color="blue" className="shrink-0">
                        User
                      </Tag>
                      <span className="line-clamp-1 text-sm">
                        {record.user_prompt}
                      </span>
                    </div>
                  )}
                  {hasFewShot && (
                    <Tag color="purple">
                      {record.few_shot_examples!.length} examples
                    </Tag>
                  )}
                  {!hasSystem && !hasUser && (
                    <span className="text-text-muted text-sm italic">
                      No content
                    </span>
                  )}
                </div>
              )
            }
          },
          {
            title: t("managePrompts.studio.prompts.columns.version", {
              defaultValue: "Version"
            }),
            key: "version",
            width: 100,
            render: (_, record) => (
              <Tooltip
                title={t("managePrompts.studio.prompts.versionTooltip", {
                  defaultValue: "Click to view version history"
                })}
              >
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleViewHistory(record)
                  }}
                  className="flex items-center gap-1 text-sm text-text-muted hover:text-primary"
                >
                  <Badge count={`v${record.version_number}`} showZero />
                  <History className="size-3" />
                </button>
              </Tooltip>
            )
          },
          {
            title: t("managePrompts.studio.prompts.columns.updated", {
              defaultValue: "Updated"
            }),
            key: "updated_at",
            width: 140,
            render: (_, record) => {
              const date = record.updated_at || record.created_at
              if (!date) return "-"
              return new Date(date).toLocaleDateString()
            }
          },
          {
            title: "",
            key: "actions",
            width: 50,
            render: (_, record) => (
              <Dropdown
                menu={{ items: getPromptActions(record) }}
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

      {/* Prompt editor drawer */}
      <PromptEditorDrawer
        open={isPromptEditorOpen}
        promptId={editingPromptId}
        projectId={selectedProjectId}
        onClose={() => {
          setPromptEditorOpen(false)
          setEditingPromptId(null)
        }}
      />

      {/* Version history drawer */}
      <VersionHistoryDrawer
        open={historyDrawerOpen}
        promptId={selectedPromptId}
        onClose={() => setHistoryDrawerOpen(false)}
      />

      {/* Execute playground drawer */}
      <ExecutePlayground
        open={isExecutePlaygroundOpen}
        promptId={selectedPromptId}
        onClose={() => setExecutePlaygroundOpen(false)}
      />
    </div>
  )
}
