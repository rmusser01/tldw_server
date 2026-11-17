import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Table,
  Skeleton,
  Input,
  Tag,
  Tooltip,
  notification,
  Dropdown
} from "antd"
import type { MenuProps } from "antd"
import {
  Plus,
  Search,
  FolderKanban,
  Archive,
  ArchiveRestore,
  Trash2,
  Pen,
  MoreHorizontal,
  FileText,
  TestTube,
  BarChart3
} from "lucide-react"
import React, { useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { usePromptStudioStore } from "@/store/prompt-studio"
import {
  listProjects,
  archiveProject,
  unarchiveProject,
  deleteProject,
  type Project
} from "@/services/prompt-studio"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import { Button } from "@/components/Common/Button"
import { ProjectFormModal } from "./ProjectFormModal"

export const ProjectsTab: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()

  const [searchText, setSearchText] = useState("")
  const [showArchived, setShowArchived] = useState(false)

  const selectedProjectId = usePromptStudioStore((s) => s.selectedProjectId)
  const setSelectedProjectId = usePromptStudioStore((s) => s.setSelectedProjectId)
  const setActiveSubTab = usePromptStudioStore((s) => s.setActiveSubTab)
  const isProjectModalOpen = usePromptStudioStore((s) => s.isProjectModalOpen)
  const setProjectModalOpen = usePromptStudioStore((s) => s.setProjectModalOpen)
  const editingProjectId = usePromptStudioStore((s) => s.editingProjectId)
  const setEditingProjectId = usePromptStudioStore((s) => s.setEditingProjectId)

  // Fetch projects
  const { data: projectsResponse, status: projectsStatus } = useQuery({
    queryKey: ["prompt-studio", "projects", { include_deleted: showArchived }],
    queryFn: () => listProjects({ per_page: 100, include_deleted: showArchived })
  })

  const projects: Project[] = (projectsResponse as any)?.data?.data ?? []

  // Archive mutation
  const archiveMutation = useMutation({
    mutationFn: (projectId: number) => archiveProject(projectId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompt-studio", "projects"] })
      notification.success({
        message: t("managePrompts.studio.projects.archived", {
          defaultValue: "Project archived"
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

  // Unarchive mutation
  const unarchiveMutation = useMutation({
    mutationFn: (projectId: number) => unarchiveProject(projectId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompt-studio", "projects"] })
      notification.success({
        message: t("managePrompts.studio.projects.unarchived", {
          defaultValue: "Project restored"
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

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: ({ projectId, permanent }: { projectId: number; permanent?: boolean }) =>
      deleteProject(projectId, permanent),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["prompt-studio", "projects"] })
      if (selectedProjectId === variables.projectId) {
        setSelectedProjectId(null)
      }
      notification.success({
        message: t("managePrompts.studio.projects.deleted", {
          defaultValue: "Project deleted"
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

  // Filter projects
  const filteredProjects = useMemo(() => {
    let items = projects
    if (!showArchived) {
      items = items.filter((p) => p.status !== "archived")
    }
    if (searchText.trim()) {
      const q = searchText.toLowerCase()
      items = items.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.description?.toLowerCase().includes(q)
      )
    }
    return items.sort(
      (a, b) =>
        new Date(b.updated_at || b.created_at || 0).getTime() -
        new Date(a.updated_at || a.created_at || 0).getTime()
    )
  }, [projects, showArchived, searchText])

  const handleSelectProject = (project: Project) => {
    setSelectedProjectId(project.id)
    setActiveSubTab("prompts")
  }

  const handleOpenCreate = () => {
    setEditingProjectId(null)
    setProjectModalOpen(true)
  }

  const handleOpenEdit = (project: Project) => {
    setEditingProjectId(project.id)
    setProjectModalOpen(true)
  }

  const handleArchive = async (project: Project) => {
    const ok = await confirmDanger({
      title: t("managePrompts.studio.projects.archiveConfirmTitle", {
        defaultValue: "Archive project?"
      }),
      content: t("managePrompts.studio.projects.archiveConfirmContent", {
        defaultValue:
          "This will archive the project. You can restore it later from the archived projects view."
      }),
      okText: t("managePrompts.studio.projects.archiveBtn", {
        defaultValue: "Archive"
      }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" })
    })
    if (ok) {
      archiveMutation.mutate(project.id)
    }
  }

  const handleDelete = async (project: Project, permanent: boolean) => {
    const ok = await confirmDanger({
      title: permanent
        ? t("managePrompts.studio.projects.deleteConfirmTitle", {
            defaultValue: "Permanently delete project?"
          })
        : t("managePrompts.studio.projects.softDeleteConfirmTitle", {
            defaultValue: "Delete project?"
          }),
      content: permanent
        ? t("managePrompts.studio.projects.deleteConfirmContent", {
            defaultValue:
              "This will permanently delete the project and all its prompts, test cases, evaluations, and optimizations. This cannot be undone."
          })
        : t("managePrompts.studio.projects.softDeleteConfirmContent", {
            defaultValue:
              "This will delete the project. You may be able to recover it depending on server settings."
          }),
      okText: t("common:delete", { defaultValue: "Delete" }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" })
    })
    if (ok) {
      deleteMutation.mutate({ projectId: project.id, permanent })
    }
  }

  const getProjectActions = (project: Project): MenuProps["items"] => {
    const isArchived = project.status === "archived"
    return [
      {
        key: "edit",
        icon: <Pen className="size-4" />,
        label: t("common:edit", { defaultValue: "Edit" }),
        onClick: () => handleOpenEdit(project)
      },
      { type: "divider" },
      isArchived
        ? {
            key: "unarchive",
            icon: <ArchiveRestore className="size-4" />,
            label: t("managePrompts.studio.projects.unarchive", {
              defaultValue: "Restore"
            }),
            onClick: () => unarchiveMutation.mutate(project.id)
          }
        : {
            key: "archive",
            icon: <Archive className="size-4" />,
            label: t("managePrompts.studio.projects.archive", {
              defaultValue: "Archive"
            }),
            onClick: () => handleArchive(project)
          },
      { type: "divider" },
      {
        key: "delete",
        icon: <Trash2 className="size-4" />,
        label: t("common:delete", { defaultValue: "Delete" }),
        danger: true,
        onClick: () => handleDelete(project, false)
      }
    ]
  }

  const projectFormModal = (
    <ProjectFormModal
      open={isProjectModalOpen}
      projectId={editingProjectId}
      onClose={() => {
        setProjectModalOpen(false)
        setEditingProjectId(null)
      }}
    />
  )

  if (projectsStatus === "pending") {
    return <Skeleton paragraph={{ rows: 8 }} />
  }

  if (projectsStatus === "success" && projects.length === 0) {
    return (
      <>
        <FeatureEmptyState
          title={t("managePrompts.studio.projects.emptyTitle", {
            defaultValue: "No projects yet"
          })}
          description={t("managePrompts.studio.projects.emptyDescription", {
            defaultValue:
              "Create a project to organize your prompts, test cases, and evaluations."
          })}
          examples={[
            t("managePrompts.studio.projects.emptyExample1", {
              defaultValue:
                "Projects help you group related prompts for different use cases or teams."
            }),
            t("managePrompts.studio.projects.emptyExample2", {
              defaultValue:
                "Each project has its own test cases and evaluation history."
            })
          ]}
          primaryActionLabel={t("managePrompts.studio.projects.createBtn", {
            defaultValue: "Create Project"
          })}
          onPrimaryAction={handleOpenCreate}
        />
        {projectFormModal}
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
            {t("managePrompts.studio.projects.createBtn", {
              defaultValue: "Create Project"
            })}
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Input
            placeholder={t("managePrompts.studio.projects.searchPlaceholder", {
              defaultValue: "Search projects..."
            })}
            prefix={<Search className="size-4 text-text-muted" />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            allowClear
            style={{ width: 220 }}
          />
          <label className="flex items-center gap-2 text-sm text-text-muted cursor-pointer">
            <input
              type="checkbox"
              checked={showArchived}
              onChange={(e) => setShowArchived(e.target.checked)}
              className="rounded border-border"
            />
            {t("managePrompts.studio.projects.showArchived", {
              defaultValue: "Show archived"
            })}
          </label>
        </div>
      </div>

      {/* Projects table */}
      <Table<Project>
        dataSource={filteredProjects}
        rowKey="id"
        size="middle"
        pagination={{
          pageSize: 10,
          showSizeChanger: true,
          showTotal: (total) =>
            t("managePrompts.studio.projects.totalCount", {
              defaultValue: "{{count}} projects",
              count: total
            })
        }}
        rowClassName={(record) =>
          record.id === selectedProjectId
            ? "bg-primary/10"
            : "cursor-pointer hover:bg-surface2"
        }
        onRow={(record) => ({
          onClick: () => handleSelectProject(record),
          onDoubleClick: () => handleSelectProject(record)
        })}
        columns={[
          {
            title: "",
            width: 40,
            render: (_, record) => (
              <FolderKanban
                className={`size-5 ${
                  record.status === "archived"
                    ? "text-text-muted"
                    : "text-primary"
                }`}
              />
            )
          },
          {
            title: t("managePrompts.studio.projects.columns.name", {
              defaultValue: "Name"
            }),
            dataIndex: "name",
            key: "name",
            render: (name: string, record) => (
              <div className="flex flex-col">
                <span className="font-medium">{name}</span>
                {record.description && (
                  <span className="text-xs text-text-muted line-clamp-1">
                    {record.description}
                  </span>
                )}
              </div>
            )
          },
          {
            title: t("managePrompts.studio.projects.columns.stats", {
              defaultValue: "Contents"
            }),
            key: "stats",
            width: 200,
            render: (_, record) => (
              <div className="flex items-center gap-3 text-sm text-text-muted">
                <Tooltip
                  title={t("managePrompts.studio.projects.promptCount", {
                    defaultValue: "Prompts"
                  })}
                >
                  <span className="flex items-center gap-1">
                    <FileText className="size-4" />
                    {record.prompt_count ?? 0}
                  </span>
                </Tooltip>
                <Tooltip
                  title={t("managePrompts.studio.projects.testCaseCount", {
                    defaultValue: "Test cases"
                  })}
                >
                  <span className="flex items-center gap-1">
                    <TestTube className="size-4" />
                    {record.test_case_count ?? 0}
                  </span>
                </Tooltip>
              </div>
            )
          },
          {
            title: t("managePrompts.studio.projects.columns.status", {
              defaultValue: "Status"
            }),
            key: "status",
            width: 100,
            render: (_, record) =>
              record.status === "archived" ? (
                <Tag color="default">
                  {t("managePrompts.studio.projects.statusArchived", {
                    defaultValue: "Archived"
                  })}
                </Tag>
              ) : (
                <Tag color="green">
                  {t("managePrompts.studio.projects.statusActive", {
                    defaultValue: "Active"
                  })}
                </Tag>
              )
          },
          {
            title: t("managePrompts.studio.projects.columns.updated", {
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
                menu={{
                  items: getProjectActions(record),
                  onClick: ({ domEvent }) => domEvent.stopPropagation()
                }}
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

      {projectFormModal}
    </div>
  )
}
