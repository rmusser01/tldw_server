import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Button,
  Form,
  Input,
  Table,
  Tag,
  Tooltip,
  Dropdown,
  Upload,
  Pagination,
  Modal,
  Switch
} from "antd"
import type { ColumnsType } from "antd/es/table"
import React from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  Plus,
  Trash2,
  Pen,
  Download,
  Upload as UploadIcon,
  Play,
  FileDown,
  FileText,
  Database,
  Copy
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { SkillDrawer } from "./SkillDrawer"
import { SkillPreview } from "./SkillPreview"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"
import type {
  SkillSummary,
  SkillResponse,
  SkillsListResponse
} from "@/types/skill"

const DEFAULT_PAGE_SIZE = 10
const SKILLS_SEARCH_DEBOUNCE_MS = 300
const SKILL_NAME_REGEX = /^[a-z][a-z0-9-]{0,63}$/

interface ImportTextFormValues {
  name?: string
  content: string
  overwrite?: boolean
}

interface SkillsSuccessAction {
  title: string
  description: string
  skillName?: string
  testLabel?: string
  viewLabel?: string
}

interface SeedSkillsResult {
  count?: number
  seeded?: unknown
}

const getResponseSkillName = (result: unknown): string | undefined => {
  if (!result || typeof result !== "object") return undefined
  const name = (result as { name?: unknown }).name
  const trimmedName = typeof name === "string" ? name.trim() : ""
  return SKILL_NAME_REGEX.test(trimmedName) ? trimmedName : undefined
}

const getSeededSkillNames = (result: SeedSkillsResult | undefined): string[] => {
  if (!Array.isArray(result?.seeded)) return []
  return result.seeded
    .map((name) => (typeof name === "string" ? name.trim() : ""))
    .filter((name): name is string => SKILL_NAME_REGEX.test(name))
}

const buildSkillInvocation = (skillName: string) => `/skill ${skillName}`

export const SkillsManager: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const queryClient = useQueryClient()
  const notification = useAntdNotification()

  const [page, setPage] = React.useState(1)
  const [pageSize, setPageSize] = React.useState(DEFAULT_PAGE_SIZE)
  const [search, setSearch] = React.useState("")
  const [debouncedSearch, setDebouncedSearch] = React.useState("")
  const [drawerOpen, setDrawerOpen] = React.useState(false)
  const [importTextOpen, setImportTextOpen] = React.useState(false)
  const [editingSkill, setEditingSkill] = React.useState<SkillResponse | null>(null)
  const [previewSkill, setPreviewSkill] = React.useState<string | null>(null)
  const [successAction, setSuccessAction] =
    React.useState<SkillsSuccessAction | null>(null)
  const [importTextForm] = Form.useForm<ImportTextFormValues>()

  const offset = (page - 1) * pageSize
  const searchQuery = debouncedSearch.trim()

  React.useEffect(() => {
    if (search === debouncedSearch) return

    const timer = window.setTimeout(() => {
      setDebouncedSearch(search)
      setPage(1)
    }, SKILLS_SEARCH_DEBOUNCE_MS)

    return () => window.clearTimeout(timer)
  }, [debouncedSearch, search])

  const {
    data,
    isLoading,
    isError,
    error,
    refetch
  } = useQuery<SkillsListResponse>({
    queryKey: ["skills", page, pageSize, searchQuery],
    queryFn: ({ signal }) =>
      tldwClient.listSkills({
        ...(searchQuery ? { q: searchQuery } : {}),
        limit: pageSize,
        offset,
        abortSignal: signal
      })
  })

  const hasLoadedSkills = data != null && !isError
  const totalSkills = data?.total ?? 0
  const hasSearch = searchQuery.length > 0
  const isLibraryEmpty =
    hasLoadedSkills && !isLoading && totalSkills === 0 && !hasSearch
  const skillCountLabel = isError
    ? t("option:skills.countUnavailable", {
        defaultValue: "Count unavailable"
      })
    : t("option:skills.countSummary", {
        defaultValue: `${totalSkills} ${totalSkills === 1 ? "skill" : "skills"}`,
        count: totalSkills
      })
  const listErrorDescription =
    error instanceof Error && error.message
      ? error.message
      : t("option:skills.loadListErrorDescription", {
          defaultValue: "Check your server connection and try again."
        })

  React.useEffect(() => {
    if (!hasLoadedSkills) return

    const lastPage = Math.max(1, Math.ceil(totalSkills / pageSize))
    if (page > lastPage) {
      setPage(lastPage)
    }
  }, [hasLoadedSkills, page, pageSize, totalSkills])

  const deleteMutation = useMutation({
    mutationFn: (name: string) => tldwClient.deleteSkill(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      setSuccessAction(null)
      notification.success({
        message: t("option:skills.deleteSuccess", { defaultValue: "Skill deleted" })
      })
    },
    onError: (err: any) => {
      notification.error({
        message: t("option:skills.deleteError", { defaultValue: "Failed to delete skill" }),
        description: err?.message
      })
    }
  })

  const importTextMutation = useMutation({
    mutationFn: (payload: {
      name?: string
      content: string
      overwrite?: boolean
    }) => tldwClient.importSkill(payload),
    onSuccess: (result, variables) => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      setImportTextOpen(false)
      importTextForm.resetFields()
      const skillName = getResponseSkillName(result) ?? variables.name
      if (skillName) {
        setSuccessAction({
          title: t("option:skills.importSuccess", { defaultValue: "Skill imported" }),
          description: t("option:skills.importSuccessActionDesc", {
            defaultValue:
              "Next, test it here or open the skill to confirm the imported details."
          }),
          skillName,
          viewLabel: t("option:skills.viewSkill", { defaultValue: "View skill" })
        })
      }
      notification.success({
        message: t("option:skills.importSuccess", { defaultValue: "Skill imported" })
      })
    },
    onError: (err: any) => {
      notification.error({
        message: t("option:skills.importError", { defaultValue: "Failed to import skill" }),
        description: err?.message
      })
    }
  })

  const seedBuiltinsMutation = useMutation({
    mutationFn: (overwrite: boolean = false) => tldwClient.seedSkills({ overwrite }),
    onSuccess: (result: SeedSkillsResult | undefined) => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      const count = Number(result?.count ?? 0)
      const seededSkillNames = getSeededSkillNames(result)
      const suggestedSkillName =
        seededSkillNames.includes("summarize") ? "summarize" : seededSkillNames[0]
      if (suggestedSkillName) {
        setSuccessAction({
          title: t("option:skills.seedSuccess", {
            defaultValue: "Built-in skills seeded"
          }),
          description: t("option:skills.seedSuccessActionDesc", {
            defaultValue:
              "Try a built-in skill now, or copy the chat invocation for later."
          }),
          skillName: suggestedSkillName,
          testLabel: t("option:skills.testSpecificSkill", {
            defaultValue: `Test ${suggestedSkillName}`,
            skillName: suggestedSkillName
          }),
          viewLabel: t("option:skills.viewSkill", { defaultValue: "View skill" })
        })
      }
      notification.success({
        message: t("option:skills.seedSuccess", { defaultValue: "Built-in skills seeded" }),
        description: t("option:skills.seedSuccessDesc", {
          defaultValue: `${count} built-in skill(s) seeded.`,
          count
        })
      })
    },
    onError: (err: any) => {
      notification.error({
        message: t("option:skills.seedError", { defaultValue: "Failed to seed built-in skills" }),
        description: err?.message
      })
    }
  })

  const handleNew = () => {
    setSuccessAction(null)
    setEditingSkill(null)
    setDrawerOpen(true)
  }

  const handleEdit = async (name: string) => {
    try {
      const skill = await tldwClient.getSkill(name)
      setEditingSkill(skill)
      setDrawerOpen(true)
    } catch (err: any) {
      notification.error({
        message: t("option:skills.loadError", { defaultValue: "Failed to load skill" }),
        description: err?.message
      })
    }
  }

  const handleDelete = (name: string) => {
    Modal.confirm({
      title: t("option:skills.deleteConfirmTitle", {
        defaultValue: "Delete skill?"
      }),
      content: t("option:skills.deleteConfirmContent", {
        defaultValue: `Are you sure you want to delete "${name}"? This cannot be undone.`,
        name
      }),
      okText: t("common:delete", { defaultValue: "Delete" }),
      okButtonProps: { danger: true },
      cancelText: t("common:cancel", { defaultValue: "Cancel" }),
      onOk: () => deleteMutation.mutateAsync(name)
    })
  }

  const handleExport = async (name: string) => {
    try {
      const blob = await tldwClient.exportSkill(name)
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `${name}.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err: any) {
      notification.error({
        message: t("option:skills.exportError", { defaultValue: "Failed to export skill" }),
        description: err?.message
      })
    }
  }

  const handleImportFile = async (file: File) => {
    try {
      const result = await tldwClient.importSkillFile(file)
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      const skillName = getResponseSkillName(result)
      if (skillName) {
        setSuccessAction({
          title: t("option:skills.importSuccess", { defaultValue: "Skill imported" }),
          description: t("option:skills.importSuccessActionDesc", {
            defaultValue:
              "Next, test it here or open the skill to confirm the imported details."
          }),
          skillName,
          viewLabel: t("option:skills.viewSkill", { defaultValue: "View skill" })
        })
      }
      notification.success({
        message: t("option:skills.importSuccess", { defaultValue: "Skill imported" })
      })
    } catch (err: any) {
      notification.error({
        message: t("option:skills.importError", { defaultValue: "Failed to import skill" }),
        description: err?.message
      })
    }
    return false // prevent antd Upload default behavior
  }

  const openImportTextModal = () => {
    setSuccessAction(null)
    importTextForm.resetFields()
    importTextForm.setFieldsValue({ overwrite: false, content: "" })
    setImportTextOpen(true)
  }

  const handleImportTextSubmit = async () => {
    try {
      const values = await importTextForm.validateFields()
      const payload: {
        name?: string
        content: string
        overwrite?: boolean
      } = {
        content: values.content,
        overwrite: Boolean(values.overwrite)
      }
      const trimmedName = values.name?.trim()
      if (trimmedName) {
        payload.name = trimmedName
      }
      await importTextMutation.mutateAsync(payload)
    } catch {
      // validation errors handled by antd
    }
  }

  const handleDrawerClose = () => {
    setDrawerOpen(false)
    setEditingSkill(null)
  }

  const handleDrawerSaved = (savedSkillName?: string) => {
    const wasCreating = !editingSkill
    queryClient.invalidateQueries({ queryKey: ["skills"] })
    handleDrawerClose()
    if (wasCreating && savedSkillName) {
      setSuccessAction({
        title: t("option:skills.createSuccess", { defaultValue: "Skill created" }),
        description: t("option:skills.createSuccessActionDesc", {
          defaultValue:
            "Next, test it here or copy the chat invocation for a conversation."
        }),
        skillName: savedSkillName,
        viewLabel: t("option:skills.viewSkill", { defaultValue: "View skill" })
      })
    }
  }

  const handleCopyInvocation = async (skillName: string) => {
    try {
      const writeText = navigator.clipboard?.writeText
      if (!writeText) {
        throw new Error(
          t("option:skills.copyInvocationUnavailable", {
            defaultValue: "Clipboard is not available in this browser context."
          })
        )
      }
      await writeText.call(navigator.clipboard, buildSkillInvocation(skillName))
      notification.success({
        message: t("option:skills.copyInvocationSuccess", {
          defaultValue: "Skill invocation copied"
        })
      })
    } catch (err: any) {
      notification.error({
        message: t("option:skills.copyInvocationError", {
          defaultValue: "Failed to copy skill invocation"
        }),
        description: err?.message
      })
    }
  }

  const columns: ColumnsType<SkillSummary> = [
    {
      title: t("option:skills.colName", { defaultValue: "Name" }),
      dataIndex: "name",
      key: "name",
      render: (name: string) => (
        <span className="font-mono text-sm">{name}</span>
      )
    },
    {
      title: t("option:skills.colDescription", { defaultValue: "Description" }),
      dataIndex: "description",
      key: "description",
      ellipsis: true,
      render: (desc: string | null) => desc || "-"
    },
    {
      title: t("option:skills.colContext", { defaultValue: "Mode" }),
      dataIndex: "context",
      key: "context",
      width: 100,
      render: (ctx: string) => (
        <Tag color={ctx === "fork" ? "blue" : "green"}>
          {ctx}
        </Tag>
      )
    },
    {
      title: t("option:skills.colActions", { defaultValue: "Actions" }),
      key: "actions",
      width: 180,
      render: (_: unknown, record: SkillSummary) => (
        <div className="flex items-center gap-1">
          <Tooltip title={t("option:skills.preview", { defaultValue: "Preview" })}>
            <Button
              type="text"
              size="small"
              icon={<Play size={14} />}
              onClick={() => setPreviewSkill(record.name)}
            />
          </Tooltip>
          <Tooltip title={t("common:edit", { defaultValue: "Edit" })}>
            <Button
              type="text"
              size="small"
              icon={<Pen size={14} />}
              onClick={() => handleEdit(record.name)}
            />
          </Tooltip>
          <Tooltip title={t("option:skills.export", { defaultValue: "Export" })}>
            <Button
              type="text"
              size="small"
              icon={<Download size={14} />}
              onClick={() => handleExport(record.name)}
            />
          </Tooltip>
          <Tooltip title={t("common:delete", { defaultValue: "Delete" })}>
            <Button
              type="text"
              size="small"
              danger
              icon={<Trash2 size={14} />}
              onClick={() => handleDelete(record.name)}
            />
          </Tooltip>
        </div>
      )
    }
  ]

  const importMenuItems = [
    {
      key: "text",
      label: (
        <span className="flex items-center gap-2">
          <FileText size={14} />
          {t("option:skills.importText", { defaultValue: "Import Text" })}
        </span>
      ),
      onClick: openImportTextModal
    },
    {
      key: "file",
      label: (
        <Upload
          accept=".md,.zip"
          showUploadList={false}
          beforeUpload={handleImportFile}
        >
          <span className="flex items-center gap-2">
            <FileDown size={14} />
            {t("option:skills.importFile", { defaultValue: "Import File (.md/.zip)" })}
          </span>
        </Upload>
      )
    }
  ]

  const seedMenuItems = [
    {
      key: "seed-missing-only",
      label: t("option:skills.seedBuiltinsMissingOnly", {
        defaultValue: "Seed Missing Only"
      }),
      onClick: () => seedBuiltinsMutation.mutate(false)
    },
    {
      key: "seed-overwrite-existing",
      label: t("option:skills.seedBuiltinsOverwrite", {
        defaultValue: "Seed and Overwrite Existing"
      }),
      onClick: () => seedBuiltinsMutation.mutate(true)
    }
  ]

  const beginnerEmptyState = (
    <div
      className="mx-auto flex max-w-xl flex-col items-center gap-3 py-8 text-center"
      data-testid="skills-empty-state"
    >
      <div className="space-y-1">
        <h2 className="text-base font-semibold text-text">
          {t("option:skills.emptyTitle", {
            defaultValue: "Start with a reusable skill"
          })}
        </h2>
        <p className="m-0 text-sm text-text-muted">
          {t("option:skills.emptyDescription", {
            defaultValue:
              "Skills are reusable instructions that can be tested here and used from chat."
          })}
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Button
          type="primary"
          icon={<Database size={14} />}
          loading={seedBuiltinsMutation.isPending}
          onClick={() => seedBuiltinsMutation.mutate(false)}
        >
          {t("option:skills.emptySeedBuiltins", {
            defaultValue: "Seed built-ins"
          })}
        </Button>
        <Button icon={<Plus size={14} />} onClick={handleNew}>
          {t("option:skills.emptyCreateFromTemplate", {
            defaultValue: "Create from template"
          })}
        </Button>
        <Button icon={<UploadIcon size={14} />} onClick={openImportTextModal}>
          {t("option:skills.emptyImportText", {
            defaultValue: "Import from text"
          })}
        </Button>
      </div>
    </div>
  )

  let tableEmptyText: React.ReactNode = beginnerEmptyState
  if (!isLibraryEmpty) {
    let emptyText = t("option:skills.emptyTable", {
      defaultValue: "No skills yet."
    })
    if (isError) {
      emptyText = t("option:skills.emptyTableError", {
        defaultValue: "Unable to load skills."
      })
    } else if (hasSearch) {
      emptyText = t("option:skills.noMatches", {
        defaultValue: "No skills match this search."
      })
    } else if (totalSkills > 0) {
      emptyText = t("option:skills.emptyCurrentPage", {
        defaultValue: "No skills on this page."
      })
    }

    tableEmptyText = emptyText
  }

  return (
    <div className="flex flex-col gap-4">
      <section
        aria-labelledby="skills-manager-title"
        className="flex flex-col gap-1"
      >
        <div className="flex flex-col gap-1 md:flex-row md:items-start md:justify-between">
          <div>
            <h1
              id="skills-manager-title"
              className="m-0 text-xl font-semibold text-text"
            >
              {t("option:skills.title", { defaultValue: "Skills" })}
            </h1>
            <p className="m-0 max-w-2xl text-sm text-text-muted">
              {t("option:skills.description", {
                defaultValue:
                  "Discover, test, create, import, and manage reusable instructions."
              })}
            </p>
          </div>
          <p className="m-0 text-sm font-medium text-text-muted">
            {skillCountLabel}
          </p>
        </div>
      </section>

      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <Input.Search
          placeholder={t("option:skills.searchPlaceholder", {
            defaultValue: "Search skills..."
          })}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          allowClear
          style={{ maxWidth: 360 }}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Dropdown menu={{ items: importMenuItems }} trigger={["click"]}>
            <Button icon={<UploadIcon size={14} />}>
              {t("option:skills.import", { defaultValue: "Import" })}
            </Button>
          </Dropdown>
          <Dropdown menu={{ items: seedMenuItems }} trigger={["click"]}>
            <Button
              icon={<Database size={14} />}
              loading={seedBuiltinsMutation.isPending}
            >
              {t("option:skills.seedBuiltins", { defaultValue: "Seed Built-ins" })}
            </Button>
          </Dropdown>
          <Button type="primary" icon={<Plus size={14} />} onClick={handleNew}>
            {t("option:skills.newSkill", { defaultValue: "New Skill" })}
          </Button>
        </div>
      </div>

      {isError && (
        <DesignSystemAlert
          variant="error"
          title={t("option:skills.loadListError", {
            defaultValue: "Failed to load skills"
          })}
          action={{
            label: t("common:tryAgain", { defaultValue: "Try again" }),
            onClick: () => void refetch()
          }}
        >
          {listErrorDescription}
        </DesignSystemAlert>
      )}

      {successAction && (
        <DesignSystemAlert
          data-testid="skills-success-actions"
          variant="success"
          title={successAction.title}
          dismissible
          onDismiss={() => setSuccessAction(null)}
        >
          <p className="m-0">{successAction.description}</p>
          {(() => {
            const skillName = successAction.skillName
            if (!skillName) return null
            const invocation = buildSkillInvocation(skillName)
            return (
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  size="small"
                  icon={<Play size={14} />}
                  onClick={() => setPreviewSkill(skillName)}
                >
                  {successAction.testLabel ??
                    t("option:skills.testRun", { defaultValue: "Test run" })}
                </Button>
                <Button
                  size="small"
                  icon={<Pen size={14} />}
                  onClick={() => void handleEdit(skillName)}
                >
                  {successAction.viewLabel ??
                    t("option:skills.viewSkill", { defaultValue: "View skill" })}
                </Button>
                <Button
                  size="small"
                  icon={<Copy size={14} />}
                  onClick={() => void handleCopyInvocation(skillName)}
                >
                  {t("option:skills.copyInvocation", {
                    defaultValue: `Copy ${invocation}`,
                    invocation
                  })}
                </Button>
              </div>
            )
          })()}
        </DesignSystemAlert>
      )}

      <Table
        dataSource={data?.skills ?? []}
        columns={columns}
        rowKey="name"
        loading={isLoading}
        pagination={false}
        size="middle"
        locale={{
          emptyText: tableEmptyText
        }}
      />

      {totalSkills > pageSize && (
        <div className="flex justify-end">
          <Pagination
            current={page}
            pageSize={pageSize}
            total={totalSkills}
            onChange={(p, ps) => {
              setPage(p)
              setPageSize(ps)
            }}
            showSizeChanger
            pageSizeOptions={["10", "20", "50"]}
          />
        </div>
      )}

      <Modal
        title={t("option:skills.importTextTitle", {
          defaultValue: "Import Skill from Text"
        })}
        open={importTextOpen}
        onCancel={() => setImportTextOpen(false)}
        onOk={handleImportTextSubmit}
        okText={t("option:skills.import", { defaultValue: "Import" })}
        okButtonProps={{ loading: importTextMutation.isPending }}
        destroyOnHidden
      >
        <Form
          form={importTextForm}
          layout="vertical"
          initialValues={{ overwrite: false }}
          autoComplete="off"
        >
          <Form.Item
            name="name"
            label={t("option:skills.nameLabel", { defaultValue: "Name" })}
            rules={[
              {
                validator: async (_, value: string | undefined) => {
                  const trimmed = (value ?? "").trim()
                  if (!trimmed) return
                  if (!SKILL_NAME_REGEX.test(trimmed)) {
                    throw new Error(
                      t("option:skills.nameInvalid", {
                        defaultValue:
                          "Must start with a letter, use only lowercase letters, numbers, and hyphens (max 64 chars)"
                      })
                    )
                  }
                }
              }
            ]}
            extra={t("option:skills.importNameOptional", {
              defaultValue: "Optional. If omitted, name from frontmatter will be used."
            })}
          >
            <Input placeholder="my-skill-name" maxLength={64} className="font-mono" />
          </Form.Item>

          <Form.Item
            name="content"
            label={t("option:skills.contentLabel", {
              defaultValue: "SKILL.md Content"
            })}
            rules={[
              {
                required: true,
                whitespace: true,
                message: t("option:skills.contentRequired", {
                  defaultValue: "Content is required"
                })
              }
            ]}
          >
            <Input.TextArea rows={14} className="font-mono text-xs" />
          </Form.Item>

          <Form.Item
            name="overwrite"
            valuePropName="checked"
            label={t("option:skills.importOverwrite", {
              defaultValue: "Overwrite existing skill"
            })}
          >
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      <SkillDrawer
        open={drawerOpen}
        skill={editingSkill}
        onClose={handleDrawerClose}
        onSaved={handleDrawerSaved}
      />

      <SkillPreview
        skillName={previewSkill}
        onClose={() => setPreviewSkill(null)}
      />
    </div>
  )
}
