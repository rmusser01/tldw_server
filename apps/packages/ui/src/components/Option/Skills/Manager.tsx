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
  Database
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { SkillDrawer } from "./SkillDrawer"
import { SkillPreview } from "./SkillPreview"
import type {
  SkillSummary,
  SkillResponse,
  SkillsListResponse
} from "@/types/skill"

const DEFAULT_PAGE_SIZE = 10
const SKILL_NAME_REGEX = /^[a-z][a-z0-9-]{0,63}$/

interface ImportTextFormValues {
  name?: string
  content: string
  overwrite?: boolean
}

export const SkillsManager: React.FC = () => {
  const { t } = useTranslation(["option", "common"])
  const queryClient = useQueryClient()
  const notification = useAntdNotification()

  const [page, setPage] = React.useState(1)
  const [pageSize, setPageSize] = React.useState(DEFAULT_PAGE_SIZE)
  const [search, setSearch] = React.useState("")
  const [drawerOpen, setDrawerOpen] = React.useState(false)
  const [importTextOpen, setImportTextOpen] = React.useState(false)
  const [editingSkill, setEditingSkill] = React.useState<SkillResponse | null>(null)
  const [previewSkill, setPreviewSkill] = React.useState<string | null>(null)
  const [importTextForm] = Form.useForm<ImportTextFormValues>()

  const offset = (page - 1) * pageSize

  const { data, isLoading } = useQuery<SkillsListResponse>({
    queryKey: ["skills", page, pageSize],
    queryFn: () => tldwClient.listSkills({ limit: pageSize, offset })
  })

  const filteredSkills = React.useMemo(() => {
    if (!data?.skills) return []
    if (!search.trim()) return data.skills
    const q = search.toLowerCase()
    return data.skills.filter(
      (s) =>
        s.name.toLowerCase().includes(q) ||
        (s.description && s.description.toLowerCase().includes(q))
      )
  }, [data?.skills, search])

  const totalSkills = data?.total ?? 0
  const hasSearch = search.trim().length > 0
  const isLibraryEmpty = !isLoading && totalSkills === 0 && !hasSearch
  const skillCountLabel = t("option:skills.countSummary", {
    defaultValue: `${totalSkills} ${totalSkills === 1 ? "skill" : "skills"}`,
    count: totalSkills
  })

  const deleteMutation = useMutation({
    mutationFn: (name: string) => tldwClient.deleteSkill(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
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
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      setImportTextOpen(false)
      importTextForm.resetFields()
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
    onSuccess: (result: { count?: number } | undefined) => {
      queryClient.invalidateQueries({ queryKey: ["skills"] })
      const count = Number(result?.count ?? 0)
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
      await tldwClient.importSkillFile(file)
      queryClient.invalidateQueries({ queryKey: ["skills"] })
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

  const handleDrawerSaved = () => {
    queryClient.invalidateQueries({ queryKey: ["skills"] })
    handleDrawerClose()
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
          {t("option:skills.import", { defaultValue: "Import" })}
        </Button>
      </div>
    </div>
  )

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

      <Table
        dataSource={filteredSkills}
        columns={columns}
        rowKey="name"
        loading={isLoading}
        pagination={false}
        size="middle"
        locale={{
          emptyText: isLibraryEmpty
            ? beginnerEmptyState
            : t("option:skills.noMatches", {
                defaultValue: hasSearch
                  ? "No skills match this search."
                  : "No skills yet."
              })
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
