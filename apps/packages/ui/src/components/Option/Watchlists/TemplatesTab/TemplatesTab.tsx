import React, { useCallback, useEffect, useState } from "react"
import {
  Button,
  Empty,
  Modal,
  Space,
  Table,
  Tooltip,
  message
} from "antd"
import type { ColumnsType } from "antd/es/table"
import { Edit, Plus, RefreshCw, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useWatchlistsStore } from "@/store/watchlists"
import {
  fetchWatchlistJobs,
  fetchWatchlistTemplates,
  deleteWatchlistTemplate
} from "@/services/watchlists"
import type { WatchlistJob, WatchlistTemplate } from "@/types/watchlists"
import { formatRelativeTime } from "@/utils/dateFormatters"
import { findActiveTemplateUsage } from "./template-usage"
import { TemplateEditor } from "./TemplateEditor"
import { useWatchlistsViewport } from "../shared/useWatchlistsViewport"

const TEMPLATE_USAGE_CHECK_PAGE_SIZE = 200
const TEMPLATE_USAGE_CHECK_MAX_PAGES = 10

export const TemplatesTab: React.FC = () => {
  const { t } = useTranslation(["watchlists", "common"])
  const { isConstrained } = useWatchlistsViewport()

  // Store state
  const templates = useWatchlistsStore((s) => s.templates)
  const templatesLoading = useWatchlistsStore((s) => s.templatesLoading)
  const selectedWatchlistId = useWatchlistsStore((s) => s.selectedWatchlistId)

  // Store actions
  const setTemplates = useWatchlistsStore((s) => s.setTemplates)
  const setTemplatesLoading = useWatchlistsStore((s) => s.setTemplatesLoading)

  // Local state for editor
  const [editorOpen, setEditorOpen] = useState(false)
  const [editingTemplate, setEditingTemplate] = useState<WatchlistTemplate | null>(null)
  const [checkingTemplateDeleteName, setCheckingTemplateDeleteName] = useState<string | null>(null)

  // Fetch templates
  const loadTemplates = useCallback(async () => {
    setTemplatesLoading(true)
    try {
      const result = await fetchWatchlistTemplates()
      setTemplates(Array.isArray(result.items) ? result.items : [])
    } catch (err) {
      console.error("Failed to fetch templates:", err)
      message.error(t("watchlists:templates.fetchError", "Failed to load templates"))
      setTemplates([])
    } finally {
      setTemplatesLoading(false)
    }
  }, [setTemplates, setTemplatesLoading, t])

  // Ensure templates is always an array for rendering
  const safeTemplates = Array.isArray(templates) ? templates : []

  // Initial load
  useEffect(() => {
    loadTemplates()
  }, [loadTemplates])

  // Handle create
  const handleCreate = () => {
    setEditingTemplate(null)
    setEditorOpen(true)
  }

  // Handle edit
  const handleEdit = (template: WatchlistTemplate) => {
    setEditingTemplate(template)
    setEditorOpen(true)
  }

  // Handle delete
  const handleDelete = async (template: WatchlistTemplate) => {
    try {
      await deleteWatchlistTemplate(template.name)
      message.success(t("watchlists:templates.deleted", "Template deleted"))
      loadTemplates()
    } catch (err) {
      console.error("Failed to delete template:", err)
      message.error(t("watchlists:templates.deleteError", "Failed to delete template"))
    }
  }

  const loadJobsForTemplateUsageCheck = useCallback(async (): Promise<WatchlistJob[]> => {
    const allJobs: WatchlistJob[] = []
    let page = 1

    while (page <= TEMPLATE_USAGE_CHECK_MAX_PAGES) {
      const response = await fetchWatchlistJobs({
        watchlist_id: selectedWatchlistId ?? undefined,
        page,
        size: TEMPLATE_USAGE_CHECK_PAGE_SIZE
      })
      const pageItems = Array.isArray(response.items) ? response.items : []
      allJobs.push(...pageItems)
      if (
        pageItems.length < TEMPLATE_USAGE_CHECK_PAGE_SIZE ||
        allJobs.length >= Number(response.total || 0)
      ) {
        break
      }
      page += 1
    }

    return allJobs
  }, [selectedWatchlistId])

  const requestDeleteConfirmation = useCallback(async (template: WatchlistTemplate) => {
    setCheckingTemplateDeleteName(template.name)
    let activeUsage: Array<{ id: number; name: string }> = []
    try {
      const jobs = await loadJobsForTemplateUsageCheck()
      activeUsage = findActiveTemplateUsage(jobs, template.name)
    } catch (err) {
      console.error("Failed to check template usage:", err)
      message.warning(
        t(
          "watchlists:templates.deleteUsageLookupError",
          "Could not verify active monitor usage before deletion."
        )
      )
    } finally {
      setCheckingTemplateDeleteName(null)
    }

    const usageCount = activeUsage.length
    Modal.confirm({
      title: usageCount > 0
        ? t(
            "watchlists:templates.deleteConfirmInUseTitle",
            "Template is used by active monitors"
          )
        : t("watchlists:templates.deleteConfirm", "Delete this template?"),
      content: usageCount > 0 ? (
        <div className="space-y-2">
          <p>
            {t(
              "watchlists:templates.deleteConfirmInUseDescription",
              "This template is referenced by {{count}} active monitor{{plural}}. Deleting it may affect scheduled reports.",
              {
                count: usageCount,
                plural: usageCount === 1 ? "" : "s"
              }
            )}
          </p>
          <ul className="list-disc pl-5">
            {activeUsage.slice(0, 5).map((job) => (
              <li key={job.id}>{job.name}</li>
            ))}
          </ul>
          {usageCount > 5 && (
            <p className="text-xs text-text-muted">
              {t(
                "watchlists:templates.deleteConfirmInUseMore",
                "+{{count}} more active monitor{{plural}}",
                {
                  count: usageCount - 5,
                  plural: usageCount - 5 === 1 ? "" : "s"
                }
              )}
            </p>
          )}
        </div>
      ) : (
        t("watchlists:templates.deleteConfirmDescription", "This action cannot be undone.")
      ),
      okText: usageCount > 0
        ? t("watchlists:templates.deleteConfirmForce", "Delete anyway")
        : t("watchlists:templates.delete", "Delete"),
      cancelText: t("common:cancel", "Cancel"),
      okButtonProps: { danger: true },
      onOk: () => handleDelete(template)
    })
  }, [handleDelete, loadJobsForTemplateUsageCheck, t])

  // Handle editor close
  const handleEditorClose = (saved?: boolean) => {
    setEditorOpen(false)
    setEditingTemplate(null)
    if (saved) {
      loadTemplates()
    }
  }

  // Table columns
  const columns: ColumnsType<WatchlistTemplate> = [
    {
      title: t("watchlists:templates.columns.name", "Name"),
      dataIndex: "name",
      key: "name",
      ellipsis: true,
      render: (name: string) => (
        <span className="font-medium">{name}</span>
      )
    },
    {
      title: t("watchlists:templates.columns.description", "Description"),
      dataIndex: "description",
      key: "description",
      ellipsis: true,
      render: (desc: string | null) => (
        <span className="text-sm text-text-muted">
          {desc || "-"}
        </span>
      )
    },
    {
      title: t("watchlists:templates.columns.format", "Format"),
      dataIndex: "format",
      key: "format",
      width: 100,
      render: (format: string) => (
        <span className="text-sm uppercase">{format || "md"}</span>
      )
    },
    {
      title: t("watchlists:templates.columns.updated", "Updated"),
      dataIndex: "updated_at",
      key: "updated_at",
      width: 150,
      render: (date: string | null) =>
        date ? (
          <span className="text-sm text-text-muted">
            {formatRelativeTime(date, t)}
          </span>
        ) : (
          <span className="text-sm text-text-subtle">-</span>
        )
    },
    {
      title: t("watchlists:templates.columns.actions", "Actions"),
      key: "actions",
      width: 140,
      align: "center",
      render: (_, record) => (
        <Space size="small">
          <Tooltip title={t("watchlists:templates.edit", "Edit")}>
            <Button
              type="text"
              size="small"
              aria-label={t("watchlists:templates.edit", "Edit")}
              icon={<Edit className="h-4 w-4" />}
              onClick={() => handleEdit(record)}
            />
          </Tooltip>
          <Tooltip title={t("watchlists:templates.delete", "Delete")}>
            <Button
              type="text"
              size="small"
              danger
              aria-label={t("watchlists:templates.delete", "Delete")}
              icon={<Trash2 className="h-4 w-4" />}
              loading={checkingTemplateDeleteName === record.name}
              onClick={() => void requestDeleteConfirmation(record)}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const renderConstrainedTemplateList = () => (
    <div className="space-y-3" data-testid="watchlists-templates-constrained-list">
      {safeTemplates.map((template) => {
        const versionCount = Number(template.history_count || template.available_versions?.length || 0)
        const formatLabel = String(template.format || "md").toUpperCase()
        return (
          <article
            key={template.name}
            className="rounded-lg border border-border bg-surface p-3"
            data-testid={`watchlists-template-card-${template.name}`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0 space-y-1">
                <div className="font-medium text-text">{template.name}</div>
                <div className="text-sm text-text-muted">
                  {template.description || t("watchlists:templates.noDescription", "No description")}
                </div>
              </div>
              <span className="rounded border border-border px-2 py-0.5 text-xs font-medium text-text-muted">
                {formatLabel}
              </span>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:templates.columns.updated", "Updated")}
                </div>
                <span className="text-text-muted">
                  {template.updated_at ? formatRelativeTime(template.updated_at, t) : "-"}
                </span>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:templates.versionLabel", "Version")}
                </div>
                <span className="text-text-muted">
                  {template.version ? `v${template.version}` : t("watchlists:templates.latestVersion", "Latest")}
                </span>
              </div>
              <div>
                <div className="text-xs font-medium text-text-subtle">
                  {t("watchlists:templates.historyLabel", "History")}
                </div>
                <span className="text-text-muted">
                  {t("watchlists:templates.versionCount", "{{count}} version{{plural}}", {
                    count: versionCount,
                    plural: versionCount === 1 ? "" : "s"
                  })}
                </span>
              </div>
            </div>

            <div className="mt-3 flex flex-wrap justify-end gap-2">
              <Button
                type="text"
                size="small"
                aria-label={t("watchlists:templates.edit", "Edit")}
                icon={<Edit className="h-4 w-4" />}
                onClick={() => handleEdit(template)}
              />
              <Button
                type="text"
                size="small"
                danger
                aria-label={t("watchlists:templates.delete", "Delete")}
                icon={<Trash2 className="h-4 w-4" />}
                loading={checkingTemplateDeleteName === template.name}
                onClick={() => void requestDeleteConfirmation(template)}
              />
            </div>
          </article>
        )
      })}
    </div>
  )

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <Button
          type="primary"
          icon={<Plus className="h-4 w-4" />}
          onClick={handleCreate}
        >
          {t("watchlists:templates.create", "Create Template")}
        </Button>
        <Button
          icon={<RefreshCw className="h-4 w-4" />}
          onClick={loadTemplates}
          loading={templatesLoading}
        >
          {t("common:refresh", "Refresh")}
        </Button>
      </div>

      {/* Description */}
      <div className="text-sm text-text-muted">
        {t("watchlists:templates.description", "Templates for generating briefing outputs.")}
      </div>

      {/* Table */}
      {safeTemplates.length === 0 && !templatesLoading ? (
        <Empty
          description={t("watchlists:templates.empty", "No templates yet")}
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        >
          <Button type="primary" onClick={handleCreate}>
            {t("watchlists:templates.createFirst", "Create your first template")}
          </Button>
        </Empty>
      ) : (
        isConstrained ? (
          renderConstrainedTemplateList()
        ) : (
          <Table
            dataSource={safeTemplates}
            columns={columns}
            rowKey="name"
            aria-label={t("watchlists:templates.tableAria", "Templates table")}
            loading={templatesLoading}
            pagination={false}
            size="middle"
          />
        )
      )}

      {/* Template Editor Modal */}
      <TemplateEditor
        template={editingTemplate}
        open={editorOpen}
        onClose={handleEditorClose}
      />
    </div>
  )
}
