import React, { useCallback, useEffect, useMemo, useState } from "react"
import {
  Alert,
  Button,
  Empty,
  Form,
  Input,
  InputNumber,
  List,
  Modal,
  Popconfirm,
  Select,
  Space,
  Spin,
  Switch,
  Tag,
  Typography,
  message
} from "antd"
import { CalendarClock, Plus } from "lucide-react"
import { useTranslation } from "react-i18next"
import { useTldwApiClient } from "@/hooks/useTldwApiClient"
import { dedupeTags } from "@/utils/tag-identity"
import type {
  ReadingDigestSchedule,
  ReadingDigestSuggestionStatus,
  ReadingDigestSuggestionsConfig
} from "@/types/collections"

const { Text } = Typography

const ALLOWED_SUGGESTION_STATUSES: ReadingDigestSuggestionStatus[] = [
  "saved",
  "reading",
  "read",
  "archived"
]

interface DigestScheduleFormValues {
  name?: string
  cron: string
  timezone: string
  enabled: boolean
  require_online: boolean
  format: "md" | "html"
  template_name?: string
  suggestions_enabled: boolean
  suggestions_limit?: number | null
  suggestions_status?: ReadingDigestSuggestionStatus[]
  suggestions_exclude_tags?: string[]
  suggestions_max_age_days?: number | null
  suggestions_include_read: boolean
  suggestions_include_archived: boolean
}

const STATUS_LABELS: Record<ReadingDigestSuggestionStatus, string> = {
  saved: "Saved",
  reading: "Reading",
  read: "Read",
  archived: "Archived"
}

const normalizeTags = (value: string[] | undefined): string[] =>
  Array.isArray(value) ? dedupeTags(value) : []

const buildSuggestionsConfig = (
  values: DigestScheduleFormValues
): ReadingDigestSuggestionsConfig | undefined => {
  if (!values.suggestions_enabled) return undefined
  const statuses = Array.from(
    new Set(
      (values.suggestions_status || [])
        .map((status) => String(status || "").trim() as ReadingDigestSuggestionStatus)
        .filter((status): status is ReadingDigestSuggestionStatus =>
          ALLOWED_SUGGESTION_STATUSES.includes(status)
        )
    )
  )

  if (statuses.length === 0) return undefined

  const payload: ReadingDigestSuggestionsConfig = {
    enabled: true,
    status: statuses,
    include_read: Boolean(values.suggestions_include_read),
    include_archived: Boolean(values.suggestions_include_archived)
  }

  if (typeof values.suggestions_limit === "number" && Number.isFinite(values.suggestions_limit)) {
    payload.limit = Math.max(1, Math.min(200, Math.floor(values.suggestions_limit)))
  }

  if (
    typeof values.suggestions_max_age_days === "number" &&
    Number.isFinite(values.suggestions_max_age_days)
  ) {
    payload.max_age_days = Math.max(1, Math.min(3650, Math.floor(values.suggestions_max_age_days)))
  }

  const excludeTags = normalizeTags(values.suggestions_exclude_tags)
  if (excludeTags.length > 0) {
    payload.exclude_tags = excludeTags
  }

  return payload
}

const getBrowserTimezone = (): string => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
  } catch {
    return "UTC"
  }
}

export const DigestSchedulesPanel: React.FC = () => {
  const { t } = useTranslation(["collections", "common"])
  const api = useTldwApiClient()
  const [form] = Form.useForm<DigestScheduleFormValues>()

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [schedules, setSchedules] = useState<ReadingDigestSchedule[]>([])
  const [modalOpen, setModalOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [editingSchedule, setEditingSchedule] = useState<ReadingDigestSchedule | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const defaultFormValues = useMemo<DigestScheduleFormValues>(
    () => ({
      name: "",
      cron: "0 8 * * *",
      timezone: getBrowserTimezone(),
      enabled: true,
      require_online: false,
      format: "md",
      template_name: "",
      suggestions_enabled: false,
      suggestions_limit: 5,
      suggestions_status: ["saved", "reading"],
      suggestions_exclude_tags: [],
      suggestions_max_age_days: null,
      suggestions_include_read: false,
      suggestions_include_archived: false
    }),
    []
  )

  const loadSchedules = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const rows = await api.listReadingDigestSchedules({ limit: 200, offset: 0 })
      setSchedules(rows)
    } catch (err: any) {
      const errorMessage = err?.message || "Failed to load digest schedules"
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }, [api])

  useEffect(() => {
    void loadSchedules()
  }, [loadSchedules])

  const openCreateModal = () => {
    setEditingSchedule(null)
    form.setFieldsValue(defaultFormValues)
    setModalOpen(true)
  }

  const openEditModal = (schedule: ReadingDigestSchedule) => {
    const existingFilters =
      schedule.filters && typeof schedule.filters === "object" ? schedule.filters : undefined
    const suggestions = existingFilters?.suggestions
    form.setFieldsValue({
      name: schedule.name || "",
      cron: schedule.cron || defaultFormValues.cron,
      timezone: schedule.timezone || defaultFormValues.timezone,
      enabled: Boolean(schedule.enabled),
      require_online: Boolean(schedule.require_online),
      format: schedule.format === "html" ? "html" : "md",
      template_name: schedule.template_name || "",
      suggestions_enabled: Boolean(suggestions?.enabled),
      suggestions_limit:
        typeof suggestions?.limit === "number" && Number.isFinite(suggestions.limit)
          ? suggestions.limit
          : defaultFormValues.suggestions_limit,
      suggestions_status:
        Array.isArray(suggestions?.status) && suggestions.status.length > 0
          ? suggestions.status.filter(
              (status): status is ReadingDigestSuggestionStatus =>
                ALLOWED_SUGGESTION_STATUSES.includes(status as ReadingDigestSuggestionStatus)
            )
          : defaultFormValues.suggestions_status,
      suggestions_exclude_tags:
        Array.isArray(suggestions?.exclude_tags) ? suggestions.exclude_tags : [],
      suggestions_max_age_days:
        typeof suggestions?.max_age_days === "number" && Number.isFinite(suggestions.max_age_days)
          ? suggestions.max_age_days
          : null,
      suggestions_include_read: Boolean(suggestions?.include_read),
      suggestions_include_archived: Boolean(suggestions?.include_archived)
    })
    setEditingSchedule(schedule)
    setModalOpen(true)
  }

  const closeModal = () => {
    setModalOpen(false)
    setEditingSchedule(null)
    form.resetFields()
  }

  const onSubmit = async () => {
    try {
      const values = await form.validateFields()
      const suggestionsConfig = buildSuggestionsConfig(values)

      const baseFilters =
        editingSchedule?.filters && typeof editingSchedule.filters === "object"
          ? { ...editingSchedule.filters }
          : {}
      if (suggestionsConfig) {
        baseFilters.suggestions = suggestionsConfig
      } else {
        delete baseFilters.suggestions
      }
      const filters = Object.keys(baseFilters).length > 0 ? baseFilters : undefined

      const payload = {
        name: values.name?.trim() || undefined,
        cron: values.cron.trim(),
        timezone: values.timezone.trim() || "UTC",
        enabled: Boolean(values.enabled),
        require_online: Boolean(values.require_online),
        format: values.format,
        template_name: values.template_name?.trim() || undefined,
        filters
      }

      setSubmitting(true)
      if (editingSchedule) {
        await api.updateReadingDigestSchedule(editingSchedule.id, payload)
        message.success(
          t("collections:digests.updated", "Digest schedule updated")
        )
      } else {
        await api.createReadingDigestSchedule(payload)
        message.success(
          t("collections:digests.created", "Digest schedule created")
        )
      }
      closeModal()
      await loadSchedules()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(err?.message || t("common:error", "Something went wrong"))
    } finally {
      setSubmitting(false)
    }
  }

  const onDelete = async (schedule: ReadingDigestSchedule) => {
    try {
      setDeletingId(schedule.id)
      await api.deleteReadingDigestSchedule(schedule.id)
      message.success(t("collections:digests.deleted", "Digest schedule deleted"))
      await loadSchedules()
    } catch (err: any) {
      message.error(err?.message || t("common:error", "Something went wrong"))
    } finally {
      setDeletingId(null)
    }
  }

  const renderSuggestionsSummary = (schedule: ReadingDigestSchedule) => {
    const suggestions = schedule.filters?.suggestions
    if (!suggestions?.enabled) return null
    const statusLabel = (suggestions.status || [])
      .map((status) => STATUS_LABELS[status] || status)
      .join(", ")
    return (
      <div className="mt-1 text-xs text-text-muted">
        {t(
          "collections:digests.suggestionsSummary",
          "Suggestions enabled: {{limit}} items, statuses: {{statuses}}",
          {
            limit: suggestions.limit ?? 5,
            statuses: statusLabel || "Saved, Reading"
          }
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-medium text-text">
            {t("collections:digests.title", "Digest Schedules")}
          </h3>
          <p className="text-sm text-text-muted">
            {t(
              "collections:digests.description",
              "Configure recurring reading digests and optional suggestion candidates."
            )}
          </p>
        </div>
        <Button type="primary" icon={<Plus className="h-4 w-4" />} onClick={openCreateModal}>
          {t("collections:digests.newSchedule", "New Schedule")}
        </Button>
      </div>

      {error && (
        <Alert
          type="error"
          title={error}
          action={
            <Button size="small" onClick={() => void loadSchedules()}>
              {t("common:retry", "Retry")}
            </Button>
          }
        />
      )}

      {loading ? (
        <div className="py-12 text-center">
          <Spin />
        </div>
      ) : schedules.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={t("collections:digests.empty", "No digest schedules yet")}
        >
          <Button type="primary" onClick={openCreateModal}>
            {t("collections:digests.createFirst", "Create your first schedule")}
          </Button>
        </Empty>
      ) : (
        <List
          dataSource={schedules}
          renderItem={(schedule) => (
            <List.Item
              actions={[
                <Button key="edit" size="small" onClick={() => openEditModal(schedule)}>
                  {t("common:edit", "Edit")}
                </Button>,
                <Popconfirm
                  key="delete"
                  title={t(
                    "collections:digests.deleteConfirm",
                    "Delete this digest schedule?"
                  )}
                  okText={t("common:delete", "Delete")}
                  cancelText={t("common:cancel", "Cancel")}
                  onConfirm={() => void onDelete(schedule)}
                >
                  <Button
                    size="small"
                    danger
                    loading={deletingId === schedule.id}
                  >
                    {t("common:delete", "Delete")}
                  </Button>
                </Popconfirm>
              ]}
            >
              <List.Item.Meta
                avatar={<CalendarClock className="h-4 w-4 mt-1 text-text-muted" />}
                title={
                  <Space size="small" wrap>
                    <span>{schedule.name || t("collections:digests.untitled", "Untitled Digest")}</span>
                    <Tag color={schedule.enabled ? "green" : "default"}>
                      {schedule.enabled
                        ? t("collections:digests.enabled", "Enabled")
                        : t("collections:digests.disabled", "Disabled")}
                    </Tag>
                    <Tag>{(schedule.format || "md").toUpperCase()}</Tag>
                  </Space>
                }
                description={
                  <div>
                    <div className="text-sm text-text-muted">
                      {t("collections:digests.scheduleLine", "Cron: {{cron}} ({{timezone}})", {
                        cron: schedule.cron,
                        timezone: schedule.timezone || "UTC"
                      })}
                    </div>
                    {renderSuggestionsSummary(schedule)}
                  </div>
                }
              />
            </List.Item>
          )}
        />
      )}

      <Modal
        title={
          editingSchedule
            ? t("collections:digests.editSchedule", "Edit Digest Schedule")
            : t("collections:digests.createSchedule", "Create Digest Schedule")
        }
        open={modalOpen}
        onCancel={closeModal}
        onOk={() => void onSubmit()}
        confirmLoading={submitting}
        okText={editingSchedule ? t("common:save", "Save") : t("common:create", "Create")}
        cancelText={t("common:cancel", "Cancel")}
        destroyOnHidden
      >
        <Form<DigestScheduleFormValues>
          layout="vertical"
          form={form}
          initialValues={defaultFormValues}
        >
          <Form.Item
            name="name"
            label={t("collections:digests.form.name", "Name")}
          >
            <Input placeholder={t("collections:digests.form.namePlaceholder", "Daily reading digest")} />
          </Form.Item>

          <Form.Item
            name="cron"
            label={t("collections:digests.form.cron", "Cron")}
            rules={[
              { required: true, message: t("collections:digests.validation.cronRequired", "Cron is required") },
              {
                validator: async (_, value: string) => {
                  const pieces = String(value || "")
                    .trim()
                    .split(/\s+/)
                    .filter(Boolean)
                  if (pieces.length !== 5) {
                    throw new Error(
                      t(
                        "collections:digests.validation.cronFormat",
                        "Cron must have exactly 5 fields"
                      )
                    )
                  }
                }
              }
            ]}
          >
            <Input placeholder="0 8 * * *" />
          </Form.Item>

          <Form.Item
            name="timezone"
            label={t("collections:digests.form.timezone", "Timezone")}
            rules={[
              {
                required: true,
                message: t("collections:digests.validation.timezoneRequired", "Timezone is required")
              }
            ]}
          >
            <Input placeholder="UTC" />
          </Form.Item>

          <div className="grid grid-cols-2 gap-3">
            <Form.Item
              label={t("collections:digests.form.enabled", "Enabled")}
              name="enabled"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
            <Form.Item
              label={t("collections:digests.form.requireOnline", "Require online")}
              name="require_online"
              valuePropName="checked"
            >
              <Switch />
            </Form.Item>
          </div>

          <Form.Item
            name="format"
            label={t("collections:digests.form.format", "Format")}
            rules={[{ required: true }]}
          >
            <Select
              options={[
                { value: "md", label: "Markdown (MD)" },
                { value: "html", label: "HTML" }
              ]}
            />
          </Form.Item>

          <Form.Item
            name="template_name"
            label={t("collections:digests.form.templateName", "Template name (optional)")}
          >
            <Input placeholder="newsletter_markdown" />
          </Form.Item>

          <div className="rounded border border-border p-3 space-y-3">
            <Form.Item
              label={t("collections:digests.form.suggestionsEnabled", "Enable suggestions")}
              name="suggestions_enabled"
              valuePropName="checked"
            >
              <Switch data-testid="digest-suggestions-toggle" />
            </Form.Item>

            <Text type="secondary" className="block text-xs">
              {t(
                "collections:digests.form.suggestionsHelper",
                "Suggestions are local-only and heuristic-based."
              )}
            </Text>

            <Form.Item noStyle shouldUpdate>
              {({ getFieldValue }) => {
                if (!getFieldValue("suggestions_enabled")) {
                  return null
                }
                return (
                  <div className="space-y-3">
                    <Form.Item
                      name="suggestions_limit"
                      label={t("collections:digests.form.suggestionsLimit", "Suggestions limit")}
                      rules={[
                        {
                          required: true,
                          message: t(
                            "collections:digests.validation.suggestionsLimitRequired",
                            "Suggestions limit is required"
                          )
                        },
                        {
                          validator: async (_, value: number | null | undefined) => {
                            const asNumber = Number(value)
                            if (!Number.isFinite(asNumber) || asNumber < 1 || asNumber > 200) {
                              throw new Error(
                                t(
                                  "collections:digests.validation.suggestionsLimitRange",
                                  "Suggestions limit must be between 1 and 200"
                                )
                              )
                            }
                          }
                        }
                      ]}
                    >
                      <InputNumber
                        min={1}
                        max={200}
                        className="w-full"
                        data-testid="digest-suggestions-limit-input"
                      />
                    </Form.Item>

                    <Form.Item
                      name="suggestions_status"
                      label={t("collections:digests.form.suggestionsStatus", "Status list")}
                      rules={[
                        {
                          required: true,
                          message: t(
                            "collections:digests.validation.suggestionsStatusRequired",
                            "Select at least one status"
                          )
                        },
                        {
                          validator: async (_, value: ReadingDigestSuggestionStatus[] | undefined) => {
                            const list = Array.isArray(value) ? value : []
                            const isValid = list.every((status) =>
                              ALLOWED_SUGGESTION_STATUSES.includes(status)
                            )
                            if (!isValid || list.length === 0) {
                              throw new Error(
                                t(
                                  "collections:digests.validation.suggestionsStatusInvalid",
                                  "Status list contains invalid values"
                                )
                              )
                            }
                          }
                        }
                      ]}
                    >
                      <Select
                        mode="multiple"
                        options={ALLOWED_SUGGESTION_STATUSES.map((status) => ({
                          value: status,
                          label: STATUS_LABELS[status]
                        }))}
                        data-testid="digest-suggestions-status-select"
                      />
                    </Form.Item>

                    <Form.Item
                      name="suggestions_exclude_tags"
                      label={t("collections:digests.form.suggestionsExcludeTags", "Exclude tags")}
                    >
                      <Select
                        mode="tags"
                        tokenSeparators={[","]}
                        placeholder={t(
                          "collections:digests.form.suggestionsExcludeTagsPlaceholder",
                          "Enter tags to exclude"
                        )}
                        data-testid="digest-suggestions-exclude-tags-input"
                      />
                    </Form.Item>

                    <Form.Item
                      name="suggestions_max_age_days"
                      label={t("collections:digests.form.suggestionsMaxAge", "Max age (days)")}
                      rules={[
                        {
                          validator: async (_, value: number | null | undefined) => {
                            if (value === null || typeof value === "undefined") return
                            const asNumber = Number(value)
                            if (!Number.isFinite(asNumber) || asNumber < 1 || asNumber > 3650) {
                              throw new Error(
                                t(
                                  "collections:digests.validation.suggestionsMaxAgeRange",
                                  "Max age must be between 1 and 3650"
                                )
                              )
                            }
                          }
                        }
                      ]}
                    >
                      <InputNumber
                        min={1}
                        max={3650}
                        className="w-full"
                        data-testid="digest-suggestions-max-age-input"
                      />
                    </Form.Item>

                    <div className="grid grid-cols-2 gap-3">
                      <Form.Item
                        name="suggestions_include_read"
                        label={t(
                          "collections:digests.form.suggestionsIncludeRead",
                          "Include read"
                        )}
                        valuePropName="checked"
                      >
                        <Switch data-testid="digest-suggestions-include-read-toggle" />
                      </Form.Item>
                      <Form.Item
                        name="suggestions_include_archived"
                        label={t(
                          "collections:digests.form.suggestionsIncludeArchived",
                          "Include archived"
                        )}
                        valuePropName="checked"
                      >
                        <Switch data-testid="digest-suggestions-include-archived-toggle" />
                      </Form.Item>
                    </div>
                  </div>
                )
              }}
            </Form.Item>
          </div>
        </Form>
      </Modal>
    </div>
  )
}

export default DigestSchedulesPanel
