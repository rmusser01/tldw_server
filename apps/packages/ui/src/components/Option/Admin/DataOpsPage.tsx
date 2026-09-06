import React, { useState, useRef, useCallback, useEffect } from "react"
import { useTranslation } from "react-i18next"
import {
  Card,
  Table,
  Tabs,
  Button,
  Input,
  InputNumber,
  Modal,
  Select,
  Space,
  Popconfirm,
  Form,
  Tag,
  Descriptions,
  message
} from "antd"
import {
  PlusOutlined,
  DeleteOutlined,
  ReloadOutlined,
  UndoOutlined,
  PlayCircleOutlined,
  EyeOutlined
} from "@ant-design/icons"
import { Alert } from "@/components/ui/primitives"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { tldwClient } from "@/services/tldw/TldwApiClient"

// ── Backups Tab ──

// Mirrors the backend's _BACKUP_DATASETS allowlist (#2917): the UI used to
// offer "chachanotes"/"users", which the API rejects, and missed three valid
// datasets entirely. All datasets except authnz are per-user and require a
// target user id.
const PER_USER_BACKUP_DATASETS = new Set([
  "media",
  "chacha",
  "prompts",
  "evaluations",
  "audit"
])

const useBackupDatasetOptions = (t: (k: string, d: string) => string) => [
  { value: "media", label: t("settings:adminDataOps.datasetMedia", "Media") },
  { value: "chacha", label: t("settings:adminDataOps.datasetChaCha", "Chats & Notes") },
  { value: "prompts", label: t("settings:adminDataOps.datasetPrompts", "Prompts") },
  { value: "evaluations", label: t("settings:adminDataOps.datasetEvaluations", "Evaluations") },
  { value: "audit", label: t("settings:adminDataOps.datasetAudit", "Audit log") },
  { value: "authnz", label: t("settings:adminDataOps.datasetAuthnz", "Users & auth (server-wide)") }
]

const BackupsTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [backups, setBackups] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [createForm] = Form.useForm()
  const [creating, setCreating] = useState(false)
  const [restoring, setRestoring] = useState<string | null>(null)
  const datasetOptions = useBackupDatasetOptions(t)
  const selectedBackupDataset = Form.useWatch("dataset", createForm)

  // Schedules
  const [schedules, setSchedules] = useState<any[]>([])
  const [schedulesLoading, setSchedulesLoading] = useState(false)
  const [scheduleForm] = Form.useForm()
  const [creatingSchedule, setCreatingSchedule] = useState(false)
  const selectedScheduleDataset = Form.useWatch("dataset", scheduleForm)

  const loadBackups = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listBackups()
      setBackups(Array.isArray(result) ? result : result?.data ?? result?.backups ?? [])
    } catch (err) {
      onGuardError(err)
    } finally {
      setLoading(false)
    }
  }, [onGuardError])

  const loadSchedules = useCallback(async () => {
    setSchedulesLoading(true)
    try {
      const result = await tldwClient.listBackupSchedules()
      setSchedules(Array.isArray(result) ? result : result?.data ?? result?.schedules ?? [])
    } catch (err) {
      onGuardError(err)
    } finally {
      setSchedulesLoading(false)
    }
  }, [onGuardError])

  useEffect(() => {
    void loadBackups()
    void loadSchedules()
  }, [loadBackups, loadSchedules])

  const handleCreateBackup = async () => {
    try {
      const values = await createForm.validateFields()
      setCreating(true)
      await tldwClient.createBackup({
        dataset: values.dataset,
        user_id: values.user_id || undefined
      })
      message.success(t("settings:adminDataOps.backupCreated", "Backup created"))
      createForm.resetFields()
      void loadBackups()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.backupCreateFailed", "Failed to create backup")))
    } finally {
      setCreating(false)
    }
  }

  const handleRestore = async (backupId: string) => {
    setRestoring(backupId)
    try {
      await tldwClient.restoreBackup(backupId)
      message.success(t("settings:adminDataOps.backupRestored", "Backup restored successfully"))
      void loadBackups()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.backupRestoreFailed", "Failed to restore backup")))
    } finally {
      setRestoring(null)
    }
  }

  const handleCreateSchedule = async () => {
    try {
      const values = await scheduleForm.validateFields()
      setCreatingSchedule(true)
      await tldwClient.createBackupSchedule({
        dataset: values.dataset,
        target_user_id: values.target_user_id || undefined,
        frequency: values.frequency,
        time_of_day: values.time_of_day,
        retention_count: values.retention_count
      })
      message.success(t("settings:adminDataOps.scheduleCreated", "Backup schedule created"))
      scheduleForm.resetFields()
      void loadSchedules()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.scheduleCreateFailed", "Failed to create schedule")))
    } finally {
      setCreatingSchedule(false)
    }
  }

  const handleDeleteSchedule = async (scheduleId: number) => {
    try {
      await tldwClient.deleteBackupSchedule(scheduleId)
      message.success(t("settings:adminDataOps.scheduleDeleted", "Schedule deleted"))
      void loadSchedules()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.scheduleDeleteFailed", "Failed to delete schedule")))
    }
  }

  const backupColumns = [
    {
      title: t("settings:adminDataOps.colDataset", "Dataset"),
      dataIndex: "dataset",
      key: "dataset",
      render: (v: string) => <Tag>{v || "unknown"}</Tag>
    },
    {
      title: t("settings:adminDataOps.colUser", "User"),
      dataIndex: "user_id",
      key: "user_id",
      width: 80,
      render: (v: number) => v ?? "\u2014"
    },
    {
      title: t("settings:adminDataOps.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      render: (v: string) => (v ? new Date(v).toLocaleString() : "\u2014")
    },
    {
      title: t("settings:adminDataOps.colSize", "Size"),
      dataIndex: "size",
      key: "size",
      width: 100,
      render: (v: number | string) => v ?? "\u2014"
    },
    {
      title: t("settings:adminDataOps.colStatus", "Status"),
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (status: string) => {
        const color = status === "completed" ? "green" : status === "failed" ? "red" : "blue"
        return <Tag color={color}>{status || "unknown"}</Tag>
      }
    },
    {
      title: t("settings:adminDataOps.colActions", "Actions"),
      key: "actions",
      width: 100,
      render: (_: any, record: any) => (
        <Popconfirm
          title={t("settings:adminDataOps.restoreConfirm", "Restore this backup? This will overwrite current data.")}
          onConfirm={() => handleRestore(record.id ?? record.backup_id)}
          okText={t("settings:adminDataOps.restore", "Restore")}
          okButtonProps={{ danger: true }}
        >
          <Button
            type="text"
            size="small"
            icon={<UndoOutlined />}
            loading={restoring === (record.id ?? record.backup_id)}
          >
            {t("settings:adminDataOps.restore", "Restore")}
          </Button>
        </Popconfirm>
      )
    }
  ]

  const scheduleColumns = [
    {
      title: t("settings:adminDataOps.colDataset", "Dataset"),
      dataIndex: "dataset",
      key: "dataset",
      render: (v: string) => <Tag>{v}</Tag>
    },
    {
      title: t("settings:adminDataOps.colSchedule", "Schedule"),
      key: "schedule",
      render: (_: unknown, r: any) => (
        <code>
          {r.frequency || "\u2014"} @ {r.time_of_day || "\u2014"}
        </code>
      )
    },
    {
      title: t("settings:adminDataOps.colRetentionCount", "Keep"),
      dataIndex: "retention_count",
      key: "retention_count",
      width: 90,
      render: (v: number) => v ?? "\u2014"
    },
    {
      title: t("settings:adminDataOps.colActions", "Actions"),
      key: "actions",
      width: 80,
      render: (_: any, record: any) => (
        <Popconfirm
          title={t("settings:adminDataOps.deleteScheduleConfirm", "Delete this schedule?")}
          onConfirm={() => handleDeleteSchedule(record.id ?? record.schedule_id)}
          okText={t("settings:adminDataOps.delete", "Delete")}
          okButtonProps={{ danger: true }}
        >
          <Button type="text" size="small" danger icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ]

  return (
    <div>
      <Card
        title={t("settings:adminDataOps.backupsCardTitle", "Backups")}
        style={{ marginBottom: 16 }}
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={() => loadBackups()}>
            {t("common:refresh", "Refresh")}
          </Button>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Form form={createForm} layout="inline">
            <Form.Item
              name="dataset"
              rules={[{ required: true, message: t("settings:adminDataOps.datasetRequired", "Dataset is required") }]}
            >
              <Select
                placeholder={t("settings:adminDataOps.datasetPlaceholder", "Dataset")}
                style={{ width: 200 }}
                options={datasetOptions}
              />
            </Form.Item>
            <Form.Item
              name="user_id"
              rules={[
                {
                  required: PER_USER_BACKUP_DATASETS.has(selectedBackupDataset),
                  message: t(
                    "settings:adminDataOps.userIdRequired",
                    "This dataset is per-user - pick the user to back up"
                  )
                }
              ]}
            >
              <InputNumber
                placeholder={
                  PER_USER_BACKUP_DATASETS.has(selectedBackupDataset)
                    ? t("settings:adminDataOps.userIdPlaceholderRequired", "User ID (required)")
                    : t("settings:adminDataOps.userIdPlaceholder", "User ID")
                }
                min={1}
                style={{ width: 160 }}
              />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleCreateBackup}
                loading={creating}
              >
                {t("settings:adminDataOps.createBackup", "Create Backup")}
              </Button>
            </Form.Item>
          </Form>
        </div>
        <Table
          dataSource={backups}
          columns={backupColumns}
          rowKey={(r) => r.id ?? r.backup_id ?? JSON.stringify(r)}
          loading={loading}
          pagination={backups.length > 20 ? { pageSize: 20 } : false}
          size="small"
          locale={{
            emptyText: t(
              "settings:adminDataOps.backupsEmpty",
              "No backups yet. Pick a dataset above and create the first one."
            )
          }}
        />
      </Card>

      <Card
        title={t("settings:adminDataOps.schedulesCardTitle", "Backup Schedules")}
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={() => loadSchedules()}>
            {t("common:refresh", "Refresh")}
          </Button>
        }
      >
        {/* Starter chips fill the form, mirroring the Monitoring alert-rule
            pattern (#2899 I1) - now speaking the API's actual schedule
            contract (frequency/time/keep-count, #2917). */}
        <div style={{ marginBottom: 8 }}>
          <Space size="small" wrap>
            {[
              {
                key: "nightly",
                label: t(
                  "settings:adminDataOps.schedulePresetNightly",
                  "Nightly at 02:00, keep 14"
                ),
                frequency: "daily",
                time_of_day: "02:00",
                retention_count: 14
              },
              {
                key: "weekly",
                label: t(
                  "settings:adminDataOps.schedulePresetWeekly",
                  "Weekly at 03:00, keep 8"
                ),
                frequency: "weekly",
                time_of_day: "03:00",
                retention_count: 8
              }
            ].map((preset) => (
              <Tag
                key={preset.key}
                style={{ cursor: "pointer" }}
                onClick={() =>
                  scheduleForm.setFieldsValue({
                    frequency: preset.frequency,
                    time_of_day: preset.time_of_day,
                    retention_count: preset.retention_count
                  })
                }
              >
                {preset.label}
              </Tag>
            ))}
          </Space>
        </div>
        <div style={{ marginBottom: 16 }}>
          <Form form={scheduleForm} layout="inline">
            <Form.Item
              name="dataset"
              rules={[{ required: true, message: t("settings:adminDataOps.datasetRequired", "Dataset is required") }]}
            >
              <Select
                placeholder={t("settings:adminDataOps.datasetPlaceholder", "Dataset")}
                style={{ width: 200 }}
                options={datasetOptions}
              />
            </Form.Item>
            <Form.Item
              name="target_user_id"
              rules={[
                {
                  required: PER_USER_BACKUP_DATASETS.has(selectedScheduleDataset),
                  message: t(
                    "settings:adminDataOps.userIdRequired",
                    "This dataset is per-user - pick the user to back up"
                  )
                }
              ]}
            >
              <InputNumber
                placeholder={
                  PER_USER_BACKUP_DATASETS.has(selectedScheduleDataset)
                    ? t("settings:adminDataOps.userIdPlaceholderRequired", "User ID (required)")
                    : t("settings:adminDataOps.userIdPlaceholder", "User ID")
                }
                min={1}
                style={{ width: 150 }}
              />
            </Form.Item>
            <Form.Item
              name="frequency"
              rules={[{ required: true, message: t("settings:adminDataOps.frequencyRequired", "Frequency is required") }]}
            >
              <Select
                placeholder={t("settings:adminDataOps.frequencyPlaceholder", "Frequency")}
                style={{ width: 130 }}
                options={[
                  { value: "daily", label: t("settings:adminDataOps.frequencyDaily", "Daily") },
                  { value: "weekly", label: t("settings:adminDataOps.frequencyWeekly", "Weekly") },
                  { value: "monthly", label: t("settings:adminDataOps.frequencyMonthly", "Monthly") }
                ]}
              />
            </Form.Item>
            <Form.Item
              name="time_of_day"
              rules={[
                { required: true, message: t("settings:adminDataOps.timeRequired", "Time is required") },
                {
                  pattern: /^\d{2}:\d{2}$/,
                  message: t("settings:adminDataOps.timeFormat", "Use 24h HH:MM, e.g. 02:00")
                }
              ]}
            >
              <Input placeholder={t("settings:adminDataOps.timePlaceholder", "Time (HH:MM)")} style={{ width: 130 }} />
            </Form.Item>
            <Form.Item
              name="retention_count"
              rules={[{ required: true, message: t("settings:adminDataOps.retentionRequired", "Retention is required") }]}
            >
              <InputNumber
                placeholder={t("settings:adminDataOps.retentionCountPlaceholder", "Backups to keep")}
                min={1}
                max={1000}
                style={{ width: 150 }}
              />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleCreateSchedule}
                loading={creatingSchedule}
              >
                {t("settings:adminDataOps.addSchedule", "Add Schedule")}
              </Button>
            </Form.Item>
          </Form>
        </div>
        <Table
          dataSource={schedules}
          columns={scheduleColumns}
          rowKey={(r) => r.id ?? r.schedule_id ?? JSON.stringify(r)}
          loading={schedulesLoading}
          pagination={false}
          size="small"
          locale={{
            emptyText: t(
              "settings:adminDataOps.schedulesEmpty",
              "No schedules yet. Recurring backups run themselves - add one above (for example: nightly at 02:00)."
            )
          }}
        />
      </Card>
    </div>
  )
}

// ── DSR Tab ──

const DsrTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [dsrs, setDsrs] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [createForm] = Form.useForm()
  const [creating, setCreating] = useState(false)
  const [executing, setExecuting] = useState<number | null>(null)

  // Preview state
  const [previewModalOpen, setPreviewModalOpen] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewData, setPreviewData] = useState<any>(null)
  const [pendingDsr, setPendingDsr] = useState<any>(null)

  const loadDsrs = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listDsrs({ limit: 100 })
      setDsrs(Array.isArray(result) ? result : result?.data ?? result?.requests ?? [])
    } catch (err) {
      onGuardError(err)
    } finally {
      setLoading(false)
    }
  }, [onGuardError])

  useEffect(() => {
    void loadDsrs()
  }, [loadDsrs])

  const handlePreview = async () => {
    try {
      const values = await createForm.validateFields()
      setPreviewLoading(true)
      const result = await tldwClient.previewDsr({
        requester_identifier: values.requester_identifier,
        request_type: values.request_type || undefined,
        categories: values.categories?.length ? values.categories : undefined
      })
      setPreviewData(result)
      setPendingDsr(values)
      setPreviewModalOpen(true)
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.dsrPreviewFailed", "Failed to preview DSR")))
    } finally {
      setPreviewLoading(false)
    }
  }

  const handleRecordDsr = async () => {
    if (!pendingDsr) return
    setCreating(true)
    try {
      await tldwClient.createDsr({
        requester_identifier: pendingDsr.requester_identifier,
        request_type: pendingDsr.request_type || "erasure",
        categories: pendingDsr.categories?.length ? pendingDsr.categories : undefined,
        client_request_id: pendingDsr.client_request_id || undefined,
        notes: pendingDsr.notes || undefined
      })
      message.success(t("settings:adminDataOps.dsrRecorded", "DSR recorded"))
      setPreviewModalOpen(false)
      setPendingDsr(null)
      setPreviewData(null)
      createForm.resetFields()
      void loadDsrs()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.dsrRecordFailed", "Failed to record DSR")))
    } finally {
      setCreating(false)
    }
  }

  const handleExecute = async (requestId: number) => {
    setExecuting(requestId)
    try {
      await tldwClient.executeDsr(requestId)
      message.success(t("settings:adminDataOps.dsrExecuted", "DSR executed"))
      void loadDsrs()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.dsrExecuteFailed", "Failed to execute DSR")))
    } finally {
      setExecuting(null)
    }
  }

  const dsrColumns = [
    {
      title: t("settings:adminDataOps.colId", "ID"),
      dataIndex: "id",
      key: "id",
      width: 60
    },
    {
      title: t("settings:adminDataOps.colRequester", "Requester"),
      dataIndex: "requester_identifier",
      key: "requester_identifier"
    },
    {
      title: t("settings:adminDataOps.colType", "Type"),
      dataIndex: "request_type",
      key: "request_type",
      width: 100,
      render: (v: string) => <Tag>{v || "erasure"}</Tag>
    },
    {
      title: t("settings:adminDataOps.colStatus", "Status"),
      dataIndex: "status",
      key: "status",
      width: 120,
      render: (status: string) => {
        const color =
          status === "completed" ? "green"
          : status === "executing" ? "blue"
          : status === "failed" ? "red"
          : "default"
        return <Tag color={color}>{status || "pending"}</Tag>
      }
    },
    {
      title: t("settings:adminDataOps.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      render: (v: string) => (v ? new Date(v).toLocaleString() : "\u2014")
    },
    {
      title: t("settings:adminDataOps.colActions", "Actions"),
      key: "actions",
      width: 100,
      render: (_: any, record: any) => {
        const canExecute =
          record.request_type === "erasure" &&
          record.status !== "completed" &&
          record.status !== "executing"
        if (!canExecute) return null
        return (
          <Popconfirm
            title={t("settings:adminDataOps.executeDsrConfirm", "Execute this DSR? This action cannot be undone.")}
            onConfirm={() => handleExecute(record.id)}
            okText={t("settings:adminDataOps.execute", "Execute")}
            okButtonProps={{ danger: true }}
          >
            <Button
              type="text"
              size="small"
              danger
              icon={<PlayCircleOutlined />}
              loading={executing === record.id}
            >
              {t("settings:adminDataOps.execute", "Execute")}
            </Button>
          </Popconfirm>
        )
      }
    }
  ]

  return (
    <div>
      <Card
        title={t("settings:adminDataOps.dsrCardTitle", "Data Subject Requests")}
        style={{ marginBottom: 16 }}
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={() => loadDsrs()}>
            {t("common:refresh", "Refresh")}
          </Button>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Form form={createForm} layout="vertical" style={{ maxWidth: 600 }}>
            <Form.Item
              name="requester_identifier"
              label={t("settings:adminDataOps.requesterLabel", "Requester Identifier")}
              rules={[{ required: true, message: t("settings:adminDataOps.requesterRequired", "Requester identifier is required") }]}
            >
              <Input placeholder={t("settings:adminDataOps.requesterPlaceholder", "Email, username, or user ID")} />
            </Form.Item>
            <Form.Item name="request_type" label={t("settings:adminDataOps.requestTypeLabel", "Request Type")} initialValue="erasure">
              <Select
                options={[
                  { value: "erasure", label: t("settings:adminDataOps.typeErasure", "Erasure (Right to be Forgotten)") },
                  { value: "export", label: t("settings:adminDataOps.typeExport", "Data Export") },
                  { value: "access", label: t("settings:adminDataOps.typeAccess", "Access Request") },
                  { value: "rectification", label: t("settings:adminDataOps.typeRectification", "Rectification") }
                ]}
              />
            </Form.Item>
            <Form.Item name="categories" label={t("settings:adminDataOps.categoriesLabel", "Categories (optional)")}>
              <Select
                mode="multiple"
                placeholder={t("settings:adminDataOps.categoriesPlaceholder", "Select categories to include")}
                allowClear
                options={[
                  { value: "media", label: t("settings:adminDataOps.categoryMedia", "Media") },
                  { value: "chats", label: t("settings:adminDataOps.categoryChats", "Chats") },
                  { value: "notes", label: t("settings:adminDataOps.categoryNotes", "Notes") },
                  { value: "embeddings", label: t("settings:adminDataOps.categoryEmbeddings", "Embeddings") },
                  { value: "profile", label: t("settings:adminDataOps.categoryProfile", "Profile") }
                ]}
              />
            </Form.Item>
            <Form.Item name="notes" label={t("settings:adminDataOps.notesLabel", "Notes (optional)")}>
              <Input.TextArea rows={2} placeholder={t("settings:adminDataOps.notesPlaceholder", "Internal notes about this request")} />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button
                  icon={<EyeOutlined />}
                  onClick={handlePreview}
                  loading={previewLoading}
                >
                  {t("settings:adminDataOps.preview", "Preview")}
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </div>
        <Table
          dataSource={dsrs}
          columns={dsrColumns}
          rowKey="id"
          loading={loading}
          pagination={dsrs.length > 20 ? { pageSize: 20 } : false}
          size="small"
        />
      </Card>

      <Modal
        title={t("settings:adminDataOps.dsrPreviewTitle", "DSR Preview")}
        open={previewModalOpen}
        onCancel={() => { setPreviewModalOpen(false); setPreviewData(null); setPendingDsr(null) }}
        footer={[
          <Button key="cancel" onClick={() => { setPreviewModalOpen(false); setPreviewData(null); setPendingDsr(null) }}>
            {t("common:cancel", "Cancel")}
          </Button>,
          <Button key="record" type="primary" onClick={handleRecordDsr} loading={creating}>
            {t("settings:adminDataOps.recordDsr", "Record DSR")}
          </Button>
        ]}
        width={600}
      >
        {previewData && (
          <div>
            <p style={{ marginBottom: 12 }}>
              {t("settings:adminDataOps.previewFoundFor", "The following data was found for")} <strong>{pendingDsr?.requester_identifier}</strong>:
            </p>
            <Descriptions bordered size="small" column={1}>
              {Object.entries(previewData?.counts ?? previewData?.data ?? previewData ?? {}).map(
                ([key, value]) => (
                  <Descriptions.Item key={key} label={key}>
                    {typeof value === "number" ? `${value} ${t("settings:adminDataOps.records", "record(s)")}` : String(value ?? "\u2014")}
                  </Descriptions.Item>
                )
              )}
            </Descriptions>
          </div>
        )}
      </Modal>
    </div>
  )
}

// ── Retention Policies Tab ──

const RetentionPoliciesTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [policies, setPolicies] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const [editValue, setEditValue] = useState<number>(0)
  const [saving, setSaving] = useState(false)

  const loadPolicies = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listRetentionPolicies()
      const arr = Array.isArray(result) ? result : result?.data ?? result?.policies ?? []
      // Handle object-style response { key: days, ... }
      if (!Array.isArray(result) && typeof result === "object" && !result?.data && !result?.policies) {
        const entries = Object.entries(result).map(([key, val]) => ({
          key,
          retention_days: typeof val === "number" ? val : (val as any)?.retention_days ?? 0
        }))
        setPolicies(entries)
      } else {
        setPolicies(arr)
      }
    } catch (err) {
      onGuardError(err)
    } finally {
      setLoading(false)
    }
  }, [onGuardError])

  useEffect(() => {
    void loadPolicies()
  }, [loadPolicies])

  const handleSave = async (policyKey: string) => {
    setSaving(true)
    try {
      await tldwClient.updateRetentionPolicy(policyKey, { retention_days: editValue })
      message.success(`${t("settings:adminDataOps.retentionUpdatedPrefix", "Retention policy")} "${policyKey}" ${t("settings:adminDataOps.retentionUpdatedSuffix", "updated")}`)
      setEditingKey(null)
      void loadPolicies()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.retentionUpdateFailed", "Failed to update retention policy")))
    } finally {
      setSaving(false)
    }
  }

  const policyColumns = [
    {
      title: t("settings:adminDataOps.colPolicyKey", "Policy Key"),
      dataIndex: "key",
      key: "key",
      render: (v: string) => <code>{v}</code>
    },
    {
      title: t("settings:adminDataOps.colRetentionDays", "Retention (days)"),
      dataIndex: "retention_days",
      key: "retention_days",
      width: 200,
      render: (days: number, record: any) => {
        if (editingKey === record.key) {
          return (
            <Space>
              <InputNumber
                value={editValue}
                min={0}
                onChange={(v) => setEditValue(v ?? 0)}
                style={{ width: 100 }}
                size="small"
              />
              <Button size="small" type="primary" onClick={() => handleSave(record.key)} loading={saving}>
                {t("common:save", "Save")}
              </Button>
              <Button size="small" onClick={() => setEditingKey(null)}>
                {t("common:cancel", "Cancel")}
              </Button>
            </Space>
          )
        }
        return (
          <Space>
            <span>{days ?? 0}</span>
            <Button
              type="link"
              size="small"
              onClick={() => {
                setEditingKey(record.key)
                setEditValue(days ?? 0)
              }}
            >
              {t("common:edit", "Edit")}
            </Button>
          </Space>
        )
      }
    }
  ]

  return (
    <Card
      title={t("settings:adminDataOps.retentionCardTitle", "Retention Policies")}
      extra={
        <Button size="small" icon={<ReloadOutlined />} onClick={() => loadPolicies()}>
          {t("common:refresh", "Refresh")}
        </Button>
      }
    >
      <Table
        dataSource={policies}
        columns={policyColumns}
        rowKey="key"
        loading={loading}
        pagination={false}
        size="small"
      />
    </Card>
  )
}

// ── Bundles Tab ──

const BundlesTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [bundles, setBundles] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [createForm] = Form.useForm()
  const [creating, setCreating] = useState(false)

  const loadBundles = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listBundles()
      setBundles(Array.isArray(result) ? result : result?.data ?? result?.bundles ?? [])
    } catch (err) {
      onGuardError(err)
    } finally {
      setLoading(false)
    }
  }, [onGuardError])

  useEffect(() => {
    void loadBundles()
  }, [loadBundles])

  const handleCreate = async () => {
    try {
      const values = await createForm.validateFields()
      setCreating(true)
      await tldwClient.createBundle({ datasets: values.datasets })
      message.success(t("settings:adminDataOps.bundleCreated", "Bundle created"))
      createForm.resetFields()
      void loadBundles()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.bundleCreateFailed", "Failed to create bundle")))
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (bundleId: string) => {
    try {
      await tldwClient.deleteBundle(bundleId)
      message.success(t("settings:adminDataOps.bundleDeleted", "Bundle deleted"))
      void loadBundles()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminDataOps.bundleDeleteFailed", "Failed to delete bundle")))
    }
  }

  const bundleColumns = [
    {
      title: t("settings:adminDataOps.colId", "ID"),
      dataIndex: "id",
      key: "id",
      width: 200,
      render: (v: string) => <code style={{ fontSize: 12 }}>{v}</code>
    },
    {
      title: t("settings:adminDataOps.colDatasets", "Datasets"),
      dataIndex: "datasets",
      key: "datasets",
      render: (datasets: string[]) =>
        Array.isArray(datasets)
          ? datasets.map((d) => <Tag key={d}>{d}</Tag>)
          : String(datasets ?? "\u2014")
    },
    {
      title: t("settings:adminDataOps.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      render: (v: string) => (v ? new Date(v).toLocaleString() : "\u2014")
    },
    {
      title: t("settings:adminDataOps.colActions", "Actions"),
      key: "actions",
      width: 80,
      render: (_: any, record: any) => (
        <Popconfirm
          title={t("settings:adminDataOps.deleteBundleConfirm", "Delete this bundle?")}
          onConfirm={() => handleDelete(record.id ?? record.bundle_id)}
          okText={t("settings:adminDataOps.delete", "Delete")}
          okButtonProps={{ danger: true }}
        >
          <Button type="text" size="small" danger icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ]

  return (
    <Card
      title={t("settings:adminDataOps.bundlesCardTitle", "Backup Bundles")}
      extra={
        <Button size="small" icon={<ReloadOutlined />} onClick={() => loadBundles()}>
          {t("common:refresh", "Refresh")}
        </Button>
      }
    >
      <div style={{ marginBottom: 16 }}>
        <Form form={createForm} layout="inline">
          <Form.Item
            name="datasets"
            rules={[{ required: true, message: t("settings:adminDataOps.datasetsRequired", "Select at least one dataset") }]}
          >
            <Select
              mode="multiple"
              placeholder={t("settings:adminDataOps.datasetsPlaceholder", "Select datasets")}
              style={{ width: 320 }}
              options={[
                { value: "media", label: t("settings:adminDataOps.datasetMedia", "Media") },
                { value: "chachanotes", label: t("settings:adminDataOps.datasetChaChaNotes", "ChaChaNotes") },
                { value: "users", label: t("settings:adminDataOps.datasetUsers", "Users") },
                { value: "evaluations", label: t("settings:adminDataOps.datasetEvaluations", "Evaluations") }
              ]}
            />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={handleCreate}
              loading={creating}
            >
              {t("settings:adminDataOps.createBundle", "Create Bundle")}
            </Button>
          </Form.Item>
        </Form>
      </div>
      <Table
        dataSource={bundles}
        columns={bundleColumns}
        rowKey={(r) => r.id ?? r.bundle_id ?? JSON.stringify(r)}
        loading={loading}
        pagination={bundles.length > 20 ? { pageSize: 20 } : false}
        size="small"
      />
    </Card>
  )
}

// ── Main Page ──

const DataOpsPage: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)
  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  // Mark initial load done (tabs load their own data)
  useEffect(() => {
    initialLoadRef.current = true
  }, [])

  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title={t("settings:adminDataOps.forbiddenTitle", "Access Denied")}>
        {t(
          "settings:adminDataOps.forbiddenBody",
          "You don't have permission to access data operations."
        )}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title={t("settings:adminDataOps.notFoundTitle", "Not Available")}>
        {t(
          "settings:adminDataOps.notFoundBody",
          "Data operations are not available on this server."
        )}
      </Alert>
    )
  }

  const tabItems = [
    {
      key: "backups",
      label: t("settings:adminDataOps.tabBackups", "Backups"),
      children: <BackupsTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "dsr",
      label: t("settings:adminDataOps.tabDsr", "Data Subject Requests"),
      children: <DsrTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "retention",
      label: t("settings:adminDataOps.tabRetention", "Retention Policies"),
      children: <RetentionPoliciesTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "bundles",
      label: t("settings:adminDataOps.tabBundles", "Bundles"),
      children: <BundlesTab onGuardError={markAdminGuardFromError} />
    }
  ]

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h1 style={{ marginBottom: 16, fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminDataOps.title", "Data Operations")}</h1>
      <Tabs items={tabItems} defaultActiveKey="backups" />
    </div>
  )
}

export default DataOpsPage
