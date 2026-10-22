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
  EyeOutlined,
  DownloadOutlined
} from "@ant-design/icons"
import { Alert } from "@/components/ui/primitives"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import type { BundleImportResult } from "@/services/tldw/domains/admin"
import { downloadBlob } from "@/utils/download-blob"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope"
import { useAdminMutation, useAdminRequestScope } from "./admin-request-scope"

import { formatAdminDateTime } from "./admin-format"

// ── Backups Tab ──

// Deterministic fallback row keys for records the server sends without an
// id: a `dataset|created_at|filename|name` composite instead of
// JSON.stringify, which leaks whole records into the DOM as keys and still
// collides on identical rows (C-S5).
const dataOpsRowKey = (r: any): string =>
  r.id ?? r.backup_id ?? r.schedule_id ?? r.bundle_id ??
  `${r.dataset ?? "ds"}|${r.created_at ?? ""}|${r.filename ?? r.name ?? ""}`

// Duplicate composites (e.g. two schedules for one dataset before the first
// run) get a fetch-order suffix (`#2`, `#3`, ...) so every row keeps a
// stable, unique key across re-renders of the same snapshot.
const tagDuplicateRowKeys = (records: any[]): any[] => {
  const seen = new Map<string, number>()
  let tagged = false
  const result = records.map((record) => {
    const base = dataOpsRowKey(record)
    const nth = (seen.get(base) ?? 0) + 1
    seen.set(base, nth)
    if (nth > 1) {
      tagged = true
      return { ...record, _rowKey: `${base}#${nth}` }
    }
    return record
  })
  return tagged ? result : records
}

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

  const getRestoreTarget = (record: {
    id?: unknown
    backup_id?: unknown
    dataset?: unknown
    user_id?: unknown
  }) => {
    const backupId = record.id ?? record.backup_id
    if (typeof backupId !== "string" || !backupId.trim()) return null
    const dataset = record.dataset
    if (typeof dataset !== "string" || (!PER_USER_BACKUP_DATASETS.has(dataset) && dataset !== "authnz")) return null
    const userId = record.user_id
    if (userId != null) {
      if (typeof userId !== "number" || !Number.isSafeInteger(userId) || userId <= 0) return null
    } else if (PER_USER_BACKUP_DATASETS.has(dataset)) {
      return null
    }
    return { backupId, dataset, userId: typeof userId === "number" ? userId : undefined }
  }

  const loadBackups = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listBackups()
      setBackups(
        tagDuplicateRowKeys(
          Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.backups ?? []
        )
      )
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
      setSchedules(
        tagDuplicateRowKeys(
          Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.schedules ?? []
        )
      )
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

  const handleRestore = async (target: ReturnType<typeof getRestoreTarget>) => {
    if (!target) return
    setRestoring(target.backupId)
    try {
      await tldwClient.restoreBackup(target.backupId, {
        dataset: target.dataset,
        user_id: target.userId,
        confirm: true
      })
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
      title: t("settings:adminDataOps.colId", "ID"),
      dataIndex: "id",
      key: "id",
      render: (v: string, record: { backup_id?: string }) => <code>{v ?? record.backup_id ?? "\u2014"}</code>
    },
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
      render: (v: string) => (v ? formatAdminDateTime(v) : "\u2014")
    },
    {
      title: t("settings:adminDataOps.colSize", "Size"),
      dataIndex: "size_bytes",
      key: "size_bytes",
      width: 100,
      render: (v: number | string, record: { size?: number | string }) => v ?? record.size ?? "\u2014"
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
      render: (_: any, record: any) => {
        const target = getRestoreTarget(record)
        return (
          <Popconfirm
            title={t("settings:adminDataOps.restoreConfirm", "Restore this backup? This will overwrite current data.")}
            description={target && (
              <div>
                <div>{t("settings:adminDataOps.backupArtifact", "Backup artifact")}: <code>{target.backupId}</code></div>
                <div>{t("settings:adminDataOps.colDataset", "Dataset")}: <strong>{target.dataset}</strong></div>
                <div>{target.dataset === "authnz"
                  ? t("settings:adminDataOps.restoreServerWide", "Server-wide users and authentication data")
                  : <>{t("settings:adminDataOps.colUser", "User")}: <strong>{target.userId}</strong></>}
                </div>
              </div>
            )}
            disabled={!target}
            onConfirm={() => handleRestore(target)}
            okText={t("settings:adminDataOps.restore", "Restore")}
            okButtonProps={{ danger: true }}
          >
            <Button
              type="text"
              size="small"
              icon={<UndoOutlined />}
              disabled={!target}
              loading={restoring === target?.backupId}
            >
              {t("settings:adminDataOps.restore", "Restore")}
            </Button>
          </Popconfirm>
        )
      }
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
          rowKey={(r) => r._rowKey ?? dataOpsRowKey(r)}
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
                  pattern: /^([01]\d|2[0-3]):[0-5]\d$/,
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
          rowKey={(r) => r._rowKey ?? dataOpsRowKey(r)}
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
      setDsrs(Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.requests ?? [])
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
      setPendingDsr({ ...values, client_request_id: crypto.randomUUID() })
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
        client_request_id: pendingDsr.client_request_id,
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
      dataIndex: "requested_at",
      key: "requested_at",
      width: 180,
      render: (v: string, record: { created_at?: string }) => {
        const value = v || record.created_at
        return value ? formatAdminDateTime(value) : "\u2014"
      }
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
                  { value: "access", label: t("settings:adminDataOps.typeAccess", "Access Request") }
                ]}
              />
            </Form.Item>
            <Form.Item name="categories" label={t("settings:adminDataOps.categoriesLabel", "Categories (optional)")}>
              <Select
                mode="multiple"
                placeholder={t("settings:adminDataOps.categoriesPlaceholder", "Select categories to include")}
                allowClear
                options={[
                  { value: "media_records", label: t("settings:adminDataOps.categoryMedia", "Media") },
                  { value: "chat_messages", label: t("settings:adminDataOps.categoryChats", "Chats") },
                  { value: "notes", label: t("settings:adminDataOps.categoryNotes", "Notes") },
                  { value: "audit_events", label: t("settings:adminDataOps.datasetAudit", "Audit log") },
                  { value: "embeddings", label: t("settings:adminDataOps.categoryEmbeddings", "Embeddings") }
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
        {/* Bounded snapshot (limit:100) — full server pagination waits on a
            truthful total; disclose the truncation instead of hiding it. */}
        <div style={{ marginBottom: 8 }}>
          <span
            style={{ color: "var(--color-text-secondary, #888)", fontSize: "0.85rem" }}
          >
            {t("settings:adminDataOps.showingFirst100Dsrs", "Showing first 100 requests")}
          </span>
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
  const { config, authorityLoading } = useCanonicalConnectionConfig()
  const { t } = useTranslation("settings")
  if (!config || authorityLoading) return <p role="status">{t("admin.resolvingConnection", "Resolving admin connection…")}</p>
  return <BundlesWorkspace key={deriveConnectionAuthorityId(config)} config={config} onGuardError={onGuardError} />
}

type BundlePreview = { file: File; userId?: number; result: BundleImportResult }
const BundlesWorkspace: React.FC<{ config: TldwConfig; onGuardError: (err: any) => void }> = ({ config, onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const captureScope = useAdminRequestScope(config)
  const { runMutation, mutating } = useAdminMutation(captureScope, onGuardError)
  const [importFile, setImportFile] = useState<File | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)
  const [importUserId, setImportUserId] = useState<number | null>(null)
  const [preview, setPreview] = useState<BundlePreview | null>(null)
  const [confirmation, setConfirmation] = useState<BundlePreview | null>(null)
  const [outcome, setOutcome] = useState<BundleImportResult | null>(null)
  const [restoreUnconfirmed, setRestoreUnconfirmed] = useState(false)
  const validUserId = importUserId !== null && Number.isSafeInteger(importUserId) && importUserId > 0
  const recognized = new Set([...PER_USER_BACKUP_DATASETS, "authnz"])
  const previewDatasets = preview?.result.validations?.map((item) => item.dataset) ?? []
  const postgresPreview = previewDatasets.includes("postgresql")
  const canRestore = Boolean(preview && !postgresPreview && preview.result.status === "compatible" &&
    previewDatasets.length && new Set(previewDatasets).size === previewDatasets.length &&
    preview.result.validations.every((item) => recognized.has(item.dataset) && item.compatible === true) &&
    (!previewDatasets.some((item) => PER_USER_BACKUP_DATASETS.has(item)) || validUserId))
  let targetServer = ""
  try {
    const url = new URL(config.serverUrl)
    targetServer = url.origin + url.pathname.replace(/\/$/, "")
  } catch { /* Invalid configuration is refused by the request transport. */ }
  const [bundles, setBundles] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [createForm] = Form.useForm()
  const [creating, setCreating] = useState(false)
  const datasetOptions = useBackupDatasetOptions(t)
  const selectedDatasets = Form.useWatch("datasets", createForm)
  const databaseSelected = Array.isArray(selectedDatasets) && selectedDatasets.includes("postgresql")
  const bundleDatasetOptions = [
    ...datasetOptions.map((option) => ({ ...option, disabled: databaseSelected })),
    { value: "postgresql", label: t("settings:adminDataOps.datasetPostgresql", "PostgreSQL database (all users and Jobs)"), disabled: Array.isArray(selectedDatasets) && selectedDatasets.some((dataset) => dataset !== "postgresql") }
  ]
  const needsUser = Array.isArray(selectedDatasets) && selectedDatasets.some((dataset) => PER_USER_BACKUP_DATASETS.has(dataset))

  const loadBundles = useCallback(async () => {
    const scope = captureScope()
    if (!scope) return
    setLoading(true)
    try {
      const result = await tldwClient.listBundles(scope)
      if (scope.abortSignal.aborted) return
      setBundles(tagDuplicateRowKeys(Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.bundles ?? []))
    } catch (err) {
      if (!scope.abortSignal.aborted) onGuardError(err)
    } finally {
      if (!scope.abortSignal.aborted) setLoading(false)
    }
  }, [captureScope, onGuardError])

  useEffect(() => {
    void loadBundles()
  }, [loadBundles])

  const handleCreate = () => runMutation(t("settings:adminDataOps.bundleCreateFailed", "Failed to create bundle"), async (scope) => {
    const values = await createForm.validateFields()
    if (scope.abortSignal.aborted) return
    const databaseScope = values.datasets.includes("postgresql")
    if (databaseScope && values.datasets.length !== 1) throw new Error(t("settings:adminDataOps.databaseExclusive", "Choose the PostgreSQL database by itself."))
    setCreating(true)
    try {
      await tldwClient.createBundle({
        datasets: values.datasets,
        user_id: values.datasets.some((dataset: string) => PER_USER_BACKUP_DATASETS.has(dataset)) ? values.user_id : undefined
      }, scope)
      if (scope.abortSignal.aborted) return
      message.success(t("settings:adminDataOps.bundleCreated", "Bundle created"))
      createForm.resetFields()
      void loadBundles()
    } catch (error) {
      if (databaseScope && deriveAdminGuardFromError(error) === "forbidden") {
        throw new Error(t("settings:adminDataOps.databasePermissionRequired", "Platform administrator access is required for whole-database backups."))
      }
      throw error
    } finally {
      if (!scope.abortSignal.aborted) setCreating(false)
    }
  })

  const handleDelete = (bundleId: string) => runMutation(t("settings:adminDataOps.bundleDeleteFailed", "Failed to delete bundle"), async (scope) => {
    await tldwClient.deleteBundle(bundleId, scope)
    if (scope.abortSignal.aborted) return
    message.success(t("settings:adminDataOps.bundleDeleted", "Bundle deleted"))
    void loadBundles()
  })

  const handleDownload = async (bundleId: string) => {
    const scope = captureScope()
    if (!scope) return
    try {
      const blob = await tldwClient.downloadBundle(bundleId, scope)
      if (scope.abortSignal.aborted) return
      downloadBlob(blob, bundleId)
      message.success(t("settings:adminDataOps.downloadStarted", "Download started"))
    } catch (error) {
      if (!scope.abortSignal.aborted) message.error(sanitizeAdminErrorMessage(error, t("settings:adminDataOps.downloadFailed", "Could not download this backup. Refresh the list and try again.")))
    }
  }
  const handleValidate = () => runMutation(t("settings:adminDataOps.validateFailed", "Backup validation failed"), async (scope) => {
    setPreview(null)
    setOutcome(null)
    setRestoreUnconfirmed(false)
    if (!importFile) throw new Error(t("settings:adminDataOps.backupFileRequired", "Choose a backup ZIP file"))
    if (importUserId !== null && !validUserId) throw new Error(t("settings:adminDataOps.bundleUserIdInvalid", "Enter a valid target user ID"))
    const result = await tldwClient.importBundle(importFile, { user_id: importUserId ?? undefined, dry_run: true }, scope).catch((error: unknown) => {
      if (error instanceof Error && /^(invalid_bundle|invalid_bundle_manifest)$/.test(sanitizeAdminErrorMessage(error, ""))) {
        error.message = t("settings:adminDataOps.invalidBackupFormat", "This backup is damaged or uses an unsupported format. Choose a complete backup ZIP from a compatible server.")
      }
      throw error
    })
    if (scope.abortSignal.aborted) return
    if (!result || typeof result.status !== "string" || !Array.isArray(result.validations) || !result.validations.every((item) => item && typeof item.dataset === "string" && typeof item.compatible === "boolean" && (item.message === undefined || typeof item.message === "string")) || !Array.isArray(result.warnings) || !result.warnings.every((item) => typeof item === "string")) {
      throw new Error(t("settings:adminDataOps.invalidValidation", "The server did not return a valid backup validation result."))
    }
    setPreview({ file: importFile, userId: importUserId ?? undefined, result })
  })
  const handleRestore = () => runMutation(t("settings:adminDataOps.restoreNotConfirmed", "Restore was not confirmed. Inspect the target data before retrying."), async (scope) => {
    if (!canRestore || !confirmation || confirmation !== preview || confirmation.file !== importFile || confirmation.userId !== (importUserId ?? undefined)) return
    const checked = confirmation
    setPreview(null)
    setRestoreUnconfirmed(true)
    const result = await tldwClient.importBundle(checked.file, { user_id: checked.userId, dry_run: false }, scope)
    if (scope.abortSignal.aborted) return
    if (!result || typeof result.status !== "string" || !Array.isArray(result.datasets_restored) || !result.datasets_restored.every((item) => typeof item === "string") || !Array.isArray(result.warnings) || !result.warnings.every((item) => typeof item === "string") || (result.rollback_failures !== undefined && (!Array.isArray(result.rollback_failures) || !result.rollback_failures.every((item) => typeof item === "string")))) {
      throw new Error(t("settings:adminDataOps.restoreNotConfirmed", "Restore was not confirmed. Inspect the target data before retrying."))
    }
    setOutcome(result)
    const expected = checked.result.validations.map((item) => item.dataset).sort()
    const actual = Array.isArray(result.datasets_restored) ? [...result.datasets_restored].sort() : []
    const complete = result.status === "imported" && expected.length === actual.length && expected.every((item, index) => item === actual[index]) && !result.rollback_failures?.length
    if (!complete) throw new Error(t("settings:adminDataOps.restoreIncomplete", "Restore is incomplete. Review the restored datasets and warnings before retrying."))
    setRestoreUnconfirmed(false)
    setConfirmation(null)
    setImportFile(null)
    if (fileInput.current) fileInput.current.value = ""
    message.success(t("settings:adminDataOps.backupRestored", "Backup restored successfully"))
    void loadBundles()
  })

  const bundleColumns = [
    {
      title: t("settings:adminDataOps.colId", "ID"),
      dataIndex: "bundle_id",
      key: "bundle_id",
      width: 200,
      render: (v: string, record: { id?: string }) => <code style={{ fontSize: 12 }}>{v ?? record.id ?? "\u2014"}</code>
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
      render: (v: string) => (v ? formatAdminDateTime(v) : "\u2014")
    },
    {
      title: t("settings:adminDataOps.colActions", "Actions"),
      key: "actions",
      width: 80,
      render: (_: any, record: any) => (
        <Space wrap>
        <Button type="text" size="small" icon={<DownloadOutlined />} aria-label={`${t("settings:adminDataOps.download", "Download")} ${record.bundle_id ?? record.id}`} onClick={() => handleDownload(record.bundle_id ?? record.id)} disabled={mutating} />
        <Popconfirm
          title={t("settings:adminDataOps.deleteBundleConfirm", "Delete this bundle?")}
          onConfirm={() => handleDelete(record.id ?? record.bundle_id)}
          okText={t("settings:adminDataOps.delete", "Delete")}
          okButtonProps={{ danger: true }}
        >
          <Button type="text" size="small" danger icon={<DeleteOutlined />} disabled={mutating} />
        </Popconfirm>
        </Space>
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
        <Form form={createForm} name="admin-bundle-create" layout="inline" disabled={mutating}>
          <Form.Item
            name="datasets"
            rules={[{ required: true, message: t("settings:adminDataOps.datasetsRequired", "Select at least one dataset") }]}
          >
            <Select
              mode="multiple"
              placeholder={t("settings:adminDataOps.datasetsPlaceholder", "Select datasets")}
              style={{ width: 320 }}
              options={bundleDatasetOptions}
            />
          </Form.Item>
          {needsUser && (
            <Form.Item
              name="user_id"
              label={t("settings:adminDataOps.bundleUserId", "Target user ID")}
              preserve={false}
              rules={[
                { required: true, message: t("settings:adminDataOps.bundleUserIdRequired", "Enter a target user ID") },
                { type: "integer", min: 1, max: Number.MAX_SAFE_INTEGER, message: t("settings:adminDataOps.bundleUserIdInvalid", "Enter a valid target user ID") }
              ]}
            >
              <InputNumber style={{ width: 160 }} />
            </Form.Item>
          )}
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
        {databaseSelected && <p role="note">{t("settings:adminDataOps.databaseCoverage", "Platform administrator only. Backs up the co-located PostgreSQL database, including all users, stored content and Jobs. Uploaded files, local SQLite, models and external stores are excluded. Recovery requires the offline helper.")}</p>}
      </div>
      <Table
        dataSource={bundles}
        columns={bundleColumns}
        rowKey={(r) => r._rowKey ?? dataOpsRowKey(r)}
        loading={loading}
        pagination={bundles.length > 20 ? { pageSize: 20 } : false}
        size="small"
      />
      <h3 style={{ marginTop: 24 }}>{t("settings:adminDataOps.restoreBundleTitle", "Restore a backup")}</h3>
      <Form layout="vertical">
        <Form.Item label={t("settings:adminDataOps.backupZipFile", "Backup ZIP file")} htmlFor="admin-bundle-file">
          <input id="admin-bundle-file" ref={fileInput} type="file" accept=".zip,application/zip" disabled={mutating} onChange={(event) => { setImportFile(event.target.files?.[0] ?? null); setPreview(null); setOutcome(null) }} />
        </Form.Item>
        <Form.Item label={t("settings:adminDataOps.restoreTargetId", "Restore target user ID")} htmlFor="admin-bundle-restore-user" extra={t("settings:adminDataOps.restoreTargetHelp", "Required for per-user datasets; leave blank for PostgreSQL database archives. Users and authentication affect the entire server.")}>
          <InputNumber id="admin-bundle-restore-user" value={importUserId} disabled={mutating} onChange={(value) => { setImportUserId(value); setPreview(null); setOutcome(null) }} />
        </Form.Item>
        <Space wrap>
          <Button onClick={handleValidate} disabled={!importFile || mutating} loading={mutating && !confirmation}>{t("settings:adminDataOps.validateBackup", "Validate backup")}</Button>
          <Button danger disabled={!canRestore || mutating} onClick={() => setConfirmation(preview)}>{t("settings:adminDataOps.restoreValidated", "Restore validated backup")}</Button>
        </Space>
      </Form>
      {preview && <div role="status" style={{ marginTop: 16, overflowWrap: "anywhere" }}>
        <p>{preview.file.name}</p>
        <p>{preview.result.status}</p>
        {preview.result.validations?.map((item) => <p key={item.dataset}><strong>{item.dataset}</strong>: {item.message}</p>)}
        {preview.result.warnings?.map((warning, index) => <p key={index}>{warning}</p>)}
        {postgresPreview && <section aria-label={t("settings:adminDataOps.databaseRecovery", "Offline PostgreSQL recovery")}>
          <h4>{t("settings:adminDataOps.databaseRecovery", "Offline PostgreSQL recovery")}</h4>
          <p>{t("settings:adminDataOps.targetServer", "Target server")}: {targetServer}</p>
          <p>{t("settings:adminDataOps.databaseOfflineSteps", "Stop all API, Jobs and Scheduler clients and prevent reconnection. Use a separate recovery target for rehearsal. This replaces the database for all users and stored Jobs.")}</p>
          <code>python Helper_Scripts/pg_backup_restore.py restore-bundle --bundle-file &lt;trusted-backup.zip&gt; --confirm-database &lt;target-database&gt; --confirm-offline</code>
          <p>{t("settings:adminDataOps.databaseRestartSteps", "After restore, verify data and review queued or leased Jobs. Explicitly decide which work to resume or cancel before a cold restart and normal sign-in.")}</p>
        </section>}
        {!canRestore && !postgresPreview && <p>{t("settings:adminDataOps.restoreNeedsValidScope", "Restore requires complete compatible validation and an explicit target for per-user data.")}</p>}
      </div>}
      {restoreUnconfirmed && <Alert variant="warning" title={t("settings:adminDataOps.restoreNotConfirmed", "Restore was not confirmed. Inspect the target data before retrying.")} />}
      {outcome && <div role="status" style={{ marginTop: 16, overflowWrap: "anywhere" }}>
        <p>{t("settings:adminDataOps.restoredDatasets", "Restored datasets")}: {outcome.datasets_restored?.join(", ") || "—"}</p>
        {outcome.warnings?.map((warning, index) => <p key={index}>{warning}</p>)}
        {outcome.rollback_failures?.map((failure, index) => <p key={index}>{failure}</p>)}
      </div>}
      <Modal title={t("settings:adminDataOps.restoreBundleConfirm", "Restore backup to this server?")} open={Boolean(confirmation)}
        onOk={handleRestore} onCancel={() => { if (!mutating) setConfirmation(null) }}
        okText={t("settings:adminDataOps.restoreBackup", "Restore backup")} cancelText={t("common:cancel", "Cancel")}
        okButtonProps={{ danger: true, disabled: !canRestore }} confirmLoading={mutating}
        cancelButtonProps={{ disabled: mutating }} closable={!mutating} maskClosable={!mutating} keyboard={!mutating}>
        <Descriptions column={1} size="small">
          <Descriptions.Item label={t("settings:adminDataOps.backupZipFile", "Backup ZIP file")}>{confirmation?.file.name}</Descriptions.Item>
          <Descriptions.Item label={t("settings:adminDataOps.targetServer", "Target server")}>{targetServer}</Descriptions.Item>
          <Descriptions.Item label={t("settings:adminDataOps.restoreTargetId", "Restore target user ID")}>{confirmation?.userId ? `User ${confirmation.userId}` : t("settings:adminDataOps.serverWide", "Server-wide")}</Descriptions.Item>
          <Descriptions.Item label={t("settings:adminDataOps.colDatasets", "Datasets")}>{confirmation?.result.validations.map((item) => item.dataset).join(", ")}</Descriptions.Item>
        </Descriptions>
        {confirmation?.result.validations.some((item) => item.dataset === "authnz") && <Alert variant="warning" title={t("settings:adminDataOps.restoreServerWide", "Server-wide users and authentication data")} />}
        <p>{t("settings:adminDataOps.restoreBundleWarning", "This replaces the selected data on this server. Use an isolated target server for a recovery rehearsal.")}</p>
      </Modal>
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
