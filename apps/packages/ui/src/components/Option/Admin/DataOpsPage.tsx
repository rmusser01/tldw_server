import React, { useState, useRef, useCallback, useEffect } from "react"
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

const BackupsTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const [backups, setBackups] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [createForm] = Form.useForm()
  const [creating, setCreating] = useState(false)
  const [restoring, setRestoring] = useState<string | null>(null)

  // Schedules
  const [schedules, setSchedules] = useState<any[]>([])
  const [schedulesLoading, setSchedulesLoading] = useState(false)
  const [scheduleForm] = Form.useForm()
  const [creatingSchedule, setCreatingSchedule] = useState(false)

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
      message.success("Backup created")
      createForm.resetFields()
      void loadBackups()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, "Failed to create backup"))
    } finally {
      setCreating(false)
    }
  }

  const handleRestore = async (backupId: string) => {
    setRestoring(backupId)
    try {
      await tldwClient.restoreBackup(backupId)
      message.success("Backup restored successfully")
      void loadBackups()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to restore backup"))
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
        cron: values.cron || undefined,
        retention_days: values.retention_days || undefined
      })
      message.success("Backup schedule created")
      scheduleForm.resetFields()
      void loadSchedules()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, "Failed to create schedule"))
    } finally {
      setCreatingSchedule(false)
    }
  }

  const handleDeleteSchedule = async (scheduleId: number) => {
    try {
      await tldwClient.deleteBackupSchedule(scheduleId)
      message.success("Schedule deleted")
      void loadSchedules()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to delete schedule"))
    }
  }

  const backupColumns = [
    {
      title: "Dataset",
      dataIndex: "dataset",
      key: "dataset",
      render: (v: string) => <Tag>{v || "unknown"}</Tag>
    },
    {
      title: "User",
      dataIndex: "user_id",
      key: "user_id",
      width: 80,
      render: (v: number) => v ?? "\u2014"
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      render: (v: string) => (v ? new Date(v).toLocaleString() : "\u2014")
    },
    {
      title: "Size",
      dataIndex: "size",
      key: "size",
      width: 100,
      render: (v: number | string) => v ?? "\u2014"
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (status: string) => {
        const color = status === "completed" ? "green" : status === "failed" ? "red" : "blue"
        return <Tag color={color}>{status || "unknown"}</Tag>
      }
    },
    {
      title: "Actions",
      key: "actions",
      width: 100,
      render: (_: any, record: any) => (
        <Popconfirm
          title="Restore this backup? This will overwrite current data."
          onConfirm={() => handleRestore(record.id ?? record.backup_id)}
          okText="Restore"
          okButtonProps={{ danger: true }}
        >
          <Button
            type="text"
            size="small"
            icon={<UndoOutlined />}
            loading={restoring === (record.id ?? record.backup_id)}
          >
            Restore
          </Button>
        </Popconfirm>
      )
    }
  ]

  const scheduleColumns = [
    {
      title: "Dataset",
      dataIndex: "dataset",
      key: "dataset",
      render: (v: string) => <Tag>{v}</Tag>
    },
    {
      title: "Cron",
      dataIndex: "cron",
      key: "cron",
      render: (v: string) => <code>{v || "\u2014"}</code>
    },
    {
      title: "Retention (days)",
      dataIndex: "retention_days",
      key: "retention_days",
      width: 130,
      render: (v: number) => v ?? "\u2014"
    },
    {
      title: "Actions",
      key: "actions",
      width: 80,
      render: (_: any, record: any) => (
        <Popconfirm
          title="Delete this schedule?"
          onConfirm={() => handleDeleteSchedule(record.id ?? record.schedule_id)}
          okText="Delete"
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
        title="Backups"
        style={{ marginBottom: 16 }}
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={() => loadBackups()}>
            Refresh
          </Button>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Form form={createForm} layout="inline">
            <Form.Item
              name="dataset"
              rules={[{ required: true, message: "Dataset is required" }]}
            >
              <Select
                placeholder="Dataset"
                style={{ width: 180 }}
                options={[
                  { value: "media", label: "Media" },
                  { value: "chachanotes", label: "ChaChaNotes" },
                  { value: "users", label: "Users" },
                  { value: "evaluations", label: "Evaluations" }
                ]}
              />
            </Form.Item>
            <Form.Item name="user_id">
              <InputNumber placeholder="User ID (optional)" min={1} style={{ width: 160 }} />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleCreateBackup}
                loading={creating}
              >
                Create Backup
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
        />
      </Card>

      <Card
        title="Backup Schedules"
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={() => loadSchedules()}>
            Refresh
          </Button>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Form form={scheduleForm} layout="inline">
            <Form.Item
              name="dataset"
              rules={[{ required: true, message: "Dataset is required" }]}
            >
              <Select
                placeholder="Dataset"
                style={{ width: 160 }}
                options={[
                  { value: "media", label: "Media" },
                  { value: "chachanotes", label: "ChaChaNotes" },
                  { value: "users", label: "Users" },
                  { value: "evaluations", label: "Evaluations" }
                ]}
              />
            </Form.Item>
            <Form.Item name="cron">
              <Input placeholder="Cron (e.g. 0 2 * * *)" style={{ width: 180 }} />
            </Form.Item>
            <Form.Item name="retention_days">
              <InputNumber placeholder="Retention days" min={1} style={{ width: 140 }} />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={handleCreateSchedule}
                loading={creatingSchedule}
              >
                Add Schedule
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
        />
      </Card>
    </div>
  )
}

// ── DSR Tab ──

const DsrTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
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
      message.error(sanitizeAdminErrorMessage(err, "Failed to preview DSR"))
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
      message.success("DSR recorded")
      setPreviewModalOpen(false)
      setPendingDsr(null)
      setPreviewData(null)
      createForm.resetFields()
      void loadDsrs()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to record DSR"))
    } finally {
      setCreating(false)
    }
  }

  const handleExecute = async (requestId: number) => {
    setExecuting(requestId)
    try {
      await tldwClient.executeDsr(requestId)
      message.success("DSR executed")
      void loadDsrs()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to execute DSR"))
    } finally {
      setExecuting(null)
    }
  }

  const dsrColumns = [
    {
      title: "ID",
      dataIndex: "id",
      key: "id",
      width: 60
    },
    {
      title: "Requester",
      dataIndex: "requester_identifier",
      key: "requester_identifier"
    },
    {
      title: "Type",
      dataIndex: "request_type",
      key: "request_type",
      width: 100,
      render: (v: string) => <Tag>{v || "erasure"}</Tag>
    },
    {
      title: "Status",
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
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      render: (v: string) => (v ? new Date(v).toLocaleString() : "\u2014")
    },
    {
      title: "Actions",
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
            title="Execute this DSR? This action cannot be undone."
            onConfirm={() => handleExecute(record.id)}
            okText="Execute"
            okButtonProps={{ danger: true }}
          >
            <Button
              type="text"
              size="small"
              danger
              icon={<PlayCircleOutlined />}
              loading={executing === record.id}
            >
              Execute
            </Button>
          </Popconfirm>
        )
      }
    }
  ]

  return (
    <div>
      <Card
        title="Data Subject Requests"
        style={{ marginBottom: 16 }}
        extra={
          <Button size="small" icon={<ReloadOutlined />} onClick={() => loadDsrs()}>
            Refresh
          </Button>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Form form={createForm} layout="vertical" style={{ maxWidth: 600 }}>
            <Form.Item
              name="requester_identifier"
              label="Requester Identifier"
              rules={[{ required: true, message: "Requester identifier is required" }]}
            >
              <Input placeholder="Email, username, or user ID" />
            </Form.Item>
            <Form.Item name="request_type" label="Request Type" initialValue="erasure">
              <Select
                options={[
                  { value: "erasure", label: "Erasure (Right to be Forgotten)" },
                  { value: "export", label: "Data Export" },
                  { value: "access", label: "Access Request" },
                  { value: "rectification", label: "Rectification" }
                ]}
              />
            </Form.Item>
            <Form.Item name="categories" label="Categories (optional)">
              <Select
                mode="multiple"
                placeholder="Select categories to include"
                allowClear
                options={[
                  { value: "media", label: "Media" },
                  { value: "chats", label: "Chats" },
                  { value: "notes", label: "Notes" },
                  { value: "embeddings", label: "Embeddings" },
                  { value: "profile", label: "Profile" }
                ]}
              />
            </Form.Item>
            <Form.Item name="notes" label="Notes (optional)">
              <Input.TextArea rows={2} placeholder="Internal notes about this request" />
            </Form.Item>
            <Form.Item>
              <Space>
                <Button
                  icon={<EyeOutlined />}
                  onClick={handlePreview}
                  loading={previewLoading}
                >
                  Preview
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
        title="DSR Preview"
        open={previewModalOpen}
        onCancel={() => { setPreviewModalOpen(false); setPreviewData(null); setPendingDsr(null) }}
        footer={[
          <Button key="cancel" onClick={() => { setPreviewModalOpen(false); setPreviewData(null); setPendingDsr(null) }}>
            Cancel
          </Button>,
          <Button key="record" type="primary" onClick={handleRecordDsr} loading={creating}>
            Record DSR
          </Button>
        ]}
        width={600}
      >
        {previewData && (
          <div>
            <p style={{ marginBottom: 12 }}>
              The following data was found for <strong>{pendingDsr?.requester_identifier}</strong>:
            </p>
            <Descriptions bordered size="small" column={1}>
              {Object.entries(previewData?.counts ?? previewData?.data ?? previewData ?? {}).map(
                ([key, value]) => (
                  <Descriptions.Item key={key} label={key}>
                    {typeof value === "number" ? `${value} record(s)` : String(value ?? "\u2014")}
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
      message.success(`Retention policy "${policyKey}" updated`)
      setEditingKey(null)
      void loadPolicies()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to update retention policy"))
    } finally {
      setSaving(false)
    }
  }

  const policyColumns = [
    {
      title: "Policy Key",
      dataIndex: "key",
      key: "key",
      render: (v: string) => <code>{v}</code>
    },
    {
      title: "Retention (days)",
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
                Save
              </Button>
              <Button size="small" onClick={() => setEditingKey(null)}>
                Cancel
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
              Edit
            </Button>
          </Space>
        )
      }
    }
  ]

  return (
    <Card
      title="Retention Policies"
      extra={
        <Button size="small" icon={<ReloadOutlined />} onClick={() => loadPolicies()}>
          Refresh
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
      message.success("Bundle created")
      createForm.resetFields()
      void loadBundles()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, "Failed to create bundle"))
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (bundleId: string) => {
    try {
      await tldwClient.deleteBundle(bundleId)
      message.success("Bundle deleted")
      void loadBundles()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to delete bundle"))
    }
  }

  const bundleColumns = [
    {
      title: "ID",
      dataIndex: "id",
      key: "id",
      width: 200,
      render: (v: string) => <code style={{ fontSize: 12 }}>{v}</code>
    },
    {
      title: "Datasets",
      dataIndex: "datasets",
      key: "datasets",
      render: (datasets: string[]) =>
        Array.isArray(datasets)
          ? datasets.map((d) => <Tag key={d}>{d}</Tag>)
          : String(datasets ?? "\u2014")
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      width: 180,
      render: (v: string) => (v ? new Date(v).toLocaleString() : "\u2014")
    },
    {
      title: "Actions",
      key: "actions",
      width: 80,
      render: (_: any, record: any) => (
        <Popconfirm
          title="Delete this bundle?"
          onConfirm={() => handleDelete(record.id ?? record.bundle_id)}
          okText="Delete"
          okButtonProps={{ danger: true }}
        >
          <Button type="text" size="small" danger icon={<DeleteOutlined />} />
        </Popconfirm>
      )
    }
  ]

  return (
    <Card
      title="Backup Bundles"
      extra={
        <Button size="small" icon={<ReloadOutlined />} onClick={() => loadBundles()}>
          Refresh
        </Button>
      }
    >
      <div style={{ marginBottom: 16 }}>
        <Form form={createForm} layout="inline">
          <Form.Item
            name="datasets"
            rules={[{ required: true, message: "Select at least one dataset" }]}
          >
            <Select
              mode="multiple"
              placeholder="Select datasets"
              style={{ width: 320 }}
              options={[
                { value: "media", label: "Media" },
                { value: "chachanotes", label: "ChaChaNotes" },
                { value: "users", label: "Users" },
                { value: "evaluations", label: "Evaluations" }
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
              Create Bundle
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
      <Alert variant="error" title="Access Denied">
        You don't have permission to access data operations.
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title="Not Available">
        Data operations are not available on this server.
      </Alert>
    )
  }

  const tabItems = [
    {
      key: "backups",
      label: "Backups",
      children: <BackupsTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "dsr",
      label: "Data Subject Requests",
      children: <DsrTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "retention",
      label: "Retention Policies",
      children: <RetentionPoliciesTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "bundles",
      label: "Bundles",
      children: <BundlesTab onGuardError={markAdminGuardFromError} />
    }
  ]

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h2 style={{ marginBottom: 16 }}>Data Operations</h2>
      <Tabs items={tabItems} defaultActiveKey="backups" />
    </div>
  )
}

export default DataOpsPage
