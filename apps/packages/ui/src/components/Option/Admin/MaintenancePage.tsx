import React, { useState, useRef, useCallback, useEffect } from "react"
import {
  Card,
  Table,
  Button,
  Input,
  Form,
  Tag,
  Space,
  Select,
  Switch,
  Popconfirm,
  Collapse,
  message
} from "antd"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { Alert as DesignSystemAlert } from "@/components/ui/primitives"

const { TextArea } = Input

const MaintenancePage: React.FC = () => {
  // Admin guard state
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // Maintenance mode state
  const [maintEnabled, setMaintEnabled] = useState(false)
  const [maintMessage, setMaintMessage] = useState("")
  const [maintAllowlist, setMaintAllowlist] = useState("")
  const [maintLoading, setMaintLoading] = useState(false)
  const [maintSaving, setMaintSaving] = useState(false)

  // Feature flags state
  const [flags, setFlags] = useState<any[]>([])
  const [flagsLoading, setFlagsLoading] = useState(false)

  // Incidents state
  const [incidents, setIncidents] = useState<any[]>([])
  const [incidentsLoading, setIncidentsLoading] = useState(false)
  const [incidentForm] = Form.useForm()
  const [creatingIncident, setCreatingIncident] = useState(false)

  // Rotation runs state
  const [rotationRuns, setRotationRuns] = useState<any[]>([])
  const [rotationLoading, setRotationLoading] = useState(false)
  const [startingRotation, setStartingRotation] = useState(false)

  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  // ── Maintenance Mode ──

  const loadMaintenanceState = useCallback(async () => {
    setMaintLoading(true)
    try {
      const state = await tldwClient.getMaintenanceState()
      setMaintEnabled(!!state?.enabled)
      setMaintMessage(state?.message || "")
      setMaintAllowlist((state?.allowlist || []).join(", "))
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setMaintLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleSaveMaintenanceState = async () => {
    setMaintSaving(true)
    try {
      const allowlistArr = maintAllowlist
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
      await tldwClient.updateMaintenanceState({
        enabled: maintEnabled,
        message: maintMessage || undefined,
        allowlist: allowlistArr.length > 0 ? allowlistArr : undefined
      })
      message.success("Maintenance state updated")
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to update maintenance state"))
    } finally {
      setMaintSaving(false)
    }
  }

  // ── Feature Flags ──

  const loadFeatureFlags = useCallback(async () => {
    setFlagsLoading(true)
    try {
      const result = await tldwClient.listFeatureFlags()
      setFlags(Array.isArray(result) ? result : [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setFlagsLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleToggleFlag = async (flagKey: string, enabled: boolean) => {
    try {
      await tldwClient.updateFeatureFlag(flagKey, { enabled })
      message.success(`Flag "${flagKey}" ${enabled ? "enabled" : "disabled"}`)
      await loadFeatureFlags()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to update feature flag"))
    }
  }

  const handleDeleteFlag = async (flagKey: string) => {
    try {
      await tldwClient.deleteFeatureFlag(flagKey)
      message.success(`Flag "${flagKey}" deleted`)
      await loadFeatureFlags()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to delete feature flag"))
    }
  }

  // ── Incidents ──

  const loadIncidents = useCallback(async () => {
    setIncidentsLoading(true)
    try {
      const result = await tldwClient.listIncidents()
      setIncidents(Array.isArray(result) ? result : [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setIncidentsLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleCreateIncident = async () => {
    try {
      const values = await incidentForm.validateFields()
      setCreatingIncident(true)
      await tldwClient.createIncident({
        title: values.title.trim(),
        severity: values.severity || undefined,
        description: values.description?.trim() || undefined
      })
      incidentForm.resetFields()
      message.success("Incident created")
      await loadIncidents()
    } catch (err: any) {
      if (err?.errorFields) return // form validation error
      message.error(sanitizeAdminErrorMessage(err, "Failed to create incident"))
    } finally {
      setCreatingIncident(false)
    }
  }

  const handleUpdateIncidentStatus = async (incidentId: number, status: string) => {
    try {
      await tldwClient.updateIncident(incidentId, { status })
      message.success("Incident updated")
      await loadIncidents()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to update incident"))
    }
  }

  const handleDeleteIncident = async (incidentId: number) => {
    try {
      await tldwClient.deleteIncident(incidentId)
      message.success("Incident deleted")
      await loadIncidents()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to delete incident"))
    }
  }

  // ── Rotation Runs ──

  const loadRotationRuns = useCallback(async () => {
    setRotationLoading(true)
    try {
      const result = await tldwClient.listRotationRuns()
      setRotationRuns(Array.isArray(result) ? result : [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setRotationLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleStartRotation = async () => {
    setStartingRotation(true)
    try {
      await tldwClient.createRotationRun()
      message.success("Rotation run started")
      await loadRotationRuns()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to start rotation run"))
    } finally {
      setStartingRotation(false)
    }
  }

  // ── Initial Load ──

  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    void loadMaintenanceState()
    void loadFeatureFlags()
    void loadIncidents()
    void loadRotationRuns()
  }, [loadMaintenanceState, loadFeatureFlags, loadIncidents, loadRotationRuns])

  // ── Feature Flags Table Columns ──

  const flagColumns = [
    {
      title: "Key",
      dataIndex: "key",
      key: "key",
      render: (key: string) => <code>{key}</code>
    },
    {
      title: "Enabled",
      dataIndex: "enabled",
      key: "enabled",
      render: (enabled: boolean, record: any) => (
        <Switch
          checked={!!enabled}
          onChange={(checked) => handleToggleFlag(record.key, checked)}
        />
      )
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: any) => (
        <Popconfirm
          title={`Delete flag "${record.key}"?`}
          onConfirm={() => handleDeleteFlag(record.key)}
        >
          <Button size="small" danger>Delete</Button>
        </Popconfirm>
      )
    }
  ]

  // ── Incidents Table Columns ──

  const incidentColumns = [
    {
      title: "Title",
      dataIndex: "title",
      key: "title"
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: string) => {
        const color = status === "resolved" ? "green" : status === "investigating" ? "orange" : "blue"
        return <Tag color={color}>{status || "open"}</Tag>
      }
    },
    {
      title: "Severity",
      dataIndex: "severity",
      key: "severity",
      render: (severity: string) => {
        const color = severity === "critical" ? "red" : severity === "high" ? "orange" : severity === "medium" ? "gold" : "default"
        return <Tag color={color}>{severity || "low"}</Tag>
      }
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014"
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: any) => (
        <Space size="small">
          <Select
            size="small"
            placeholder="Update status"
            style={{ width: 140 }}
            onChange={(val) => handleUpdateIncidentStatus(record.id, val)}
            options={[
              { value: "investigating", label: "Investigating" },
              { value: "identified", label: "Identified" },
              { value: "monitoring", label: "Monitoring" },
              { value: "resolved", label: "Resolved" }
            ]}
          />
          <Popconfirm
            title="Delete this incident?"
            onConfirm={() => handleDeleteIncident(record.id)}
          >
            <Button size="small" danger>Delete</Button>
          </Popconfirm>
        </Space>
      )
    }
  ]

  // ── Rotation Runs Table Columns ──

  const rotationColumns = [
    {
      title: "ID",
      dataIndex: "id",
      key: "id"
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: string) => {
        const color = status === "completed" ? "green" : status === "running" ? "blue" : status === "failed" ? "red" : "default"
        return <Tag color={color}>{status || "unknown"}</Tag>
      }
    },
    {
      title: "Started",
      dataIndex: "started_at",
      key: "started_at",
      render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014"
    },
    {
      title: "Completed",
      dataIndex: "completed_at",
      key: "completed_at",
      render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014"
    }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return (
      <DesignSystemAlert variant="error" title="Access Denied">
        You don't have permission to access the maintenance console.
      </DesignSystemAlert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <DesignSystemAlert variant="warning" title="Not Available">
        The maintenance console is not available on this server.
      </DesignSystemAlert>
    )
  }

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h2 style={{ marginBottom: 16 }}>Maintenance Console</h2>

      {/* Maintenance Mode Card */}
      <Card title="Maintenance Mode" loading={maintLoading} style={{ marginBottom: 16 }}>
        <Space orientation="vertical" style={{ width: "100%" }} size="middle">
          <Space>
            <span>Maintenance Mode:</span>
            <Switch
              checked={maintEnabled}
              onChange={(checked) => setMaintEnabled(checked)}
              checkedChildren="ON"
              unCheckedChildren="OFF"
            />
          </Space>
          <div>
            <label style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>Message:</label>
            <TextArea
              rows={2}
              value={maintMessage}
              onChange={(e) => setMaintMessage(e.target.value)}
              placeholder="Maintenance message displayed to users..."
            />
          </div>
          <div>
            <label style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>Allowlist (comma-separated IPs or usernames):</label>
            <Input
              value={maintAllowlist}
              onChange={(e) => setMaintAllowlist(e.target.value)}
              placeholder="e.g. 192.168.1.1, admin"
            />
          </div>
          <Button type="primary" onClick={handleSaveMaintenanceState} loading={maintSaving}>
            Save Changes
          </Button>
        </Space>
      </Card>

      {/* Feature Flags Card */}
      <Card title="Feature Flags" style={{ marginBottom: 16 }}>
        <Table
          dataSource={flags}
          columns={flagColumns}
          rowKey="key"
          loading={flagsLoading}
          pagination={false}
          size="small"
        />
      </Card>

      {/* Incidents Card */}
      <Card
        title="Incidents"
        style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Button onClick={() => loadIncidents()} size="small">
              Refresh
            </Button>
          </Space>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Form form={incidentForm} layout="inline">
            <Form.Item
              name="title"
              rules={[{ required: true, message: "Title is required" }]}
            >
              <Input placeholder="Incident title" style={{ width: 200 }} />
            </Form.Item>
            <Form.Item name="severity">
              <Select
                placeholder="Severity"
                style={{ width: 130 }}
                allowClear
                options={[
                  { value: "low", label: "Low" },
                  { value: "medium", label: "Medium" },
                  { value: "high", label: "High" },
                  { value: "critical", label: "Critical" }
                ]}
              />
            </Form.Item>
            <Form.Item name="description">
              <Input placeholder="Description (optional)" style={{ width: 200 }} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" onClick={handleCreateIncident} loading={creatingIncident}>
                Create Incident
              </Button>
            </Form.Item>
          </Form>
        </div>
        <Table
          dataSource={incidents}
          columns={incidentColumns}
          rowKey="id"
          loading={incidentsLoading}
          pagination={false}
          size="small"
        />
      </Card>

      {/* Rotation Runs (Collapsible) */}
      <Collapse
        items={[
          {
            key: "rotation-runs",
            label: "Rotation Runs",
            children: (
              <>
                <Space style={{ marginBottom: 12 }}>
                  <Button
                    type="primary"
                    onClick={handleStartRotation}
                    loading={startingRotation}
                  >
                    Start Run
                  </Button>
                  <Button onClick={() => loadRotationRuns()} size="small">
                    Refresh
                  </Button>
                </Space>
                <Table
                  dataSource={rotationRuns}
                  columns={rotationColumns}
                  rowKey="id"
                  loading={rotationLoading}
                  pagination={false}
                  size="small"
                />
              </>
            )
          }
        ]}
      />
    </div>
  )
}

export default MaintenancePage
