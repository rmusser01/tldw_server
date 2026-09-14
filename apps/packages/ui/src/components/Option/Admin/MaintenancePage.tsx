import React, { useState, useRef, useCallback, useEffect } from "react"
import { useTranslation } from "react-i18next"
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
import { Alert } from "@/components/ui/primitives"

const { TextArea } = Input

const MaintenancePage: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
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
      message.success(t("settings:adminMaintenance.maintUpdated", "Maintenance state updated"))
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMaintenance.maintUpdateFailed", "Failed to update maintenance state")))
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
      message.success(
        t("settings:adminMaintenance.flagToggled", {
          defaultValue: 'Flag "{{flag}}" {{state}}',
          flag: flagKey,
          state: enabled
            ? t("settings:adminMaintenance.flagStateEnabled", "enabled")
            : t("settings:adminMaintenance.flagStateDisabled", "disabled")
        })
      )
      await loadFeatureFlags()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMaintenance.flagUpdateFailed", "Failed to update feature flag")))
    }
  }

  const handleDeleteFlag = async (flagKey: string) => {
    try {
      await tldwClient.deleteFeatureFlag(flagKey)
      message.success(
        t("settings:adminMaintenance.flagDeleted", {
          defaultValue: 'Flag "{{flag}}" deleted',
          flag: flagKey
        })
      )
      await loadFeatureFlags()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMaintenance.flagDeleteFailed", "Failed to delete feature flag")))
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
      message.success(t("settings:adminMaintenance.incidentCreated", "Incident created"))
      await loadIncidents()
    } catch (err: any) {
      if (err?.errorFields) return // form validation error
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMaintenance.incidentCreateFailed", "Failed to create incident")))
    } finally {
      setCreatingIncident(false)
    }
  }

  const handleUpdateIncidentStatus = async (incidentId: number, status: string) => {
    try {
      await tldwClient.updateIncident(incidentId, { status })
      message.success(t("settings:adminMaintenance.incidentUpdated", "Incident updated"))
      await loadIncidents()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMaintenance.incidentUpdateFailed", "Failed to update incident")))
    }
  }

  const handleDeleteIncident = async (incidentId: number) => {
    try {
      await tldwClient.deleteIncident(incidentId)
      message.success(t("settings:adminMaintenance.incidentDeleted", "Incident deleted"))
      await loadIncidents()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMaintenance.incidentDeleteFailed", "Failed to delete incident")))
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
      message.success(t("settings:adminMaintenance.rotationStarted", "Rotation run started"))
      await loadRotationRuns()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMaintenance.rotationStartFailed", "Failed to start rotation run")))
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
      title: t("settings:adminMaintenance.colKey", "Key"),
      dataIndex: "key",
      key: "key",
      render: (key: string) => <code>{key}</code>
    },
    {
      title: t("settings:adminMaintenance.colEnabled", "Enabled"),
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
      title: t("settings:adminMaintenance.colActions", "Actions"),
      key: "actions",
      render: (_: any, record: any) => (
        <Popconfirm
          title={t("settings:adminMaintenance.deleteFlagConfirm", {
            defaultValue: 'Delete flag "{{flag}}"?',
            flag: record.key
          })}
          onConfirm={() => handleDeleteFlag(record.key)}
        >
          <Button size="small" danger>{t("common:delete", "Delete")}</Button>
        </Popconfirm>
      )
    }
  ]

  // ── Incidents Table Columns ──

  const incidentColumns = [
    {
      title: t("settings:adminMaintenance.colTitle", "Title"),
      dataIndex: "title",
      key: "title"
    },
    {
      title: t("settings:adminMaintenance.colStatus", "Status"),
      dataIndex: "status",
      key: "status",
      render: (status: string) => {
        const color = status === "resolved" ? "green" : status === "investigating" ? "orange" : "blue"
        return <Tag color={color}>{status || t("settings:adminMaintenance.statusOpen", "open")}</Tag>
      }
    },
    {
      title: t("settings:adminMaintenance.colSeverity", "Severity"),
      dataIndex: "severity",
      key: "severity",
      render: (severity: string) => {
        const color = severity === "critical" ? "red" : severity === "high" ? "orange" : severity === "medium" ? "gold" : "default"
        return <Tag color={color}>{severity || t("settings:adminMaintenance.severityLow", "low")}</Tag>
      }
    },
    {
      title: t("settings:adminMaintenance.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014"
    },
    {
      title: t("settings:adminMaintenance.colActions", "Actions"),
      key: "actions",
      render: (_: any, record: any) => (
        <Space size="small">
          <Select
            size="small"
            placeholder={t("settings:adminMaintenance.updateStatusPlaceholder", "Update status")}
            style={{ width: 140 }}
            onChange={(val) => handleUpdateIncidentStatus(record.id, val)}
            options={[
              { value: "investigating", label: t("settings:adminMaintenance.statusInvestigating", "Investigating") },
              { value: "identified", label: t("settings:adminMaintenance.statusIdentified", "Identified") },
              { value: "monitoring", label: t("settings:adminMaintenance.statusMonitoring", "Monitoring") },
              { value: "resolved", label: t("settings:adminMaintenance.statusResolved", "Resolved") }
            ]}
          />
          <Popconfirm
            title={t("settings:adminMaintenance.deleteIncidentConfirm", "Delete this incident?")}
            onConfirm={() => handleDeleteIncident(record.id)}
          >
            <Button size="small" danger>{t("common:delete", "Delete")}</Button>
          </Popconfirm>
        </Space>
      )
    }
  ]

  // ── Rotation Runs Table Columns ──

  const rotationColumns = [
    {
      title: t("settings:adminMaintenance.colId", "ID"),
      dataIndex: "id",
      key: "id"
    },
    {
      title: t("settings:adminMaintenance.colStatus", "Status"),
      dataIndex: "status",
      key: "status",
      render: (status: string) => {
        const color = status === "completed" ? "green" : status === "running" ? "blue" : status === "failed" ? "red" : "default"
        return <Tag color={color}>{status || t("settings:adminMaintenance.statusUnknown", "unknown")}</Tag>
      }
    },
    {
      title: t("settings:adminMaintenance.colStarted", "Started"),
      dataIndex: "started_at",
      key: "started_at",
      render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014"
    },
    {
      title: t("settings:adminMaintenance.colCompleted", "Completed"),
      dataIndex: "completed_at",
      key: "completed_at",
      render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014"
    }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title={t("settings:adminMaintenance.forbiddenTitle", "Access Denied")}>
        {t(
          "settings:adminMaintenance.forbiddenBody",
          "You don't have permission to access the maintenance console."
        )}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title={t("settings:adminMaintenance.notFoundTitle", "Not Available")}>
        {t(
          "settings:adminMaintenance.notFoundBody",
          "The maintenance console is not available on this server."
        )}
      </Alert>
    )
  }

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h1 style={{ marginBottom: 16, fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminMaintenance.title", "Maintenance Console")}</h1>

      {/* Maintenance Mode Card */}
      <Card title={t("settings:adminMaintenance.maintCardTitle", "Maintenance Mode")} loading={maintLoading} style={{ marginBottom: 16 }}>
        <Space orientation="vertical" style={{ width: "100%" }} size="middle">
          <Space>
            <span>{t("settings:adminMaintenance.maintToggleLabel", "Maintenance Mode:")}</span>
            <Switch
              checked={maintEnabled}
              onChange={(checked) => setMaintEnabled(checked)}
              checkedChildren={t("settings:adminMaintenance.maintOn", "ON")}
              unCheckedChildren={t("settings:adminMaintenance.maintOff", "OFF")}
            />
          </Space>
          <div>
            <label style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>{t("settings:adminMaintenance.maintMessageLabel", "Message:")}</label>
            <TextArea
              rows={2}
              value={maintMessage}
              onChange={(e) => setMaintMessage(e.target.value)}
              placeholder={t("settings:adminMaintenance.maintMessagePlaceholder", "Maintenance message displayed to users...")}
            />
          </div>
          <div>
            <label style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>{t("settings:adminMaintenance.maintAllowlistLabel", "Allowlist (comma-separated IPs or usernames):")}</label>
            <Input
              value={maintAllowlist}
              onChange={(e) => setMaintAllowlist(e.target.value)}
              placeholder={t("settings:adminMaintenance.maintAllowlistPlaceholder", "e.g. 192.168.1.1, admin")}
            />
          </div>
          <Button type="primary" onClick={handleSaveMaintenanceState} loading={maintSaving}>
            {t("settings:adminMaintenance.saveChanges", "Save Changes")}
          </Button>
        </Space>
      </Card>

      {/* Feature Flags Card */}
      <Card title={t("settings:adminMaintenance.flagsCardTitle", "Feature Flags")} style={{ marginBottom: 16 }}>
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
        title={t("settings:adminMaintenance.incidentsCardTitle", "Incidents")}
        style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Button onClick={() => loadIncidents()} size="small">
              {t("common:refresh", "Refresh")}
            </Button>
          </Space>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Form form={incidentForm} layout="inline">
            <Form.Item
              name="title"
              rules={[{ required: true, message: t("settings:adminMaintenance.titleRequired", "Title is required") }]}
            >
              <Input placeholder={t("settings:adminMaintenance.incidentTitlePlaceholder", "Incident title")} style={{ width: 200 }} />
            </Form.Item>
            <Form.Item name="severity">
              <Select
                placeholder={t("settings:adminMaintenance.severityPlaceholder", "Severity")}
                style={{ width: 130 }}
                allowClear
                options={[
                  { value: "low", label: t("settings:adminMaintenance.severityOptionLow", "Low") },
                  { value: "medium", label: t("settings:adminMaintenance.severityOptionMedium", "Medium") },
                  { value: "high", label: t("settings:adminMaintenance.severityOptionHigh", "High") },
                  { value: "critical", label: t("settings:adminMaintenance.severityOptionCritical", "Critical") }
                ]}
              />
            </Form.Item>
            <Form.Item name="description">
              <Input placeholder={t("settings:adminMaintenance.descriptionPlaceholder", "Description (optional)")} style={{ width: 200 }} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" onClick={handleCreateIncident} loading={creatingIncident}>
                {t("settings:adminMaintenance.createIncident", "Create Incident")}
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
            label: t("settings:adminMaintenance.rotationRunsLabel", "Rotation Runs"),
            children: (
              <>
                <Space style={{ marginBottom: 12 }}>
                  <Button
                    type="primary"
                    onClick={handleStartRotation}
                    loading={startingRotation}
                  >
                    {t("settings:adminMaintenance.startRun", "Start Run")}
                  </Button>
                  <Button onClick={() => loadRotationRuns()} size="small">
                    {t("common:refresh", "Refresh")}
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
