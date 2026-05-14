import React, { useState, useRef, useCallback, useEffect, useMemo } from "react"
import {
  AutoComplete,
  Card,
  Table,
  Descriptions,
  Button,
  InputNumber,
  Form,
  Tag,
  Space,
  Alert,
  Select,
  Switch,
  Popconfirm,
  Tooltip,
  Typography,
  message
} from "antd"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { CollapsibleSection } from "./CollapsibleSection"
import {
  tldwClient,
  type SandboxAdminRuntimeDiagnosticsItem,
  type SandboxAdminRuntimeDiagnosticsResponse
} from "@/services/tldw/TldwApiClient"
import { getDesignSystemState } from "@/design-system"

/** Format a stat value for display — handles objects, arrays, booleans, numbers */
function formatStatValue(value: unknown): React.ReactNode {
  if (value === null || value === undefined) return "\u2014"
  if (typeof value === "boolean") return value ? "Yes" : "No"
  if (typeof value === "number") return value.toLocaleString()
  if (typeof value === "string") return value || "\u2014"
  if (Array.isArray(value)) {
    if (value.length === 0) return "(empty)"
    return (
      <ul style={{ margin: 0, paddingLeft: 16, listStyle: "disc" }}>
        {value.map((item, i) => (
          <li key={i} style={{ fontSize: 12 }}>{formatStatValue(item)}</li>
        ))}
      </ul>
    )
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
    if (entries.length === 0) return "(empty)"
    return (
      <dl style={{ margin: 0 }}>
        {entries.map(([k, v]) => (
          <div key={k} style={{ display: "flex", gap: 8, fontSize: 12, lineHeight: 1.6 }}>
            <dt style={{ color: "#666", minWidth: 100 }}>{k}:</dt>
            <dd style={{ margin: 0 }}>{formatStatValue(v)}</dd>
          </div>
        ))}
      </dl>
    )
  }
  return String(value)
}

/** Make a stat key more human-readable */
function formatStatKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatRuntimeCode(value: string | null | undefined): string {
  if (!value) return "\u2014"
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

const RUNTIME_READINESS_TAG_COLORS: Record<string, string> = {
  ready: "green",
  unavailable: "red",
  host_gated: "gold",
  scaffold: "blue",
  unsupported: "default",
  not_applicable: "default"
}

const RUNTIME_WARNING_LABELS: Record<string, string> = {
  host_local_boundary: "Host-local boundary",
  not_untrusted_eligible: "Not untrusted eligible",
  sandbox_exec_deprecated: "sandbox-exec deprecated",
  weaker_than_vm_isolation: "Weaker than VM isolation"
}

type SandboxDiagnosticsErrorState = {
  title: string
  description: string
  alertType: "warning" | "error"
}

const MonitoringDashboardPage: React.FC = () => {
  // Admin guard state
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // Current user ID for alert assignment
  const [currentUserId, setCurrentUserId] = useState<number | null>(null)

  // System overview state
  const [systemStats, setSystemStats] = useState<any>(null)
  const [statsLoading, setStatsLoading] = useState(false)
  const [securityStatus, setSecurityStatus] = useState<any>(null)
  const [securityLoading, setSecurityLoading] = useState(false)
  const [sandboxDiagnostics, setSandboxDiagnostics] = useState<SandboxAdminRuntimeDiagnosticsResponse | null>(null)
  const [sandboxDiagnosticsLoading, setSandboxDiagnosticsLoading] = useState(false)
  const [sandboxDiagnosticsError, setSandboxDiagnosticsError] = useState<SandboxDiagnosticsErrorState | null>(null)

  // Alert rules state
  const [alertRules, setAlertRules] = useState<any[]>([])
  const [rulesLoading, setRulesLoading] = useState(false)
  const [ruleForm] = Form.useForm()
  const [creatingRule, setCreatingRule] = useState(false)

  // Alert history state
  const [alertHistory, setAlertHistory] = useState<any[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)

  // Activity state
  const [activity, setActivity] = useState<any>(null)
  const [activityLoading, setActivityLoading] = useState(false)

  // Staleness indicator & auto-refresh
  const [lastRefreshedAt, setLastRefreshedAt] = useState<Date | null>(null)
  const [autoRefreshInterval, setAutoRefreshInterval] = useState<number>(0)
  const [timeSinceRefresh, setTimeSinceRefresh] = useState("")

  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  // ── System Overview ──

  const loadSystemStats = useCallback(async () => {
    setStatsLoading(true)
    try {
      const stats = await tldwClient.getSystemStats()
      setSystemStats(stats)
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setStatsLoading(false)
    }
  }, [markAdminGuardFromError])

  const loadSecurityStatus = useCallback(async () => {
    setSecurityLoading(true)
    try {
      const status = await tldwClient.getSecurityAlertStatus()
      setSecurityStatus(status)
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setSecurityLoading(false)
    }
  }, [markAdminGuardFromError])

  const loadSandboxDiagnostics = useCallback(async () => {
    setSandboxDiagnosticsLoading(true)
    try {
      const diagnostics = await tldwClient.getSandboxRuntimeDiagnostics()
      setSandboxDiagnostics(diagnostics)
      setSandboxDiagnosticsError(null)
    } catch (err: any) {
      const guardState = deriveAdminGuardFromError(err)
      const isForbidden = guardState === "forbidden"
      setSandboxDiagnostics(null)
      setSandboxDiagnosticsError({
        title: isForbidden
          ? "Sandbox diagnostics access denied"
          : "Sandbox diagnostics unavailable",
        description: sanitizeAdminErrorMessage(
          err,
          isForbidden
            ? "You don't have permission to view sandbox runtime diagnostics."
            : "Sandbox runtime diagnostics are not available."
        ),
        alertType: isForbidden ? "error" : "warning"
      })
    } finally {
      setSandboxDiagnosticsLoading(false)
    }
  }, [])

  // ── Alert Rules ──

  const loadAlertRules = useCallback(async () => {
    setRulesLoading(true)
    try {
      const result = await tldwClient.listAlertRules()
      setAlertRules(Array.isArray(result) ? result : [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setRulesLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleCreateRule = async () => {
    try {
      const values = await ruleForm.validateFields()
      setCreatingRule(true)
      await tldwClient.createAlertRule({
        metric: values.metric.trim(),
        operator: values.operator,
        threshold: values.threshold,
        duration_minutes: values.duration_minutes,
        severity: values.severity,
        enabled: values.enabled ?? true
      })
      ruleForm.resetFields()
      message.success("Alert rule created")
      await loadAlertRules()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(sanitizeAdminErrorMessage(err, "Failed to create alert rule"))
    } finally {
      setCreatingRule(false)
    }
  }

  const handleDeleteRule = async (ruleId: number) => {
    try {
      await tldwClient.deleteAlertRule(ruleId)
      message.success("Alert rule deleted")
      await loadAlertRules()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to delete alert rule"))
    }
  }

  // ── Alert History ──

  const loadAlertHistory = useCallback(async () => {
    setHistoryLoading(true)
    try {
      const result = await tldwClient.listAlertHistory()
      setAlertHistory(Array.isArray(result) ? result : [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setHistoryLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleAssignAlert = async (alertId: string, userId: number | null) => {
    try {
      await tldwClient.assignAlert(alertId, { assigned_to_user_id: userId })
      message.success("Alert assigned")
      await loadAlertHistory()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to assign alert"))
    }
  }

  const handleSnoozeAlert = async (alertId: string, until: string) => {
    try {
      await tldwClient.snoozeAlert(alertId, { until })
      message.success("Alert snoozed")
      await loadAlertHistory()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to snooze alert"))
    }
  }

  const handleEscalateAlert = async (alertId: string) => {
    try {
      await tldwClient.escalateAlert(alertId)
      message.success("Alert escalated")
      await loadAlertHistory()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to escalate alert"))
    }
  }

  // ── Activity ──

  const loadActivity = useCallback(async () => {
    setActivityLoading(true)
    try {
      const result = await tldwClient.getDashboardActivity({ days: 7 })
      setActivity(result)
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setActivityLoading(false)
    }
  }, [markAdminGuardFromError])

  // Refresh all sections and update timestamp
  const refreshAll = useCallback(() => {
    void loadSystemStats()
    void loadSecurityStatus()
    void loadSandboxDiagnostics()
    void loadAlertRules()
    void loadAlertHistory()
    void loadActivity()
    setLastRefreshedAt(new Date())
  }, [loadSystemStats, loadSecurityStatus, loadSandboxDiagnostics, loadAlertRules, loadAlertHistory, loadActivity])

  // ── Initial Load ──

  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    void loadSystemStats()
    void loadSecurityStatus()
    void loadSandboxDiagnostics()
    void loadAlertRules()
    void loadAlertHistory()
    void loadActivity()
    setLastRefreshedAt(new Date())
    void tldwClient.getCurrentUserProfile().then(
      (profile: any) => {
        if (profile?.id) setCurrentUserId(profile.id)
      },
      () => { /* non-critical */ }
    )
  }, [loadSystemStats, loadSecurityStatus, loadSandboxDiagnostics, loadAlertRules, loadAlertHistory, loadActivity])

  // Auto-refresh timer
  useEffect(() => {
    if (autoRefreshInterval <= 0) return
    const id = setInterval(refreshAll, autoRefreshInterval * 1000)
    return () => clearInterval(id)
  }, [autoRefreshInterval, refreshAll])

  // Update "last updated X ago" text every 10 seconds
  useEffect(() => {
    const tick = () => {
      if (!lastRefreshedAt) { setTimeSinceRefresh(""); return }
      const secs = Math.floor((Date.now() - lastRefreshedAt.getTime()) / 1000)
      if (secs < 10) setTimeSinceRefresh("just now")
      else if (secs < 60) setTimeSinceRefresh(`${secs}s ago`)
      else setTimeSinceRefresh(`${Math.floor(secs / 60)}m ago`)
    }
    tick()
    const id = setInterval(tick, 10_000)
    return () => clearInterval(id)
  }, [lastRefreshedAt])

  // Derive metric name suggestions from system stats keys
  const metricOptions = useMemo(() => {
    const keys: string[] = []
    if (systemStats && typeof systemStats === "object") {
      keys.push(...Object.keys(systemStats))
    }
    if (keys.length === 0) {
      keys.push("cpu_usage", "memory_percent", "disk_usage", "active_connections", "request_count")
    }
    return keys.map((k) => ({ value: k, label: k }))
  }, [systemStats])

  // Starter alert rules for empty state
  const starterRules = useMemo(() => [
    { metric: "cpu_usage", operator: ">", threshold: 90, duration_minutes: 5, severity: "high" },
    { metric: "memory_percent", operator: ">", threshold: 85, duration_minutes: 10, severity: "medium" },
    { metric: "disk_usage", operator: ">", threshold: 95, duration_minutes: 1, severity: "critical" }
  ], [])

  const handleCreateStarterRule = async (rule: typeof starterRules[0]) => {
    try {
      setCreatingRule(true)
      await tldwClient.createAlertRule({ ...rule, enabled: true })
      message.success(`Starter rule created: ${rule.metric} ${rule.operator} ${rule.threshold}`)
      await loadAlertRules()
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, "Failed to create starter rule"))
    } finally {
      setCreatingRule(false)
    }
  }

  // ── Alert Rules Table Columns ──

  const ruleColumns = [
    { title: "Metric", dataIndex: "metric", key: "metric", render: (metric: string) => <code>{metric}</code> },
    { title: "Operator", dataIndex: "operator", key: "operator" },
    { title: "Threshold", dataIndex: "threshold", key: "threshold" },
    { title: "Duration (min)", dataIndex: "duration_minutes", key: "duration_minutes", render: (val: number | null) => val ?? "\u2014" },
    {
      title: "Severity", dataIndex: "severity", key: "severity",
      render: (severity: string) => {
        const color = severity === "critical" ? "red" : severity === "high" ? "orange" : severity === "medium" ? "gold" : "default"
        return <Tag color={color}>{severity || "low"}</Tag>
      }
    },
    {
      title: "Enabled", dataIndex: "enabled", key: "enabled",
      render: (enabled: boolean) => <Tag color={enabled !== false ? "green" : "default"}>{enabled !== false ? "Yes" : "No"}</Tag>
    },
    {
      title: "Actions", key: "actions",
      render: (_: any, record: any) => (
        <Popconfirm title="Delete this alert rule?" onConfirm={() => handleDeleteRule(record.id)}>
          <Button size="small" danger>Delete</Button>
        </Popconfirm>
      )
    }
  ]

  // ── Alert History Table Columns ──

  const historyColumns = [
    { title: "Alert", dataIndex: "alert", key: "alert", render: (alert: string, record: any) => alert || record.metric || record.id || "\u2014" },
    {
      title: "Severity", dataIndex: "severity", key: "severity",
      render: (severity: string) => {
        const color = severity === "critical" ? "red" : severity === "high" ? "orange" : severity === "medium" ? "gold" : "default"
        return <Tag color={color}>{severity || "low"}</Tag>
      }
    },
    { title: "Time", dataIndex: "triggered_at", key: "triggered_at", render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014" },
    {
      title: "Status", dataIndex: "status", key: "status",
      render: (status: string) => {
        const color = status === "resolved" ? "green" : status === "snoozed" ? "blue" : status === "escalated" ? "red" : "orange"
        return <Tag color={color}>{status || "active"}</Tag>
      }
    },
    {
      title: "Actions", key: "actions",
      render: (_: any, record: any) => {
        const identity = String(record.id ?? record.alert ?? "")
        return (
          <Space size="small">
            <Popconfirm title="Assign this alert?" description="This will assign the alert to you (or unassign)." onConfirm={() => handleAssignAlert(identity, currentUserId)}>
              <Button size="small">Assign</Button>
            </Popconfirm>
            <Select size="small" placeholder="Snooze" style={{ width: 100 }} onChange={(minutes: number) => { const until = new Date(Date.now() + minutes * 60 * 1000).toISOString(); handleSnoozeAlert(identity, until) }} options={[{ value: 30, label: "30 min" }, { value: 60, label: "1 hour" }, { value: 240, label: "4 hours" }, { value: 1440, label: "24 hours" }]} />
            <Popconfirm title="Escalate this alert?" onConfirm={() => handleEscalateAlert(identity)}>
              <Button size="small" danger>Escalate</Button>
            </Popconfirm>
          </Space>
        )
      }
    }
  ]

  const sandboxRuntimeColumns = [
    {
      title: "Runtime",
      dataIndex: "name",
      key: "name",
      render: (name: string) => <code>{name}</code>
    },
    {
      title: "Readiness",
      dataIndex: "readiness",
      key: "readiness",
      render: (readiness: string) => (
        <Tag color={RUNTIME_READINESS_TAG_COLORS[readiness] || "default"}>
          {formatRuntimeCode(readiness)}
        </Tag>
      )
    },
    {
      title: "Boundary",
      dataIndex: "boundary_class",
      key: "boundary_class",
      render: (boundaryClass: string | null | undefined) => formatRuntimeCode(boundaryClass)
    },
    {
      title: "VM-grade",
      dataIndex: "vm_grade_isolation",
      key: "vm_grade_isolation",
      render: (vmGrade: boolean) => (
        <Tag color={vmGrade ? "green" : "orange"}>{vmGrade ? "Yes" : "No"}</Tag>
      )
    },
    {
      title: "Untrusted",
      dataIndex: "untrusted_eligible",
      key: "untrusted_eligible",
      render: (eligible: boolean) => (
        <Tag color={eligible ? "green" : "red"}>{eligible ? "Eligible" : "Not eligible"}</Tag>
      )
    },
    {
      title: "Warnings",
      dataIndex: "isolation_warnings",
      key: "isolation_warnings",
      render: (warnings: string[] | undefined) => {
        const values = Array.isArray(warnings) ? warnings : []
        if (values.length === 0) return <Tag color="green">None</Tag>
        return (
          <Space size={[4, 4]} wrap>
            {values.map((warning) => (
              <Tag key={warning} color="gold">
                {RUNTIME_WARNING_LABELS[warning] || formatRuntimeCode(warning)}
              </Tag>
            ))}
          </Space>
        )
      }
    },
    {
      title: "Action",
      dataIndex: "recommended_action",
      key: "recommended_action",
      render: (action: string | undefined) => formatRuntimeCode(action || "none")
    }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return <Alert type="error" title="Access Denied" description="You don't have permission to access the monitoring dashboard." showIcon />
  }
  if (adminGuard === "notFound") {
    return <Alert type="warning" title="Not Available" description="The monitoring dashboard is not available on this server." showIcon />
  }

  const activityEntries = Array.isArray(activity?.entries) ? activity.entries : Array.isArray(activity) ? activity : []
  const sandboxRuntimeRows: SandboxAdminRuntimeDiagnosticsItem[] = Array.isArray(sandboxDiagnostics?.runtimes)
    ? sandboxDiagnostics.runtimes
    : []
  const hostLocalWarningRuntimes = Array.isArray(sandboxDiagnostics?.summary?.host_local_warning_runtimes)
    ? sandboxDiagnostics.summary.host_local_warning_runtimes
    : []
  const readyStateLabel = getDesignSystemState("ready")?.label ?? "ready"
  const unavailableStateLabel =
    getDesignSystemState("unavailable")?.label ?? "unavailable"

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h2 style={{ marginBottom: 4 }}>Monitoring &amp; Alerting</h2>
      <Typography.Paragraph type="secondary" style={{ marginBottom: 16 }}>
        Monitor your tldw server&apos;s health and set up alerts for important metrics. Create rules below to get notified when something needs attention.
      </Typography.Paragraph>

      {/* System Overview Card */}
      <Card title="System Overview" loading={statsLoading || securityLoading} style={{ marginBottom: 16 }} extra={
        <Space size="small" align="center">
          {timeSinceRefresh && <Typography.Text type="secondary" style={{ fontSize: 12 }}>Updated {timeSinceRefresh}</Typography.Text>}
          <Select size="small" value={autoRefreshInterval} onChange={setAutoRefreshInterval} style={{ width: 90 }} options={[{ value: 0, label: "Off" }, { value: 30, label: "30s" }, { value: 60, label: "1min" }, { value: 300, label: "5min" }]} />
          <Button size="small" onClick={refreshAll}>Refresh</Button>
        </Space>
      }>
        <Space orientation="vertical" style={{ width: "100%" }} size="middle">
          {systemStats && (
            <div>
              <strong>System Stats:</strong>
              <Table dataSource={Object.entries(systemStats).map(([key, value]) => ({ key, stat: key, rawValue: value }))} columns={[{ title: "Stat", dataIndex: "stat", key: "stat", render: (s: string) => formatStatKey(s) }, { title: "Value", dataIndex: "rawValue", key: "value", render: (v: unknown) => formatStatValue(v) }]} rowKey="key" pagination={false} size="small" />
            </div>
          )}
          {securityStatus && (
            <div>
              <strong>Security Alert Status:</strong>
              <Table dataSource={Object.entries(securityStatus).map(([key, value]) => ({ key, field: key, rawValue: value }))} columns={[{ title: "Field", dataIndex: "field", key: "field", render: (s: string) => formatStatKey(s) }, { title: "Value", dataIndex: "rawValue", key: "value", render: (v: unknown) => formatStatValue(v) }]} rowKey="key" pagination={false} size="small" />
            </div>
          )}
          {!systemStats && !securityStatus && !statsLoading && !securityLoading && (
            <Alert type="info" title="No system data available yet." showIcon />
          )}
        </Space>
      </Card>

      <Card title="Sandbox Runtime Isolation" loading={sandboxDiagnosticsLoading} style={{ marginBottom: 16 }} extra={<Button onClick={() => loadSandboxDiagnostics()} size="small">Refresh</Button>}>
        <Space orientation="vertical" style={{ width: "100%" }} size="middle">
          {sandboxDiagnosticsError && (
            <Alert
              type={sandboxDiagnosticsError.alertType}
              title={sandboxDiagnosticsError.title}
              description={sandboxDiagnosticsError.description}
              showIcon
            />
          )}
          {sandboxDiagnostics?.summary && (
            <Descriptions size="small" column={5}>
              <Descriptions.Item label="Total">
                {sandboxDiagnostics.summary.total}
              </Descriptions.Item>
              <Descriptions.Item label={readyStateLabel}>
                {sandboxDiagnostics.summary.ready}
              </Descriptions.Item>
              <Descriptions.Item label={unavailableStateLabel}>
                {sandboxDiagnostics.summary.unavailable}
              </Descriptions.Item>
              <Descriptions.Item label="Host-gated">
                {sandboxDiagnostics.summary.host_gated}
              </Descriptions.Item>
              <Descriptions.Item label="Scaffold">
                {sandboxDiagnostics.summary.scaffold}
              </Descriptions.Item>
            </Descriptions>
          )}
          {hostLocalWarningRuntimes.length > 0 && (
            <Alert
              type="warning"
              title="Host-local sandbox runtimes require operator review"
              description={
                <span>
                  {hostLocalWarningRuntimes.join(", ")} run on host-local boundaries,
                  are not VM-grade isolation, and are not eligible for untrusted code.
                </span>
              }
              showIcon
            />
          )}
          {sandboxRuntimeRows.length > 0 ? (
            <Table<SandboxAdminRuntimeDiagnosticsItem>
              dataSource={sandboxRuntimeRows}
              columns={sandboxRuntimeColumns as any}
              rowKey="name"
              pagination={false}
              size="small"
            />
          ) : !sandboxDiagnosticsLoading && !sandboxDiagnosticsError ? (
            <Alert type="info" title="No sandbox runtime diagnostics available yet." showIcon />
          ) : null}
        </Space>
      </Card>

      {/* Alert Rules Card */}
      <Card title="Alert Rules" style={{ marginBottom: 16 }} extra={<Button onClick={() => loadAlertRules()} size="small">Refresh</Button>}>
        <div style={{ marginBottom: 16 }}>
          <Form form={ruleForm} layout="inline" style={{ flexWrap: "wrap", gap: "8px 0" }}>
            <Form.Item name="metric" rules={[{ required: true, message: "Metric is required" }]}>
              <AutoComplete placeholder="e.g. cpu_usage" style={{ width: 180 }} options={metricOptions} filterOption={(input, option) => (option?.value as string)?.toLowerCase().includes(input.toLowerCase())} />
            </Form.Item>
            <Form.Item name="operator" rules={[{ required: true, message: "Operator is required" }]}>
              <Select placeholder="Operator" style={{ width: 100 }} options={[{ value: ">", label: ">" }, { value: ">=", label: ">=" }, { value: "<", label: "<" }, { value: "<=", label: "<=" }, { value: "==", label: "==" }]} />
            </Form.Item>
            <Form.Item name="threshold" rules={[{ required: true, message: "Threshold is required" }]} tooltip="The value to compare against (e.g. 90 for 90%)">
              <InputNumber placeholder="Threshold" style={{ width: 120 }} />
            </Form.Item>
            <Form.Item name="duration_minutes" rules={[{ required: true, message: "Duration is required" }]} tooltip="How long the threshold must be exceeded before alerting (1-1440 minutes)">
              <InputNumber placeholder="Duration (min)" style={{ width: 130 }} min={1} max={1440} />
            </Form.Item>
            <Form.Item name="severity" rules={[{ required: true, message: "Severity is required" }]} tooltip="Critical: immediate attention. High: investigate soon. Medium: monitor closely. Low: informational.">
              <Select placeholder="Severity" style={{ width: 120 }} options={[{ value: "low", label: "Low" }, { value: "medium", label: "Medium" }, { value: "high", label: "High" }, { value: "critical", label: "Critical" }]} />
            </Form.Item>
            <Form.Item name="enabled" valuePropName="checked" initialValue={true}>
              <Switch checkedChildren="Enabled" unCheckedChildren="Disabled" defaultChecked />
            </Form.Item>
            <Form.Item>
              <Button type="primary" onClick={handleCreateRule} loading={creatingRule}>Create Rule</Button>
            </Form.Item>
          </Form>
        </div>
        {alertRules.length === 0 && !rulesLoading ? (
          <Alert type="info" showIcon message="No alert rules configured" description={
            <div>
              <p style={{ marginBottom: 8 }}>Create your first rule using the form above, or try a starter rule:</p>
              <Space wrap>
                {starterRules.map((rule) => (
                  <Button key={rule.metric} size="small" onClick={() => handleCreateStarterRule(rule)} loading={creatingRule}>
                    {rule.metric} {rule.operator} {rule.threshold} for {rule.duration_minutes}min
                  </Button>
                ))}
              </Space>
              <p style={{ marginTop: 8 }}><Typography.Text type="secondary" style={{ fontSize: 12 }}>These are common rules &mdash; your server may use different metric names.</Typography.Text></p>
            </div>
          } style={{ marginBottom: 16 }} />
        ) : (
          <Table dataSource={alertRules} columns={ruleColumns} rowKey="id" loading={rulesLoading} pagination={false} size="small" />
        )}
      </Card>

      {/* Alert History Card */}
      <Card title="Alert History" style={{ marginBottom: 16 }} extra={<Button onClick={() => loadAlertHistory()} size="small">Refresh</Button>}>
        <Table dataSource={alertHistory} columns={historyColumns} rowKey={(record) => String(record.id ?? record.alert ?? record.triggered_at ?? "unknown")} loading={historyLoading} pagination={{ pageSize: 20 }} size="small" />
      </Card>

      {/* Activity (Collapsible) */}
      <CollapsibleSection title="Recent Activity" description="Dashboard activity over the last 7 days" defaultOpen>
        {activityLoading ? (
          <Card loading={true} />
        ) : activityEntries.length > 0 ? (
          <Table dataSource={activityEntries.map((entry: any, idx: number) => ({ ...entry, _key: idx }))} columns={[
            { title: "Time", dataIndex: "timestamp", key: "timestamp", render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014" },
            { title: "Action", dataIndex: "action", key: "action" },
            { title: "User", dataIndex: "user", key: "user", render: (val: string) => val || "\u2014" },
            { title: "Details", dataIndex: "details", key: "details", render: (val: unknown) => formatStatValue(val) }
          ]} rowKey="_key" pagination={false} size="small" />
        ) : (
          <Alert type="info" title="No recent activity data available." showIcon />
        )}
      </CollapsibleSection>
    </div>
  )
}

export default MonitoringDashboardPage
