import React, { useState, useRef, useCallback, useEffect, useMemo } from "react"
import { useTranslation } from "react-i18next"
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
  Select,
  Switch,
  Popconfirm,
  Typography,
  message
} from "antd"
import type { ColumnsType } from "antd/es/table"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { CollapsibleSection } from "./CollapsibleSection"
import { Alert } from "@/components/ui/primitives"
import {
  RecoveryCallout,
  buildCapabilityState,
  type CapabilityStateDescriptor
} from "@/components/ui/state"
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
            <dt style={{ color: "#666", minWidth: 100 }}>{formatStatKey(k)}:</dt>
            <dd style={{ margin: 0 }}>{formatStatValue(v)}</dd>
          </div>
        ))}
      </dl>
    )
  }
  return String(value)
}

const STAT_KEY_ACRONYMS: Record<string, string> = {
  kb: "KB",
  mb: "MB",
  gb: "GB",
  tb: "TB",
  cpu: "CPU",
  gpu: "GPU",
  mcp: "MCP",
  acp: "ACP",
  llm: "LLM",
  api: "API",
  id: "ID",
  url: "URL",
  pct: "%"
}

/** Make a stat key human-readable: total_used_mb -> "Total Used MB",
 * mcp_invocations_today -> "MCP Invocations Today", new_last_30d -> "New (last 30 days)". */
function formatStatKey(key: string): string {
  return key
    .split("_")
    .map(
      (part) =>
        STAT_KEY_ACRONYMS[part.toLowerCase()] ??
        part.charAt(0).toUpperCase() + part.slice(1)
    )
    .join(" ")
    .replace(/\bLast 30d\b/i, "(last 30 days)")
    .replace(/\bLast 7d\b/i, "(last 7 days)")
    .replace(/\bLast 24h\b/i, "(last 24 hours)")
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

const READY_STATE_LABEL = getDesignSystemState("ready")?.label ?? "ready"
const UNAVAILABLE_STATE_LABEL =
  getDesignSystemState("unavailable")?.label ?? "unavailable"
const SANDBOX_RUNTIME_DIAGNOSTICS_PATH =
  "/api/v1/sandbox/admin/runtime-diagnostics"

type SandboxDiagnosticsErrorState = CapabilityStateDescriptor
type DashboardRecord = Record<string, unknown>
type AlertRuleFormValues = {
  metric: string
  operator: string
  threshold: number
  duration_minutes: number
  severity: string
  enabled?: boolean
}
type StarterRule = Omit<AlertRuleFormValues, "enabled">
type AlertRuleRow = Partial<AlertRuleFormValues> & {
  id: number
}
type AlertHistoryRow = {
  id?: string | number
  alert?: string
  metric?: string
  severity?: string
  status?: string
  triggered_at?: string
}
type ActivityEntry = {
  timestamp?: string
  action?: string
  user?: string
  details?: unknown
  [key: string]: unknown
}
type ActivityState = { entries?: ActivityEntry[] } | ActivityEntry[]
type ActivityRow = ActivityEntry & { _key: number }
type CurrentUserProfile = {
  id?: number | string | null
}

const hasAntdValidationError = (
  error: unknown
): error is { errorFields: unknown[] } => (
  Boolean(error) &&
  typeof error === "object" &&
  "errorFields" in error
)

const MonitoringDashboardPage: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  // Admin guard state
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // Current user ID for alert assignment
  const [currentUserId, setCurrentUserId] = useState<number | null>(null)

  // System overview state
  const [systemStats, setSystemStats] = useState<DashboardRecord | null>(null)
  const [statsLoading, setStatsLoading] = useState(false)
  const [securityStatus, setSecurityStatus] = useState<DashboardRecord | null>(null)
  const [securityLoading, setSecurityLoading] = useState(false)
  const [sandboxDiagnostics, setSandboxDiagnostics] = useState<SandboxAdminRuntimeDiagnosticsResponse | null>(null)
  const [sandboxDiagnosticsLoading, setSandboxDiagnosticsLoading] = useState(false)
  const [sandboxDiagnosticsError, setSandboxDiagnosticsError] = useState<SandboxDiagnosticsErrorState | null>(null)
  const [sandboxDiagnosticsMissing, setSandboxDiagnosticsMissing] = useState(false)

  // Alert rules state
  const [alertRules, setAlertRules] = useState<AlertRuleRow[]>([])
  const [rulesLoading, setRulesLoading] = useState(false)
  const [ruleForm] = Form.useForm<AlertRuleFormValues>()
  const [creatingRule, setCreatingRule] = useState(false)

  // Alert history state
  const [alertHistory, setAlertHistory] = useState<AlertHistoryRow[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)

  // Activity state
  const [activity, setActivity] = useState<ActivityState | null>(null)
  const [activityLoading, setActivityLoading] = useState(false)

  // Staleness indicator & auto-refresh
  const [lastRefreshedAt, setLastRefreshedAt] = useState<Date | null>(null)
  const [autoRefreshInterval, setAutoRefreshInterval] = useState<number>(0)
  const [timeSinceRefresh, setTimeSinceRefresh] = useState("")

  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: unknown) => {
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
      setSandboxDiagnosticsMissing(false)
    } catch (err: unknown) {
      const guardState = deriveAdminGuardFromError(err)
      const isForbidden = guardState === "forbidden"
      const isMissing = guardState === "notFound"
      setSandboxDiagnostics(null)
      setSandboxDiagnosticsMissing(isMissing)
      setSandboxDiagnosticsError(
        buildCapabilityState({
          featureName: "Sandbox diagnostics",
          capabilityName: "Sandbox Admin Runtime Diagnostics API",
          endpoint: SANDBOX_RUNTIME_DIAGNOSTICS_PATH,
          method: "GET",
          error: err,
          title: isForbidden
            ? t("settings:adminMonitoring.sandboxForbiddenTitle", "Sandbox diagnostics access denied")
            : isMissing
              ? t("settings:adminMonitoring.sandboxMissingTitle", "Sandbox diagnostics not available on this server")
              : t("settings:adminMonitoring.sandboxUnavailableTitle", "Sandbox diagnostics unavailable"),
          message: isForbidden
            ? t("settings:adminMonitoring.sandboxForbiddenBody", "You don't have permission to view sandbox runtime diagnostics.")
            : isMissing
              ? t("settings:adminMonitoring.sandboxMissingBody", "This server does not expose the sandbox runtime diagnostics API. This is expected when the sandbox module is not enabled.")
              : t("settings:adminMonitoring.sandboxUnavailableBody", "Sandbox runtime diagnostics are not available.")
        })
      )
    } finally {
      setSandboxDiagnosticsLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `t` is stable for the session
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
      message.success(t("settings:adminMonitoring.ruleCreated", "Alert rule created"))
      await loadAlertRules()
    } catch (err: unknown) {
      if (hasAntdValidationError(err)) return
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMonitoring.ruleCreateFailed", "Failed to create alert rule")))
    } finally {
      setCreatingRule(false)
    }
  }

  const handleDeleteRule = async (ruleId: number) => {
    try {
      await tldwClient.deleteAlertRule(ruleId)
      message.success(t("settings:adminMonitoring.ruleDeleted", "Alert rule deleted"))
      await loadAlertRules()
    } catch (err: unknown) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMonitoring.ruleDeleteFailed", "Failed to delete alert rule")))
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
      message.success(t("settings:adminMonitoring.alertAssigned", "Alert assigned"))
      await loadAlertHistory()
    } catch (err: unknown) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMonitoring.alertAssignFailed", "Failed to assign alert")))
    }
  }

  const handleSnoozeAlert = async (alertId: string, until: string) => {
    try {
      await tldwClient.snoozeAlert(alertId, { until })
      message.success(t("settings:adminMonitoring.alertSnoozed", "Alert snoozed"))
      await loadAlertHistory()
    } catch (err: unknown) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMonitoring.alertSnoozeFailed", "Failed to snooze alert")))
    }
  }

  const handleEscalateAlert = async (alertId: string) => {
    try {
      await tldwClient.escalateAlert(alertId)
      message.success(t("settings:adminMonitoring.alertEscalated", "Alert escalated"))
      await loadAlertHistory()
    } catch (err: unknown) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMonitoring.alertEscalateFailed", "Failed to escalate alert")))
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
      (profile: CurrentUserProfile) => {
        const profileId =
          typeof profile?.id === "number"
            ? profile.id
            : typeof profile?.id === "string" && profile.id.trim()
              ? Number(profile.id)
              : null
        if (profileId !== null && Number.isFinite(profileId) && profileId > 0) {
          setCurrentUserId(profileId)
        }
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
      if (secs < 10) setTimeSinceRefresh(t("settings:adminMonitoring.justNow", "just now"))
      else if (secs < 60) setTimeSinceRefresh(`${secs}${t("settings:adminMonitoring.secondsAgoSuffix", "s ago")}`)
      else setTimeSinceRefresh(`${Math.floor(secs / 60)}${t("settings:adminMonitoring.minutesAgoSuffix", "m ago")}`)
    }
    tick()
    const id = setInterval(tick, 10_000)
    return () => clearInterval(id)
  }, [lastRefreshedAt, t])

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
  const starterRules = useMemo<StarterRule[]>(() => [
    { metric: "cpu_usage", operator: ">", threshold: 90, duration_minutes: 5, severity: "high" },
    { metric: "memory_percent", operator: ">", threshold: 85, duration_minutes: 10, severity: "medium" },
    { metric: "disk_usage", operator: ">", threshold: 95, duration_minutes: 1, severity: "critical" }
  ], [])

  const handleCreateStarterRule = async (rule: typeof starterRules[0]) => {
    try {
      setCreatingRule(true)
      await tldwClient.createAlertRule({ ...rule, enabled: true })
      message.success(`${t("settings:adminMonitoring.starterRuleCreated", "Starter rule created:")} ${rule.metric} ${rule.operator} ${rule.threshold}`)
      await loadAlertRules()
    } catch (err: unknown) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminMonitoring.starterRuleFailed", "Failed to create starter rule")))
    } finally {
      setCreatingRule(false)
    }
  }

  // ── Alert Rules Table Columns ──

  const ruleColumns: ColumnsType<AlertRuleRow> = [
    { title: t("settings:adminMonitoring.colMetric", "Metric"), dataIndex: "metric", key: "metric", render: (metric: string) => <code>{metric}</code> },
    { title: t("settings:adminMonitoring.colOperator", "Operator"), dataIndex: "operator", key: "operator" },
    { title: t("settings:adminMonitoring.colThreshold", "Threshold"), dataIndex: "threshold", key: "threshold" },
    { title: t("settings:adminMonitoring.colDuration", "Duration (min)"), dataIndex: "duration_minutes", key: "duration_minutes", render: (val: number | null) => val ?? "\u2014" },
    {
      title: t("settings:adminMonitoring.colSeverity", "Severity"), dataIndex: "severity", key: "severity",
      render: (severity: string) => {
        const color = severity === "critical" ? "red" : severity === "high" ? "orange" : severity === "medium" ? "gold" : "default"
        return <Tag color={color}>{severity || "low"}</Tag>
      }
    },
    {
      title: t("settings:adminMonitoring.colEnabled", "Enabled"), dataIndex: "enabled", key: "enabled",
      render: (enabled: boolean) => <Tag color={enabled !== false ? "green" : "default"}>{enabled !== false ? t("settings:adminMonitoring.yes", "Yes") : t("settings:adminMonitoring.no", "No")}</Tag>
    },
    {
      title: t("settings:adminMonitoring.colActions", "Actions"), key: "actions",
      render: (_value: unknown, record: AlertRuleRow) => (
        <Popconfirm title={t("settings:adminMonitoring.deleteRuleConfirm", "Delete this alert rule?")} onConfirm={() => handleDeleteRule(record.id)}>
          <Button size="small" danger>{t("settings:adminMonitoring.delete", "Delete")}</Button>
        </Popconfirm>
      )
    }
  ]

  // ── Alert History Table Columns ──

  const historyColumns: ColumnsType<AlertHistoryRow> = [
    { title: t("settings:adminMonitoring.colAlert", "Alert"), dataIndex: "alert", key: "alert", render: (alert: string | undefined, record: AlertHistoryRow) => alert || record.metric || record.id || "\u2014" },
    {
      title: t("settings:adminMonitoring.colSeverity", "Severity"), dataIndex: "severity", key: "severity",
      render: (severity: string) => {
        const color = severity === "critical" ? "red" : severity === "high" ? "orange" : severity === "medium" ? "gold" : "default"
        return <Tag color={color}>{severity || "low"}</Tag>
      }
    },
    { title: t("settings:adminMonitoring.colTime", "Time"), dataIndex: "triggered_at", key: "triggered_at", render: (val: string) => val ? new Date(val).toLocaleString() : "\u2014" },
    {
      title: t("settings:adminMonitoring.colStatus", "Status"), dataIndex: "status", key: "status",
      render: (status: string) => {
        const color = status === "resolved" ? "green" : status === "snoozed" ? "blue" : status === "escalated" ? "red" : "orange"
        return <Tag color={color}>{status || "active"}</Tag>
      }
    },
    {
      title: t("settings:adminMonitoring.colActions", "Actions"), key: "actions",
      render: (_value: unknown, record: AlertHistoryRow) => {
        const identity = String(record.id ?? record.alert ?? "")
        return (
          <Space size="small">
            <Popconfirm title={t("settings:adminMonitoring.assignConfirm", "Assign this alert?")} description={t("settings:adminMonitoring.assignConfirmBody", "This will assign the alert to you (or unassign).")} onConfirm={() => handleAssignAlert(identity, currentUserId)}>
              <Button size="small">{t("settings:adminMonitoring.assign", "Assign")}</Button>
            </Popconfirm>
            <Select size="small" placeholder={t("settings:adminMonitoring.snooze", "Snooze")} style={{ width: 100 }} onChange={(minutes: number) => { const until = new Date(Date.now() + minutes * 60 * 1000).toISOString(); handleSnoozeAlert(identity, until) }} options={[{ value: 30, label: t("settings:adminMonitoring.snooze30m", "30 min") }, { value: 60, label: t("settings:adminMonitoring.snooze1h", "1 hour") }, { value: 240, label: t("settings:adminMonitoring.snooze4h", "4 hours") }, { value: 1440, label: t("settings:adminMonitoring.snooze24h", "24 hours") }]} />
            <Popconfirm title={t("settings:adminMonitoring.escalateConfirm", "Escalate this alert?")} onConfirm={() => handleEscalateAlert(identity)}>
              <Button size="small" danger>{t("settings:adminMonitoring.escalate", "Escalate")}</Button>
            </Popconfirm>
          </Space>
        )
      }
    }
  ]

  const sandboxRuntimeColumns: ColumnsType<SandboxAdminRuntimeDiagnosticsItem> = [
    {
      title: t("settings:adminMonitoring.colRuntime", "Runtime"),
      dataIndex: "name",
      key: "name",
      render: (name: string) => <code>{name}</code>
    },
    {
      title: t("settings:adminMonitoring.colReadiness", "Readiness"),
      dataIndex: "readiness",
      key: "readiness",
      render: (readiness: string) => (
        <Tag color={RUNTIME_READINESS_TAG_COLORS[readiness] || "default"}>
          {formatRuntimeCode(readiness)}
        </Tag>
      )
    },
    {
      title: t("settings:adminMonitoring.colBoundary", "Boundary"),
      dataIndex: "boundary_class",
      key: "boundary_class",
      render: (boundaryClass: string | null | undefined) => formatRuntimeCode(boundaryClass)
    },
    {
      title: t("settings:adminMonitoring.colVmGrade", "VM-grade"),
      dataIndex: "vm_grade_isolation",
      key: "vm_grade_isolation",
      render: (vmGrade: boolean) => (
        <Tag color={vmGrade ? "green" : "orange"}>{vmGrade ? t("settings:adminMonitoring.yes", "Yes") : t("settings:adminMonitoring.no", "No")}</Tag>
      )
    },
    {
      title: t("settings:adminMonitoring.colUntrusted", "Untrusted"),
      dataIndex: "untrusted_eligible",
      key: "untrusted_eligible",
      render: (eligible: boolean) => (
        <Tag color={eligible ? "green" : "red"}>{eligible ? t("settings:adminMonitoring.eligible", "Eligible") : t("settings:adminMonitoring.notEligible", "Not eligible")}</Tag>
      )
    },
    {
      title: t("settings:adminMonitoring.colWarnings", "Warnings"),
      dataIndex: "isolation_warnings",
      key: "isolation_warnings",
      render: (warnings: string[] | undefined) => {
        const values = Array.isArray(warnings) ? warnings : []
        if (values.length === 0) return <Tag color="green">{t("settings:adminMonitoring.none", "None")}</Tag>
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
      title: t("settings:adminMonitoring.colAction", "Action"),
      dataIndex: "recommended_action",
      key: "recommended_action",
      render: (action: string | undefined) => formatRuntimeCode(action || "none")
    }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title={t("settings:adminMonitoring.forbiddenTitle", "Access Denied")}>
        {t(
          "settings:adminMonitoring.forbiddenBody",
          "You don't have permission to access the monitoring dashboard."
        )}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title={t("settings:adminMonitoring.notFoundTitle", "Not Available")}>
        {t(
          "settings:adminMonitoring.notFoundBody",
          "The monitoring dashboard is not available on this server."
        )}
      </Alert>
    )
  }

  const activityEntries: ActivityEntry[] =
    activity && !Array.isArray(activity) && Array.isArray(activity.entries)
      ? activity.entries
      : Array.isArray(activity)
        ? activity
        : []
  const activityRows: ActivityRow[] = activityEntries.map((entry, idx) => ({
    ...entry,
    _key: idx
  }))
  const activityColumns: ColumnsType<ActivityRow> = [
    { title: t("settings:adminMonitoring.colTime", "Time"), dataIndex: "timestamp", key: "timestamp", render: (val: string | undefined) => val ? new Date(val).toLocaleString() : "\u2014" },
    { title: t("settings:adminMonitoring.colAction", "Action"), dataIndex: "action", key: "action" },
    { title: t("settings:adminMonitoring.colUser", "User"), dataIndex: "user", key: "user", render: (val: string | undefined) => val || "\u2014" },
    { title: t("settings:adminMonitoring.colDetails", "Details"), dataIndex: "details", key: "details", render: (val: unknown) => formatStatValue(val) }
  ]
  const sandboxRuntimeRows: SandboxAdminRuntimeDiagnosticsItem[] = Array.isArray(sandboxDiagnostics?.runtimes)
    ? sandboxDiagnostics.runtimes
    : []
  const hostLocalWarningRuntimes = Array.isArray(sandboxDiagnostics?.summary?.host_local_warning_runtimes)
    ? sandboxDiagnostics.summary.host_local_warning_runtimes
    : []

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h1 style={{ marginBottom: 4, fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminMonitoring.title", "Monitoring & Alerting")}</h1>
      <Typography.Paragraph type="secondary" style={{ marginBottom: 16 }}>
        {t(
          "settings:adminMonitoring.description",
          "Monitor your tldw server's health and set up alerts for important metrics. Create rules below to get notified when something needs attention."
        )}{" "}
        <a href="/admin/server">
          {t(
            "settings:adminMonitoring.serverAdminCrossLink",
            "Users, storage, and session management live in Server Admin."
          )}
        </a>
      </Typography.Paragraph>

      {/* System Overview Card */}
      <Card title={t("settings:adminMonitoring.systemOverviewTitle", "System Overview")} loading={statsLoading || securityLoading} style={{ marginBottom: 16 }} extra={
        <Space size="small" align="center">
          {timeSinceRefresh && <Typography.Text type="secondary" style={{ fontSize: 12 }}>{t("settings:adminMonitoring.updated", "Updated")} {timeSinceRefresh}</Typography.Text>}
          <Select size="small" value={autoRefreshInterval} onChange={setAutoRefreshInterval} style={{ width: 90 }} options={[{ value: 0, label: t("settings:adminMonitoring.refreshOff", "Off") }, { value: 30, label: t("settings:adminMonitoring.refresh30s", "30s") }, { value: 60, label: t("settings:adminMonitoring.refresh1min", "1min") }, { value: 300, label: t("settings:adminMonitoring.refresh5min", "5min") }]} />
          <Button size="small" onClick={refreshAll}>{t("common:refresh", "Refresh")}</Button>
        </Space>
      }>
        <Space orientation="vertical" style={{ width: "100%" }} size="middle">
          {systemStats && (
            <div>
              <strong>{t("settings:adminMonitoring.systemStatsLabel", "System Stats:")}</strong>
              <Table dataSource={Object.entries(systemStats).map(([key, value]) => ({ key, stat: key, rawValue: value }))} columns={[{ title: t("settings:adminMonitoring.colStat", "Stat"), dataIndex: "stat", key: "stat", render: (s: string) => formatStatKey(s) }, { title: t("settings:adminMonitoring.colValue", "Value"), dataIndex: "rawValue", key: "value", render: (v: unknown) => formatStatValue(v) }]} rowKey="key" pagination={false} size="small" />
            </div>
          )}
          {securityStatus && (
            <div>
              <strong>{t("settings:adminMonitoring.securityStatusLabel", "Security Alert Status:")}</strong>
              <Table dataSource={Object.entries(securityStatus).map(([key, value]) => ({ key, field: key, rawValue: value }))} columns={[{ title: t("settings:adminMonitoring.colField", "Field"), dataIndex: "field", key: "field", render: (s: string) => formatStatKey(s) }, { title: t("settings:adminMonitoring.colValue", "Value"), dataIndex: "rawValue", key: "value", render: (v: unknown) => formatStatValue(v) }]} rowKey="key" pagination={false} size="small" />
            </div>
          )}
          {!systemStats && !securityStatus && !statsLoading && !securityLoading && (
            <Alert title={t("settings:adminMonitoring.systemDataEmpty", "No system data available yet.")} />
          )}
        </Space>
      </Card>

      <Card title={t("settings:adminMonitoring.sandboxCardTitle", "Sandbox Runtime Isolation")} loading={sandboxDiagnosticsLoading} style={{ marginBottom: 16 }} extra={<Button onClick={() => loadSandboxDiagnostics()} size="small">{t("common:refresh", "Refresh")}</Button>}>
        <Space orientation="vertical" style={{ width: "100%" }} size="middle">
          {sandboxDiagnosticsError && (
            <RecoveryCallout
              state={sandboxDiagnosticsError.state}
              title={sandboxDiagnosticsError.title}
              message={sandboxDiagnosticsError.message}
              diagnostics={sandboxDiagnosticsError.diagnostics}
              role="alert"
              primaryAction={
                // Retrying a missing endpoint can never succeed — offer the
                // action only for transient failures (2026-09 audit S9).
                sandboxDiagnosticsMissing
                  ? undefined
                  : {
                      label: t("settings:adminMonitoring.retryDiagnostics", "Retry diagnostics"),
                      onClick: () => void loadSandboxDiagnostics()
                    }
              }
            />
          )}
          {sandboxDiagnostics?.summary && (
            <Descriptions size="small" column={5}>
              <Descriptions.Item label={t("settings:adminMonitoring.sandboxTotal", "Total")}>
                {sandboxDiagnostics.summary.total}
              </Descriptions.Item>
              <Descriptions.Item label={READY_STATE_LABEL}>
                {sandboxDiagnostics.summary.ready}
              </Descriptions.Item>
              <Descriptions.Item label={UNAVAILABLE_STATE_LABEL}>
                {sandboxDiagnostics.summary.unavailable}
              </Descriptions.Item>
              <Descriptions.Item label={t("settings:adminMonitoring.sandboxHostGated", "Host-gated")}>
                {sandboxDiagnostics.summary.host_gated}
              </Descriptions.Item>
              <Descriptions.Item label={t("settings:adminMonitoring.sandboxScaffold", "Scaffold")}>
                {sandboxDiagnostics.summary.scaffold}
              </Descriptions.Item>
            </Descriptions>
          )}
          {hostLocalWarningRuntimes.length > 0 && (
            <Alert
              variant="warning"
              title={t("settings:adminMonitoring.hostLocalWarningTitle", "Host-local sandbox runtimes require operator review")}
            >
              {
                <span>
                  {hostLocalWarningRuntimes.join(", ")}{" "}
                  {t(
                    "settings:adminMonitoring.hostLocalWarningBody",
                    "run on host-local boundaries, are not VM-grade isolation, and are not eligible for untrusted code."
                  )}
                </span>
              }
            </Alert>
          )}
          {sandboxRuntimeRows.length > 0 ? (
            <Table<SandboxAdminRuntimeDiagnosticsItem>
              dataSource={sandboxRuntimeRows}
              columns={sandboxRuntimeColumns}
              rowKey="name"
              pagination={false}
              size="small"
            />
          ) : !sandboxDiagnosticsLoading && !sandboxDiagnosticsError ? (
            <Alert title={t("settings:adminMonitoring.sandboxEmpty", "No sandbox runtime diagnostics available yet.")} />
          ) : null}
        </Space>
      </Card>

      {/* Alert Rules Card */}
      <Card title={t("settings:adminMonitoring.alertRulesTitle", "Alert Rules")} style={{ marginBottom: 16 }} extra={<Button onClick={() => loadAlertRules()} size="small">{t("common:refresh", "Refresh")}</Button>}>
        <div style={{ marginBottom: 16 }}>
          <Form form={ruleForm} layout="inline" style={{ flexWrap: "wrap", gap: "8px 0" }}>
            <Form.Item name="metric" rules={[{ required: true, message: t("settings:adminMonitoring.metricRequired", "Metric is required") }]}>
              <AutoComplete placeholder={t("settings:adminMonitoring.metricPlaceholder", "e.g. cpu_usage")} style={{ width: 180 }} options={metricOptions} filterOption={(input, option) => (option?.value as string)?.toLowerCase().includes(input.toLowerCase())} />
            </Form.Item>
            <Form.Item name="operator" rules={[{ required: true, message: t("settings:adminMonitoring.operatorRequired", "Operator is required") }]}>
              <Select placeholder={t("settings:adminMonitoring.operatorPlaceholder", "Operator")} style={{ width: 100 }} options={[{ value: ">", label: ">" }, { value: ">=", label: ">=" }, { value: "<", label: "<" }, { value: "<=", label: "<=" }, { value: "==", label: "==" }]} />
            </Form.Item>
            <Form.Item name="threshold" rules={[{ required: true, message: t("settings:adminMonitoring.thresholdRequired", "Threshold is required") }]} tooltip={t("settings:adminMonitoring.thresholdTooltip", "The value to compare against (e.g. 90 for 90%)")}>
              <InputNumber placeholder={t("settings:adminMonitoring.thresholdPlaceholder", "Threshold")} style={{ width: 120 }} />
            </Form.Item>
            <Form.Item name="duration_minutes" rules={[{ required: true, message: t("settings:adminMonitoring.durationRequired", "Duration is required") }]} tooltip={t("settings:adminMonitoring.durationTooltip", "How long the threshold must be exceeded before alerting (1-1440 minutes)")}>
              <InputNumber placeholder={t("settings:adminMonitoring.durationPlaceholder", "Duration (min)")} style={{ width: 130 }} min={1} max={1440} />
            </Form.Item>
            <Form.Item name="severity" rules={[{ required: true, message: t("settings:adminMonitoring.severityRequired", "Severity is required") }]} tooltip={t("settings:adminMonitoring.severityTooltip", "Critical: immediate attention. High: investigate soon. Medium: monitor closely. Low: informational.")}>
              <Select placeholder={t("settings:adminMonitoring.severityPlaceholder", "Severity")} style={{ width: 120 }} options={[{ value: "low", label: t("settings:adminMonitoring.severityLow", "Low") }, { value: "medium", label: t("settings:adminMonitoring.severityMedium", "Medium") }, { value: "high", label: t("settings:adminMonitoring.severityHigh", "High") }, { value: "critical", label: t("settings:adminMonitoring.severityCritical", "Critical") }]} />
            </Form.Item>
            <Form.Item name="enabled" valuePropName="checked" initialValue={true}>
              <Switch checkedChildren={t("settings:adminMonitoring.enabled", "Enabled")} unCheckedChildren={t("settings:adminMonitoring.disabled", "Disabled")} defaultChecked />
            </Form.Item>
            <Form.Item>
              <Button type="primary" onClick={handleCreateRule} loading={creatingRule}>{t("settings:adminMonitoring.createRule", "Create Rule")}</Button>
            </Form.Item>
          </Form>
        </div>
        {alertRules.length === 0 && !rulesLoading ? (
          <Alert title={t("settings:adminMonitoring.noRulesTitle", "No alert rules configured")} className="mb-4">
            <div>
              <p style={{ marginBottom: 8 }}>{t("settings:adminMonitoring.noRulesHint", "Create your first rule using the form above, or try a starter rule:")}</p>
              <Space wrap>
                {starterRules.map((rule) => (
                  <Button key={rule.metric} size="small" onClick={() => handleCreateStarterRule(rule)} loading={creatingRule}>
                    {rule.metric} {rule.operator} {rule.threshold} for {rule.duration_minutes}min
                  </Button>
                ))}
              </Space>
              <p style={{ marginTop: 8 }}><Typography.Text type="secondary" style={{ fontSize: 12 }}>{t("settings:adminMonitoring.starterRulesNote", "These are common rules — your server may use different metric names.")}</Typography.Text></p>
            </div>
          </Alert>
        ) : (
          <Table dataSource={alertRules} columns={ruleColumns} rowKey="id" loading={rulesLoading} pagination={false} size="small" />
        )}
      </Card>

      {/* Alert History Card */}
      <Card title={t("settings:adminMonitoring.alertHistoryTitle", "Alert History")} style={{ marginBottom: 16 }} extra={<Button onClick={() => loadAlertHistory()} size="small">{t("common:refresh", "Refresh")}</Button>}>
        <Table dataSource={alertHistory} columns={historyColumns} rowKey={(record) => String(record.id ?? record.alert ?? record.triggered_at ?? "unknown")} loading={historyLoading} pagination={{ pageSize: 20 }} size="small" locale={{ emptyText: t("settings:adminMonitoring.alertHistoryEmpty", "No alert activity recorded yet. Actions on alerts (acknowledge, snooze, escalate) appear here.") }} />
      </Card>

      {/* Activity (Collapsible) */}
      <CollapsibleSection title={t("settings:adminMonitoring.recentActivityTitle", "Recent Activity")} description={t("settings:adminMonitoring.recentActivityDescription", "Dashboard activity over the last 7 days")} defaultOpen>
        {activityLoading ? (
          <Card loading={true} />
        ) : activityRows.length > 0 ? (
          <Table dataSource={activityRows} columns={activityColumns} rowKey="_key" pagination={false} size="small" />
        ) : (
          <Alert title={t("settings:adminMonitoring.activityEmpty", "No recent activity data available.")} />
        )}
      </CollapsibleSection>
    </div>
  )
}

export default MonitoringDashboardPage
