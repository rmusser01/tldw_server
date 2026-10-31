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
import { formatAdminDateTime } from "./admin-format"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { Alert } from "@/components/ui/primitives"
import { isValidEmailAddress, normalizeEmailAddress } from "../Watchlists/JobsTab/email-utils"

const { TextArea } = Input

// Draft state for the maintenance banner lives here, not in MaintenancePage,
// so typing in the message/allowlist inputs (or flipping the toggle)
// re-renders this form only - the flags/incidents/rotation tables below keep
// their props untouched (C-S5). Commit is explicit via the existing Save
// Changes button, matching the page's pre-existing apply pattern.
const MaintenanceBannerForm: React.FC<{
  initialEnabled: boolean
  initialMessage: string
  initialAllowlist: string
  saving: boolean
  onSave: (values: { enabled: boolean; message: string; allowlist: string }) => void
}> = ({ initialEnabled, initialMessage, initialAllowlist, saving, onSave }) => {
  const { t } = useTranslation(["settings", "common"])
  const [enabled, setEnabled] = useState(initialEnabled)
  const [message, setMessage] = useState(initialMessage)
  const [allowlist, setAllowlist] = useState(initialAllowlist)

  return (
    <Space orientation="vertical" style={{ width: "100%" }} size="middle">
      <Space>
        <span>{t("settings:adminMaintenance.maintToggleLabel", "Maintenance Mode:")}</span>
        <Switch
          aria-label={t("settings:adminMaintenance.maintenanceModeLabel", "Maintenance mode")}
          checked={enabled}
          onChange={(checked) => setEnabled(checked)}
          checkedChildren={t("settings:adminMaintenance.maintOn", "ON")}
          unCheckedChildren={t("settings:adminMaintenance.maintOff", "OFF")}
        />
      </Space>
      <div>
        <label htmlFor="maintenance-message" style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>{t("settings:adminMaintenance.maintMessageLabel", "Message:")}</label>
        <TextArea
          id="maintenance-message"
          rows={2}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={t("settings:adminMaintenance.maintMessagePlaceholder", "Maintenance message displayed to users...")}
        />
      </div>
      <div>
        <label htmlFor="maintenance-allowlist" style={{ display: "block", marginBottom: 4, fontWeight: 500 }}>{t("settings:adminMaintenance.maintAllowlistIdsEmailsLabel", "Allowed user IDs or emails (comma-separated):")}</label>
        <Input
          id="maintenance-allowlist"
          value={allowlist}
          onChange={(e) => setAllowlist(e.target.value)}
          placeholder={t("settings:adminMaintenance.maintAllowlistIdsEmailsPlaceholder", "e.g. 7, alice@example.com")}
        />
      </div>
      <Button
        type="primary"
        loading={saving}
        onClick={() => onSave({ enabled, message, allowlist })}
      >
        {t("settings:adminMaintenance.saveChanges", "Save Changes")}
      </Button>
    </Space>
  )
}
type MaintenanceFlag = {
  key: string; scope?: string; org_id?: number | null; user_id?: number | null;
  target_user_ids?: number[]; rollout_percent?: number; variant_value?: string | null;
}

const getFlagTarget = (flag: MaintenanceFlag): {
  scope: "global" | "org" | "user"; org_id?: number; user_id?: number;
} | null => {
  if (typeof flag.key !== "string" || !flag.key.trim()) return null
  const scope = flag.scope
  if (scope !== "global" && scope !== "org" && scope !== "user") return null
  for (const id of [flag.org_id, flag.user_id]) {
    if (id != null && (typeof id !== "number" || !Number.isSafeInteger(id) || id <= 0)) return null
  }
  if ((scope === "org" && flag.org_id == null) || (scope === "user" && flag.user_id == null)) return null
  return { scope, org_id: flag.org_id ?? undefined, user_id: flag.user_id ?? undefined }
}

const MaintenancePage: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  // Admin guard state
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // Maintenance mode state. The message/allowlist/enabled values here are
  // the last-loaded server snapshot: they seed MaintenanceBannerForm (via
  // the revision-keyed remount below) and no longer change on keystrokes.
  const [maintEnabled, setMaintEnabled] = useState(false)
  const [maintMessage, setMaintMessage] = useState("")
  const [maintAllowlist, setMaintAllowlist] = useState("")
  const [maintStateRevision, setMaintStateRevision] = useState(0)
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
      setMaintAllowlist([...(state?.allowlist_user_ids ?? []), ...(state?.allowlist_emails ?? [])].join(", "))
      setMaintStateRevision((n) => n + 1)
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setMaintLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleSaveMaintenanceState = async (values: {
    enabled: boolean
    message: string
    allowlist: string
  }) => {
    setMaintSaving(true)
    try {
      const allowlistArr = values.allowlist
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean)
      const userIds = allowlistArr.filter((entry) => /^\d+$/.test(entry)).map(Number)
      const emails = allowlistArr.filter((entry) => !/^\d+$/.test(entry))
      if (userIds.some((id) => !Number.isSafeInteger(id) || id <= 0) || emails.some((email) => !isValidEmailAddress(email))) {
        message.error(t("settings:adminMaintenance.allowlistIdsEmailsInvalid", "Enter positive user IDs or valid email addresses"))
        return
      }
      await tldwClient.updateMaintenanceState({
        enabled: values.enabled,
        message: values.message || undefined,
        allowlist_user_ids: userIds,
        allowlist_emails: emails.map(normalizeEmailAddress)
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

  const handleToggleFlag = async (flag: MaintenanceFlag, enabled: boolean) => {
    const target = getFlagTarget(flag)
    if (!target || !Array.isArray(flag.target_user_ids) || flag.target_user_ids.some((id) => !Number.isSafeInteger(id) || id <= 0) || typeof flag.rollout_percent !== "number" || !Number.isInteger(flag.rollout_percent) || flag.rollout_percent < 0 || flag.rollout_percent > 100 || !(flag.variant_value === null || typeof flag.variant_value === "string")) {
      message.error(t("settings:adminMaintenance.flagDetailsRequired", "Reload complete flag details before changing this flag"))
      return
    }
    try {
      await tldwClient.updateFeatureFlag(flag.key, { enabled, ...target, target_user_ids: flag.target_user_ids, rollout_percent: flag.rollout_percent, variant_value: flag.variant_value })
      message.success(
        t("settings:adminMaintenance.flagToggled", {
          defaultValue: 'Flag "{{flag}}" {{state}}',
          flag: flag.key,
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

  const handleDeleteFlag = async (flag: MaintenanceFlag) => {
    const target = getFlagTarget(flag)
    if (!target) {
      message.error(t("settings:adminMaintenance.flagTargetRequired", "Reload a valid flag target before deleting"))
      return
    }
    try {
      await tldwClient.deleteFeatureFlag(flag.key, target)
      message.success(
        t("settings:adminMaintenance.flagDeleted", {
          defaultValue: 'Flag "{{flag}}" deleted',
          flag: flag.key
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
        summary: values.description?.trim() || undefined
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
      message.success(t("settings:adminMaintenance.rotationPreviewQueued", "Rotation preview queued"))
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
      title: t("settings:adminMaintenance.colTarget", "Target"),
      key: "target",
      render: (_: unknown, record: MaintenanceFlag) => {
        const target = getFlagTarget(record)
        if (!target) return t("settings:adminMaintenance.unknownFlagTarget", "Unknown target")
        const organization = target.org_id == null ? null : `${t("settings:adminMaintenance.scopeOrganization", "Organization")} #${target.org_id}`
        const user = target.user_id == null ? null : `${t("settings:adminMaintenance.scopeUser", "User")} #${target.user_id}`
        if (target.scope === "org") return [organization, user].filter(Boolean).join(" · ")
        if (target.scope === "user") return [user, organization].filter(Boolean).join(" · ")
        if (!organization && !user) return t("settings:adminMaintenance.scopeGlobal", "Server-wide")
        return [t("settings:adminMaintenance.globalFlag", "Global flag"), organization, user].filter(Boolean).join(" · ")
      }
    },
    {
      title: t("settings:adminMaintenance.colEnabled", "Enabled"),
      dataIndex: "enabled",
      key: "enabled",
      render: (enabled: boolean, record: MaintenanceFlag) => (
        <Switch
          checked={!!enabled}
          disabled={!getFlagTarget(record)}
          onChange={(checked) => handleToggleFlag(record, checked)}
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
          onConfirm={() => handleDeleteFlag(record)}
        >
          <Button size="small" danger disabled={!getFlagTarget(record)}>{t("common:delete", "Delete")}</Button>
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
      render: (val: string) => val ? formatAdminDateTime(val) : "\u2014"
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
              { value: "open", label: t("settings:adminMaintenance.statusOpen", "Open") },
              { value: "mitigating", label: t("settings:adminMaintenance.statusMitigating", "Mitigating") },
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
    { title: t("settings:adminMaintenance.colMode", "Mode"), dataIndex: "mode", key: "mode" },
    { title: t("settings:adminMaintenance.colScope", "Scope"), dataIndex: "scope_summary", key: "scope_summary" },
    { title: t("settings:adminMaintenance.colAffected", "Affected"), dataIndex: "affected_count", key: "affected_count", render: (value: number | null) => value ?? "\u2014" },
    { title: t("settings:adminMaintenance.colFailure", "Failure"), dataIndex: "error_message", key: "error_message", render: (value: string | null) => value ? sanitizeAdminErrorMessage({ message: value }, "Operation failed") : "\u2014" },
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
        const color = status === "complete" ? "green" : status === "running" ? "blue" : status === "failed" ? "red" : "default"
        return <Tag color={color}>{status || t("settings:adminMaintenance.statusUnknown", "unknown")}</Tag>
      }
    },
    {
      title: t("settings:adminMaintenance.colStarted", "Started"),
      dataIndex: "started_at",
      key: "started_at",
      render: (val: string) => val ? formatAdminDateTime(val) : "\u2014"
    },
    {
      title: t("settings:adminMaintenance.colCompleted", "Completed"),
      dataIndex: "completed_at",
      key: "completed_at",
      render: (val: string) => val ? formatAdminDateTime(val) : "\u2014"
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

      {/* Maintenance Mode Card — banner drafts live in MaintenanceBannerForm
          so banner keystrokes never re-render the tables below (C-S5). */}
      <Card title={t("settings:adminMaintenance.maintCardTitle", "Maintenance Mode")} loading={maintLoading} style={{ marginBottom: 16 }}>
        <MaintenanceBannerForm
          key={maintStateRevision}
          initialEnabled={maintEnabled}
          initialMessage={maintMessage}
          initialAllowlist={maintAllowlist}
          saving={maintSaving}
          onSave={handleSaveMaintenanceState}
        />
      </Card>

      {/* Feature Flags Card */}
      <Card title={t("settings:adminMaintenance.flagsCardTitle", "Feature Flags")} style={{ marginBottom: 16 }}>
        <Table
          dataSource={flags}
          columns={flagColumns}
          rowKey={(record: MaintenanceFlag) => `${record.key}:${record.scope}:${record.org_id ?? ""}:${record.user_id ?? ""}`}
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
                    {t("settings:adminMaintenance.previewRotation", "Preview rotation")}
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
