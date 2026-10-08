import React, { useState, useCallback, useEffect } from "react"
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
  Statistic,
  Form,
  Tag,
  message
} from "antd"
import { ReloadOutlined } from "@ant-design/icons"
import { Alert } from "@/components/ui/primitives"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { serverSupportsPath } from "@/services/tldw/capability-probe"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const BILLING_OVERVIEW_PATH = "/api/v1/admin/billing/overview"

// ── Overview Tab ──

// GET /admin/storage-quotas/summary returns {total_quotas, items:
// [{quota_mb, used_mb, ...}], pagination} - the flat total_used_mb /
// avg_utilization_pct fields never existed, so aggregate the page here.
export const aggregateStorageSummary = (summary: any) => {
  const quotaItems: Array<{ quota_mb?: number; used_mb?: number }> = summary?.items ?? []
  const totalQuotaMb = quotaItems.reduce((sum, q) => sum + (q.quota_mb ?? 0), 0)
  const totalUsedMb = quotaItems.reduce((sum, q) => sum + (q.used_mb ?? 0), 0)
  return {
    totalQuotas: summary?.total_quotas ?? quotaItems.length,
    totalQuotaMb,
    totalUsedMb,
    utilizationPct: totalQuotaMb > 0 ? (totalUsedMb / totalQuotaMb) * 100 : 0,
    hasMore: Boolean(summary?.has_more ?? summary?.pagination?.has_more)
  }
}

const OverviewTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [overview, setOverview] = useState<any>(null)
  const [storageSummary, setStorageSummary] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const loadOverview = useCallback(async () => {
    setLoading(true)
    try {
      const [billing, storage] = await Promise.allSettled([
        tldwClient.getBillingOverview(),
        // 200 is the endpoint's max page size; hasMore flags any remainder.
        tldwClient.getStorageQuotaSummary({ limit: 200 })
      ])
      if (billing.status === "fulfilled") {
        setOverview(billing.value)
      } else {
        onGuardError(billing.reason)
      }
      if (storage.status === "fulfilled") {
        setStorageSummary(storage.value)
      }
    } catch (err) {
      onGuardError(err)
    } finally {
      setLoading(false)
    }
  }, [onGuardError])

  useEffect(() => {
    loadOverview()
  }, [loadOverview])

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Button icon={<ReloadOutlined />} onClick={loadOverview} loading={loading}>
          {t("common:refresh", "Refresh")}
        </Button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 16, marginBottom: 24 }}>
        <Card>
          <Statistic title={t("settings:adminBilling.mrr", "Monthly Recurring Revenue")} value={overview?.mrr ?? "N/A"} prefix="$" loading={loading} />
        </Card>
        <Card>
          <Statistic title={t("settings:adminBilling.activeSubscriptions", "Active Subscriptions")} value={overview?.active_subscriptions ?? 0} loading={loading} />
        </Card>
        <Card>
          <Statistic title={t("settings:adminBilling.canceledSubscriptions", "Canceled Subscriptions")} value={overview?.canceled_subscriptions ?? 0} loading={loading} />
        </Card>
        <Card>
          <Statistic title={t("settings:adminBilling.pastDue", "Past Due")} value={overview?.past_due_subscriptions ?? 0} loading={loading} />
        </Card>
      </div>

      {overview?.plan_distribution && (
        <Card title={t("settings:adminBilling.planDistribution", "Plan Distribution")} style={{ marginBottom: 24 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 16 }}>
            {Object.entries(overview.plan_distribution).map(([plan, count]) => (
              <Statistic key={plan} title={plan} value={count as number} />
            ))}
          </div>
        </Card>
      )}

      {storageSummary &&
        (() => {
          const storage = aggregateStorageSummary(storageSummary)
          return (
            <Card title={t("settings:adminBilling.storageSummary", "Storage Summary")}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 16 }}>
                <Statistic
                  title={t("settings:adminBilling.totalQuotas", "Quota Records")}
                  value={storage.totalQuotas}
                  suffix={storage.hasMore ? "+" : undefined}
                />
                <Statistic
                  title={t("settings:adminBilling.totalUsedMb", "Total Used (MB)")}
                  value={storage.totalUsedMb}
                  precision={1}
                />
                <Statistic
                  title={t("settings:adminBilling.totalQuotaMb", "Total Quota (MB)")}
                  value={storage.totalQuotaMb}
                  precision={1}
                />
                <Statistic
                  title={t("settings:adminBilling.avgUtilization", "Utilization")}
                  value={storage.utilizationPct}
                  suffix="%"
                  precision={1}
                />
              </div>
              {storage.hasMore && (
                <p style={{ marginTop: 12, marginBottom: 0, color: "var(--color-text-secondary, #888)", fontSize: "0.85rem" }}>
                  {t(
                    "settings:adminBilling.storageTruncated",
                    "Totals cover the first 200 quota records; this server has more."
                  )}
                </p>
              )}
            </Card>
          )
        })()}
    </div>
  )
}

// ── Subscriptions Tab ──

const SubscriptionsTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [subscriptions, setSubscriptions] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [statusFilter, setStatusFilter] = useState<string>("all")

  // Override modal
  const [overrideModal, setOverrideModal] = useState<{ visible: boolean; userId: number | null }>({ visible: false, userId: null })
  const [overrideForm] = Form.useForm()
  const [overriding, setOverriding] = useState(false)

  // Credits modal
  const [creditsModal, setCreditsModal] = useState<{ visible: boolean; userId: number | null }>({ visible: false, userId: null })
  const [creditsForm] = Form.useForm()
  const [granting, setGranting] = useState(false)

  const loadSubscriptions = useCallback(async () => {
    setLoading(true)
    try {
      const params: any = { limit: 100 }
      if (statusFilter !== "all") params.status = statusFilter
      const result = await tldwClient.listAllSubscriptions(params)
      // The server returns the {items, total} envelope; legacy shapes
      // (data/subscriptions arrays) stay as fallbacks for older builds.
      setSubscriptions(Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.subscriptions ?? [])
    } catch (err) {
      onGuardError(err)
    } finally {
      setLoading(false)
    }
  }, [onGuardError, statusFilter])

  useEffect(() => {
    loadSubscriptions()
  }, [loadSubscriptions])

  const handleOverride = async () => {
    if (!overrideModal.userId) return
    setOverriding(true)
    try {
      const values = await overrideForm.validateFields()
      await tldwClient.overrideUserPlan(overrideModal.userId, {
        plan_id: values.plan_id,
        reason: values.reason || undefined
      })
      message.success(t("settings:adminBilling.planOverridden", "Plan overridden successfully"))
      setOverrideModal({ visible: false, userId: null })
      overrideForm.resetFields()
      loadSubscriptions()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(
        sanitizeAdminErrorMessage(err, t("settings:adminBilling.overrideFailed", "Failed to override the user plan"))
      )
    } finally {
      setOverriding(false)
    }
  }

  const handleGrantCredits = async () => {
    if (!creditsModal.userId) return
    setGranting(true)
    try {
      const values = await creditsForm.validateFields()
      await tldwClient.grantCredits(creditsModal.userId, {
        amount: values.amount,
        reason: values.reason || undefined
      })
      message.success(t("settings:adminBilling.creditsGranted", "Credits granted successfully"))
      setCreditsModal({ visible: false, userId: null })
      creditsForm.resetFields()
      loadSubscriptions()
    } catch (err: any) {
      if (err?.errorFields) return
      message.error(
        sanitizeAdminErrorMessage(err, t("settings:adminBilling.grantCreditsFailed", "Failed to grant credits"))
      )
    } finally {
      setGranting(false)
    }
  }

  const statusColor = (status: string) => {
    switch (status) {
      case "active": return "green"
      case "canceled": return "red"
      case "past_due": return "orange"
      default: return "default"
    }
  }

  const columns = [
    {
      title: t("settings:adminBilling.colUserId", "User ID"),
      dataIndex: "user_id",
      key: "user_id",
      width: 100
    },
    {
      title: t("settings:adminBilling.colUsername", "Username"),
      dataIndex: "username",
      key: "username"
    },
    {
      title: t("settings:adminBilling.colPlan", "Plan"),
      dataIndex: "plan_id",
      key: "plan_id"
    },
    {
      title: t("settings:adminBilling.colStatus", "Status"),
      dataIndex: "status",
      key: "status",
      render: (status: string) => <Tag color={statusColor(status)}>{status}</Tag>
    },
    {
      title: t("settings:adminBilling.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      render: (val: string) => val ? new Date(val).toLocaleDateString() : "N/A"
    },
    {
      title: t("settings:adminBilling.colActions", "Actions"),
      key: "actions",
      render: (_: any, record: any) => (
        <Space>
          <Button
            size="small"
            onClick={() => {
              setOverrideModal({ visible: true, userId: record.user_id })
              overrideForm.setFieldsValue({ plan_id: record.plan_id })
            }}
          >
            {t("settings:adminBilling.overridePlan", "Override Plan")}
          </Button>
          <Button
            size="small"
            onClick={() => setCreditsModal({ visible: true, userId: record.user_id })}
          >
            {t("settings:adminBilling.grantCredits", "Grant Credits")}
          </Button>
        </Space>
      )
    }
  ]

  return (
    <div>
      <div style={{ marginBottom: 16, display: "flex", gap: 12, alignItems: "center" }}>
        <Select
          value={statusFilter}
          onChange={setStatusFilter}
          style={{ width: 160 }}
          options={[
            { value: "all", label: t("settings:adminBilling.allStatuses", "All Statuses") },
            { value: "active", label: t("settings:adminBilling.statusActive", "Active") },
            { value: "canceled", label: t("settings:adminBilling.statusCanceled", "Canceled") },
            { value: "past_due", label: t("settings:adminBilling.statusPastDue", "Past Due") }
          ]}
        />
        <Button icon={<ReloadOutlined />} onClick={loadSubscriptions} loading={loading}>
          {t("common:refresh", "Refresh")}
        </Button>
        {/* Bounded snapshot (limit:100) — full server pagination waits on a
            truthful total; disclose the truncation instead of hiding it. */}
        <span
          style={{ color: "var(--color-text-secondary, #888)", fontSize: "0.85rem" }}
        >
          {t("settings:adminBilling.showingFirst100Subscriptions", "Showing first 100 subscriptions")}
        </span>
      </div>

      <Table
        dataSource={subscriptions}
        columns={columns}
        rowKey={(r) => r.user_id ?? r.id ?? Math.random()}
        loading={loading}
        pagination={{ pageSize: 20 }}
        size="small"
      />

      <Modal
        title={t("settings:adminBilling.overrideModalTitle", { defaultValue: "Override Plan - User {{userId}}", userId: overrideModal.userId })}
        open={overrideModal.visible}
        onOk={handleOverride}
        onCancel={() => { setOverrideModal({ visible: false, userId: null }); overrideForm.resetFields() }}
        confirmLoading={overriding}
      >
        <Form form={overrideForm} layout="vertical">
          <Form.Item name="plan_id" label={t("settings:adminBilling.planIdLabel", "Plan ID")} rules={[{ required: true, message: t("settings:adminBilling.planIdRequired", "Plan ID is required") }]}>
            <Input placeholder={t("settings:adminBilling.planIdPlaceholder", "e.g. pro, enterprise, free")} />
          </Form.Item>
          <Form.Item name="reason" label="Reason">
            <Input.TextArea rows={2} placeholder={t("settings:adminBilling.overrideReasonPlaceholder", "Optional reason for the override")} />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={t("settings:adminBilling.creditsModalTitle", { defaultValue: "Grant Credits - User {{userId}}", userId: creditsModal.userId })}
        open={creditsModal.visible}
        onOk={handleGrantCredits}
        onCancel={() => { setCreditsModal({ visible: false, userId: null }); creditsForm.resetFields() }}
        confirmLoading={granting}
      >
        <Form form={creditsForm} layout="vertical">
          <Form.Item name="amount" label={t("settings:adminBilling.amountLabel", "Amount")} rules={[{ required: true, message: t("settings:adminBilling.amountRequired", "Amount is required") }]}>
            <InputNumber min={1} style={{ width: "100%" }} placeholder={t("settings:adminBilling.creditAmountPlaceholder", "Credit amount")} />
          </Form.Item>
          <Form.Item name="reason" label="Reason">
            <Input.TextArea rows={2} placeholder={t("settings:adminBilling.creditsReasonPlaceholder", "Optional reason for granting credits")} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

// ── Billing Events Tab ──

const BillingEventsTab: React.FC<{ onGuardError: (err: any) => void }> = ({ onGuardError }) => {
  const { t } = useTranslation(["settings", "common"])
  const [events, setEvents] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const loadEvents = useCallback(async () => {
    setLoading(true)
    try {
      const result = await tldwClient.listBillingEvents({ limit: 100 })
      // Same {items, total} envelope as subscriptions; legacy fallbacks kept.
      setEvents(Array.isArray(result) ? result : result?.items ?? result?.data ?? result?.events ?? [])
    } catch (err) {
      onGuardError(err)
    } finally {
      setLoading(false)
    }
  }, [onGuardError])

  useEffect(() => {
    loadEvents()
  }, [loadEvents])

  const columns = [
    {
      title: t("settings:adminBilling.colEventType", "Event Type"),
      dataIndex: "event_type",
      key: "event_type",
      render: (val: string) => <Tag>{val}</Tag>
    },
    {
      title: t("settings:adminBilling.colUserId", "User ID"),
      dataIndex: "user_id",
      key: "user_id",
      width: 100
    },
    {
      title: t("settings:adminBilling.colAmount", "Amount"),
      dataIndex: "amount",
      key: "amount",
      render: (val: number) => val != null ? `$${val.toFixed(2)}` : "N/A"
    },
    {
      title: t("settings:adminBilling.colDescription", "Description"),
      dataIndex: "description",
      key: "description",
      ellipsis: true
    },
    {
      title: t("settings:adminBilling.colCreated", "Created"),
      dataIndex: "created_at",
      key: "created_at",
      render: (val: string) => val ? new Date(val).toLocaleString() : "N/A"
    }
  ]

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        <Button icon={<ReloadOutlined />} onClick={loadEvents} loading={loading}>
          {t("common:refresh", "Refresh")}
        </Button>{" "}
        {/* Bounded snapshot (limit:100) — same truncation disclosure as
            subscriptions; full pagination waits on a truthful total. */}
        <span
          style={{ color: "var(--color-text-secondary, #888)", fontSize: "0.85rem" }}
        >
          {t("settings:adminBilling.showingFirst100Events", "Showing first 100 events")}
        </span>
      </div>

      <Table
        dataSource={events}
        columns={columns}
        rowKey={(r) => r.id ?? r.event_id ?? Math.random()}
        loading={loading}
        pagination={{ pageSize: 25 }}
        size="small"
      />
    </div>
  )
}

// ── Main Page ──

const BillingDashboardPage: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  const { config: connectionConfig, loading: connectionConfigLoading } = useCanonicalConnectionConfig()
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)
  // null = probe pending/unknown; false = the server lacks the billing routes.
  // The tabs render immediately either way - the probe only downgrades the
  // billing content in place, it never gates the first render.
  const [billingSupported, setBillingSupported] = useState<boolean | null>(null)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  // Capability probe through the shared session cache (one openapi.json fetch
  // per server URL). Tri-state: false = the fetched spec definitively lacks
  // the billing routes (downgrade in place); null = probe unknown/failed or
  // the 4s failsafe fired - no downgrade, the runtime endpoints report their
  // own errors instead of the probe gating the render.
  useEffect(() => {
    if (connectionConfigLoading) return
    const serverUrl = connectionConfig?.serverUrl?.trim()
    if (!serverUrl) return
    let cancelled = false
    let timeoutId: ReturnType<typeof setTimeout> | undefined
    const probeTimedOut = new Promise<null>((resolve) => {
      timeoutId = setTimeout(() => resolve(null), 4000)
    })
    void Promise.race([
      serverSupportsPath(serverUrl, BILLING_OVERVIEW_PATH).catch(() => null),
      probeTimedOut
    ]).then((supported) => {
      if (!cancelled && supported !== null) setBillingSupported(supported)
    })
    return () => {
      cancelled = true
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [connectionConfig?.serverUrl, connectionConfigLoading])

  // The page heading (and its Usage cross-link) renders in every state -
  // guard panels included - so the page never loses its h1 (#2898 L2).
  const pageHeading = (
    <span style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminBilling.title", "Billing Dashboard")}</h1>
      <a href="/admin/usage" style={{ fontSize: "0.85rem" }}>
        {t("settings:adminBilling.usageCrossLink", "Request and token volumes live in Usage Analytics")}
      </a>
    </span>
  )

  // Once the probe has DEFINITIVELY ruled the billing routes absent, a 404
  // from the speculative overview call is route absence, not an anomaly:
  // suppress the page-level notFound guard so the inline downgrade notice is
  // the durable end state, whichever of the two resolves first. Forbidden
  // (403) always passes through.
  const effectiveAdminGuard =
    adminGuard === "notFound" && billingSupported === false ? null : adminGuard

  if (effectiveAdminGuard === "forbidden") {
    return (
      <div style={{ padding: 24 }}>
        {pageHeading}
        <Alert variant="error" title={t("settings:adminBilling.forbiddenTitle", "Access Denied")}>
          {t("settings:adminBilling.forbiddenBody", "You do not have permission to view the billing dashboard.")}
        </Alert>
      </div>
    )
  }

  if (effectiveAdminGuard === "notFound") {
    return (
      <div style={{ padding: 24 }}>
        {pageHeading}
        <Alert variant="warning" title={t("settings:adminBilling.notFoundTitle", "Not available on this server")}>
          {t(
            "settings:adminBilling.notFoundBody",
            "Billing endpoints are not enabled here. Billing applies to multi-user deployments with subscription management configured; single-user servers do not use it."
          )}
        </Alert>
      </div>
    )
  }

  // The billing routes are missing on this server (e.g. single-user
  // deployments): downgrade the tab content in place - RateLimitingPage's
  // lazy per-endpoint style - instead of swapping the whole page for a guard.
  const unavailableNotice = billingSupported === false ? (
    <Alert variant="warning" title={t("settings:adminBilling.notFoundTitle", "Not available on this server")}>
      {t(
        "settings:adminBilling.notFoundBody",
        "Billing endpoints are not enabled here. Billing applies to multi-user deployments with subscription management configured; single-user servers do not use it."
      )}
    </Alert>
  ) : null

  const tabItems = [
    {
      key: "overview",
      label: t("settings:adminBilling.tabOverview", "Overview"),
      children: unavailableNotice ?? <OverviewTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "subscriptions",
      label: t("settings:adminBilling.tabSubscriptions", "Subscriptions"),
      children: unavailableNotice ?? <SubscriptionsTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "events",
      label: t("settings:adminBilling.tabEvents", "Billing Events"),
      children: unavailableNotice ?? <BillingEventsTab onGuardError={markAdminGuardFromError} />
    }
  ]

  return (
    <div style={{ padding: 24 }}>
      {pageHeading}
      <Tabs defaultActiveKey="overview" items={tabItems} />
    </div>
  )
}

export default BillingDashboardPage
