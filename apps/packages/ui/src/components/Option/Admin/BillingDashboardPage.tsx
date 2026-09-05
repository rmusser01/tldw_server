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
import { tldwClient } from "@/services/tldw/TldwApiClient"

const BILLING_OVERVIEW_PATH = "/api/v1/admin/billing/overview"

// ── Overview Tab ──

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
        tldwClient.getStorageQuotaSummary()
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

      {storageSummary && (
        <Card title={t("settings:adminBilling.storageSummary", "Storage Summary")}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 16 }}>
            <Statistic title={t("settings:adminBilling.totalUsers", "Total Users")} value={storageSummary.total_users ?? 0} />
            <Statistic
              title={t("settings:adminBilling.totalUsedMb", "Total Used (MB)")}
              value={storageSummary.total_used_mb ?? 0}
              precision={1}
            />
            <Statistic
              title={t("settings:adminBilling.totalQuotaMb", "Total Quota (MB)")}
              value={storageSummary.total_quota_mb ?? 0}
              precision={1}
            />
            <Statistic
              title={t("settings:adminBilling.avgUtilization", "Avg Utilization")}
              value={storageSummary.avg_utilization_pct ?? 0}
              suffix="%"
              precision={1}
            />
          </div>
        </Card>
      )}
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
      setSubscriptions(Array.isArray(result) ? result : result?.data ?? result?.subscriptions ?? [])
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
      setEvents(Array.isArray(result) ? result : result?.data ?? result?.events ?? [])
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
        </Button>
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
  const initialLoadRef = useRef(false)
  const [capabilityCheckResolved, setCapabilityCheckResolved] = useState(false)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  useEffect(() => {
    if (initialLoadRef.current || connectionConfigLoading) return
    initialLoadRef.current = true
    let cancelled = false

    const checkBillingSupport = async () => {
      const serverUrl = connectionConfig?.serverUrl?.trim()
      if (!serverUrl) {
        if (!cancelled) {
          setCapabilityCheckResolved(true)
        }
        return
      }

      try {
        const response = await fetch(`${serverUrl}/openapi.json`)
        if (response.ok) {
          const spec = await response.json()
          const paths =
            spec && typeof spec === "object" && spec.paths && typeof spec.paths === "object"
              ? (spec.paths as Record<string, unknown>)
              : null
          if (!cancelled && (!paths || !(BILLING_OVERVIEW_PATH in paths))) {
            setAdminGuard("notFound")
          }
        }
      } catch {
        // Ignore capability probe failures and let the page try runtime endpoints.
      } finally {
        if (!cancelled) {
          setCapabilityCheckResolved(true)
        }
      }
    }

    void checkBillingSupport()

    return () => {
      cancelled = true
    }
  }, [connectionConfig?.serverUrl, connectionConfigLoading])

  // Failsafe: the capability probe must never hold the whole page in a
  // skeleton (e.g. when the connection config never finishes loading or the
  // openapi fetch hangs). After a short grace period, render and let the
  // runtime endpoints report their own errors.
  useEffect(() => {
    if (capabilityCheckResolved) return
    const timer = setTimeout(() => setCapabilityCheckResolved(true), 4000)
    return () => clearTimeout(timer)
  }, [capabilityCheckResolved])

  if (!capabilityCheckResolved) {
    return (
      <div style={{ padding: 24 }}>
        <Card loading />
      </div>
    )
  }

  if (adminGuard === "forbidden") {
    return (
      <div style={{ padding: 24 }}>
        <Alert variant="error" title={t("settings:adminBilling.forbiddenTitle", "Access Denied")}>
          {t("settings:adminBilling.forbiddenBody", "You do not have permission to view the billing dashboard.")}
        </Alert>
      </div>
    )
  }

  if (adminGuard === "notFound") {
    return (
      <div style={{ padding: 24 }}>
        <Alert variant="warning" title={t("settings:adminBilling.notFoundTitle", "Not available on this server")}>
          {t(
            "settings:adminBilling.notFoundBody",
            "Billing endpoints are not enabled here. Billing applies to multi-user deployments with subscription management configured; single-user servers do not use it."
          )}
        </Alert>
      </div>
    )
  }

  const tabItems = [
    {
      key: "overview",
      label: t("settings:adminBilling.tabOverview", "Overview"),
      children: <OverviewTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "subscriptions",
      label: t("settings:adminBilling.tabSubscriptions", "Subscriptions"),
      children: <SubscriptionsTab onGuardError={markAdminGuardFromError} />
    },
    {
      key: "events",
      label: t("settings:adminBilling.tabEvents", "Billing Events"),
      children: <BillingEventsTab onGuardError={markAdminGuardFromError} />
    }
  ]

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminBilling.title", "Billing Dashboard")}</h1>
      <Tabs defaultActiveKey="overview" items={tabItems} />
    </div>
  )
}

export default BillingDashboardPage
