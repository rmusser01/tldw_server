import React, { useState, useRef, useCallback, useEffect } from "react"
import { useTranslation } from "react-i18next"
import {
  Card,
  Table,
  Button,
  Select,
  Space,
  Collapse,
  Statistic,
  message
} from "antd"
import { Alert } from "@/components/ui/primitives"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const downloadCsv = (data: string, filename: string) => {
  const blob = new Blob([data], { type: "text/csv" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}


// Dollar amounts: whole cents normally; sub-cent LLM costs keep 4 decimals
// so tiny real spend does not round to an unhelpful $0.00.
const formatUsd = (value: number): string => {
  if (!Number.isFinite(value) || value === 0) return "$0.00"
  return Math.abs(value) < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(2)}`
}

const UsageAnalyticsPage: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  // Admin guard state
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // Date range
  const [dateRange, setDateRange] = useState<string>("7d")

  // Daily usage state
  const [dailyUsage, setDailyUsage] = useState<any[]>([])
  const [dailyLoading, setDailyLoading] = useState(false)
  const [dailyExporting, setDailyExporting] = useState(false)

  // Top users state
  const [topUsage, setTopUsage] = useState<any[]>([])
  const [topLoading, setTopLoading] = useState(false)
  const [topExporting, setTopExporting] = useState(false)

  // LLM usage state
  const [llmUsage, setLlmUsage] = useState<any[]>([])
  const [llmLoading, setLlmLoading] = useState(false)
  const [llmSummary, setLlmSummary] = useState<any>(null)
  const [llmSummaryLoading, setLlmSummaryLoading] = useState(false)
  const [topSpenders, setTopSpenders] = useState<any[]>([])
  const [topSpendersLoading, setTopSpendersLoading] = useState(false)

  // Provider analytics state
  const [providerAnalytics, setProviderAnalytics] = useState<any[]>([])
  const [providerLoading, setProviderLoading] = useState(false)

  const initialLoadRef = useRef(false)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  // Compute date params from range selection
  const getDateParams = useCallback(() => {
    const end = new Date()
    const start = new Date()
    const days = dateRange === "30d" ? 30 : 7
    start.setDate(start.getDate() - days)
    return {
      start_date: start.toISOString().split("T")[0],
      end_date: end.toISOString().split("T")[0]
    }
  }, [dateRange])

  // ── Daily Usage ──

  const loadDailyUsage = useCallback(async () => {
    setDailyLoading(true)
    try {
      const result = await tldwClient.getDailyUsage(getDateParams())
      setDailyUsage(Array.isArray(result) ? result : result?.data ?? [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setDailyLoading(false)
    }
  }, [getDateParams, markAdminGuardFromError])

  const handleExportDailyCsv = async () => {
    setDailyExporting(true)
    try {
      const csv = await tldwClient.exportDailyUsageCsv()
      downloadCsv(csv, "daily_usage.csv")
      message.success(t("settings:adminUsage.dailyCsvExported", "Daily usage CSV exported"))
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminUsage.dailyCsvExportFailed", "Failed to export daily usage CSV")))
    } finally {
      setDailyExporting(false)
    }
  }

  // ── Top Users ──

  const loadTopUsage = useCallback(async () => {
    setTopLoading(true)
    try {
      const result = await tldwClient.getTopUsage({ limit: 20 })
      setTopUsage(Array.isArray(result) ? result : result?.data ?? [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setTopLoading(false)
    }
  }, [markAdminGuardFromError])

  const handleExportTopCsv = async () => {
    setTopExporting(true)
    try {
      const csv = await tldwClient.exportTopUsageCsv()
      downloadCsv(csv, "top_usage.csv")
      message.success(t("settings:adminUsage.topCsvExported", "Top usage CSV exported"))
    } catch (err: any) {
      message.error(sanitizeAdminErrorMessage(err, t("settings:adminUsage.topCsvExportFailed", "Failed to export top usage CSV")))
    } finally {
      setTopExporting(false)
    }
  }

  // ── LLM Usage ──

  const loadLlmUsage = useCallback(async () => {
    setLlmLoading(true)
    try {
      const result = await tldwClient.getLlmUsage({ limit: 50 })
      setLlmUsage(Array.isArray(result) ? result : result?.data ?? [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setLlmLoading(false)
    }
  }, [markAdminGuardFromError])

  const loadLlmSummary = useCallback(async () => {
    setLlmSummaryLoading(true)
    try {
      const result = await tldwClient.getLlmUsageSummary()
      setLlmSummary(result)
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setLlmSummaryLoading(false)
    }
  }, [markAdminGuardFromError])

  const loadTopSpenders = useCallback(async () => {
    setTopSpendersLoading(true)
    try {
      const result = await tldwClient.getLlmTopSpenders({ limit: 10 })
      setTopSpenders(Array.isArray(result) ? result : result?.data ?? [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setTopSpendersLoading(false)
    }
  }, [markAdminGuardFromError])

  // ── Provider Analytics ──

  const loadProviderAnalytics = useCallback(async () => {
    setProviderLoading(true)
    try {
      const result = await tldwClient.getRouterAnalyticsProviders({ range: dateRange })
      setProviderAnalytics(Array.isArray(result) ? result : result?.data ?? [])
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setProviderLoading(false)
    }
  }, [dateRange, markAdminGuardFromError])

  // ── Initial Load ──

  useEffect(() => {
    if (initialLoadRef.current) return
    initialLoadRef.current = true
    void loadDailyUsage()
    void loadTopUsage()
    void loadLlmUsage()
    void loadLlmSummary()
    void loadTopSpenders()
    void loadProviderAnalytics()
  }, [loadDailyUsage, loadTopUsage, loadLlmUsage, loadLlmSummary, loadTopSpenders, loadProviderAnalytics])

  // Reload when date range changes (after initial load)
  useEffect(() => {
    if (!initialLoadRef.current) return
    void loadDailyUsage()
    void loadProviderAnalytics()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dateRange])

  // ── Table Columns ──

  const dailyColumns = [
    { title: t("settings:adminUsage.colDate", "Date"), dataIndex: "date", key: "date" },
    { title: t("settings:adminUsage.colRequests", "Requests"), dataIndex: "requests", key: "requests", render: (v: number) => v?.toLocaleString() ?? "\u2014" },
    { title: t("settings:adminUsage.colBytesIn", "Bytes In"), dataIndex: "bytes_in", key: "bytes_in", render: (v: number) => v != null ? v.toLocaleString() : "\u2014" },
    { title: t("settings:adminUsage.colBytesOut", "Bytes Out"), dataIndex: "bytes_out", key: "bytes_out", render: (v: number) => v != null ? v.toLocaleString() : "\u2014" },
    { title: t("settings:adminUsage.colErrors", "Errors"), dataIndex: "errors", key: "errors", render: (v: number) => v?.toLocaleString() ?? "0" },
    { title: t("settings:adminUsage.colUniqueUsers", "Unique Users"), dataIndex: "unique_users", key: "unique_users", render: (v: number) => v?.toLocaleString() ?? "\u2014" }
  ]

  const topUsageColumns = [
    { title: t("settings:adminUsage.colUsername", "Username"), dataIndex: "username", key: "username" },
    { title: t("settings:adminUsage.colRequests", "Requests"), dataIndex: "requests", key: "requests", render: (v: number) => v?.toLocaleString() ?? "\u2014" },
    { title: t("settings:adminUsage.colBytes", "Bytes"), dataIndex: "bytes", key: "bytes", render: (v: number) => v != null ? v.toLocaleString() : "\u2014" },
    { title: t("settings:adminUsage.colErrors", "Errors"), dataIndex: "errors", key: "errors", render: (v: number) => v?.toLocaleString() ?? "0" }
  ]

  const llmColumns = [
    { title: t("settings:adminUsage.colProvider", "Provider"), dataIndex: "provider", key: "provider" },
    { title: t("settings:adminUsage.colModel", "Model"), dataIndex: "model", key: "model" },
    { title: t("settings:adminUsage.colTokens", "Tokens"), dataIndex: "tokens", key: "tokens", render: (v: number) => v?.toLocaleString() ?? "\u2014" },
    { title: t("settings:adminUsage.colCost", "Cost"), dataIndex: "cost", key: "cost", render: (v: number) => v != null ? formatUsd(v) : "\u2014" }
  ]

  const topSpenderColumns = [
    { title: t("settings:adminUsage.colUser", "User"), dataIndex: "username", key: "username" },
    { title: t("settings:adminUsage.colTotalTokens", "Total Tokens"), dataIndex: "total_tokens", key: "total_tokens", render: (v: number) => v?.toLocaleString() ?? "\u2014" },
    { title: t("settings:adminUsage.colTotalCost", "Total Cost"), dataIndex: "total_cost", key: "total_cost", render: (v: number) => v != null ? formatUsd(v) : "\u2014" }
  ]

  const providerColumns = [
    { title: t("settings:adminUsage.colProvider", "Provider"), dataIndex: "provider", key: "provider" },
    { title: t("settings:adminUsage.colSuccessRate", "Success Rate"), dataIndex: "success_rate", key: "success_rate", render: (v: number) => v != null ? `${(v * 100).toFixed(1)}%` : "\u2014" },
    { title: t("settings:adminUsage.colAvgLatency", "Avg Latency (ms)"), dataIndex: "avg_latency_ms", key: "avg_latency_ms", render: (v: number) => v != null ? v.toFixed(0) : "\u2014" },
    { title: t("settings:adminUsage.colRequests", "Requests"), dataIndex: "total_requests", key: "total_requests", render: (v: number) => v?.toLocaleString() ?? "\u2014" }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title={t("settings:adminUsage.forbiddenTitle", "Access Denied")}>
        {t("settings:adminUsage.forbiddenBody", "You don't have permission to access usage analytics.")}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title={t("settings:adminUsage.notFoundTitle", "Not Available")}>
        {t("settings:adminUsage.notFoundBody", "Usage analytics is not available on this server.")}
      </Alert>
    )
  }

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminUsage.title", "Usage Analytics")}</h1>
        <Select
          value={dateRange}
          onChange={(val) => setDateRange(val)}
          style={{ width: 140 }}
          options={[
            { value: "7d", label: t("settings:adminUsage.last7Days", "Last 7 days") },
            { value: "30d", label: t("settings:adminUsage.last30Days", "Last 30 days") }
          ]}
        />
      </div>

      {/* Daily Usage Card */}
      <Card
        title={t("settings:adminUsage.dailyCardTitle", "Daily Usage")}
        style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Button size="small" onClick={handleExportDailyCsv} loading={dailyExporting}>
              {t("settings:adminUsage.exportCsv", "Export CSV")}
            </Button>
            <Button size="small" onClick={() => loadDailyUsage()}>
              {t("common:refresh", "Refresh")}
            </Button>
          </Space>
        }
      >
        <Table
          dataSource={dailyUsage}
          columns={dailyColumns}
          rowKey="date"
          loading={dailyLoading}
          pagination={false}
          size="small"
        />
      </Card>

      {/* Top Users Card */}
      <Card
        title={t("settings:adminUsage.topUsersCardTitle", "Top Users")}
        style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Button size="small" onClick={handleExportTopCsv} loading={topExporting}>
              {t("settings:adminUsage.exportCsv", "Export CSV")}
            </Button>
            <Button size="small" onClick={() => loadTopUsage()}>
              {t("common:refresh", "Refresh")}
            </Button>
          </Space>
        }
      >
        <Table
          dataSource={topUsage}
          columns={topUsageColumns}
          rowKey="username"
          loading={topLoading}
          pagination={false}
          size="small"
        />
      </Card>

      {/* LLM Usage Card */}
      <Card
        title={t("settings:adminUsage.llmCardTitle", "LLM Usage")}
        style={{ marginBottom: 16 }}
        extra={
          <Button size="small" onClick={() => { void loadLlmUsage(); void loadLlmSummary(); void loadTopSpenders() }}>
            {t("common:refresh", "Refresh")}
          </Button>
        }
      >
        {/* Summary stats */}
        {llmSummary && (
          <Space size="large" style={{ marginBottom: 16 }}>
            <Statistic
              title={t("settings:adminUsage.totalTokens", "Total Tokens")}
              value={llmSummary.total_tokens ?? llmSummary.totalTokens ?? 0}
              loading={llmSummaryLoading}
            />
            <Statistic
              title={t("settings:adminUsage.totalCost", "Total Cost")}
              value={formatUsd(llmSummary.total_cost ?? llmSummary.totalCost ?? 0)}
              loading={llmSummaryLoading}
            />
          </Space>
        )}

        <Table
          dataSource={llmUsage}
          columns={llmColumns}
          rowKey={(r) => `${r.provider}-${r.model}`}
          loading={llmLoading}
          pagination={false}
          size="small"
          style={{ marginBottom: 16 }}
        />

        {/* Top Spenders sub-table */}
        {topSpenders.length > 0 && (
          <>
            <h4 style={{ marginTop: 8, marginBottom: 8 }}>{t("settings:adminUsage.topSpenders", "Top Spenders")}</h4>
            <Table
              dataSource={topSpenders}
              columns={topSpenderColumns}
              rowKey="username"
              loading={topSpendersLoading}
              pagination={false}
              size="small"
            />
          </>
        )}
      </Card>

      {/* Provider Analytics (Collapsible) */}
      <Collapse
        items={[
          {
            key: "provider-analytics",
            label: t("settings:adminUsage.providerAnalytics", "Provider Analytics"),
            children: (
              <>
                <Space style={{ marginBottom: 12 }}>
                  <Button size="small" onClick={() => loadProviderAnalytics()}>
                    {t("common:refresh", "Refresh")}
                  </Button>
                </Space>
                <Table
                  dataSource={providerAnalytics}
                  columns={providerColumns}
                  rowKey="provider"
                  loading={providerLoading}
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

export default UsageAnalyticsPage
