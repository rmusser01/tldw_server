import React, { useState, useRef, useCallback, useEffect } from "react"
import { useTranslation } from "react-i18next"
import {
  Card,
  Table,
  Tag,
  Button,
  Space,
  Progress,
  message
} from "antd"
import {
  deriveAdminGuardFromError,
  sanitizeAdminErrorMessage
} from "./admin-error-utils"
import { Alert } from "@/components/ui/primitives"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { tldwClient } from "@/services/tldw/TldwApiClient"

const ADMIN_RATE_LIMITS_PATH = "/api/v1/admin/rate-limits"
const ADMIN_RATE_LIMITS_UNAVAILABLE_MESSAGE =
  "Rate limits listing endpoint is not available on this server."

const RateLimitingPage: React.FC = () => {
  const { t } = useTranslation(["settings", "common"])
  const { config: connectionConfig, loading: connectionConfigLoading } = useCanonicalConnectionConfig()
  // Admin guard state
  const [adminGuard, setAdminGuard] = useState<"forbidden" | "notFound" | null>(null)

  // Governor policy state
  const [policy, setPolicy] = useState<any>(null)
  const [policyLoading, setPolicyLoading] = useState(false)

  // Coverage state
  const [coverage, setCoverage] = useState<any>(null)
  const [coverageLoading, setCoverageLoading] = useState(false)

  // Rate limits state
  const [rateLimits, setRateLimits] = useState<any[]>([])
  const [rateLimitsLoading, setRateLimitsLoading] = useState(false)
  const [rateLimitsError, setRateLimitsError] = useState<string | null>(null)

  const initialLoadRef = useRef(false)
  const rateLimitsSupportedRef = useRef<boolean | null>(null)

  const markAdminGuardFromError = useCallback((err: any) => {
    const guardState = deriveAdminGuardFromError(err)
    if (guardState) setAdminGuard(guardState)
  }, [])

  // ── Governor Policy ──

  const loadPolicy = useCallback(async () => {
    setPolicyLoading(true)
    try {
      const result = await tldwClient.getGovernorPolicy()
      setPolicy(result)
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setPolicyLoading(false)
    }
  }, [markAdminGuardFromError])

  // ── Coverage Audit ──

  const loadCoverage = useCallback(async () => {
    setCoverageLoading(true)
    try {
      // Request the full route lists - the endpoint's default caps them at
      // 50, which silently hid most of a 558-route audit (#2890).
      const result = await tldwClient.getGovernorCoverage({ limit: 5000 })
      setCoverage(result)
    } catch (err) {
      markAdminGuardFromError(err)
    } finally {
      setCoverageLoading(false)
    }
  }, [markAdminGuardFromError])

  // ── Admin Rate Limits ──

  const loadRateLimits = useCallback(async () => {
    if (connectionConfigLoading) {
      return
    }
    setRateLimitsLoading(true)
    setRateLimitsError(null)
    try {
      if (rateLimitsSupportedRef.current == null) {
        const serverUrl = connectionConfig?.serverUrl?.trim()
        if (serverUrl) {
          try {
            const response = await fetch(`${serverUrl}/openapi.json`)
            if (response.ok) {
              const spec = await response.json()
              const paths =
                spec && typeof spec === "object" && spec.paths && typeof spec.paths === "object"
                  ? (spec.paths as Record<string, unknown>)
                  : null
              rateLimitsSupportedRef.current = Boolean(paths && ADMIN_RATE_LIMITS_PATH in paths)
            }
          } catch {
            rateLimitsSupportedRef.current = null
          }
        }
      }

      if (rateLimitsSupportedRef.current === false) {
        setRateLimits([])
        setRateLimitsError(ADMIN_RATE_LIMITS_UNAVAILABLE_MESSAGE)
        return
      }

      const result = await tldwClient.listAdminRateLimits()
      setRateLimits(Array.isArray(result) ? result : [])
    } catch (err: any) {
      // This endpoint may not exist yet; handle gracefully
      const status = err?.status ?? err?.response?.status
      if (status === 404 || status === 405) {
        rateLimitsSupportedRef.current = false
        setRateLimits([])
        setRateLimitsError(ADMIN_RATE_LIMITS_UNAVAILABLE_MESSAGE)
      } else {
        markAdminGuardFromError(err)
      }
    } finally {
      setRateLimitsLoading(false)
    }
  }, [connectionConfig?.serverUrl, connectionConfigLoading, markAdminGuardFromError])

  // ── Initial Load ──

  useEffect(() => {
    if (initialLoadRef.current || connectionConfigLoading) return
    initialLoadRef.current = true
    void loadPolicy()
    void loadCoverage()
    void loadRateLimits()
  }, [connectionConfigLoading, loadPolicy, loadCoverage, loadRateLimits])

  // ── Coverage Table Columns ──

  // The diag endpoint returns protected_routes/unprotected_routes (+ counts);
  // older builds used protected/unprotected. Reading the wrong keys rendered
  // "78.9% coverage" above "0 routes | 0 routes" (2026-09 audit finding P11).
  const protectedRoutes = coverage?.protected_routes ?? coverage?.protected ?? []
  const unprotectedRoutes =
    coverage?.unprotected_routes ?? coverage?.unprotected ?? []
  const protectedCount = coverage?.protected_count ?? protectedRoutes.length
  const unprotectedCount = coverage?.unprotected_count ?? unprotectedRoutes.length
  const coveragePct = coverage?.coverage_pct ?? (
    protectedRoutes.length + unprotectedRoutes.length > 0
      ? Math.round((protectedRoutes.length / (protectedRoutes.length + unprotectedRoutes.length)) * 100)
      : 0
  )

  const routeColumns = [
    {
      title: t("settings:adminRateLimiting.colRoute", "Route"),
      dataIndex: "route",
      key: "route",
      // The diag endpoint emits { method, path } objects; legacy payloads used
      // plain strings or { route }. Never render a raw object into the cell.
      render: (val: string, record: any) => (
        <code>{val || record?.path || String(record?.route ?? "")}</code>
      )
    },
    {
      title: t("settings:adminRateLimiting.colMethod", "Method"),
      dataIndex: "method",
      key: "method",
      render: (val: string) => val ? <Tag>{val}</Tag> : "\u2014"
    }
  ]

  const rateLimitColumns = [
    {
      title: t("settings:adminRateLimiting.colScope", "Scope"),
      dataIndex: "scope",
      key: "scope",
      render: (val: string) => <Tag color={val === "role" ? "blue" : "green"}>{val || "unknown"}</Tag>
    },
    {
      title: t("settings:adminRateLimiting.colId", "ID"),
      dataIndex: "id",
      key: "id"
    },
    {
      title: t("settings:adminRateLimiting.colResource", "Resource"),
      dataIndex: "resource",
      key: "resource",
      render: (val: string) => <code>{val}</code>
    },
    {
      title: t("settings:adminRateLimiting.colLimitPerMin", "Limit / min"),
      dataIndex: "limit_per_min",
      key: "limit_per_min"
    },
    {
      title: t("settings:adminRateLimiting.colBurst", "Burst"),
      dataIndex: "burst",
      key: "burst"
    }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title={t("settings:adminRateLimiting.forbiddenTitle", "Access Denied")}>
        {t(
          "settings:adminRateLimiting.forbiddenBody",
          "You don't have permission to access rate limiting administration."
        )}
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title={t("settings:adminRateLimiting.notFoundTitle", "Not Available")}>
        {t(
          "settings:adminRateLimiting.notFoundBody",
          "Rate limiting administration is not available on this server."
        )}
      </Alert>
    )
  }

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h1 style={{ marginBottom: 16, fontSize: "1.5rem", fontWeight: 600 }}>{t("settings:adminRateLimiting.title", "Rate Limiting & Resource Governor")}</h1>

      {/* Governor Policy Card */}
      <Card
        title={t("settings:adminRateLimiting.policyCardTitle", "Resource Governor Policy")}
        loading={policyLoading}
        style={{ marginBottom: 16 }}
        extra={
          <Button onClick={() => loadPolicy()} size="small">
            {t("common:refresh", "Refresh")}
          </Button>
        }
      >
        {policy ? (
          <Space orientation="vertical" style={{ width: "100%" }}>
            <div>
              <strong>{t("settings:adminRateLimiting.policyStatus", "Status:")}</strong>{" "}
              <Tag color={policy.status === "ok" ? "green" : policy.status === "unavailable" ? "orange" : "red"}>
                {policy.status || "unknown"}
              </Tag>
            </div>
            <div>
              <strong>{t("settings:adminRateLimiting.policyStore", "Store:")}</strong> {policy.store || "file"}
              {(policy.store || "file") === "file" && (
                <span style={{ color: "var(--color-text-secondary, #888)" }}>
                  {" "}
                  {t(
                    "settings:adminRateLimiting.policyStoreHint",
                    "(policies live in the YAML at [ResourceGovernor] policy_path in Config_Files/config.txt; edit on the server and restart)"
                  )}
                </span>
              )}
            </div>
            <div>
              <strong>{t("settings:adminRateLimiting.policyVersion", "Version:")}</strong> {policy.version ?? "\u2014"}
            </div>
            <div>
              <strong>{t("settings:adminRateLimiting.policyCount", "Policies Count:")}</strong> {policy.policies_count ?? 0}
            </div>
            {policy.policy_ids && (
              <div>
                <strong>{t("settings:adminRateLimiting.policyIds", "Policy IDs:")}</strong>{" "}
                {policy.policy_ids.map((pid: string) => (
                  <Tag key={pid} style={{ marginBottom: 4 }}>{pid}</Tag>
                ))}
              </div>
            )}
          </Space>
        ) : (
          <Alert title={t("settings:adminRateLimiting.policyEmpty", "No policy data loaded yet.")} />
        )}
      </Card>

      {/* Coverage Audit Card */}
      <Card
        title={t("settings:adminRateLimiting.coverageCardTitle", "Endpoint Coverage Audit")}
        loading={coverageLoading}
        style={{ marginBottom: 16 }}
        extra={
          <Button onClick={() => loadCoverage()} size="small">
            {t("common:refresh", "Refresh")}
          </Button>
        }
      >
        {coverage ? (
          <Space orientation="vertical" style={{ width: "100%" }} size="middle">
            <div style={{ maxWidth: 300 }}>
              <strong>{t("settings:adminRateLimiting.coverageLabel", "Coverage:")}</strong>
              <Progress
                percent={coveragePct}
                status={coveragePct >= 80 ? "success" : coveragePct >= 50 ? "normal" : "exception"}
                style={{ marginTop: 4 }}
              />
            </div>
            <div>
              <strong>{t("settings:adminRateLimiting.protectedLabel", "Protected:")}</strong> {protectedCount} routes |{" "}
              <strong>{t("settings:adminRateLimiting.unprotectedLabel", "Unprotected:")}</strong> {unprotectedCount} routes
            </div>
            {unprotectedRoutes.length > 0 && (
              <div>
                <strong>{t("settings:adminRateLimiting.unprotectedRoutes", "Unprotected Routes:")}</strong>
                {unprotectedRoutes.length < unprotectedCount && (
                  // Older servers ignore the limit param and still cap the
                  // list - never present a partial audit as complete (#2890).
                  <div style={{ color: "var(--color-text-secondary, #888)" }}>
                    {t(
                      "settings:adminRateLimiting.unprotectedTruncated",
                      "Showing the first {{shown}} of {{total}} unprotected routes reported by the server.",
                      {
                        shown: unprotectedRoutes.length,
                        total: unprotectedCount
                      }
                    )}
                  </div>
                )}
                <Table
                  dataSource={unprotectedRoutes.map((r: any, i: number) =>
                    typeof r === "string" ? { route: r, key: i } : { ...r, key: i }
                  )}
                  columns={routeColumns}
                  pagination={{ pageSize: 25, showSizeChanger: false }}
                  size="small"
                  style={{ marginTop: 8 }}
                />
              </div>
            )}
          </Space>
        ) : (
          <Alert title={t("settings:adminRateLimiting.coverageEmpty", "No coverage data loaded yet.")} />
        )}
      </Card>

      {/* Rate Limits Card */}
      <Card
        title={t("settings:adminRateLimiting.overridesCardTitle", "Per-user rate limit overrides")}
        style={{ marginBottom: 16 }}
        extra={
          <Button onClick={() => loadRateLimits()} size="small">
            {t("common:refresh", "Refresh")}
          </Button>
        }
      >
        <p style={{ marginTop: 0, color: "var(--color-text-secondary, #888)" }}>
          {t(
            "settings:adminRateLimiting.overridesDescription",
            "Overrides created for specific users or API keys. Baseline limits come from the resource governor policy above."
          )}{" "}
          {/* The only UI that creates an override today is the API Keys
              create dialog - link the workflow instead of dead-ending (#2895). */}
          <a href="/admin/api-keys">
            {t(
              "settings:adminRateLimiting.overridesCreateLink",
              "Set a per-key limit when creating an API key in API Keys."
            )}
          </a>
        </p>
        {rateLimitsError ? (
          <Alert title={rateLimitsError} />
        ) : (
          <Table
            dataSource={rateLimits}
            columns={rateLimitColumns}
            rowKey={(record) => `${record.scope}-${record.id}-${record.resource}`}
            loading={rateLimitsLoading}
            pagination={false}
            size="small"
            locale={{
              emptyText: t(
                "settings:adminRateLimiting.overridesEmpty",
                "No per-user overrides configured. The governor policy's baseline limits still apply."
              )
            }}
          />
        )}
      </Card>
    </div>
  )
}

export default RateLimitingPage
