import React, { useState, useRef, useCallback, useEffect } from "react"
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
      const result = await tldwClient.getGovernorCoverage()
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

  const protectedRoutes = coverage?.protected ?? []
  const unprotectedRoutes = coverage?.unprotected ?? []
  const coveragePct = coverage?.coverage_pct ?? (
    protectedRoutes.length + unprotectedRoutes.length > 0
      ? Math.round((protectedRoutes.length / (protectedRoutes.length + unprotectedRoutes.length)) * 100)
      : 0
  )

  const routeColumns = [
    {
      title: "Route",
      dataIndex: "route",
      key: "route",
      render: (val: string, record: any) => <code>{val || record}</code>
    },
    {
      title: "Method",
      dataIndex: "method",
      key: "method",
      render: (val: string) => val ? <Tag>{val}</Tag> : "\u2014"
    }
  ]

  const rateLimitColumns = [
    {
      title: "Scope",
      dataIndex: "scope",
      key: "scope",
      render: (val: string) => <Tag color={val === "role" ? "blue" : "green"}>{val || "unknown"}</Tag>
    },
    {
      title: "ID",
      dataIndex: "id",
      key: "id"
    },
    {
      title: "Resource",
      dataIndex: "resource",
      key: "resource",
      render: (val: string) => <code>{val}</code>
    },
    {
      title: "Limit / min",
      dataIndex: "limit_per_min",
      key: "limit_per_min"
    },
    {
      title: "Burst",
      dataIndex: "burst",
      key: "burst"
    }
  ]

  // ── Render ──

  if (adminGuard === "forbidden") {
    return (
      <Alert variant="error" title="Access Denied">
        You don't have permission to access rate limiting administration.
      </Alert>
    )
  }
  if (adminGuard === "notFound") {
    return (
      <Alert variant="warning" title="Not Available">
        Rate limiting administration is not available on this server.
      </Alert>
    )
  }

  return (
    <div style={{ padding: "24px", maxWidth: 1200 }}>
      <h2 style={{ marginBottom: 16 }}>Rate Limiting &amp; Resource Governor</h2>

      {/* Governor Policy Card */}
      <Card
        title="Resource Governor Policy"
        loading={policyLoading}
        style={{ marginBottom: 16 }}
        extra={
          <Button onClick={() => loadPolicy()} size="small">
            Refresh
          </Button>
        }
      >
        {policy ? (
          <Space orientation="vertical" style={{ width: "100%" }}>
            <div>
              <strong>Status:</strong>{" "}
              <Tag color={policy.status === "ok" ? "green" : policy.status === "unavailable" ? "orange" : "red"}>
                {policy.status || "unknown"}
              </Tag>
            </div>
            <div>
              <strong>Store:</strong> {policy.store || "file"}
            </div>
            <div>
              <strong>Version:</strong> {policy.version ?? "\u2014"}
            </div>
            <div>
              <strong>Policies Count:</strong> {policy.policies_count ?? 0}
            </div>
            {policy.policy_ids && (
              <div>
                <strong>Policy IDs:</strong>{" "}
                {policy.policy_ids.map((pid: string) => (
                  <Tag key={pid} style={{ marginBottom: 4 }}>{pid}</Tag>
                ))}
              </div>
            )}
          </Space>
        ) : (
          <Alert title="No policy data loaded yet." />
        )}
      </Card>

      {/* Coverage Audit Card */}
      <Card
        title="Endpoint Coverage Audit"
        loading={coverageLoading}
        style={{ marginBottom: 16 }}
        extra={
          <Button onClick={() => loadCoverage()} size="small">
            Refresh
          </Button>
        }
      >
        {coverage ? (
          <Space orientation="vertical" style={{ width: "100%" }} size="middle">
            <div style={{ maxWidth: 300 }}>
              <strong>Coverage:</strong>
              <Progress
                percent={coveragePct}
                status={coveragePct >= 80 ? "success" : coveragePct >= 50 ? "normal" : "exception"}
                style={{ marginTop: 4 }}
              />
            </div>
            <div>
              <strong>Protected:</strong> {protectedRoutes.length} routes |{" "}
              <strong>Unprotected:</strong> {unprotectedRoutes.length} routes
            </div>
            {unprotectedRoutes.length > 0 && (
              <div>
                <strong>Unprotected Routes:</strong>
                <Table
                  dataSource={unprotectedRoutes.map((r: any, i: number) =>
                    typeof r === "string" ? { route: r, key: i } : { ...r, key: i }
                  )}
                  columns={routeColumns}
                  pagination={false}
                  size="small"
                  style={{ marginTop: 8 }}
                />
              </div>
            )}
          </Space>
        ) : (
          <Alert title="No coverage data loaded yet." />
        )}
      </Card>

      {/* Rate Limits Card */}
      <Card
        title="Admin Rate Limits"
        style={{ marginBottom: 16 }}
        extra={
          <Button onClick={() => loadRateLimits()} size="small">
            Refresh
          </Button>
        }
      >
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
            locale={{ emptyText: "No rate limits configured" }}
          />
        )}
      </Card>
    </div>
  )
}

export default RateLimitingPage
