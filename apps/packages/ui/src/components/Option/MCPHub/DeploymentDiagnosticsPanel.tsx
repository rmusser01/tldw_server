import { useEffect, useState } from "react"
import { Card, Descriptions, Tag } from "antd"

import {
  buildBrowserHttpBase,
  resolveBrowserTransport,
  resolveBrowserTransportMode
} from "@/services/tldw/browser-networking"
import { useMcpToolsStore } from "@/store/mcp-tools"

type DeploymentDiagnosticsEnv = {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE?: string
  NEXT_PUBLIC_API_URL?: string
}

type DeploymentDiagnosticsPanelProps = {
  env?: DeploymentDiagnosticsEnv
  pageOrigin?: string
}

type ServerReadinessSnapshot = {
  state?: string
  degradedChecks?: string[]
  healthUrl?: string
  httpStatus?: number
  healthStatus?: string
  errorMessage?: string
  checkedAt?: string
}

const SERVER_READINESS_STATE_EVENT = "tldw:server-readiness-state"

const trimTrailingSlash = (value: string): string => value.replace(/\/+$/, "")

const readDeploymentEnv = (): DeploymentDiagnosticsEnv => ({
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL
})

const getPageOrigin = (pageOrigin?: string): string => {
  if (pageOrigin) {
    return trimTrailingSlash(pageOrigin)
  }
  if (typeof window !== "undefined" && window.location?.origin) {
    return trimTrailingSlash(window.location.origin)
  }
  return "unknown"
}

const buildHealthUrl = (apiOrigin: string): string =>
  apiOrigin ? `${trimTrailingSlash(apiOrigin)}/api/v1/health` : "/api/v1/health"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const normalizeReadinessSnapshot = (value: unknown): ServerReadinessSnapshot | null => {
  if (!isRecord(value)) return null
  return {
    state: typeof value.state === "string" ? value.state : undefined,
    degradedChecks: Array.isArray(value.degradedChecks)
      ? value.degradedChecks.filter((item): item is string => typeof item === "string")
      : undefined,
    healthUrl: typeof value.healthUrl === "string" ? value.healthUrl : undefined,
    httpStatus: typeof value.httpStatus === "number" ? value.httpStatus : undefined,
    healthStatus: typeof value.healthStatus === "string" ? value.healthStatus : undefined,
    errorMessage: typeof value.errorMessage === "string" ? value.errorMessage : undefined,
    checkedAt: typeof value.checkedAt === "string" ? value.checkedAt : undefined
  }
}

const readReadinessSnapshot = (): ServerReadinessSnapshot | null => {
  if (typeof window === "undefined") return null
  return normalizeReadinessSnapshot(
    (window as typeof window & { __tldwServerReadinessState?: unknown })
      .__tldwServerReadinessState
  )
}

const getHealthTagColor = (state: string): string => {
  if (state === "healthy" || state === "ok" || state === "ready") return "green"
  if (state === "unhealthy" || state === "blocked") return "red"
  if (state === "degraded" || state === "unavailable") return "orange"
  return "default"
}

export const DeploymentDiagnosticsPanel = ({
  env,
  pageOrigin
}: DeploymentDiagnosticsPanelProps) => {
  const [readinessSnapshot, setReadinessSnapshot] =
    useState<ServerReadinessSnapshot | null>(() => readReadinessSnapshot())
  const healthState = useMcpToolsStore((state) => state.healthState)
  const deploymentEnv = env ?? readDeploymentEnv()
  const resolvedPageOrigin = getPageOrigin(pageOrigin)
  const mode = resolveBrowserTransportMode(
    deploymentEnv.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  )

  let apiOrigin = ""
  let apiOriginLabel = "relative (same origin)"
  let healthUrl = "/api/v1/health"
  let configIssue: string | null = null
  try {
    const transport = resolveBrowserTransport({
      surface: "webui-page",
      deploymentMode: deploymentEnv.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
      pageOrigin: resolvedPageOrigin,
      apiOrigin: deploymentEnv.NEXT_PUBLIC_API_URL
    })
    apiOrigin = buildBrowserHttpBase(transport)
    apiOriginLabel = apiOrigin || "relative (same origin)"
    healthUrl = buildHealthUrl(apiOrigin)
  } catch (err) {
    apiOriginLabel = "not configured"
    healthUrl = "not available"
    configIssue = err instanceof Error ? err.message : "Invalid deployment networking config."
  }

  useEffect(() => {
    setReadinessSnapshot(readReadinessSnapshot())
    const handleReadinessState = (event: Event) => {
      setReadinessSnapshot(
        normalizeReadinessSnapshot(
          (event as CustomEvent<ServerReadinessSnapshot>).detail
        )
      )
    }
    window.addEventListener(SERVER_READINESS_STATE_EVENT, handleReadinessState)
    return () => {
      window.removeEventListener(SERVER_READINESS_STATE_EVENT, handleReadinessState)
    }
  }, [])

  const lastHealthStatus =
    readinessSnapshot?.healthStatus || readinessSnapshot?.state || "unknown"
  const lastStatusCode =
    typeof readinessSnapshot?.httpStatus === "number"
      ? String(readinessSnapshot.httpStatus)
      : "not recorded"

  return (
    <Card size="small" title="Deployment Diagnostics">
      <Descriptions size="small" column={{ xs: 1, sm: 2, md: 3 }}>
        <Descriptions.Item label="Deployment mode">
          <Tag color={mode === "quickstart" ? "blue" : "purple"}>{mode}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="Request mode">
          {mode === "quickstart" ? "same-origin proxy" : "direct API"}
        </Descriptions.Item>
        <Descriptions.Item label="Page origin">
          {resolvedPageOrigin}
        </Descriptions.Item>
        <Descriptions.Item label="API origin">
          {apiOriginLabel}
        </Descriptions.Item>
        <Descriptions.Item label="Health URL">
          {healthUrl}
        </Descriptions.Item>
        <Descriptions.Item label="MCP health">
          <Tag
            color={
              healthState === "healthy"
                ? "green"
                : healthState === "unhealthy"
                  ? "red"
                  : healthState === "unavailable"
                    ? "orange"
                    : "default"
            }
          >
            {healthState}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="Last health status">
          <Tag color={getHealthTagColor(lastHealthStatus)}>
            {lastHealthStatus}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="Last status code">
          {lastStatusCode}
        </Descriptions.Item>
        <Descriptions.Item label="Last checked">
          {readinessSnapshot?.checkedAt || "not recorded"}
        </Descriptions.Item>
        {readinessSnapshot?.healthUrl ? (
          <Descriptions.Item label="Readiness health URL" span={3}>
            {readinessSnapshot.healthUrl}
          </Descriptions.Item>
        ) : null}
        {readinessSnapshot?.errorMessage ? (
          <Descriptions.Item label="Last health error" span={3}>
            {readinessSnapshot.errorMessage}
          </Descriptions.Item>
        ) : null}
        {configIssue ? (
          <Descriptions.Item label="Config issue" span={3}>
            {configIssue}
          </Descriptions.Item>
        ) : null}
      </Descriptions>
    </Card>
  )
}
