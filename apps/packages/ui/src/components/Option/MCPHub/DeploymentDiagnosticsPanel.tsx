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

export const DeploymentDiagnosticsPanel = ({
  env,
  pageOrigin
}: DeploymentDiagnosticsPanelProps) => {
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
        {configIssue ? (
          <Descriptions.Item label="Config issue" span={3}>
            {configIssue}
          </Descriptions.Item>
        ) : null}
      </Descriptions>
    </Card>
  )
}
