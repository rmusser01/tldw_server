import React from "react"
import { RecoveryCallout, buildCapabilityState } from "@/components/ui/state"

export const ACP_HEALTH_ENDPOINT = "/api/v1/acp/health"

export type ACPHealthSnapshot = {
  overall?: string
  status?: number
  rawMessage?: string
  runner?: {
    status?: string
    agent_type?: string
  }
  agents?: Array<{
    agent_type?: string
    status?: string
  }>
  [key: string]: unknown
}

type ACPPlaygroundRecoveryPredicateInput = {
  healthData?: ACPHealthSnapshot | null
  isHealthLoading: boolean
}

export type ACPPlaygroundRecoveryProps = ACPPlaygroundRecoveryPredicateInput & {
  onRetry: () => void
  serverUrl?: string | null
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const extractHealthMessage = (payload: unknown): string | undefined => {
  if (!isRecord(payload)) return undefined
  const candidate = payload.detail ?? payload.message ?? payload.error
  return typeof candidate === "string" && candidate.trim() ? candidate : undefined
}

export const normalizeACPHealthSnapshot = (
  payload: unknown,
  fallback?: Pick<ACPHealthSnapshot, "status" | "rawMessage" | "overall">
): ACPHealthSnapshot => {
  const data = isRecord(payload) ? payload : {}
  const overall =
    fallback?.overall ??
    (typeof data.overall === "string" && data.overall.trim()
      ? data.overall
      : undefined)

  return {
    ...data,
    overall,
    status: fallback?.status,
    rawMessage: extractHealthMessage(payload) ?? fallback?.rawMessage
  }
}

export const shouldShowAcpPlaygroundRecovery = ({
  healthData,
  isHealthLoading
}: ACPPlaygroundRecoveryPredicateInput): boolean =>
  !isHealthLoading && healthData?.overall === "unavailable"

const buildAcpPlaygroundCapabilityState = (
  healthData: ACPHealthSnapshot,
  serverUrl?: string | null
) => {
  const state = buildCapabilityState({
    featureName: "ACP Playground",
    capabilityName: "ACP session orchestration",
    endpoint: ACP_HEALTH_ENDPOINT,
    method: "GET",
    serverUrl,
    status: healthData.status,
    rawMessage: healthData.rawMessage,
    reason: "unsupported"
  })

  if (state.state === "auth_required" || state.state === "permission_denied") {
    return state
  }

  return {
    ...state,
    title: "ACP Playground is unavailable on this server",
    message: "The connected server does not advertise ACP session orchestration."
  }
}

export const ACPPlaygroundRecovery: React.FC<ACPPlaygroundRecoveryProps> = ({
  healthData,
  isHealthLoading,
  onRetry,
  serverUrl
}) => {
  if (!shouldShowAcpPlaygroundRecovery({ healthData, isHealthLoading })) {
    return null
  }

  const capabilityState = buildAcpPlaygroundCapabilityState(healthData, serverUrl)

  return (
    <RecoveryCallout
      state={capabilityState.state}
      title={capabilityState.title}
      message={capabilityState.message}
      diagnostics={capabilityState.diagnostics}
      primaryAction={{
        label: "Try again",
        onClick: onRetry
      }}
      data-testid="acp-playground-capability-recovery"
    />
  )
}
