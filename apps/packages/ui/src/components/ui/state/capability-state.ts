import type { DesignSystemStateKey } from "@/design-system"
import type { StateAction } from "./ActionGroup"
import type { StatePanelDiagnostic } from "./StatePanel"

export type CapabilityStateKind =
  | "empty"
  | "unavailable"
  | "missing_worker"
  | "auth_required"
  | "permission_denied"
  | "not_configured"
  | "degraded"
  | "unsupported_version"
  | "network_failure"

export type CapabilityStateDescriptor = {
  kind: CapabilityStateKind
  state: DesignSystemStateKey
  title: string
  message: string
  diagnostics?: StatePanelDiagnostic[]
  primaryAction?: StateAction
  secondaryActions?: StateAction[]
}

export type CapabilityDiagnosticInput = {
  method?: string
  endpoint?: string
  status?: number | string
  serverUrl?: string
  rawMessage?: string
}

export type CapabilityStateInput = CapabilityDiagnosticInput & {
  kind: CapabilityStateKind
  featureName: string
  capabilityName?: string
  primaryAction?: StateAction
  secondaryActions?: StateAction[]
}

const KIND_TO_STATE: Record<CapabilityStateKind, DesignSystemStateKey> = {
  empty: "empty",
  unavailable: "unavailable",
  missing_worker: "degraded",
  auth_required: "auth_required",
  permission_denied: "permission_denied",
  not_configured: "setup_required",
  degraded: "degraded",
  unsupported_version: "unavailable",
  network_failure: "unavailable"
}

const statusFromError = (error: unknown): number | undefined => {
  if (!error || typeof error !== "object") {
    return undefined
  }

  const maybeError = error as { status?: unknown; response?: { status?: unknown } }
  const status = maybeError.status ?? maybeError.response?.status

  return typeof status === "number" ? status : undefined
}

const messageFromError = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message
  }

  if (error && typeof error === "object" && "message" in error) {
    const message = (error as { message?: unknown }).message

    return typeof message === "string" ? message : ""
  }

  return typeof error === "string" ? error : ""
}

export const classifyCapabilityError = (error: unknown): CapabilityStateKind => {
  const status = statusFromError(error)

  if (status === 401) return "auth_required"
  if (status === 403) return "permission_denied"
  if (status === 404 || status === 410) return "unavailable"
  if (status && status >= 500) return "unavailable"

  const message = messageFromError(error).toLowerCase()

  if (
    message.includes("fetch failed") ||
    message.includes("failed to fetch") ||
    message.includes("connection refused") ||
    message.includes("network")
  ) {
    return "network_failure"
  }

  if (
    message.includes("not configured") ||
    message.includes("missing config") ||
    message.includes("missing provider") ||
    message.includes("api key")
  ) {
    return "not_configured"
  }

  return "unavailable"
}

export const buildCapabilityDiagnostics = ({
  method,
  endpoint,
  status,
  serverUrl,
  rawMessage
}: CapabilityDiagnosticInput): StatePanelDiagnostic[] | undefined => {
  const diagnostics: StatePanelDiagnostic[] = []

  if (method) {
    diagnostics.push({ label: "Method", value: method })
  }

  if (endpoint) {
    diagnostics.push({ label: "Endpoint", value: endpoint, code: true })
  }

  if (status !== undefined) {
    diagnostics.push({ label: "Status", value: String(status) })
  }

  if (serverUrl) {
    diagnostics.push({ label: "Server URL", value: serverUrl, code: true })
  }

  if (rawMessage) {
    diagnostics.push({ label: "Raw message", value: rawMessage })
  }

  return diagnostics.length > 0 ? diagnostics : undefined
}

const titleForKind = (
  kind: CapabilityStateKind,
  featureName: string
): string => {
  switch (kind) {
    case "empty":
      return `No ${featureName.toLowerCase()} yet`
    case "degraded":
      return `${featureName} are partially available`
    case "missing_worker":
      return `${featureName} need a worker`
    case "auth_required":
      return `${featureName} need credentials`
    case "permission_denied":
      return `${featureName} need permission`
    case "not_configured":
      return `${featureName} need setup`
    case "unsupported_version":
      return `${featureName} need a newer server`
    case "network_failure":
      return `${featureName} cannot reach the server`
    case "unavailable":
    default:
      return `${featureName} are unavailable`
  }
}

const messageForKind = ({
  kind,
  featureName,
  capabilityName
}: Pick<
  CapabilityStateInput,
  "kind" | "featureName" | "capabilityName"
>): string => {
  const capability = capabilityName ?? featureName.toLowerCase()

  switch (kind) {
    case "empty":
      return `The feature is available, but there is no ${featureName.toLowerCase()} data yet.`
    case "degraded":
      return "Some data loaded, but part of this feature is limited."
    case "missing_worker":
      return "A background worker or service required for this feature is not running."
    case "auth_required":
      return "Connect or sign in before this feature can load."
    case "permission_denied":
      return "The current account cannot access this feature."
    case "not_configured":
      return "Required provider, server, or feature setup is missing."
    case "unsupported_version":
      return `The connected server is older or does not expose the ${capability} capability.`
    case "network_failure":
      return "The frontend cannot reach the configured server."
    case "unavailable":
    default:
      return `This server does not expose the ${capability} capability.`
  }
}

export const buildCapabilityState = (
  input: CapabilityStateInput
): CapabilityStateDescriptor => ({
  kind: input.kind,
  state: KIND_TO_STATE[input.kind],
  title: titleForKind(input.kind, input.featureName),
  message: messageForKind(input),
  diagnostics: buildCapabilityDiagnostics(input),
  primaryAction: input.primaryAction,
  secondaryActions: input.secondaryActions
})
