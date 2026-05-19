import type { DesignSystemStateKey } from "@/design-system"
import type { StatePanelDiagnostic } from "./StatePanel"

export type CapabilityStateReason =
  | "unsupported"
  | "missing_config"
  | "partial"
  | "network"
  | "unknown"

export type CapabilityRecoveryState = Extract<
  DesignSystemStateKey,
  | "unavailable"
  | "setup_required"
  | "auth_required"
  | "permission_denied"
  | "degraded"
  | "error"
>

export type CapabilityStateOptions = {
  featureName: string
  capabilityName: string
  endpoint?: string
  method?: string
  serverUrl?: string | null
  status?: number | string | null
  error?: unknown
  reason?: CapabilityStateReason
  rawMessage?: string
  partialErrors?: string[]
  title?: string
  message?: string
}

export type CapabilityStateDescriptor = {
  state: CapabilityRecoveryState
  title: string
  message: string
  diagnostics?: StatePanelDiagnostic[]
}

const NETWORK_ERROR_PATTERN =
  /(failed to fetch|network error|load failed|err_connection|connection refused|could not establish connection|extension messaging timeout|receiving end does not exist)/i

const lowerFirst = (value: string): string => {
  const trimmed = value.trim()
  if (!trimmed) return trimmed
  return trimmed.charAt(0).toLowerCase() + trimmed.slice(1)
}

const singularDataName = (featureName: string): string => {
  const lower = lowerFirst(featureName)
  if (lower.endsWith(" tasks")) {
    return lower.slice(0, -1)
  }
  return lower
}

const parseStatus = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    const status = Math.trunc(value)
    return status >= 100 && status <= 599 ? status : null
  }
  if (typeof value === "string" && value.trim()) {
    const numeric = Number(value)
    if (Number.isFinite(numeric)) {
      return parseStatus(numeric)
    }
    const match = value.match(/\b([1-5]\d{2})\b/)
    return match ? parseStatus(Number(match[1])) : null
  }
  return null
}

export const getCapabilityErrorStatus = (error: unknown): number | null => {
  const candidate = error as
    | {
        status?: unknown
        statusCode?: unknown
        response?: { status?: unknown }
        message?: unknown
      }
    | null
    | undefined

  return (
    parseStatus(candidate?.status) ??
    parseStatus(candidate?.statusCode) ??
    parseStatus(candidate?.response?.status) ??
    parseStatus(candidate?.message) ??
    parseStatus(error)
  )
}

export const getCapabilityRawMessage = (error: unknown): string | undefined => {
  if (error == null) return undefined
  if (error instanceof Error && error.message) return error.message
  if (typeof error === "string") return error
  const candidate = error as { message?: unknown } | null | undefined
  if (typeof candidate?.message === "string" && candidate.message.trim()) {
    return candidate.message
  }
  try {
    return JSON.stringify(error)
  } catch {
    return String(error)
  }
}

const classifyCapabilityState = (
  options: CapabilityStateOptions,
  status: number | null,
  rawMessage?: string
): CapabilityRecoveryState => {
  if (options.reason === "partial") return "degraded"
  if (options.reason === "missing_config") return "setup_required"
  if (status === 401) return "auth_required"
  if (status === 403) return "permission_denied"
  if (options.reason === "unsupported" || status === 404 || status === 405 || status === 422) {
    return "unavailable"
  }
  if (options.reason === "network" || NETWORK_ERROR_PATTERN.test(rawMessage || "")) {
    return "unavailable"
  }
  return "error"
}

const defaultTitle = (
  options: CapabilityStateOptions,
  state: CapabilityRecoveryState
): string => {
  const feature = options.featureName.trim()
  const lowerFeature = lowerFirst(feature)

  switch (state) {
    case "auth_required":
      return `Sign in before using ${lowerFeature}`
    case "permission_denied":
      return `You do not have access to ${lowerFeature}`
    case "setup_required":
      return `${feature} needs setup`
    case "degraded":
      return `${feature} are partially available`
    case "unavailable":
      if (options.reason === "network") {
        return `Cannot reach ${feature}`
      }
      return NETWORK_ERROR_PATTERN.test(getCapabilityRawMessage(options.error) || "")
        ? `Cannot reach ${feature}`
        : `${feature} are unavailable on this server`
    case "error":
    default:
      return `Unable to load ${lowerFeature}`
  }
}

const defaultMessage = (
  options: CapabilityStateOptions,
  state: CapabilityRecoveryState
): string => {
  const capability = options.capabilityName.trim()
  const dataName = singularDataName(options.featureName)

  switch (state) {
    case "auth_required":
      return "Connect or repair your tldw credentials, then try again."
    case "permission_denied":
      return `Use an account with access to ${capability}.`
    case "setup_required":
      return `Configure ${capability}, then try again.`
    case "degraded":
      return `Some ${dataName} data loaded, but one dependency could not be reached.`
    case "unavailable":
      if (options.reason === "network" || NETWORK_ERROR_PATTERN.test(getCapabilityRawMessage(options.error) || "")) {
        return "The frontend cannot reach the connected server. Check the server and try again."
      }
      return `The connected server does not advertise ${capability}.`
    case "error":
    default:
      return `The ${capability} overview could not be loaded. Try again or open diagnostics.`
  }
}

const pushDiagnostic = (
  diagnostics: StatePanelDiagnostic[],
  label: string,
  value: unknown,
  code = false
) => {
  if (value === null || value === undefined || value === "") return
  diagnostics.push({ label, value: String(value), code })
}

export const buildCapabilityState = (
  options: CapabilityStateOptions
): CapabilityStateDescriptor => {
  const status =
    parseStatus(options.status) ?? getCapabilityErrorStatus(options.error)
  const rawMessage = options.rawMessage ?? getCapabilityRawMessage(options.error)
  const state = classifyCapabilityState(options, status, rawMessage)
  const diagnostics: StatePanelDiagnostic[] = []

  pushDiagnostic(diagnostics, "Request method", options.method)
  pushDiagnostic(diagnostics, "Request path", options.endpoint, true)
  pushDiagnostic(diagnostics, "Configured server URL", options.serverUrl, true)
  pushDiagnostic(diagnostics, "Status", status)
  pushDiagnostic(diagnostics, "Raw message", rawMessage)
  if (options.partialErrors?.length) {
    pushDiagnostic(diagnostics, "Partial errors", options.partialErrors.join("; "))
  }

  return {
    state,
    title: options.title ?? defaultTitle(options, state),
    message: options.message ?? defaultMessage(options, state),
    diagnostics: diagnostics.length > 0 ? diagnostics : undefined
  }
}
