import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  buildChatSurfaceScopeKeyFromConfig,
  derivePromptAssistAuthorizationRevision
} from "@/services/chat-surface-scope"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"

export type CallerCapabilities = Readonly<{
  user_id: number
  can_read_scheduled_tasks: boolean
  can_read_notifications: boolean
  can_read_monitoring_alerts: boolean
}>

export type CallerCapabilityDecision =
  | "allowed"
  | "denied"
  | "unknown"
  | "unsupported"

export const callerCapabilityErrorStatus = (
  error: unknown
): number | undefined => {
  const status =
    error && typeof error === "object"
      ? (error as { status?: unknown }).status
      : undefined
  return typeof status === "number" ? status : undefined
}

const authorityKey = (config: TldwConfig): string =>
  JSON.stringify([
    config.serverUrl,
    config.authMode,
    config.authSource ?? "manual",
    config.orgId ?? null,
    derivePromptAssistAuthorizationRevision(config)
  ])

/** Discover this request's authenticated owner; no profile or legacy auth/me dependency. */
export async function loadCallerCapabilities(signal: AbortSignal): Promise<{
  capabilities: CallerCapabilities
  scopeKey: string
}> {
  const captured: { config?: TldwConfig } = {}
  let data: unknown
  let failure: unknown
  try {
    data = await tldwClient.requestWithCurrentConfig<unknown>(
      (config: TldwConfig) => {
        signal.throwIfAborted()
        captured.config = { ...config }
        return {
          path: "/api/v1/users/me/capabilities",
          method: "GET",
          abortSignal: signal,
          suppressBackendUnavailableEvent: true,
          expectedStatuses: [401, 403, 404, 405, 410, 501]
        }
      }
    )
  } catch (error) {
    failure = error
  }
  signal.throwIfAborted()
  if (!captured.config)
    throw failure ?? new Error("Caller authority is unavailable")

  // Re-read effective credentials after transport, including rotation/rejection records.
  const current = await tldwClient.ensureConfigForRequest(true)
  signal.throwIfAborted()
  if (authorityKey(current) !== authorityKey(captured.config)) {
    throw new Error("Caller authority changed during capability discovery")
  }
  if (failure) throw failure

  const value =
    data && typeof data === "object" ? (data as Record<string, unknown>) : null
  if (
    !value ||
    typeof value.user_id !== "number" ||
    !Number.isSafeInteger(value.user_id) ||
    value.user_id <= 0 ||
    [
      "can_read_scheduled_tasks",
      "can_read_notifications",
      "can_read_monitoring_alerts"
    ].some((key) => typeof value[key] !== "boolean")
  ) {
    throw new Error("Invalid caller capability response")
  }
  const tokenUser = deriveScopedUserId({
    authMode: "multi-user",
    accessToken: captured.config.accessToken
  })
  if (
    !isHostedTldwDeployment() &&
    captured.config.authMode === "multi-user" &&
    captured.config.authSource !== "cookie-session" &&
    tokenUser !== "user:anonymous" &&
    tokenUser !== `user:${value.user_id}`
  ) {
    throw new Error(
      "Capability response does not match the authenticated caller"
    )
  }
  return {
    capabilities: Object.freeze({
      user_id: value.user_id,
      can_read_scheduled_tasks: value.can_read_scheduled_tasks as boolean,
      can_read_notifications: value.can_read_notifications as boolean,
      can_read_monitoring_alerts: value.can_read_monitoring_alerts as boolean
    }),
    scopeKey: `${buildChatSurfaceScopeKeyFromConfig(captured.config, { userId: value.user_id })}:${captured.config.authSource ?? "manual"}`
  }
}
