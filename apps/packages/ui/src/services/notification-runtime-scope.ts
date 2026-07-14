import {
  buildNotificationScopeKey,
  type NotificationScopeInput
} from "@/services/notification-lifecycle"

export type NotificationRuntimeConfig = NotificationScopeInput

export const parseNotificationRuntimeConfig = (
  value: unknown
): NotificationRuntimeConfig | null => {
  if (!value || typeof value !== "object") return null
  const config = value as Record<string, unknown>
  if (typeof config.serverUrl !== "string" || typeof config.authMode !== "string") {
    return null
  }
  const serverUrl = config.serverUrl.trim()
  const authMode = config.authMode.trim().toLowerCase()
  if (!serverUrl || (authMode !== "single-user" && authMode !== "multi-user")) return null

  if (
    (Object.prototype.hasOwnProperty.call(config, "accessToken") &&
      typeof config.accessToken !== "string") ||
    (Object.prototype.hasOwnProperty.call(config, "apiKey") &&
      typeof config.apiKey !== "string")
  ) {
    return null
  }
  const accessToken = typeof config.accessToken === "string" ? config.accessToken.trim() : ""
  const apiKey = typeof config.apiKey === "string" ? config.apiKey.trim() : ""
  if (authMode === "multi-user" ? !accessToken : !apiKey) return null

  return {
    serverUrl,
    authMode,
    orgId:
      typeof config.orgId === "string" || typeof config.orgId === "number"
        ? config.orgId
        : null,
    userId:
      typeof config.userId === "string" || typeof config.userId === "number"
        ? config.userId
        : null,
    accessToken,
    apiKey
  }
}

export const notificationRecordKeyForConfig = (value: unknown): string | null => {
  const config = parseNotificationRuntimeConfig(value)
  return config ? `tldw:${buildNotificationScopeKey(config)}` : null
}
