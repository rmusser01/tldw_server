import type { ChatModelSettings } from "./model"

export const normalizeModelSettingsScope = (
  provider: unknown,
  model: unknown
): string | null => {
  const normalizedProvider =
    typeof provider === "string" ? provider.trim().toLowerCase() : ""
  const normalizedModel = typeof model === "string" ? model.trim() : ""

  if (!normalizedProvider || !normalizedModel) return null

  return `${normalizedProvider}:${normalizedModel}`
}

export const stripUndefinedScopedSettings = (
  settings: Partial<ChatModelSettings> = {}
): Partial<ChatModelSettings> => {
  const next: Partial<ChatModelSettings> = {}

  for (const [key, value] of Object.entries(settings)) {
    if (value !== undefined) {
      ;(next as Record<string, unknown>)[key] = value
    }
  }

  return next
}

export const mergeGlobalAndScopedSettings = (
  globalDefaults: Partial<ChatModelSettings> = {},
  scopedSettings: Partial<ChatModelSettings> = {}
): ChatModelSettings => ({
  ...stripUndefinedScopedSettings(globalDefaults),
  ...stripUndefinedScopedSettings(scopedSettings)
})
