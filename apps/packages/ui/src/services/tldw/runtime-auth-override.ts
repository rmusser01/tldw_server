import { isPlaceholderApiKey } from "@/utils/api-key"

let runtimeSingleUserApiKey: string | null = null

const normalizeApiKey = (value?: string | null): string | null => {
  const normalized = String(value || "").trim()
  if (!normalized || /\s/.test(normalized)) return null
  if (isPlaceholderApiKey(normalized)) return null
  return normalized
}

export const setRuntimeSingleUserApiKeyOverride = (
  value?: string | null
): void => {
  runtimeSingleUserApiKey = normalizeApiKey(value)
}

export const getRuntimeSingleUserApiKeyOverride = (): string | null =>
  runtimeSingleUserApiKey

export const clearRuntimeAuthOverride = (): void => {
  runtimeSingleUserApiKey = null
}
