export type ProviderAvailability = {
  is_configured?: boolean
  provider_enabled?: boolean
  availability?: string
}

export type TldwProviderEntry =
  | string
  | {
      name?: unknown
      provider?: unknown
      id?: unknown
      is_configured?: unknown
      configured?: unknown
      provider_enabled?: unknown
      enabled?: unknown
      availability?: unknown
      [key: string]: unknown
    }

export type TldwProvidersResponse =
  | TldwProviderEntry[]
  | {
      providers?: TldwProviderEntry[]
      [key: string]: unknown
    }

const PROVIDER_KEY_ALIASES: Record<string, string> = {
  "custom-openai-api": "custom_openai_api",
  custom_openai_api: "custom_openai_api",
  customopenaiapi: "custom_openai_api",
  customopenai: "custom_openai_api",
  "custom-openai-api-2": "custom_openai_api2",
  custom_openai_api_2: "custom_openai_api2",
  custom_openai_api2: "custom_openai_api2",
  customopenaiapi2: "custom_openai_api2",
  customopenai2: "custom_openai_api2",
  gemini: "google",
  "llama.cpp": "llama",
  "llama-cpp": "llama",
  llama_cpp: "llama",
  llamacpp: "llama",
  oobabooga: "ooba",
  tabbyapi: "tabby",
  "z.ai": "zai",
  z_ai: "zai"
}

export const toNonEmptyProviderString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

export const toOptionalProviderBoolean = (
  value: unknown
): boolean | undefined => (typeof value === "boolean" ? value : undefined)

export const normalizeProviderAvailabilityKey = (
  value: unknown
): string | null => {
  const raw = toNonEmptyProviderString(value)?.toLowerCase()
  if (!raw || raw === "unknown") return null

  const noWhitespace = raw.replace(/\s+/g, "")
  const compact = noWhitespace.replace(/[._-]+/g, "")
  return (
    PROVIDER_KEY_ALIASES[noWhitespace] ??
    PROVIDER_KEY_ALIASES[compact] ??
    noWhitespace
  )
}

const asProviderRecord = (
  provider: TldwProviderEntry
): Exclude<TldwProviderEntry, string> | null =>
  provider && typeof provider === "object" ? provider : null

const normalizeProviders = (
  payload: TldwProvidersResponse | null | undefined
): TldwProviderEntry[] => {
  if (Array.isArray(payload)) return payload
  if (payload && Array.isArray(payload.providers)) return payload.providers
  return []
}

const hasOptionalBoolean = (
  record: Record<string, unknown>,
  keys: string[]
): boolean => keys.some((key) => typeof record[key] === "boolean")

const hasNonEmptyString = (
  record: Record<string, unknown>,
  key: string
): boolean => toNonEmptyProviderString(record[key]) != null

const shouldEnrichModelRecord = (model: unknown): boolean => {
  if (!model || typeof model !== "object") return false
  const record = model as Record<string, unknown>
  const type = toNonEmptyProviderString(record.type)?.toLowerCase()
  if (type && type !== "chat") return false

  const hasConfigured = hasOptionalBoolean(record, [
    "is_configured",
    "provider_is_configured",
    "provider_configured",
    "configured"
  ])
  const hasEnabled = hasOptionalBoolean(record, [
    "provider_enabled",
    "enabled"
  ])
  const hasAvailability = hasNonEmptyString(record, "availability")

  return !hasConfigured || !hasEnabled || !hasAvailability
}

export const shouldFetchProviderAvailability = (models: unknown[]): boolean =>
  models.some(shouldEnrichModelRecord)

export const buildProviderAvailabilityMap = async (
  fetchProviders: () => Promise<TldwProvidersResponse>
): Promise<Map<string, ProviderAvailability>> => {
  const providerAvailability = new Map<string, ProviderAvailability>()
  let providersPayload: TldwProvidersResponse | null = null

  try {
    providersPayload = await fetchProviders()
  } catch (error) {
    if (import.meta.env?.DEV) {
      console.warn("tldw_server: provider availability fetch failed", error)
    }
    return providerAvailability
  }

  const providers = normalizeProviders(providersPayload)
  for (const provider of providers) {
    try {
      const record = asProviderRecord(provider)
      const key = normalizeProviderAvailabilityKey(
        typeof provider === "string"
          ? provider
          : record?.name ?? record?.provider ?? record?.id
      )
      if (!key) continue
      providerAvailability.set(key, {
        is_configured:
          toOptionalProviderBoolean(record?.is_configured) ??
          toOptionalProviderBoolean(record?.configured),
        provider_enabled:
          toOptionalProviderBoolean(record?.provider_enabled) ??
          toOptionalProviderBoolean(record?.enabled),
        availability:
          toNonEmptyProviderString(record?.availability) ?? undefined
      })
    } catch (error) {
      if (import.meta.env?.DEV) {
        console.warn("tldw_server: provider availability enrichment failed", {
          error,
          provider
        })
      }
    }
  }

  return providerAvailability
}
