/** Provider health as reported by the server, reduced to what the UI can claim. */
export type ProviderHealthSignal = "working" | "unreachable" | "unknown"

export type ProviderAvailability = {
  default_model?: string
  health?: ProviderHealthSignal
  is_configured?: boolean
  provider_enabled?: boolean
  availability?: string
  readiness_reason_code?: string
  readiness_message?: string
  chat_provider?: string
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
      readiness_reason_code?: unknown
      readiness_message?: unknown
      chat_provider?: unknown
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
  "openai-compatible": "custom_openai_api",
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
  localllm: "local",
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
  const normalized = noWhitespace.replace(/_/g, "-")
  const customSlot =
    /^(?:custom-openai-api-?|custom-openai-|openai-compatible-|customopenai)(\d{1,2})$/.exec(
      normalized
    ) ?? /^custom-openai(\d{1,2})-api$/.exec(normalized)
  const slot = Number(customSlot?.[1])
  if (slot >= 1 && slot <= 99)
    return slot === 1 ? "custom_openai_api" : `custom_openai_api${slot}`
  return (
    PROVIDER_KEY_ALIASES[noWhitespace] ??
    PROVIDER_KEY_ALIASES[compact] ??
    noWhitespace
  )
}

/**
 * Reduce the server's provider health report to a claim the UI can make.
 * "healthy" counts only once the provider has answered at least once, and a
 * provider whose latest calls failed is unreachable even before the server's
 * periodic re-evaluation changes its status.
 */
export const toProviderHealthSignal = (health: unknown): ProviderHealthSignal => {
  if (typeof health === "string") {
    const status = health.trim().toLowerCase()
    if (status === "working" || status === "unreachable") return status
    return "unknown"
  }
  if (!health || typeof health !== "object") return "unknown"
  const record = health as Record<string, unknown>
  const status = typeof record.status === "string" ? record.status.trim().toLowerCase() : ""
  const successCount = Number(record.success_count ?? 0)
  const consecutiveFailures = Number(record.consecutive_failures ?? 0)
  if (status === "unhealthy" || status === "circuit_open" || status === "open") {
    return "unreachable"
  }
  if (consecutiveFailures > 0) return "unreachable"
  if ((status === "healthy" || status === "degraded") && successCount > 0) {
    return "working"
  }
  return "unknown"
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

const shouldEnrichModelRecord = (model: unknown): boolean => {
  if (!model || typeof model !== "object" || Array.isArray(model)) return false
  const type = toNonEmptyProviderString(
    (model as Record<string, unknown>).type
  )?.toLowerCase()
  // The provider default is needed even when model availability is complete.
  return !type || type === "chat"
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

  const defaultProvider = normalizeProviderAvailabilityKey(
    !Array.isArray(providersPayload) ? providersPayload?.default_provider : null
  )
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
        default_model:
          key === defaultProvider
            ? toNonEmptyProviderString(record?.default_model) ?? undefined
            : undefined,
        is_configured:
          toOptionalProviderBoolean(record?.is_configured) ??
          toOptionalProviderBoolean(record?.configured),
        provider_enabled:
          toOptionalProviderBoolean(record?.provider_enabled) ??
          toOptionalProviderBoolean(record?.enabled),
        availability:
          toNonEmptyProviderString(record?.availability) ?? undefined,
        readiness_reason_code:
          toNonEmptyProviderString(record?.readiness_reason_code) ?? undefined,
        readiness_message:
          toNonEmptyProviderString(record?.readiness_message) ?? undefined,
        chat_provider:
          toNonEmptyProviderString(record?.chat_provider) ?? undefined,
        health: toProviderHealthSignal(record?.health)
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
