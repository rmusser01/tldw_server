import type { TldwModel } from "./TldwApiClient"

export interface TldwModelsMetadataClient {
  getModelsMetadata(options?: { refreshOpenRouter?: boolean }): Promise<unknown>
  getProviders(): Promise<unknown>
}

interface ProviderAvailability {
  is_configured?: boolean
  provider_enabled?: boolean
  availability?: string
}

function toNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function toOptionalBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object"
}

function isLikelyModelId(value: string): boolean {
  if (/\s/.test(value)) return false
  return /[/:._-]/.test(value)
}

function providerKey(value: unknown): string | null {
  return toNonEmptyString(value)?.toLowerCase() ?? null
}

export function buildProviderAvailabilityMap(
  providersPayload: unknown
): Map<string, ProviderAvailability> {
  const providerAvailability = new Map<string, ProviderAvailability>()
  const providers =
    isRecord(providersPayload) && Array.isArray(providersPayload.providers)
      ? providersPayload.providers
      : []

  for (const provider of providers) {
    if (!isRecord(provider)) continue
    const key = providerKey(provider.name)
    if (!key) continue
    providerAvailability.set(key, {
      is_configured:
        toOptionalBoolean(provider.is_configured) ??
        toOptionalBoolean(provider.configured),
      provider_enabled: toOptionalBoolean(provider.enabled),
      availability: toNonEmptyString(provider.availability) ?? undefined
    })
  }

  return providerAvailability
}

export function normalizeTldwModels(
  metadataPayload: unknown,
  providerAvailability = new Map<string, ProviderAvailability>()
): TldwModel[] {
  const list =
    Array.isArray(metadataPayload) && metadataPayload.length > 0
      ? metadataPayload
      : isRecord(metadataPayload) && Array.isArray(metadataPayload.models)
        ? metadataPayload.models
        : []

  return list.map((entry: unknown) => {
    const m = isRecord(entry) ? entry : {}
    const rawModel =
      toNonEmptyString(m.model) || toNonEmptyString(m.model_id)
    const rawName = toNonEmptyString(m.name)
    const rawId = toNonEmptyString(m.id)
    const canonicalModelId =
      rawModel ||
      (rawName && isLikelyModelId(rawName) ? rawName : null) ||
      rawId ||
      rawName ||
      "unknown-model"
    const displayName =
      rawName && !isLikelyModelId(rawName) && rawName !== canonicalModelId
        ? `${rawName} (${canonicalModelId})`
        : canonicalModelId
    const provider = String(m.provider || "default")
    const inheritedAvailability = providerAvailability.get(
      provider.toLowerCase()
    )
    const capabilities = isRecord(m.capabilities) ? m.capabilities : undefined

    return {
      id: canonicalModelId,
      name: displayName,
      provider,
      description: toNonEmptyString(m.description) ?? undefined,
      capabilities: Array.isArray(m.capabilities)
        ? m.capabilities.map((v: unknown) => String(v))
        : Array.isArray(m.features)
          ? m.features.map((v: unknown) => String(v))
          : capabilities,
      context_length:
        typeof m.context_length === "number"
          ? m.context_length
          : typeof m.context_window === "number"
            ? m.context_window
            : typeof m.contextLength === "number"
              ? m.contextLength
              : undefined,
      vision: Boolean(capabilities?.vision ?? m.vision),
      function_calling: Boolean(
        (capabilities &&
          (capabilities.function_calling || capabilities.tool_use)) ??
          m.function_calling
      ),
      json_output: Boolean(capabilities?.json_mode ?? m.json_output),
      type: typeof m.type === "string" ? m.type : undefined,
      modalities:
        isRecord(m.modalities)
          ? {
              input: Array.isArray(m.modalities.input)
                ? m.modalities.input.map((v: unknown) => String(v))
                : undefined,
              output: Array.isArray(m.modalities.output)
                ? m.modalities.output.map((v: unknown) => String(v))
                : undefined
            }
          : {
              input: Array.isArray(m.input_modality)
                ? m.input_modality.map((v: unknown) => String(v))
                : Array.isArray(m.input_modalities)
                  ? m.input_modalities.map((v: unknown) => String(v))
                  : typeof m.input_modality === "string"
                    ? [String(m.input_modality)]
                    : undefined,
              output: Array.isArray(m.output_modality)
                ? m.output_modality.map((v: unknown) => String(v))
                : Array.isArray(m.output_modalities)
                  ? m.output_modalities.map((v: unknown) => String(v))
                  : typeof m.output_modality === "string"
                    ? [String(m.output_modality)]
                    : undefined
            },
      is_configured:
        toOptionalBoolean(m.is_configured) ??
        toOptionalBoolean(m.provider_configured) ??
        inheritedAvailability?.is_configured,
      provider_enabled:
        toOptionalBoolean(m.provider_enabled) ??
        toOptionalBoolean(m.enabled) ??
        inheritedAvailability?.provider_enabled,
      availability:
        toNonEmptyString(m.availability) ?? inheritedAvailability?.availability
    }
  })
}

export async function getNormalizedTldwModels(
  client: TldwModelsMetadataClient,
  options?: { refreshOpenRouter?: boolean }
): Promise<TldwModel[]> {
  const meta = await client.getModelsMetadata(options)
  let providerAvailability = new Map<string, ProviderAvailability>()

  try {
    providerAvailability = buildProviderAvailabilityMap(await client.getProviders())
  } catch {
    // Older servers may not expose provider listings; keep legacy behavior.
  }

  return normalizeTldwModels(meta, providerAvailability)
}
