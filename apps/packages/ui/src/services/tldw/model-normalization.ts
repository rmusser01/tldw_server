import type { TldwModel } from "./TldwApiClient"
import {
  buildProviderAvailabilityMap,
  normalizeProviderAvailabilityKey,
  shouldFetchProviderAvailability,
  toNonEmptyProviderString,
  toOptionalProviderBoolean,
  type ProviderAvailability,
  type TldwProvidersResponse
} from "./model-provider-availability"

export interface TldwModelsMetadataClient {
  getModelsMetadata(options?: { refreshOpenRouter?: boolean }): Promise<unknown>
  getProviders(): Promise<TldwProvidersResponse>
}

function toNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
}

function isLikelyModelId(value: string): boolean {
  if (/\s/.test(value)) return false
  return /[/:._-]/.test(value)
}

function extractModelMetadataList(metadataPayload: unknown): unknown[] {
  return (
    Array.isArray(metadataPayload) && metadataPayload.length > 0
      ? metadataPayload
      : isRecord(metadataPayload) && Array.isArray(metadataPayload.models)
        ? metadataPayload.models
        : []
  )
}

export function normalizeTldwModels(
  metadataPayload: unknown,
  providerAvailability = new Map<string, ProviderAvailability>()
): TldwModel[] {
  const list = extractModelMetadataList(metadataPayload)

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
    const provider = toNonEmptyString(m.provider) || "default"
    const inheritedAvailability = providerAvailability.get(
      normalizeProviderAvailabilityKey(provider) || ""
    )
    const capabilities = isRecord(m.capabilities) ? m.capabilities : undefined

    return {
      ...m,
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
        toOptionalProviderBoolean(m.is_configured) ??
        toOptionalProviderBoolean(m.provider_configured) ??
        toOptionalProviderBoolean(m.configured) ??
        inheritedAvailability?.is_configured,
      provider_is_configured:
        toOptionalProviderBoolean(m.provider_is_configured) ??
        toOptionalProviderBoolean(m.provider_configured) ??
        toOptionalProviderBoolean(m.configured) ??
        inheritedAvailability?.is_configured,
      provider_enabled:
        toOptionalProviderBoolean(m.provider_enabled) ??
        toOptionalProviderBoolean(m.enabled) ??
        inheritedAvailability?.provider_enabled,
      availability:
        toNonEmptyProviderString(m.availability) ??
        inheritedAvailability?.availability,
      catalog_only: toOptionalProviderBoolean(m.catalog_only)
    }
  })
}

export async function getNormalizedTldwModels(
  client: TldwModelsMetadataClient,
  options?: { refreshOpenRouter?: boolean }
): Promise<TldwModel[]> {
  const meta = await client.getModelsMetadata(options)
  const list = extractModelMetadataList(meta)
  const providerAvailability = shouldFetchProviderAvailability(list)
    ? await buildProviderAvailabilityMap(() => client.getProviders())
    : new Map<string, ProviderAvailability>()

  return normalizeTldwModels(meta, providerAvailability)
}
