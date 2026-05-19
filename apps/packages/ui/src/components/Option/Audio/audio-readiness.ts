import type { TldwTtsProvidersInfo } from "@/services/tldw/audio-providers"

export type ReadinessState = "ready" | "warning" | "blocked" | "unknown"
export type CapabilityValue = "supported" | "unsupported" | "unknown"
export type MetadataSource = "health" | "static_catalog" | "provider" | "unknown"
export type AvailabilityState = "ready" | "on_demand" | "unavailable" | "unknown"

export type ReadinessItem = {
  id: string
  label: string
  state: ReadinessState
  detail: string
  source?: MetadataSource
}

export type CapabilityDescription = {
  label: string
  tone: "success" | "error" | "default"
}

export type SttModelCatalogEntry = {
  value: string
  label?: string
  description?: string
}

export type SttModelCatalog = {
  categories?: Record<string, SttModelCatalogEntry[]>
  all_models?: string[]
}

export type SttModelHealth = {
  provider?: string | null
  available?: boolean
  usable?: boolean
  on_demand?: boolean
  message?: string | null
}

export type SttModelOption = {
  id: string
  label: string
  description?: string
  category?: string
  provider?: string
  availability: AvailabilityState
  readinessMessage?: string
  capabilities: {
    batch: CapabilityValue
    streaming: CapabilityValue
    diarization: CapabilityValue
    timestamps: CapabilityValue
    segments: CapabilityValue
  }
  sources: Partial<
    Record<
      | keyof SttModelOption["capabilities"]
      | "availability"
      | "label"
      | "description",
      MetadataSource
    >
  >
}

export type BuildSttModelOptionsArgs = {
  catalog?: SttModelCatalog | null
  healthByModel?: Record<string, SttModelHealth | undefined>
}

export type BuildTtsReadinessItemsArgs = {
  provider: string
  hasAudio: boolean
  providersInfo?: TldwTtsProvidersInfo | null
  elevenLabsApiKey?: string | null
}

const UNKNOWN_CAPABILITIES: SttModelOption["capabilities"] = {
  batch: "unknown",
  streaming: "unknown",
  diarization: "unknown",
  timestamps: "unknown",
  segments: "unknown"
}

export function describeCapabilityValue(
  value: CapabilityValue
): CapabilityDescription {
  if (value === "supported") {
    return { label: "Supported", tone: "success" }
  }
  if (value === "unsupported") {
    return { label: "Unsupported", tone: "error" }
  }
  return { label: "Unknown", tone: "default" }
}

const availabilityFromHealth = (
  health?: SttModelHealth
): AvailabilityState => {
  if (!health) return "unknown"
  if (health.on_demand) return "on_demand"
  if (health.usable || health.available) return "ready"
  return "unavailable"
}

export function buildSttModelOptions({
  catalog,
  healthByModel = {}
}: BuildSttModelOptionsArgs): SttModelOption[] {
  const metadataById = new Map<
    string,
    { label?: string; description?: string; category?: string }
  >()

  for (const [category, entries] of Object.entries(catalog?.categories ?? {})) {
    for (const entry of entries ?? []) {
      if (!entry?.value) continue
      metadataById.set(entry.value, {
        label: entry.label,
        description: entry.description,
        category
      })
    }
  }

  const ids = Array.from(
    new Set([
      ...(catalog?.all_models ?? []),
      ...Array.from(metadataById.keys())
    ])
  ).sort()

  return ids.map((id) => {
    const metadata = metadataById.get(id)
    const health = healthByModel[id]
    const availability = availabilityFromHealth(health)
    const sources: SttModelOption["sources"] = {}
    if (metadata?.label) sources.label = "static_catalog"
    if (metadata?.description) sources.description = "static_catalog"
    if (health) sources.availability = "health"

    return {
      id,
      label: metadata?.label || id,
      description: metadata?.description,
      category: metadata?.category,
      provider: health?.provider || undefined,
      availability,
      readinessMessage: health?.message || undefined,
      capabilities: { ...UNKNOWN_CAPABILITIES },
      sources
    }
  })
}

export function buildSttReadinessItems({
  modelOptions,
  loading,
  error
}: {
  modelOptions: SttModelOption[]
  loading: boolean
  error?: string | null
}): ReadinessItem[] {
  if (loading) {
    return [
      {
        id: "stt-models-loading",
        label: "STT models",
        state: "unknown",
        detail: "Loading transcription model catalog."
      }
    ]
  }

  if (error) {
    return [
      {
        id: "stt-models-error",
        label: "STT models",
        state: "warning",
        detail: error
      }
    ]
  }

  if (modelOptions.length === 0) {
    return [
      {
        id: "stt-models-empty",
        label: "STT models",
        state: "blocked",
        detail: "No transcription models reported by the server."
      }
    ]
  }

  const ready = modelOptions.filter((model) => model.availability === "ready").length
  const onDemand = modelOptions.filter(
    (model) => model.availability === "on_demand"
  ).length
  const unavailable = modelOptions.filter(
    (model) => model.availability === "unavailable"
  ).length
  const unknown = modelOptions.filter(
    (model) => model.availability === "unknown"
  ).length
  const state: ReadinessState =
    ready > 0 || onDemand > 0
      ? "ready"
      : unavailable > 0
        ? "blocked"
        : "unknown"
  const detailParts = [
    `${modelOptions.length} listed`,
    ready > 0 ? `${ready} ready` : null,
    onDemand > 0 ? `${onDemand} on demand` : null,
    unavailable > 0 ? `${unavailable} unavailable` : null,
    unknown > 0 ? `${unknown} unknown` : null
  ].filter(Boolean)

  return [
    {
      id: "stt-models-summary",
      label: "STT models",
      state,
      detail: detailParts.join(", "),
      source: ready > 0 || onDemand > 0 || unavailable > 0 ? "health" : "static_catalog"
    }
  ]
}

export function buildTtsReadinessItems({
  provider,
  hasAudio,
  providersInfo,
  elevenLabsApiKey
}: BuildTtsReadinessItemsArgs): ReadinessItem[] {
  const items: ReadinessItem[] = [
    {
      id: "browser-preview",
      label: "Browser preview",
      state: "ready",
      detail: "Available in this browser without server setup."
    }
  ]

  if (provider === "browser") {
    return items
  }

  if (provider === "elevenlabs" && !elevenLabsApiKey) {
    items.push({
      id: "elevenlabs-credentials",
      label: "ElevenLabs setup",
      state: "blocked",
      detail: "API key required before generation."
    })
    return items
  }

  if (provider === "elevenlabs") {
    items.push({
      id: "elevenlabs-credentials",
      label: "ElevenLabs setup",
      state: "ready",
      detail: "API key saved. Voices and models load directly from ElevenLabs.",
      source: "provider"
    })
    return items
  }

  if (!hasAudio) {
    items.push({
      id: "tts-server-audio",
      label: "Server TTS",
      state: "blocked",
      detail: "tldw audio/speech API not detected."
    })
    return items
  }

  const providerInfo = providersInfo?.providers?.[provider]
  const formats = providerInfo?.formats?.length
    ? ` Formats: ${providerInfo.formats.join(", ")}.`
    : ""
  items.push({
    id: `${provider}-provider`,
    label: providerInfo?.provider_name || provider,
    state: providerInfo || provider === "openai" || provider === "tldw"
      ? "ready"
      : "unknown",
    detail: `Current provider selected.${formats}`,
    source: providerInfo ? "provider" : "unknown"
  })

  return items
}
