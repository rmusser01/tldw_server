import { fetchTtsProviders } from "./audio-providers"
import { tldwClient } from "./TldwApiClient"

export type TldwTtsModel = {
  id: string
  label: string
}

const FALLBACK_MODELS = [
  "tts-1",
  "tts-1-hd",
  "kokoro",
  "kitten_tts",
  "KittenML/kitten-tts-nano-0.8",
  "KittenML/kitten-tts-nano-0.8-int8",
  "KittenML/kitten-tts-micro-0.8",
  "KittenML/kitten-tts-mini-0.8",
  "higgs",
  "dia",
  "chatterbox",
  "chatterbox-emotion",
  "chatterbox-multilingual",
  "chatterbox-turbo",
  "vibevoice",
  "vibevoice_realtime",
  "neutts",
  "pocket_tts",
  "index_tts",
  "index_tts2",
  "lux_tts",
  "qwen3_tts",
  "supertonic",
  "supertonic2",
  "echo_tts"
]

const normalizeModelId = (value: unknown): string | null => {
  if (value === null || value === undefined) return null
  if (typeof value === "string" || typeof value === "number") {
    const id = String(value).trim()
    return id ? id : null
  }
  if (typeof value === "object") {
    const candidate =
      (value as { id?: unknown }).id ??
      (value as { model_id?: unknown }).model_id ??
      (value as { modelId?: unknown }).modelId ??
      (value as { name?: unknown }).name ??
      (value as { label?: unknown }).label ??
      (value as { value?: unknown }).value
    if (candidate !== undefined) {
      return normalizeModelId(candidate)
    }
  }
  return null
}

const collectModelIds = (
  source: unknown,
  seen: Set<string>,
  output: string[]
) => {
  if (!source) return
  if (Array.isArray(source)) {
    for (const entry of source) {
      const id = normalizeModelId(entry)
      if (id && !seen.has(id)) {
        seen.add(id)
        output.push(id)
      }
    }
    return
  }
  if (typeof source === "object") {
    const record = source as Record<string, unknown>
    const hasKnownKey =
      "id" in record ||
      "model_id" in record ||
      "modelId" in record ||
      "name" in record ||
      "label" in record ||
      "value" in record
    if (hasKnownKey) {
      const id = normalizeModelId(record)
      if (id && !seen.has(id)) {
        seen.add(id)
        output.push(id)
      }
      return
    }
    for (const [key, value] of Object.entries(record)) {
      const id = normalizeModelId(value) ?? normalizeModelId(key)
      if (id && !seen.has(id)) {
        seen.add(id)
        output.push(id)
      }
    }
  }
}

const extractModelsFromProviders = (
  providersInfo: Awaited<ReturnType<typeof fetchTtsProviders>> | null,
  seen: Set<string>
): string[] => {
  if (!providersInfo) return []
  const output: string[] = []
  const modelListKeys = [
    "models",
    "model_ids",
    "available_models",
    "supported_models",
    "tts_models",
    "model_list",
    "modelList"
  ]
  const singleModelKeys = [
    "model",
    "model_id",
    "modelId",
    "default_model",
    "defaultModel",
    "default_model_id",
    "defaultModelId",
    "preferred_model",
    "preferredModel"
  ]
  const providers = providersInfo.providers || {}
  for (const entry of Object.values(providers)) {
    if (!entry || typeof entry !== "object") continue
    const record = entry as Record<string, unknown>
    for (const key of modelListKeys) {
      collectModelIds(record[key], seen, output)
    }
    for (const key of singleModelKeys) {
      collectModelIds(record[key], seen, output)
    }
  }
  if (output.length > 0) {
    return output
  }
  for (const key of modelListKeys) {
    collectModelIds((providersInfo as any)?.[key], seen, output)
  }
  for (const key of singleModelKeys) {
    collectModelIds((providersInfo as any)?.[key], seen, output)
  }
  return output
}

export const fetchTldwTtsModels = async (): Promise<TldwTtsModel[]> => {
  const seen = new Set<string>()
  const models: string[] = []

  try {
    const providersInfo = await fetchTtsProviders()
    const providerModels = extractModelsFromProviders(providersInfo, seen)
    if (providerModels.length > 0) {
      return providerModels.map((id) => ({
        id,
        label: id
      }))
    }
  } catch {
    // Ignore and fall back to spec/manual list below.
  }

  try {
    const spec = await tldwClient.getOpenAPISpec()
    const modelSchema =
      spec?.components?.schemas?.OpenAISpeechRequest?.properties?.model

    if (modelSchema) {
      if (Array.isArray(modelSchema.enum)) {
        for (const v of modelSchema.enum) {
          const id = String(v).trim()
          if (id && !seen.has(id)) {
            seen.add(id)
            models.push(id)
          }
        }
      } else if (typeof modelSchema.description === "string") {
        const desc = modelSchema.description as string
        const match = desc.match(/Supported models?:\s*([^\.]+)/i)
        if (match && match[1]) {
          const parts = match[1]
            .split(",")
            .map((s) =>
              s
                .trim()
                .replace(/^and\s+/i, "")
                .replace(/[.)]+$/g, "")
            )
            .filter(Boolean)
          for (const p of parts) {
            const id = p
            if (id && !seen.has(id)) {
              seen.add(id)
              models.push(id)
            }
          }
        }
      }
    }
  } catch {
    // Ignore and fall back to static list below.
  }

  for (const id of FALLBACK_MODELS) {
    if (!seen.has(id)) {
      seen.add(id)
      models.push(id)
    }
  }

  return models.map((id) => ({
    id,
    label: id
  }))
}
