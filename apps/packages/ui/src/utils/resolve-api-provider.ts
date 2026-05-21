import { tldwModels } from "@/services/tldw"
import { inferProviderFromModel } from "@/utils/provider-registry"

type ResolveApiProviderOptions = {
  modelId?: string | null
  explicitProvider?: string | null
  providerHint?: string | null
}

type ResolveExplicitProviderForSelectedModelOptions = {
  currentSelectedModel?: string | null
  requestedSelectedModel?: string | null
  explicitProvider?: string | null
}

export type ProviderQualifiedModelSelection = {
  raw: string
  modelId: string
  provider?: string
  isProviderQualified: boolean
}

const PROVIDER_ALIASES: Record<string, string> = {
  "custom_openai_api": "custom-openai-api",
  "custom-openai-api2": "custom-openai-api-2",
  "custom_openai_api_2": "custom-openai-api-2",
  "local_llm": "local-llm",
  "llamacpp": "llama.cpp"
}

const KNOWN_PROVIDER_KEYS = new Set<string>([
  "openai",
  "anthropic",
  "cohere",
  "groq",
  "qwen",
  "openrouter",
  "deepseek",
  "mistral",
  "google",
  "gemini",
  "huggingface",
  "moonshot",
  "zai",
  "llama.cpp",
  "kobold",
  "ooba",
  "tabbyapi",
  "vllm",
  "local-llm",
  "ollama",
  "aphrodite",
  "mlx",
  "custom-openai-api",
  "custom-openai-api-2",
  "custom",
  "together",
  "xai",
  "siliconflow",
  "volcengine",
  "tencentcloud",
  "alibabacloud",
  "fireworks",
  "novita",
  "chutes",
  "bedrock"
])

const PROVIDER_MODEL_PREFIX_RULES: Array<[string, string]> = [
  ["deepseek", "deepseek"],
  ["moonshot", "moonshot"],
  ["claude", "anthropic"],
  ["gpt-", "openai"],
  ["o1-", "openai"],
  ["o3-", "openai"],
  ["gemini", "google"],
  ["mistral", "mistral"],
  ["qwen", "qwen"],
  ["zai", "zai"],
  ["glm", "zai"],
  ["groq", "groq"]
]

const normalizeProvider = (value: unknown): string => {
  const trimmed = String(value || "").trim().toLowerCase().replace(/\s+/g, "")
  if (!trimmed || trimmed === "unknown") return ""
  return PROVIDER_ALIASES[trimmed] ?? trimmed
}

const normalizeModelId = (value: unknown): string =>
  String(value || "").trim().replace(/^tldw:/i, "")

export const AUTO_MODEL_ID = "auto"

export const isAutoModelId = (value: unknown): boolean =>
  normalizeModelId(value).toLowerCase() === AUTO_MODEL_ID

const normalizeKnownProvider = (value: unknown): string => {
  const normalized = normalizeProvider(value)
  if (!normalized) return ""
  if (!KNOWN_PROVIDER_KEYS.has(normalized)) return ""
  return normalized
}

export const parseProviderQualifiedModelSelection = (
  value: unknown
): ProviderQualifiedModelSelection => {
  const raw = String(value || "").trim()
  const fallback = {
    raw,
    modelId: raw,
    provider: undefined,
    isProviderQualified: false
  }

  const parseCandidate = (
    candidate: string
  ): Omit<ProviderQualifiedModelSelection, "raw"> | null => {
    const separatorIndex = candidate.indexOf(":")
    if (separatorIndex <= 0 || separatorIndex === candidate.length - 1) {
      return null
    }

    const provider = normalizeKnownProvider(candidate.slice(0, separatorIndex))
    if (!provider) return null

    const modelId = candidate.slice(separatorIndex + 1).trim()
    if (!modelId) return null

    return {
      modelId,
      provider,
      isProviderQualified: true
    }
  }

  const parsed = parseCandidate(raw)
  if (parsed) {
    return {
      raw,
      ...parsed
    }
  }

  const normalized = normalizeModelId(raw)
  const normalizedParsed =
    normalized !== raw ? parseCandidate(normalized) : null
  if (normalizedParsed) {
    return {
      raw,
      ...normalizedParsed
    }
  }

  const separatorIndex = raw.indexOf(":")
  if (separatorIndex <= 0 || separatorIndex === raw.length - 1) {
    return fallback
  }

  const provider = normalizeKnownProvider(raw.slice(0, separatorIndex))
  if (!provider) return fallback

  const modelId = raw.slice(separatorIndex + 1).trim()
  if (!modelId) return fallback

  return {
    raw,
    modelId,
    provider,
    isProviderQualified: true
  }
}

export const resolveExplicitProviderForSelectedModel = ({
  currentSelectedModel,
  requestedSelectedModel,
  explicitProvider
}: ResolveExplicitProviderForSelectedModelOptions): string | undefined => {
  const requestedSelection = parseProviderQualifiedModelSelection(
    requestedSelectedModel
  )
  if (requestedSelection.provider) {
    return requestedSelection.provider
  }

  const normalizedExplicitProvider = normalizeProvider(explicitProvider)
  const normalizedRequestedModel = normalizeModelId(requestedSelection.modelId)
  if (!normalizedRequestedModel) {
    return normalizedExplicitProvider || undefined
  }

  const currentSelection = parseProviderQualifiedModelSelection(
    currentSelectedModel
  )
  const normalizedCurrentModel = normalizeModelId(currentSelection.modelId)
  if (!normalizedCurrentModel) {
    return undefined
  }

  if (
    currentSelection.provider &&
    normalizedRequestedModel === normalizedCurrentModel
  ) {
    return currentSelection.provider
  }

  return normalizedExplicitProvider &&
    normalizedRequestedModel === normalizedCurrentModel
    ? normalizedExplicitProvider
    : undefined
}

const inferInlineProvider = (normalizedModelId: string): string => {
  const slashIndex = normalizedModelId.indexOf("/")
  if (slashIndex <= 0) return ""
  const prefix = normalizeProvider(normalizedModelId.slice(0, slashIndex))
  if (!prefix || !KNOWN_PROVIDER_KEYS.has(prefix)) return ""
  return prefix
}

const inferProviderFromPrefix = (normalizedModelId: string): string => {
  const lower = normalizedModelId.toLowerCase()
  for (const [prefix, provider] of PROVIDER_MODEL_PREFIX_RULES) {
    if (lower.startsWith(prefix)) {
      return provider
    }
  }
  return ""
}

const inferProviderFromServerModelCatalog = async (
  normalizedModelId: string
): Promise<string> => {
  if (!normalizedModelId) return ""
  try {
    const models = await tldwModels.getModels()
    const needle = normalizedModelId.toLowerCase()
    const matched = models.find((model) => {
      const id = normalizeModelId(model.id).toLowerCase()
      if (id === needle) return true
      const name = String(model.name || "").trim().toLowerCase()
      return name === needle
    })
    return normalizeProvider(matched?.provider)
  } catch {
    return ""
  }
}

export const resolveApiProviderForModel = async ({
  modelId,
  providerHint,
  explicitProvider
}: ResolveApiProviderOptions): Promise<string | undefined> => {
  const modelSelection = parseProviderQualifiedModelSelection(modelId)
  const explicit =
    modelSelection.provider || normalizeProvider(explicitProvider)
  if (explicit) return explicit

  const rawModelId = modelSelection.modelId
  const normalizedModelId = normalizeModelId(rawModelId)
  const isTldwScopedModel = /^tldw:/i.test(rawModelId)
  if (isAutoModelId(rawModelId)) {
    return undefined
  }

  // For models selected from the server catalog (`tldw:`), trust the
  // catalog provider first. This avoids misrouting OpenRouter namespaced
  // IDs like "anthropic/..." to Anthropic directly.
  if (isTldwScopedModel) {
    const catalogProvider = await inferProviderFromServerModelCatalog(
      normalizedModelId
    )
    if (catalogProvider) return catalogProvider
  }

  const inlineProvider = inferInlineProvider(normalizedModelId)
  if (inlineProvider) return inlineProvider

  if (!isTldwScopedModel) {
    const catalogProvider = await inferProviderFromServerModelCatalog(
      normalizedModelId
    )
    if (catalogProvider) return catalogProvider
  }

  const hint = normalizeProvider(providerHint)
  if (hint) return hint

  const inferredFromRegistry = normalizeKnownProvider(
    inferProviderFromModel(normalizedModelId, "llm")
  )
  if (inferredFromRegistry) return inferredFromRegistry

  const inferredFromPrefix = normalizeKnownProvider(
    inferProviderFromPrefix(normalizedModelId)
  )
  if (inferredFromPrefix) return inferredFromPrefix

  return undefined
}
