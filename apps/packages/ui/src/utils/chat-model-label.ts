/**
 * One way to name a chat model in the UI (CC-06, #3111).
 *
 * The composer chip, the picker, the streaming message header and the context
 * rails used to show the same model as "Provider / mod…", "provider_key:model"
 * or "tldw:model". Everything user-facing goes through these helpers: the model
 * name comes first and the provider's display name second. Raw provider keys
 * and the internal `tldw:` prefix never reach a label.
 */
import { removeModelSuffix } from "@/db/model-provider-utils"
import { getProviderDisplayName } from "@/utils/provider-registry"
import { normalizeProviderAvailabilityKey } from "@/services/tldw/model-provider-availability"
import { parseProviderQualifiedModelSelection } from "@/utils/resolve-api-provider"

type ModelDescriptor = Record<string, unknown> | null | undefined

export type ChatModelLabelParts = {
  /** The model's nickname or id, without any provider prefix. */
  modelName: string
  /** The provider's display name, or "" when the provider is unknown. */
  providerName: string
}

const toText = (value: unknown): string => {
  if (typeof value !== "string" && typeof value !== "number") return ""
  return String(value).trim()
}

const stripInternalPrefix = (value: string): string =>
  value.replace(/^tldw:/i, "").trim()

// "tldw" is the server transport, not a model provider.
const UNKNOWN_PROVIDERS = new Set(["", "unknown", "other", "default", "tldw"])

/** The display name for a provider key such as `custom_openai_api` or `ollama`. */
export const formatChatProviderName = (provider: unknown): string => {
  const raw = toText(provider)
  if (UNKNOWN_PROVIDERS.has(raw.toLowerCase())) return ""
  const direct = getProviderDisplayName(raw)
  if (direct && direct !== raw) return direct
  const normalized = normalizeProviderAvailabilityKey(raw)
  if (normalized) {
    const viaAlias = getProviderDisplayName(normalized)
    if (viaAlias && viaAlias !== normalized) return viaAlias
  }
  return raw
}

/**
 * Split a model into its display name and provider name. Accepts a catalog
 * descriptor, a provider-qualified id (`ollama:llama3`, `tldw:llama3`) or both.
 */
export const getChatModelLabelParts = ({
  model,
  modelId,
  provider
}: {
  model?: ModelDescriptor
  modelId?: unknown
  provider?: unknown
}): ChatModelLabelParts => {
  const rawId = stripInternalPrefix(
    toText(modelId) || toText(model?.model) || toText(model?.id) || toText(model?.name)
  )
  const selection = parseProviderQualifiedModelSelection(rawId)
  const nickname = stripInternalPrefix(toText(model?.nickname))
  const modelName = nickname || stripInternalPrefix(selection.modelId) || rawId
  const providerCandidates = [
    provider,
    model?.provider,
    (model?.details as Record<string, unknown> | undefined)?.provider,
    selection.provider
  ]
  let providerName = ""
  for (const candidate of providerCandidates) {
    providerName = formatChatProviderName(candidate)
    if (providerName) break
  }
  return { modelName, providerName }
}

/** "gemma3:1b (Ollama)": the one-line model label used outside the picker. */
export const formatChatModelLabel = (
  input: Parameters<typeof getChatModelLabelParts>[0]
): string => {
  const { modelName, providerName } = getChatModelLabelParts(input)
  if (!modelName) return providerName
  return providerName ? `${modelName} (${providerName})` : modelName
}

const ACCOUNT_MODEL_PATH = /accounts\/[^/]+\/models\//g

/**
 * The name in an assistant message header. A streaming reply is named after
 * the selected model key (`custom_openai_api:mock-gpt-4o`), a finished one
 * after the model; both read as the model name. Character and assistant names
 * pass through unchanged.
 */
export const formatChatMessageAuthorName = (name: unknown): string => {
  const raw = removeModelSuffix(toText(name).replace(ACCOUNT_MODEL_PATH, ""))
  const internal = /^tldw:/i.test(raw)
  const selection = parseProviderQualifiedModelSelection(stripInternalPrefix(raw))
  if (!internal && !selection.isProviderQualified) return raw
  return removeModelSuffix(
    stripInternalPrefix(selection.modelId).replace(ACCOUNT_MODEL_PATH, "")
  )
}
