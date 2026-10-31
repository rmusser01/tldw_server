/**
 * Which optional sampling settings the selected chat model's provider accepts
 * (#3283). Providers reject request fields they don't support, so the chat
 * request sends Top K and Min P only where the server's model catalog lists
 * them (`sampling_fields`), and the settings controls say when they aren't.
 */
import { chatModelReadinessKey } from "@/utils/chat-model-readiness"
import { formatChatProviderName } from "@/utils/chat-model-label"
import { splitChatModelId } from "@/utils/chat-model-id"

type CatalogModel = {
  model?: unknown
  provider?: unknown
  sampling_fields?: unknown
}

export type SamplingSupport = {
  /** True or false when the catalog says; null when it doesn't (not sent). */
  topK: boolean | null
  minP: boolean | null
  /** The provider's display name, for "Not supported by <provider>". */
  providerName: string
}

const UNKNOWN: SamplingSupport = { topK: null, minP: null, providerName: "" }

const findSelectedModel = (
  models: readonly CatalogModel[],
  selectedModel: string
): CatalogModel | undefined => {
  const selectedKey = chatModelReadinessKey(undefined, selectedModel)
  const exact = models.find(
    (model) => chatModelReadinessKey(model.provider, model.model) === selectedKey
  )
  if (exact) return exact
  // A selection without a provider ("tldw:gpt-4o") matches a unique model id.
  const selectedId = splitChatModelId(selectedModel).modelId.toLowerCase()
  const byId = models.filter(
    (model) =>
      splitChatModelId(model.model, model.provider).modelId.toLowerCase() === selectedId
  )
  return byId.length === 1 ? byId[0] : undefined
}

export const resolveSamplingSupport = (
  models: readonly CatalogModel[] | null | undefined,
  selectedModel: string | null | undefined
): SamplingSupport => {
  const selected = selectedModel?.trim()
  if (!models?.length || !selected) return UNKNOWN
  const model = findSelectedModel(models, selected)
  if (!model) return UNKNOWN
  const providerName = formatChatProviderName(model.provider)
  if (!Array.isArray(model.sampling_fields)) return { ...UNKNOWN, providerName }
  return {
    topK: model.sampling_fields.includes("top_k"),
    minP: model.sampling_fields.includes("min_p"),
    providerName
  }
}
