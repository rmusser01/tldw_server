import {
  getCanonicalModelKey,
  getFavoriteModelKeyCandidates
} from "@/hooks/playground/modelSelectorUtils"
import { toProviderHealthSignal } from "@/services/tldw/model-provider-availability"
import { normalizeKnownApiProviderKey } from "./resolve-api-provider"

type StartupModelDescriptor = {
  model?: unknown
  provider?: unknown
  is_server_default?: unknown
  /** Server-observed provider reachability (see toProviderHealthSignal). */
  provider_health?: unknown
}

type ResolveStartupSelectedModelParams = {
  currentModel: string | null | undefined
  models: StartupModelDescriptor[] | null | undefined
  preferredModelIds?: string[] | null | undefined
  isCurrentModelHydrating?: boolean
  arePreferencesHydrating?: boolean
}

const normalizeModelId = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

export const resolveStartupSelectedModel = ({
  currentModel,
  models,
  preferredModelIds,
  isCurrentModelHydrating = false,
  arePreferencesHydrating = false
}: ResolveStartupSelectedModelParams): string | null => {
  if (isCurrentModelHydrating || arePreferencesHydrating) {
    return null
  }

  if (normalizeModelId(currentModel)) {
    return null
  }

  const modelList = Array.isArray(models) ? models : []
  if (modelList.length === 0) {
    return null
  }

  const preferred = new Set(
    (preferredModelIds || [])
      .map((id) => normalizeModelId(id))
      .filter((id): id is string => Boolean(id))
  )

  const health = (model: StartupModelDescriptor) =>
    toProviderHealthSignal(model?.provider_health)

  // Favourites, then the server default, then a provider the server has seen
  // answer, then the first model not known to be down (CC-01, #3111).
  const choice =
    modelList.find(
      (model) =>
        normalizeModelId(model?.model) &&
        getFavoriteModelKeyCandidates(model).some((key) => preferred.has(key))
    ) ??
    modelList.find(
      (model) =>
        model?.is_server_default === true &&
        normalizeModelId(model.model) &&
        health(model) !== "unreachable"
    ) ??
    modelList.find(
      (model) => normalizeModelId(model?.model) && health(model) === "working"
    ) ??
    modelList.find(
      (model) => normalizeModelId(model?.model) && health(model) !== "unreachable"
    ) ??
    modelList.find((model) => normalizeModelId(model?.model))

  if (!choice) return null
  return normalizeKnownApiProviderKey(choice.provider)
    ? getCanonicalModelKey(choice)
    : normalizeModelId(choice.model)
}
