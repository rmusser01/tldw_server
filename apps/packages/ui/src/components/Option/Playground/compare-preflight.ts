import { getCanonicalModelKey } from "@/hooks/playground/modelSelectorUtils"

export type CompareModelMeta = {
  capabilities: Set<string>
  contextLength: number | null
}

const toFinitePositiveNumber = (value: unknown): number | null => {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null
  }
  return value
}

export const buildCompareModelMetaById = (
  models: unknown[]
): Map<string, CompareModelMeta> => {
  const map = new Map<string, CompareModelMeta>()
  const safeModels = Array.isArray(models) ? models : []

  for (const rawModel of safeModels) {
    const model = (rawModel ?? {}) as Record<string, any>
    const modelId = String(model?.model || "").trim()
    if (!modelId) continue

    const capabilitiesRaw = Array.isArray(model?.capabilities)
      ? model.capabilities
      : Array.isArray(model?.details?.capabilities)
        ? model.details.capabilities
        : []

    const contextLength =
      toFinitePositiveNumber(model?.context_length) ??
      toFinitePositiveNumber(model?.contextLength) ??
      toFinitePositiveNumber(model?.context_window) ??
      toFinitePositiveNumber(model?.details?.context_length) ??
      toFinitePositiveNumber(model?.details?.contextLength) ??
      toFinitePositiveNumber(model?.details?.context_window)

    const metadata: CompareModelMeta = {
      capabilities: new Set(
        capabilitiesRaw.map((capability: unknown) =>
          String(capability || "").toLowerCase()
        )
      ),
      contextLength
    }
    map.set(modelId, metadata)
    map.set(getCanonicalModelKey(model), metadata)
  }

  return map
}

export const compareModelsSupportCapability = (
  modelIds: string[],
  capability: string,
  modelMetaById: Map<string, CompareModelMeta>
): boolean => {
  if (!Array.isArray(modelIds) || modelIds.length === 0) return false
  const normalizedCapability = String(capability || "").toLowerCase()
  return modelIds.every((modelId) =>
    modelMetaById.get(String(modelId))?.capabilities.has(normalizedCapability)
  )
}

export type CompareCapabilityLabels = {
  vision: string
  tools: string
  streaming: string
  context: string
}

const CONTEXT_WINDOW_MISMATCH_RATIO = 4

export const getCompareCapabilityIncompatibilities = (params: {
  modelIds: string[]
  modelMetaById: Map<string, CompareModelMeta>
  labels: CompareCapabilityLabels
}): string[] => {
  const { modelIds, modelMetaById, labels } = params
  if (!Array.isArray(modelIds) || modelIds.length < 2) return []

  const mismatches: string[] = []
  const capabilityChecks: Array<{ key: keyof CompareCapabilityLabels; label: string }> = [
    { key: "vision", label: labels.vision },
    { key: "tools", label: labels.tools },
    { key: "streaming", label: labels.streaming }
  ]

  for (const capability of capabilityChecks) {
    const withCapability = modelIds.filter((modelId) =>
      modelMetaById.get(String(modelId))?.capabilities.has(capability.key)
    ).length
    if (withCapability > 0 && withCapability < modelIds.length) {
      mismatches.push(capability.label)
    }
  }

  const contextLengths = modelIds
    .map((modelId) => modelMetaById.get(String(modelId))?.contextLength)
    .filter(
      (value): value is number =>
        typeof value === "number" && Number.isFinite(value) && value > 0
    )

  if (contextLengths.length >= 2) {
    const minContext = Math.min(...contextLengths)
    const maxContext = Math.max(...contextLengths)
    if (minContext > 0 && maxContext / minContext >= CONTEXT_WINDOW_MISMATCH_RATIO) {
      mismatches.push(labels.context)
    }
  }

  return mismatches
}
