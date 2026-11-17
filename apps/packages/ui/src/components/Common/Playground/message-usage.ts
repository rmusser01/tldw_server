const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value !== "number" || !Number.isFinite(value)) return null
  return value
}

export type MessageUsage = {
  promptTokens: number
  completionTokens: number
  totalTokens: number
}

export const resolveMessageUsage = (generationInfo: unknown): MessageUsage => {
  if (!generationInfo || typeof generationInfo !== "object") {
    return { promptTokens: 0, completionTokens: 0, totalTokens: 0 }
  }

  const payload = generationInfo as Record<string, any>
  const usage = (payload.usage || {}) as Record<string, any>

  const promptTokens =
    toFiniteNumber(payload.prompt_eval_count) ??
    toFiniteNumber(payload.prompt_tokens) ??
    toFiniteNumber(payload.input_tokens) ??
    toFiniteNumber(usage.prompt_tokens) ??
    toFiniteNumber(usage.input_tokens) ??
    0
  const completionTokens =
    toFiniteNumber(payload.eval_count) ??
    toFiniteNumber(payload.completion_tokens) ??
    toFiniteNumber(payload.output_tokens) ??
    toFiniteNumber(usage.completion_tokens) ??
    toFiniteNumber(usage.output_tokens) ??
    0
  const totalTokens =
    toFiniteNumber(payload.total_tokens) ??
    toFiniteNumber(payload.total_token_count) ??
    toFiniteNumber(usage.total_tokens) ??
    promptTokens + completionTokens

  return {
    promptTokens: Math.max(0, Math.round(promptTokens)),
    completionTokens: Math.max(0, Math.round(completionTokens)),
    totalTokens: Math.max(0, Math.round(totalTokens))
  }
}

export const resolveMessageCostUsd = (generationInfo: unknown): number | null => {
  if (!generationInfo || typeof generationInfo !== "object") {
    return null
  }

  const payload = generationInfo as Record<string, any>
  const usage = (payload.usage || {}) as Record<string, any>
  const pricing = (payload.pricing || {}) as Record<string, any>
  const cost =
    toFiniteNumber(payload.total_cost_usd) ??
    toFiniteNumber(payload.estimated_cost_usd) ??
    toFiniteNumber(payload.cost_usd) ??
    toFiniteNumber(payload.total_cost) ??
    toFiniteNumber(payload.estimated_cost) ??
    toFiniteNumber(usage.total_cost_usd) ??
    toFiniteNumber(usage.estimated_cost_usd) ??
    toFiniteNumber(usage.cost_usd) ??
    toFiniteNumber(pricing.total_cost_usd) ??
    toFiniteNumber(pricing.estimated_cost_usd) ??
    toFiniteNumber(pricing.cost_usd)

  if (cost == null || cost < 0) return null
  return cost
}

export type SavedGenerationMetadata = {
  status?: "complete" | "length" | "interrupted" | "stopped" | "error"
  model?: string
  provider?: string
  finishReason?: string
  usage?: Partial<
    Record<"prompt_tokens" | "completion_tokens" | "total_tokens", number>
  >
}

/** Saved presentation only: never creates recovery flags or history authority. */
export const resolveSavedGenerationMetadata = (
  value: unknown
): SavedGenerationMetadata | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null
  const metadata = value as Record<string, unknown>
  const saved: SavedGenerationMetadata = {}
  if (
    ["complete", "length", "interrupted", "stopped", "error"].includes(
      metadata.generation_status as string
    )
  )
    saved.status =
      metadata.generation_status as SavedGenerationMetadata["status"]

  const model =
    typeof metadata.model_id === "string" ? metadata.model_id.trim() : ""
  if (
    model &&
    [...model].length <= 256 &&
    !/[\p{C}\p{Z}]/u.test(model.replaceAll(" ", ""))
  )
    saved.model = model
  for (const [source, target] of [
    ["provider", "provider"],
    ["finish_reason", "finishReason"]
  ] as const) {
    const raw = metadata[source]
    if (
      typeof raw === "string" &&
      /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}$/.test(raw.trim())
    )
      saved[target] = raw.trim()
  }

  if (
    metadata.usage &&
    typeof metadata.usage === "object" &&
    !Array.isArray(metadata.usage)
  ) {
    const usage = metadata.usage as Record<string, unknown>
    for (const key of [
      "prompt_tokens",
      "completion_tokens",
      "total_tokens"
    ] as const) {
      const count = usage[key]
      if (
        typeof count === "number" &&
        Number.isInteger(count) &&
        count >= 0 &&
        count <= 2 ** 53
      ) {
        saved.usage ??= {}
        saved.usage[key] = count
      }
    }
  }
  return Object.keys(saved).length ? saved : null
}
