import type { ChatModelSettings } from "@/store/model"

const NUMERIC_CHAT_MODEL_SETTING_KEYS: ReadonlySet<keyof ChatModelSettings> =
  new Set([
    "temperature",
    "topK",
    "topP",
    "numCtx",
    "seed",
    "numGpu",
    "numPredict",
    "minP",
    "repeatLastN",
    "repeatPenalty",
    "tfsZ",
    "numKeep",
    "numThread",
    "historyMessageLimit",
    "llamaThinkingBudgetTokens"
  ])

export function normalizeCurrentChatModelSettingValue<
  K extends keyof ChatModelSettings
>(key: K, value: ChatModelSettings[K] | string | null | undefined) {
  if (!NUMERIC_CHAT_MODEL_SETTING_KEYS.has(key)) {
    return value as ChatModelSettings[K]
  }

  if (value == null) return undefined
  if (typeof value === "number") {
    return Number.isFinite(value) ? (value as ChatModelSettings[K]) : undefined
  }
  if (typeof value !== "string") return value as ChatModelSettings[K]

  const trimmed = value.trim()
  if (!trimmed) return undefined

  const parsed = Number(trimmed)
  return Number.isFinite(parsed)
    ? (parsed as ChatModelSettings[K])
    : undefined
}
