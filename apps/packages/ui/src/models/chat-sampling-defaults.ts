/**
 * Sampling values a chat request carries when the current chat does not set
 * them (XS-16).
 *
 * ChatTldw always sends temperature, top_p and both penalties. When nothing
 * sets them it falls back to CHAT_REQUEST_SAMPLING_FALLBACKS, so these are
 * the values the side panel shows as "Default (x)".
 */
export const CHAT_REQUEST_SAMPLING_FALLBACKS = Object.freeze({
  temperature: 0.7,
  topP: 1,
  frequencyPenalty: 0,
  presencePenalty: 0
})

export type InheritedSampling = {
  temperature: number
  topP: number
  frequencyPenalty: number
  presencePenalty: number
}

type SavedDefaultModelSettings = {
  temperature?: unknown
  topP?: unknown
  frequencyPenalty?: unknown
  presencePenalty?: unknown
}

const finiteOrUndefined = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined

/**
 * What the request sends for a parameter the current chat leaves unset,
 * given the user's saved default model settings.
 *
 * pageAssistModel falls back to the saved defaults for temperature, top_p and
 * both penalties (#3283); ChatTldw sends its own fallback when none is set.
 * Keep this in step with pageAssistModel (models/index.ts);
 * pageAssistModel.sampling-defaults.test.ts fails if the two disagree.
 */
export const resolveInheritedSampling = (
  savedDefaults: SavedDefaultModelSettings | null | undefined
): InheritedSampling => ({
  temperature:
    finiteOrUndefined(savedDefaults?.temperature) ??
    CHAT_REQUEST_SAMPLING_FALLBACKS.temperature,
  topP:
    finiteOrUndefined(savedDefaults?.topP) ??
    CHAT_REQUEST_SAMPLING_FALLBACKS.topP,
  frequencyPenalty:
    finiteOrUndefined(savedDefaults?.frequencyPenalty) ??
    CHAT_REQUEST_SAMPLING_FALLBACKS.frequencyPenalty,
  presencePenalty:
    finiteOrUndefined(savedDefaults?.presencePenalty) ??
    CHAT_REQUEST_SAMPLING_FALLBACKS.presencePenalty
})
