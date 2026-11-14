/**
 * Split a chat model id into its model and provider (CC-06, #3111).
 *
 * A selection key such as `ollama:mistral:latest` names a provider first, but
 * many local model ids look the same: `mistral:latest` and `qwen:7b` are
 * Ollama models whose family shares its name with a provider key. Reading
 * those as "provider mistral, model latest" shows the wrong name and lets two
 * models share one status. A leading `provider:` is therefore dropped only when
 *   - a provider was given and the prefix names that same provider, or
 *   - no provider was given, the prefix is a lowercase provider key, and the
 *     rest is not an Ollama-style tag (`latest`, `7b`, `v0.3`, `instruct`).
 * Text with spaces (a character name such as "Google: Gemini Buddy") is never
 * split. The internal `tldw:` prefix is always dropped.
 */
import { normalizeProviderAvailabilityKey } from "@/services/tldw/model-provider-availability"
import { parseProviderQualifiedModelSelection } from "@/utils/resolve-api-provider"

export type ChatModelIdParts = {
  /** The model id without the `tldw:` prefix or a provider prefix. */
  modelId: string
  /** The provider key from the prefix, when the id named its provider. */
  provider?: string
}

// "tldw" is the server transport, not a model provider.
const PLACEHOLDER_PROVIDERS = new Set(["", "unknown", "other", "default", "tldw"])

// Ollama-style tags: latest, instruct, text, 7b, 0.5b, 8x7b, v0.3, 7b-instruct-q4_K_M.
const LOCAL_MODEL_TAG = /^(?:latest|instruct|text|v?\d[\w.]*(?:-[\w.]+)*)$/i

const toText = (value: unknown): string => {
  if (typeof value !== "string" && typeof value !== "number") return ""
  return String(value).trim()
}

/** The provider key a caller supplied, or "" for none or a placeholder. */
const toProviderHint = (provider: unknown): string => {
  const raw = toText(provider).toLowerCase()
  if (PLACEHOLDER_PROVIDERS.has(raw)) return ""
  return normalizeProviderAvailabilityKey(raw) ?? ""
}

export const stripInternalModelPrefix = (value: string): string =>
  value.replace(/^tldw:/i, "").trim()

export const splitChatModelId = (
  value: unknown,
  provider?: unknown
): ChatModelIdParts => {
  const raw = stripInternalModelPrefix(toText(value))
  const unsplit: ChatModelIdParts = { modelId: raw }
  if (!raw || /\s/.test(raw)) return unsplit
  const selection = parseProviderQualifiedModelSelection(raw)
  if (!selection.isProviderQualified || !selection.provider) return unsplit
  const modelId = stripInternalModelPrefix(selection.modelId)
  if (!modelId) return unsplit

  const hint = toProviderHint(provider)
  if (hint) {
    return normalizeProviderAvailabilityKey(selection.provider) === hint
      ? { modelId, provider: selection.provider }
      : unsplit
  }
  const prefix = raw.slice(0, raw.indexOf(":"))
  if (prefix !== prefix.toLowerCase() || LOCAL_MODEL_TAG.test(modelId)) return unsplit
  return { modelId, provider: selection.provider }
}
