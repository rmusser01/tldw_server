import type { JobOutputPrefs } from "@/types/watchlists"

const TEMPLATE_KEYS = new Set(["default_name", "default_version", "default_format"])
const RETENTION_KEYS = new Set(["default_seconds", "temporary_seconds"])
const DELIVERY_CHANNEL_KEYS = new Set(["email", "chatbook"])
const AUTO_OUTPUT_KEYS = new Set(["enabled", "type", "format", "template_name", "template_version"])
const TOP_LEVEL_KEYS = new Set([
  "generate_audio",
  "audio_voice",
  "audio_speed",
  "target_audio_minutes",
  "background_audio_uri",
  "voice_map",
  "retention_days",
  "template_name",
  "delivery_config"
])

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const cloneValue = <T>(value: T): T => {
  try {
    return JSON.parse(JSON.stringify(value)) as T
  } catch {
    return value
  }
}

const normalizePrefs = (prefs: JobOutputPrefs | null | undefined): Record<string, unknown> =>
  isRecord(prefs) ? cloneValue(prefs) : {}

const mergeKnownNestedPrefs = (
  baseValue: unknown,
  presetValue: unknown,
  presetHasKey: boolean,
  knownKeys: Set<string>
): unknown | undefined => {
  if (!presetHasKey) {
    if (!isRecord(baseValue)) return undefined
    const preserved = Object.fromEntries(
      Object.entries(baseValue)
        .filter(([key]) => !knownKeys.has(key))
        .map(([key, value]) => [key, cloneValue(value)])
    )
    return Object.keys(preserved).length > 0 ? preserved : undefined
  }

  if (presetValue == null) return undefined
  if (!isRecord(presetValue)) return cloneValue(presetValue)

  const preserved = isRecord(baseValue)
    ? Object.fromEntries(
      Object.entries(baseValue)
        .filter(([key]) => !knownKeys.has(key))
        .map(([key, value]) => [key, cloneValue(value)])
    )
    : {}

  for (const [key, value] of Object.entries(presetValue)) {
    preserved[key] = cloneValue(value)
  }
  return Object.keys(preserved).length > 0 ? preserved : undefined
}

export const applyOutputPresetToPrefs = ({
  baseOutputPrefs,
  presetOutputPrefs
}: {
  baseOutputPrefs?: JobOutputPrefs | null
  presetOutputPrefs?: JobOutputPrefs | null
}): JobOutputPrefs => {
  const base = normalizePrefs(baseOutputPrefs)
  const preset = normalizePrefs(presetOutputPrefs)
  const result: Record<string, unknown> = cloneValue(base)

  const nestedSpecs: Array<[string, Set<string>]> = [
    ["template", TEMPLATE_KEYS],
    ["retention", RETENTION_KEYS],
    ["deliveries", DELIVERY_CHANNEL_KEYS],
    ["auto_output", AUTO_OUTPUT_KEYS]
  ]

  for (const [key, knownKeys] of nestedSpecs) {
    const mergedValue = mergeKnownNestedPrefs(
      base[key],
      preset[key],
      Object.prototype.hasOwnProperty.call(preset, key),
      knownKeys
    )
    if (mergedValue === undefined) {
      delete result[key]
    } else {
      result[key] = mergedValue
    }
  }

  for (const key of TOP_LEVEL_KEYS) {
    delete result[key]
    if (Object.prototype.hasOwnProperty.call(preset, key) && preset[key] != null) {
      result[key] = cloneValue(preset[key])
    }
  }

  const handledKeys = new Set([...nestedSpecs.map(([key]) => key), ...TOP_LEVEL_KEYS])
  for (const [key, value] of Object.entries(preset)) {
    if (!handledKeys.has(key)) {
      result[key] = cloneValue(value)
    }
  }

  return result as JobOutputPrefs
}
