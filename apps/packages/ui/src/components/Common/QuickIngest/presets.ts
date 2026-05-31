import type { IngestPreset, PresetConfig, CommonOptions, TypeDefaults } from "./types"
import { shouldSubmitQuickIngestAdvancedField } from "@/services/tldw/quick-ingest-chunking"

/**
 * Preset configurations for Quick Ingest.
 * Each preset defines a complete set of options for different use cases.
 */
export type PresetMap = Record<Exclude<IngestPreset, "custom">, PresetConfig>

export const DEFAULT_PRESETS: PresetMap = {
  quick: {
    common: {
      perform_analysis: false,
      perform_chunking: false,
      overwrite_existing: false,
      chunking_mode: "auto",
      auto_chunking_goal: "balanced",
      auto_chunking_use_llm: false
    },
    storeRemote: true,
    reviewBeforeStorage: false,
    typeDefaults: {
      audio: { diarize: false },
      document: { ocr: false },
      video: { captions: false }
    },
    advancedValues: {}
  },
  standard: {
    common: {
      perform_analysis: true,
      perform_chunking: true,
      overwrite_existing: false,
      chunking_mode: "auto",
      auto_chunking_goal: "balanced",
      auto_chunking_use_llm: false
    },
    storeRemote: true,
    reviewBeforeStorage: false,
    typeDefaults: {
      audio: { diarize: false },
      document: { ocr: true },
      video: { captions: true }
    },
    advancedValues: {}
  },
  deep: {
    common: {
      perform_analysis: true,
      perform_chunking: true,
      overwrite_existing: true,
      chunking_mode: "auto",
      auto_chunking_goal: "balanced",
      auto_chunking_use_llm: false
    },
    storeRemote: true,
    reviewBeforeStorage: true,
    typeDefaults: {
      audio: { diarize: true },
      document: { ocr: true },
      video: { captions: true }
    },
    advancedValues: {}
  }
}

export const mergePresetConfig = (
  base: PresetConfig,
  override?: Partial<PresetConfig>
): PresetConfig => ({
  common: {
    ...base.common,
    ...(override?.common ?? {})
  },
  storeRemote: override?.storeRemote ?? base.storeRemote,
  reviewBeforeStorage:
    override?.reviewBeforeStorage ?? base.reviewBeforeStorage,
  typeDefaults: {
    audio: {
      ...base.typeDefaults?.audio,
      ...(override?.typeDefaults?.audio ?? {})
    },
    document: {
      ...base.typeDefaults?.document,
      ...(override?.typeDefaults?.document ?? {})
    },
    video: {
      ...base.typeDefaults?.video,
      ...(override?.typeDefaults?.video ?? {})
    }
  },
  advancedValues: {
    ...(base.advancedValues ?? {}),
    ...(override?.advancedValues ?? {})
  }
})

export const resolvePresetMap = (
  overrides?: Partial<PresetMap> | null
): PresetMap => ({
  quick: mergePresetConfig(DEFAULT_PRESETS.quick, overrides?.quick),
  standard: mergePresetConfig(DEFAULT_PRESETS.standard, overrides?.standard),
  deep: mergePresetConfig(DEFAULT_PRESETS.deep, overrides?.deep)
})

/**
 * Metadata for each preset (labels, descriptions, icons).
 * Keys reference i18n translation keys.
 */
export const PRESET_META: Record<
  IngestPreset,
  { labelKey: string; descriptionKey: string; icon: string }
> = {
  quick: {
    labelKey: "preset.quick",
    descriptionKey: "preset.quick.description",
    icon: "⚡"
  },
  standard: {
    labelKey: "preset.standard",
    descriptionKey: "preset.standard.description",
    icon: "★"
  },
  deep: {
    labelKey: "preset.deep",
    descriptionKey: "preset.deep.description",
    icon: "🔬"
  },
  custom: {
    labelKey: "preset.custom",
    descriptionKey: "preset.custom.description",
    icon: "⚙️"
  }
}

/**
 * List of preset keys in display order.
 */
export const PRESET_ORDER: IngestPreset[] = ["quick", "standard", "deep", "custom"]

/**
 * Default preset for new users.
 */
export const DEFAULT_PRESET: Exclude<IngestPreset, "custom"> = "standard"

const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  if (!value || typeof value !== "object") return false
  const proto = Object.getPrototypeOf(value)
  return proto === Object.prototype || proto === null
}

const normalizeAdvancedValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    return value.map(normalizeAdvancedValue)
  }
  if (isPlainObject(value)) {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, entryValue]) => entryValue !== undefined)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, entryValue]) => [key, normalizeAdvancedValue(entryValue)])
    )
  }
  return value
}

const serializeAdvancedValues = (values?: Record<string, unknown>) => {
  return JSON.stringify(normalizeAdvancedValue(values ?? {}))
}

const filterApplicableAdvancedValues = (
  values: Record<string, unknown> | undefined,
  common: CommonOptions,
) =>
  Object.fromEntries(
    Object.entries(values ?? {}).filter(([fieldName]) =>
      shouldSubmitQuickIngestAdvancedField(fieldName, common)
    )
  )

/**
 * Check if current configuration matches a specific preset.
 */
export function configMatchesPreset(
  config: {
    common: CommonOptions
    storeRemote: boolean
    reviewBeforeStorage: boolean
    typeDefaults: TypeDefaults
    advancedValues?: Record<string, unknown>
  },
  presetName: Exclude<IngestPreset, "custom">,
  presetMap: PresetMap = DEFAULT_PRESETS
): boolean {
  const preset = presetMap[presetName]

  // Check common options
  if (
    config.common.perform_analysis !== preset.common.perform_analysis ||
    config.common.perform_chunking !== preset.common.perform_chunking ||
    config.common.overwrite_existing !== preset.common.overwrite_existing ||
    (config.common.chunking_mode ?? "auto") !==
      (preset.common.chunking_mode ?? "auto") ||
    (config.common.auto_chunking_goal ?? "balanced") !==
      (preset.common.auto_chunking_goal ?? "balanced") ||
    Boolean(config.common.auto_chunking_use_llm) !==
      Boolean(preset.common.auto_chunking_use_llm)
  ) {
    return false
  }

  // Check storage options
  if (
    config.storeRemote !== preset.storeRemote ||
    config.reviewBeforeStorage !== preset.reviewBeforeStorage
  ) {
    return false
  }

  // Check type defaults (use ?? false to handle undefined as false)
  const audioDiarize = config.typeDefaults?.audio?.diarize ?? false
  const presetAudioDiarize = preset.typeDefaults?.audio?.diarize ?? false
  if (audioDiarize !== presetAudioDiarize) {
    return false
  }

  const audioLanguage = (config.typeDefaults?.audio?.language || "").trim()
  const presetAudioLanguage = (preset.typeDefaults?.audio?.language || "").trim()
  if (audioLanguage !== presetAudioLanguage) {
    return false
  }

  const documentOcr = config.typeDefaults?.document?.ocr ?? false
  const presetDocumentOcr = preset.typeDefaults?.document?.ocr ?? false
  if (documentOcr !== presetDocumentOcr) {
    return false
  }

  const videoCaptions = config.typeDefaults?.video?.captions ?? false
  const presetVideoCaptions = preset.typeDefaults?.video?.captions ?? false
  if (videoCaptions !== presetVideoCaptions) {
    return false
  }

  if (
    serializeAdvancedValues(
      filterApplicableAdvancedValues(config.advancedValues, config.common)
    ) !==
    serializeAdvancedValues(
      filterApplicableAdvancedValues(preset.advancedValues, preset.common)
    )
  ) {
    return false
  }

  return true
}

/**
 * Detect which preset matches the current configuration.
 * Returns 'custom' if no preset matches.
 */
export function detectPreset(config: {
  common: CommonOptions
  storeRemote: boolean
  reviewBeforeStorage: boolean
  typeDefaults: TypeDefaults
  advancedValues?: Record<string, unknown>
}, presetMap: PresetMap = DEFAULT_PRESETS): IngestPreset {
  for (const name of ["quick", "standard", "deep"] as const) {
    if (configMatchesPreset(config, name, presetMap)) {
      return name
    }
  }
  return "custom"
}

/**
 * Get the configuration for a preset.
 * For 'custom', returns undefined (caller should use current values).
 */
export function getPresetConfig(
  preset: IngestPreset,
  presetMap: PresetMap = DEFAULT_PRESETS
): PresetConfig | undefined {
  if (preset === "custom") {
    return undefined
  }
  return presetMap[preset]
}
