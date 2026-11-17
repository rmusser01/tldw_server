import {
  defineSetting,
  getSetting,
  setSetting
} from "@/services/settings/registry"

export const SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "webp"] as const
export type ImageOutputFormat = (typeof SUPPORTED_IMAGE_FORMATS)[number]

export type ImageBackendConfig = {
  negativePrompt?: string
  width?: number
  height?: number
  steps?: number
  cfgScale?: number
  seed?: number
  sampler?: string
  model?: string
  format?: ImageOutputFormat
  extraParams?: string | Record<string, unknown>
}

export type ImageBackendConfigMap = Record<string, ImageBackendConfig>

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const coerceConfigMap = (value: unknown): ImageBackendConfigMap => {
  if (!isRecord(value)) return {}
  return value as ImageBackendConfigMap
}

const IMAGE_BACKEND_CONFIGS_SETTING = defineSetting(
  "imageBackendConfigs",
  {} as ImageBackendConfigMap,
  coerceConfigMap
)

export const getImageBackendConfigs = async (options?: { throwOnError?: boolean }): Promise<ImageBackendConfigMap> =>
  getSetting(IMAGE_BACKEND_CONFIGS_SETTING, options)

export const setImageBackendConfigs = async (configs: ImageBackendConfigMap) =>
  setSetting(IMAGE_BACKEND_CONFIGS_SETTING, configs)

const normalizeBackendToken = (value: string): string =>
  value.toLowerCase().replace(/[^a-z0-9]/g, "")

const stripBackendPrefix = (value: string): string =>
  value
    .replace(/^tldw:/i, "")
    .replace(/^tldw_server[-_: ]*/i, "")
    .trim()

export const resolveImageBackendConfig = (
  backend: string,
  configs: ImageBackendConfigMap
): ImageBackendConfig | undefined => {
  if (!backend) return undefined
  const direct = configs[backend]
  if (direct) return direct

  const stripped = stripBackendPrefix(backend)
  if (stripped && configs[stripped]) return configs[stripped]

  const targetKey = normalizeBackendToken(stripped || backend)
  if (!targetKey) return undefined
  const matchKey = Object.keys(configs).find((key) => {
    const normalized = normalizeBackendToken(stripBackendPrefix(key) || key)
    return normalized === targetKey
  })
  return matchKey ? configs[matchKey] : undefined
}

const coerceString = (value: unknown): string | undefined => {
  if (typeof value !== "string") return undefined
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : undefined
}

const coerceNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

export const normalizeImageBackendConfig = (
  config?: ImageBackendConfig | null
): ImageBackendConfig => {
  if (!config || !isRecord(config)) return {}

  const formatRaw = coerceString(config.format)
  const normalizedFormat = SUPPORTED_IMAGE_FORMATS.includes(
    formatRaw as ImageOutputFormat
  )
    ? (formatRaw as ImageOutputFormat)
    : undefined

  const normalized: ImageBackendConfig = {
    negativePrompt: coerceString(config.negativePrompt),
    width: coerceNumber(config.width),
    height: coerceNumber(config.height),
    steps: coerceNumber(config.steps),
    cfgScale: coerceNumber(config.cfgScale),
    seed: coerceNumber(config.seed),
    sampler: coerceString(config.sampler),
    model: coerceString(config.model),
    format: normalizedFormat,
    extraParams: config.extraParams
  }

  return normalized
}

export const parseExtraParams = (
  value: ImageBackendConfig["extraParams"]
): Record<string, unknown> | undefined => {
  if (!value) return undefined
  if (isRecord(value)) return value
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return undefined
    try {
      const parsed = JSON.parse(trimmed)
      return isRecord(parsed) ? parsed : undefined
    } catch {
      return undefined
    }
  }
  return undefined
}
