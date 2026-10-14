export interface LlamacppServerArgsInput {
  contextSize: number
  gpuLayers: number
  threads?: number
  threadsBatch?: number
  batchSize?: number
  ubatchSize?: number
  mlock?: boolean
  noMmap?: boolean
  noKvOffload?: boolean
  numa?: boolean | "distribute" | "isolate" | "numactl"
  cpuMoe?: boolean
  nCpuMoe?: number
  streamingLlm?: boolean
  splitMode?: "none" | "layer" | "row"
  rowSplit?: boolean
  tensorSplit?: string
  mainGpu?: number
  cacheType?: string
  cacheTypeK?: string
  cacheTypeV?: string
  ropeFreqBase?: number
  ropeFreqScale?: number
  compressPosEmb?: number
  flashAttn?: "auto" | "on" | "off"
  mmproj?: string
  mmprojUrl?: string
  mmprojAuto?: boolean
  mmprojOffload?: boolean
  imageMinTokens?: number
  imageMaxTokens?: number
  draftModel?: string
  draftMax?: number
  draftMin?: number
  draftPMin?: number
  ctxSizeDraft?: number
  gpuLayersDraft?: number
  cpuMoeDraft?: boolean
  nCpuMoeDraft?: number
  host?: string
  port?: number
  extraFlags?: string
  customArgs?: Record<string, any>
}

export interface LlamacppServerArgs {
  [key: string]: any
}

/** Defaults for the llama.cpp launch form; shared by the admin page and panel. */
export const DEFAULT_LLAMACPP_SERVER_ARGS_INPUT: LlamacppServerArgsInput = {
  contextSize: 4096,
  gpuLayers: 0,
  cacheType: "f16",
  splitMode: "layer",
  rowSplit: false,
  mlock: false,
  noMmap: false,
  noKvOffload: false,
  streamingLlm: false,
  cpuMoe: false,
  mmprojAuto: true,
  mmprojOffload: true,
  flashAttn: "auto",
  customArgs: {}
}

const SCALAR_NUMBER_PATTERN = /^-?\d+(\.\d+)?$/

const coerceScalarValue = (raw: string): any => {
  const value = raw.trim()
  if (!value) return value

  const lower = value.toLowerCase()
  if (lower === "true") return true
  if (lower === "false") return false
  if (SCALAR_NUMBER_PATTERN.test(value)) return Number(value)

  if (
    (value.startsWith("{") && value.endsWith("}")) ||
    (value.startsWith("[") && value.endsWith("]"))
  ) {
    try {
      return JSON.parse(value)
    } catch {
      return value
    }
  }

  return value
}

const parseTensorSplit = (value?: string): number[] | string | undefined => {
  if (!value) return undefined
  const normalized = value.trim()
  if (!normalized) return undefined

  const parts = normalized
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)

  if (parts.length === 0) return undefined

  const parsed = parts.map((part) => Number(part))
  if (parsed.every((item) => Number.isFinite(item))) {
    return parsed
  }

  return normalized
}

const normalizeFlagKey = (key: string): string =>
  key
    .trim()
    .replace(/^--?/, "")
    .replace(/-/g, "_")

const parseExtraFlags = (value?: string): Record<string, any> => {
  if (!value?.trim()) return {}

  const parsed: Record<string, any> = {}
  const segments = value
    .split(/[\n,]/)
    .map((segment) => segment.trim())
    .filter(Boolean)

  for (const segment of segments) {
    const eqIndex = segment.indexOf("=")
    if (eqIndex === -1) {
      const key = normalizeFlagKey(segment)
      if (key) parsed[key] = true
      continue
    }

    const key = normalizeFlagKey(segment.slice(0, eqIndex))
    const raw = segment.slice(eqIndex + 1)
    if (!key) continue
    parsed[key] = coerceScalarValue(raw)
  }

  return parsed
}

const normalizeCustomArgKeys = (
  input: Record<string, any> | undefined
): Record<string, any> => {
  if (!input) return {}
  return Object.entries(input).reduce<Record<string, any>>((acc, [rawKey, rawValue]) => {
    const key = normalizeFlagKey(rawKey)
    if (!key) return acc
    acc[key] = rawValue
    return acc
  }, {})
}

export const buildLlamacppServerArgs = ({
  contextSize,
  gpuLayers,
  threads,
  threadsBatch,
  batchSize,
  ubatchSize,
  mlock,
  noMmap,
  noKvOffload,
  numa,
  cpuMoe,
  nCpuMoe,
  streamingLlm,
  splitMode,
  rowSplit,
  tensorSplit,
  mainGpu,
  cacheType,
  cacheTypeK,
  cacheTypeV,
  ropeFreqBase,
  ropeFreqScale,
  compressPosEmb,
  flashAttn,
  mmproj,
  mmprojUrl,
  mmprojAuto,
  mmprojOffload,
  imageMinTokens,
  imageMaxTokens,
  draftModel,
  draftMax,
  draftMin,
  draftPMin,
  ctxSizeDraft,
  gpuLayersDraft,
  cpuMoeDraft,
  nCpuMoeDraft,
  host,
  port,
  extraFlags,
  customArgs
}: LlamacppServerArgsInput): LlamacppServerArgs => {
  const args: LlamacppServerArgs = {
    ctx_size: contextSize,
    n_gpu_layers: gpuLayers
  }

  if (threads !== undefined) args.threads = threads
  if (threadsBatch !== undefined) args.threads_batch = threadsBatch
  if (batchSize !== undefined) args.batch_size = batchSize
  if (ubatchSize !== undefined) args.ubatch_size = ubatchSize

  if (mlock) args.mlock = true
  if (noMmap) args.no_mmap = true
  if (noKvOffload) args.no_kv_offload = true

  if (numa !== undefined) args.numa = numa
  if (cpuMoe) args.cpu_moe = true
  if (nCpuMoe !== undefined) args.n_cpu_moe = nCpuMoe
  if (streamingLlm) args.streaming_llm = true

  if (rowSplit) {
    args.split_mode = "row"
  } else if (splitMode && splitMode !== "layer") {
    args.split_mode = splitMode
  }

  const parsedTensorSplit = parseTensorSplit(tensorSplit)
  if (parsedTensorSplit !== undefined) args.tensor_split = parsedTensorSplit

  if (mainGpu !== undefined) args.main_gpu = mainGpu

  const resolvedCacheTypeK = cacheTypeK || cacheType
  const resolvedCacheTypeV = cacheTypeV || cacheType
  if (resolvedCacheTypeK) args.cache_type_k = resolvedCacheTypeK
  if (resolvedCacheTypeV) args.cache_type_v = resolvedCacheTypeV

  if (ropeFreqBase !== undefined) args.rope_freq_base = ropeFreqBase
  if (ropeFreqScale !== undefined) {
    args.rope_freq_scale = ropeFreqScale
  } else if (compressPosEmb !== undefined && compressPosEmb > 0) {
    args.rope_freq_scale = 1 / compressPosEmb
  }

  if (flashAttn && flashAttn !== "auto") args.flash_attn = flashAttn

  if (mmproj?.trim()) args.mmproj = mmproj.trim()
  if (mmprojUrl?.trim()) args.mmproj_url = mmprojUrl.trim()
  if (mmprojAuto === false) args.no_mmproj = true
  if (mmprojOffload === false) args.no_mmproj_offload = true
  if (imageMinTokens !== undefined) args.image_min_tokens = imageMinTokens
  if (imageMaxTokens !== undefined) args.image_max_tokens = imageMaxTokens

  if (draftModel?.trim()) args.model_draft = draftModel.trim()
  if (draftMax !== undefined) args.draft_max = draftMax
  if (draftMin !== undefined) args.draft_min = draftMin
  if (draftPMin !== undefined) args.draft_p_min = draftPMin
  if (ctxSizeDraft !== undefined) args.ctx_size_draft = ctxSizeDraft
  if (gpuLayersDraft !== undefined) args.gpu_layers_draft = gpuLayersDraft
  if (cpuMoeDraft) args.cpu_moe_draft = true
  if (nCpuMoeDraft !== undefined) args.n_cpu_moe_draft = nCpuMoeDraft

  if (host?.trim()) args.host = host.trim()
  if (port !== undefined) args.port = port

  const parsedExtraFlags = parseExtraFlags(extraFlags)
  const normalizedCustomArgs = normalizeCustomArgKeys(customArgs)

  return { ...parsedExtraFlags, ...args, ...normalizedCustomArgs }
}
