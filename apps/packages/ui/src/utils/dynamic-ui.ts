import type {
  DynamicUIActionPayload,
  DynamicUIActionUserMetadata,
  DynamicUIEnvelope,
  DynamicUIRendererId
} from "@/types/dynamic-ui"

const SUPPORTED_RENDERERS = new Set<DynamicUIRendererId>(["openui"])
const SENSITIVE_KEY_SEGMENTS = new Set(["password", "token", "secret", "credential", "key", "auth"])
const MAX_ACTION_STRING_LENGTH = 128
const MAX_ACTION_VALUES_BYTES = 16_384
const MAX_JSON_DEPTH = 8

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  if (!isRecord(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

const isBlobLike = (value: unknown): boolean => {
  if (!isRecord(value)) return false
  const tag = Object.prototype.toString.call(value)
  return tag === "[object Blob]" || tag === "[object File]"
}

const isStrictJSONValue = (value: unknown, depth = 0, seen = new WeakSet<object>()): boolean => {
  if (depth > MAX_JSON_DEPTH) return false
  if (value == null) return true
  if (typeof value === "string" || typeof value === "boolean") return true
  if (typeof value === "number") return Number.isFinite(value)
  if (typeof value === "function" || typeof value === "symbol" || typeof value === "undefined") return false
  if (Array.isArray(value)) {
    if (seen.has(value)) return false
    seen.add(value)
    return value.every((entry) => isStrictJSONValue(entry, depth + 1, seen))
  }
  if (!isPlainObject(value) || isBlobLike(value)) return false
  if (Object.getOwnPropertySymbols(value).length > 0) return false
  if (seen.has(value)) return false
  seen.add(value)
  return Object.values(value).every((entry) => isStrictJSONValue(entry, depth + 1, seen))
}

const getSerializedByteLength = (value: string): number => {
  if (typeof TextEncoder === "function") {
    return new TextEncoder().encode(value).byteLength
  }
  return value.length
}

const keySegments = (key: string): string[] =>
  key
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .split(/[^A-Za-z0-9]+/)
    .flatMap((segment) => {
      const normalized = segment.toLowerCase()
      return normalized === "apikey" ? ["api", "key"] : [normalized]
    })
    .filter(Boolean)

const isSensitiveActionValueKey = (key: string): boolean =>
  keySegments(key).some((segment) => SENSITIVE_KEY_SEGMENTS.has(segment))

export const preflightOpenUISource = (source: unknown): { ok: boolean; reason?: string } => {
  if (typeof source !== "string") return { ok: false, reason: "source_not_string" }
  const trimmed = source.trim()
  if (trimmed.length === 0) return { ok: false, reason: "empty_source" }
  if (!/root\s*=/.test(trimmed)) return { ok: false, reason: "missing_root_assignment" }
  if (!/<[A-Z][A-Za-z0-9.]*/.test(trimmed)) return { ok: false, reason: "missing_component_markup" }
  return { ok: true }
}

export const normalizeDynamicUIEnvelope = (value: unknown): DynamicUIEnvelope | null => {
  if (!isRecord(value)) return null
  if (value.renderer !== "openui") return null
  if (!SUPPORTED_RENDERERS.has(value.renderer)) return null
  if (value.version !== "v1") return null

  const preflight = preflightOpenUISource(value.source)
  if (!preflight.ok || typeof value.source !== "string") return null

  return {
    renderer: "openui",
    version: "v1",
    source: value.source.trim(),
    ...(isRecord(value.state) ? { state: value.state } : {}),
    ...(Array.isArray(value.capabilities)
      ? { capabilities: value.capabilities.filter((entry): entry is string => typeof entry === "string") }
      : {})
  }
}

export const buildDynamicUIEnvelope = (
  renderer: DynamicUIRendererId,
  source: string
): DynamicUIEnvelope | null =>
  normalizeDynamicUIEnvelope({ renderer, version: "v1", source })

export const normalizeDynamicUIActionPayload = (
  value: unknown,
  options: { currentMessageIds: Set<string> }
): DynamicUIActionPayload | null => {
  if (!isRecord(value)) return null
  if (value.renderer !== "openui") return null
  if (value.actionType !== "submit") return null

  const sourceMessageId = typeof value.sourceMessageId === "string" ? value.sourceMessageId.trim() : ""
  const actionId = typeof value.actionId === "string" ? value.actionId.trim() : ""
  if (!sourceMessageId || !options.currentMessageIds.has(sourceMessageId)) return null
  if (!actionId || actionId.length > MAX_ACTION_STRING_LENGTH) return null
  if (!isRecord(value.values)) return null
  if (!isStrictJSONValue(value.values)) return null

  let serialized = ""
  try {
    serialized = JSON.stringify(value.values)
  } catch {
    return null
  }
  if (getSerializedByteLength(serialized) > MAX_ACTION_VALUES_BYTES) return null

  return {
    renderer: "openui",
    sourceMessageId,
    actionId,
    actionType: "submit",
    values: value.values
  }
}

export const shouldBlockDynamicUIActionValues = (value: unknown, depth = 0): boolean => {
  if (depth > MAX_JSON_DEPTH) return false
  if (Array.isArray(value)) {
    return value.some((entry) => shouldBlockDynamicUIActionValues(entry, depth + 1))
  }
  if (!isRecord(value)) return false
  return Object.entries(value).some(
    ([key, entry]) =>
      isSensitiveActionValueKey(key) ||
      shouldBlockDynamicUIActionValues(entry, depth + 1)
  )
}

const formatSubmittedValue = (value: unknown): string => {
  if (typeof value === "string") return value
  const serialized = JSON.stringify(value)
  return typeof serialized === "string" ? serialized : String(value)
}

export const formatDynamicUIActionUserMessage = (
  payload: DynamicUIActionUserMetadata
): string => {
  const lines = [`OpenUI action: ${payload.actionType} ${payload.actionId}`, "", "Submitted values:"]
  for (const [key, value] of Object.entries(payload.values)) {
    lines.push(`- ${key}: ${formatSubmittedValue(value)}`)
  }
  return lines.join("\n")
}
