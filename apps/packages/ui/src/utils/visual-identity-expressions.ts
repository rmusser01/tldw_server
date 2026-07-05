export const VISUAL_IDENTITY_CUSTOM_EXPRESSION_PREFIX = "custom:"

export const VISUAL_IDENTITY_EXPRESSION_OPTIONS = [
  { key: "neutral", label: "Neutral" },
  { key: "happy", label: "Happy" },
  { key: "excited", label: "Excited" },
  { key: "sad", label: "Sad" },
  { key: "angry", label: "Angry" },
  { key: "thinking", label: "Thinking" },
  { key: "confused", label: "Confused" },
  { key: "surprised", label: "Surprised" }
] as const

export type VisualIdentityCanonicalExpressionKey =
  (typeof VISUAL_IDENTITY_EXPRESSION_OPTIONS)[number]["key"]

export type VisualIdentityExpressionKey =
  | VisualIdentityCanonicalExpressionKey
  | `${typeof VISUAL_IDENTITY_CUSTOM_EXPRESSION_PREFIX}${string}`

const CANONICAL_EXPRESSION_KEYS = new Set<string>(
  VISUAL_IDENTITY_EXPRESSION_OPTIONS.map((option) => option.key)
)

const VISUAL_IDENTITY_EXPRESSION_ALIASES: Record<
  string,
  VisualIdentityCanonicalExpressionKey
> = {
  default: "neutral",
  normal: "neutral",
  calm: "neutral",
  joy: "happy",
  joyful: "happy",
  cheerful: "happy",
  hype: "excited",
  thrilled: "excited",
  upset: "sad",
  sorrowful: "sad",
  mad: "angry",
  annoyed: "angry",
  furious: "angry",
  anger: "angry",
  thoughtful: "thinking",
  pondering: "thinking",
  unsure: "confused",
  puzzled: "confused",
  shocked: "surprised",
  astonished: "surprised"
}

const sanitizeExpressionToken = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")

export const normalizeVisualIdentityExpressionKey = (
  value: unknown
): VisualIdentityExpressionKey | null => {
  if (typeof value !== "string") return null

  const rawValue = value.trim()
  if (!rawValue) return null

  if (rawValue.toLowerCase().startsWith(VISUAL_IDENTITY_CUSTOM_EXPRESSION_PREFIX)) {
    const customToken = sanitizeExpressionToken(
      rawValue.slice(VISUAL_IDENTITY_CUSTOM_EXPRESSION_PREFIX.length)
    )
    return customToken
      ? `${VISUAL_IDENTITY_CUSTOM_EXPRESSION_PREFIX}${customToken}`
      : null
  }

  const normalized = sanitizeExpressionToken(rawValue)
  if (!normalized) return null
  if (CANONICAL_EXPRESSION_KEYS.has(normalized)) {
    return normalized as VisualIdentityCanonicalExpressionKey
  }
  return (
    VISUAL_IDENTITY_EXPRESSION_ALIASES[normalized] ??
    `${VISUAL_IDENTITY_CUSTOM_EXPRESSION_PREFIX}${normalized}`
  )
}

export const isVisualIdentityCustomExpressionKey = (value: unknown): boolean => {
  const normalized = normalizeVisualIdentityExpressionKey(value)
  return Boolean(
    normalized?.startsWith(VISUAL_IDENTITY_CUSTOM_EXPRESSION_PREFIX)
  )
}

export const getVisualIdentityExpressionDisplayLabel = (value: unknown): string => {
  const normalized = normalizeVisualIdentityExpressionKey(value)
  if (!normalized) return ""

  const customPrefix = VISUAL_IDENTITY_CUSTOM_EXPRESSION_PREFIX
  const labelSource = normalized.startsWith(customPrefix)
    ? normalized.slice(customPrefix.length)
    : normalized

  return labelSource
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}
