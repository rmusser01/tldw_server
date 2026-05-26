export const FEATURE_ROLLOUT_SUBJECT_ID_STORAGE_KEY =
  "tldw:feature-rollout:subject-id:v1"

export const FEATURE_ROLLOUT_PERCENTAGE_STORAGE_KEYS = {
  research_workspace_provenance_v1:
    "tldw:feature-rollout:research_workspace_provenance_v1:percentage",
  research_workspace_status_guardrails_v1:
    "tldw:feature-rollout:research_workspace_status_guardrails_v1:percentage",
  watchlists_ia_reduced_nav_v1:
    "tldw:feature-rollout:watchlists_ia_reduced_nav_v1:percentage"
} as const

const ROLLOUT_PERCENT_MIN = 0
const ROLLOUT_PERCENT_MAX = 100
const DEFAULT_ROLLOUT_PERCENTAGE = 100
const DEFAULT_BUCKET_COUNT = 100

const clampRolloutPercentage = (value: number): number =>
  Math.min(
    ROLLOUT_PERCENT_MAX,
    Math.max(ROLLOUT_PERCENT_MIN, Math.floor(value))
  )

const parseRolloutPercentage = (value: unknown): number | null => {
  const parsedValue =
    typeof value === "number"
      ? value
      : typeof value === "string" && value.trim().length > 0
        ? Number(value.trim())
        : Number.NaN
  if (!Number.isFinite(parsedValue)) return null
  return clampRolloutPercentage(parsedValue)
}

export const normalizeRolloutPercentage = (
  value: unknown,
  fallbackPercentage: number = DEFAULT_ROLLOUT_PERCENTAGE
): number => {
  const parsedValue = parseRolloutPercentage(value)
  if (parsedValue != null) return parsedValue

  const fallbackValue =
    typeof fallbackPercentage === "number" && Number.isFinite(fallbackPercentage)
      ? fallbackPercentage
      : DEFAULT_ROLLOUT_PERCENTAGE
  return clampRolloutPercentage(fallbackValue)
}

export const resolveRolloutPercentageFromCandidates = (
  candidates: readonly unknown[],
  fallbackPercentage: number = DEFAULT_ROLLOUT_PERCENTAGE
): number => {
  for (const candidate of candidates) {
    if (candidate == null) continue
    if (typeof candidate === "string" && candidate.trim().length === 0) continue
    const parsedCandidate = parseRolloutPercentage(candidate)
    if (parsedCandidate != null) return parsedCandidate
  }

  return normalizeRolloutPercentage(undefined, fallbackPercentage)
}

const fnv1a32 = (value: string): number => {
  let hash = 2166136261
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

export const computeRolloutBucket = (
  flagKey: string,
  subjectId: string,
  bucketCount: number = DEFAULT_BUCKET_COUNT
): number => {
  const safeBucketCount = Math.max(1, Math.floor(bucketCount))
  const seed = `${flagKey}:${subjectId}`
  return fnv1a32(seed) % safeBucketCount
}

export const isFlagEnabledForRollout = ({
  flagKey,
  subjectId,
  rolloutPercentage
}: {
  flagKey: string
  subjectId: string
  rolloutPercentage: number
}): boolean => {
  const normalizedPercentage = normalizeRolloutPercentage(rolloutPercentage)
  if (normalizedPercentage <= ROLLOUT_PERCENT_MIN) return false
  if (normalizedPercentage >= ROLLOUT_PERCENT_MAX) return true

  return (
    computeRolloutBucket(flagKey, subjectId, DEFAULT_BUCKET_COUNT) <
    normalizedPercentage
  )
}

export const createRolloutSubjectId = (): string =>
  `rs-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
