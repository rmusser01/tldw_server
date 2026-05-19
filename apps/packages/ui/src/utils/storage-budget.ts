export const STORAGE_BUDGET_DEFAULT_MB = 5

export const STORAGE_THRESHOLDS = { warning: 0.80, exceeded: 0.95 } as const

const STORAGE_BUDGET_VITE_ENV = "VITE_WORKSPACE_STORAGE_PAYLOAD_BUDGET_MB"
const STORAGE_BUDGET_NEXT_ENV = "NEXT_PUBLIC_WORKSPACE_STORAGE_PAYLOAD_BUDGET_MB"

/**
 * Estimate localStorage storage cost of a string.
 * localStorage uses UTF-16 internally, so str.length (UTF-16 code units)
 * is the most accurate and performant proxy for quota consumption.
 */
export const estimateStorageCost = (str: string): number => {
  return str.length
}

const parseStorageBudgetCandidateMb = (
  candidate: unknown
): number | null => {
  if (typeof candidate === "number" && Number.isFinite(candidate) && candidate > 0) {
    return candidate
  }
  if (typeof candidate !== "string") return null
  const parsed = Number(candidate.trim())
  if (!Number.isFinite(parsed) || parsed <= 0) return null
  return parsed
}

/** Estimate total localStorage bytes used by keys matching a prefix (default: all keys). */
export const estimateLocalStorageUsageBytes = (
  storage: Storage,
  prefix?: string
): number => {
  let totalBytes = 0
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i)
    if (!key) continue
    if (prefix && !key.startsWith(prefix)) continue
    const value = storage.getItem(key)
    if (value == null) continue
    totalBytes += estimateStorageCost(key) + estimateStorageCost(value)
  }
  return totalBytes
}

/** Resolve the localStorage budget in bytes. Checks env vars, defaults to 5 MB. */
export const resolveStorageBudgetBytes = (): number => {
  const viteEnv = (import.meta as unknown as { env?: Record<string, unknown> }).env
  const viteBudgetMb = parseStorageBudgetCandidateMb(
    viteEnv?.[STORAGE_BUDGET_VITE_ENV]
  )
  if (viteBudgetMb != null) {
    return Math.round(viteBudgetMb * 1024 * 1024)
  }

  const nextProcess =
    typeof globalThis !== "undefined"
      ? (globalThis as { process?: { env?: Record<string, string | undefined> } })
          .process
      : undefined
  const nextBudgetMb = parseStorageBudgetCandidateMb(
    nextProcess?.env?.[STORAGE_BUDGET_NEXT_ENV]
  )
  if (nextBudgetMb != null) {
    return Math.round(nextBudgetMb * 1024 * 1024)
  }

  return STORAGE_BUDGET_DEFAULT_MB * 1024 * 1024
}
