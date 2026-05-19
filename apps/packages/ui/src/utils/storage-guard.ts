import { estimateLocalStorageUsageBytes, estimateStorageCost, resolveStorageBudgetBytes, STORAGE_THRESHOLDS } from "./storage-budget"
import { STORAGE_QUOTA_REFRESH_EVENT } from "@/store/storage-quota-events"

export type StorageGuardResult = {
  canWrite: boolean
  currentRatio: number
  wouldExceed: boolean
  recommendation: string | null
}

/**
 * Advisory pre-write check. Does NOT block writes — callers decide.
 * @param existingKey - If updating an existing key, pass it to subtract its current size
 * @param storage - Storage instance to check (defaults to window.localStorage)
 */
export function checkStorageBeforeWrite(
  estimatedBytes: number,
  existingKey?: string,
  storage: Storage = window.localStorage,
): StorageGuardResult {
  try {
    const totalUsedBytes = estimateLocalStorageUsageBytes(storage) // no prefix = all keys
    const browserLimit = resolveStorageBudgetBytes()
    // Subtract existing value size when overwriting a key (avoids double-counting)
    let existingSize = 0
    if (existingKey) {
      const existing = storage.getItem(existingKey)
      if (existing != null) {
        existingSize = estimateStorageCost(existingKey) + estimateStorageCost(existing)
      }
    }
    const effectiveUsed = totalUsedBytes - existingSize
    const currentRatio = browserLimit > 0 ? effectiveUsed / browserLimit : 0
    const wouldExceed = (effectiveUsed + estimatedBytes) >= browserLimit * STORAGE_THRESHOLDS.exceeded

    let recommendation: string | null = null
    if (wouldExceed) {
      recommendation = "Storage is nearly full. Consider archiving old workspaces before saving."
    } else if (currentRatio >= STORAGE_THRESHOLDS.warning) {
      recommendation = "Storage is getting full. Consider cleaning up old data soon."
    }

    return {
      canWrite: !wouldExceed,
      currentRatio: Math.min(currentRatio, 1),
      wouldExceed,
      recommendation
    }
  } catch {
    // If we can't check, assume it's fine
    return { canWrite: true, currentRatio: 0, wouldExceed: false, recommendation: null }
  }
}

/**
 * Dispatch a refresh event to update any mounted StorageQuotaBanner.
 * Call this after a successful localStorage write.
 */
export function notifyStorageWrite(): void {
  try {
    window.dispatchEvent(new CustomEvent(STORAGE_QUOTA_REFRESH_EVENT))
  } catch { /* ignore */ }
}

export { estimateStorageCost }
