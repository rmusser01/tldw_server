import { useState, useEffect, useCallback, useMemo, useRef } from "react"
import { estimateLocalStorageUsageBytes, resolveStorageBudgetBytes, STORAGE_THRESHOLDS } from "@/utils/storage-budget"
import { STORAGE_QUOTA_REFRESH_EVENT } from "@/store/storage-quota-events"
import { WORKSPACE_STORAGE_QUOTA_EVENT, WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"

export type StorageQuotaLevel = "ok" | "warning" | "exceeded"

export type StorageQuotaState = {
  usedBytes: number
  budgetBytes: number
  ratio: number
  level: StorageQuotaLevel
  availableBytes: number
  canWrite: (estimatedBytes: number) => boolean
  refresh: () => void
}

export const resolveLevel = (ratio: number): StorageQuotaLevel => {
  if (ratio >= STORAGE_THRESHOLDS.exceeded) return "exceeded"
  if (ratio >= STORAGE_THRESHOLDS.warning) return "warning"
  return "ok"
}

export function useStorageQuota(): StorageQuotaState {
  const budgetBytes = useMemo(() => resolveStorageBudgetBytes(), [])
  const [usedBytes, setUsedBytes] = useState(() => {
    try {
      return estimateLocalStorageUsageBytes(window.localStorage, WORKSPACE_STORAGE_KEY)
    } catch {
      return 0
    }
  })

  const refresh = useCallback(() => {
    try {
      setUsedBytes(estimateLocalStorageUsageBytes(window.localStorage, WORKSPACE_STORAGE_KEY))
    } catch { /* ignore */ }
  }, [])

  // Debounced refresh for cross-tab storage events (fires frequently)
  const storageDebounceRef = useRef<ReturnType<typeof setTimeout>>()
  const debouncedRefresh = useCallback(() => {
    clearTimeout(storageDebounceRef.current)
    storageDebounceRef.current = setTimeout(refresh, 500)
  }, [refresh])

  // Listen for storage events (cross-tab), quota refresh events, and workspace quota events
  useEffect(() => {
    window.addEventListener("storage", debouncedRefresh)
    window.addEventListener(STORAGE_QUOTA_REFRESH_EVENT, refresh)
    window.addEventListener(WORKSPACE_STORAGE_QUOTA_EVENT, refresh)

    return () => {
      clearTimeout(storageDebounceRef.current)
      window.removeEventListener("storage", debouncedRefresh)
      window.removeEventListener(STORAGE_QUOTA_REFRESH_EVENT, refresh)
      window.removeEventListener(WORKSPACE_STORAGE_QUOTA_EVENT, refresh)
    }
  }, [refresh, debouncedRefresh])

  const ratio = budgetBytes > 0 ? Math.min(usedBytes / budgetBytes, 1) : 0
  const level = resolveLevel(ratio)
  const availableBytes = Math.max(budgetBytes - usedBytes, 0)

  const canWrite = useCallback(
    (estimatedBytes: number) => usedBytes + estimatedBytes < budgetBytes * STORAGE_THRESHOLDS.exceeded,
    [usedBytes, budgetBytes]
  )

  return { usedBytes, budgetBytes, ratio, level, availableBytes, canWrite, refresh }
}
