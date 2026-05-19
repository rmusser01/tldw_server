// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from "vitest"
import { checkStorageBeforeWrite, notifyStorageWrite } from "../storage-guard"
import * as storageBudget from "../storage-budget"

describe("storage-guard", () => {
  beforeEach(() => localStorage.clear())

  it("returns canWrite true when storage is empty", () => {
    const result = checkStorageBeforeWrite(100)
    expect(result.canWrite).toBe(true)
    expect(result.wouldExceed).toBe(false)
    expect(result.recommendation).toBeNull()
  })

  it("returns recommendation when ratio >= 80%", () => {
    vi.spyOn(storageBudget, "estimateLocalStorageUsageBytes").mockReturnValue(4.3 * 1024 * 1024)

    const result = checkStorageBeforeWrite(100)

    expect(result.currentRatio).toBeGreaterThanOrEqual(0.80)
    expect(result.recommendation).toBeTruthy()
  })

  it("dispatches refresh event via notifyStorageWrite", () => {
    const handler = vi.fn()
    window.addEventListener("tldw:storage-quota-refresh", handler)
    notifyStorageWrite()
    expect(handler).toHaveBeenCalledTimes(1)
    window.removeEventListener("tldw:storage-quota-refresh", handler)
  })
})
