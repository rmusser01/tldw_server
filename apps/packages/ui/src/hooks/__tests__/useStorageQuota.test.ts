import { describe, expect, it, beforeEach } from "vitest"
import {
  estimateStorageCost,
  estimateLocalStorageUsageBytes,
  resolveStorageBudgetBytes,
  STORAGE_BUDGET_DEFAULT_MB
} from "../../utils/storage-budget"
import { resolveLevel } from "../useStorageQuota"

describe("storage-budget utilities", () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it("estimates UTF-8 byte length correctly for ASCII", () => {
    expect(estimateStorageCost("hello")).toBe(5)
    expect(estimateStorageCost("")).toBe(0)
  })

  it("returns string length (UTF-16 code units) for localStorage quota estimation", () => {
    // localStorage uses UTF-16 internally, so str.length (UTF-16 code units)
    // is the correct proxy — "é" and "你" are each 1 UTF-16 code unit
    expect(estimateStorageCost("é")).toBe(1)
    expect(estimateStorageCost("你")).toBe(1)
    expect(estimateStorageCost("abc")).toBe(3)
  })

  it("estimates localStorage usage", () => {
    localStorage.setItem("tldw-test-key", "test-value")
    const bytes = estimateLocalStorageUsageBytes(localStorage)
    expect(bytes).toBeGreaterThan(0)
    localStorage.removeItem("tldw-test-key")
  })

  it("filters by prefix", () => {
    localStorage.setItem("tldw-a", "value")
    localStorage.setItem("other-b", "value")
    const tldwBytes = estimateLocalStorageUsageBytes(localStorage, "tldw")
    const allBytes = estimateLocalStorageUsageBytes(localStorage)
    expect(tldwBytes).toBeLessThan(allBytes)
  })

  it("returns 0 for empty storage", () => {
    expect(estimateLocalStorageUsageBytes(localStorage)).toBe(0)
  })

  it("resolves default budget to 5 MB", () => {
    expect(resolveStorageBudgetBytes()).toBe(STORAGE_BUDGET_DEFAULT_MB * 1024 * 1024)
  })
})

describe("StorageQuotaLevel resolution", () => {
  it("returns ok at 0%", () => {
    expect(resolveLevel(0)).toBe("ok")
  })

  it("returns ok below 80%", () => {
    expect(resolveLevel(0.79)).toBe("ok")
  })

  it("returns warning at 80%", () => {
    expect(resolveLevel(0.80)).toBe("warning")
  })

  it("returns warning between 80% and 95%", () => {
    expect(resolveLevel(0.90)).toBe("warning")
  })

  it("returns exceeded at 95%", () => {
    expect(resolveLevel(0.95)).toBe("exceeded")
  })

  it("returns exceeded above 95%", () => {
    expect(resolveLevel(0.99)).toBe("exceeded")
    expect(resolveLevel(1.0)).toBe("exceeded")
  })
})
