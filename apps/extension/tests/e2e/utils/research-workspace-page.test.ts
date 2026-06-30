import { describe, expect, it } from "vitest"

import { shouldUseDomClickFallback } from "../../../../test-utils/research-workspace/page"

describe("ResearchWorkspaceParityPage click fallback", () => {
  it("falls back to DOM click for Playwright actionability timeouts", () => {
    const error = new Error("TimeoutError: locator.click: Timeout 3000ms exceeded.")

    expect(shouldUseDomClickFallback(error)).toBe(true)
  })

  it("does not fall back for unrelated assertion errors", () => {
    const error = new Error("Expected source count to match seeded fixtures.")

    expect(shouldUseDomClickFallback(error)).toBe(false)
  })
})
