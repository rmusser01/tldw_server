import { describe, expect, it } from "vitest"
import {
  isWatchlistsConstrainedViewport,
  WATCHLISTS_CONSTRAINED_BREAKPOINT
} from "../useWatchlistsViewport"

describe("watchlists constrained viewport resolver", () => {
  it("treats extension-sized widths as constrained", () => {
    expect(isWatchlistsConstrainedViewport(390)).toBe(true)
    expect(isWatchlistsConstrainedViewport(420)).toBe(true)
  })

  it("treats desktop breakpoint and wider widths as unconstrained", () => {
    expect(WATCHLISTS_CONSTRAINED_BREAKPOINT).toBe(768)
    expect(isWatchlistsConstrainedViewport(768)).toBe(false)
    expect(isWatchlistsConstrainedViewport(1024)).toBe(false)
  })

  it("handles invalid or missing width as desktop safe", () => {
    expect(isWatchlistsConstrainedViewport(undefined)).toBe(false)
    expect(isWatchlistsConstrainedViewport(Number.NaN)).toBe(false)
    expect(isWatchlistsConstrainedViewport(-1)).toBe(false)
  })
})
