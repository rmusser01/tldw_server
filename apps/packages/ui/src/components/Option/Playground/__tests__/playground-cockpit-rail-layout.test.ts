// CO-04 (#3113): at 1024px both rails left the transcript a few hundred px.
// The layout is width-aware: below a threshold only one rail is open at a
// time, and when even one would squeeze the transcript it overlays it. The
// saved preferences are inputs only; the resolver never changes them.
import { describe, expect, it } from "vitest"

import {
  COCKPIT_BOTH_RAILS_MIN_WIDTH,
  COCKPIT_ONE_RAIL_MIN_WIDTH,
  resolveCockpitRailLayout
} from "../playground-cockpit-rail-layout"

describe("resolveCockpitRailLayout", () => {
  it("keeps both rails inline until the shell has been measured", () => {
    expect(
      resolveCockpitRailLayout({
        width: null,
        contextVisible: true,
        runtimeVisible: true,
        preferred: "context"
      })
    ).toEqual({ context: "inline", runtime: "inline" })
  })

  it("keeps both rails inline when the transcript still gets a usable width", () => {
    expect(
      resolveCockpitRailLayout({
        width: COCKPIT_BOTH_RAILS_MIN_WIDTH,
        contextVisible: true,
        runtimeVisible: true,
        preferred: "context"
      })
    ).toEqual({ context: "inline", runtime: "inline" })
  })

  it("opens one rail at a time below the threshold, the most recently opened one", () => {
    const width = COCKPIT_BOTH_RAILS_MIN_WIDTH - 1
    expect(
      resolveCockpitRailLayout({
        width,
        contextVisible: true,
        runtimeVisible: true,
        preferred: "context"
      })
    ).toEqual({ context: "inline", runtime: "hidden" })
    expect(
      resolveCockpitRailLayout({
        width,
        contextVisible: true,
        runtimeVisible: true,
        preferred: "runtime"
      })
    ).toEqual({ context: "hidden", runtime: "inline" })
  })

  it("shows the only open rail even when it was not the preferred one", () => {
    expect(
      resolveCockpitRailLayout({
        width: COCKPIT_BOTH_RAILS_MIN_WIDTH - 1,
        contextVisible: false,
        runtimeVisible: true,
        preferred: "context"
      })
    ).toEqual({ context: "hidden", runtime: "inline" })
  })

  it("overlays the transcript when even one inline rail would squeeze it", () => {
    expect(
      resolveCockpitRailLayout({
        width: COCKPIT_ONE_RAIL_MIN_WIDTH - 1,
        contextVisible: true,
        runtimeVisible: true,
        preferred: "runtime"
      })
    ).toEqual({ context: "hidden", runtime: "overlay" })
  })

  it("puts the thresholds where a 1024px window with the app sidebar lands", () => {
    // 1024px minus a 256px sidebar is 768px: one rail at most, as an overlay.
    expect(COCKPIT_BOTH_RAILS_MIN_WIDTH).toBeGreaterThan(1024)
    expect(COCKPIT_ONE_RAIL_MIN_WIDTH).toBeGreaterThan(768)
  })
})
