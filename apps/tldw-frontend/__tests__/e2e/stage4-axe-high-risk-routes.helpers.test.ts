import { describe, expect, it } from "vitest"

import { getRedirectDispositionForA11yScan } from "@web/e2e/smoke/stage4-axe-high-risk-routes.helpers"

describe("getRedirectDispositionForA11yScan", () => {
  it("keeps stable redirect-prone routes in the scan", () => {
    expect(
      getRedirectDispositionForA11yScan({
        routePath: "/login",
        finalPath: "/login",
        mayRedirectWhenUnavailable: true
      })
    ).toEqual({ shouldSkip: false })
  })

  it("skips redirect-prone routes that moved before the scan", () => {
    expect(
      getRedirectDispositionForA11yScan({
        routePath: "/login",
        finalPath: "/settings/tldw",
        mayRedirectWhenUnavailable: true
      })
    ).toEqual({
      shouldSkip: true,
      message:
        "Route /login redirected to /settings/tldw; feature is unavailable in this runtime"
    })
  })

  it("skips redirect-prone routes that reload during the scan", () => {
    expect(
      getRedirectDispositionForA11yScan({
        routePath: "/login",
        finalPath: "/login",
        mayRedirectWhenUnavailable: true,
        navigationObservedDuringScan: true
      })
    ).toEqual({
      shouldSkip: true,
      message:
        "Route /login reloaded during accessibility scan; feature is unavailable in this runtime"
    })
  })

  it("never skips stable routes that are not marked redirect-prone", () => {
    expect(
      getRedirectDispositionForA11yScan({
        routePath: "/chat",
        finalPath: "/chat",
        mayRedirectWhenUnavailable: false,
        navigationObservedDuringScan: true
      })
    ).toEqual({ shouldSkip: false })
  })
})
