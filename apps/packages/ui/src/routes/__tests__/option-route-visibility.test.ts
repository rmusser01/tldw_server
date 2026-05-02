import { describe, expect, it } from "vitest"

import { isHostedVisibleOptionPath } from "../option-route-visibility"

describe("hosted option route visibility", () => {
  it("keeps audio explainer routes visible in hosted mode", () => {
    expect(isHostedVisibleOptionPath("/tts")).toBe(true)
    expect(isHostedVisibleOptionPath("/stt")).toBe(true)
  })
})
