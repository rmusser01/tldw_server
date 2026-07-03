import { describe, expect, it } from "vitest"

import { COCKPIT_LEFT_RESTORE_WRAPPER_CLASS } from "../chat-rail-positioning"

describe("chat rail positioning contract", () => {
  it("keeps the cockpit context restore trigger clear of the app navigation rail", () => {
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS.split(" ")).toEqual(
      expect.arrayContaining([
        "absolute",
        "left-0",
        "top-[clamp(18rem,36vh,24rem)]",
        "z-50"
      ])
    )
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS).not.toContain("fixed")
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS).not.toContain("left-12")
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS).not.toContain("top-1/2")
  })
})
