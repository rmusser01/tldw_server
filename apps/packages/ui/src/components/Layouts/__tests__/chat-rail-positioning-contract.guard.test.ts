import { describe, expect, it } from "vitest"

import { COCKPIT_LEFT_RESTORE_WRAPPER_CLASS } from "../chat-rail-positioning"

describe("chat rail positioning contract", () => {
  it("keeps the cockpit context restore trigger attached to the left edge", () => {
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS.split(" ")).toEqual(
      expect.arrayContaining([
        "fixed",
        "left-0",
        "top-[clamp(18rem,36vh,24rem)]",
        "z-50"
      ])
    )
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS).not.toContain("left-12")
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS).not.toContain("top-1/2")
  })
})
