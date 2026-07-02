import { describe, expect, it } from "vitest"

import {
  CHAT_RAIL_EDGE_TRIGGER_CLASS,
  COCKPIT_LEFT_RESTORE_WRAPPER_CLASS
} from "../chat-rail-positioning"

describe("chat rail positioning contract", () => {
  it("keeps the collapsed chat rail trigger attached to the upper left edge", () => {
    expect(CHAT_RAIL_EDGE_TRIGGER_CLASS.split(" ")).toEqual(
      expect.arrayContaining([
        "absolute",
        "left-0",
        "top-[clamp(8rem,20vh,14rem)]",
        "h-28",
        "w-10",
        "rounded-r-lg",
        "border-l-0"
      ])
    )
    expect(CHAT_RAIL_EDGE_TRIGGER_CLASS).not.toContain("top-[calc(50%_-_8rem)]")
  })

  it("offsets the cockpit context restore trigger from the collapsed chat rail edge", () => {
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS.split(" ")).toEqual(
      expect.arrayContaining([
        "absolute",
        "left-12",
        "top-[clamp(3rem,18vh,10rem)]",
        "z-50"
      ])
    )
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS).not.toContain("left-0")
    expect(COCKPIT_LEFT_RESTORE_WRAPPER_CLASS).not.toContain("top-1/2")
  })
})
