import { describe, expect, it } from "vitest"

import {
  CHAT_RAIL_EDGE_TRIGGER_CLASS,
  COCKPIT_LEFT_RESTORE_WRAPPER_CLASS
} from "../chat-rail-positioning"

describe("chat rail positioning contract", () => {
  it("keeps the collapsed chat rail trigger attached lower on the left edge", () => {
    expect(CHAT_RAIL_EDGE_TRIGGER_CLASS.split(" ")).toEqual(
      expect.arrayContaining([
        "absolute",
        "left-0",
        "top-[clamp(10rem,24vh,15rem)]",
        "h-20",
        "w-8",
        "rounded-r-lg",
        "border-l-0"
      ])
    )
    expect(CHAT_RAIL_EDGE_TRIGGER_CLASS).not.toContain("top-[calc(50%_-_8rem)]")
  })

  it("keeps the cockpit context restore trigger attached to the left edge below the chat trigger", () => {
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
