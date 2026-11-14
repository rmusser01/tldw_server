import { describe, expect, it } from "vitest"

import { sidepanelScrollToLatestBottom } from "../sidepanel-scroll-latest"

describe("sidepanelScrollToLatestBottom", () => {
  it("floats the pill 24px above a pinned composer", () => {
    expect(sidepanelScrollToLatestBottom(300, 128, true)).toBe("324px")
  })

  it("keeps the pill clear of a pinned composer that is not measured yet", () => {
    expect(sidepanelScrollToLatestBottom(0, 160, true)).toBe("160px")
    expect(sidepanelScrollToLatestBottom(40, 160, true)).toBe("128px")
  })

  it("sits near the bottom when the composer is unpinned and scrolls with the messages (G19)", () => {
    expect(sidepanelScrollToLatestBottom(300, 128, false)).toBe(
      "calc(16px + env(safe-area-inset-bottom, 0px))"
    )
  })
})
