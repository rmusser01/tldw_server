import { describe, expect, it } from "vitest"

import { toUnreadCount } from "@/utils/notifications"

describe("toUnreadCount", () => {
  it("normalizes missing and invalid values to zero", () => {
    expect(toUnreadCount(undefined)).toBe(0)
    expect(toUnreadCount(null)).toBe(0)
    expect(toUnreadCount("abc")).toBe(0)
  })

  it("preserves numeric unread counts", () => {
    expect(toUnreadCount(4)).toBe(4)
    expect(toUnreadCount("12")).toBe(12)
  })
})
