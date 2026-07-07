import { describe, expect, it } from "vitest"
import {
  isSettingsNavItemActive,
  resolveCurrentSettingsNavItem
} from "../settings-active-route"
import type { SettingsNavItem } from "../settings-nav"

const ITEM_CHAT = {
  to: "/settings/chat",
  labelToken: "settings:chatSettingsNav",
  icon: (() => null) as any
} satisfies SettingsNavItem

const ITEM_CHAT_ADVANCED = {
  to: "/settings/chat/advanced",
  labelToken: "settings:chatAdvancedNav",
  icon: (() => null) as any
} satisfies SettingsNavItem

describe("settings active-route matching", () => {
  it("matches exact route", () => {
    expect(isSettingsNavItemActive("/settings/chat", "/settings/chat")).toBe(true)
  })

  it("does not mark root settings active for child settings routes", () => {
    expect(isSettingsNavItemActive("/settings/preferences", "/settings")).toBe(
      false
    )
  })

  it("matches nested route under the same section", () => {
    expect(isSettingsNavItemActive("/settings/chat/advanced", "/settings/chat")).toBe(
      true
    )
  })

  it("does not match partial prefixes", () => {
    expect(isSettingsNavItemActive("/settings/chatbot", "/settings/chat")).toBe(
      false
    )
  })

  it("normalizes trailing slashes and query fragments", () => {
    expect(isSettingsNavItemActive("/settings/chat/?tab=debug", "/settings/chat")).toBe(
      true
    )
  })

  it("prefers the most specific matching settings route", () => {
    const matched = resolveCurrentSettingsNavItem("/settings/chat/advanced/tools", [
      { items: [ITEM_CHAT, ITEM_CHAT_ADVANCED] }
    ])

    expect(matched?.to).toBe("/settings/chat/advanced")
  })
})
