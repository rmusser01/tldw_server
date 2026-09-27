import { beforeEach, describe, expect, it } from "vitest"

import {
  clearSettingsReturnTo,
  getSettingsReturnTo,
  resolveSettingsNavigationUrl,
  setSettingsReturnTo
} from "@/utils/settings-return"

describe("settings return target", () => {
  beforeEach(() => {
    sessionStorage.clear()
    clearSettingsReturnTo()
  })

  it("stores non-settings routes for return navigation", () => {
    setSettingsReturnTo("/media")

    expect(getSettingsReturnTo()).toBe("/media")
  })

  it("stores chat context for chat return targets", () => {
    setSettingsReturnTo("/chat", {
      historyId: "history-123",
      serverChatId: "server-chat-456"
    })

    expect(getSettingsReturnTo()).toBe(
      "/chat?settingsHistoryId=history-123&settingsServerChatId=server-chat-456"
    )
  })

  it("does not overwrite return target with settings routes", () => {
    setSettingsReturnTo("/chat", {
      historyId: "history-abc"
    })
    setSettingsReturnTo("/settings/tldw")

    expect(getSettingsReturnTo()).toBe("/chat?settingsHistoryId=history-abc")
  })

  it.each([
    {
      currentUrl: "moz-extension://profile/options.html#/settings/prompt",
      destination: "/prompts",
      expected: "moz-extension://profile/options.html#/prompts"
    },
    {
      currentUrl: "chrome-extension://extension/options.html#/settings/prompt",
      destination: "/settings/chat",
      expected: "chrome-extension://extension/options.html#/settings/chat"
    },
    {
      currentUrl: "moz-extension://profile/options.html#/settings/prompt",
      destination: "moz-extension://profile/options.html#/prompts",
      expected: "moz-extension://profile/options.html#/prompts"
    },
    {
      currentUrl: "https://app.test/settings/prompt",
      destination: "/chat",
      expected: "https://app.test/chat"
    }
  ])("resolves $currentUrl navigation against its real host", ({
    currentUrl,
    destination,
    expected
  }) => {
    expect(resolveSettingsNavigationUrl(destination, currentUrl)).toBe(expected)
  })

  it("rejects a navigation destination on another host", () => {
    expect(resolveSettingsNavigationUrl(
      "https://other.test/chat",
      "moz-extension://profile/options.html#/settings/prompt"
    )).toBeNull()
    expect(resolveSettingsNavigationUrl(
      "moz-extension://other/prompts",
      "moz-extension://profile/options.html#/settings/prompt"
    )).toBeNull()
  })
})
