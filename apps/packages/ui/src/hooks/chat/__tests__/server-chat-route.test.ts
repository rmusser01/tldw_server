import { afterEach, describe, expect, it } from "vitest"
import {
  buildServerChatPath,
  isCharacterChatRouteSearch,
  isServerChatNavigationPending,
  planServerChatNavigation,
  readServerChatRouteId,
  reportMountedChatRoute,
  resetServerChatRouteState,
  settleServerChatNavigation,
  withServerChatRouteId
} from "../server-chat-route"

describe("chat page addresses (XP-05)", () => {
  afterEach(() => resetServerChatRouteState())

  it("names a saved chat by its server id", () => {
    expect(buildServerChatPath("d64f6a52-aaaa")).toBe("/chat?chat=d64f6a52-aaaa")
    expect(readServerChatRouteId("?chat=d64f6a52-aaaa")).toBe("d64f6a52-aaaa")
  })

  it("ignores a missing or malformed chat id", () => {
    expect(readServerChatRouteId("")).toBeNull()
    expect(readServerChatRouteId("?chat=")).toBeNull()
    expect(readServerChatRouteId("?chat=%2Fapi%2Fv1")).toBeNull()
  })

  it("keeps the rest of the query when the chat changes", () => {
    expect(withServerChatRouteId("?tab=history&chat=old", "new")).toBe("?tab=history&chat=new")
    expect(withServerChatRouteId("?tab=history&chat=old", null)).toBe("?tab=history")
    expect(withServerChatRouteId("?chat=old", null)).toBe("")
  })

  it("leaves character chat routes to the character intent", () => {
    expect(isCharacterChatRouteSearch("?mode=character&chatId=abc")).toBe(true)
    expect(isCharacterChatRouteSearch("?chat=abc")).toBe(false)
  })

  it("adds a history entry when switching from one saved chat to another", () => {
    reportMountedChatRoute({ namesChat: true })
    expect(planServerChatNavigation("chat-b")).toEqual({ to: "/chat?chat=chat-b", replace: false })
  })

  it("replaces the entry of a chat page that named no chat", () => {
    reportMountedChatRoute({ namesChat: false })
    expect(planServerChatNavigation("chat-b")).toEqual({ to: "/chat?chat=chat-b", replace: true })
  })

  it("adds a history entry when opening a chat from another page", () => {
    reportMountedChatRoute(null)
    expect(planServerChatNavigation("chat-b").replace).toBe(false)
  })

  it("holds the address sync until the planned navigation lands", () => {
    planServerChatNavigation("chat-b")
    expect(isServerChatNavigationPending("chat-b")).toBe(true)
    expect(isServerChatNavigationPending("chat-a")).toBe(false)
    settleServerChatNavigation("chat-b")
    expect(isServerChatNavigationPending("chat-b")).toBe(false)
  })
})
