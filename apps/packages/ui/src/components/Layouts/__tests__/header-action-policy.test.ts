import { describe, expect, it } from "vitest"
import { getHeaderActionPolicy } from "../header-action-policy"

describe("getHeaderActionPolicy", () => {
  it("enables chat session actions on the main chat route", () => {
    expect(getHeaderActionPolicy("/chat")).toMatchObject({
      showChatSessionActions: true,
      showChatTitle: true,
      showSessionModeBadge: true,
      showShareConversation: true,
    })
  })

  it.each([
    "/knowledge",
    "/media",
    "/sources",
    "/settings",
    "/mcp-hub",
    "/stt",
    "/tts",
    "/quick-chat-popout",
  ])("hides chat session actions on %s", (pathname) => {
    expect(getHeaderActionPolicy(pathname)).toMatchObject({
      showChatSessionActions: false,
      showChatTitle: false,
      showSessionModeBadge: false,
      showShareConversation: false,
    })
  })

  it("normalizes trailing slashes before classifying chat routes", () => {
    expect(getHeaderActionPolicy("/chat/").showChatSessionActions).toBe(true)
  })
})
