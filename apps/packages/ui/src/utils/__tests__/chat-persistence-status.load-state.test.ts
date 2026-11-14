import { describe, expect, it } from "vitest"
import { resolveChatPersistenceKind } from "../chat-persistence-status"

describe("chat persistence status while a server chat cannot be loaded (CS-06)", () => {
  it("reports the chat as not loaded rather than saved", () => {
    expect(
      resolveChatPersistenceKind({
        temporaryChat: false,
        serverChatId: "chat-1",
        serverSaveStatus: "saved",
        serverChatNotLoaded: true
      })
    ).toBe("serverNotLoaded")
  })

  it("keeps the save status once the chat loads", () => {
    expect(
      resolveChatPersistenceKind({
        temporaryChat: false,
        serverChatId: "chat-1",
        serverSaveStatus: "saved",
        serverChatNotLoaded: false
      })
    ).toBe("server")
  })
})
