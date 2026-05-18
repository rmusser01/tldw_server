// @vitest-environment jsdom

import { describe, expect, it, vi } from "vitest"
import {
  CHARACTER_CHAT_MODE_INTENT_EVENT,
  dispatchCharacterChatModeIntent,
  getCharacterChatRouteIntent
} from "../character-chat-mode-intent"

describe("character chat mode intent", () => {
  it("parses first-class character chat URL intent", () => {
    expect(getCharacterChatRouteIntent("?mode=character&characterId=char-1")).toEqual({
      mode: "character",
      characterId: "char-1"
    })
    expect(getCharacterChatRouteIntent("?mode=character&character_id=42")).toEqual({
      mode: "character",
      characterId: "42"
    })
  })

  it("ignores non-character chat modes", () => {
    expect(getCharacterChatRouteIntent("?mode=rag&characterId=char-1")).toBeNull()
    expect(getCharacterChatRouteIntent("?characterId=char-1")).toBeNull()
  })

  it("dispatches durable character chat mode intent", () => {
    const listener = vi.fn()
    window.addEventListener(CHARACTER_CHAT_MODE_INTENT_EVENT, listener)

    try {
      dispatchCharacterChatModeIntent({
        source: "test",
        characterId: "char-1"
      })
    } finally {
      window.removeEventListener(CHARACTER_CHAT_MODE_INTENT_EVENT, listener)
    }

    expect(listener).toHaveBeenCalledWith(
      expect.objectContaining({
        detail: { source: "test", characterId: "char-1" }
      })
    )
  })
})
