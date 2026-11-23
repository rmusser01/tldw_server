import { describe, expect, it, vi } from "vitest"
import { startSavedChat } from "../start-saved-chat"

// The header's New chat, the phone focus bar and `/new` share this path (CO-01, CC-11).
describe("startSavedChat", () => {
  it("starts a saved chat with no character once the clear goes ahead", () => {
    const calls: string[] = []
    const result = startSavedChat({
      clearChat: () => {
        calls.push("clear")
        return true
      },
      setTemporaryChat: (temporary) => calls.push(`temporary:${temporary}`),
      setSelectedCharacter: (character) => calls.push(`character:${character}`)
    })
    expect(result).toBe(true)
    expect(calls).toEqual(["clear", "temporary:false", "character:null"])
  })

  it("leaves the chat mode and character alone when the clear is refused", () => {
    const setTemporaryChat = vi.fn()
    const setSelectedCharacter = vi.fn()
    const result = startSavedChat({
      clearChat: () => false,
      setTemporaryChat,
      setSelectedCharacter
    })
    expect(result).toBe(false)
    expect(setTemporaryChat).not.toHaveBeenCalled()
    expect(setSelectedCharacter).not.toHaveBeenCalled()
  })
})
