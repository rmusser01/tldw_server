// @vitest-environment jsdom
import { renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import {
  usePlaygroundVoiceChat,
  type UsePlaygroundVoiceChatDeps
} from "../usePlaygroundVoiceChat"

vi.mock("@/components/Chat/composer/hooks/useComposerVoiceChat", () => ({
  useComposerVoiceChat: () => ({
    speechAvailable: false,
    speechUsesServer: false,
    dictationStrategy: { autoFallbackActive: false }
  })
}))

const deps = (state: string): UsePlaygroundVoiceChatDeps =>
  ({
    voiceConversationAvailability: { available: true, message: null },
    voiceChatEnabled: true,
    setVoiceChatEnabled: vi.fn(),
    voiceChat: { state },
    voiceChatMessages: { abandonTurn: vi.fn() },
    t: (_key: string, fallback?: unknown) => (typeof fallback === "string" ? fallback : _key)
  }) as unknown as UsePlaygroundVoiceChatDeps

// G14 follow-up: a live voice chat names the tab. Leaving the chat page must
// keep the title the next route set (XP-19), not restore the chat's.
describe("usePlaygroundVoiceChat tab title", () => {
  afterEach(() => {
    document.title = ""
  })

  it("names the tab while voice chat listens and gives the title back when it stops", () => {
    document.title = "Chat | tldw"
    const view = renderHook((state: string) => usePlaygroundVoiceChat(deps(state)), {
      initialProps: "listening"
    })
    expect(document.title).toBe("\u{1F3A4} Listening - Chat")
    view.rerender("idle")
    expect(document.title).toBe("Chat | tldw")
  })

  it("keeps the next route's title when the chat page unmounts", () => {
    document.title = "Chat | tldw"
    const view = renderHook(() => usePlaygroundVoiceChat(deps("listening")))
    document.title = "Notes | tldw"
    view.unmount()
    expect(document.title).toBe("Notes | tldw")
  })
})
