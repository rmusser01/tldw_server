import { describe, expect, it, vi } from "vitest"
import { notifyServerChatLoadFailure } from "../server-chat-load-failure-notice"

const t = ((key: string, options?: { defaultValue?: string }) =>
  options?.defaultValue ?? key) as never

describe("saved chat load failure notice (CS-06)", () => {
  it("adds no toast when the transcript already shows the failure panel", () => {
    const notification = { error: vi.fn() }

    notifyServerChatLoadFailure({ notification, t, messagesShown: 0 })

    expect(notification.error).not.toHaveBeenCalled()
  })

  it("explains the failure in plain words, never with the raw request line", () => {
    const notification = { error: vi.fn() }

    notifyServerChatLoadFailure({ notification, t, messagesShown: 4 })

    expect(notification.error).toHaveBeenCalledTimes(1)
    const payload = notification.error.mock.calls[0][0]
    expect(payload.message).toBe("Couldn't load this conversation.")
    expect(`${payload.message} ${payload.description}`).not.toMatch(/GET |\/api\/v1/)
  })
})
