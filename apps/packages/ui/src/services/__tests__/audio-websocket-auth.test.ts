import { describe, expect, it, vi } from "vitest"

import {
  buildAudioWebSocketUrl,
  sendAudioWebSocketAuthFrame
} from "@/services/tldw/audio-websocket-auth"

describe("audio websocket auth helpers", () => {
  it.each([
    ["/api/v1/audio/stream/tts"],
    ["/api/v1/audio/stream/transcribe"],
    ["/api/v1/audio/chat/stream"]
  ])("builds a bare websocket url for %s", (path) => {
    const url = buildAudioWebSocketUrl("http://127.0.0.1:8000/", path)

    expect(url).toBe(`ws://127.0.0.1:8000${path}`)
    expect(url).not.toContain("token=")
  })

  it("rejects token-in-query construction", () => {
    expect(() =>
      buildAudioWebSocketUrl(
        "http://127.0.0.1:8000/",
        "/api/v1/audio/stream/tts?token=secret"
      )
    ).toThrow("Audio WebSocket tokens must be sent in the auth frame")
  })

  it("sends the backend-supported auth frame", () => {
    const ws = { send: vi.fn() }

    sendAudioWebSocketAuthFrame(ws, "secret-token")

    expect(ws.send).toHaveBeenCalledWith(
      JSON.stringify({ type: "auth", token: "secret-token" })
    )
  })
})
