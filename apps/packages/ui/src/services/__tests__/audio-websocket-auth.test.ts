import { readFileSync } from "node:fs"
import path from "node:path"

import { describe, expect, it, vi } from "vitest"

describe("audio websocket auth helpers", () => {
  it.each([
    ["/api/v1/audio/stream/tts"],
    ["/api/v1/audio/stream/transcribe"],
    ["/api/v1/audio/chat/stream"]
  ])("builds a bare websocket url for %s", async (endpointPath) => {
    const { buildAudioWebSocketUrl } = await import(
      "@/services/tldw/audio-websocket-auth"
    )

    const url = buildAudioWebSocketUrl("http://127.0.0.1:8000/", endpointPath)

    expect(url).toBe(`ws://127.0.0.1:8000${endpointPath}`)
    expect(url).not.toContain("token=")
  })

  it("rejects token-in-query construction", async () => {
    const { buildAudioWebSocketUrl } = await import(
      "@/services/tldw/audio-websocket-auth"
    )

    expect(() =>
      buildAudioWebSocketUrl(
        "http://127.0.0.1:8000/",
        "/api/v1/audio/stream/tts?token=secret"
      )
    ).toThrow("Audio WebSocket tokens must be sent in the auth frame")
  })

  it("sends the backend-supported auth frame", async () => {
    const { sendAudioWebSocketAuthFrame } = await import(
      "@/services/tldw/audio-websocket-auth"
    )
    const ws = { send: vi.fn() }

    sendAudioWebSocketAuthFrame(ws, "secret-token")

    expect(ws.send).toHaveBeenCalledWith(
      JSON.stringify({ type: "auth", token: "secret-token" })
    )
  })
})

describe("speech TTS websocket contract", () => {
  it("opens a token-free URL and sends auth before prompt payloads", () => {
    const source = readFileSync(
      path.resolve(
        __dirname,
        "../../components/Option/Speech/SpeechPlaygroundPage.tsx"
      ),
      "utf8"
    )

    expect(source).not.toContain("/api/v1/audio/stream/tts?token=")
    expect(source).not.toContain("encodeURIComponent(token)")

    const indexOfMatch = (regex: RegExp) => {
      const match = source.match(regex)
      expect(match).not.toBeNull()
      return match?.index ?? -1
    }

    const authFrameIndex = indexOfMatch(
      /sendAudioWebSocketAuthFrame\s*\(\s*ws\s*,\s*token\s*\)/
    )
    const promptFrameIndex = indexOfMatch(/type\s*:\s*["']prompt["']/)
    const authFailureHandlerIndex = indexOfMatch(
      new RegExp(
        [
          String.raw`try\s*\{\s*sendAudioWebSocketAuthFrame\s*\(\s*ws\s*,\s*token\s*\)`,
          String.raw`[\s\S]*?\}\s*catch\s*\([^)]*\)\s*\{`,
          String.raw`[\s\S]*?ws\.close\s*\(\s*\)`
        ].join("")
      )
    )

    expect(authFrameIndex).toBeLessThan(promptFrameIndex)
    expect(authFailureHandlerIndex).toBeLessThan(promptFrameIndex)
  })
})
