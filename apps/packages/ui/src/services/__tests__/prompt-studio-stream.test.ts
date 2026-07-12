import { afterEach, beforeEach, describe, expect, it } from "vitest"
import {
  buildPromptStudioWebSocketUrl,
  isPromptStudioStatusEvent
} from "@/services/prompt-studio-stream"

describe("buildPromptStudioWebSocketUrl", () => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  const originalWindow = globalThis.window

  beforeEach(() => {
    Object.defineProperty(globalThis, "window", {
      value: {
        location: {
          origin: "http://127.0.0.1:8080",
          protocol: "http:"
        }
      },
      configurable: true
    })
  })

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
    Object.defineProperty(globalThis, "window", {
      value: originalWindow,
      configurable: true
    })
  })

  it("uses the webui origin for quickstart websocket urls", () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"

    const url = buildPromptStudioWebSocketUrl({
      serverUrl: "http://127.0.0.1:8000/",
      authMode: "single-user",
      apiKey: "abc123",
      accessToken: ""
    })

    expect(url).toBe("ws://127.0.0.1:8080/api/v1/prompt-studio/ws?api_key=abc123")
  })

  it("uses a secret-free page-origin url for cookie sessions", () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const url = buildPromptStudioWebSocketUrl(
      {
        serverUrl: "https://remote.example.test",
        authMode: "single-user",
        authSource: "cookie-session",
        apiKey: "stale-key",
        accessToken: "stale-token"
      },
      42
    )

    expect(url).toBe(
      "ws://127.0.0.1:8080/api/v1/prompt-studio/ws?project_id=42"
    )
  })

  it("builds single-user websocket url with api key", () => {
    const url = buildPromptStudioWebSocketUrl({
      serverUrl: "http://127.0.0.1:8000/",
      authMode: "single-user",
      apiKey: "abc123",
      accessToken: ""
    })

    expect(url).toBe("ws://127.0.0.1:8000/api/v1/prompt-studio/ws?api_key=abc123")
  })

  it("builds multi-user websocket url with bearer token and project id", () => {
    const url = buildPromptStudioWebSocketUrl(
      {
        serverUrl: "https://example.com",
        authMode: "multi-user",
        apiKey: "",
        accessToken: "jwt-token"
      },
      42
    )

    expect(url).toBe(
      "wss://example.com/api/v1/prompt-studio/ws?token=jwt-token&project_id=42"
    )
  })
})

describe("isPromptStudioStatusEvent", () => {
  it("accepts known realtime status event payloads", () => {
    expect(isPromptStudioStatusEvent({ type: "job_progress" })).toBe(true)
    expect(isPromptStudioStatusEvent({ type: "subscribed" })).toBe(true)
  })

  it("rejects unknown payloads", () => {
    expect(isPromptStudioStatusEvent({ type: "unrelated" })).toBe(false)
    expect(isPromptStudioStatusEvent(null)).toBe(false)
    expect(isPromptStudioStatusEvent("job_progress")).toBe(false)
  })
})
