import { afterEach, beforeEach, describe, expect, it } from "vitest"

import { buildPersonaWebSocketUrl } from "@/services/persona-stream"

describe("buildPersonaWebSocketUrl", () => {
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

  it("uses the webui origin for quickstart websocket urls and keeps auth out of the url", () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"

    const { url, protocols } = buildPersonaWebSocketUrl({
      serverUrl: "http://127.0.0.1:8000/",
      authMode: "single-user",
      apiKey: "abc123",
      accessToken: ""
    })

    expect(url).toBe("ws://127.0.0.1:8080/api/v1/persona/stream")
    expect(url).not.toContain("abc123")
    expect(protocols).toEqual(["bearer", "abc123"])
  })

  it("carries the api key in the subprotocol for single-user mode (not the url)", () => {
    const { url, protocols } = buildPersonaWebSocketUrl({
      serverUrl: "http://127.0.0.1:8000/",
      authMode: "single-user",
      apiKey: "abc123",
      accessToken: ""
    })

    expect(url).toBe("ws://127.0.0.1:8000/api/v1/persona/stream")
    expect(url).not.toContain("api_key")
    expect(url).not.toContain("abc123")
    expect(protocols).toEqual(["bearer", "abc123"])
  })

  it("uses a secret-free page-origin url for cookie sessions", () => {
    const { url, protocols } = buildPersonaWebSocketUrl({
      serverUrl: "https://remote.example.test",
      authMode: "single-user",
      authSource: "cookie-session",
      apiKey: "stale-key",
      accessToken: "stale-token"
    })

    expect(url).toBe("ws://127.0.0.1:8080/api/v1/persona/stream")
    expect(protocols).toEqual([])
  })

  it("carries the jwt in the subprotocol for multi-user mode (not the url)", () => {
    const { url, protocols } = buildPersonaWebSocketUrl({
      serverUrl: "https://example.com",
      authMode: "multi-user",
      apiKey: "",
      accessToken: "jwt-token"
    })

    expect(url).toBe("wss://example.com/api/v1/persona/stream")
    expect(url).not.toContain("token=")
    expect(url).not.toContain("jwt-token")
    expect(protocols).toEqual(["bearer", "jwt-token"])
  })

  it("falls back to the query-string token when the api key is not subprotocol-safe", () => {
    // A user-set custom key with RFC 6455 separators (`/`, `=`) can't be a
    // WebSocket subprotocol value; falling back keeps persona voice working
    // instead of throwing at `new WebSocket(url, ["bearer", key])`.
    const { url, protocols } = buildPersonaWebSocketUrl({
      serverUrl: "http://127.0.0.1:8000/",
      authMode: "single-user",
      apiKey: "weird/key=with+seps",
      accessToken: ""
    })

    expect(protocols).toEqual([])
    expect(url).toContain("api_key=")
    expect(url).toContain(encodeURIComponent("weird/key=with+seps"))
  })

  it("keeps a token-safe key (token_urlsafe / hex style) in the subprotocol", () => {
    const { url, protocols } = buildPersonaWebSocketUrl({
      serverUrl: "http://127.0.0.1:8000/",
      authMode: "single-user",
      apiKey: "abc-123_XYZ",
      accessToken: ""
    })

    expect(protocols).toEqual(["bearer", "abc-123_XYZ"])
    expect(url).not.toContain("abc-123_XYZ")
  })

  it("throws when auth secret is missing for selected auth mode", () => {
    expect(() =>
      buildPersonaWebSocketUrl({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "",
        accessToken: ""
      })
    ).toThrowError(/API key missing/i)

    expect(() =>
      buildPersonaWebSocketUrl({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "multi-user",
        apiKey: "",
        accessToken: ""
      })
    ).toThrowError(/Not authenticated/i)
  })
})
