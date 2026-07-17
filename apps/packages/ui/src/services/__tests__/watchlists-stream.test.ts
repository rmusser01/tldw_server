import { afterEach, beforeEach, describe, expect, it } from "vitest"
import {
  buildWatchlistsRunWebSocketUrl,
  parseWatchlistsRunStreamPayload
} from "../watchlists-stream"

describe("buildWatchlistsRunWebSocketUrl", () => {
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

    const url = buildWatchlistsRunWebSocketUrl(
      {
        serverUrl: "http://127.0.0.1:8000/",
        authMode: "single-user",
        apiKey: "abc123",
        accessToken: ""
      },
      42
    )

    expect(url).toBe(
      "ws://127.0.0.1:8080/api/v1/watchlists/runs/42/stream?api_key=abc123"
    )
  })

  it("uses a secret-free page-origin url for cookie sessions", () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const url = buildWatchlistsRunWebSocketUrl(
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
      "ws://127.0.0.1:8080/api/v1/watchlists/runs/42/stream"
    )
  })

  it("builds api-key websocket url for single-user mode", () => {
    const url = buildWatchlistsRunWebSocketUrl(
      {
        serverUrl: "http://127.0.0.1:8000/",
        authMode: "single-user",
        apiKey: "abc123",
        accessToken: ""
      },
      42
    )

    expect(url).toBe(
      "ws://127.0.0.1:8000/api/v1/watchlists/runs/42/stream?api_key=abc123"
    )
  })

  it("builds token websocket url for multi-user mode", () => {
    const url = buildWatchlistsRunWebSocketUrl(
      {
        serverUrl: "https://example.com",
        authMode: "multi-user",
        apiKey: "",
        accessToken: "jwt-token"
      },
      7
    )

    expect(url).toBe(
      "wss://example.com/api/v1/watchlists/runs/7/stream?token=jwt-token"
    )
  })

  it("throws for invalid run id", () => {
    expect(() =>
      buildWatchlistsRunWebSocketUrl(
        {
          serverUrl: "https://example.com",
          authMode: "single-user",
          apiKey: "abc",
          accessToken: ""
        },
        0
      )
    ).toThrowError(/invalid run id/i)
  })
})

describe("parseWatchlistsRunStreamPayload", () => {
  it("parses snapshot payloads", () => {
    const event = parseWatchlistsRunStreamPayload({
      type: "snapshot",
      run: {
        id: 10,
        job_id: 2,
        status: "running",
        started_at: "2026-02-18T20:00:00Z",
        finished_at: null
      },
      stats: { items_found: "5", items_ingested: 3 },
      error_msg: null,
      log_tail: "line 1",
      log_truncated: true
    })

    expect(event).toEqual({
      type: "snapshot",
      run: {
        id: 10,
        job_id: 2,
        status: "running",
        started_at: "2026-02-18T20:00:00Z",
        finished_at: null
      },
      stats: { items_found: 5, items_ingested: 3 },
      error_msg: null,
      log_tail: "line 1",
      log_truncated: true
    })
  })

  it("parses run updates and log events", () => {
    expect(
      parseWatchlistsRunStreamPayload({
        type: "run_update",
        run: { id: 10, job_id: 2, status: "completed" },
        stats: { items_ingested: 8 }
      })
    ).toEqual({
      type: "run_update",
      run: { id: 10, job_id: 2, status: "completed", started_at: null, finished_at: null },
      stats: { items_ingested: 8 },
      error_msg: null
    })

    expect(
      parseWatchlistsRunStreamPayload({
        type: "log",
        text: "processed 5 items"
      })
    ).toEqual({
      type: "log",
      text: "processed 5 items"
    })
  })

  it("rejects malformed payloads", () => {
    expect(parseWatchlistsRunStreamPayload(null)).toBeNull()
    expect(parseWatchlistsRunStreamPayload({ type: "snapshot", run: {} })).toBeNull()
    expect(parseWatchlistsRunStreamPayload({ type: "log", text: 10 })).toBeNull()
  })
})
