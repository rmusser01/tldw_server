import { describe, expect, it, vi } from "vitest"

import {
  isAwaitingFirstConnectionCheck,
  notifySendNeedsConnection
} from "../composer-connection"
import { ConnectionPhase, type ConnectionState } from "@/types/connection"

const state = (patch: Partial<ConnectionState>): ConnectionState => ({
  phase: ConnectionPhase.SEARCHING,
  serverUrl: "http://tldw.test",
  lastCheckedAt: null,
  lastError: null,
  lastStatusCode: null,
  isConnected: false,
  isChecking: false,
  consecutiveFailures: 0,
  offlineBypass: false,
  knowledgeStatus: "unknown",
  knowledgeLastCheckedAt: null,
  knowledgeError: null,
  mode: "normal",
  configStep: "none",
  errorKind: "none",
  hasCompletedFirstRun: true,
  userPersona: null,
  lastConfigUpdatedAt: null,
  checksSinceConfigChange: 0,
  ...patch
})

// CC-03 (#3101): "offline" only after a check has failed.
describe("isAwaitingFirstConnectionCheck", () => {
  it("is true before any check has finished", () => {
    expect(isAwaitingFirstConnectionCheck(state({}))).toBe(true)
    expect(isAwaitingFirstConnectionCheck(state({ isChecking: true }))).toBe(true)
  })

  it("is false once a check has failed, including while it retries", () => {
    const failedAt = Date.now()
    expect(
      isAwaitingFirstConnectionCheck(
        state({ phase: ConnectionPhase.ERROR, lastCheckedAt: failedAt, consecutiveFailures: 1 })
      )
    ).toBe(false)
    expect(
      isAwaitingFirstConnectionCheck(
        state({ isChecking: true, lastCheckedAt: failedAt, consecutiveFailures: 1 })
      )
    ).toBe(false)
    // Auth failures reset the failure count but still record the check.
    expect(
      isAwaitingFirstConnectionCheck(
        state({ isChecking: true, lastCheckedAt: failedAt, consecutiveFailures: 0 })
      )
    ).toBe(false)
  })

  it("is false when connected, unconfigured or in demo mode", () => {
    expect(
      isAwaitingFirstConnectionCheck(
        state({ phase: ConnectionPhase.CONNECTED, isConnected: true })
      )
    ).toBe(false)
    expect(
      isAwaitingFirstConnectionCheck(state({ phase: ConnectionPhase.UNCONFIGURED }))
    ).toBe(false)
    expect(isAwaitingFirstConnectionCheck(state({ mode: "demo" }))).toBe(false)
  })

  it("tolerates partial connection states", () => {
    expect(
      isAwaitingFirstConnectionCheck({ phase: "disconnected", isConnected: false } as never)
    ).toBe(false)
  })
})

describe("notifySendNeedsConnection", () => {
  const t = (_key: string, fallback: string) => fallback

  it("says the app is still connecting during the first check", () => {
    const info = vi.fn()
    notifySendNeedsConnection({ info } as never, t, true)
    expect(info).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("Still connecting to your tldw server")
      })
    )
  })

  it("asks the user to connect once a check has failed", () => {
    const info = vi.fn()
    notifySendNeedsConnection({ info } as never, t, false)
    expect(info).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.stringContaining("Connect to your tldw server before sending")
      })
    )
  })

  it("reuses one notice so repeated Enter presses do not stack toasts", () => {
    const info = vi.fn()
    notifySendNeedsConnection({ info } as never, t, false)
    notifySendNeedsConnection({ info } as never, t, false)
    const keys = info.mock.calls.map(([config]) => config.key)
    expect(keys[0]).toBeTruthy()
    expect(keys[1]).toBe(keys[0])
  })
})
