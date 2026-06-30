import { describe, expect, it } from "vitest"
import { ConnectionPhase, deriveConnectionUxState, type ConnectionState } from "../connection"

const makeState = (overrides: Partial<ConnectionState> = {}): ConnectionState => ({
  phase: ConnectionPhase.SEARCHING,
  serverUrl: "http://127.0.0.1:8000",
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
  hasCompletedFirstRun: false,
  userPersona: null,
  lastConfigUpdatedAt: null,
  checksSinceConfigChange: 0,
  ...overrides
})

describe("deriveConnectionUxState", () => {
  it.each([
    [
      "unconfigured first run",
      {
        phase: ConnectionPhase.UNCONFIGURED,
        configStep: "none"
      },
      "unconfigured"
    ],
    [
      "server URL entry",
      {
        phase: ConnectionPhase.UNCONFIGURED,
        configStep: "url"
      },
      "configuring_url"
    ],
    [
      "auth entry",
      {
        phase: ConnectionPhase.UNCONFIGURED,
        configStep: "auth"
      },
      "configuring_auth"
    ],
    [
      "health test running",
      {
        phase: ConnectionPhase.SEARCHING,
        isChecking: true
      },
      "testing"
    ],
    [
      "connected and knowledge ready",
      {
        phase: ConnectionPhase.CONNECTED,
        isConnected: true,
        knowledgeStatus: "ready"
      },
      "connected_ok"
    ],
    [
      "connected with partial health",
      {
        phase: ConnectionPhase.CONNECTED,
        isConnected: true,
        errorKind: "partial"
      },
      "connected_degraded"
    ],
    [
      "connected with offline knowledge",
      {
        phase: ConnectionPhase.CONNECTED,
        isConnected: true,
        knowledgeStatus: "offline"
      },
      "connected_degraded"
    ],
    [
      "auth failure",
      {
        phase: ConnectionPhase.ERROR,
        errorKind: "auth"
      },
      "error_auth"
    ],
    [
      "unreachable backend",
      {
        phase: ConnectionPhase.ERROR,
        errorKind: "unreachable"
      },
      "error_unreachable"
    ],
    [
      "demo mode",
      {
        mode: "demo",
        phase: ConnectionPhase.CONNECTED,
        isConnected: true
      },
      "demo_mode"
    ]
  ] as const)("maps %s to %s", (_label, overrides, expected) => {
    expect(deriveConnectionUxState(makeState(overrides))).toBe(expected)
  })

  it("keeps connected UX during background checks", () => {
    const state = makeState({
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      isChecking: true
    })
    expect(deriveConnectionUxState(state)).toBe("connected_ok")
  })

  it("shows testing while actively searching before connection", () => {
    const state = makeState({
      phase: ConnectionPhase.SEARCHING,
      isConnected: false,
      isChecking: true
    })
    expect(deriveConnectionUxState(state)).toBe("testing")
  })
})
