import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  ConnectionPhase,
  deriveConnectionUxState,
  type ConnectionState,
  type ConnectionUxState
} from "@/types/connection"

vi.mock("@/services/tldw-server-url", () => ({
  getStoredTldwServerURL: vi.fn(async () => null)
}))

vi.mock("@/services/api-send", () => ({
  apiSend: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: vi.fn(),
    initialize: vi.fn(),
    ragHealth: vi.fn(),
    updateConfig: vi.fn(),
    clearManualSingleUserCredentials: vi.fn()
  }
}))

vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: vi.fn(() => null),
  isCookieSessionConfigInvalidated: vi.fn(() => false)
}))

import { apiSend } from "@/services/api-send"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { getRuntimeSingleUserApiKeyOverride, isCookieSessionConfigInvalidated } from "@/services/tldw/runtime-auth-override"
import { CONNECTION_TIMEOUT_MS, useConnectionStore } from "../connection"

const mockedApiSend = vi.mocked(apiSend)
const mockedClient = vi.mocked(tldwClient, true)
const mockedRuntimeApiKey = vi.mocked(getRuntimeSingleUserApiKeyOverride)
const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
// Fixed wall-clock so state-setup timestamps are deterministic across runs
// (workflow code must use real Date.now(), but test fixtures should not).
const FIXED_NOW_MS = 1_700_000_000_000

const setConnectionState = (overrides: Record<string, unknown>) => {
  const prev = useConnectionStore.getState().state
  useConnectionStore.setState({
    state: {
      ...prev,
      ...overrides
    }
  })
}

const ageLastCheck = () => {
  setConnectionState({
    lastCheckedAt: Date.now() - 60_000,
    isChecking: false
  })
}

const createBaseConnectionState = (): ConnectionState => ({
  phase: ConnectionPhase.UNCONFIGURED,
  serverUrl: null,
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
  checksSinceConfigChange: 0
})

type ConnectionUxMatrixCase = {
  name: string
  overrides: Partial<ConnectionState>
  expected: ConnectionUxState
}

const connectionUxMatrix: ConnectionUxMatrixCase[] = [
  {
    name: "unconfigured first run",
    overrides: {},
    expected: "unconfigured"
  },
  {
    name: "URL configuration step",
    overrides: { configStep: "url" },
    expected: "configuring_url"
  },
  {
    name: "auth configuration step",
    overrides: { configStep: "auth" },
    expected: "configuring_auth"
  },
  {
    name: "active health check from setup",
    overrides: { configStep: "health", isChecking: true },
    expected: "testing"
  },
  {
    name: "searching health check",
    overrides: {
      phase: ConnectionPhase.SEARCHING,
      configStep: "health",
      isChecking: true
    },
    expected: "testing"
  },
  {
    name: "connected ready backend",
    overrides: {
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      serverUrl: "http://127.0.0.1:8000",
      knowledgeStatus: "ready",
      hasCompletedFirstRun: true
    },
    expected: "connected_ok"
  },
  {
    name: "connected partial error",
    overrides: {
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      serverUrl: "http://127.0.0.1:8000",
      errorKind: "partial",
      knowledgeStatus: "ready",
      hasCompletedFirstRun: true
    },
    expected: "connected_degraded"
  },
  {
    name: "connected with offline knowledge",
    overrides: {
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      serverUrl: "http://127.0.0.1:8000",
      knowledgeStatus: "offline",
      hasCompletedFirstRun: true
    },
    expected: "connected_degraded"
  },
  {
    name: "auth error",
    overrides: {
      phase: ConnectionPhase.ERROR,
      errorKind: "auth",
      serverUrl: "http://127.0.0.1:8000"
    },
    expected: "error_auth"
  },
  {
    name: "unreachable error",
    overrides: {
      phase: ConnectionPhase.ERROR,
      errorKind: "unreachable",
      serverUrl: "http://127.0.0.1:8000"
    },
    expected: "error_unreachable"
  },
  {
    name: "demo mode",
    overrides: {
      mode: "demo",
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      offlineBypass: true
    },
    expected: "demo_mode"
  }
]

describe("connection store stability", () => {
  const originalChrome = (
    globalThis as typeof globalThis & {
      chrome?: unknown
    }
  ).chrome

  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.removeItem("__tldw_allow_offline")
    localStorage.removeItem("__tldw_force_unconfigured")
    localStorage.removeItem("__tldw_first_run_complete")

    setConnectionState({
      phase: ConnectionPhase.CONNECTED,
      serverUrl: "http://127.0.0.1:8000",
      isConnected: true,
      isChecking: false,
      lastCheckedAt: Date.now() - 60_000,
      lastError: null,
      lastStatusCode: null,
      errorKind: "none",
      knowledgeStatus: "ready",
      knowledgeError: null,
      knowledgeLastCheckedAt: Date.now(),
      consecutiveFailures: 0
    })

    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    } as any)
    mockedClient.initialize.mockResolvedValue(undefined)
    mockedClient.updateConfig.mockReset()
    mockedClient.ragHealth.mockResolvedValue({ status: "healthy" } as any)
    mockedRuntimeApiKey.mockReturnValue(null)
    vi.mocked(isCookieSessionConfigInvalidated).mockReturnValue(false)
  })

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
    if (typeof originalChrome === "undefined") {
      Reflect.deleteProperty(globalThis, "chrome")
      return
    }

    Object.defineProperty(globalThis, "chrome", {
      value: originalChrome,
      configurable: true
    })
  })

  it.each(connectionUxMatrix)(
    "maps $name to $expected in the setup connection UX state matrix",
    ({ overrides, expected }) => {
      expect(
        deriveConnectionUxState({
          ...createBaseConnectionState(),
          ...overrides
        })
      ).toBe(expected)
    }
  )

  it("keeps connected state through transient unreachable checks before threshold", async () => {
    mockedApiSend.mockResolvedValue({
      ok: false,
      status: 0,
      error: "timeout"
    })

    await useConnectionStore.getState().checkOnce()
    let state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(state.errorKind).toBe("partial")
    expect(state.consecutiveFailures).toBe(1)

    ageLastCheck()
    await useConnectionStore.getState().checkOnce()
    state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(state.consecutiveFailures).toBe(2)

    ageLastCheck()
    await useConnectionStore.getState().checkOnce()
    state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.ERROR)
    expect(state.isConnected).toBe(false)
    expect(state.errorKind).toBe("unreachable")
    expect(state.consecutiveFailures).toBe(3)
  })

  it("falls back to localStorage when chrome storage lacks the first-run flag", async () => {
    Object.defineProperty(globalThis, "chrome", {
      value: {
        storage: {
          local: {
            get: vi.fn((key: string, callback: (value: Record<string, unknown>) => void) => {
              callback({})
            })
          }
        }
      },
      configurable: true
    })

    localStorage.setItem("__tldw_first_run_complete", "true")
    mockedApiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "alive" }
    })

    setConnectionState({
      hasCompletedFirstRun: false,
      phase: ConnectionPhase.SEARCHING,
      isConnected: false,
      lastCheckedAt: Date.now() - 60_000
    })

    await useConnectionStore.getState().checkOnce()

    const state = useConnectionStore.getState().state
    expect(state.hasCompletedFirstRun).toBe(true)
  })

  it("connects unverified multi-user sessions without profile or operator health permissions", async () => {
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "ordinary-user-token"
    })
    mockedApiSend.mockImplementation(async ({ path }) => path === "/api/v1/auth/sessions"
      ? { ok: true, status: 200, data: [] }
      : { ok: false, status: 403, error: "Email not verified or missing system.logs" })
    await useConnectionStore.getState().checkOnce()
    expect(useConnectionStore.getState().state.isConnected).toBe(true)
    expect(mockedApiSend).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/auth/sessions", noAuth: false }))
    expect(mockedApiSend).not.toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/health/live" }))
  })

  it("uses lightweight health liveness endpoint and resets failure streak on success", async () => {
    setConnectionState({
      consecutiveFailures: 2,
      errorKind: "partial",
      lastError: "timeout"
    })
    mockedApiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "alive" }
    })

    await useConnectionStore.getState().checkOnce()

    const state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(state.consecutiveFailures).toBe(0)
    expect(state.lastError).toBeNull()
    expect(mockedApiSend).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/health/live",
        method: "GET",
        timeoutMs: CONNECTION_TIMEOUT_MS
      })
    )
  })

  it.each([
    ["active", true], ["invalidated", false], ["foreign-origin", false]
  ])("uses %s cookie-session readiness without an API key", async (kind, expected) => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: kind === "foreign-origin" ? "https://foreign.example" : window.location.origin,
      authMode: "single-user", authSource: "cookie-session"
    })
    vi.mocked(isCookieSessionConfigInvalidated).mockReturnValue(kind === "invalidated")
    mockedApiSend.mockResolvedValue({ ok: true, status: 200, data: { status: "alive" } })
    await useConnectionStore.getState().checkOnce()
    expect(useConnectionStore.getState().state.isConnected).toBe(expected)
    if (expected) expect(mockedApiSend).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/auth/sessions", noAuth: false }))
    else expect(useConnectionStore.getState().state.configStep).toBe("auth")
  })

  it.each(["absent", "expired", "revoked"])(
    "rejects %s cookies even when persisted transport metadata and public liveness are valid",
    async (reason) => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      mockedClient.getConfig.mockResolvedValue({
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session"
      })
      mockedApiSend.mockImplementation(async ({ path }) =>
        path === "/api/v1/auth/sessions"
          ? { ok: false, status: 401, error: `Session ${reason}` }
          : { ok: true, status: 200, data: { status: "alive" } }
      )

      await useConnectionStore.getState().checkOnce()

      expect(useConnectionStore.getState().state).toMatchObject({
        phase: ConnectionPhase.ERROR,
        isConnected: false,
        errorKind: "auth",
        lastStatusCode: 401
      })
      expect(mockedClient.ragHealth).not.toHaveBeenCalled()
    }
  )

  it("treats runtime single-user auth as configured without persisting an api key", async () => {
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    } as any)
    mockedRuntimeApiKey.mockReturnValue("runtime-key")
    mockedApiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "alive" }
    })

    await useConnectionStore.getState().checkOnce()

    const state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(mockedApiSend).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/health/live",
        method: "GET",
        noAuth: false
      })
    )
  })

  it.each(["   ", "CHANGE_ME_TO_SECURE_API_KEY"])(
    "treats invalid runtime single-user auth %s as missing credentials",
    async (runtimeKey) => {
      mockedClient.getConfig.mockResolvedValue({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user"
      } as any)
      mockedRuntimeApiKey.mockReturnValue(runtimeKey)

      await useConnectionStore.getState().checkOnce()

      const state = useConnectionStore.getState().state
      expect(state.phase).toBe(ConnectionPhase.UNCONFIGURED)
      expect(state.isConnected).toBe(false)
      expect(state.configStep).toBe("auth")
      expect(mockedApiSend).not.toHaveBeenCalled()
    }
  )

  it("can force a fresh health check after a recent connected state", async () => {
    setConnectionState({
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      isChecking: false,
      lastCheckedAt: Date.now(),
      consecutiveFailures: 1,
      errorKind: "partial",
      lastError: "previous transient failure"
    })
    mockedApiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "alive" }
    })

    await useConnectionStore.getState().checkOnce({ force: true })

    const state = useConnectionStore.getState().state
    expect(mockedApiSend).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/health/live",
        method: "GET"
      })
    )
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(state.consecutiveFailures).toBe(0)
    expect(state.lastError).toBeNull()
  })

  it("surfaces a CORS hint for cross-origin network-blocked health checks", async () => {
    setConnectionState({
      phase: ConnectionPhase.SEARCHING,
      serverUrl: "http://192.168.5.186:8000",
      isConnected: false,
      isChecking: false,
      lastCheckedAt: Date.now() - 60_000,
      consecutiveFailures: 0
    })
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://192.168.5.186:8000",
      authMode: "single-user",
      apiKey: "test-key"
    } as any)
    mockedApiSend.mockResolvedValue({
      ok: false,
      status: 0,
      error: "NetworkError when attempting to fetch resource."
    })

    await useConnectionStore.getState().checkOnce()

    const state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.ERROR)
    expect(state.errorKind).toBe("unreachable")
    expect(state.lastError).toContain("Likely CORS mismatch")
    expect(state.lastError).toContain("ALLOWED_ORIGINS")
  })

  it("surfaces a CORS/network hint for aborted cross-origin health checks", async () => {
    setConnectionState({
      phase: ConnectionPhase.SEARCHING,
      serverUrl: "http://192.168.5.186:8000",
      isConnected: false,
      isChecking: false,
      lastCheckedAt: Date.now() - 60_000,
      consecutiveFailures: 0
    })
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://192.168.5.186:8000",
      authMode: "single-user",
      apiKey: "test-key"
    } as any)
    mockedApiSend.mockResolvedValue({
      ok: false,
      status: 0,
      error: "The operation was aborted."
    })

    await useConnectionStore.getState().checkOnce()

    const state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.ERROR)
    expect(state.errorKind).toBe("unreachable")
    expect(state.lastError).toContain("Likely CORS mismatch")
    expect(state.lastError).toContain("ALLOWED_ORIGINS")
  })

  it("recovers from stale LAN host by switching to current browser host when probe succeeds", async () => {
    setConnectionState({
      phase: ConnectionPhase.SEARCHING,
      serverUrl: "http://192.168.5.186:8000",
      isConnected: false,
      isChecking: false,
      lastCheckedAt: Date.now() - 60_000,
      consecutiveFailures: 0
    })
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://192.168.5.186:8000",
      authMode: "single-user",
      apiKey: "test-key"
    } as any)
    mockedApiSend
      .mockResolvedValueOnce({
        ok: false,
        status: 0,
        error: "NetworkError when attempting to fetch resource."
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        data: { status: "alive" }
      })

    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue({
        ok: true,
        status: 200
      } as Response)

    const originalWindow = globalThis.window
    Object.defineProperty(globalThis, "window", {
      value: {
        location: {
          origin: "http://192.168.5.184:3000",
          hostname: "192.168.5.184"
        }
      },
      configurable: true
    })

    try {
      await useConnectionStore.getState().checkOnce()
    } finally {
      fetchMock.mockRestore()
      Object.defineProperty(globalThis, "window", {
        value: originalWindow,
        configurable: true
      })
    }

    const state = useConnectionStore.getState().state
    expect(mockedClient.updateConfig).toHaveBeenCalledWith({
      serverUrl: "http://192.168.5.184:8000"
    })
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(state.serverUrl).toBe("http://192.168.5.184:8000")
    expect(mockedApiSend).toHaveBeenCalledTimes(2)
  })

  it("canonicalizes quickstart webui health checks to the current page origin", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    setConnectionState({
      phase: ConnectionPhase.SEARCHING,
      serverUrl: "http://127.0.0.1:8000",
      isConnected: false,
      isChecking: false,
      lastCheckedAt: Date.now() - 60_000,
      consecutiveFailures: 0
    })
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    } as any)
    mockedApiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "alive" }
    })

    const fetchMock = vi.spyOn(globalThis, "fetch")

    const originalWindow = globalThis.window
    Object.defineProperty(globalThis, "window", {
      value: {
        location: {
          origin: "http://192.168.5.184:3000",
          hostname: "192.168.5.184",
          protocol: "http:"
        }
      },
      configurable: true
    })

    try {
      await useConnectionStore.getState().checkOnce()
    } finally {
      fetchMock.mockRestore()
      Object.defineProperty(globalThis, "window", {
        value: originalWindow,
        configurable: true
      })
    }

    const state = useConnectionStore.getState().state
    expect(mockedClient.updateConfig).toHaveBeenCalledWith({
      serverUrl: "http://192.168.5.184:3000"
    })
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(state.serverUrl).toBe("http://192.168.5.184:3000")
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("canonicalizes explicit custom hosts to the webui origin in quickstart mode", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    setConnectionState({
      phase: ConnectionPhase.SEARCHING,
      serverUrl: "https://api.example.test:9443",
      isConnected: false,
      isChecking: false,
      lastCheckedAt: Date.now() - 60_000,
      consecutiveFailures: 0
    })
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "https://api.example.test:9443",
      authMode: "single-user",
      apiKey: "test-key"
    } as any)
    mockedApiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "alive" }
    })

    const fetchMock = vi.spyOn(globalThis, "fetch")

    const originalWindow = globalThis.window
    Object.defineProperty(globalThis, "window", {
      value: {
        location: {
          origin: "http://192.168.5.184:3000",
          hostname: "192.168.5.184",
          protocol: "http:"
        }
      },
      configurable: true
    })

    try {
      await useConnectionStore.getState().checkOnce()
    } finally {
      Object.defineProperty(globalThis, "window", {
        value: originalWindow,
        configurable: true
      })
      fetchMock.mockRestore()
    }

    const state = useConnectionStore.getState().state
    expect(mockedClient.updateConfig).toHaveBeenCalledWith({
      serverUrl: "http://192.168.5.184:3000"
    })
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(state.serverUrl).toBe("http://192.168.5.184:3000")
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("uses the shared quickstart liveness path for recovery probes", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    setConnectionState({
      phase: ConnectionPhase.SEARCHING,
      serverUrl: "http://192.168.5.186:8000",
      isConnected: false,
      isChecking: false,
      lastCheckedAt: Date.now() - 60_000,
      consecutiveFailures: 0
    })
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://192.168.5.186:8000",
      authMode: "single-user",
      apiKey: "test-key"
    } as any)
    mockedApiSend
      .mockResolvedValueOnce({
        ok: false,
        status: 0,
        error: "NetworkError when attempting to fetch resource."
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        data: { status: "alive" }
      })

    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200
    } as Response)

    const originalWindow = globalThis.window
    Object.defineProperty(globalThis, "window", {
      value: {
        location: {
          origin: "http://192.168.5.184:3000",
          hostname: "192.168.5.184",
          protocol: "http:"
        }
      },
      configurable: true
    })

    try {
      await useConnectionStore.getState().checkOnce()

      expect(fetchMock).toHaveBeenCalledWith(
        "/api/v1/health/live",
        expect.objectContaining({
          method: "GET",
          credentials: "omit"
        })
      )
    } finally {
      Object.defineProperty(globalThis, "window", {
        value: originalWindow,
        configurable: true
      })
      fetchMock.mockRestore()
    }
  })

  it("preserves persisted first-run completion when offline bypass is enabled", async () => {
    setConnectionState({
      hasCompletedFirstRun: false,
      phase: ConnectionPhase.SEARCHING,
      isConnected: false,
      isChecking: false
    })
    localStorage.setItem("__tldw_first_run_complete", "true")
    localStorage.setItem("__tldw_allow_offline", "true")

    await useConnectionStore.getState().checkOnce()

    const state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.offlineBypass).toBe(true)
    expect(state.hasCompletedFirstRun).toBe(true)
  })

  it("preserves persisted first-run completion through a successful health check", async () => {
    setConnectionState({
      hasCompletedFirstRun: false,
      phase: ConnectionPhase.SEARCHING,
      isConnected: false,
      isChecking: false
    })
    localStorage.setItem("__tldw_first_run_complete", "true")
    mockedApiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "alive" }
    })

    await useConnectionStore.getState().checkOnce()

    const state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.CONNECTED)
    expect(state.isConnected).toBe(true)
    expect(state.hasCompletedFirstRun).toBe(true)
  })

  it("treats a server URL without single-user credentials as unconfigured auth instead of connected", async () => {
    setConnectionState({
      phase: ConnectionPhase.SEARCHING,
      serverUrl: "http://127.0.0.1:8000",
      isConnected: false,
      isChecking: false,
      lastCheckedAt: Date.now() - 60_000,
      configStep: "health",
      hasCompletedFirstRun: true
    })
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    } as any)
    mockedApiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "alive" }
    })

    await useConnectionStore.getState().checkOnce()

    const state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.UNCONFIGURED)
    expect(state.serverUrl).toBe("http://127.0.0.1:8000")
    expect(state.configStep).toBe("auth")
    expect(state.isConnected).toBe(false)
    expect(state.errorKind).toBe("none")
    expect(mockedApiSend).not.toHaveBeenCalled()
  })

  it("begins onboarding without clearing first-run completion when a saved server still needs auth", async () => {
    setConnectionState({
      phase: ConnectionPhase.ERROR,
      serverUrl: "http://127.0.0.1:8000",
      isConnected: false,
      isChecking: false,
      errorKind: "auth",
      configStep: "health",
      hasCompletedFirstRun: true
    })
    localStorage.setItem("__tldw_first_run_complete", "true")
    mockedClient.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: ""
    } as any)

    await useConnectionStore.getState().beginOnboarding()

    const state = useConnectionStore.getState().state
    expect(state.serverUrl).toBe("http://127.0.0.1:8000")
    expect(state.configStep).toBe("auth")
    expect(state.hasCompletedFirstRun).toBe(true)
    expect(localStorage.getItem("__tldw_first_run_complete")).toBe("true")
  })

  it("restarts onboarding from the beginning only when explicitly requested", async () => {
    setConnectionState({
      phase: ConnectionPhase.ERROR,
      serverUrl: "http://127.0.0.1:8000",
      isConnected: false,
      isChecking: false,
      errorKind: "auth",
      configStep: "auth",
      hasCompletedFirstRun: true
    })
    localStorage.setItem("__tldw_first_run_complete", "true")

    await (useConnectionStore.getState() as any).restartOnboarding()

    const state = useConnectionStore.getState().state
    expect(state.phase).toBe(ConnectionPhase.UNCONFIGURED)
    expect(state.configStep).toBe("url")
    expect(state.hasCompletedFirstRun).toBe(false)
    expect(localStorage.getItem("__tldw_first_run_complete")).toBeNull()
    expect(mockedClient.clearManualSingleUserCredentials).toHaveBeenCalledOnce()
  })

  it("revokes readiness immediately on onboarding restart and discards its pending health check", async () => {
    setConnectionState({
      phase: ConnectionPhase.CONNECTED, isConnected: true, isChecking: false,
      lastCheckedAt: 0, knowledgeStatus: "ready", knowledgeLastCheckedAt: Date.now()
    })
    let finishHealth: (value: unknown) => void = () => {}
    mockedApiSend.mockImplementationOnce(() => new Promise((resolve) => { finishHealth = resolve }))
    const oldCheck = useConnectionStore.getState().checkOnce({ force: true })
    await vi.waitFor(() => expect(mockedApiSend).toHaveBeenCalledOnce())
    let finishClear: () => void = () => {}
    mockedClient.clearManualSingleUserCredentials.mockImplementationOnce(() => new Promise<void>((resolve) => { finishClear = resolve }))
    const restart = useConnectionStore.getState().restartOnboarding()
    await vi.waitFor(() => expect(mockedClient.clearManualSingleUserCredentials).toHaveBeenCalledOnce())
    const connectedWhileClearing = useConnectionStore.getState().state.isConnected
    finishClear()
    await restart
    finishHealth({ ok: true, status: 200, data: {} })
    await oldCheck
    expect(connectedWhileClearing).toBe(false)
    expect(useConnectionStore.getState().state.isConnected).toBe(false)
  })

  it("exits demo mode when entering onboarding so setup does not look connected", async () => {
    setConnectionState({
      mode: "demo",
      phase: ConnectionPhase.CONNECTED,
      serverUrl: null,
      isConnected: true,
      isChecking: false,
      configStep: "health",
      errorKind: "none",
      hasCompletedFirstRun: true
    })
    mockedClient.getConfig.mockResolvedValue(null as any)

    await useConnectionStore.getState().beginOnboarding()

    const state = useConnectionStore.getState().state
    expect(state.mode).toBe("normal")
    expect(state.phase).toBe(ConnectionPhase.UNCONFIGURED)
    expect(state.isConnected).toBe(false)
    expect(state.configStep).toBe("url")
  })

  it("keeps concurrent onboarding edits but discards readiness for a replaced server (H7)", async () => {
    setConnectionState({
      phase: ConnectionPhase.SEARCHING,
      isConnected: false,
      isChecking: false,
      configStep: "url",
      hasCompletedFirstRun: false,
      userPersona: null,
      knowledgeStatus: "ready",
      knowledgeLastCheckedAt: FIXED_NOW_MS,
      lastCheckedAt: FIXED_NOW_MS - 60_000,
      consecutiveFailures: 0,
      errorKind: "none"
    })

    // Gate the health check so it stays in-flight while other actions run.
    let releaseHealth: (value: {
      ok: boolean
      status: number
      data?: unknown
    }) => void = () => {}
    const healthGate = new Promise<{
      ok: boolean
      status: number
      data?: unknown
    }>((resolve) => {
      releaseHealth = resolve
    })
    mockedApiSend.mockReturnValue(healthGate as never)

    // Start the (slow) health check but do not await it yet.
    const checkPromise = useConnectionStore.getState().checkOnce({ force: true })

    // While it is in-flight, concurrent onboarding actions mutate the store.
    await useConnectionStore
      .getState()
      .setConfigPartial({ serverUrl: "http://concurrent.test:9999" })
    await useConnectionStore.getState().markFirstRunComplete()

    // Let the slow health check complete. Its terminal write must merge onto the
    // LATEST state, not the snapshot captured before the concurrent edits.
    releaseHealth({ ok: true, status: 200, data: { status: "alive" } })
    await checkPromise

    const final = useConnectionStore.getState().state
    expect(final.configStep).toBe("auth")
    expect(final.hasCompletedFirstRun).toBe(true)
    expect(final.phase).toBe(ConnectionPhase.UNCONFIGURED)
    expect(final.isConnected).toBe(false)
    expect(final.serverUrl).toBe("http://concurrent.test:9999")
  })

  it.each([
    { serverUrl: "https://server-b.test" },
    { apiKey: "replacement-key" },
    { orgId: 2 },
    { authSource: "cookie-session" as const }
  ])("discards pending readiness after a settings authority replacement: %j", async (change) => {
    let current: TldwConfig = { serverUrl: "https://server-a.test", authMode: "single-user", apiKey: "old-key", orgId: 1 }
    mockedClient.getConfig.mockImplementation(async () => current)
    mockedClient.updateConfig.mockImplementation(async (update) => { current = { ...current, ...update } })
    let resolveProbe: (value: unknown) => void = () => {}
    mockedApiSend.mockImplementationOnce(() => new Promise((resolve) => { resolveProbe = resolve }) as never)
    const previousCheck = useConnectionStore.getState().checkOnce({ force: true })
    await vi.waitFor(() => expect(mockedApiSend).toHaveBeenCalledTimes(1))
    await useConnectionStore.getState().setConfigPartial(change)
    const connectedAfterChange = useConnectionStore.getState().state.isConnected
    resolveProbe({ ok: true, status: 200, data: {} })
    await previousCheck
    expect(connectedAfterChange).toBe(false)
    expect(useConnectionStore.getState().state).toMatchObject({ phase: ConnectionPhase.UNCONFIGURED, isConnected: false })
    expect(mockedClient.ragHealth).not.toHaveBeenCalled()
  })

  it("setServerUrl requires the replacement server's own successful probe", async () => {
    let current: TldwConfig = { serverUrl: "https://server-a.test", authMode: "single-user", apiKey: "old-key" }
    mockedClient.getConfig.mockImplementation(async () => current)
    mockedClient.updateConfig.mockImplementation(async (update) => { current = { ...current, ...update } })
    const probes: Array<(value: unknown) => void> = []
    mockedApiSend.mockImplementation(() => new Promise((resolve) => { probes.push(resolve) }) as never)
    const oldCheck = useConnectionStore.getState().checkOnce({ force: true })
    await vi.waitFor(() => expect(probes).toHaveLength(1))
    const changed = useConnectionStore.getState().setServerUrl("https://server-b.test")
    try {
      await vi.waitFor(() => expect(probes).toHaveLength(2))
    } catch (error) {
      probes[0]({ ok: true, status: 200, data: {} })
      await oldCheck
      throw error
    }
    probes[0]({ ok: true, status: 200, data: {} })
    await oldCheck
    expect(useConnectionStore.getState().state).toMatchObject({ serverUrl: "https://server-b.test", isConnected: false, isChecking: true })
    probes[1]({ ok: true, status: 200, data: {} })
    await changed
    expect(useConnectionStore.getState().state).toMatchObject({ serverUrl: "https://server-b.test", isConnected: true })
  })

  it("invalidates a same-tab authority event synchronously but ignores benign updates", () => {
    window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: false } }))
    expect(useConnectionStore.getState().state.isConnected).toBe(true)
    window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    expect(useConnectionStore.getState().state.isConnected).toBe(false)
  })

  it("discards pending readiness when another tab removes configured credentials", async () => {
    const current = { serverUrl: "https://server-a.test", authMode: "single-user" as const, apiKey: "old-key" }
    mockedClient.getConfig.mockResolvedValue(current)
    let resolveProbe: (value: unknown) => void = () => {}
    mockedApiSend.mockImplementationOnce(() => new Promise((resolve) => { resolveProbe = resolve }) as never)
    const oldCheck = useConnectionStore.getState().checkOnce({ force: true })
    await vi.waitFor(() => expect(mockedApiSend).toHaveBeenCalledTimes(1))
    const cleared = { ...current, apiKey: undefined }
    mockedClient.getConfig.mockResolvedValue(cleared)
    window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: JSON.stringify(current), newValue: JSON.stringify(cleared) }))
    const connectedAfterRemoval = useConnectionStore.getState().state.isConnected
    resolveProbe({ ok: true, status: 200, data: {} })
    await oldCheck
    expect(connectedAfterRemoval).toBe(false)
    await useConnectionStore.getState().checkOnce()
    expect(useConnectionStore.getState().state.isConnected).toBe(false)
    expect(mockedApiSend).toHaveBeenCalledTimes(1)
  })

  it("does not invalidate readiness for cross-tab rotation of the same known principal", () => {
    const current = { serverUrl: "https://server.test", authMode: "multi-user", accessToken: "e30.eyJzdWIiOiJhbGljZSJ9.sig" }
    window.dispatchEvent(new StorageEvent("storage", { key: "tldwConfig", oldValue: JSON.stringify(current), newValue: JSON.stringify({ ...current, accessToken: "e30.eyJzdWIiOiJhbGljZSIsImlhdCI6Mn0.sig" }) }))
    expect(useConnectionStore.getState().state.isConnected).toBe(true)
  })

  it("rechecks normalized quickstart config after its own authority update without looping", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    let current: TldwConfig = { serverUrl: "https://old-server.test", authMode: "single-user", apiKey: "configured-key" }
    mockedClient.getConfig.mockImplementation(async () => current)
    mockedClient.updateConfig.mockImplementation(async (update) => {
      current = { ...current, ...update }
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    })
    mockedApiSend.mockResolvedValue({ ok: true, status: 200, data: {} })
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(useConnectionStore.getState().state).toMatchObject({ serverUrl: window.location.origin, isConnected: true })
    expect(mockedClient.updateConfig).toHaveBeenCalledTimes(1)
    expect(mockedApiSend).toHaveBeenCalledTimes(1)
  })

  it("does not normalize foreign cookie metadata into local authority during revalidation", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    let current: TldwConfig = { serverUrl: "https://foreign.test", authMode: "single-user", authSource: "cookie-session" }
    mockedClient.getConfig.mockImplementation(async () => current)
    mockedClient.updateConfig.mockImplementation(async (update) => {
      current = { ...current, ...update }
      window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
    })
    mockedApiSend.mockResolvedValue({ ok: true, status: 200, data: {} })
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(useConnectionStore.getState().state.isConnected).toBe(false)
    expect(mockedClient.updateConfig).not.toHaveBeenCalled()
    expect(mockedApiSend).not.toHaveBeenCalled()
  })

  it.each(["tldwConfig", "tldwCookieSessionConfig"])("invalidates removed %s transport metadata in another tab", (key) => {
    window.dispatchEvent(new StorageEvent("storage", {
      key, oldValue: JSON.stringify({ serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" }), newValue: null
    }))
    expect(useConnectionStore.getState().state.isConnected).toBe(false)
  })

  it("ignores a concurrent checkOnce while one is already in flight (H7 guard)", async () => {
    setConnectionState({
      phase: ConnectionPhase.SEARCHING,
      isConnected: false,
      isChecking: false,
      knowledgeStatus: "ready",
      knowledgeLastCheckedAt: Date.now(),
      lastCheckedAt: Date.now() - 60_000
    })

    let releaseHealth: (value: {
      ok: boolean
      status: number
      data?: unknown
    }) => void = () => {}
    const healthGate = new Promise<{
      ok: boolean
      status: number
      data?: unknown
    }>((resolve) => {
      releaseHealth = resolve
    })
    mockedApiSend.mockReturnValue(healthGate as never)

    // The in-flight guard is claimed synchronously (before the first await), so
    // the second call must bail before issuing its own health request.
    const first = useConnectionStore.getState().checkOnce({ force: true })
    const second = useConnectionStore.getState().checkOnce({ force: true })

    releaseHealth({ ok: true, status: 200, data: { status: "alive" } })
    await Promise.all([first, second])

    expect(mockedApiSend).toHaveBeenCalledTimes(1)
  })

  it.each(["manual-key", "cookie-session"])("invalidates logout authority until %s reconnect verifies", async (kind) => {
    if (kind === "cookie-session") {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
      mockedClient.getConfig.mockResolvedValue({
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session"
      })
      vi.mocked(isCookieSessionConfigInvalidated).mockReturnValue(true)
    } else {
      mockedClient.getConfig.mockResolvedValue({ serverUrl: "http://127.0.0.1:8000", authMode: "single-user" })
    }
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
    expect(useConnectionStore.getState().state).toMatchObject({
      phase: ConnectionPhase.UNCONFIGURED, configStep: "auth", isConnected: false,
      isChecking: false, knowledgeStatus: "unknown", lastCheckedAt: null
    })
    await useConnectionStore.getState().checkOnce()
    expect(mockedApiSend).not.toHaveBeenCalled()

    if (kind === "cookie-session") {
      vi.mocked(isCookieSessionConfigInvalidated).mockReturnValue(false)
    } else {
      mockedClient.getConfig.mockResolvedValue({ serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "reentered-key" })
    }
    let resolveProbe: (value: unknown) => void = () => {}
    mockedApiSend.mockImplementation(() => new Promise((resolve) => { resolveProbe = resolve }) as never)
    const reconnect = useConnectionStore.getState().checkOnce()
    await vi.waitFor(() => expect(mockedApiSend).toHaveBeenCalledTimes(1))
    expect(useConnectionStore.getState().state.isConnected).toBe(false)
    resolveProbe({ ok: true, status: 200, data: {} })
    await reconnect
    expect(useConnectionStore.getState().state.isConnected).toBe(true)
    expect(mockedApiSend).toHaveBeenCalledWith(expect.objectContaining({
      path: kind === "cookie-session" ? "/api/v1/auth/sessions" : "/api/v1/health/live",
      noAuth: false
    }))
  })

  it("discards a pre-logout health result without releasing the reconnect check guard", async () => {
    const resolvers: Array<(value: unknown) => void> = []
    mockedApiSend.mockImplementation(() => new Promise((resolve) => { resolvers.push(resolve) }) as never)
    const oldCheck = useConnectionStore.getState().checkOnce({ force: true })
    await vi.waitFor(() => expect(resolvers).toHaveLength(1))
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
    const reconnect = useConnectionStore.getState().checkOnce({ force: true })
    await vi.waitFor(() => expect(resolvers).toHaveLength(2))

    resolvers[0]({ ok: true, status: 200, data: {} })
    await oldCheck
    expect(useConnectionStore.getState().state).toMatchObject({ isConnected: false, isChecking: true })
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(resolvers).toHaveLength(2)
    expect(mockedClient.ragHealth).not.toHaveBeenCalled()

    resolvers[1]({ ok: true, status: 200, data: {} })
    await reconnect
    expect(useConnectionStore.getState().state.isConnected).toBe(true)
  })

  it("discards a pre-logout knowledge result without restoring private readiness", async () => {
    setConnectionState({ knowledgeStatus: "unknown", knowledgeLastCheckedAt: null })
    mockedApiSend.mockResolvedValue({ ok: true, status: 200, data: {} })
    let resolveRag: (value: unknown) => void = () => {}
    mockedClient.ragHealth.mockImplementationOnce(() => new Promise((resolve) => { resolveRag = resolve }) as never)
    const oldCheck = useConnectionStore.getState().checkOnce({ force: true })
    await vi.waitFor(() => expect(mockedClient.ragHealth).toHaveBeenCalledTimes(1))
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
    resolveRag({ status: "healthy" })
    await oldCheck
    expect(useConnectionStore.getState().state).toMatchObject({ isConnected: false, knowledgeStatus: "unknown" })
  })

  it("releases the in-flight guard when a step before the health check throws (H7 deadlock)", async () => {
    // A first-run flag in storage makes checkOnce run its first-run-sync set(...)
    // BEFORE it flips isChecking, so a throw there leaves isChecking false and the
    // synchronous in-flight guard as the only thing that could block a retry.
    setConnectionState({
      phase: ConnectionPhase.CONNECTED,
      serverUrl: "http://127.0.0.1:8000",
      isConnected: true,
      isChecking: false,
      hasCompletedFirstRun: false,
      userPersona: null,
      knowledgeStatus: "ready",
      knowledgeLastCheckedAt: FIXED_NOW_MS,
      lastCheckedAt: FIXED_NOW_MS - 60_000
    })
    localStorage.setItem("__tldw_first_run_complete", "true")
    mockedApiSend.mockResolvedValue({
      ok: true,
      status: 200,
      data: { status: "alive" }
    } as never)

    // Throw from a store subscriber to simulate a pre-`try` step failing.
    const unsubscribe = useConnectionStore.subscribe(() => {
      throw new Error("pre-check boom")
    })
    await expect(
      useConnectionStore.getState().checkOnce({ force: true })
    ).rejects.toThrow("pre-check boom")
    unsubscribe()

    // If the guard had leaked, this second checkOnce would bail before issuing a
    // health request; it must run and reach apiSend instead.
    mockedApiSend.mockClear()
    await useConnectionStore.getState().checkOnce({ force: true })
    expect(mockedApiSend).toHaveBeenCalled()
  })
})
