import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  ConnectionPhase,
  deriveConnectionUxState,
  type ConnectionState,
  type ConnectionUxState
} from "@/types/connection"

vi.mock("@/services/tldw-server", () => ({
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
    updateConfig: vi.fn()
  }
}))

import { apiSend } from "@/services/api-send"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { CONNECTION_TIMEOUT_MS, useConnectionStore } from "../connection"

const mockedApiSend = vi.mocked(apiSend)
const mockedClient = vi.mocked(tldwClient, true)
const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

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
    mockedClient.ragHealth.mockResolvedValue({ status: "healthy" } as any)
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
})
