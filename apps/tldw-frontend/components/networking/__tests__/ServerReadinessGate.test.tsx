import { act, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const connectionStoreMock = vi.hoisted(() => ({
  store: {
    state: {
      serverUrl: ""
    }
  }
}))

vi.mock("@tldw/ui/store/connection", () => ({
  useConnectionStore: (
    selector: (store: typeof connectionStoreMock.store) => unknown
  ) => selector(connectionStoreMock.store)
}))

describe("ServerReadinessGate", () => {
  const expectReadinessStatus = () => {
    const status = screen.getByRole("status")
    expect(status).toHaveTextContent(/Checking server readiness|Retrying server readiness/)
    expect(status).toHaveTextContent(/Loading|Retrying/)
  }

  beforeEach(() => {
    connectionStoreMock.store.state.serverUrl = ""
  })

  afterEach(() => {
    try {
      localStorage.removeItem("__tldw_allow_offline")
      localStorage.removeItem("__tldw_test_bypass")
    } catch {
      // ignore test storage availability
    }
    delete (window as unknown as { __tldwServerReadinessState?: unknown })
      .__tldwServerReadinessState
    vi.restoreAllMocks()
    vi.useRealTimers()
    vi.unstubAllEnvs()
    vi.resetModules()
  })

  it("accepts the backend healthy status envelope", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: "healthy" })
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(screen.getByText("App ready")).toBeInTheDocument()
    })

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/health",
      expect.objectContaining({ method: "GET" })
    )
  })

  it("uses the saved server URL over the page and environment origin", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    connectionStoreMock.store.state.serverUrl = " http://10.0.0.5:9000/ "
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: "healthy" })
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(screen.getByText("App ready")).toBeInTheDocument()
    })

    expect(fetchMock).toHaveBeenCalledWith(
      "http://10.0.0.5:9000/api/v1/health",
      expect.objectContaining({ method: "GET" })
    )
  })

  it("normalizes the saved server URL to its origin before probing health", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    connectionStoreMock.store.state.serverUrl =
      " https://user:secret@example.test/base/path?api_key=hidden "
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: "healthy" })
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(screen.getByText("App ready")).toBeInTheDocument()
    })

    expect(fetchMock).toHaveBeenCalledWith(
      "https://example.test/api/v1/health",
      expect.objectContaining({ method: "GET" })
    )
  })

  it("falls back to the environment origin for invalid saved server URLs", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    connectionStoreMock.store.state.serverUrl = "not a url"
    const warnMock = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: "healthy" })
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(screen.getByText("App ready")).toBeInTheDocument()
    })

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/health",
      expect.objectContaining({ method: "GET" })
    )
    expect(warnMock).toHaveBeenCalledWith(
      "Ignoring invalid tldw server URL for readiness health check."
    )
  })

  it("falls back to the environment origin for unsupported saved server URL protocols", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    connectionStoreMock.store.state.serverUrl =
      "ftp://user:secret@example.test/base/path"
    const warnMock = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: "healthy" })
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(screen.getByText("App ready")).toBeInTheDocument()
    })

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/health",
      expect.objectContaining({ method: "GET" })
    )
    expect(warnMock).toHaveBeenCalledWith(
      "Ignoring unsupported tldw server URL protocol for readiness health check.",
      { protocol: "ftp:" }
    )
    expect(warnMock.mock.calls.flat().map(String).join(" ")).not.toContain(
      "secret"
    )
  })

  it("restarts readiness checks when the configured server URL changes", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    connectionStoreMock.store.state.serverUrl = "http://old.example:8000"
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ status: "unavailable" })
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    const { rerender } = render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://old.example:8000/api/v1/health",
        expect.objectContaining({ method: "GET" })
      )
    })

    connectionStoreMock.store.state.serverUrl = "http://new.example:9000"
    rerender(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://new.example:9000/api/v1/health",
        expect.objectContaining({ method: "GET" })
      )
    })
  })

  it("falls back before config hydration and restarts after config is available", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ status: "unavailable" })
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    const { rerender } = render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/health",
        expect.objectContaining({ method: "GET" })
      )
    })

    connectionStoreMock.store.state.serverUrl = "http://configured.local:8123"
    rerender(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://configured.local:8123/api/v1/health",
        expect.objectContaining({ method: "GET" })
      )
    })
  })

  it("accepts degraded health as enterable for partial-content responses when allowed", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 206,
      json: async () => ({ status: "degraded" })
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate allowDegraded>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(screen.getByText("App ready")).toBeInTheDocument()
    })

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("publishes bounded readiness diagnostics for degraded health", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 206,
      json: async () => ({
        status: "degraded",
        checks: { mcp: { status: "degraded" } }
      })
    } as Response)

    const readinessEvents: Array<Record<string, unknown>> = []
    const handleReadinessState = (event: Event) => {
      readinessEvents.push(
        (event as CustomEvent<Record<string, unknown>>).detail
      )
    }
    window.addEventListener(
      "tldw:server-readiness-state",
      handleReadinessState
    )

    try {
      const { ServerReadinessGate } = await import("../ServerReadinessGate")

      render(
        <ServerReadinessGate allowDegraded>
          <div>App ready</div>
        </ServerReadinessGate>
      )

      await waitFor(() => {
        expect(screen.getByText("App ready")).toBeInTheDocument()
      })

      await waitFor(() => {
        expect(readinessEvents.at(-1)).toEqual(
          expect.objectContaining({
            state: "degraded",
            healthUrl: "http://127.0.0.1:8000/api/v1/health",
            httpStatus: 206,
            healthStatus: "degraded",
            degradedChecks: ["mcp"]
          })
        )
      })
      expect(
        (window as unknown as { __tldwServerReadinessState?: unknown })
          .__tldwServerReadinessState
      ).toEqual(readinessEvents.at(-1))
    } finally {
      window.removeEventListener(
        "tldw:server-readiness-state",
        handleReadinessState
      )
    }
  })

  it("accepts ok and allowed degraded status envelopes from normal successful responses", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ status: "ok" })
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ status: "degraded" })
      } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    const { unmount } = render(
      <ServerReadinessGate>
        <div>Ok ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(screen.getByText("Ok ready")).toBeInTheDocument()
    })
    unmount()

    render(
      <ServerReadinessGate allowDegraded>
        <div>Degraded ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(screen.getByText("Degraded ready")).toBeInTheDocument()
    })

    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("keeps retrying instead of entering the app for explicitly unhealthy responses", async () => {
    vi.useFakeTimers()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")

    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ status: "unhealthy" })
    } as Response)

    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    expectReadinessStatus()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_000)
    })

    expect(screen.queryByText("App ready")).toBeNull()
    expectReadinessStatus()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("keeps malformed health responses behind the readiness screen", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => {
        throw new Error("invalid json")
      }
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expectReadinessStatus()
    })
    expect(screen.queryByText("App ready")).toBeNull()
  })

  it("reports non-enterable non-json health responses by status", async () => {
    vi.useFakeTimers()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => {
        throw new Error("Unexpected token <")
      }
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await act(async () => {
      await vi.advanceTimersByTimeAsync(16_000)
    })

    expect(screen.queryByText("App ready")).not.toBeInTheDocument()
    await act(async () => {
      await vi.runOnlyPendingTimersAsync()
    })
    expect(
      (window as unknown as { __tldwServerReadinessState?: unknown })
        .__tldwServerReadinessState
    ).toEqual(
      expect.objectContaining({
        state: "blocked",
        httpStatus: 503
      })
    )
    expect(
      (
        window as unknown as {
          __tldwServerReadinessState?: { errorMessage?: string }
        }
      )
        .__tldwServerReadinessState?.errorMessage
    ).toBeUndefined()
  })

  it("preserves degraded checks when timeout blocks an unaccepted degraded response", async () => {
    vi.useFakeTimers()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      status: 206,
      json: async () => ({
        status: "degraded",
        checks: { mcp: { status: "degraded" } }
      })
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await act(async () => {
      await vi.advanceTimersByTimeAsync(16_000)
    })

    expect(screen.queryByText("App ready")).not.toBeInTheDocument()
    await act(async () => {
      await vi.runOnlyPendingTimersAsync()
    })
    expect(
      (window as unknown as { __tldwServerReadinessState?: unknown })
        .__tldwServerReadinessState
    ).toEqual(
      expect.objectContaining({
        state: "blocked",
        healthStatus: "degraded",
        degradedChecks: ["mcp"]
      })
    )
  })

  it("restarts readiness checks when leaving a bypass route after timing out", async () => {
    vi.useFakeTimers()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")

    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false
    } as Response)

    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    const { rerender } = render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    expectReadinessStatus()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(16_000)
    })

    expect(screen.queryByText("App ready")).not.toBeInTheDocument()

    await act(async () => {
      rerender(
        <ServerReadinessGate bypass>
          <div>Settings ready</div>
        </ServerReadinessGate>
      )
    })

    expect(screen.getByText("Settings ready")).toBeInTheDocument()

    await act(async () => {
      rerender(
        <ServerReadinessGate bypass={false}>
          <div>App ready</div>
        </ServerReadinessGate>
      )
    })

    expectReadinessStatus()
    expect(screen.queryByText("App ready")).toBeNull()
  })

  it("shows actionable recovery when health checks fail until timeout", async () => {
    vi.useFakeTimers()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")

    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false
    } as Response)

    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <main data-testid="knowledge-main-region" />
      </ServerReadinessGate>
    )

    expectReadinessStatus()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(16_000)
    })

    expect(screen.queryByTestId("knowledge-main-region")).not.toBeInTheDocument()
    expect(screen.queryByTestId("server-readiness-route-content")).not.toBeInTheDocument()
    expect(screen.getAllByRole("main")).toHaveLength(1)
    expect(
      screen.getByRole("heading", { name: /Backend readiness check failed/i })
    ).toBeInTheDocument()
    expect(screen.getByText("http://127.0.0.1:8000/api/v1/health")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Health & diagnostics" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Server settings" })).toBeInTheDocument()
  })

  it("shows actionable recovery when the health request stalls until timeout", async () => {
    vi.useFakeTimers()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")

    vi.spyOn(globalThis, "fetch").mockImplementation(
      () => new Promise<Response>(() => undefined)
    )

    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <main data-testid="knowledge-main-region" />
      </ServerReadinessGate>
    )

    expectReadinessStatus()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(16_000)
    })

    expect(screen.queryByTestId("knowledge-main-region")).not.toBeInTheDocument()
    expect(screen.queryByTestId("server-readiness-route-content")).not.toBeInTheDocument()
    expect(screen.getAllByRole("main")).toHaveLength(1)
    expect(
      screen.getByRole("heading", { name: /Backend readiness check failed/i })
    ).toBeInTheDocument()
    expect(screen.getByText("http://127.0.0.1:8000/api/v1/health")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Health & diagnostics" })).toBeInTheDocument()
  })

  it("bypasses health checks when bypass is enabled", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate bypass>
        <div>Settings ready</div>
      </ServerReadinessGate>
    )

    expect(screen.getByText("Settings ready")).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("bypasses health checks when the offline E2E flag is enabled", async () => {
    localStorage.setItem("__tldw_allow_offline", "true")
    const fetchMock = vi.spyOn(globalThis, "fetch")
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <div>App ready offline</div>
      </ServerReadinessGate>
    )

    expect(screen.getByText("App ready offline")).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
