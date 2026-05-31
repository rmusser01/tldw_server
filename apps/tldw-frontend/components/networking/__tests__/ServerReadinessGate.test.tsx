import { act, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

describe("ServerReadinessGate", () => {
  const expectReadinessStatus = () => {
    const status = screen.getByRole("status")
    expect(status).toHaveTextContent(/Checking server readiness|Retrying server readiness/)
    expect(status).toHaveTextContent(/Loading|Retrying/)
  }

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

    expect(screen.getByText("App ready")).toBeInTheDocument()
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

    expect(screen.getByText("App ready")).toBeInTheDocument()
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

    expect(screen.getByText("App ready")).toBeInTheDocument()

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
