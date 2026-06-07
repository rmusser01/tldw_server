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
    vi.restoreAllMocks()
    vi.useRealTimers()
    vi.unstubAllEnvs()
  })

  it("accepts the backend healthy status envelope", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
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

    expect(screen.getByTestId("knowledge-main-region")).toBeInTheDocument()
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

    expect(screen.getByTestId("knowledge-main-region")).toBeInTheDocument()
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
