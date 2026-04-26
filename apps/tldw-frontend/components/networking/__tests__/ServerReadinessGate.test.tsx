import { act, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

describe("ServerReadinessGate", () => {
  afterEach(() => {
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

    expect(screen.getByText("Waiting for server...")).toBeInTheDocument()

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

    expect(screen.getByText("Waiting for server...")).toBeInTheDocument()
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
})
