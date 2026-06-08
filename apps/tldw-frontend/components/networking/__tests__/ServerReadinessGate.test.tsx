import { act, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  connectionState: {
    serverUrl: null as string | null,
    lastConfigUpdatedAt: null as number | null
  }
}))

vi.mock("@/store/connection", () => ({
  useConnectionStore: (selector?: (store: { state: typeof mocks.connectionState }) => unknown) => {
    const store = { state: mocks.connectionState }
    return typeof selector === "function" ? selector(store) : store
  }
}))

describe("ServerReadinessGate", () => {
  const expectReadinessStatus = () => {
    const status = screen.getByRole("status")
    expect(status).toHaveTextContent(/Checking server readiness|Retrying server readiness/)
    expect(status).toHaveTextContent(/Loading|Retrying/)
  }

  beforeEach(() => {
    vi.resetModules()
    mocks.connectionState = {
      serverUrl: null,
      lastConfigUpdatedAt: null
    }
  })

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

  it("uses the saved server URL for readiness even when the page origin differs", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://localhost:18001")
    mocks.connectionState = {
      serverUrl: "http://127.0.0.1:8000",
      lastConfigUpdatedAt: 1700000000000
    }
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

  it("shows one blocking recovery state without exposing route content", async () => {
    vi.useFakeTimers()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")

    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false
    } as Response)

    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate>
        <main data-testid="route-main" aria-label="Route content">
          Route content
        </main>
      </ServerReadinessGate>
    )

    await act(async () => {
      await vi.advanceTimersByTimeAsync(16_000)
    })

    expect(screen.getByTestId("server-readiness-recovery")).toBeInTheDocument()
    expect(screen.queryByTestId("route-main")).not.toBeInTheDocument()
    expect(document.querySelectorAll("main")).toHaveLength(1)
  })

  it("restarts readiness checks when the configured server URL changes", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://localhost:18001")
    mocks.connectionState = {
      serverUrl: "http://127.0.0.1:8000",
      lastConfigUpdatedAt: 1700000000000
    }
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false
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

    mocks.connectionState = {
      serverUrl: "http://192.168.1.50:8000",
      lastConfigUpdatedAt: 1700000001000
    }

    await act(async () => {
      rerender(
        <ServerReadinessGate>
          <div>App ready</div>
        </ServerReadinessGate>
      )
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://192.168.1.50:8000/api/v1/health",
        expect.objectContaining({ method: "GET" })
      )
    })
  })

  it("falls back before config hydration and restarts after config is available", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://localhost:18001")
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false
    } as Response)
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    const { rerender } = render(
      <ServerReadinessGate>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://localhost:18001/api/v1/health",
        expect.objectContaining({ method: "GET" })
      )
    })

    mocks.connectionState = {
      serverUrl: "http://127.0.0.1:8000",
      lastConfigUpdatedAt: 1700000002000
    }

    await act(async () => {
      rerender(
        <ServerReadinessGate>
          <div>App ready</div>
        </ServerReadinessGate>
      )
    })

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/health",
        expect.objectContaining({ method: "GET" })
      )
    })
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

    expect(screen.getByTestId("server-readiness-recovery")).toBeInTheDocument()
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
    expect(
      screen.getByRole("heading", { name: /Backend readiness check failed/i })
    ).toBeInTheDocument()
    expect(screen.getByText("http://127.0.0.1:8000/api/v1/health")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Health & diagnostics" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Server settings" })).toBeInTheDocument()
    expect(document.querySelectorAll("main")).toHaveLength(1)
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
    expect(
      screen.getByRole("heading", { name: /Backend readiness check failed/i })
    ).toBeInTheDocument()
    expect(screen.getByText("http://127.0.0.1:8000/api/v1/health")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Health & diagnostics" })).toBeInTheDocument()
    expect(document.querySelectorAll("main")).toHaveLength(1)
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
