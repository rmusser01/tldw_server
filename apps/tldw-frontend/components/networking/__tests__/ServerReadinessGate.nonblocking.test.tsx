import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
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

const okHealth = () =>
  ({ ok: true, status: 200, json: async () => ({ status: "ok" }) }) as Response
const unavailableHealth = () =>
  ({ ok: false, status: 503, json: async () => ({ status: "unavailable" }) }) as
    Response

describe("ServerReadinessGate non-blocking sessions (W1)", () => {
  beforeEach(() => {
    connectionStoreMock.store.state.serverUrl = ""
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
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

  it("renders children immediately while the first health probe is in flight", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(() => new Promise<Response>(() => undefined))
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate nonBlocking>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    expect(screen.getByText("App ready")).toBeInTheDocument()
    expect(screen.queryByTestId("server-reconnect-banner")).toBeNull()
    expect(screen.queryByTestId("server-readiness-recovery")).toBeNull()
    expect(
      screen.queryByText(/Checking server readiness/i)
    ).toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("keeps children mounted with a reconnect banner while health fails, even after the deadline", async () => {
    vi.useFakeTimers()
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => unavailableHealth())
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate nonBlocking>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(screen.getByText("App ready")).toBeInTheDocument()
    expect(screen.getByTestId("server-reconnect-banner")).toHaveTextContent(
      /Reconnecting to the tldw server/i
    )
    expect(screen.queryByTestId("server-readiness-recovery")).toBeNull()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(16_000)
    })

    expect(screen.getByText("App ready")).toBeInTheDocument()
    expect(screen.getByTestId("server-reconnect-banner")).toHaveTextContent(
      /could not reach the tldw server/i
    )
    expect(
      screen.getByRole("button", { name: /retry connection/i })
    ).toBeInTheDocument()
    expect(screen.queryByTestId("server-readiness-recovery")).toBeNull()
    // Retries ran until the deadline, then stopped.
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2)
  })

  it("clears the reconnect banner when a scheduled retry succeeds", async () => {
    vi.useFakeTimers()
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(unavailableHealth())
      .mockResolvedValue(okHealth())
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate nonBlocking>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(screen.getByTestId("server-reconnect-banner")).toBeInTheDocument()

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_100)
    })

    expect(screen.getByText("App ready")).toBeInTheDocument()
    expect(screen.queryByTestId("server-reconnect-banner")).toBeNull()
    expect(screen.queryByTestId("server-readiness-nonblocking-shell")).toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("retries automatically when connectivity returns after the deadline", async () => {
    vi.useFakeTimers()
    let healthy = false
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      healthy ? okHealth() : unavailableHealth()
    )
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate nonBlocking>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await act(async () => {
      await vi.advanceTimersByTimeAsync(16_000)
    })
    expect(screen.getByTestId("server-reconnect-banner")).toHaveTextContent(
      /could not reach the tldw server/i
    )

    healthy = true
    await act(async () => {
      window.dispatchEvent(new Event("online"))
      await vi.advanceTimersByTimeAsync(0)
    })

    expect(screen.getByText("App ready")).toBeInTheDocument()
    expect(screen.queryByTestId("server-reconnect-banner")).toBeNull()
  })

  it("shows the degraded banner for degraded health without allowDegraded when non-blocking", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      ({ ok: true, status: 200, json: async () => ({ status: "degraded" }) }) as
        Response
    )
    const { ServerReadinessGate } = await import("../ServerReadinessGate")

    render(
      <ServerReadinessGate nonBlocking>
        <div>App ready</div>
      </ServerReadinessGate>
    )

    expect(await screen.findByText("App ready")).toBeInTheDocument()
    // Children render immediately (checking); the degraded shell appears once
    // the probe settles on "degraded".
    expect(
      await screen.findByTestId("server-readiness-degraded-shell")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("server-reconnect-banner")).toBeNull()
  })

  it.each(["unavailable", "degraded"] as const)(
    "retains the checking draft and mounted instance when health becomes %s",
    async (status) => {
      vi.useFakeTimers()
      let resolveHealth!: (response: Response) => void
      vi.spyOn(globalThis, "fetch").mockImplementation(
        () => new Promise<Response>((resolve) => { resolveHealth = resolve })
      )
      const { ServerReadinessGate } = await import("../ServerReadinessGate")
      let mounts = 0
      let unmounts = 0
      const Draft = () => {
        const [draft, setDraft] = React.useState("")
        React.useEffect(() => {
          mounts += 1
          return () => { unmounts += 1 }
        }, [])
        return <input aria-label="Draft" value={draft} onChange={event => setDraft(event.target.value)} />
      }
      const { unmount } = render(<ServerReadinessGate nonBlocking><Draft /></ServerReadinessGate>)
      const input = screen.getByRole("textbox", { name: "Draft" })
      fireEvent.change(input, { target: { value: "Unsaved research question" } })

      await act(async () => {
        resolveHealth({ status: status === "degraded" ? 200 : 503,
          json: async () => ({ status }) } as Response)
        await vi.advanceTimersByTimeAsync(0)
      })

      expect(screen.getByRole("textbox", { name: "Draft" })).toHaveValue("Unsaved research question")
      expect(screen.getByRole("textbox", { name: "Draft" })).toBe(input)
      expect({ mounts, unmounts }).toEqual({ mounts: 1, unmounts: 0 })
      unmount()
      expect({ mounts, unmounts }).toEqual({ mounts: 1, unmounts: 1 })
    }
  )

  it("retains a local draft across timeout, manual retry, degradation, recovery and bypass", async () => {
    vi.useFakeTimers()
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(unavailableHealth())
    const { ServerReadinessGate } = await import("../ServerReadinessGate")
    let mounts = 0
    const Draft = () => {
      const [draft, setDraft] = React.useState("")
      React.useEffect(() => { mounts += 1 }, [])
      return <input aria-label="Draft" value={draft} onChange={event => setDraft(event.target.value)} />
    }
    const tree = (bypass = false, configuredServerUrl = "http://first.test") => (
      <ServerReadinessGate nonBlocking bypass={bypass} configuredServerUrl={configuredServerUrl}>
        <Draft />
      </ServerReadinessGate>
    )
    const { rerender } = render(tree())
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    const input = screen.getByRole("textbox", { name: "Draft" })
    fireEvent.change(input, { target: { value: "Keep this draft" } })
    const expectDraft = () => {
      expect(screen.getByRole("textbox", { name: "Draft" })).toBe(input)
      expect(input).toHaveValue("Keep this draft")
      expect(mounts).toBe(1)
    }

    await act(async () => { await vi.advanceTimersByTimeAsync(15_000) })
    expect(screen.getByTestId("server-reconnect-banner")).toHaveTextContent(/could not reach/i)
    expectDraft()

    let resolveRetry!: (response: Response) => void
    fetchMock.mockImplementationOnce(() => new Promise<Response>(resolve => { resolveRetry = resolve }))
    fireEvent.click(screen.getByRole("button", { name: /retry connection/i }))
    expect(screen.queryByTestId("server-reconnect-banner")).toBeNull()
    expectDraft()
    await act(async () => {
      resolveRetry({ status: 200, json: async () => ({ status: "degraded", checks: { mcp: { status: "degraded" } } }) } as Response)
      await vi.advanceTimersByTimeAsync(0)
    })
    expect(screen.getByTestId("server-readiness-degraded-shell")).toBeInTheDocument()
    expectDraft()

    rerender(tree(true))
    expect(screen.queryByTestId("server-readiness-degraded-shell")).toBeNull()
    expectDraft()
    fetchMock.mockResolvedValue(okHealth())
    rerender(tree(false, "http://recovered.test"))
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(screen.queryByTestId("server-reconnect-banner")).toBeNull()
    expect(screen.queryByTestId("server-readiness-degraded-shell")).toBeNull()
    expectDraft()
    rerender(tree(true, "http://recovered.test"))
    expectDraft()
  })
})

describe("ServerReadinessGate warmed probe reuse (W1)", () => {
  beforeEach(() => {
    connectionStoreMock.store.state.serverUrl = ""
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
    vi.unstubAllEnvs()
    vi.resetModules()
  })

  it("reuses a settled warmed probe instead of refetching health", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(okHealth())
    const { ServerReadinessGate, warmServerReadinessHealth } = await import(
      "../ServerReadinessGate"
    )

    warmServerReadinessHealth("http://127.0.0.1:8000")
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)

    render(
      <ServerReadinessGate configuredServerUrl="http://127.0.0.1:8000">
        <div>App ready</div>
      </ServerReadinessGate>
    )

    await waitFor(() => {
      expect(screen.getByText("App ready")).toBeInTheDocument()
    })
    // The gate consumed the warmed result: still exactly one health request.
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("joins a warmed in-flight probe instead of issuing a parallel request", async () => {
    let resolveHealth!: (value: Response) => void
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(
        () => new Promise<Response>((resolve) => { resolveHealth = resolve })
      )
    const { ServerReadinessGate, warmServerReadinessHealth } = await import(
      "../ServerReadinessGate"
    )

    warmServerReadinessHealth("http://127.0.0.1:8000")
    expect(fetchMock).toHaveBeenCalledTimes(1)

    render(
      <ServerReadinessGate configuredServerUrl="http://127.0.0.1:8000">
        <div>App ready</div>
      </ServerReadinessGate>
    )

    // Gate joined the in-flight probe: no second request while it is pending.
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("App ready")).toBeNull()

    await act(async () => {
      resolveHealth(okHealth())
      await Promise.resolve()
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(screen.getByText("App ready")).toBeInTheDocument()
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
