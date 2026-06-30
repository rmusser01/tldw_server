import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const renderGate = async (allowDegraded = false) => {
  const { ServerReadinessGate } = await import(
    "@web/components/networking/ServerReadinessGate"
  )
  return render(
    <ServerReadinessGate allowDegraded={allowDegraded}>
      <div data-testid="page-content">Chat page</div>
    </ServerReadinessGate>
  )
}

const mockHealth = (body: unknown, ok = true) => {
  vi.mocked(fetch).mockResolvedValueOnce({
    ok,
    status: ok ? 200 : 503,
    json: async () => body
  } as Response)
}

const ReadinessEventChild = ({
  onReadiness
}: {
  onReadiness: (detail: unknown) => void
}) => {
  React.useEffect(() => {
    const handleReadiness = (event: Event) => {
      onReadiness(
        (event as CustomEvent<{ state?: string; degradedChecks?: string[] }>)
          .detail
      )
    }
    window.addEventListener("tldw:server-readiness-state", handleReadiness)
    return () => {
      window.removeEventListener(
        "tldw:server-readiness-state",
        handleReadiness
      )
    }
  }, [onReadiness])

  return <div data-testid="readiness-event-child">Mounted child</div>
}

describe("ServerReadinessGate degraded health", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubEnv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")
    vi.stubGlobal("fetch", vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
    vi.restoreAllMocks()
  })

  it("renders chat content immediately with a scoped warning when degraded health is allowed", async () => {
    const readinessListener = vi.fn()
    window.addEventListener("tldw:server-readiness-state", readinessListener)
    try {
      mockHealth({
        status: "degraded",
        checks: {
          database: { status: "healthy" },
          chacha_notes: {
            status: "degraded",
            last_error: "sqlite corruption"
          }
        }
      })

      await renderGate(true)

      expect(await screen.findByTestId("page-content")).toBeInTheDocument()
      expect(screen.getByTestId("server-readiness-degraded-shell")).toHaveClass(
        "server-readiness-degraded-shell"
      )
      expect(screen.getByRole("status")).toHaveTextContent(
        "Server partially degraded"
      )
      expect(screen.getByRole("status")).toHaveTextContent("chacha_notes")
      expect(screen.getByRole("link", { name: /open health/i })).toHaveAttribute(
        "href",
        "/settings/health"
      )
      expect(readinessListener).toHaveBeenCalledWith(
        expect.objectContaining({
          detail: expect.objectContaining({
            state: "degraded",
            degradedChecks: ["chacha_notes"]
          })
        })
      )
    } finally {
      window.removeEventListener(
        "tldw:server-readiness-state",
        readinessListener
      )
    }
  })

  it("emits degraded readiness after allowed children mount", async () => {
    const childReadinessListener = vi.fn()
    const { ServerReadinessGate } = await import(
      "@web/components/networking/ServerReadinessGate"
    )
    mockHealth({
      status: "degraded",
      checks: {
        chacha_notes: { status: "degraded" }
      }
    })

    render(
      <ServerReadinessGate allowDegraded>
        <ReadinessEventChild onReadiness={childReadinessListener} />
      </ServerReadinessGate>
    )

    expect(await screen.findByTestId("readiness-event-child")).toBeInTheDocument()
    await waitFor(() => {
      expect(childReadinessListener).toHaveBeenCalledWith(
        expect.objectContaining({
          state: "degraded",
          degradedChecks: ["chacha_notes"]
        })
      )
    })
  })

  it("keeps degraded health behind the readiness screen when degraded health is not allowed", async () => {
    mockHealth({
      status: "degraded",
      checks: { chacha_notes: { status: "degraded" } }
    })

    await renderGate(false)

    expect(
      await screen.findByText("Retrying server readiness")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("page-content")).toBeNull()
  })

  it("keeps unhealthy health behind the readiness screen even when degraded health is allowed", async () => {
    mockHealth({ status: "unhealthy" })

    await renderGate(true)

    expect(
      await screen.findByText("Retrying server readiness")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("page-content")).toBeNull()
  })

  it("keeps failed health checks behind the readiness screen", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error("network down"))

    await renderGate(true)

    expect(
      await screen.findByText("Retrying server readiness")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("page-content")).toBeNull()
  })
})
