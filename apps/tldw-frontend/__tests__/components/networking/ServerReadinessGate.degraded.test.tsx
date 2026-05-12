import React from "react"
import { render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { ServerReadinessGate } from "@web/components/networking/ServerReadinessGate"

const renderGate = (allowDegraded = false) =>
  render(
    <ServerReadinessGate allowDegraded={allowDegraded}>
      <div data-testid="page-content">Chat page</div>
    </ServerReadinessGate>
  )

const mockHealth = (body: unknown, ok = true) => {
  vi.mocked(fetch).mockResolvedValueOnce({
    ok,
    status: ok ? 200 : 503,
    json: async () => body
  } as Response)
}

describe("ServerReadinessGate degraded health", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("renders chat content immediately with a scoped warning when degraded health is allowed", async () => {
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

    renderGate(true)

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
  })

  it("keeps degraded health behind the readiness screen when degraded health is not allowed", async () => {
    mockHealth({
      status: "degraded",
      checks: { chacha_notes: { status: "degraded" } }
    })

    renderGate(false)

    expect(
      await screen.findByText("Retrying server readiness")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("page-content")).toBeNull()
  })

  it("keeps unhealthy health behind the readiness screen even when degraded health is allowed", async () => {
    mockHealth({ status: "unhealthy" })

    renderGate(true)

    expect(
      await screen.findByText("Retrying server readiness")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("page-content")).toBeNull()
  })

  it("keeps failed health checks behind the readiness screen", async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error("network down"))

    renderGate(true)

    expect(
      await screen.findByText("Retrying server readiness")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("page-content")).toBeNull()
  })
})
