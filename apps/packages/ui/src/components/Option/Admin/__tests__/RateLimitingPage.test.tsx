// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  useCanonicalConnectionConfig: vi.fn(),
  getGovernorPolicy: vi.fn(),
  getGovernorCoverage: vi.fn(),
  listAdminRateLimits: vi.fn()
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: (...args: unknown[]) => mocks.useCanonicalConnectionConfig(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getGovernorPolicy: (...args: unknown[]) => mocks.getGovernorPolicy(...args),
    getGovernorCoverage: (...args: unknown[]) => mocks.getGovernorCoverage(...args),
    listAdminRateLimits: (...args: unknown[]) => mocks.listAdminRateLimits(...args)
  }
}))

import RateLimitingPage from "../RateLimitingPage"

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const expectDesignSystemAlertForText = async (text: string) => {
  const title = await screen.findByText(text)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  return alert as HTMLElement
}

describe("RateLimitingPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }

    mocks.useCanonicalConnectionConfig.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      },
      loading: false
    })
    mocks.getGovernorPolicy.mockResolvedValue({
      status: "ok",
      store: "file",
      version: 1,
      policies_count: 0
    })
    mocks.getGovernorCoverage.mockResolvedValue({
      protected: [],
      unprotected: [],
      coverage_pct: 100
    })
    mocks.listAdminRateLimits.mockResolvedValue([])
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/admin/rate-limits": {}
        }
      })
    })
  })

  it("renders forbidden guard feedback through the design-system Alert primitive", async () => {
    mocks.getGovernorPolicy.mockRejectedValueOnce({ status: 403 })

    render(<RateLimitingPage />)

    const alert = await expectDesignSystemAlertForText("Access Denied")
    expect(alert).toHaveAttribute("role", "alert")
    expect(alert).toHaveTextContent(
      "You don't have permission to access rate limiting administration."
    )
  })

  it("renders missing-endpoint guard feedback through the design-system Alert primitive", async () => {
    mocks.getGovernorPolicy.mockRejectedValueOnce({ status: 404 })

    render(<RateLimitingPage />)

    const alert = await expectDesignSystemAlertForText("Not Available")
    expect(alert).toHaveAttribute("role", "alert")
    expect(alert).toHaveTextContent(
      "Rate limiting administration is not available on this server."
    )
  })

  it("renders empty policy and coverage feedback through the design-system Alert primitive", async () => {
    mocks.getGovernorPolicy.mockResolvedValueOnce(null)
    mocks.getGovernorCoverage.mockResolvedValueOnce(null)

    render(<RateLimitingPage />)

    const policyAlert = await expectDesignSystemAlertForText("No policy data loaded yet.")
    expect(policyAlert).toHaveAttribute("role", "status")

    const coverageAlert = await expectDesignSystemAlertForText("No coverage data loaded yet.")
    expect(coverageAlert).toHaveAttribute("role", "status")
  })

  it("shows an unsupported-state message without calling admin rate-limits when the route is absent", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {}
      })
    })

    render(<RateLimitingPage />)

    expect(
      await screen.findByText("Rate limits listing endpoint is not available on this server.")
    ).toBeInTheDocument()
    const alert = await expectDesignSystemAlertForText(
      "Rate limits listing endpoint is not available on this server."
    )
    expect(alert).toHaveAttribute("role", "status")
    expect(mocks.listAdminRateLimits).not.toHaveBeenCalled()
  })

  it("reads the diag coverage payload's real field names (protected_routes/counts)", async () => {
    // Regression: the page read coverage.protected/unprotected while the diag
    // endpoint returns protected_routes/unprotected_routes (+ counts), so it
    // rendered "78.9%" above "Protected: 0 routes | Unprotected: 0 routes".
    mocks.getGovernorCoverage.mockResolvedValueOnce({
      total_routes: 10,
      protected_count: 8,
      unprotected_count: 2,
      coverage_pct: 80,
      protected_routes: Array.from({ length: 8 }, (_, i) => ({
        method: "GET",
        path: `/api/v1/protected-${i}`
      })),
      unprotected_routes: [
        { method: "GET", path: "/api/v1/open-a" },
        { method: "POST", path: "/api/v1/open-b" }
      ]
    })

    render(<RateLimitingPage />)

    await screen.findByText("/api/v1/open-a")
    const summary = screen.getByText("Protected:").closest("div")
    expect(summary?.textContent).toContain("Protected: 8 routes")
    expect(summary?.textContent).toContain("Unprotected: 2 routes")
    expect(screen.getByText("/api/v1/open-b")).toBeInTheDocument()
  })

  it("explains that empty per-user overrides still leave baseline limits active", async () => {
    render(<RateLimitingPage />)

    expect(
      await screen.findByText(
        "No per-user overrides configured. The governor policy's baseline limits still apply."
      )
    ).toBeInTheDocument()
  })
})
