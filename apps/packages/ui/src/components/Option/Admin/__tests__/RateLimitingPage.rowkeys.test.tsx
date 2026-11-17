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

import { clearCapabilityProbeCacheForTests } from "@/services/tldw/capability-probe"
import RateLimitingPage from "../RateLimitingPage"

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const tableRowKeys = () =>
  Array.from(document.querySelectorAll("tr.ant-table-row")).map((row) =>
    row.getAttribute("data-row-key")
  )

describe("RateLimitingPage unprotected-route row keys (C-S5)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    clearCapabilityProbeCacheForTests()

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

  it("keys unprotected routes by the route/path value, not the list index", async () => {
    mocks.getGovernorCoverage.mockResolvedValueOnce({
      total_routes: 3,
      protected_count: 0,
      unprotected_count: 3,
      coverage_pct: 0,
      protected_routes: [],
      unprotected_routes: [
        { method: "GET", path: "/api/v1/open-a" },
        { method: "POST", path: "/api/v1/open-b" },
        "/api/v1/open-c"
      ]
    })

    render(<RateLimitingPage />)

    await screen.findByText("/api/v1/open-a")
    expect(tableRowKeys()).toEqual([
      "/api/v1/open-a",
      "/api/v1/open-b",
      "/api/v1/open-c"
    ])
  })
})
