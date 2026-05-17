import { describe, expect, it, vi } from "vitest"

import {
  buildCapabilityDiagnostics,
  buildCapabilityState,
  classifyCapabilityError
} from "../capability-state"

describe("capability state mapping", () => {
  it("maps endpoint 404s to unavailable state without using raw copy as primary text", () => {
    const descriptor = buildCapabilityState({
      kind: "unavailable",
      featureName: "Sources",
      capabilityName: "ingestion sources",
      method: "GET",
      endpoint: "/api/v1/sources",
      status: 404,
      rawMessage: "Not Found (GET /api/v1/sources)",
      primaryAction: { label: "Check server setup", onClick: vi.fn() }
    })

    expect(descriptor.state).toBe("unavailable")
    expect(descriptor.title).toBe("Sources are unavailable")
    expect(descriptor.message).toBe(
      "This server does not expose the ingestion sources capability."
    )
    expect(descriptor.message).not.toContain("/api/v1/sources")
    expect(descriptor.primaryAction?.label).toBe("Check server setup")
    expect(descriptor.diagnostics).toEqual([
      { label: "Method", value: "GET" },
      { label: "Endpoint", value: "/api/v1/sources", code: true },
      { label: "Status", value: "404" },
      { label: "Raw message", value: "Not Found (GET /api/v1/sources)" }
    ])
  })

  it("classifies common capability failures into design-system states", () => {
    expect(classifyCapabilityError({ status: 401 })).toBe("auth_required")
    expect(classifyCapabilityError({ status: 403 })).toBe("permission_denied")
    expect(classifyCapabilityError({ status: 404 })).toBe("unavailable")
    expect(classifyCapabilityError(new TypeError("fetch failed"))).toBe(
      "network_failure"
    )
    expect(classifyCapabilityError(new Error("connection refused"))).toBe(
      "network_failure"
    )
    expect(classifyCapabilityError(new Error("provider not configured"))).toBe(
      "not_configured"
    )
  })

  it("keeps optional diagnostics out of the descriptor when no details exist", () => {
    expect(buildCapabilityDiagnostics({})).toBeUndefined()
  })

  it("uses degraded language for partial data", () => {
    const descriptor = buildCapabilityState({
      kind: "degraded",
      featureName: "Scheduled tasks",
      rawMessage: "Watchlist jobs unavailable",
      primaryAction: { label: "Refresh", onClick: vi.fn() }
    })

    expect(descriptor.state).toBe("degraded")
    expect(descriptor.title).toBe("Scheduled tasks are partially available")
    expect(descriptor.message).toBe(
      "Some data loaded, but part of this feature is limited."
    )
    expect(descriptor.diagnostics).toEqual([
      { label: "Raw message", value: "Watchlist jobs unavailable" }
    ])
  })
})
