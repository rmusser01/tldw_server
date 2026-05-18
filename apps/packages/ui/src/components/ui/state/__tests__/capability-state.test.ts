import { describe, expect, it, vi } from "vitest"

import {
  buildCapabilityDiagnostics,
  buildCapabilityState,
  classifyCapabilityError,
  messageFromError,
  statusFromError
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
    expect(classifyCapabilityError({ status: 500 })).toBe("error")
    expect(classifyCapabilityError({ response: { status: 503 } })).toBe("error")
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

  it("exports shared error parsing helpers", () => {
    expect(statusFromError({ response: { status: 503 } })).toBe(503)
    expect(messageFromError({ message: "service unavailable" })).toBe(
      "service unavailable"
    )
    expect(messageFromError("plain failure")).toBe("plain failure")
  })

  it("uses request-error copy for server failures", () => {
    const descriptor = buildCapabilityState({
      kind: classifyCapabilityError({ status: 500 }),
      featureName: "Workspace integrations",
      capabilityName: "workspace integrations",
      method: "GET",
      endpoint: "/api/v1/integrations/workspace",
      status: 500,
      rawMessage: "Internal Server Error"
    })

    expect(descriptor.state).toBe("error")
    expect(descriptor.title).toBe("Workspace integrations could not load")
    expect(descriptor.message).toBe(
      "The request failed before this feature could load. Check diagnostics or try again."
    )
  })

  it("allows callers to provide pretranslated primary copy", () => {
    const descriptor = buildCapabilityState({
      kind: "error",
      featureName: "Workspace integrations",
      title: "Integrations failed to load",
      message: "Retry after checking the server."
    })

    expect(descriptor.title).toBe("Integrations failed to load")
    expect(descriptor.message).toBe("Retry after checking the server.")
  })

  it("redacts credentials, paths, queries, and fragments from diagnostic server URLs", () => {
    const diagnostics = buildCapabilityDiagnostics({
      serverUrl: "https://user:secret@example.test:8443/base?api_key=secret#token"
    })

    expect(diagnostics).toEqual([
      {
        label: "Server URL",
        value: "https://example.test:8443",
        code: true
      }
    ])
    expect(diagnostics?.[0]?.value).not.toContain("user")
    expect(diagnostics?.[0]?.value).not.toContain("secret")
    expect(diagnostics?.[0]?.value).not.toContain("api_key")
  })

  it("redacts paths and credentials from no-scheme diagnostic server URLs", () => {
    expect(buildCapabilityDiagnostics({
      serverUrl: "user:secret@example.test:8000/base?api_key=secret#token"
    })).toEqual([
      {
        label: "Server URL",
        value: "example.test:8000",
        code: true
      }
    ])
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
