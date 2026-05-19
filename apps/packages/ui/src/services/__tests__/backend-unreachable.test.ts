import { describe, expect, it } from "vitest"

import {
  classifyBackendUnreachableError,
  type BackendUnreachableClassification
} from "@/services/backend-unreachable"

describe("classifyBackendUnreachableError", () => {
  it("classifies status 0 network-style errors as backend unreachable", () => {
    const result = classifyBackendUnreachableError(
      Object.assign(
        new Error("Network request failed while fetching server metadata."),
        {
          status: 0
        }
      )
    )

    expect(result).toMatchObject({
      kind: "backend_unreachable"
    })
  })

  it("classifies NetworkError when attempting to fetch resource", () => {
    const result = classifyBackendUnreachableError(
      new Error("NetworkError when attempting to fetch resource.")
    )

    expect(result).toMatchObject({
      kind: "backend_unreachable"
    })
  })

  it("classifies Failed to fetch", () => {
    const result = classifyBackendUnreachableError(new Error("Failed to fetch"))

    expect(result).toMatchObject({
      kind: "backend_unreachable"
    })
  })

  it("does not classify feature-specific failed-to-fetch HTTP errors", () => {
    const cases = [
      new Error("Failed to fetch file report.pdf: HTTP 404"),
    ]

    for (const error of cases) {
      expect(classifyBackendUnreachableError(error)).toMatchObject({
        kind: "other"
      })
    }
  })

  it("does not classify abort-style transport errors", () => {
    const cases = [
      Object.assign(new Error("AbortError"), { status: 0, code: "REQUEST_ABORTED" }),
      Object.assign(new Error("REQUEST_ABORTED"), { status: 0 }),
      new Error("The operation was aborted.")
    ]

    for (const error of cases) {
      expect(classifyBackendUnreachableError(error)).toMatchObject({
        kind: "other"
      })
    }
  })

  it("does not classify unrelated runtime errors", () => {
    const result = classifyBackendUnreachableError(new Error("Unexpected state transition"))

    expect(result).toMatchObject({
      kind: "other"
    })
  })

  it("enriches diagnostics from a recent __tldwLastRequestError payload", () => {
    const result = classifyBackendUnreachableError(
      Object.assign(new Error("Failed to fetch"), {
        status: 0
      }),
      {
        nowMs: Date.parse("2026-03-27T12:00:10.000Z"),
        recentRequestError: {
          method: "GET",
          path: "/api/v1/health",
          status: 0,
          error: "NetworkError when attempting to fetch resource.",
          source: "direct",
          at: "2026-03-27T12:00:00.000Z"
        },
        recentRequestErrorFreshnessMs: 60_000
      }
    )

    expect(result).toMatchObject({
      kind: "backend_unreachable",
      method: "GET",
      path: "/api/v1/health",
      recentRequestError: {
        method: "GET",
        path: "/api/v1/health",
        status: 0,
        error: "NetworkError when attempting to fetch resource.",
        source: "direct"
      }
    })
  })

  // ---- Subtype classification tests ----

  describe("subtype: cors", () => {
    it("classifies 'blocked by CORS policy' as cors subtype", () => {
      const result = classifyBackendUnreachableError(
        new Error("Access to fetch at 'http://localhost:8000' has been blocked by CORS policy")
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "cors",
        title: "Cross-origin request blocked"
      })
      expect((result as BackendUnreachableClassification).fixHint).toContain("ALLOWED_ORIGINS")
    })

    it("classifies 'No Access-Control-Allow-Origin header' as cors subtype", () => {
      const result = classifyBackendUnreachableError(
        new Error("No 'Access-Control-Allow-Origin' header is present on the requested resource")
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "cors"
      })
    })

    it("classifies 'Cross-Origin Request Blocked' (Firefox) as cors subtype", () => {
      const result = classifyBackendUnreachableError(
        new Error("Cross-Origin Request Blocked: The Same Origin Policy disallows reading")
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "cors"
      })
    })

    it("includes the server URL in the cors message when provided", () => {
      const result = classifyBackendUnreachableError(
        new Error("blocked by CORS policy"),
        { serverUrl: "http://localhost:8000" }
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "cors"
      })
      expect((result as BackendUnreachableClassification).message).toContain("http://localhost:8000")
    })
  })

  describe("subtype: connection_refused", () => {
    it("classifies ECONNREFUSED as connection_refused", () => {
      const result = classifyBackendUnreachableError(
        new Error("request to http://localhost:8000/health failed, reason: connect ECONNREFUSED 127.0.0.1:8000")
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "connection_refused",
        title: "Cannot connect to the API server"
      })
      expect((result as BackendUnreachableClassification).fixHint).toContain("curl")
    })

    it("classifies net::ERR_CONNECTION_REFUSED as connection_refused", () => {
      const result = classifyBackendUnreachableError(
        new Error("net::ERR_CONNECTION_REFUSED")
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "connection_refused"
      })
    })

    it("classifies generic 'Failed to fetch' as connection_refused (most common cause)", () => {
      const result = classifyBackendUnreachableError(
        new Error("Failed to fetch")
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "connection_refused"
      })
    })
  })

  describe("subtype: auth_failed", () => {
    it("classifies HTTP 401 as auth_failed", () => {
      const result = classifyBackendUnreachableError(
        Object.assign(new Error("Unauthorized"), { status: 401 })
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "auth_failed",
        title: "Authentication failed"
      })
      expect((result as BackendUnreachableClassification).fixHint).toContain("API key")
    })
  })

  describe("subtype: forbidden", () => {
    it("classifies HTTP 403 as forbidden", () => {
      const result = classifyBackendUnreachableError(
        Object.assign(new Error("Forbidden"), { status: 403 })
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "forbidden",
        title: "Access denied"
      })
      expect((result as BackendUnreachableClassification).fixHint).toContain("permissions")
    })
  })

  describe("subtype: timeout", () => {
    it("classifies 'timeout exceeded' as timeout", () => {
      const result = classifyBackendUnreachableError(
        new Error("timeout of 30000ms exceeded")
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "timeout",
        title: "Server took too long to respond"
      })
      expect((result as BackendUnreachableClassification).fixHint).toContain("starting up")
    })

    it("classifies 'request timed out' as timeout", () => {
      const result = classifyBackendUnreachableError(
        new Error("The request timed out")
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "timeout"
      })
    })

    it("classifies net::ERR_TIMED_OUT as timeout", () => {
      const result = classifyBackendUnreachableError(
        new Error("net::ERR_TIMED_OUT")
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "timeout"
      })
    })

    it("classifies TimeoutError name as timeout", () => {
      const err = new Error("timeout")
      err.name = "TimeoutError"
      const result = classifyBackendUnreachableError(err)

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "timeout"
      })
    })
  })

  describe("subtype: server_error", () => {
    it("classifies HTTP 500 as server_error", () => {
      const result = classifyBackendUnreachableError(
        Object.assign(new Error("Internal Server Error"), { status: 500 })
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "server_error",
        title: "Server encountered an error"
      })
      expect((result as BackendUnreachableClassification).message).toContain("500")
      expect((result as BackendUnreachableClassification).fixHint).toContain("server logs")
    })

    it("classifies HTTP 502 as server_error", () => {
      const result = classifyBackendUnreachableError(
        Object.assign(new Error("Bad Gateway"), { status: 502 })
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "server_error"
      })
    })

    it("classifies HTTP 503 as server_error", () => {
      const result = classifyBackendUnreachableError(
        Object.assign(new Error("Service Unavailable"), { status: 503 })
      )

      expect(result).toMatchObject({
        kind: "backend_unreachable",
        subtype: "server_error"
      })
    })
  })

  describe("fixHint and title populated for all subtypes", () => {
    it("always includes a subtype field on backend_unreachable results", () => {
      const transportErrors = [
        new Error("Failed to fetch"),
        new Error("NetworkError when attempting to fetch resource."),
        Object.assign(new Error("Unauthorized"), { status: 401 }),
        Object.assign(new Error("Forbidden"), { status: 403 }),
        Object.assign(new Error("Internal Server Error"), { status: 500 }),
        new Error("timeout of 5000ms exceeded"),
        new Error("blocked by CORS policy"),
        new Error("net::ERR_CONNECTION_REFUSED")
      ]

      for (const error of transportErrors) {
        const result = classifyBackendUnreachableError(error)
        expect(result.kind).toBe("backend_unreachable")
        if (result.kind === "backend_unreachable") {
          expect(result.subtype).toBeTruthy()
          expect(result.title).toBeTruthy()
          expect(result.message).toBeTruthy()
        }
      }
    })
  })
})
