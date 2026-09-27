import { describe, it, expect } from "vitest"
import { classifyError } from "../ErrorClassification"

describe("classifyError", () => {
  it.each([
    { extraction_failures: [{ code: "extraction_timeout" }, { code: "extraction_failed" }] },
    { extraction_failures: [{ code: "constructor" }] },
    { extraction_failures: [null] },
    { extraction_failures: [{ code: "extraction_timeout" }], errors: ["Source timed out", "Storage failed"] },
  ])("does not infer a safe retry from mixed or malformed extraction metadata", (data) => {
    expect(classifyError("request timed out", data).retryable).toBe(false)
  })

  it("reads the confirmed extraction code inside a completed job envelope", () => {
    expect(classifyError("generic failure", { result: { extraction_failures: [{ code: "extraction_timeout" }] } }).retryable).toBe(true)
  })

  // -------------------------------------------------------------------------
  // Timeout category
  // -------------------------------------------------------------------------
  describe("timeout category", () => {
    it.each(["timeout", "Request timed out", "deadline exceeded"])(
      'returns timeout for "%s"',
      (msg) => {
        const result = classifyError(msg)
        expect(result.classification).toBe("timeout")
        expect(result.retryable).toBe(true)
        expect(result.badgeLabel).toContain("Timeout")
      }
    )
  })

  // -------------------------------------------------------------------------
  // Queue-full category
  // -------------------------------------------------------------------------
  describe("queue-full category", () => {
    it.each([
      "HTTP 429 Too Many Requests",
      "User 1 has reached the maximum concurrent job limit (5)",
    ])('returns queue-full guidance for "%s"', (msg) => {
      const result = classifyError(msg)
      expect(result.classification).toBe("server")
      expect(result.retryable).toBe(false)
      expect(result.badgeLabel).toContain("Queue Full")
      expect(result.userMessage).toContain("maximum number of ingest jobs")
    })
  })

  // -------------------------------------------------------------------------
  // Auth category
  // -------------------------------------------------------------------------
  describe("auth category", () => {
    it.each([
      "HTTP 401 response",
      "status 403",
      "unauthorized access",
      "forbidden resource",
    ])('returns auth for "%s"', (msg) => {
      const result = classifyError(msg)
      expect(result.classification).toBe("auth")
      expect(result.retryable).toBe(false)
      expect(result.badgeLabel).toContain("Auth")
    })

    it("returns configuration guidance before generic HTTP 400 handling", () => {
      const result = classifyError("HTTP 400: tldw server not configured")

      expect(result.classification).toBe("auth")
      expect(result.retryable).toBe(false)
      expect(result.badgeLabel).toContain("Config")
      expect(result.userMessage).toContain("server is not configured")
    })

    it("does not relabel unrelated server configuration errors", () => {
      const result = classifyError(
        "Third-party server configuration is invalid"
      )

      expect(result.badgeLabel).not.toContain("Config")
      expect(result.userMessage).not.toContain("server is not configured")
    })
  })

  // -------------------------------------------------------------------------
  // Validation category
  // -------------------------------------------------------------------------
  describe("validation category", () => {
    it.each([
      "HTTP 400 bad request",
      "invalid input provided",
      "unsupported format detected",
    ])('returns validation for "%s"', (msg) => {
      const result = classifyError(msg)
      expect(result.classification).toBe("validation")
      expect(result.retryable).toBe(false)
      expect(result.badgeLabel).toContain("Format")
    })
  })

  // -------------------------------------------------------------------------
  // Server category
  // -------------------------------------------------------------------------
  describe("server category", () => {
    it.each([
      "HTTP 500 error",
      "503 service unavailable",
      "internal server error occurred",
    ])('returns server for "%s"', (msg) => {
      const result = classifyError(msg)
      expect(result.classification).toBe("server")
      expect(result.retryable).toBe(true)
      expect(result.badgeLabel).toContain("Server")
    })
  })

  // -------------------------------------------------------------------------
  // Network category
  // -------------------------------------------------------------------------
  describe("network category", () => {
    it.each([
      "ECONNREFUSED on port 8000",
      "fetch failed for url",
      "network error",
    ])('returns network for "%s"', (msg) => {
      const result = classifyError(msg)
      expect(result.classification).toBe("network")
      expect(result.retryable).toBe(true)
      expect(result.badgeLabel).toContain("Network")
    })
  })

  // -------------------------------------------------------------------------
  // Unknown / fallback category
  // -------------------------------------------------------------------------
  describe("unknown category", () => {
    it("does not promise that an unknown failure is retryable", () => {
      const result = classifyError(undefined)
      expect(result.classification).toBe("unknown")
      expect(result.retryable).toBe(false)
      expect(result.badgeLabel).toContain("Error")
    })

    it("returns unknown without a retry promise for empty string", () => {
      const result = classifyError("")
      expect(result.classification).toBe("unknown")
      expect(result.retryable).toBe(false)
    })

    it("returns unknown for unrecognized error message", () => {
      const result = classifyError("something completely different happened")
      expect(result.classification).toBe("unknown")
      expect(result.retryable).toBe(false)
    })
  })

  // -------------------------------------------------------------------------
  // Priority / overlap tests
  // -------------------------------------------------------------------------
  describe("priority ordering", () => {
    it('"connection timeout" matches timeout, not network', () => {
      const result = classifyError("connection timeout while uploading")
      expect(result.classification).toBe("timeout")
    })

    it('"timed out" beats "network" even when both could match', () => {
      const result = classifyError("network request timed out")
      expect(result.classification).toBe("timeout")
    })
  })

  // -------------------------------------------------------------------------
  // Structural integrity of returned categories
  // -------------------------------------------------------------------------
  describe("returned object structure", () => {
    it("has all expected fields for every classification", () => {
      const messages = [
        "timeout",
        "401",
        "invalid",
        "500",
        "ECONNREFUSED",
        undefined,
        "xyz",
      ]
      for (const msg of messages) {
        const result = classifyError(msg)
        expect(result).toHaveProperty("classification")
        expect(result).toHaveProperty("retryable")
        expect(result).toHaveProperty("badgeLabel")
        expect(result).toHaveProperty("badgeColor")
        expect(result).toHaveProperty("userMessage")
        expect(result).toHaveProperty("suggestion")
        expect(typeof result.badgeLabel).toBe("string")
        expect(typeof result.badgeColor).toBe("string")
      }
    })
  })
})
