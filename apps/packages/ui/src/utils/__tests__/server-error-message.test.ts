import { describe, expect, it } from "vitest"
import {
  buildServerLogHint,
  extractServerCorrelationId,
  sanitizeServerErrorMessage
} from "../server-error-message"

describe("sanitizeServerErrorMessage", () => {
  it("redacts API endpoints and filesystem paths", () => {
    const message = sanitizeServerErrorMessage(
      "Request failed: 500 GET /api/v1/chatbooks/export (/Users/macbook-dev/secrets/config.yml)",
      "fallback message"
    )

    expect(message).toContain("GET [server-endpoint]")
    expect(message).toContain("[redacted-path]")
    expect(message).not.toContain("/api/v1/chatbooks/export")
    expect(message).not.toContain("/Users/macbook-dev/secrets/config.yml")
  })

  it("redacts standalone API endpoints with query strings", () => {
    const message = sanitizeServerErrorMessage(
      "/api/v1/chats?limit=50&token=sk_chat_secret returned 404",
      "fallback message"
    )

    expect(message).toContain("[server-endpoint]")
    expect(message).not.toContain("/api/v1/chats")
    expect(message).not.toContain("sk_chat_secret")
  })

  it("redacts full server URLs", () => {
    const message = sanitizeServerErrorMessage(
      "POST https://localhost:8000/api/v1/chatbooks/import failed",
      "fallback message"
    )

    expect(message).toContain("[server-url]")
    expect(message).not.toContain("https://localhost:8000")
  })

  it("redacts token-like secrets", () => {
    const message = sanitizeServerErrorMessage(
      "Request failed token=sk_secret_inline Authorization: Bearer sk-secret-inline api_key=api_secret_value",
      "fallback message"
    )

    expect(message).toContain("token=[redacted-secret]")
    expect(message).toContain("api_key=[redacted-secret]")
    expect(message).toContain("Bearer [redacted-secret]")
    expect(message).not.toContain("sk_secret_inline")
    expect(message).not.toContain("sk-secret-inline")
    expect(message).not.toContain("api_secret_value")
  })

  it("redacts prefixed assignment-style secret keys", () => {
    const message = sanitizeServerErrorMessage(
      "access_token=access-secret refresh_token=refresh-secret client_secret=client-secret api-key=api-key-secret",
      "fallback"
    )

    expect(message).toContain("access_token=[redacted-secret]")
    expect(message).toContain("refresh_token=[redacted-secret]")
    expect(message).toContain("client_secret=[redacted-secret]")
    expect(message).toContain("api-key=[redacted-secret]")
    expect(message).not.toContain("access-secret")
    expect(message).not.toContain("refresh-secret")
    expect(message).not.toContain("client-secret")
    expect(message).not.toContain("api-key-secret")
  })

  it("uses fallback when error content is empty", () => {
    expect(sanitizeServerErrorMessage("", "fallback message")).toBe(
      "fallback message"
    )
  })

  it("extracts correlation ID from common backend error formats", () => {
    const correlationId = extractServerCorrelationId(
      "Request failed: 500 request_id=abc123-xyz789"
    )
    expect(correlationId).toBe("abc123-xyz789")
  })

  it("builds log hint with correlation ID when available", () => {
    const hint = buildServerLogHint(
      "error: correlation-id: req-456-789",
      "fallback hint"
    )
    expect(hint).toContain("req-456-789")
  })

  it("falls back to generic hint when no correlation ID exists", () => {
    expect(buildServerLogHint("Request failed: 500", "fallback hint")).toBe(
      "fallback hint"
    )
  })
})
