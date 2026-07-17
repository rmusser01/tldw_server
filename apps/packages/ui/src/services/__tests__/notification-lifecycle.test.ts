import { describe, expect, it } from "vitest"

import {
  buildNotificationScopeKey,
  classifyNotificationError,
  nextReconnectDelay,
  readHttpStatus,
  reduceNotificationLifecycle,
  type NotificationLifecycleState
} from "../notification-lifecycle"

const JWT_WITH_SUB = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJ1c2VyLTQyIn0.signature"

describe("notification lifecycle", () => {
  it("defines every lifecycle state", () => {
    const states: NotificationLifecycleState[] = [
      "idle",
      "connecting",
      "active",
      "degraded",
      "auth-required",
      "unavailable"
    ]

    expect(states).toHaveLength(6)
  })

  it("reads only structured HTTP status fields", () => {
    expect(readHttpStatus({ status: 401 })).toBe(401)
    expect(readHttpStatus({ statusCode: 403 })).toBe(403)
    expect(readHttpStatus({ status: "503", statusCode: 429 })).toBe(429)
    expect(readHttpStatus(new Error("HTTP 500"))).toBeNull()
  })

  it("classifies authentication and terminal client failures", () => {
    expect(classifyNotificationError({ status: 401 })).toEqual({
      kind: "auth-required"
    })
    expect(classifyNotificationError({ statusCode: 403 })).toEqual({
      kind: "unavailable"
    })
    expect(classifyNotificationError({ status: 404 })).toEqual({
      kind: "unavailable"
    })
  })

  it.each([408, 425, 429, 500, 503])("classifies HTTP %s as retryable", (status) => {
    expect(classifyNotificationError({ status })).toMatchObject({
      kind: "retry"
    })
  })

  it("classifies status-free network failures as retryable", () => {
    expect(classifyNotificationError(new TypeError("Failed to fetch"))).toMatchObject({
      kind: "retry"
    })
  })

  it("classifies abort as idle and non-retryable", () => {
    expect(classifyNotificationError(Object.assign(new Error("cancelled"), { name: "AbortError" }))).toEqual({
      kind: "idle"
    })
  })

  it("honors a longer Retry-After delay", () => {
    expect(classifyNotificationError({ status: 503, retryAfter: 40 })).toEqual({
      kind: "retry",
      delayMs: 40_000
    })
  })

  it("caps exponential backoff and supports deterministic injected jitter", () => {
    expect(nextReconnectDelay({ attempt: 20 })).toBe(30_000)
    expect(
      nextReconnectDelay({
        attempt: 2,
        baseDelayMs: 1_000,
        maxDelayMs: 30_000,
        jitter: 0
      })
    ).toBe(3_200)
    expect(
      nextReconnectDelay({
        attempt: 2,
        baseDelayMs: 1_000,
        maxDelayMs: 30_000,
        jitter: 1
      })
    ).toBe(4_800)
  })

  it("reduces lifecycle transitions without runtime side effects", () => {
    expect(reduceNotificationLifecycle("idle", { type: "start" })).toBe("connecting")
    expect(reduceNotificationLifecycle("connecting", { type: "open" })).toBe("active")
    expect(reduceNotificationLifecycle("active", { type: "retry" })).toBe("degraded")
    expect(reduceNotificationLifecycle("degraded", { type: "reconnect" })).toBe("connecting")
    expect(reduceNotificationLifecycle("active", { type: "auth-required" })).toBe("auth-required")
    expect(reduceNotificationLifecycle("active", { type: "unavailable" })).toBe("unavailable")
    expect(reduceNotificationLifecycle("unavailable", { type: "stop" })).toBe("idle")
  })

  it("normalizes server and principal scope like chat surfaces", () => {
    const first = buildNotificationScopeKey({
      serverUrl: "HTTPS://Example.COM:443/api/v1/",
      authMode: "multi-user",
      orgId: 7,
      userId: null,
      accessToken: JWT_WITH_SUB
    })
    const equivalent = buildNotificationScopeKey({
      serverUrl: "https://example.com/api/v1",
      authMode: "MULTI-USER",
      orgId: "7",
      userId: "user-42"
    })

    expect(first).toBe(equivalent)
    expect(first).toContain("user:user-42")
  })

  it("changes scope across principals and never leaks single-user API keys", () => {
    const alpha = buildNotificationScopeKey({
      serverUrl: "https://example.com",
      authMode: "single-user",
      orgId: null,
      userId: null,
      apiKey: "alpha-secret"
    })
    const beta = buildNotificationScopeKey({
      serverUrl: "https://example.com",
      authMode: "single-user",
      orgId: null,
      userId: null,
      apiKey: "beta-secret"
    })

    expect(alpha).not.toBe(beta)
    expect(alpha).not.toContain("alpha-secret")
    expect(beta).not.toContain("beta-secret")
  })
})
