import { describe, expect, it } from "vitest"
import type { AuthSession } from "@/services/auth-sessions"
import type { NotificationItem } from "@/services/notifications"
import {
  isOwnSignIn,
  isProtectedAlertError,
  visibleSignInAlerts
} from "@/services/sign-in-alert"

const session = (over: Partial<AuthSession>): AuthSession => ({
  id: 1,
  ip_address: "10.0.0.1",
  user_agent: "UA",
  device_label: "Chrome on macOS",
  created_at: "2026-01-01T00:00:00Z",
  last_activity: "2026-01-01T00:00:00Z",
  expires_at: "2026-02-01T00:00:00Z",
  is_current: false,
  same_install: false,
  ...over
})

const alert = (over: Partial<NotificationItem>): NotificationItem => ({
  id: 1,
  kind: "new_sign_in",
  title: "t",
  message: "m",
  severity: "warning",
  source_job_id: 9,
  created_at: "2026-01-01T00:00:00Z",
  ...over
})

describe("isOwnSignIn", () => {
  it("is true when the current session is the source", () => {
    expect(isOwnSignIn(7, [session({ id: 7, is_current: true })])).toBe(true)
  })
  it("compares ids as strings", () => {
    expect(isOwnSignIn("7", [session({ id: 7, is_current: true })])).toBe(true)
  })
  it("is true when the source row is from this install", () => {
    const rows = [session({ id: 1, is_current: true }), session({ id: 2, same_install: true })]
    expect(isOwnSignIn(2, rows)).toBe(true)
  })
  it("is false for another install even with the same ip and label", () => {
    const rows = [session({ id: 1, is_current: true }), session({ id: 2, same_install: false })]
    expect(isOwnSignIn(2, rows)).toBe(false)
  })
  it("is false when the source session is gone", () => {
    expect(isOwnSignIn(5, [session({ id: 1, is_current: true })])).toBe(false)
  })
  it("is false without a current session (API key connection)", () => {
    expect(isOwnSignIn(2, [session({ id: 2, same_install: true })])).toBe(false)
  })
  it("is false without a source id", () => {
    expect(isOwnSignIn(null, [session({ id: 1, is_current: true })])).toBe(false)
  })
})

describe("visibleSignInAlerts", () => {
  it("drops read, dismissed, other-kind and own alerts, newest first", () => {
    const sessions = [session({ id: 1, is_current: true })]
    const items = [
      alert({ id: 1, created_at: "2026-01-01T00:00:00Z" }),
      alert({ id: 2, created_at: "2026-01-03T00:00:00Z" }),
      alert({ id: 3, read_at: "2026-01-04T00:00:00Z" }),
      alert({ id: 4, dismissed_at: "2026-01-04T00:00:00Z" }),
      alert({ id: 5, kind: "other" }),
      alert({ id: 6, source_job_id: 1 })
    ]
    expect(visibleSignInAlerts(items, sessions).map((i) => i.id)).toEqual([2, 1])
  })
})

describe("isProtectedAlertError", () => {
  it("matches the ApiError shape", () => {
    expect(isProtectedAlertError({ status: 403, errorCode: "new_sign_in_protected" })).toBe(true)
  })
  it("matches the bgRequest shape", () => {
    expect(
      isProtectedAlertError({ status: 403, details: { detail: { code: "new_sign_in_protected" } } })
    ).toBe(true)
  })
  it("rejects other codes and plain errors", () => {
    expect(isProtectedAlertError({ status: 403, errorCode: "interactive_session_required" })).toBe(false)
    expect(isProtectedAlertError(new Error("x"))).toBe(false)
  })
})
