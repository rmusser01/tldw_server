import { describe, expect, it } from "vitest"
import { createTokenRefreshError } from "@/services/tldw/auth-refresh-error"

describe("createTokenRefreshError", () => {
  it("redacts sensitive keys from the failed response body but keeps the session code", () => {
    const error = createTokenRefreshError({
      ok: false,
      status: 401,
      data: { detail: { code: "session_replaced", trace: "secret stack" } }
    } as never) as Error & { details?: { detail: Record<string, unknown> } }
    expect(error.details?.detail).toEqual({ code: "session_replaced", trace: "[REDACTED]" })
  })

  it("attaches no details when the response was ok", () => {
    const error = createTokenRefreshError({ ok: true, status: 200, data: { access: "x" } } as never)
    expect((error as { details?: unknown }).details).toBeUndefined()
  })
})
