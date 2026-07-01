import { beforeEach, describe, expect, it } from "vitest"

import { addRequestHistory, getRequestHistory } from "@web/lib/history"

describe("request history", () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it("redacts credential headers before writing request history", () => {
    addRequestHistory({
      id: "req-1",
      method: "GET",
      url: "/api/v1/config",
      timestamp: "2026-06-30T00:00:00.000Z",
      requestHeaders: {
        Authorization: "Bearer secret-token",
        "x-api-key": "secret-api-key",
        "content-type": "application/json"
      }
    })

    const raw = localStorage.getItem("tldw-request-history") || ""
    expect(raw).not.toContain("secret-token")
    expect(raw).not.toContain("secret-api-key")

    const [item] = getRequestHistory()
    expect(item.requestHeaders).toEqual({
      Authorization: "[REDACTED]",
      "x-api-key": "[REDACTED]",
      "content-type": "application/json"
    })
  })

  it("migrates existing unredacted request history on read", () => {
    localStorage.setItem("tldw-request-history", JSON.stringify([
      {
        id: "req-1",
        method: "GET",
        url: "/api/v1/config",
        timestamp: "2026-06-30T00:00:00.000Z",
        requestHeaders: {
          Authorization: "Bearer old-secret",
          Cookie: "sid=old-cookie",
          Accept: "application/json"
        }
      }
    ]))

    const [item] = getRequestHistory()

    expect(item.requestHeaders).toEqual({
      Authorization: "[REDACTED]",
      Cookie: "[REDACTED]",
      Accept: "application/json"
    })
    const raw = localStorage.getItem("tldw-request-history") || ""
    expect(raw).not.toContain("old-secret")
    expect(raw).not.toContain("old-cookie")
  })
})
