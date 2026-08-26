import { describe, expect, it } from "vitest"

import { isExplicitRequestCancellation } from "../request-events"

describe("explicit request cancellation classification", () => {
  it.each([
    Object.assign(new Error("browser-specific text"), { name: "AbortError" }),
    Object.assign(new Error("browser-specific text"), { name: "aborterror" }),
    Object.assign(new Error("request stopped"), { code: "REQUEST_ABORTED" }),
    Object.assign(new Error("request stopped"), { code: "request_aborted" }),
    "AbortError",
    "REQUEST_ABORTED",
    "The operation was aborted.",
    "signal is aborted without reason",
    "Request aborted",
    { message: "The user aborted a request." },
  ])("recognizes established cancellation shape %#", (value) => {
    expect(isExplicitRequestCancellation(value)).toBe(true)
  })

  it.each([
    new Error("Request timed out"),
    { code: "ETIMEDOUT", message: "The request timed out" },
    "Abort option is disabled",
    "Failed to fetch",
    null,
  ])("does not suppress non-cancellation failure %#", (value) => {
    expect(isExplicitRequestCancellation(value)).toBe(false)
  })
})
