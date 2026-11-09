import { describe, expect, it } from "vitest"
import { isTimeoutLikeError, resolveRequestTimeout } from "@/utils/request-timeout"

describe("request-timeout", () => {
  it("treats AbortError as timeout-like", () => {
    const err = new Error("Aborted")
    err.name = "AbortError"
    expect(isTimeoutLikeError(err)).toBe(true)
  })

  it("treats timeout wording as timeout-like", () => {
    expect(isTimeoutLikeError(new Error("Request timed out"))).toBe(true)
    expect(isTimeoutLikeError("Extension messaging timeout")).toBe(true)
  })

  it("does not classify unrelated errors as timeout-like", () => {
    expect(isTimeoutLikeError(new Error("Unauthorized"))).toBe(false)
  })
})


describe("managed model startup timeout", () => {
  it.each([
    "/api/v1/llamacpp/profiles/cold/start",
    "/api/v1/llamacpp/profiles/cold/resume",
    "/api/v1/llamacpp/start-by-model",
    "/api/v1/llamacpp/start_server"
  ])("lets %s outlast cold loading while retaining explicit overrides", path => {
    expect(resolveRequestTimeout({ requestTimeoutMs: 10000 }, path)).toBe(120000)
    expect(resolveRequestTimeout({ requestTimeoutMs: "10000" }, path)).toBe(120000)
    expect(resolveRequestTimeout(null, path)).toBe(120000)
    expect(resolveRequestTimeout({ requestTimeoutMs: 10000 }, path, 2000)).toBe(2000)
  })

  it.each([
    "/api/v1/llamacpp/profiles/cold/stop",
    "/api/v1/llamacpp/profiles/cold/start-other",
    "/api/v1/llamacpp/profiles/cold/starting",
    "/api/v1/llamacpp/start_server-other"
  ])("keeps the configured short timeout for %s", path => {
    expect(resolveRequestTimeout({ requestTimeoutMs: 10000 }, path)).toBe(10000)
  })
})
