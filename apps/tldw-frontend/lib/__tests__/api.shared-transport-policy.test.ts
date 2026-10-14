import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

/**
 * TASK-13526 transport Stage A: the web client adopts the shared timeout and
 * Retry-After policies from packages/ui's request-core instead of its own
 * fixed 30s timeout and parseInt-only parsing.
 */
describe("web api client shared transport policy (source contract)", () => {
  const source = fs.readFileSync(
    path.resolve(__dirname, "../api.ts"),
    "utf8"
  )

  it("derives timeouts through the shared policy with the web default seeding requestTimeoutMs", () => {
    expect(source).toContain(
      "deriveRequestTimeout("
    )
    expect(source).toMatch(
      /requestTimeoutMs:\s*api\.defaults\.timeout/
    )
    // The bare fixed-timeout read must not creep back.
    expect(source).not.toMatch(
      /const timeout = config\.timeout \?\? api\.defaults\.timeout;/
    )
  })

  it("parses Retry-After with the shared parser and converts ms back to the web's seconds semantics", () => {
    expect(source).toContain("parseRetryAfter(headers.get('retry-after'))")
    expect(source).toContain("Math.ceil(ms / 1000)")
    expect(source).not.toContain("parseInt(retryAfterHeader, 10)")
  })
})
