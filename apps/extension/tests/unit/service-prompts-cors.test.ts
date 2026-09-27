import { describe, expect, it } from "vitest"
import { fixtureCorsHeaders } from "../e2e/utils/cors"

const allowedOrigins = new Set([
  "http://127.0.0.1:8080",
  "chrome-extension://selected-test-extension"
])

describe("service prompt fixture CORS", () => {
  it.each([
    undefined, "null", "https://attacker.example", "http://127.0.0.1:8081",
    "chrome-extension://other-extension"
  ])("does not authorize an unselected origin: %s", (origin) => {
    const headers = fixtureCorsHeaders(origin, allowedOrigins)
    expect(headers["access-control-allow-origin"]).toBeUndefined()
    expect(headers["access-control-allow-credentials"]).toBeUndefined()
  })

  it.each([...allowedOrigins])("authorizes the selected surface: %s", (origin) => {
    expect(fixtureCorsHeaders(origin, allowedOrigins)).toEqual({
      "access-control-allow-origin": origin,
      "access-control-allow-credentials": "true",
      vary: "Origin"
    })
  })
})
