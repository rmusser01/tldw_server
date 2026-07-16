import { describe, expect, it, vi } from "vitest"

import {
  mockPowerUserSkillsLibrary,
  validateSkillsFixtureRequest,
} from "../../../tldw-frontend/e2e/utils/skills-fixtures"

type FixtureRequest = {
  method: () => string
  url: () => string
  headers: () => Record<string, string>
}

type RequestContract = {
  origin: string
  apiKey?: string
}

const contract = {
  origin: "http://skills-parity.invalid",
  apiKey: "skills-parity-test-key",
}

const makeRequest = (
  overrides: {
    method?: string
    url?: string
    apiKey?: string
  } = {},
): FixtureRequest => ({
  method: () => overrides.method ?? "GET",
  url: () => overrides.url ?? `${contract.origin}/api/v1/skills`,
  headers: () => overrides.apiKey === undefined
    ? {}
    : { "x-api-key": overrides.apiKey },
})

describe("Skills fixture request contract", () => {
  it("rejects a request from the wrong origin", () => {
    const request = makeRequest({
      url: "http://wrong-origin.invalid/api/v1/skills",
      apiKey: contract.apiKey,
    })

    expect(() => validateSkillsFixtureRequest(request, "GET", contract)).toThrow(
      "Expected request origin http://skills-parity.invalid",
    )
  })

  it("rejects the wrong request method", () => {
    const request = makeRequest({ method: "POST", apiKey: contract.apiKey })

    expect(() => validateSkillsFixtureRequest(request, "GET", contract)).toThrow(
      "Expected GET request, received POST",
    )
  })

  it("normalizes request and expected methods", () => {
    const request = makeRequest({ method: "get", apiKey: contract.apiKey })

    expect(() => validateSkillsFixtureRequest(request, "gEt", contract)).not.toThrow()
  })

  it("reports malformed request URLs with a bounded contract error", () => {
    const malformedUrl = `not-a-url-${"x".repeat(20)}${"\0".repeat(500)}`
    const request = makeRequest({
      url: malformedUrl,
      apiKey: contract.apiKey,
    })

    let thrown: unknown
    try {
      validateSkillsFixtureRequest(request, "GET", contract)
    } catch (error) {
      thrown = error
    }
    expect(thrown).toBeInstanceOf(Error)
    const message = (thrown as Error).message
    expect(message).toMatch(
      /^Unable to validate Skills fixture request origin: malformed URL "not-a-url-x+/,
    )
    expect(message).toMatch(/\.\.\."$/)
    expect(message.length).toBeLessThanOrEqual(200)
    expect(message).not.toContain(malformedUrl)
  })

  it("rejects a missing API key", () => {
    expect(() => validateSkillsFixtureRequest(makeRequest(), "GET", contract)).toThrow(
      "Expected x-api-key request header",
    )
  })

  it("rejects the wrong API key", () => {
    const request = makeRequest({ apiKey: "wrong-key" })

    expect(() => validateSkillsFixtureRequest(request, "GET", contract)).toThrow(
      "Unexpected x-api-key request header",
    )
  })

  it("accepts a request that matches the method, origin, and API key", () => {
    const request = makeRequest({ apiKey: contract.apiKey })

    expect(() => validateSkillsFixtureRequest(request, "GET", contract)).not.toThrow()
  })

  it("supports a public origin-and-method-only contract", () => {
    const publicContract = {
      origin: contract.origin,
    }

    expect(() => validateSkillsFixtureRequest(
      makeRequest(),
      "GET",
      publicContract,
    )).not.toThrow()
    expect(() => validateSkillsFixtureRequest(
      makeRequest({ url: "http://wrong-origin.invalid/api/v1/skills" }),
      "GET",
      publicContract,
    )).toThrow("Expected request origin http://skills-parity.invalid")
    expect(() => validateSkillsFixtureRequest(
      makeRequest({ method: "POST" }),
      "GET",
      publicContract,
    )).toThrow("Expected GET request, received POST")
  })

  it("keeps origin and API-key validation optional for WebUI fixtures", () => {
    const request = makeRequest({
      url: "http://127.0.0.1:3000/api/v1/skills",
    })

    expect(() => validateSkillsFixtureRequest(request, "GET")).not.toThrow()
  })

  it("registers only the dedicated route for the Skills context endpoint", async () => {
    const routeMatchers: Array<string | RegExp> = []
    const page = {
      route: vi.fn(async (matcher: string | RegExp) => {
        routeMatchers.push(matcher)
      }),
    }

    await mockPowerUserSkillsLibrary(page as never)

    const contextUrl = `${contract.origin}/api/v1/skills/context`
    const detailUrl = `${contract.origin}/api/v1/skills/target-research-formatter`
    const matchingRoutes = (url: string) => routeMatchers.filter(
      (matcher) => matcher instanceof RegExp && matcher.test(url),
    )

    expect(matchingRoutes(contextUrl)).toHaveLength(1)
    expect(matchingRoutes(detailUrl)).toHaveLength(1)
  })
})
