import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import handler from "@web/pages/openapi.json"
import {
  createApiRequest,
  createApiResponse
} from "./api/test-utils"

const ORIGINAL_ENV = {
  TLDW_INTERNAL_API_ORIGIN: process.env.TLDW_INTERNAL_API_ORIGIN,
  TLDW_SERVER_URL: process.env.TLDW_SERVER_URL,
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL
}

const restoreEnv = () => {
  for (const [key, value] of Object.entries(ORIGINAL_ENV)) {
    if (value === undefined) {
      delete process.env[key]
    } else {
      process.env[key] = value
    }
  }
}

describe("WebUI OpenAPI same-origin proxy", () => {
  beforeEach(() => {
    restoreEnv()
    vi.restoreAllMocks()
  })

  afterEach(() => {
    restoreEnv()
    vi.restoreAllMocks()
  })

  it("proxies same-origin /openapi.json to the configured backend OpenAPI document", async () => {
    process.env.TLDW_INTERNAL_API_ORIGIN = "http://127.0.0.1:8000/"
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ openapi: "3.1.0", paths: { "/api/v1/media": {} } }), {
        status: 200,
        headers: { "content-type": "application/json" }
      })
    )
    const req = createApiRequest({
      method: "GET",
      url: "/openapi.json"
    })
    const res = createApiResponse()

    await handler(req, res)

    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/openapi.json",
      expect.objectContaining({
        method: "GET",
        headers: expect.objectContaining({
          accept: "application/json"
        })
      })
    )
    expect(res.statusCode).toBe(200)
    expect(res.headers["cache-control"]).toContain("no-store")
    expect(res.headers["content-type"]).toContain("application/json")
    expect(res.body).toEqual({
      openapi: "3.1.0",
      paths: { "/api/v1/media": {} }
    })
  })
})
