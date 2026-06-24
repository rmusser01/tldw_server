import { afterEach, beforeEach, describe, expect, it } from "vitest"
import handler from "@web/pages/api/_tldw-webui/runtime-config"
import {
  createApiRequest,
  createApiResponse
} from "./test-utils"

const ORIGINAL_ENV = {
  AUTH_MODE: process.env.AUTH_MODE,
  SINGLE_USER_API_KEY: process.env.SINGLE_USER_API_KEY,
  TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH,
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
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

const configureRuntimeAuth = () => {
  process.env.AUTH_MODE = "single_user"
  process.env.SINGLE_USER_API_KEY = "runtime-single-user-key"
  process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH = "1"
  process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
}

const callRuntimeConfig = async (headers: Record<string, string> = {}) => {
  const req = createApiRequest({
    method: "GET",
    url: "/api/_tldw-webui/runtime-config",
    headers: {
      host: "127.0.0.1:8080",
      ...headers
    }
  })
  const res = createApiResponse()
  await handler(req, res)
  return res
}

describe("WebUI runtime config API", () => {
  beforeEach(() => {
    restoreEnv()
    configureRuntimeAuth()
  })

  afterEach(() => {
    restoreEnv()
  })

  it("returns runtime single-user auth for local quickstart requests", async () => {
    const res = await callRuntimeConfig()

    expect(res.statusCode).toBe(200)
    expect(res.headers["cache-control"]).toContain("no-store")
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: true,
        authMode: "single-user",
        apiKey: "runtime-single-user-key"
      },
      networking: {
        deploymentMode: "quickstart"
      }
    })
  })

  it.each([
    ["disabled exposure", { TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: "0" }],
    ["true exposure flag", { TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: "true" }],
    ["yes exposure flag", { TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: "yes" }],
    ["multi-user auth mode", { AUTH_MODE: "multi_user" }],
    ["hyphenated single-user auth mode", { AUTH_MODE: "single-user" }],
    ["placeholder key", { SINGLE_USER_API_KEY: "change-me" }],
    ["repo placeholder key", { SINGLE_USER_API_KEY: "CHANGE_ME_TO_SECURE_API_KEY" }],
    ["repo placeholder key prefix", { SINGLE_USER_API_KEY: "CHANGE_ME_BEFORE_RUNNING" }],
    ["whitespace-bearing key", { SINGLE_USER_API_KEY: " runtime-key " }],
    ["blank key", { SINGLE_USER_API_KEY: "   " }]
  ])("returns unavailable for %s", async (_name, envPatch) => {
    Object.assign(process.env, envPatch)

    const res = await callRuntimeConfig()

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })

  it.each([
    ["api.example.test"],
    ["192.168.1.50:8080"],
    ["0.0.0.0:8080"]
  ])("returns unavailable for non-loopback host %s", async (host) => {
    const res = await callRuntimeConfig({ host })

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
  })

  it.each([
    ["forwarded", "for=203.0.113.10;host=localhost:8080"],
    ["x-forwarded-for", "203.0.113.10"],
    ["x-forwarded-host", "localhost:8080"],
    ["x-real-ip", "203.0.113.10"]
  ])("returns unavailable when %s is present", async (header, value) => {
    const res = await callRuntimeConfig({ [header]: value })

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
  })

  it("rejects non-GET methods without exposing auth", async () => {
    const req = createApiRequest({
      method: "POST",
      headers: { host: "127.0.0.1:8080" }
    })
    const res = createApiResponse()

    await handler(req, res)

    expect(res.statusCode).toBe(405)
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })
})
