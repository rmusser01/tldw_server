import { afterEach, beforeEach, describe, expect, it } from "vitest"
import handler from "@web/pages/api/_tldw-webui/runtime-config"
import { createApiRequest, createApiResponse } from "./test-utils"

const ORIGINAL_ENV = {
  AUTH_MODE: process.env.AUTH_MODE,
  SINGLE_USER_API_KEY: process.env.SINGLE_USER_API_KEY,
  TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH,
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  TLDW_INTERNAL_API_ORIGIN: process.env.TLDW_INTERNAL_API_ORIGIN,
  SINGLE_USER_SESSION_COOKIE_NAME: process.env.SINGLE_USER_SESSION_COOKIE_NAME
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
  process.env.TLDW_INTERNAL_API_ORIGIN = "http://app:8000"
  delete process.env.SINGLE_USER_SESSION_COOKIE_NAME
}

const callRuntimeConfig = async (
  headers: Record<string, string> = {},
  remoteAddress: string | null = "127.0.0.1"
) => {
  const req = createApiRequest({
    method: "GET",
    url: "/api/_tldw-webui/runtime-config",
    headers: {
      host: "127.0.0.1:8080",
      ...headers
    }
  })
  if (remoteAddress !== null) {
    Object.defineProperty(req, "socket", {
      configurable: true,
      value: { remoteAddress }
    })
  }
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

  it("returns only cookie-session capability for local quickstart requests", async () => {
    const res = await callRuntimeConfig()

    expect(res.statusCode).toBe(200)
    expect(res.headers["cache-control"]).toContain("no-store")
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: true,
        authMode: "single-user",
        transport: "cookie-session"
      },
      networking: {
        deploymentMode: "quickstart"
      }
    })
    expect(res.body.runtimeAuth).toEqual({
      available: true,
      authMode: "single-user",
      transport: "cookie-session"
    })
    expect(JSON.stringify(res.body ?? "")).not.toContain("runtime-single-user-key")
  })

  it.each(["custom_session", "CSRF_TOKEN"])(
    "keeps valid custom session cookie name %s server-only",
    async (cookieName) => {
      process.env.SINGLE_USER_SESSION_COOKIE_NAME = cookieName

      const res = await callRuntimeConfig()

      expect(res.statusCode).toBe(200)
      expect(res.body.runtimeAuth).toEqual({
        available: true,
        authMode: "single-user",
        transport: "cookie-session"
      })
      expect(JSON.stringify(res.body)).not.toContain(cookieName)
    }
  )

  it.each([
    "",
    "csrf_token",
    "__Host-session",
    "__Http-session",
    "__secure-session",
    "invalid name",
    "session=value",
    "session;name",
    "/session"
  ])(
    "returns unavailable for invalid cookie name %j",
    async (cookieName) => {
      process.env.SINGLE_USER_SESSION_COOKIE_NAME = cookieName

      const res = await callRuntimeConfig()

      expect(res.statusCode).toBe(200)
      expect(res.body.runtimeAuth).toEqual({ available: false })
      if (cookieName) expect(JSON.stringify(res.body)).not.toContain(cookieName)
      expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
    }
  )

  it.each([
    ["disabled exposure", { TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: "0" }],
    ["true exposure flag", { TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: "true" }],
    ["yes exposure flag", { TLDW_WEBUI_EXPOSE_RUNTIME_AUTH: "yes" }],
    ["multi-user auth mode", { AUTH_MODE: "multi_user" }],
    ["hyphenated single-user auth mode", { AUTH_MODE: "single-user" }],
    ["placeholder key", { SINGLE_USER_API_KEY: "change-me" }],
    ["repo placeholder key", { SINGLE_USER_API_KEY: "CHANGE_ME_TO_SECURE_API_KEY" }],
    ["repo placeholder key prefix", { SINGLE_USER_API_KEY: "CHANGE_ME_BEFORE_RUNNING" }],
    ["migration hyphen placeholder key", { SINGLE_USER_API_KEY: "your-api-key-here" }],
    ["migration underscore placeholder key", { SINGLE_USER_API_KEY: "your_api_key_here" }],
    ["default key", { SINGLE_USER_API_KEY: "default" }],
    ["test key", { SINGLE_USER_API_KEY: "test-key" }],
    ["short key", { SINGLE_USER_API_KEY: "short-key" }],
    ["whitespace-bearing key", { SINGLE_USER_API_KEY: " runtime-key " }],
    ["blank key", { SINGLE_USER_API_KEY: "   " }]
  ])("returns unavailable for %s", async (_name, envPatch) => {
    Object.assign(process.env, envPatch)

    const res = await callRuntimeConfig()
    const bodyText = JSON.stringify(res.body)
    const patchedApiKey = envPatch.SINGLE_USER_API_KEY

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(res.body.runtimeAuth).toEqual({ available: false })
    expect(res.body).toMatchObject({
      runtimeAuth: expect.not.objectContaining({
        apiKey: expect.any(String)
      })
    })
    expect(bodyText).not.toContain("runtime-single-user-key")
    if (patchedApiKey?.trim()) {
      expect(bodyText).not.toContain(patchedApiKey)
    }
  })

  it("returns unavailable when the exposure flag is omitted", async () => {
    delete process.env.TLDW_WEBUI_EXPOSE_RUNTIME_AUTH

    const res = await callRuntimeConfig()

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(res.body.runtimeAuth).toEqual({ available: false })
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })

  it.each([["127.0.0.1"], ["::1"], ["::ffff:127.0.0.1"]])(
    "returns runtime single-user auth for loopback peer %s",
    async (remoteAddress) => {
      const res = await callRuntimeConfig({}, remoteAddress)

      expect(res.statusCode).toBe(200)
      expect(res.body).toMatchObject({
        runtimeAuth: {
          available: true,
          authMode: "single-user",
          transport: "cookie-session"
        }
      })
    }
  )

  it.each(["127.0.0.1", "::1", "::ffff:127.0.0.1", "172.17.0.1"])(
    "accepts the canonical forwarding quartet injected for trusted peer %s",
    async (remoteAddress) => {
      const res = await callRuntimeConfig(
        {
          host: "localhost:8080",
          "x-forwarded-for": remoteAddress,
          "x-forwarded-host": "localhost:8080",
          "x-forwarded-port": "8080",
          "x-forwarded-proto": "http"
        },
        remoteAddress
      )

      expect(res.body.runtimeAuth).toMatchObject({ available: true })
    }
  )

  it.each([
    ["peer", "x-forwarded-for", "203.0.113.10"],
    ["host", "x-forwarded-host", "example.test:8080"],
    ["port", "x-forwarded-port", "8443"],
    ["protocol", "x-forwarded-proto", "https"]
  ])(
    "rejects a canonical forwarding quartet with mismatched %s",
    async (_name, header, value) => {
      const res = await callRuntimeConfig({
        host: "localhost:8080",
        "x-forwarded-for": "127.0.0.1",
        "x-forwarded-host": "localhost:8080",
        "x-forwarded-port": "8080",
        "x-forwarded-proto": "http",
        [header]: value
      })

      expect(res.body.runtimeAuth).toEqual({ available: false })
    }
  )

  it.each([["172.17.0.1"], ["172.18.0.1"], ["::ffff:172.18.0.1"], ["192.168.65.1"]])(
    "returns runtime single-user auth for quickstart Docker gateway peer %s",
    async (remoteAddress) => {
      const res = await callRuntimeConfig({}, remoteAddress)

      expect(res.statusCode).toBe(200)
      expect(res.body).toMatchObject({
        runtimeAuth: {
          available: true,
          authMode: "single-user",
          transport: "cookie-session"
        }
      })
    }
  )

  it.each([["192.168.1.50"], ["10.0.0.5"], ["172.17.0.2"], ["172.32.0.1"]])(
    "returns unavailable for nonlocal spoof peer %s",
    async (remoteAddress) => {
      const res = await callRuntimeConfig({ host: "127.0.0.1:8080" }, remoteAddress)

      expect(res.statusCode).toBe(200)
      expect(res.body).toMatchObject({
        runtimeAuth: {
          available: false
        }
      })
      expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
    }
  )

  it("returns unavailable for a Docker gateway peer outside quickstart deployment mode", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "production"

    const res = await callRuntimeConfig({}, "172.17.0.1")

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })

  it("returns unavailable for a loopback peer outside quickstart deployment mode", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "production"

    const res = await callRuntimeConfig({}, "127.0.0.1")

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(res.body).toMatchObject({
      runtimeAuth: expect.not.objectContaining({
        apiKey: expect.any(String)
      })
    })
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })

  it("returns unavailable for a spoofed loopback host from a non-loopback peer", async () => {
    const res = await callRuntimeConfig({ host: "127.0.0.1:8080" }, "203.0.113.10")

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })

  it("returns unavailable when the peer address is absent", async () => {
    const res = await callRuntimeConfig({}, null)

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })

  it.each([["api.example.test"], ["192.168.1.50:8080"], ["0.0.0.0:8080"]])(
    "returns unavailable for non-loopback host %s",
    async (host) => {
      const res = await callRuntimeConfig({ host })

      expect(res.statusCode).toBe(200)
      expect(res.body).toMatchObject({
        runtimeAuth: {
          available: false
        }
      })
      expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
    }
  )

  it.each([
    ["forwarded", "forwarded", "for=203.0.113.10;host=localhost:8080"],
    ["x-forwarded-for", "x-forwarded-for", "203.0.113.10"],
    ["empty x-forwarded-for", "x-forwarded-for", ""],
    ["x-forwarded-host", "x-forwarded-host", "localhost:8080"],
    ["x-forwarded-proto", "x-forwarded-proto", "http"],
    ["x-real-ip", "x-real-ip", "203.0.113.10"]
  ])("returns unavailable when %s is present", async (_name, header, value) => {
    const res = await callRuntimeConfig({ [header]: value })

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })

  it("rejects non-GET methods without exposing auth", async () => {
    const req = createApiRequest({
      method: "POST",
      headers: { host: "127.0.0.1:8080" }
    })
    const res = createApiResponse()

    await handler(req, res)

    expect(res.statusCode).toBe(405)
    expect(JSON.stringify(res.body ?? "")).not.toContain("runtime-single-user-key")
  })

  it.each([
    ["missing origin", ""],
    ["relative origin", "/api"],
    ["non-HTTP origin", "ftp://app:8000"],
    ["credential-bearing origin", "http://user:pass@app:8000"],
    ["path-bearing origin", "http://app:8000/backend"],
    ["query-bearing origin", "http://app:8000/?target=other"],
    ["fragment-bearing origin", "http://app:8000/#backend"],
    ["empty-query marker", "http://app:8000?"],
    ["empty-fragment marker", "http://app:8000#"],
    ["dot-segment path", "http://app:8000/./"],
    ["collapsed dot-segment path", "http://app:8000/a/../"],
    ["noncanonical host case", "http://APP:8000"],
    ["default port", "http://app:80"],
    ["surrounding whitespace", " http://app:8000 "]
  ])("returns unavailable for %s internal API origin", async (_name, origin) => {
    process.env.TLDW_INTERNAL_API_ORIGIN = origin

    const res = await callRuntimeConfig()

    expect(res.statusCode).toBe(200)
    expect(res.body).toMatchObject({
      runtimeAuth: {
        available: false
      }
    })
    expect(res.body.runtimeAuth).toEqual({ available: false })
    expect(JSON.stringify(res.body)).not.toContain("runtime-single-user-key")
  })
})
