import { readFileSync } from "node:fs"
import path from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

const appDir = path.resolve(__dirname, "..")
const repoRoot = path.resolve(appDir, "..", "..")
const nextConfigPath = path.join(appDir, "next.config.mjs")
const validateNetworkingConfigPath = path.join(appDir, "scripts", "validate-networking-config.mjs")
const makefilePath = path.join(repoRoot, "Makefile")
const webuiComposePath = path.join(repoRoot, "Dockerfiles", "docker-compose.webui.yml")
const singleUserComposePath = path.join(repoRoot, "Dockerfiles", "docker-compose.single-user.yml")
const baseComposePath = path.join(repoRoot, "Dockerfiles", "docker-compose.yml")
const hostStorageComposePath = path.join(repoRoot, "Dockerfiles", "docker-compose.host-storage.yml")

const ORIGINAL_ENV = {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  TLDW_INTERNAL_API_ORIGIN: process.env.TLDW_INTERNAL_API_ORIGIN,
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL
}

const restoreEnv = () => {
  if (ORIGINAL_ENV.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE === undefined) {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  } else {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = ORIGINAL_ENV.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  }

  if (ORIGINAL_ENV.TLDW_INTERNAL_API_ORIGIN === undefined) {
    delete process.env.TLDW_INTERNAL_API_ORIGIN
  } else {
    process.env.TLDW_INTERNAL_API_ORIGIN = ORIGINAL_ENV.TLDW_INTERNAL_API_ORIGIN
  }

  if (ORIGINAL_ENV.NEXT_PUBLIC_API_URL === undefined) {
    delete process.env.NEXT_PUBLIC_API_URL
  } else {
    process.env.NEXT_PUBLIC_API_URL = ORIGINAL_ENV.NEXT_PUBLIC_API_URL
  }
}

const loadNextConfig = async (env: {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE?: string
  TLDW_INTERNAL_API_ORIGIN?: string
  NEXT_PUBLIC_API_URL?: string
}) => {
  if (env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE === undefined) {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  } else {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  }

  if (env.TLDW_INTERNAL_API_ORIGIN === undefined) {
    delete process.env.TLDW_INTERNAL_API_ORIGIN
  } else {
    process.env.TLDW_INTERNAL_API_ORIGIN = env.TLDW_INTERNAL_API_ORIGIN
  }

  if (env.NEXT_PUBLIC_API_URL === undefined) {
    delete process.env.NEXT_PUBLIC_API_URL
  } else {
    process.env.NEXT_PUBLIC_API_URL = env.NEXT_PUBLIC_API_URL
  }

  const moduleUrl = pathToFileURL(nextConfigPath)
  moduleUrl.searchParams.set("t", `${Date.now()}-${Math.random()}`)
  const mod = await import(moduleUrl.href)
  return mod.default
}

const loadValidateNetworkingConfig = async () => {
  const moduleUrl = pathToFileURL(validateNetworkingConfigPath)
  moduleUrl.searchParams.set("t", `${Date.now()}-${Math.random()}`)
  const mod = await import(moduleUrl.href)
  return mod.validateNetworkingConfig as (env?: Record<string, string | undefined>) => {
    deploymentMode: string
    internalApiOrigin: string
    publicApiUrl: string
  }
}

describe("frontend quickstart networking", () => {
  beforeEach(() => {
    restoreEnv()
  })

  afterEach(() => {
    restoreEnv()
  })

  it("adds a quickstart same-origin proxy rewrite for /api/:path*", async () => {
    const nextConfig = await loadNextConfig({
      NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart",
      TLDW_INTERNAL_API_ORIGIN: "http://app:8000"
    })

    expect(nextConfig.rewrites).toEqual(expect.any(Function))

    const rewrites = await nextConfig.rewrites()

    expect(rewrites).toEqual(
      expect.arrayContaining([
        {
          source: "/api/v1/media",
          destination: "http://app:8000/api/v1/media/"
        },
        {
          source: "/api/:path*/",
          destination: "http://app:8000/api/:path*/"
        },
        {
          source: "/api/:path*",
          destination: "http://app:8000/api/:path*"
        }
      ])
    )
  })

  it("preserves API trailing slashes so quickstart rewrites hit backend-canonical routes", async () => {
    const nextConfig = await loadNextConfig({
      NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart",
      TLDW_INTERNAL_API_ORIGIN: "http://app:8000"
    })

    expect(nextConfig.skipTrailingSlashRedirect).toBe(true)
  })

  it("requires TLDW_INTERNAL_API_ORIGIN in quickstart mode", async () => {
    const validateNetworkingConfig = await loadValidateNetworkingConfig()

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart",
        TLDW_INTERNAL_API_ORIGIN: ""
      })
    ).toThrow(/TLDW_INTERNAL_API_ORIGIN/i)
  })

  it.each([
    ["relative URL", "/api"],
    ["non-HTTP URL", "ftp://app:8000"],
    ["credentials", "http://user:pass@app:8000"],
    ["path", "http://app:8000/backend"],
    ["query", "http://app:8000/?target=other"],
    ["fragment", "http://app:8000/#backend"],
    ["empty query marker", "http://app:8000?"],
    ["empty fragment marker", "http://app:8000#"],
    ["dot-segment path", "http://app:8000/./"],
    ["collapsed dot-segment path", "http://app:8000/a/../"],
    ["noncanonical host case", "http://APP:8000"],
    ["default port", "http://app:80"],
    ["surrounding whitespace", " http://app:8000 "]
  ])("rejects an internal quickstart API origin with %s", async (_name, origin) => {
    const validateNetworkingConfig = await loadValidateNetworkingConfig()

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart",
        TLDW_INTERNAL_API_ORIGIN: origin
      })
    ).toThrow(/TLDW_INTERNAL_API_ORIGIN/i)
  })

  it("returns the canonical internal quickstart API origin", async () => {
    const validateNetworkingConfig = await loadValidateNetworkingConfig()

    expect(
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart",
        TLDW_INTERNAL_API_ORIGIN: "http://app:8000/"
      })
    ).toMatchObject({ internalApiOrigin: "http://app:8000" })
  })

  it("rejects an absolute NEXT_PUBLIC_API_URL in quickstart mode", async () => {
    const validateNetworkingConfig = await loadValidateNetworkingConfig()

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart",
        TLDW_INTERNAL_API_ORIGIN: "http://app:8000",
        NEXT_PUBLIC_API_URL: "http://127.0.0.1:8000"
      })
    ).toThrow(/NEXT_PUBLIC_API_URL/i)
  })

  it("requires a valid absolute NEXT_PUBLIC_API_URL in advanced mode", async () => {
    const validateNetworkingConfig = await loadValidateNetworkingConfig()

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced",
        NEXT_PUBLIC_API_URL: ""
      })
    ).toThrow(/NEXT_PUBLIC_API_URL/i)

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced",
        NEXT_PUBLIC_API_URL: "/api"
      })
    ).toThrow(/NEXT_PUBLIC_API_URL/i)

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced",
        NEXT_PUBLIC_API_URL: "ftp://api.example.test"
      })
    ).toThrow(/NEXT_PUBLIC_API_URL/i)

    expect(
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced",
        NEXT_PUBLIC_API_URL: "https://api.example.test"
      })
    ).toMatchObject({
      deploymentMode: "advanced",
      publicApiUrl: "https://api.example.test"
    })
  })

  it("does not add the quickstart proxy rewrite outside quickstart mode", async () => {
    const nextConfig = await loadNextConfig({
      NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced",
      TLDW_INTERNAL_API_ORIGIN: "http://app:8000",
      NEXT_PUBLIC_API_URL: "https://api.example.test"
    })

    const rewrites = await nextConfig.rewrites()

    expect(rewrites).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          source: "/api/:path*",
          destination: "http://app:8000/api/:path*"
        })
      ])
    )
  })

  it("fails next config loading when advanced mode omits NEXT_PUBLIC_API_URL", async () => {
    await expect(
      loadNextConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced",
        TLDW_INTERNAL_API_ORIGIN: "http://app:8000"
      })
    ).rejects.toThrow(/NEXT_PUBLIC_API_URL/i)
  })

  it("normalizes a trailing slash from the internal quickstart API origin", async () => {
    const nextConfig = await loadNextConfig({
      NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart",
      TLDW_INTERNAL_API_ORIGIN: "http://app:8000/"
    })

    const rewrites = await nextConfig.rewrites()

    expect(rewrites).toEqual(
      expect.arrayContaining([
        {
          source: "/api/:path*/",
          destination: "http://app:8000/api/:path*/"
        },
        {
          source: "/api/:path*",
          destination: "http://app:8000/api/:path*"
        }
      ])
    )
  })

  it("defaults quickstart Makefile wiring to quickstart deployment mode", () => {
    const makefile = readFileSync(makefilePath, "utf8")

    expect(makefile).toContain("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE ?= quickstart")
    expect(makefile).toContain("TLDW_INTERNAL_API_ORIGIN ?= http://app:8000")
  })

  it("keeps server single-user keys out of WebUI public compose interpolation", () => {
    const makefile = readFileSync(makefilePath, "utf8")
    const compose = readFileSync(webuiComposePath, "utf8")
    const startTarget = makefile.match(
      /start-docker-single:\n(?<recipe>[\s\S]*?)\n\nverify-docker-single:/
    )

    expect(startTarget?.groups?.recipe).toBeTruthy()
    expect(startTarget?.groups?.recipe).not.toMatch(/grep '\^SINGLE_USER_API_KEY='/)
    expect(compose).toContain("NEXT_PUBLIC_X_API_KEY: ${NEXT_PUBLIC_X_API_KEY:-}")
    expect(compose).toContain("NEXT_PUBLIC_X_API_KEY=${NEXT_PUBLIC_X_API_KEY:-}")
    expect(compose).not.toContain("NEXT_PUBLIC_X_API_KEY:-${SINGLE_USER_API_KEY")
    expect(makefile).toContain(
      'docker compose --env-file "$(TLDW_ENV_FILE)" -f "$(DOCKER_SINGLE_COMPOSE)" -f "$(DOCKER_WEBUI_COMPOSE)"'
    )
  })

  it("keeps the resolved WebUI quickstart key out of echoed Make output", () => {
    const makefile = readFileSync(makefilePath, "utf8")
    const startTarget = makefile.match(
      /start-docker-single:\n(?<recipe>[\s\S]*?)\n\nverify-docker-single:/
    )

    expect(startTarget?.groups?.recipe).toBeTruthy()
    expect(startTarget?.groups?.recipe).not.toContain("grep '^SINGLE_USER_API_KEY='")
    expect(startTarget?.groups?.recipe).not.toContain("cut -d= -f2-")
    expect(startTarget?.groups?.recipe).not.toContain('NEXT_PUBLIC_X_API_KEY="$$(grep')
  })

  it("passes runtime auth env to the WebUI container without changing the loopback port binding", () => {
    const compose = readFileSync(webuiComposePath, "utf8")

    expect(compose).toContain("- AUTH_MODE=${AUTH_MODE:-single_user}")
    expect(compose).toContain("- SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}")
    expect(compose).toContain(
      "- TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=${TLDW_WEBUI_EXPOSE_RUNTIME_AUTH:-0}"
    )
    expect(compose).toContain('"127.0.0.1:8080:3000"')
    expect(compose).toContain(
      "- TLDW_INTERNAL_API_ORIGIN=${TLDW_INTERNAL_API_ORIGIN:-http://app:8000}"
    )
    expect(compose).toContain(
      "- SINGLE_USER_SESSION_COOKIE_NAME=${SINGLE_USER_SESSION_COOKIE_NAME:-tldw_single_user_session}"
    )
  })

  it.each([
    ["base", baseComposePath],
    ["single-user", singleUserComposePath],
    ["host-storage", hostStorageComposePath]
  ])("passes the defaulted session cookie name to the %s backend", (_name, composePath) => {
    const compose = readFileSync(composePath, "utf8")

    expect(compose).toContain(
      "- SINGLE_USER_SESSION_COOKIE_NAME=${SINGLE_USER_SESSION_COOKIE_NAME:-tldw_single_user_session}"
    )
  })

  it("enables CSRF with non-Secure cookies only in the loopback HTTP WebUI overlay", () => {
    const compose = readFileSync(webuiComposePath, "utf8")

    expect(compose).toContain("- CSRF_ENABLED=${CSRF_ENABLED:-1}")
    expect(compose).toContain("- SESSION_COOKIE_SECURE=${SESSION_COOKIE_SECURE:-0}")
  })

  it("provides an explicit cookie lifecycle browser regression command", () => {
    const packageJson = readFileSync(
      path.join(appDir, "package.json"),
      "utf8"
    )

    expect(packageJson).toContain('"e2e:cookie-lifecycle"')
    expect(packageJson).toContain("node scripts/playwright-cookie-lifecycle.mjs")

    const launcher = readFileSync(
      path.join(appDir, "scripts", "playwright-cookie-lifecycle.mjs"),
      "utf8"
    )
    expect(launcher).toContain('reservePorts(["api", "hostile", "web"])')
    expect(launcher).toContain('TLDW_COOKIE_LIFECYCLE: "1"')
  })

  it("enables remote setup writes for the Docker single-user app service", () => {
    const compose = readFileSync(singleUserComposePath, "utf8")

    expect(compose).toContain("- TLDW_SETUP_ALLOW_REMOTE=${TLDW_SETUP_ALLOW_REMOTE:-1}")
    expect(compose).toContain("- AUTH_MODE=${AUTH_MODE:-single_user}")
    expect(compose).toContain("- SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}")
  })
})
