import path from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, describe, expect, it } from "vitest"

const appDir = path.resolve(__dirname, "..")
const originalEnvironment = {
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
  TLDW_INTERNAL_API_ORIGIN: process.env.TLDW_INTERNAL_API_ORIGIN,
  TLDW_NEXT_BUILD_CPUS: process.env.TLDW_NEXT_BUILD_CPUS,
}

afterEach(() => {
  for (const [name, value] of Object.entries(originalEnvironment)) {
    if (value === undefined) {
      delete process.env[name]
    } else {
      process.env[name] = value
    }
  }
})

describe("Next production build resources", () => {
  it("honors the explicit build CPU limit used by the production image", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    process.env.TLDW_INTERNAL_API_ORIGIN = "http://app:8000"
    process.env.TLDW_NEXT_BUILD_CPUS = "2"

    const moduleUrl = pathToFileURL(path.join(appDir, "next.config.mjs"))
    moduleUrl.searchParams.set("t", `${Date.now()}-${Math.random()}`)
    const { default: nextConfig } = await import(moduleUrl.href)

    expect(nextConfig.experimental?.cpus).toBe(2)
    expect(nextConfig.experimental?.webpackMemoryOptimizations).toBe(true)
  })
})
