import { readFileSync } from "node:fs"
import path from "node:path"
import { pathToFileURL } from "node:url"
import { describe, expect, it } from "vitest"

describe("Next dev watch guard", () => {
  const configSource = readFileSync(
    path.resolve(__dirname, "../next.config.mjs"),
    "utf8",
  )
  const packageJson = JSON.parse(
    readFileSync(path.resolve(__dirname, "../package.json"), "utf8"),
  ) as { scripts?: Record<string, string> }
  const loadNextConfig = async () => {
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8001"
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    delete process.env.TLDW_INTERNAL_API_ORIGIN
    const moduleUrl = `${pathToFileURL(
      path.resolve(__dirname, "../next.config.mjs"),
    ).href}?watch-guard`
    const mod = await import(moduleUrl)
    return mod.default
  }

  it("uses webpack for default local dev so watch ignores are honored", () => {
    expect(packageJson.scripts?.dev).toBe("next dev --webpack")
    expect(packageJson.scripts?.["dev:turbopack"]).toBe("next dev")
  })

  it("ignores backend runtime output for webpack dev fallback", () => {
    expect(configSource).toContain("backendRuntimeWatchIgnorePatterns")
    expect(configSource).toContain("../../Databases")
    expect(configSource).toContain("../../tldw_Server_API/Databases")
    expect(configSource).toContain("config.watchOptions")
    expect(configSource).toContain("ignored")
  })

  it("normalizes webpack ignored watch patterns to strings", async () => {
    const nextConfig = await loadNextConfig()
    const webpackConfig = nextConfig.webpack({
      resolve: { alias: {} },
      watchOptions: {
        ignored: [/node_modules/],
      },
    })

    expect(webpackConfig.watchOptions.ignored.every(
      (item: unknown) => typeof item === "string" && item.length > 0,
    )).toBe(true)
    expect(webpackConfig.watchOptions.ignored).toEqual(
      expect.arrayContaining([
        expect.stringContaining("../../Databases"),
        expect.stringContaining("../../tldw_Server_API/Databases"),
      ]),
    )
  })
})
