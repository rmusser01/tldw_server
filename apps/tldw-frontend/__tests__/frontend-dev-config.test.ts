import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { pathToFileURL } from "node:url"
import { describe, expect, it } from "vitest"

const appDir = path.resolve(__dirname, "..")

const loadPackageJson = () =>
  JSON.parse(readFileSync(path.join(appDir, "package.json"), "utf8")) as {
    scripts?: Record<string, string>
  }

const turbopackBuildScript =
  "node scripts/build-with-profile.mjs --bundler=turbopack && node scripts/verify-shared-token-sync.mjs --dir .next"
const webpackBuildScript =
  "node scripts/build-with-profile.mjs --bundler=webpack && node scripts/verify-shared-token-sync.mjs --dir .next"

const loadNextConfig = async () => {
  process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
  process.env.TLDW_INTERNAL_API_ORIGIN = "http://app:8000"
  delete process.env.NEXT_PUBLIC_API_URL
  const moduleUrl = pathToFileURL(path.join(appDir, "next.config.mjs")).href
  const mod = await import(moduleUrl)
  return mod.default
}

describe("frontend dev config", () => {
  it("allows localhost loopback dev origins", async () => {
    const nextConfig = await loadNextConfig()

    expect(nextConfig.allowedDevOrigins).toEqual(
      expect.arrayContaining(["localhost", "127.0.0.1", "[::1]"])
    )
  })

  it("provides a webpack dev fallback script", () => {
    const packageJson = loadPackageJson()

    expect(packageJson.scripts?.["dev:webpack"]).toBe("next dev --webpack")
  })

  it("routes Turbopack build entrypoints through the profile wrapper", () => {
    const packageJson = loadPackageJson()

    expect(packageJson.scripts?.build).toBe(turbopackBuildScript)
    expect(packageJson.scripts?.["build:prod"]).toBe(
      `cross-env TLDW_BUILD_PROFILE=production ${turbopackBuildScript}`
    )
    expect(packageJson.scripts?.["build:dev"]).toBe(
      `cross-env TLDW_BUILD_PROFILE=development ${turbopackBuildScript}`
    )
  })

  it("routes webpack compile entrypoints through the same profile wrapper", () => {
    const packageJson = loadPackageJson()

    expect(packageJson.scripts?.compile).toBe(webpackBuildScript)
    expect(packageJson.scripts?.["compile:prod"]).toBe(
      `cross-env TLDW_BUILD_PROFILE=production ${webpackBuildScript}`
    )
    expect(packageJson.scripts?.["compile:dev"]).toBe(
      `cross-env TLDW_BUILD_PROFILE=development ${webpackBuildScript}`
    )
  })

  it("resolves shared ui aliases to the sibling workspace package", async () => {
    const nextConfig = await loadNextConfig()
    const expectedSharedUiSrc = path.resolve(appDir, "../packages/ui/src")

    expect(existsSync(expectedSharedUiSrc)).toBe(true)
    expect(
      path.resolve(appDir, nextConfig.turbopack.resolveAlias["@tldw/ui"])
    ).toBe(expectedSharedUiSrc)

    const webpackConfig = nextConfig.webpack({ resolve: { alias: {} } })

    expect(webpackConfig.resolve.alias["@tldw/ui"]).toBe(expectedSharedUiSrc)
    expect(webpackConfig.resolve.alias["@"]).toBe(expectedSharedUiSrc)
    expect(webpackConfig.resolve.alias["~"]).toBe(expectedSharedUiSrc)
  })
})
