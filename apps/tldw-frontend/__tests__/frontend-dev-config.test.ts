import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, describe, expect, it, vi } from "vitest"

const appDir = path.resolve(__dirname, "..")

const loadPackageJson = () =>
  JSON.parse(readFileSync(path.join(appDir, "package.json"), "utf8")) as {
    scripts?: Record<string, string>
  }

const buildSteps = (bundler: "turbopack" | "webpack", profile?: string) => [
  `${profile ? `cross-env TLDW_BUILD_PROFILE=${profile} ` : ""}node scripts/build-with-profile.mjs --bundler=${bundler}`,
  "node scripts/verify-shared-token-sync.mjs --dir .next",
  "node scripts/check-bundle-budget.mjs --dir .next",
]

const loadNextConfig = async () => {
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
  vi.stubEnv("TLDW_INTERNAL_API_ORIGIN", "http://app:8000")
  vi.stubEnv("NEXT_PUBLIC_API_URL", undefined)
  const moduleUrl = pathToFileURL(path.join(appDir, "next.config.mjs")).href
  const mod = await import(moduleUrl)
  return mod.default
}

describe("frontend dev config", () => {
  afterEach(() => vi.unstubAllEnvs())

  it("allows localhost loopback dev origins", async () => {
    const nextConfig = await loadNextConfig()

    expect(nextConfig.allowedDevOrigins).toEqual(
      expect.arrayContaining(["localhost", "127.0.0.1", "[::1]"])
    )
  })

  it("uses the qualified Turbopack dev runtime with an explicit webpack fallback", () => {
    const packageJson = loadPackageJson()

    // Qualified by the live-backend comparison documented in README.md.
    expect(packageJson.scripts?.dev).toBe("next dev")
    expect(packageJson.scripts?.["dev:webpack"]).toBe("next dev --webpack")
    expect(packageJson.scripts?.["dev:turbopack"]).toBe("next dev")
  })

  it("routes Turbopack build entrypoints through the profile wrapper", () => {
    const packageJson = loadPackageJson()

    expect(packageJson.scripts?.build?.split(/\s*&&\s*/)).toEqual(buildSteps("turbopack"))
    expect(packageJson.scripts?.["build:prod"]?.split(/\s*&&\s*/)).toEqual(buildSteps("turbopack", "production"))
    expect(packageJson.scripts?.["build:dev"]?.split(/\s*&&\s*/)).toEqual(buildSteps("turbopack", "development"))
  })

  it("routes webpack compile entrypoints through the same profile wrapper", () => {
    const packageJson = loadPackageJson()

    expect(packageJson.scripts?.compile?.split(/\s*&&\s*/)).toEqual(buildSteps("webpack"))
    expect(packageJson.scripts?.["compile:prod"]?.split(/\s*&&\s*/)).toEqual(buildSteps("webpack", "production"))
    expect(packageJson.scripts?.["compile:dev"]?.split(/\s*&&\s*/)).toEqual(buildSteps("webpack", "development"))
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
