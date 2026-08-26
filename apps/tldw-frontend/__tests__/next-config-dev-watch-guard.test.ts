import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
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
  const webpack = (
    createRequire(path.resolve(__dirname, "../package.json"))(
      "next/dist/compiled/webpack/webpack",
    ) as { webpack: { validate(config: unknown): void } }
  ).webpack
  const workspaceRoot = path.resolve(__dirname, "../../..")
  const normalizePath = (value: string) => value.split(path.sep).join("/")
  const runtimePaths = [
    "Databases/user_databases/1/Media_DB_v2.db",
    "tldw_Server_API/Databases/runtime.db",
    "tldw_Server_API/Logs/server.log",
    "logs/runtime.log",
  ].map((runtimePath) =>
    normalizePath(path.resolve(workspaceRoot, runtimePath)),
  )
  const backendPatterns = [
    "Databases/**",
    "tldw_Server_API/Databases/**",
    "tldw_Server_API/Logs/**",
    "logs/**",
  ].map((pattern) => normalizePath(path.resolve(workspaceRoot, pattern)))
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

  it("uses the qualified Turbopack runtime while preserving an explicit webpack fallback", () => {
    expect(packageJson.scripts?.dev).toBe("next dev")
    expect(packageJson.scripts?.["dev:webpack"]).toBe("next dev --webpack")
    expect(packageJson.scripts?.["dev:turbopack"]).toBe("next dev")
    expect(packageJson.scripts?.["dev:webpack"]).not.toBe(
      packageJson.scripts?.["dev:turbopack"],
    )
  })

  it("ignores backend runtime output for webpack dev fallback", () => {
    expect(configSource).toContain("backendRuntimeWatchIgnorePatterns")
    expect(configSource).toContain("config.watchOptions")
    expect(configSource).toContain("ignored")
  })

  it("combines a standalone RegExp with backend runtime roots", async () => {
    const nextConfig = await loadNextConfig()
    const existingRegex = /node_modules/
    const webpackConfig = nextConfig.webpack({
      resolve: { alias: {} },
      watchOptions: {
        ignored: existingRegex,
      },
    })
    const ignored = webpackConfig.watchOptions.ignored

    expect(() => webpack.validate({ watchOptions: { ignored } })).not.toThrow()
    expect(ignored).toBeInstanceOf(RegExp)
    const nodeModulesPath = normalizePath(
      path.resolve(workspaceRoot, "node_modules/webpack"),
    )
    expect(ignored.test(nodeModulesPath)).toBe(true)
    for (const runtimePath of runtimePaths) {
      expect(ignored.test(runtimePath)).toBe(true)
    }
  })

  it("removes stateful RegExp flags while preserving other flags", async () => {
    const nextConfig = await loadNextConfig()
    const webpackConfig = nextConfig.webpack({
      resolve: { alias: {} },
      watchOptions: { ignored: /node_modules/gim },
    })
    const ignored = webpackConfig.watchOptions.ignored
    const nodeModulesPath = normalizePath(
      path.resolve(workspaceRoot, "node_modules/webpack"),
    )

    expect(() => webpack.validate({ watchOptions: { ignored } })).not.toThrow()
    expect(ignored.flags).toBe("im")
    expect(ignored.test(nodeModulesPath)).toBe(true)
    expect(ignored.test(nodeModulesPath)).toBe(true)
  })

  it("preserves string-array ignores and appends backend runtime globs", async () => {
    const nextConfig = await loadNextConfig()
    const existingIgnored = ["**/node_modules/**", "**/.next/**"]
    const webpackConfig = nextConfig.webpack({
      resolve: { alias: {} },
      watchOptions: { ignored: existingIgnored },
    })
    const ignored = webpackConfig.watchOptions.ignored

    expect(ignored).toEqual([...existingIgnored, ...backendPatterns])
    expect(backendPatterns.every((pattern) => path.isAbsolute(pattern))).toBe(true)
    expect(backendPatterns.every((pattern) => !pattern.includes("\\"))).toBe(true)
    expect(() => webpack.validate({ watchOptions: { ignored } })).not.toThrow()
  })

  it("filters unsupported values from ignored arrays", async () => {
    const nextConfig = await loadNextConfig()
    const existingGlob = "**/.next/**"
    const webpackConfig = nextConfig.webpack({
      resolve: { alias: {} },
      watchOptions: {
        ignored: [
          existingGlob,
          /node_modules/,
          () => true,
          { test: true },
          "",
          "   ",
        ],
      },
    })
    const ignored = webpackConfig.watchOptions.ignored

    expect(ignored).toEqual([existingGlob, ...backendPatterns])
    expect(() => webpack.validate({ watchOptions: { ignored } })).not.toThrow()
  })

  it("normalizes a standalone string ignore to a string array", async () => {
    const nextConfig = await loadNextConfig()
    const webpackConfig = nextConfig.webpack({
      resolve: { alias: {} },
      watchOptions: { ignored: "**/.next/**" },
    })
    const ignored = webpackConfig.watchOptions.ignored

    expect(ignored).toEqual(["**/.next/**", ...backendPatterns])
    expect(() => webpack.validate({ watchOptions: { ignored } })).not.toThrow()
  })
})
