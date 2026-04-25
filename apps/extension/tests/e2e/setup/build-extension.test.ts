import fs from "node:fs"
import path from "node:path"

import { afterEach, describe, expect, it, vi } from "vitest"

const realReadFileSync = fs.readFileSync
const realExistsSync = fs.existsSync
const realStatSync = fs.statSync

afterEach(() => {
  vi.resetModules()
  vi.restoreAllMocks()
})

describe("extension playwright globalSetup", () => {
  it("skips rebuilding when a non-dev .output chrome build already exists", async () => {
    const execSync = vi.fn()
    vi.doMock("node:child_process", () => ({
      execSync
    }))

    const moduleUrl = new URL(import.meta.url)
    const setupDir = path.dirname(moduleUrl.pathname)
    const projectRoot = path.resolve(setupDir, "..", "..", "..")
    const outputChromePath = path.resolve(projectRoot, ".output/chrome-mv3")
    const outputManifestPath = path.join(outputChromePath, "manifest.json")
    const outputBackgroundPath = path.join(outputChromePath, "background.js")
    const outputOptionsPath = path.join(outputChromePath, "options.html")

    vi.spyOn(fs, "existsSync").mockImplementation((targetPath) => {
      const normalized = String(targetPath)
      if (normalized.startsWith(path.resolve(projectRoot, "build/chrome-mv3"))) {
        return false
      }
      if (normalized === outputChromePath) return true
      if (normalized === outputManifestPath) return true
      if (normalized === outputBackgroundPath) return true
      if (normalized === outputOptionsPath) return true
      return realExistsSync(targetPath)
    })

    vi.spyOn(fs, "readFileSync").mockImplementation((targetPath, options) => {
      if (String(targetPath) === outputOptionsPath) {
        return "<html><body>production build</body></html>" as any
      }
      return realReadFileSync(targetPath as any, options as any)
    })

    vi.spyOn(fs, "statSync").mockImplementation((targetPath, options) => {
      if (String(targetPath) === outputManifestPath) {
        return {
          mtimeMs: Date.now()
        } as fs.Stats
      }
      return realStatSync(targetPath as any, options as any)
    })

    vi.spyOn(fs, "readdirSync").mockImplementation(() => [])

    const globalSetup = (await import("./build-extension")).default

    await globalSetup()

    expect(execSync).not.toHaveBeenCalled()
  })

  it("prefers a valid .output build over an invalid build/chrome-mv3 directory", async () => {
    const execSync = vi.fn()
    vi.doMock("node:child_process", () => ({
      execSync
    }))

    const moduleUrl = new URL(import.meta.url)
    const setupDir = path.dirname(moduleUrl.pathname)
    const projectRoot = path.resolve(setupDir, "..", "..", "..")
    const buildChromePath = path.resolve(projectRoot, "build/chrome-mv3")
    const buildManifestPath = path.join(buildChromePath, "manifest.json")
    const buildBackgroundPath = path.join(buildChromePath, "background.js")
    const buildOptionsPath = path.join(buildChromePath, "options.html")
    const buildSidepanelPath = path.join(buildChromePath, "sidepanel.html")
    const outputChromePath = path.resolve(projectRoot, ".output/chrome-mv3")
    const outputManifestPath = path.join(outputChromePath, "manifest.json")
    const outputBackgroundPath = path.join(outputChromePath, "background.js")
    const outputOptionsPath = path.join(outputChromePath, "options.html")

    vi.spyOn(fs, "existsSync").mockImplementation((targetPath) => {
      const normalized = String(targetPath)
      if (normalized === buildChromePath) return true
      if (normalized === buildManifestPath) return false
      if (normalized === buildBackgroundPath) return false
      if (normalized === buildOptionsPath) return false
      if (normalized === buildSidepanelPath) return false
      if (normalized === outputChromePath) return true
      if (normalized === outputManifestPath) return true
      if (normalized === outputBackgroundPath) return true
      if (normalized === outputOptionsPath) return true
      return realExistsSync(targetPath)
    })

    vi.spyOn(fs, "readFileSync").mockImplementation((targetPath, options) => {
      if (String(targetPath) === outputOptionsPath) {
        return "<html><body>production build</body></html>" as any
      }
      return realReadFileSync(targetPath as any, options as any)
    })

    vi.spyOn(fs, "statSync").mockImplementation((targetPath, options) => {
      if (String(targetPath) === outputManifestPath) {
        return {
          mtimeMs: Date.now()
        } as fs.Stats
      }
      return realStatSync(targetPath as any, options as any)
    })

    vi.spyOn(fs, "readdirSync").mockImplementation(() => [])

    const globalSetup = (await import("./build-extension")).default

    await globalSetup()

    expect(execSync).not.toHaveBeenCalled()
  })

  it("rebuilds with the explicit production chrome script when no valid build exists", async () => {
    const execSync = vi.fn()
    vi.doMock("node:child_process", () => ({
      execSync
    }))

    vi.spyOn(fs, "existsSync").mockReturnValue(false)
    vi.spyOn(fs, "readdirSync").mockImplementation(() => [])

    const globalSetup = (await import("./build-extension")).default

    await globalSetup()

    expect(execSync).toHaveBeenCalledWith("npm run build:chrome:prod", {
      cwd: expect.any(String),
      stdio: "inherit"
    })
  })

  it("falls back to the direct production wrapper command when package-manager scripts fail", async () => {
    const execSync = vi
      .fn()
      .mockImplementationOnce(() => {
        throw new Error("npm missing")
      })
      .mockImplementationOnce(() => {
        throw new Error("bun missing")
      })
      .mockImplementation(() => undefined)

    vi.doMock("node:child_process", () => ({
      execSync
    }))

    vi.spyOn(fs, "existsSync").mockReturnValue(false)
    vi.spyOn(fs, "readdirSync").mockImplementation(() => [])

    const globalSetup = (await import("./build-extension")).default

    await globalSetup()

    expect(execSync).toHaveBeenNthCalledWith(1, "npm run build:chrome:prod", {
      cwd: expect.any(String),
      stdio: "inherit"
    })
    expect(execSync).toHaveBeenNthCalledWith(2, "bun run build:chrome:prod", {
      cwd: expect.any(String),
      stdio: "inherit"
    })
    expect(execSync).toHaveBeenNthCalledWith(
      3,
      "node scripts/build-with-profile.mjs --browser=chrome",
      {
        cwd: expect.any(String),
        env: expect.objectContaining({
          TLDW_BUILD_PROFILE: "production"
        }),
        stdio: "inherit"
      }
    )
  })

  it("surfaces every failed build attempt when all build commands fail", async () => {
    const execSync = vi
      .fn()
      .mockImplementationOnce(() => {
        throw new Error("npm missing")
      })
      .mockImplementationOnce(() => {
        throw new Error("bun missing")
      })
      .mockImplementationOnce(() => {
        throw new Error("wrapper failed")
      })

    vi.doMock("node:child_process", () => ({
      execSync
    }))

    vi.spyOn(fs, "existsSync").mockReturnValue(false)
    vi.spyOn(fs, "readdirSync").mockImplementation(() => [])

    const globalSetup = (await import("./build-extension")).default

    try {
      await globalSetup()
      throw new Error("expected globalSetup to fail")
    } catch (error) {
      expect(error).toBeInstanceOf(AggregateError)
      const aggregate = error as AggregateError
      expect(aggregate.message).toBe("All extension build attempts failed.")
      expect(aggregate.errors).toHaveLength(3)
      expect(String(aggregate.errors[0])).toContain("npm run build:chrome:prod")
      expect(String(aggregate.errors[1])).toContain("bun run build:chrome:prod")
      expect(String(aggregate.errors[2])).toContain(
        "node scripts/build-with-profile.mjs --browser=chrome"
      )
    }
  })
})
