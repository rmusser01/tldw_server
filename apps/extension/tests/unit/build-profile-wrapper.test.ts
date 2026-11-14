import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it, vi } from "vitest"

import {
  getExportedArtifactDir,
  getExportedZipName,
} from "../../scripts/build-with-profile.mjs"
import {
  finalizeZipArtifact,
  getZipArtifactSearchRoots,
} from "../../scripts/zip-with-profile.mjs"

const appDir = path.resolve(__dirname, "..", "..")

const loadPackageJson = () =>
  JSON.parse(fs.readFileSync(path.join(appDir, "package.json"), "utf8")) as {
    scripts?: Record<string, string>
  }

describe("extension build profile wrapper", () => {
  it("keeps production exported install directories unsuffixed", () => {
    expect(getExportedArtifactDir("chrome-mv3", "production")).toBe(
      path.join("build", "chrome-mv3")
    )
  })

  it("exports suffixed dev install directories without renaming canonical internal roots", () => {
    expect(getExportedArtifactDir("chrome-mv3", "development")).toBe(
      path.join("build", "chrome-mv3-dev")
    )
  })

  it("suffixes dev archives", () => {
    expect(getExportedZipName("chrome", "development")).toContain("-dev")
  })

  it("renames dev archives instead of leaving an unsuffixed copy behind", () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-zip-profile-"))
    const createdZip = path.join(tempRoot, "tldw-assistant-0.1.0-chrome.zip")

    try {
      fs.writeFileSync(createdZip, "zip")

      const finalizedZip = finalizeZipArtifact(createdZip, "development")

      expect(path.basename(finalizedZip)).toBe("tldw-assistant-0.1.0-chrome-dev.zip")
      expect(fs.existsSync(finalizedZip)).toBe(true)
      expect(fs.existsSync(createdZip)).toBe(false)
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  })

  it("searches WXT zip artifacts under the canonical .output root first", () => {
    expect(getZipArtifactSearchRoots(appDir)).toEqual([
      path.join(appDir, ".output"),
      path.join(appDir, "build"),
    ])
  })

  it("routes browser-specific build scripts through the profile wrapper", () => {
    const packageJson = loadPackageJson()

    expect(packageJson.scripts?.["build:prod"]).toBe(
      "bun run locales:sync && bun run build:chrome:prod && bun run build:firefox:prod && bun run build:edge:prod"
    )
    expect(packageJson.scripts?.["build:dev"]).toBe(
      "bun run locales:sync && bun run build:chrome:dev && bun run build:firefox:dev && bun run build:edge:dev"
    )
    expect(packageJson.scripts?.["build:chrome"]).toBe(
      "node scripts/build-with-profile.mjs --browser=chrome"
    )
    expect(packageJson.scripts?.["build:chrome:prod"]).toBe(
      "cross-env TLDW_BUILD_PROFILE=production node scripts/build-with-profile.mjs --browser=chrome"
    )
    expect(packageJson.scripts?.["build:chrome:dev"]).toBe(
      "cross-env TLDW_BUILD_PROFILE=development node scripts/build-with-profile.mjs --browser=chrome"
    )
    expect(packageJson.scripts?.["build:firefox"]).toBe(
      "node scripts/build-with-profile.mjs --browser=firefox"
    )
    expect(packageJson.scripts?.["build:firefox:prod"]).toBe(
      "cross-env TLDW_BUILD_PROFILE=production node scripts/build-with-profile.mjs --browser=firefox"
    )
    expect(packageJson.scripts?.["build:firefox:dev"]).toBe(
      "cross-env TLDW_BUILD_PROFILE=development node scripts/build-with-profile.mjs --browser=firefox"
    )
    expect(packageJson.scripts?.["build:edge"]).toBe(
      "node scripts/build-with-profile.mjs --browser=edge"
    )
    expect(packageJson.scripts?.["build:edge:prod"]).toBe(
      "cross-env TLDW_BUILD_PROFILE=production node scripts/build-with-profile.mjs --browser=edge"
    )
    expect(packageJson.scripts?.["build:edge:dev"]).toBe(
      "cross-env TLDW_BUILD_PROFILE=development node scripts/build-with-profile.mjs --browser=edge"
    )
  })

  it("routes archive scripts through the profile wrapper", () => {
    const packageJson = loadPackageJson()

    expect(packageJson.scripts?.zip).toBe(
      "bun run locales:sync && node scripts/zip-with-profile.mjs --browser=chrome"
    )
    expect(packageJson.scripts?.["zip:prod"]).toBe(
      "bun run locales:sync && cross-env TLDW_BUILD_PROFILE=production node scripts/zip-with-profile.mjs --browser=chrome"
    )
    expect(packageJson.scripts?.["zip:dev"]).toBe(
      "bun run locales:sync && cross-env TLDW_BUILD_PROFILE=development node scripts/zip-with-profile.mjs --browser=chrome"
    )
    expect(packageJson.scripts?.["zip:firefox"]).toBe(
      "bun run locales:sync && node scripts/zip-with-profile.mjs --browser=firefox"
    )
    expect(packageJson.scripts?.["zip:firefox:prod"]).toBe(
      "bun run locales:sync && cross-env TLDW_BUILD_PROFILE=production node scripts/zip-with-profile.mjs --browser=firefox"
    )
    expect(packageJson.scripts?.["zip:firefox:dev"]).toBe(
      "bun run locales:sync && cross-env TLDW_BUILD_PROFILE=development node scripts/zip-with-profile.mjs --browser=firefox"
    )
  })
})


afterEach(() => {
  vi.doUnmock("node:child_process")
  vi.resetModules()
  vi.restoreAllMocks()
})

describe.each(["build", "zip"] as const)("%s child heap policy", (mode) => {
  it.each([undefined, "--max-old-space-size=8192", ""])(
    "defaults only absent options and preserves parent env (%s)",
    async (nodeOptions) => {
      const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-build-env-"))
      const stopped = new Error("captured WXT child")
      const execFileSync = vi.fn((command: string) => {
        if (command === "git") return "dev\n"
        throw stopped
      })
      vi.doMock("node:child_process", () => ({ execFileSync }))
      const run = mode === "build"
        ? (await import("../../scripts/build-with-profile.mjs")).buildWithProfile
        : (await import("../../scripts/zip-with-profile.mjs")).zipWithProfile
      const env: NodeJS.ProcessEnv = { TLDW_BUILD_PROFILE: "production" }
      if (nodeOptions !== undefined) env.NODE_OPTIONS = nodeOptions
      const parentOptions = process.env.NODE_OPTIONS
      try {
        expect(() => run({ argv: ["--browser=chrome"], cwd: tempRoot, env })).toThrow(stopped)
        expect(execFileSync).toHaveBeenLastCalledWith(
          process.platform === "win32" ? "wxt.cmd" : "wxt",
          [mode],
          expect.objectContaining({
            cwd: tempRoot,
            stdio: "inherit",
            env: {
              ...env,
              TARGET: "chrome",
              NODE_OPTIONS: nodeOptions ?? "--max-old-space-size=6144",
            },
          })
        )
        expect(env).toEqual({
          TLDW_BUILD_PROFILE: "production",
          ...(nodeOptions === undefined ? {} : { NODE_OPTIONS: nodeOptions }),
        })
        expect(process.env.NODE_OPTIONS).toBe(parentOptions)
      } finally {
        fs.rmSync(tempRoot, { recursive: true, force: true })
      }
    }
  )
})
