import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it, vi } from "vitest"

const originalEnv = { ...process.env }

afterEach(() => {
  process.env = { ...originalEnv }
  vi.resetModules()
  vi.restoreAllMocks()
})

describe("zipWithProfile", () => {
  it("detects overwritten zip artifacts when WXT rewrites an existing archive", async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-zip-with-profile-"))
    const buildDir = path.join(tempRoot, "build")
    const zipPath = path.join(buildDir, "tldw-assistant-chrome.zip")
    fs.mkdirSync(buildDir, { recursive: true })
    fs.writeFileSync(zipPath, "before", "utf8")
    const originalMtime = fs.statSync(zipPath).mtimeMs
    const nextTimestamp = new Date(originalMtime + 5_000)

    const { detectCreatedZipArtifact } = await import("../../scripts/zip-with-profile.mjs")

    fs.writeFileSync(zipPath, "after", "utf8")
    fs.utimesSync(zipPath, nextTimestamp, nextTimestamp)

    const createdZip = detectCreatedZipArtifact(
      [path.join(tempRoot, ".output"), buildDir],
      new Map([[zipPath, originalMtime]])
    )

    expect(createdZip).toBe(zipPath)
  })

  it("searches for zip artifacts under the provided cwd", async () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-zip-with-profile-"))
    const buildDir = path.join(tempRoot, "build")
    const zipPath = path.join(buildDir, "tldw-assistant-chrome.zip")
    fs.mkdirSync(buildDir, { recursive: true })

    const execFileSync = vi.fn(() => {
      fs.writeFileSync(zipPath, "after", "utf8")
    })

    vi.doMock("node:child_process", () => ({
      execFileSync,
    }))
    vi.doMock("../../scripts/resolve-build-profile.mjs", () => ({
      getCurrentGitBranch: () => "dev",
      resolveBuildProfile: () => "production",
    }))

    const { zipWithProfile } = await import("../../scripts/zip-with-profile.mjs")

    const result = zipWithProfile({
      argv: ["--browser=chrome"],
      cwd: tempRoot,
      env: {
        ...process.env,
        TLDW_BUILD_PROFILE: "production",
      },
    })

    expect(result).toBe(zipPath)
    expect(execFileSync).toHaveBeenCalledWith(
      expect.any(String),
      ["zip"],
      expect.objectContaining({
        cwd: tempRoot,
        env: expect.objectContaining({
          TARGET: "chrome",
        }),
        stdio: "inherit",
      }),
    )
  })
})
