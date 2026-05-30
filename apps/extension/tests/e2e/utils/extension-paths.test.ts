import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it } from "vitest"

import {
  prepareExtensionLaunchPath,
  prioritizeExtensionBuildCandidates
} from "./extension-paths"

let tempRoots: string[] = []

afterEach(() => {
  for (const tempRoot of tempRoots) {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
  tempRoots = []
})

describe("prioritizeExtensionBuildCandidates", () => {
  it("prefers .output/chrome-mv3 over build/chrome-mv3", () => {
    const repoRoot = path.resolve("/tmp/tldw-extension")
    const buildPath = path.join(repoRoot, "build", "chrome-mv3")
    const outputPath = path.join(repoRoot, ".output", "chrome-mv3")

    expect(
      prioritizeExtensionBuildCandidates([buildPath, outputPath])
    ).toEqual([outputPath, buildPath])
  })

  it("keeps custom extension directories ahead of standard build outputs", () => {
    const repoRoot = path.resolve("/tmp/tldw-extension")
    const customPath = path.join(repoRoot, "fixtures", "packed-extension")
    const buildPath = path.join(repoRoot, "build", "chrome-mv3")
    const outputPath = path.join(repoRoot, ".output", "chrome-mv3")

    expect(
      prioritizeExtensionBuildCandidates([customPath, buildPath, outputPath])
    ).toEqual([customPath, outputPath, buildPath])
  })

  it("can stage a Chrome smoke copy with only a minimal default locale", () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-paths-"))
    tempRoots.push(tempRoot)
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(path.join(extensionDir, "_locales", "en"), { recursive: true })
    fs.mkdirSync(path.join(extensionDir, "_locales", "de"), { recursive: true })
    fs.mkdirSync(path.join(extensionDir, "chunks"), { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, default_locale: "en" }),
      "utf8"
    )
    fs.writeFileSync(path.join(extensionDir, "options.html"), "<html></html>", "utf8")
    fs.writeFileSync(
      path.join(extensionDir, "_locales", "en", "messages.json"),
      JSON.stringify({ appName: { message: "tldw Assistant" } }),
      "utf8"
    )
    fs.writeFileSync(
      path.join(extensionDir, "_locales", "de", "messages.json"),
      JSON.stringify({ appName: { message: "tldw Assistent" } }),
      "utf8"
    )
    fs.writeFileSync(path.join(extensionDir, "chunks", "page.js"), "export {}", "utf8")

    const stagedPath = prepareExtensionLaunchPath(extensionDir, {
      minimalLocales: true,
      rootDir: path.join(tempRoot, "staged")
    })

    expect(stagedPath).not.toBe(extensionDir)
    expect(fs.existsSync(path.join(stagedPath, "manifest.json"))).toBe(true)
    expect(fs.existsSync(path.join(stagedPath, "options.html"))).toBe(true)
    expect(fs.existsSync(path.join(stagedPath, "chunks", "page.js"))).toBe(true)
    expect(
      fs.existsSync(path.join(stagedPath, "_locales", "de", "messages.json"))
    ).toBe(false)
    expect(
      JSON.parse(
        fs.readFileSync(
          path.join(stagedPath, "_locales", "en", "messages.json"),
          "utf8"
        )
      )
    ).toEqual({})
  })
})
