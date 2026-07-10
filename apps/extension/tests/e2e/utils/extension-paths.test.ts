import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it } from "vitest"

import {
  prepareExtensionLaunchPath,
  prioritizeExtensionBuildCandidates
} from "./extension-paths"

const EXPECTED_E2E_MANIFEST_KEY =
  "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAjI1q+ZCGeQEsFkXz8Jcx9BHxpWcxr4egilGW2LKpyDcxbd+2id2k0WtauiWSS+eBfvJWRnonnIjZQ/6jkNbN41z+G6Wp5HzHJaGHB609GO4LWW5kVkPo0h+KkSSEVjoXTyRQZO3ViwDbne3gqHVJmnKGWV+Tz6X2se3GwCah3I0AG2290/E4aweSV6OG/SRD15MCiDTImSCNa7WXhMQtqN61o+b8MGr3t5eN3E2UCKMFYAFH017EuRQ46vn8q29O7ATaEwHnB0U/7g9zyi3OKhCU5bI9XhZNoRH/iZqOajz5vVu4Pbq6Wq0Vu2Y1nHIjOQi4XADuUrd4ZFyQWkDFcwIDAQAB"

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

  it("can stage a Chrome smoke copy with only the real default locale catalog", () => {
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
      deterministicManifestKey: true,
      minimalLocales: true,
      rootDir: path.join(tempRoot, "staged")
    })

    expect(stagedPath).not.toBe(extensionDir)
    const stagedManifest = JSON.parse(
      fs.readFileSync(path.join(stagedPath, "manifest.json"), "utf8")
    )
    expect(stagedManifest.key).toBe(EXPECTED_E2E_MANIFEST_KEY)
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
    ).toEqual({ appName: { message: "tldw Assistant" } })
  })

  it("stages a deterministic manifest key without minimal-locale mode", () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-paths-"))
    tempRoots.push(tempRoot)
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(path.join(extensionDir, "_locales", "en"), { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, default_locale: "en" }),
      "utf8"
    )
    fs.writeFileSync(path.join(extensionDir, "background.js"), "// background", "utf8")
    fs.writeFileSync(
      path.join(extensionDir, "_locales", "en", "messages.json"),
      JSON.stringify({ appName: { message: "tldw Assistant" } }),
      "utf8"
    )

    const stagedPath = prepareExtensionLaunchPath(extensionDir, {
      deterministicManifestKey: true,
      minimalLocales: false,
      rootDir: path.join(tempRoot, "staged")
    })

    expect(stagedPath).not.toBe(extensionDir)
    expect(
      JSON.parse(fs.readFileSync(path.join(stagedPath, "manifest.json"), "utf8")).key
    ).toBe(EXPECTED_E2E_MANIFEST_KEY)
    expect(fs.existsSync(path.join(stagedPath, "background.js"))).toBe(true)
    expect(
      fs.existsSync(path.join(stagedPath, "_locales", "en", "messages.json"))
    ).toBe(true)
  })

  it("does not add a deterministic manifest key unless requested", () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-paths-"))
    tempRoots.push(tempRoot)
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(path.join(extensionDir, "_locales", "en"), { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, default_locale: "en" }),
      "utf8"
    )
    fs.writeFileSync(
      path.join(extensionDir, "_locales", "en", "messages.json"),
      JSON.stringify({ appName: { message: "tldw Assistant" } }),
      "utf8"
    )

    const stagedPath = prepareExtensionLaunchPath(extensionDir, {
      minimalLocales: true,
      rootDir: path.join(tempRoot, "staged")
    })

    const stagedManifest = JSON.parse(
      fs.readFileSync(path.join(stagedPath, "manifest.json"), "utf8")
    )
    expect(stagedManifest.key).toBeUndefined()
  })

  it("uses the manifest default locale catalog when staging minimal locales", () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-paths-"))
    tempRoots.push(tempRoot)
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(path.join(extensionDir, "_locales", "ja"), { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, default_locale: "ja" }),
      "utf8"
    )
    fs.writeFileSync(
      path.join(extensionDir, "_locales", "ja", "messages.json"),
      JSON.stringify({ appName: { message: "tldw Assistant JA" } }),
      "utf8"
    )

    const stagedPath = prepareExtensionLaunchPath(extensionDir, {
      minimalLocales: true,
      rootDir: path.join(tempRoot, "staged")
    })

    expect(
      fs.existsSync(path.join(stagedPath, "_locales", "ja", "messages.json"))
    ).toBe(true)
    expect(
      fs.existsSync(path.join(stagedPath, "_locales", "en", "messages.json"))
    ).toBe(false)
    expect(
      JSON.parse(
        fs.readFileSync(
          path.join(stagedPath, "_locales", "ja", "messages.json"),
          "utf8"
        )
      )
    ).toEqual({ appName: { message: "tldw Assistant JA" } })
  })

  it("can stage a lightweight default locale stub for built-extension launches", () => {
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-paths-"))
    tempRoots.push(tempRoot)
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(path.join(extensionDir, "_locales", "en"), { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, default_locale: "en" }),
      "utf8"
    )
    fs.writeFileSync(
      path.join(extensionDir, "_locales", "en", "messages.json"),
      JSON.stringify({ appName: { message: "tldw Assistant" } }),
      "utf8"
    )

    const stagedPath = prepareExtensionLaunchPath(extensionDir, {
      minimalLocales: true,
      preserveDefaultLocaleCatalog: false,
      rootDir: path.join(tempRoot, "staged")
    })

    expect(
      fs.readFileSync(
        path.join(stagedPath, "_locales", "en", "messages.json"),
        "utf8"
      )
    ).toBe("{}\n")
  })
})
