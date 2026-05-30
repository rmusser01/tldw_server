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

describe("launchWithExtension", () => {
  it("passes userDataDir to resolveExtensionId when extension targets are unavailable", async () => {
    process.env.TLDW_E2E_EXTENSION_TARGET_WAIT_MS = "1"

    const extensionId = "a".repeat(32)
    const resolveExtensionId = vi.fn().mockResolvedValue(extensionId)

    const page = {
      waitForTimeout: vi.fn().mockResolvedValue(undefined),
      goto: vi.fn().mockResolvedValue(undefined),
      waitForFunction: vi.fn().mockResolvedValue(undefined),
    }

    const context = {
      serviceWorkers: vi.fn(() => []),
      backgroundPages: vi.fn(() => []),
      waitForEvent: vi.fn(() => new Promise(() => {})),
      addInitScript: vi.fn().mockResolvedValue(undefined),
      newPage: vi.fn().mockResolvedValue(page),
    }

    const launchPersistentContext = vi.fn().mockResolvedValue(context)

    vi.doMock("@playwright/test", () => ({
      BrowserContext: class BrowserContext {},
      Page: class Page {},
      chromium: {
        launchPersistentContext,
      },
    }))

    vi.doMock("./extension-id", () => ({
      resolveExtensionId,
    }))

    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-launch-"))
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(extensionDir, { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, name: "Test Extension", version: "1.0.0" }),
      "utf8",
    )
    fs.writeFileSync(path.join(extensionDir, "background.js"), "// background", "utf8")
    fs.writeFileSync(path.join(extensionDir, "options.html"), "<html></html>", "utf8")
    fs.writeFileSync(path.join(extensionDir, "sidepanel.html"), "<html></html>", "utf8")

    try {
      const { launchWithExtension } = await import("./extension")

      const result = await launchWithExtension(extensionDir)

      expect(resolveExtensionId).toHaveBeenCalledTimes(1)
      expect(resolveExtensionId).toHaveBeenCalledWith(
        context,
        expect.objectContaining({
          userDataDir: expect.stringContaining("tmp-playwright-profile/user-data-"),
        }),
      )
      expect(result.extensionId).toBe(extensionId)
      expect(result.optionsUrl).toBe(`chrome-extension://${extensionId}/options.html`)
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  })

  it("uses CI chromium channel and env launch timeout when no explicit timeout is passed", async () => {
    process.env.CI = "true"
    process.env.TLDW_E2E_EXTENSION_TARGET_WAIT_MS = "1"
    process.env.TLDW_E2E_EXTENSION_LAUNCH_TIMEOUT_MS = "90000"

    const resolveExtensionId = vi.fn().mockResolvedValue("b".repeat(32))
    const page = {
      waitForTimeout: vi.fn().mockResolvedValue(undefined),
      goto: vi.fn().mockResolvedValue(undefined),
      waitForFunction: vi.fn().mockResolvedValue(undefined),
    }
    const context = {
      serviceWorkers: vi.fn(() => []),
      backgroundPages: vi.fn(() => []),
      waitForEvent: vi.fn(() => new Promise(() => {})),
      addInitScript: vi.fn().mockResolvedValue(undefined),
      newPage: vi.fn().mockResolvedValue(page),
    }
    const launchPersistentContext = vi.fn().mockResolvedValue(context)

    vi.doMock("@playwright/test", () => ({
      BrowserContext: class BrowserContext {},
      Page: class Page {},
      chromium: {
        launchPersistentContext,
      },
    }))

    vi.doMock("./extension-id", () => ({
      resolveExtensionId,
    }))

    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-launch-"))
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(extensionDir, { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, name: "Test Extension", version: "1.0.0" }),
      "utf8",
    )
    fs.writeFileSync(path.join(extensionDir, "background.js"), "// background", "utf8")
    fs.writeFileSync(path.join(extensionDir, "options.html"), "<html></html>", "utf8")
    fs.writeFileSync(path.join(extensionDir, "sidepanel.html"), "<html></html>", "utf8")

    try {
      const { launchWithExtension } = await import("./extension")

      await launchWithExtension(extensionDir)

      expect(launchPersistentContext).toHaveBeenCalledWith(
        expect.stringContaining("tmp-playwright-profile/user-data-"),
        expect.objectContaining({
          timeout: 90000,
          channel: "chromium",
          headless: true,
          ignoreDefaultArgs: ["--disable-extensions"],
        }),
      )
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  })

  it("allows CI extension launches to opt into headed mode via env override", async () => {
    process.env.CI = "true"
    process.env.TLDW_E2E_EXTENSION_HEADLESS = "0"
    process.env.TLDW_E2E_EXTENSION_TARGET_WAIT_MS = "1"

    const resolveExtensionId = vi.fn().mockResolvedValue("c".repeat(32))
    const page = {
      waitForTimeout: vi.fn().mockResolvedValue(undefined),
      goto: vi.fn().mockResolvedValue(undefined),
      waitForFunction: vi.fn().mockResolvedValue(undefined),
    }
    const context = {
      serviceWorkers: vi.fn(() => []),
      backgroundPages: vi.fn(() => []),
      waitForEvent: vi.fn(() => new Promise(() => {})),
      addInitScript: vi.fn().mockResolvedValue(undefined),
      newPage: vi.fn().mockResolvedValue(page),
    }
    const launchPersistentContext = vi.fn().mockResolvedValue(context)

    vi.doMock("@playwright/test", () => ({
      BrowserContext: class BrowserContext {},
      Page: class Page {},
      chromium: {
        launchPersistentContext,
      },
    }))

    vi.doMock("./extension-id", () => ({
      resolveExtensionId,
    }))

    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-launch-"))
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(extensionDir, { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, name: "Test Extension", version: "1.0.0" }),
      "utf8",
    )
    fs.writeFileSync(path.join(extensionDir, "background.js"), "// background", "utf8")
    fs.writeFileSync(path.join(extensionDir, "options.html"), "<html></html>", "utf8")
    fs.writeFileSync(path.join(extensionDir, "sidepanel.html"), "<html></html>", "utf8")

    try {
      const { launchWithExtension } = await import("./extension")

      await launchWithExtension(extensionDir)

      expect(launchPersistentContext).toHaveBeenCalledWith(
        expect.stringContaining("tmp-playwright-profile/user-data-"),
        expect.objectContaining({
          headless: false,
          channel: "chromium",
        }),
      )
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  })

  it("opens an explicit sidepanel target when requested", async () => {
    process.env.TLDW_E2E_EXTENSION_TARGET_WAIT_MS = "1"

    const resolveExtensionId = vi.fn().mockResolvedValue("d".repeat(32))
    const page = {
      waitForTimeout: vi.fn().mockResolvedValue(undefined),
      goto: vi.fn().mockResolvedValue(undefined),
      waitForFunction: vi.fn().mockResolvedValue(undefined),
      locator: vi.fn(() => ({
        waitFor: vi.fn().mockResolvedValue(undefined),
      })),
      bringToFront: vi.fn().mockResolvedValue(undefined),
    }
    const context = {
      serviceWorkers: vi.fn(() => []),
      backgroundPages: vi.fn(() => []),
      waitForEvent: vi.fn(() => new Promise(() => {})),
      addInitScript: vi.fn().mockResolvedValue(undefined),
      newPage: vi.fn().mockResolvedValue(page),
    }
    const launchPersistentContext = vi.fn().mockResolvedValue(context)

    vi.doMock("@playwright/test", () => ({
      BrowserContext: class BrowserContext {},
      Page: class Page {},
      chromium: {
        launchPersistentContext,
      },
    }))

    vi.doMock("./extension-id", () => ({
      resolveExtensionId,
    }))

    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-extension-launch-"))
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(extensionDir, { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, name: "Test Extension", version: "1.0.0" }),
      "utf8",
    )
    fs.writeFileSync(path.join(extensionDir, "background.js"), "// background", "utf8")
    fs.writeFileSync(path.join(extensionDir, "options.html"), "<html></html>", "utf8")
    fs.writeFileSync(path.join(extensionDir, "sidepanel.html"), "<html></html>", "utf8")

    try {
      const { launchWithExtension } = await import("./extension")

      const result = await launchWithExtension(extensionDir)
      await result.openSidepanel("?view=chat")

      expect(page.goto).toHaveBeenCalledWith(
        `chrome-extension://${"d".repeat(32)}/sidepanel.html?view=chat`,
        { waitUntil: "domcontentloaded" }
      )
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  })
})
