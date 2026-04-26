import fs from "node:fs"
import os from "node:os"
import path from "node:path"

import { afterEach, describe, expect, it, vi } from "vitest"

import { normalizeBuiltExtensionSeedConfig } from "./extension-build"

const originalEnv = { ...process.env }

afterEach(() => {
  process.env = { ...originalEnv }
  vi.resetModules()
  vi.restoreAllMocks()
})

describe("normalizeBuiltExtensionSeedConfig", () => {
  it("wraps a plain connection config under tldwConfig for built extension storage", () => {
    const seedConfig = {
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    }

    const normalized = normalizeBuiltExtensionSeedConfig(seedConfig)

    expect(normalized.connectionConfig).toEqual(seedConfig)
    expect(normalized.storagePayload).toMatchObject({
      __tldw_first_run_complete: true,
      tldw_skip_landing_hub: true,
      quickIngestInspectorIntroDismissed: true,
      quickIngestOnboardingDismissed: true,
      tldwConfig: seedConfig,
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "test-key"
    })
  })

  it("preserves a full seeded storage payload without nesting it again", () => {
    const seedConfig = {
      __tldw_first_run_complete: true,
      tldw_skip_landing_hub: true,
      quickIngestInspectorIntroDismissed: true,
      quickIngestOnboardingDismissed: true,
      "tldw:workflow:landing-config": {
        showOnFirstRun: true,
        dismissedAt: 123,
        completedWorkflows: []
      },
      tldwConfig: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      }
    }

    const normalized = normalizeBuiltExtensionSeedConfig(seedConfig)

    expect(normalized.connectionConfig).toEqual(seedConfig.tldwConfig)
    expect(normalized.storagePayload).toMatchObject({
      tldwConfig: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      }
    })
    expect(normalized.storagePayload.tldwConfig).not.toHaveProperty("tldwConfig")
  })

  it("launches built extensions with crashpad-disabled Chromium options", async () => {
    process.env.CI = "true"
    process.env.TLDW_E2E_EXTENSION_TARGET_WAIT_MS = "1"

    const resolveExtensionId = vi.fn().mockResolvedValue("e".repeat(32))
    const page = {
      waitForTimeout: vi.fn().mockResolvedValue(undefined),
      goto: vi.fn().mockResolvedValue(undefined),
      waitForFunction: vi.fn().mockResolvedValue(undefined),
      evaluate: vi.fn().mockResolvedValue(undefined),
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
      chromium: {
        launchPersistentContext,
      },
    }))

    vi.doMock("./extension-id", () => ({
      resolveExtensionId,
    }))

    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "tldw-built-extension-launch-"))
    const extensionDir = path.join(tempRoot, "chrome-mv3")
    fs.mkdirSync(extensionDir, { recursive: true })
    fs.writeFileSync(
      path.join(extensionDir, "manifest.json"),
      JSON.stringify({ manifest_version: 3, name: "Built Test Extension", version: "1.0.0" }),
      "utf8",
    )
    fs.writeFileSync(path.join(extensionDir, "background.js"), "// background", "utf8")
    fs.writeFileSync(path.join(extensionDir, "options.html"), "<html></html>", "utf8")
    fs.writeFileSync(path.join(extensionDir, "sidepanel.html"), "<html></html>", "utf8")

    vi.doMock("./extension-paths", () => ({
      prioritizeExtensionBuildCandidates: () => [extensionDir],
    }))

    try {
      const { launchWithBuiltExtension } = await import("./extension-build")

      await launchWithBuiltExtension()

      expect(launchPersistentContext).toHaveBeenCalledWith(
        expect.stringContaining("tmp-playwright-profile/user-data-"),
        expect.objectContaining({
          headless: true,
          channel: "chromium",
          acceptDownloads: true,
          ignoreDefaultArgs: ["--disable-extensions"],
          args: expect.arrayContaining([
            `--disable-extensions-except=${extensionDir}`,
            `--load-extension=${extensionDir}`,
            "--no-crashpad",
            "--disable-crash-reporter",
            "--crash-dumps-dir=/tmp",
          ]),
        }),
      )
    } finally {
      fs.rmSync(tempRoot, { recursive: true, force: true })
    }
  })
})
