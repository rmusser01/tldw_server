import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { afterEach, describe, expect, it, vi } from "vitest"

import { normalizeBuiltExtensionSeedConfig } from "./extension-build"

const originalEnv = { ...process.env }

const setupBuiltExtensionLaunchTest = async () => {
  vi.resetModules()
  process.env.CI = "true"
  process.env.TLDW_E2E_EXTENSION_TARGET_WAIT_MS = "1"

  const resolveExtensionId = vi.fn().mockResolvedValue("e".repeat(32))
  const page = {
    waitForTimeout: vi.fn().mockResolvedValue(undefined),
    goto: vi.fn().mockResolvedValue(undefined),
    waitForFunction: vi.fn().mockResolvedValue(undefined),
    evaluate: vi.fn().mockResolvedValue(undefined)
  }
  const context = {
    serviceWorkers: vi.fn(() => []),
    backgroundPages: vi.fn(() => []),
    waitForEvent: vi.fn(() => new Promise(() => {})),
    addInitScript: vi.fn().mockResolvedValue(undefined),
    newPage: vi.fn().mockResolvedValue(page),
    close: vi.fn().mockResolvedValue(undefined)
  }
  const launchPersistentContext = vi.fn().mockResolvedValue(context)

  vi.doMock("@playwright/test", () => ({
    chromium: {
      launchPersistentContext
    }
  }))
  vi.doMock("./extension-id", () => ({ resolveExtensionId }))

  const tempRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "tldw-built-extension-launch-")
  )
  const extensionDir = path.join(tempRoot, "chrome-mv3")
  fs.mkdirSync(extensionDir, { recursive: true })
  fs.writeFileSync(
    path.join(extensionDir, "manifest.json"),
    JSON.stringify({
      manifest_version: 3,
      name: "Built Test Extension",
      version: "1.0.0"
    }),
    "utf8"
  )
  fs.writeFileSync(
    path.join(extensionDir, "background.js"),
    "// background",
    "utf8"
  )
  fs.writeFileSync(
    path.join(extensionDir, "options.html"),
    "<html></html>",
    "utf8"
  )
  fs.writeFileSync(
    path.join(extensionDir, "sidepanel.html"),
    "<html></html>",
    "utf8"
  )

  const prepareExtensionLaunchPath = vi.fn(
    (extensionPath: string) => extensionPath
  )
  vi.doMock("./extension-paths", () => ({
    prepareExtensionLaunchPath,
    prioritizeExtensionBuildCandidates: () => [extensionDir]
  }))

  const { launchWithBuiltExtension } = await import("./extension-build")

  return {
    cleanup: () => fs.rmSync(tempRoot, { recursive: true, force: true }),
    context,
    extensionDir,
    launchPersistentContext,
    launchWithBuiltExtension,
    page,
    prepareExtensionLaunchPath,
    resolveExtensionId
  }
}

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
    expect(normalized.storagePayload.tldwConfig).not.toHaveProperty(
      "tldwConfig"
    )
  })

  it("launches built extensions with crashpad-disabled Chromium options", async () => {
    const {
      cleanup,
      context,
      extensionDir,
      launchPersistentContext,
      launchWithBuiltExtension,
      page,
      prepareExtensionLaunchPath,
      resolveExtensionId
    } = await setupBuiltExtensionLaunchTest()

    try {
      await launchWithBuiltExtension()

      expect(page.goto).toHaveBeenCalledWith(
        `chrome-extension://${"e".repeat(32)}/options.html`
      )
      expect(prepareExtensionLaunchPath).toHaveBeenCalledWith(
        extensionDir,
        expect.objectContaining({
          preserveDefaultLocaleCatalog: false,
          rootDir: expect.stringContaining("tmp-playwright-profile/user-data-")
        })
      )
      expect(resolveExtensionId).toHaveBeenCalledWith(
        context,
        expect.objectContaining({
          extensionPath: extensionDir,
          userDataDir: expect.stringContaining(
            "tmp-playwright-profile/user-data-"
          )
        })
      )
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
            "--crash-dumps-dir=/tmp"
          ])
        })
      )
    } finally {
      cleanup()
    }
  })

  it("keeps strict profile data and Chromium paths beneath profileRoot without inherited secrets", async () => {
    const {
      cleanup,
      extensionDir,
      launchPersistentContext,
      launchWithBuiltExtension,
      prepareExtensionLaunchPath
    } = await setupBuiltExtensionLaunchTest()
    const profileRoot = fs.mkdtempSync(
      path.join(os.tmpdir(), "tldw-strict-profile-")
    )
    process.env.OPENAI_API_KEY = "must-not-reach-chromium"

    try {
      await launchWithBuiltExtension({ profileRoot })

      const [userDataDir, launchOptions] = launchPersistentContext.mock.calls[0]
      const strictRoot = path.resolve(profileRoot)
      const homeDir = launchOptions.env.HOME

      expect(userDataDir).toMatch(new RegExp(`^${strictRoot}/user-data-`))
      expect(homeDir).toMatch(new RegExp(`^${strictRoot}/home-`))
      expect(launchOptions.env).toMatchObject({
        HOME: homeDir,
        TMPDIR: path.join(strictRoot, "tmp"),
        TMP: path.join(strictRoot, "tmp"),
        TEMP: path.join(strictRoot, "tmp")
      })
      expect(launchOptions.env).not.toHaveProperty("OPENAI_API_KEY")
      expect(launchOptions.args).toContain(
        `--crash-dumps-dir=${path.join(strictRoot, "crash-dumps")}`
      )
      expect(prepareExtensionLaunchPath).toHaveBeenCalledWith(
        extensionDir,
        expect.objectContaining({
          rootDir: path.join(userDataDir, "extension-launch")
        })
      )
    } finally {
      cleanup()
      fs.rmSync(profileRoot, { recursive: true, force: true })
    }
  })

  it("prepares a targeted options page before its first navigation", async () => {
    const { cleanup, context, launchWithBuiltExtension, page } =
      await setupBuiltExtensionLaunchTest()
    const events: string[] = []
    let preparedPage: unknown
    page.goto.mockImplementation(async (url: string) => {
      events.push(`goto:${url}`)
    })

    try {
      await launchWithBuiltExtension({
        optionsTarget: "/skills",
        prepareOptionsPage: async ({ page: pageToPrepare }) => {
          await Promise.resolve()
          preparedPage = pageToPrepare
          events.push("prepare")
        }
      })

      expect(events).toEqual([
        "prepare",
        `goto:chrome-extension://${"e".repeat(32)}/options.html#/skills`
      ])
      expect(preparedPage).toBe(page)
      expect(context.newPage).toHaveBeenCalledTimes(1)
    } finally {
      cleanup()
    }
  })

  it("closes the persistent context when targeted page preparation fails", async () => {
    const { cleanup, context, launchWithBuiltExtension, page } =
      await setupBuiltExtensionLaunchTest()
    const preparationError = new Error("options preparation failed")

    try {
      await expect(
        launchWithBuiltExtension({
          prepareOptionsPage: () => {
            throw preparationError
          }
        })
      ).rejects.toBe(preparationError)

      expect(context.close).toHaveBeenCalledTimes(1)
      expect(page.goto).not.toHaveBeenCalled()
    } finally {
      cleanup()
    }
  })

  it("preserves the preparation error when persistent context cleanup also fails", async () => {
    const { cleanup, context, launchWithBuiltExtension } =
      await setupBuiltExtensionLaunchTest()
    const preparationError = new Error("options preparation failed")
    const cleanupError = new Error("context cleanup failed")
    context.close.mockRejectedValueOnce(cleanupError)

    try {
      await expect(
        launchWithBuiltExtension({
          prepareOptionsPage: () => {
            throw preparationError
          }
        })
      ).rejects.toBe(preparationError)

      expect(context.close).toHaveBeenCalledTimes(1)
      expect(preparationError.cause).toBe(cleanupError)
    } finally {
      cleanup()
    }
  })
})
