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
    evaluate: vi.fn().mockResolvedValue({ ok: true }),
  }
  const context = {
    serviceWorkers: vi.fn(() => []),
    backgroundPages: vi.fn(() => []),
    waitForEvent: vi.fn(() => new Promise(() => {})),
    addInitScript: vi.fn().mockResolvedValue(undefined),
    newPage: vi.fn().mockResolvedValue(page),
    close: vi.fn().mockResolvedValue(undefined),
  }
  const launchPersistentContext = vi.fn().mockResolvedValue(context)

  vi.doMock("@playwright/test", () => ({
    chromium: {
      launchPersistentContext,
    },
  }))
  vi.doMock("./extension-id", () => ({ resolveExtensionId }))

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

  const prepareExtensionLaunchPath = vi.fn(
    (extensionPath: string, options: { deterministicManifestKey?: boolean; rootDir: string }) =>
      options.deterministicManifestKey
        ? path.join(options.rootDir, "staged-extension")
        : extensionPath
  )
  vi.doMock("./extension-paths", () => ({
    prepareExtensionLaunchPath,
    prioritizeExtensionBuildCandidates: () => [extensionDir],
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
    resolveExtensionId,
  }
}

afterEach(() => {
  process.env = { ...originalEnv }
  vi.useRealTimers()
  vi.unstubAllGlobals()
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
    const {
      cleanup,
      context,
      extensionDir,
      launchPersistentContext,
      launchWithBuiltExtension,
      page,
      prepareExtensionLaunchPath,
      resolveExtensionId,
    } = await setupBuiltExtensionLaunchTest()
    process.env.LEGACY_INHERITED_VALUE = "preserved"

    try {
      await launchWithBuiltExtension()

      expect(page.goto).toHaveBeenCalledWith(
        `chrome-extension://${"e".repeat(32)}/options.html`,
      )
      expect(prepareExtensionLaunchPath).toHaveBeenCalledWith(
        extensionDir,
        expect.objectContaining({
          preserveDefaultLocaleCatalog: false,
          rootDir: expect.stringContaining("tmp-playwright-profile/user-data-"),
          deterministicManifestKey: false,
        }),
      )
      expect(resolveExtensionId).toHaveBeenCalledWith(
        context,
        expect.objectContaining({
          extensionPath: extensionDir,
          userDataDir: expect.stringContaining("tmp-playwright-profile/user-data-"),
        }),
      )
      expect(launchPersistentContext).toHaveBeenCalledWith(
        expect.stringContaining("tmp-playwright-profile/user-data-"),
        expect.objectContaining({
          headless: true,
          channel: "chromium",
          acceptDownloads: true,
          ignoreDefaultArgs: ["--disable-extensions"],
          env: expect.objectContaining({
            LEGACY_INHERITED_VALUE: "preserved",
          }),
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
      cleanup()
    }
  })

  it("keeps strict profile data and Chromium paths beneath profileRoot without inherited secrets", async () => {
    const {
      cleanup,
      extensionDir,
      launchPersistentContext,
      launchWithBuiltExtension,
      prepareExtensionLaunchPath,
      resolveExtensionId,
    } = await setupBuiltExtensionLaunchTest()
    const profileRoot = fs.mkdtempSync(
      path.join(os.tmpdir(), "tldw-strict-profile-")
    )
    process.env.OPENAI_API_KEY = "must-not-reach-chromium"
    process.env.ANTHROPIC_API_KEY = "must-not-reach-chromium"
    process.env.SYSTEMROOT = "C:\\Windows"
    process.env.WINDIR = "C:\\Windows"
    process.env.COMSPEC = "C:\\Windows\\System32\\cmd.exe"
    process.env.PATHEXT = ".COM;.EXE;.BAT;.CMD"
    process.env.SSL_CERT_FILE = "/host/cert.pem"
    process.env.LD_LIBRARY_PATH = "/host/lib"

    try {
      await launchWithBuiltExtension({ profileRoot })

      const [userDataDir, launchOptions] = launchPersistentContext.mock.calls[0]
      const strictRoot = path.resolve(profileRoot)
      const homeDir = launchOptions.env.HOME
      const strictLaunchPath = path.join(
        userDataDir,
        "extension-launch",
        "staged-extension"
      )

      expect(path.dirname(userDataDir)).toBe(strictRoot)
      expect(path.basename(userDataDir)).toMatch(/^user-data-/)
      expect(path.dirname(homeDir)).toBe(strictRoot)
      expect(path.basename(homeDir)).toMatch(/^home-/)
      expect(launchOptions.env).toMatchObject({
        HOME: homeDir,
        USERPROFILE: homeDir,
        TMPDIR: path.join(strictRoot, "tmp"),
        TMP: path.join(strictRoot, "tmp"),
        TEMP: path.join(strictRoot, "tmp"),
        APPDATA: path.join(strictRoot, "appdata"),
        LOCALAPPDATA: path.join(strictRoot, "localappdata"),
        XDG_CACHE_HOME: path.join(strictRoot, "xdg-cache"),
        XDG_CONFIG_HOME: path.join(strictRoot, "xdg-config"),
        SYSTEMROOT: "C:\\Windows",
        WINDIR: "C:\\Windows",
        COMSPEC: "C:\\Windows\\System32\\cmd.exe",
        PATHEXT: ".COM;.EXE;.BAT;.CMD",
        SSL_CERT_FILE: "/host/cert.pem",
        LD_LIBRARY_PATH: "/host/lib"
      })
      expect(launchOptions.env).not.toHaveProperty("OPENAI_API_KEY")
      expect(launchOptions.env).not.toHaveProperty("ANTHROPIC_API_KEY")
      for (const directory of [
        path.join(strictRoot, "tmp"),
        path.join(strictRoot, "crash-dumps"),
        path.join(strictRoot, "appdata"),
        path.join(strictRoot, "localappdata"),
        path.join(strictRoot, "xdg-cache"),
        path.join(strictRoot, "xdg-config")
      ]) {
        expect(fs.existsSync(directory)).toBe(true)
      }
      expect(launchOptions.args).toContain(
        `--crash-dumps-dir=${path.join(strictRoot, "crash-dumps")}`
      )
      expect(prepareExtensionLaunchPath).toHaveBeenCalledWith(
        extensionDir,
        expect.objectContaining({
          rootDir: path.join(userDataDir, "extension-launch"),
          deterministicManifestKey: true,
        }),
      )
      expect(launchOptions.args).toEqual(
        expect.arrayContaining([
          `--disable-extensions-except=${strictLaunchPath}`,
          `--load-extension=${strictLaunchPath}`,
        ])
      )
      expect(resolveExtensionId).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({ extensionPath: strictLaunchPath }),
      )
      expect(path.relative(strictRoot, strictLaunchPath)).not.toMatch(/^\.\./)
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
        },
      })

      expect(events).toEqual([
        "prepare",
        `goto:chrome-extension://${"e".repeat(32)}/options.html#/skills`,
      ])
      expect(preparedPage).toBe(page)
      expect(context.newPage).toHaveBeenCalledTimes(1)
    } finally {
      cleanup()
    }
  })

  it("closes the persistent context when extension id resolution fails", async () => {
    const { cleanup, context, launchWithBuiltExtension, resolveExtensionId } =
      await setupBuiltExtensionLaunchTest()
    const resolutionError = new Error("extension id resolution failed")
    resolveExtensionId.mockRejectedValueOnce(resolutionError)

    try {
      await expect(launchWithBuiltExtension()).rejects.toBe(resolutionError)

      expect(context.close).toHaveBeenCalledTimes(1)
      expect(context.newPage).not.toHaveBeenCalled()
    } finally {
      cleanup()
    }
  })

  it("closes the persistent context when targeted page preparation fails", async () => {
    const { cleanup, context, launchWithBuiltExtension, page } =
      await setupBuiltExtensionLaunchTest()
    const preparationError = new Error("options preparation failed")

    try {
      await expect(launchWithBuiltExtension({
        prepareOptionsPage: () => {
          throw preparationError
        },
      })).rejects.toBe(preparationError)

      expect(context.close).toHaveBeenCalledTimes(1)
      expect(page.goto).not.toHaveBeenCalled()
    } finally {
      cleanup()
    }
  })

  it("closes the persistent context when the background worker is not ready", async () => {
    const { cleanup, context, launchWithBuiltExtension, page } =
      await setupBuiltExtensionLaunchTest()
    page.evaluate.mockResolvedValueOnce(null)

    try {
      await expect(launchWithBuiltExtension()).rejects.toThrow(
        "Extension background worker did not become ready",
      )

      expect(context.close).toHaveBeenCalledTimes(1)
    } finally {
      cleanup()
    }
  })

  it("executes the bounded diagnostics readiness probe", async () => {
    const { cleanup, launchWithBuiltExtension, page } =
      await setupBuiltExtensionLaunchTest()

    try {
      await launchWithBuiltExtension()
      const [probe, timeoutMs] = page.evaluate.mock.calls[0] as [
        (timeoutMs: number) => Promise<unknown>,
        number,
      ]
      let lastError: { message: string } | undefined
      const sendMessage = vi.fn(
        (
          _message: unknown,
          callback: (response: unknown) => void,
        ) => callback({ ok: true }),
      )
      vi.stubGlobal("chrome", {
        runtime: {
          sendMessage,
          get lastError() {
            return lastError
          },
        },
      })

      expect(timeoutMs).toBe(30_000)
      await expect(probe(timeoutMs)).resolves.toEqual({ ok: true })
      expect(sendMessage).toHaveBeenCalledWith(
        { type: "tldw:diagnostics" },
        expect.any(Function),
      )

      let attempts = 0
      sendMessage.mockImplementation((_message, callback) => {
        attempts += 1
        lastError = attempts === 1 ? { message: "no receiver" } : undefined
        callback({ ok: true })
      })
      await expect(probe(timeoutMs)).resolves.toEqual({ ok: true })
      expect(attempts).toBe(2)

      lastError = undefined
      sendMessage.mockImplementation(() => undefined)
      vi.useFakeTimers()
      const timedOutProbe = probe(timeoutMs)
      await vi.advanceTimersByTimeAsync(timeoutMs)
      await expect(timedOutProbe).resolves.toBeNull()
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
      await expect(launchWithBuiltExtension({
        prepareOptionsPage: () => {
          throw preparationError
        },
      })).rejects.toSatisfy((error: unknown) => {
        expect(error).toBeInstanceOf(AggregateError)
        expect((error as AggregateError).errors).toEqual([
          preparationError,
          cleanupError,
        ])
        return true
      })

      expect(context.close).toHaveBeenCalledTimes(1)
    } finally {
      cleanup()
    }
  })
})
