import { chromium } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'

import { resolveExtensionHeadlessMode } from './extension-common'
import { resolveExtensionId } from './extension-id'
import { prioritizeExtensionBuildCandidates } from './extension-paths'

type LaunchOptions = {
  seedConfig?: Record<string, any>
  allowOffline?: boolean
  seedLocalStorage?: Record<string, any>
  launchTimeoutMs?: number
}

const BASE_EXTENSION_STORAGE_SEED = {
  __tldw_first_run_complete: true,
  tldw_skip_landing_hub: true,
  quickIngestInspectorIntroDismissed: true,
  quickIngestOnboardingDismissed: true,
  "tldw:workflow:landing-config": {
    showOnFirstRun: true,
    dismissedAt: Date.now(),
    completedWorkflows: []
  }
}

const CONNECTION_CONFIG_KEYS = [
  "serverUrl",
  "authMode",
  "apiKey",
  "accessToken",
  "refreshToken",
  "orgId"
] as const

const isRecord = (value: unknown): value is Record<string, any> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const extractConnectionConfig = (
  value: unknown
): Record<string, any> | null => {
  if (!isRecord(value)) return null

  const config: Record<string, any> = {}
  for (const key of CONNECTION_CONFIG_KEYS) {
    if (typeof value[key] !== "undefined") {
      config[key] = value[key]
    }
  }

  return Object.keys(config).length > 0 ? config : null
}

export const normalizeBuiltExtensionSeedConfig = (
  seedConfig?: Record<string, any> | null
): {
  storagePayload: Record<string, any>
  connectionConfig: Record<string, any> | null
} => {
  if (!isRecord(seedConfig)) {
    return {
      storagePayload: { ...BASE_EXTENSION_STORAGE_SEED },
      connectionConfig: null
    }
  }

  const nestedConnectionConfig = extractConnectionConfig(seedConfig.tldwConfig)
  const rootConnectionConfig = extractConnectionConfig(seedConfig)
  const connectionConfig = nestedConnectionConfig || rootConnectionConfig

  if (nestedConnectionConfig) {
    return {
      storagePayload: {
        ...BASE_EXTENSION_STORAGE_SEED,
        ...seedConfig
      },
      connectionConfig
    }
  }

  if (rootConnectionConfig) {
    return {
      storagePayload: {
        ...BASE_EXTENSION_STORAGE_SEED,
        tldwConfig: rootConnectionConfig,
        ...seedConfig
      },
      connectionConfig
    }
  }

  return {
    storagePayload: {
      ...BASE_EXTENSION_STORAGE_SEED,
      ...seedConfig
    },
    connectionConfig: null
  }
}

async function waitForStorageSeed(page: any) {
  await page.waitForFunction(
    () =>
      new Promise<boolean>((resolve) => {
        if (typeof chrome === 'undefined' || !chrome.storage?.local) {
          resolve(false)
          return
        }
        chrome.storage.local.get('__e2eSeeded', (items) => {
          resolve(Boolean(items?.__e2eSeeded))
        })
      }),
    undefined,
    { timeout: 10000 }
  )
}

function makeTempProfileDirs() {
  const root = path.resolve('tmp-playwright-profile')
  fs.mkdirSync(root, { recursive: true })
  const homeDir = fs.mkdtempSync(path.join(root, 'home-'))
  const userDataDir = fs.mkdtempSync(path.join(root, 'user-data-'))
  return { homeDir, userDataDir }
}

function isExtensionBuildDir(dir: string): boolean {
  if (!dir || !fs.existsSync(dir)) {
    return false
  }

  const manifestPath = path.join(dir, 'manifest.json')
  if (!fs.existsSync(manifestPath)) {
    return false
  }

  const backgroundPath = path.join(dir, 'background.js')
  const optionsPath = path.join(dir, 'options.html')
  const sidepanelPath = path.join(dir, 'sidepanel.html')
  return fs.existsSync(backgroundPath) && (fs.existsSync(optionsPath) || fs.existsSync(sidepanelPath))
}

function resolveChromiumExecutablePath(explicitPath?: string): string | undefined {
  const fromEnv = String(explicitPath || "").trim()
  if (fromEnv) {
    return fromEnv
  }

  const userHome = process.env.HOME
  if (!userHome) {
    return undefined
  }

  const playwrightCacheRoot = path.join(userHome, "Library", "Caches", "ms-playwright")
  if (!fs.existsSync(playwrightCacheRoot)) {
    return undefined
  }

  let chromiumDirs: string[] = []
  try {
    chromiumDirs = fs
      .readdirSync(playwrightCacheRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && entry.name.startsWith("chromium-"))
      .map((entry) => entry.name)
      .sort((a, b) => {
        const aVersion = Number.parseInt(a.split("-")[1] || "0", 10)
        const bVersion = Number.parseInt(b.split("-")[1] || "0", 10)
        return bVersion - aVersion
      })
  } catch {
    return undefined
  }

  const platformDirs = ["chrome-mac-arm64", "chrome-mac-x64", "chrome-mac"]
  for (const chromiumDir of chromiumDirs) {
    for (const platformDir of platformDirs) {
      const candidate = path.join(
        playwrightCacheRoot,
        chromiumDir,
        platformDir,
        "Google Chrome for Testing.app",
        "Contents",
        "MacOS",
        "Google Chrome for Testing"
      )
      if (fs.existsSync(candidate)) {
        return candidate
      }
    }
  }

  return undefined
}

function resolvePlaywrightChannel(): string | undefined {
  const explicitChannel = String(
    process.env.TLDW_E2E_PLAYWRIGHT_CHANNEL || ""
  ).trim()
  if (explicitChannel) {
    return explicitChannel
  }

  return process.env.CI ? "chromium" : undefined
}

const projectRoot = path.resolve(__dirname, '..', '..', '..')

const resolveSidepanelUrl = (baseUrl: string, target?: string): string => {
  const normalized = String(target || "").trim()
  if (!normalized) return baseUrl
  if (normalized.startsWith("?") || normalized.startsWith("#")) {
    return `${baseUrl}${normalized}`
  }
  return `${baseUrl}#${normalized.startsWith("/") ? normalized : `/${normalized}`}`
}

export async function launchWithBuiltExtension(
  { seedConfig, allowOffline, seedLocalStorage, launchTimeoutMs }: LaunchOptions = {}
) {
  const normalizedSeed = normalizeBuiltExtensionSeedConfig(seedConfig)
  const seedStoragePayload = seedConfig ? normalizedSeed.storagePayload : null
  const seedConnectionConfig = normalizedSeed.connectionConfig
  const rawCandidates = prioritizeExtensionBuildCandidates([
    path.resolve(projectRoot, 'build/chrome-mv3'),
    path.resolve(projectRoot, '.output/chrome-mv3')
  ])
  const candidates = rawCandidates.filter(isExtensionBuildDir)
  const extensionPath = candidates[0]
  if (!extensionPath) {
    const invalidCandidates = rawCandidates.filter((p) => fs.existsSync(p) && !candidates.includes(p))
    const invalidHint = invalidCandidates.length
      ? `Ignored invalid extension directories (missing manifest or required assets): ${invalidCandidates.join(', ')}. `
      : ''
    throw new Error(
      `${invalidHint}No built extension found. Tried: ${rawCandidates.join(', ')}. ` +
      `Run "npm run build:chrome" from apps/extension.`
    )
  }
  const configuredLaunchTimeout = Number.parseInt(
    String(process.env.TLDW_E2E_EXTENSION_LAUNCH_TIMEOUT_MS || ""),
    10
  )
  const effectiveLaunchTimeoutMs =
    launchTimeoutMs ??
    (Number.isFinite(configuredLaunchTimeout) && configuredLaunchTimeout > 0
      ? configuredLaunchTimeout
      : 30000)

  const { homeDir, userDataDir } = makeTempProfileDirs()
  const executablePath = resolveChromiumExecutablePath(
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
  )
  const channel = resolvePlaywrightChannel()
  const headless = resolveExtensionHeadlessMode()
  const context = await chromium.launchPersistentContext(userDataDir, {
    timeout: effectiveLaunchTimeoutMs,
    headless,
    channel,
    acceptDownloads: true,
    ignoreDefaultArgs: ['--disable-extensions'],
    env: {
      ...process.env,
      HOME: homeDir
    },
    executablePath: executablePath || undefined,
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
      '--no-crashpad',
      '--disable-crash-reporter',
      '--crash-dumps-dir=/tmp'
    ]
  })

  const configuredTargetWait = Number.parseInt(
    String(process.env.TLDW_E2E_EXTENSION_TARGET_WAIT_MS || ""),
    10
  )
  const targetWaitMs =
    Number.isFinite(configuredTargetWait) && configuredTargetWait > 0
      ? configuredTargetWait
      : 30000

  // Enable E2E debug logs in extension pages.
  await context.addInitScript(() => {
    try {
      ;(globalThis as any).__tldw_e2e_debug = true
    } catch {
      // ignore flag set failures
    }
  })

  // Test-only: redirect sync storage writes to local to avoid sync quota limits.
  await context.addInitScript(() => {
    const patchStorage = (storage: any) => {
      if (!storage?.sync || !storage?.local) return
      const local = storage.local
      const sync = storage.sync
      const methods = ["get", "set", "remove", "clear", "getBytesInUse"]
      for (const method of methods) {
        if (typeof local?.[method] === "function") {
          sync[method] = local[method].bind(local)
        }
      }
    }
    try {
      if (typeof chrome !== "undefined") {
        patchStorage(chrome.storage)
      }
      if (typeof browser !== "undefined") {
        patchStorage((browser as any).storage)
      }
    } catch {
      // ignore patch failures
    }
  })

  // Seed storage before any extension pages load to bypass connection checks
  await context.addInitScript(
    (cfg, allowOfflineFlag) => {
      try {
        if (typeof chrome === 'undefined' || !chrome.storage?.local) return
        const setLocal = (data: Record<string, any>, done?: () => void) => {
          // @ts-ignore
          const setter = chrome?.storage?.local?.set
          if (typeof setter === 'function') {
            setter(data, () => done?.())
          } else {
            done?.()
          }
        }
        const setSync = (data: Record<string, any>, done?: () => void) => {
          // @ts-ignore
          const setter = chrome?.storage?.sync?.set
          if (typeof setter === 'function') {
            setter(data, () => done?.())
          } else {
            done?.()
          }
        }
        const finalize = () => {
          setLocal({ __e2eSeeded: true })
          setSync({ __e2eSeeded: true })
        }

        chrome.storage.local.get('__e2eSeeded', (items) => {
          if (items?.__e2eSeeded) return
          let pending = 0
          const done = () => {
            pending -= 1
            if (pending <= 0) finalize()
          }

          if (allowOfflineFlag) {
            pending += 1
            setLocal({ __tldw_allow_offline: true }, done)
          }

          if (cfg && typeof cfg === "object") {
            pending += 1
            setLocal(cfg, done)
            pending += 1
            setSync(cfg, done)
          }

          if (pending === 0) finalize()
        })
      } catch {
        // ignore storage write failures in isolated contexts
      }
    },
    seedStoragePayload,
    allowOffline || false
  )

  // Wait for SW/background
  const waitForTargets = async () => {
    if (context.serviceWorkers().length || context.backgroundPages().length) return
    await Promise.race([
      context.waitForEvent('serviceworker').catch(() => null),
      context.waitForEvent('backgroundpage').catch(() => null),
      new Promise((r) => setTimeout(r, targetWaitMs))
    ])
  }
  await waitForTargets()

  // Seed storage via service worker before any extension pages load.
  // This avoids a race where the options UI checks connection before storage is ready.
  const sw = context.serviceWorkers()[0]
  if (sw) {
    await sw.evaluate(() => {
      try {
        const patchStorage = (storage: any) => {
          if (!storage?.sync || !storage?.local) return
          const local = storage.local
          const sync = storage.sync
          const methods = ["get", "set", "remove", "clear", "getBytesInUse"]
          for (const method of methods) {
            if (typeof local?.[method] === "function") {
              sync[method] = local[method].bind(local)
            }
          }
        }
        if (typeof chrome !== "undefined") {
          patchStorage(chrome.storage)
        }
        if (typeof browser !== "undefined") {
          patchStorage((browser as any).storage)
        }
      } catch {
        // ignore patch failures
      }
    })

    if (seedConfig) {
      await sw.evaluate(({ cfg, allowOfflineFlag }) => {
        return new Promise<void>((resolve) => {
          const setLocal = (data: Record<string, any>, done?: () => void) => {
            try {
              chrome.storage.local.set(data, () => done?.())
            } catch {
              done?.()
            }
          }
          const setSync = (data: Record<string, any>, done?: () => void) => {
            try {
              chrome.storage.sync.set(data, () => done?.())
            } catch {
              done?.()
            }
          }
          chrome.storage.local.clear(() => {
            chrome.storage.sync.clear(() => {
              let pending = 0
              const done = () => {
                pending -= 1
                if (pending <= 0) resolve()
              }
              if (allowOfflineFlag) {
                pending += 1
                setLocal({ __tldw_allow_offline: true }, done)
              }
              pending += 1
              setSync(cfg, done)
              pending += 1
              setLocal(cfg, done)
              pending += 1
              setSync({ __e2eSeeded: true }, done)
              pending += 1
              setLocal({ __e2eSeeded: true }, done)
            })
          })
        })
      }, { cfg: seedStoragePayload, allowOfflineFlag: allowOffline || false })
    } else {
      await sw.evaluate(() => {
        return new Promise<void>((resolve) => {
          chrome.storage.local.clear(() => {
            chrome.storage.sync.clear(() => {
              chrome.storage.local.set({ __e2eSeeded: true }, () => {
                chrome.storage.sync.set({ __e2eSeeded: true }, () => {
                  resolve()
                })
              })
            })
          })
        })
      })
    }
  }

  // Seed localStorage for tutorials and other non-extension storage
  if (seedLocalStorage) {
    await context.addInitScript((localStorageData) => {
      if (typeof localStorage === 'undefined') return
      for (const [key, value] of Object.entries(localStorageData)) {
        localStorage.setItem(key, typeof value === 'string' ? value : JSON.stringify(value))
      }
    }, seedLocalStorage)
  }

  const extensionId = await resolveExtensionId(context, { userDataDir })
  const optionsUrl = `chrome-extension://${extensionId}/options.html`
  const sidepanelUrl = `chrome-extension://${extensionId}/sidepanel.html`

  const page = await context.newPage()
  await page.goto(optionsUrl)
  await waitForStorageSeed(page)

  // When seeding config, proactively hydrate the connection store so tests do
  // not race first-run onboarding checks on initial mount.
  if (seedConfig) {
    await page
      .waitForFunction(
        () =>
          typeof (window as any).__tldw_useConnectionStore?.getState ===
          "function",
        undefined,
        { timeout: 15_000 }
      )
      .catch(() => undefined)

    await page
      .evaluate(async (cfg) => {
        const store = (window as any).__tldw_useConnectionStore
        if (!store?.getState) return
        const actions = store.getState()

        try {
          if (cfg && typeof cfg === "object" && typeof actions.setConfigPartial === "function") {
            await actions.setConfigPartial(cfg)
          }
        } catch {
          // ignore config hydration failures in test contexts
        }

        try {
          if (typeof actions.markFirstRunComplete === "function") {
            await actions.markFirstRunComplete()
          }
        } catch {
          // ignore flag write failures in test contexts
        }

        try {
          if (typeof actions.checkOnce === "function") {
            await actions.checkOnce()
          }
        } catch {
          // ignore connection check failures in test contexts
        }
      }, seedConnectionConfig)
      .catch(() => undefined)
  }

  async function openSidepanel(target?: string) {
    const p = await context.newPage()
    await p.goto(resolveSidepanelUrl(sidepanelUrl, target))
    return p
  }

  return { context, page, openSidepanel, extensionId, optionsUrl, sidepanelUrl }
}
