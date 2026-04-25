import { BrowserContext, Page, chromium } from '@playwright/test'
import path from 'path'
import fs from 'fs'

import { resolveExtensionHeadlessMode } from './extension-common'
import { resolveExtensionId } from './extension-id'
import { prioritizeExtensionBuildCandidates } from './extension-paths'

async function waitForStorageSeed(page: Page) {
  await page.waitForFunction(
    () =>
      new Promise<boolean>((resolve) => {
        // Ensure we're in an extension context with chrome.storage
        if (typeof chrome === 'undefined' || !chrome.storage?.local) {
          resolve(false)
          return
        }

        // Check both local and sync for the sentinel since we seed both
        chrome.storage.local.get('__e2eSeeded', (localItems) => {
          if (localItems?.__e2eSeeded) {
            chrome.storage.sync.get('__e2eSeeded', (syncItems) => {
              resolve(!!syncItems?.__e2eSeeded)
            })
          } else {
            resolve(false)
          }
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
  const fromEnv = String(explicitPath || '').trim()
  if (fromEnv) {
    return fromEnv
  }

  const userHome = process.env.HOME
  if (!userHome) {
    return undefined
  }

  const playwrightCacheRoot = path.join(userHome, 'Library', 'Caches', 'ms-playwright')
  if (!fs.existsSync(playwrightCacheRoot)) {
    return undefined
  }

  let chromiumDirs: string[] = []
  try {
    chromiumDirs = fs
      .readdirSync(playwrightCacheRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && entry.name.startsWith('chromium-'))
      .map((entry) => entry.name)
      .sort((a, b) => {
        const aVersion = Number.parseInt(a.split('-')[1] || '0', 10)
        const bVersion = Number.parseInt(b.split('-')[1] || '0', 10)
        return bVersion - aVersion
      })
  } catch {
    return undefined
  }

  const platformDirs = ['chrome-mac-arm64', 'chrome-mac-x64', 'chrome-mac']
  for (const chromiumDir of chromiumDirs) {
    for (const platformDir of platformDirs) {
      const candidate = path.join(
        playwrightCacheRoot,
        chromiumDir,
        platformDir,
        'Google Chrome for Testing.app',
        'Contents',
        'MacOS',
        'Google Chrome for Testing'
      )
      if (fs.existsSync(candidate)) {
        return candidate
      }
    }
  }

  return undefined
}

function resolvePlaywrightChannel(): string | undefined {
  const explicitChannel = String(process.env.TLDW_E2E_PLAYWRIGHT_CHANNEL || '').trim()
  if (explicitChannel) {
    return explicitChannel
  }

  return process.env.CI ? 'chromium' : undefined
}

export interface LaunchWithExtensionResult {
  context: BrowserContext
  page: Page
  extensionId: string
  optionsUrl: string
  sidepanelUrl: string
  openSidepanel: (target?: string) => Promise<Page>
}

const resolveSidepanelUrl = (baseUrl: string, target?: string): string => {
  const normalized = String(target || "").trim()
  if (!normalized) return baseUrl
  if (normalized.startsWith("?") || normalized.startsWith("#")) {
    return `${baseUrl}${normalized}`
  }
  return `${baseUrl}#${normalized.startsWith("/") ? normalized : `/${normalized}`}`
}

export async function launchWithExtension(
  extensionPath: string,
  {
    seedConfig,
    seedLocalStorage,
    launchTimeoutMs
  }: {
    seedConfig?: Record<string, any>
    seedLocalStorage?: Record<string, any>
    launchTimeoutMs?: number
  } = {}
): Promise<LaunchWithExtensionResult> {
  const isDevBuild = (dir: string) => {
    const optionsPath = path.join(dir, 'options.html')
    if (!fs.existsSync(optionsPath)) return false
    const html = fs.readFileSync(optionsPath, 'utf8')
    return (
      html.includes('http://localhost:') ||
      html.includes('/@vite/client') ||
      html.includes('virtual:wxt-html-plugins')
    )
  }

  // Pick the first existing extension build so tests work whether dev output or prod build is present.
  const rawCandidates = prioritizeExtensionBuildCandidates([
    extensionPath,
    path.resolve('.output/chrome-mv3'),
    path.resolve('build/chrome-mv3')
  ]).filter((p) => p && fs.existsSync(p))
  const candidates = rawCandidates.filter(isExtensionBuildDir)
  const allowDev = ['1', 'true', 'yes'].includes(
    String(process.env.TLDW_E2E_ALLOW_DEV || '').toLowerCase()
  )
  const prodCandidates = candidates.filter((p) => !isDevBuild(p))
  const devCandidates = candidates.filter((p) => isDevBuild(p))
  const extPath =
    prodCandidates[0] || (allowDev ? devCandidates[0] : undefined)
  if (!extPath) {
    const invalidCandidates = rawCandidates.filter((p) => !candidates.includes(p))
    const invalidHint = invalidCandidates.length
      ? `Ignored invalid extension directories (missing manifest or required assets): ${invalidCandidates.join(', ')}. `
      : ''
    const devHint = devCandidates.length
      ? 'Found only dev-server builds. Run "bun run build:chrome" or start the dev server and set TLDW_E2E_ALLOW_DEV=1.'
      : 'Run "bun run build:chrome" first.'
    throw new Error(
      `No production extension build found. ${invalidHint}Tried: ${rawCandidates.join(
        ', '
      )}. ${devHint}`
    )
  }

  const { homeDir, userDataDir } = makeTempProfileDirs()

  const configuredLaunchTimeout = Number.parseInt(
    String(process.env.TLDW_E2E_EXTENSION_LAUNCH_TIMEOUT_MS || ""),
    10
  )
  const effectiveLaunchTimeoutMs =
    launchTimeoutMs ??
    (Number.isFinite(configuredLaunchTimeout) && configuredLaunchTimeout > 0
      ? configuredLaunchTimeout
      : 30000)

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
      `--disable-extensions-except=${extPath}`,
      `--load-extension=${extPath}`,
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

  // Wait for background targets to appear (service worker or background page)
  const waitForTargets = async () => {
    // Already present?
    if (context.serviceWorkers().length || context.backgroundPages().length) return
    await Promise.race([
      context.waitForEvent('serviceworker').catch(() => null),
      context.waitForEvent('backgroundpage').catch(() => null),
      new Promise((r) => setTimeout(r, targetWaitMs))
    ])
  }
  await waitForTargets()

  // Log service worker status for debugging
  const sw = context.serviceWorkers()[0]
  if (sw) {
    console.log('[E2E_DEBUG] Service worker found:', sw.url())
    // Attach console logging to service worker
    sw.on('console', (msg) => {
      console.log('[SW_CONSOLE]', msg.type(), msg.text())
    })
  } else {
    console.log('[E2E_DEBUG] No service worker found after waiting')
  }

  const extensionId = await resolveExtensionId(context, { userDataDir })
  const optionsUrl = `chrome-extension://${extensionId}/options.html`
  const sidepanelUrl = `chrome-extension://${extensionId}/sidepanel.html`

  // --- CRITICAL FIX: Seed storage via service worker BEFORE page loads ---
  // This ensures the connection check (which runs early in page load) finds config.
  // The addInitScript approach below runs too late - after React mounts and checks.
  if (seedConfig && sw) {
    await sw.evaluate((cfg) => {
      return new Promise<void>((resolve) => {
        chrome.storage.local.clear(() => {
          chrome.storage.sync.clear(() => {
            chrome.storage.sync.set(cfg, () => {
              chrome.storage.local.set(cfg, () => {
                chrome.storage.sync.set({ __e2eSeeded: true }, () => {
                  chrome.storage.local.set({ __e2eSeeded: true }, () => {
                    resolve()
                  })
                })
              })
            })
          })
        })
      })
    }, seedConfig)
  } else if (sw) {
    // Clear storage when no seedConfig provided
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

  // Ensure each test run starts from a clean extension storage state so
  // first-run onboarding and connection flows behave deterministically.
  // Only clear once per browser context to avoid races when opening extra tabs.
  // IMPORTANT: Seed both chrome.storage.local AND chrome.storage.sync because
  // @plasmohq/storage defaults to sync, but some code explicitly uses local.
  // NOTE: This addInitScript is kept for subsequent page loads within the same context.
  if (seedConfig) {
    await context.addInitScript((cfg) => {
      if (typeof chrome === 'undefined' || !chrome.storage?.local) {
        return
      }

      chrome.storage.local.get('__e2eSeeded', (items) => {
        if (items?.__e2eSeeded) return
        // Clear and seed both storage areas
        chrome.storage.local.clear(() => {
          chrome.storage.sync.clear(() => {
            // Seed local storage
            chrome.storage.local.set(cfg, () => {
              chrome.storage.local.set({ __e2eSeeded: true }, () => {
                // Seed sync storage (used by @plasmohq/storage default)
                chrome.storage.sync.set(cfg, () => {
                  chrome.storage.sync.set({ __e2eSeeded: true }, () => {
                    // Both storage areas seeded
                  })
                })
              })
            })
          })
        })
      })
    }, seedConfig)
  } else {
    await context.addInitScript(() => {
      if (typeof chrome === 'undefined' || !chrome.storage?.local) {
        return
      }

      chrome.storage.local.get('__e2eSeeded', (items) => {
        if (items?.__e2eSeeded) return
        // Clear both storage areas
        chrome.storage.local.clear(() => {
          chrome.storage.sync.clear(() => {
            chrome.storage.local.set({ __e2eSeeded: true }, () => {
              chrome.storage.sync.set({ __e2eSeeded: true }, () => {
                // Both storage areas cleared
              })
            })
          })
        })
      })
    })
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

  const page = await context.newPage()
  // Ensure the extension is ready before navigating
  await page.waitForTimeout(250)
  await page.goto(optionsUrl)
  // Wait until storage has been cleared/seeded (sentinel set)
  await waitForStorageSeed(page)

  // If config was seeded, wait for the actual tldwConfig to be present in storage.
  // This ensures the background service worker can read the config when it tries to.
  if (seedConfig) {
    await page.waitForFunction(
      () =>
        new Promise<boolean>((resolve) => {
          if (typeof chrome === 'undefined' || !chrome.storage?.sync) {
            resolve(false)
            return
          }
          chrome.storage.sync.get('tldwConfig', (items) => {
            resolve(!!items?.tldwConfig?.serverUrl)
          })
        }),
      undefined,
      { timeout: 5000 }
    )

    // --- Wait for React app and connection store to mount ---
    // The store is exposed on window when the React app mounts. We must wait
    // for it to exist before we can call checkOnce() or wait for connected state.
    await page.waitForFunction(
      () => {
        const root = document.querySelector('#root')
        if (!root) return false
        const store = (window as any).__tldw_useConnectionStore
        return !!store?.getState
      },
      undefined,
      { timeout: 15000 }
    )

    // --- CRITICAL FIX: Force connection re-check after storage is seeded ---
    // The connection store may have already run checkOnce() before storage was seeded.
    // Even though we seed via SW first, there can still be timing issues where the
    // React app mounts and checks before the SW seeding completes. Force a re-check.
    await page.evaluate(async () => {
      const store = (window as any).__tldw_useConnectionStore
      if (store?.getState?.()?.checkOnce) {
        await store.getState().checkOnce()
      }
    })

    // --- CRITICAL FIX: Wait for actual connected state, not just storage presence ---
    // This ensures the connection check has completed and succeeded before tests proceed.
    // Timeout must exceed CONNECTION_TIMEOUT_MS (20 seconds) in connection.tsx to avoid
    // race conditions when the server health check takes the full timeout duration.
    await page.waitForFunction(
      () => {
        const store = (window as any).__tldw_useConnectionStore
        if (!store) return false
        const state = store.getState()?.state
        // Allow connected OR offlineBypass (demo mode)
        return state?.isConnected === true || state?.offlineBypass === true
      },
      undefined,
      { timeout: 25000 }
    )
  }

  async function openSidepanel(target?: string) {
    const p = await context.newPage()
    await p.goto(resolveSidepanelUrl(sidepanelUrl, target), {
      waitUntil: 'domcontentloaded'
    })
    // Ensure the sidepanel tab is visible; some UI only renders when visible.
    try {
      await p.bringToFront()
    } catch {
      // ignore bringToFront failures in headless contexts
    }
    // Ensure the sidepanel app has a root to mount into before returning.
    const root = p.locator('#root')
    try {
      await root.waitFor({ state: 'visible', timeout: 10000 })
    } catch {
      // Ignore if the root is not visible yet; downstream tests will assert.
    }
    await waitForStorageSeed(p)
    return p
  }

  return { context, page, extensionId, optionsUrl, sidepanelUrl, openSidepanel }
}
